import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.models.head import head_dict
from graphgym.models.layer import (GeneralLayer, GeneralMultiLayer,
                                   GeneralRecurrentLayer, GRUGraphRecurrentLayer,
                                   GraphRecurrentLayerWrapper,
                                   BatchNorm1dNode, BatchNorm1dEdge,
                                   layer_dict)
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.init import init_weights
from graphgym.models.feature_encoder import node_encoder_dict, \
    edge_encoder_dict

# from graphgym.contrib.stage import *
import graphgym.register as register
from graphgym.register import register_network

from fl_models import ReshapeH

########### Layer ############
# Methods to construct layers.
recurrent_layer_types = [
    'dcrnn', 'evolve_gcn_o', 'evolve_gcn_h', 'gconv_gru', 'gconv_lstm',
    'gconv_lstm_baseline', 'tgcn', 'edge_conv_gru'
]


def GNNLayer(dim_in, dim_out, has_act=True, id=0):
    if cfg.gnn.layer_type in recurrent_layer_types:
        # For baseline Recurrent GNNs, instantiate directly.
        return layer_dict[cfg.gnn.layer_type](dim_in, dim_out, id=id)
    else:
        # In ROLAND case, call the GeneralRecurrentLayer wrapper layer to
        # convert cfg.gnn.layer_type object to a recurrent layer that updates
        # batch.node_states as well.
        if cfg.gnn.embed_update_method == 'moving_average':
            # The original implementation.
            # TODO: use an unified implementation.
            return GeneralRecurrentLayer(cfg.gnn.layer_type, dim_in, dim_out,
                                         has_act, id=id)
        else:
            return GraphRecurrentLayerWrapper(cfg.gnn.layer_type, dim_in, dim_out,
                                              has_act, id=id)


def GNNPreMP(dim_in, dim_out):
    r"""Constructs preprocessing layers: dim_in --> dim_out --> dim_out --> ... --> dim_out"""
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out,
                             final_act=True)


########### Block: multiple layers ############

class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipBlock, self).__init__()
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, dim_out, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = \
                torch.cat((node_feature, self.f(batch).node_feature), 1)
        else:
            raise ValueError(
                'cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch


########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    r"""Simple Stage that stacks GNN layers"""

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, id=i)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


########### Model: start + stage + head ############

class GNN(nn.Module):
    r"""The General GNN model"""

    def __init__(self, dim_in, dim_out, glob_shape, task_type, **kwargs):
        r"""Initializes the GNN model.

        Args:
            dim_in, dim_out: dimensions of in and out channels.
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        # Customise Configuration according to task_type
        if task_type == 'LP':
            cfg.dataset.task = 'link_pred'
            cfg.dataset.edge_encoder = True
            cfg.gnn.layer_type = 'generalconv'
        else:
            cfg.dataset.task = 'node'
            cfg.dataset.edge_encoder = False
            cfg.gnn.layer_type = 'gcnconv'

        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]
        # Currently only for OGB datasets
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)
        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp >= 1:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.reshape = ReshapeH(glob_shape)

        self.apply(init_weights)

    def forward(self, batch):
        ''' Reshape node_feature to global node feature '''
        batch.node_feature = self.reshape.reshape_to_fill(batch.node_feature, batch.subnodes)
        for module in self.children():
            batch = module(batch)
        return batch          

register_network('gnn_recurrent', GNN)
