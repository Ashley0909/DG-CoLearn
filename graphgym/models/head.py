""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.models.layer import MLP
from graphgym.models.pooling import pooling_dict

from graphgym.contrib.head import *
import graphgym.register as register


########### Head ############

class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''

    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=cfg.gnn.layers_post_mp, bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return batch.node_feature[batch.node_label_index], \
                   batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        # return pred, label
        return pred, label, copy.deepcopy([tensor.detach().clone() for tensor in batch.node_states]) # Return the node embed from current snapshot as well


class GNNEdgeHead(nn.Module):
    r"""The GNN head module for edge prediction tasks. This module takes a (batch of) graphs and
    outputs ...
    """

    def __init__(self, dim_in: int, dim_out: int):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(GNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        if cfg.model.edge_decoding == 'concat':
            # Only use node features.
            self.layer_post_mp = MLP(dim_in * 2, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        elif cfg.model.edge_decoding == 'edgeconcat':
            # Use both node and edge features.
            self.layer_post_mp = MLP(dim_in * 2 + cfg.dataset.edge_dim, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2, edge: \
                self.layer_post_mp(torch.cat((v1, v2, edge), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(dim_in, dim_in,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError('Unknown edge decoding {}.'.format(
                    cfg.model.edge_decoding))

    def _apply_index(self, batch):
        return batch.node_feature[batch.edge_label_index], \
               batch.edge_label

    def forward_pred(self, batch):
        pred, label = self._apply_index(batch)
        if hasattr(batch, 'device'):
            # TODO: optionally move the head-prediction to cpu to allow for
            #  higher throughput (e.g., for MRR computations).
            raise NotImplementedError
        nodes_first = pred[0]  # node features of the source node of each edge.
        nodes_second = pred[1]
        if cfg.model.edge_decoding == 'edgeconcat':
            edge_feature = torch.index_select(
                batch.edge_feature, 0, batch.edge_split_index)
            pred = self.decode_module(nodes_first, nodes_second, edge_feature)
        else:
            pred = self.decode_module(nodes_first, nodes_second)
        return pred, label


    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat' and \
            cfg.model.edge_decoding != 'edgeconcat':
            batch = self.layer_post_mp(batch)
        pred, label = self.forward_pred(batch)
        return pred, label, copy.deepcopy([tensor.detach().clone() for tensor in batch.node_states]) # Return the node embed from current snapshot as well

class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''

    def __init__(self, dim_in, dim_out):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=cfg.gnn.layers_post_mp, bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.graph_label

    def forward(self, batch):
        if cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch,
                                         batch.node_id_index)
        else:
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


# Head models for external interface
head_dict = {
    'node': GNNNodeHead,
    'edge': GNNEdgeHead,
    'link_pred': GNNEdgeHead,
    'graph': GNNGraphHead
}

head_dict = {**register.head_dict, **head_dict}
