import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

class ReshapeH():
    def __init__(self, glob_shape):
        self.glob_shape = glob_shape

    def reshape_to_fill(self, h, subnodes):
        if h.shape[0] == self.glob_shape:
            return h
        
        reshaped = torch.zeros((self.glob_shape, h.shape[1]))
        # reshaped = copy.deepcopy(self.empty_h)
        reshaped[subnodes] = h

        return reshaped
    
class HopFusion(nn.Module): # Implementation of Paper (adding two hops NE together)
    def __init__(self, dim_out=16):
        super().__init__()
        self.dim_out = dim_out
        self.proj = None

    def forward(self, h0, h1):
        h_cat = torch.cat([h0, h1], dim=-1) # Shape: [1, cat_size]
        input_dim = h_cat.shape[1]

        if self.proj is None:
            self.proj = nn.Linear(input_dim, self.dim_out)
            
        return self.proj(h_cat) # Shape: [1, dim_out]
    
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ROLANDGNN(torch.nn.Module):
    def __init__(self, device, input_dim, num_nodes, output_dim, dropout=0.3, update='moving', loss=nn.BCEWithLogitsLoss):
        
        super(ROLANDGNN, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess BERT repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        self.device = device
        hidden_conv_1 = 128 #64
        hidden_conv_2 = 128 #32
        self.preprocess1 = Linear(input_dim, 256).to(self.device)
        self.bn1 = nn.BatchNorm1d(256).to(self.device)
        self.preprocess2 = Linear(256, 128).to(self.device)
        self.bn2 = nn.BatchNorm1d(128).to(self.device)
        self.conv1 = GCNConv(128, hidden_conv_1).to(self.device)
        self.bn3 = nn.BatchNorm1d(hidden_conv_1).to(self.device)
        self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2).to(self.device)
        self.bn4 = nn.BatchNorm1d(hidden_conv_2).to(self.device)
        self.postprocess1 = Linear(hidden_conv_2, output_dim).to(self.device)
        self.nc_mlp = nn.Sequential(
            Linear(hidden_conv_2, 128),
            nn.ReLU(),
            Linear(128, output_dim)
        ).to(self.device)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.dropout = dropout
        self.update = update
        if update=='moving':
            self.tau = torch.Tensor([0]).to(self.device)
        elif update=='learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0])).to(self.device)
        elif update=='gru':
            self.gru1 = nn.GRUCell(hidden_conv_1, hidden_conv_1).to(self.device)
            self.gru2 = nn.GRUCell(hidden_conv_2, hidden_conv_2).to(self.device)
        elif update=='mlp':
            self.mlp1 = Linear(hidden_conv_1*2, hidden_conv_1).to(self.device)
            self.mlp2 = Linear(hidden_conv_2*2, hidden_conv_2).to(self.device)
        else:
            assert(0<=update<=1)
            self.tau = torch.Tensor([update]).to(self.device)
        self.previous_embeddings = [torch.zeros((num_nodes, hidden_conv_1)).to(self.device), torch.zeros((num_nodes, hidden_conv_2)).to(self.device)]

        empty_h = torch.zeros((num_nodes, hidden_conv_1)).to(self.device)
        self.reshape = ReshapeH(empty_h)
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()

    def forward(self, x, edge_index, task_type, edge_label_index=None, subnodes=None, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None: #None if test
            self.previous_embeddings = [previous_embeddings[0].detach().clone().to(self.device),previous_embeddings[1].detach().clone().to(self.device)]
            train = True
        else:
            train = False
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone().to(self.device) # tau -- past weight
        if not isinstance(self.update, str):
            self.tau = torch.Tensor([self.update]).to(self.device)
        
        current_embeddings = [torch.Tensor([]).to(self.device),torch.Tensor([]).to(self.device),torch.Tensor([]).to(self.device)]
        
        #Preprocess text
        h = self.preprocess1(x)
        h = self.bn1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.preprocess2(h)
        h = self.bn2(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        #Reshape h since clients have different number of nodes
        h = self.reshape.reshape_to_fill(h, subnodes)

        """ Obtain 0-hop NE """
        current_embeddings[0] = self.gru1(h, self.previous_embeddings[0].clone()).clone().to(self.device)

        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = self.bn3(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        #Embedding Update after first layer (only when training)
        if train == True:
            if self.update=='gru':
                h_temp = self.gru1(h, self.previous_embeddings[0].clone())
                h = h_temp
            elif self.update=='mlp':
                hin = torch.cat((h, self.previous_embeddings[0].clone()), dim=1)
                h_temp = self.mlp1(hin)
                h = h_temp
            else:
                h_temp = self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h
                h = h_temp
    
        current_embeddings[1] = h.clone().to(self.device)
        
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = self.bn4(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        #Embedding Update after second layer (only when training)
        if train == True:
            if self.update=='gru':
                h_temp = self.gru2(h, self.previous_embeddings[1].clone())
                h = h_temp
            elif self.update=='mlp':
                hin = torch.cat((h, self.previous_embeddings[1].clone()), dim=1)
                h_temp = self.mlp2(hin)
                h = h_temp
            else:
                h_temp = self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h
                h = h_temp
    
        current_embeddings[2] = h.clone().to(self.device)

        #HADAMARD MLP (For Link Prediction)
        if task_type == "LP":
            h_src = h[edge_label_index[0].long()].clone().to(self.device)
            h_dst = h[edge_label_index[1].long()].clone().to(self.device)
            h_hadamard = torch.mul(h_src.clone(), h_dst.clone()) #hadamard product
            h = self.postprocess1(h_hadamard.clone())
            out = torch.sum(h.clone(), dim=-1) # sum up the values in a row
        elif task_type == "NC":
            out = self.nc_mlp(h.clone())
        else:
            print('E> Invalid task type specified. Options are {LP, NC}')
            exit(-1)
        
        #return both 
        #i) the predictions for the current snapshot
        #ii) the embeddings of current snapshot

        return out, current_embeddings