import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

# torch.set_printoptions(edgeitems=1000, linewidth=1000)

''' Define machine learning models '''
class MLmodelReg(nn.Module):
    """
    Linear regression model implemented as a single-layer NN
    """
    def __init__(self, in_features, out_features):
        super(MLmodelReg, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # in = independent vars, out = dependent vars

        # init weights
        self.init_weights()

    def init_weights(self, seed=None):
        # common random seed is vital for the effectiveness of averaging, see fig. 1 in
        # McMahan, B.'s paper
        if seed:
            torch.manual_seed(seed)
        else:  # all zero init, deprecated (likely to drop in a local-minimal trap)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        y_pred = self.linear(x)  # y_pred = w * x - b
        return y_pred


class MLmodelSVM(nn.Module):
    """
    Linear Support Vector Machine implemented as a single-layer NN plus regularization
    hyperplane: wx - b = 0
    + samples: wx - b > 1
    - samples: wx - b < -1
    Loss function =  ||w||/2 + C*sum{ max[0, 1 - y(wx - b)]^2 }, C is a hyper-param
    Guide: http://bytepawn.com/svm-with-pytorch.html
    """
    def __init__(self, in_features):
        super(MLmodelSVM, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def forward(self, x):
        y_hat = torch.mv(x, self.w) - self.b  # y_pred = w * x - b
        return y_hat.reshape(-1, 1)


class svmLoss(nn.Module):
    """
    Loss function class for linear SVM
        reduction: reduction method
    """
    def __init__(self, reduction='mean'):
        super(svmLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        """
        Loss calculation
        ||w||/2 + C*sum{ max[0, 1 - y*y_hat]^2 }, the regularization term is implemented in optim.SGD with weight_decay
        where y_hat = wx - b
        :param y_hat: output y as a tensor
        :param y: labels as a tensor
        :return: loss
        """
        tmp = 1 - (y * y_hat)
        # sample-wise max, (x + |x|)/2 = max(x, 0)
        abs_tmp = torch.sqrt(tmp**2).detach()
        tmp += abs_tmp  # differentiable abs
        tmp /= 2
        tmp = tmp ** 2
        if self.reduction == 'sum':
            return torch.sum(tmp)
        elif self.reduction == 'mean':
            return torch.mean(tmp)
        else:
            print('E> Wrong reduction method specified')


class MLmodelCNN(nn.Module):
    def __init__(self, classes=10):
        super(MLmodelCNN, self).__init__()
        # input = 28*28 PIL.ToTensor()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 20 [(1*)5*5] conv. kernels
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 50 [(20*)5*5] conv. kernels
        self.fc1 = nn.Linear(4 * 4 * 50, 500)  # linear 1
        self.fc2 = nn.Linear(500, classes)  # linear 2

        self.init_weights(seed=1)

    def init_weights(self, seed=None):
        # common random seed is vital for the effectiveness of averaging, see fig. 1 in
        # McMahan, B.'s paper
        if seed:
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.conv1.weight)  # current best practice
            nn.init.xavier_uniform_(self.conv2.weight)
        else:  # deprecated (likely to drop in a local-minimal trap)
            nn.init.zeros_(self.conv1.weight)
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class BasicBlock(nn.Module):
    # Basic Block for ResNet
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MLmodelResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(MLmodelResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ReshapeH():
    def __init__(self, empty_h):
        self.empty_h = empty_h

    def reshape_to_fill(self, h, subnodes):
        if h.shape == self.empty_h.shape: # only NC will have misaligned NE
            return h
        
        # reshaped = self.empty_h[layer]
        reshaped = copy.deepcopy(self.empty_h)
        reshaped[subnodes] = h

        return reshaped

class ROLANDGNN(torch.nn.Module):
    def __init__(self, device, input_dim, num_nodes, output_dim, dropout=0.0, update='moving', loss=nn.BCEWithLogitsLoss):
        
        super(ROLANDGNN, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess BERT repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        self.device = device
        hidden_conv_1 = 64
        hidden_conv_2 = 32
        self.preprocess1 = Linear(input_dim, 256).to(self.device)
        self.preprocess2 = Linear(256, 128).to(self.device)
        self.conv1 = GCNConv(128, hidden_conv_1).to(self.device) #, add_self_loops=False
        self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2).to(self.device) #, add_self_loops=False
        self.postprocess1 = Linear(hidden_conv_2, output_dim).to(self.device)
        
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
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update]).to(self.device)
        self.previous_embeddings = [torch.zeros((num_nodes, hidden_conv_1)).to(self.device), torch.zeros((num_nodes, hidden_conv_2)).to(self.device)]
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()

    def customise_pe(self, num_nodes):
        empty_h = torch.zeros((num_nodes, 64)).to(self.device) # To reshape after the 1st conv layer
        self.reshape = ReshapeH(empty_h)

        self.previous_embeddings = [torch.zeros((num_nodes, 64)).to(self.device), torch.zeros((num_nodes, 32)).to(self.device)]


    def forward(self, x, edge_index, task_type, edge_label_index=None, subnodes=None, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone().to(self.device),previous_embeddings[1].clone().to(self.device)]
            train = True
        else:
            train = False
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone().to(self.device) # tau -- past weight
        if not isinstance(self.update, str):
            self.tau = torch.Tensor([self.update]).to(self.device)
        
        current_embeddings = [torch.Tensor([]).to(self.device),torch.Tensor([]).to(self.device)]
        
        #Preprocess text
        h = self.preprocess1(x)
        h = F.leaky_relu(h,inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h,inplace=True)
        h = F.dropout(h, p=self.dropout, inplace=True)

        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h,inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True)

        #Reshape h for NC (clients have different number of nodes)
        h = self.reshape.reshape_to_fill(h, subnodes)

        #Embedding Update after first layer (only when training)
        if train == True:
            if self.update=='gru':
                h = self.gru1(h, self.previous_embeddings[0].clone()).detach()
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[0].clone()),dim=1)
                h = self.mlp1(hin).detach()
            else:
                h = (self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h.clone()).detach()
    
        current_embeddings[0] = h.clone()
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h,inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True)

        #Embedding Update after second layer (only when training)
        if train == True:
            if self.update=='gru':
                h = self.gru2(h, self.previous_embeddings[1].clone()).detach()
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[1].clone()),dim=1)
                h = self.mlp2(hin).detach()
            else:
                h = (self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h.clone()).detach()
    
        current_embeddings[1] = h.clone()

        #HADAMARD MLP (For Link Prediction)
        if task_type == "LP":
            h_src = h[edge_label_index[0]].to(self.device)
            h_dst = h[edge_label_index[1]].to(self.device)
            h_hadamard = torch.mul(h_src, h_dst) #hadamard product
            h = self.postprocess1(h_hadamard)
            out = torch.sum(h.clone(), dim=-1).clone()
        elif task_type == "NC":
            out = self.postprocess1(h)
        else:
            print('E> Invalid task type specified. Options are {LP, NC}')
            exit(-1)
        
        #return both 
        #i) the predictions for the current snapshot
        #ii) the embeddings of current snapshot

        return out, current_embeddings