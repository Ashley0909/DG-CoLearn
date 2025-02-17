import os
import copy
import numpy as np
import torch
import metis
from torch_geometric import datasets as torchgeometric_datasets
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from fl_clients import EdgeDevice
from utils import sample_dirichlet, normalize, process_data, localise_idx

class FLLPDataset:
    def __init__(self, x, subnodes, edge_index, edge_label_index, edge_label, previous_edge_index, client=None):
        self.x = x
        self.subnodes = subnodes
        self.edge_index = edge_index  # edge_indces of pos edges
        self.edge_label_index = edge_label_index # edge indices of pos and neg edges
        self.y = edge_label  # 1/0 labels of combined edges (pos and neg)
        self.previous_edge_index = previous_edge_index
        self.length = x.shape[0]
        self.eilength = edge_index.shape[1]
        self.ellength = edge_label.shape[0]
        self.location = client

    def __len__(self):
        return len(self.x)

    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, EdgeDevice)
        self.location = client

class FLNCDataset:
    def __init__(self, x, subnodes, edge_index, previous_edge_index, y, client=None):
        self.x = x
        self.subnodes = subnodes
        self.edge_index = edge_index
        self.y = y
        self.previous_edge_index = previous_edge_index
        self.location = client

    def __len__(self):
        return len(self.x)
    
    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, EdgeDevice)
        self.location = client

class FLFedDataset:
    def __init__(self, fbd_list):
        self.fbd_list = fbd_list  # a list of FLBaseDatasets / FLLPDataset / FLNCDataset
        self.total_datasize = 0
        for fbd in self.fbd_list:
            self.total_datasize += len(fbd)  # mnist: train=60000, test=10000
        
    def __len__(self):
        return len(self.fbd_list)
    
    def __getitem__(self, item):
        return self.fbd_list[item]

def load_gnndata(task_cfg):
    if task_cfg.task_type == 'LP':
        if task_cfg.dataset == 'bitcoinOTC':
            data = torchgeometric_datasets.BitcoinOTC(task_cfg.path)
        elif task_cfg.dataset == 'UCI':
            # path = download_url('http://snap.stanford.edu/data/CollegeMsg.txt.gz', task_cfg.path) # Download data if needed
            # extract_gz(path, task_cfg.path)
            # os.unlink(path)
            txt_path = os.path.join(task_cfg.path, "CollegeMsg.txt")
            data = process_data(txt_path)
        else:
            print('E> Invalid link prediction dataset specified. Options are {bitcoinOTC, UCI}')
            exit(-1)

        num_snapshots = len(data)
        label = 2 # positive or negative edges
        hidden_conv1, hidden_conv2 = 64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]),torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(data[0].num_nodes)])]
        weights = torch.zeros(data[0].num_nodes)
        task_cfg.in_dim = data[0].num_node_features
        task_cfg.out_dim = data[0].num_nodes  # number of nodes is not the output dimension, I just used out_dim to store num_nodes for init_global_model

    elif task_cfg.task_type == 'NC':
        data = np.load('./data/{}.npz'.format(task_cfg.dataset))
        adjs = data['adjs']
        feature = data['attmats']
        label = data['labels']
        assert adjs.shape[1] == adjs.shape[2] == feature.shape[0] == label.shape[0]
        assert adjs.shape[0] == feature.shape[1]
        num_snapshots = adjs.shape[0]
        num_nodes = feature.shape[0]
        print("total number of nodes", num_nodes)
        hidden_conv1, hidden_conv2 = 64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
        weights = torch.zeros(num_nodes)
        task_cfg.in_dim = feature.shape[2]
        task_cfg.out_dim = num_nodes  # number of nodes is not the output dimension, I just used out_dim to store num_nodes for init_global_model

        for node in range(num_nodes):
            adjs[:, node, node] = 0
        adjs = [adjs[t, :, :] for t in range(adjs.shape[0])]
        feature = [feature[:, t, :] for t in range(feature.shape[1])]
        label = np.argmax(label, axis=1)
        task_cfg.num_classes = max(label) + 1
        adjs = [torch.tensor(adj, dtype=torch.long).to_sparse() for adj in adjs]
        indices = [adj.indices() for adj in adjs]
        data = [Data(edge_index=index, num_nodes=num_nodes) for index in indices]
        feature = [torch.tensor(feat, dtype=torch.float) for feat in feature]
        label = torch.tensor(label, dtype=torch.long)
        for graph, feat in zip(data, feature):
            graph.x = feat
            graph.y = label

    """ Split each snapshot into train, val and test """
    train_list, val_list, test_list, data_size = partition_data(task_cfg.task_type, num_snapshots, data) # data_size is to record the number of training data (#edge/#node)

    return num_snapshots, train_list, val_list, test_list, data_size, {'last_embeddings': last_embeddings, 'weights': weights, 'num_nodes': data[0].num_nodes, 'y': label}

def partition_data(task_type, num_snapshots, data):
    """ Partition data in train using t, val using t+1 and test using t+2 """
    train_list, val_list, test_list, data_size = [], [], [], []

    for i in range(num_snapshots - 2): # There are num_snapshots rounds of training
        g_t0 = copy.deepcopy(data[i])
        g_t1 = copy.deepcopy(data[i+1])
        g_t2 = copy.deepcopy(data[i+2])

        if task_type == 'LP':
            g_t0.x = torch.Tensor([[1] for _ in range(g_t0.num_nodes)])
            g_t1.x = torch.Tensor([[1] for _ in range(g_t1.num_nodes)])
            g_t2.x = torch.Tensor([[1] for _ in range(g_t2.num_nodes)])

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # All for training in time t
            train_data, _, _ = transform(g_t0)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # All for validation in time t+1
            val_data, _, _ = transform(g_t1)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # All for test in time t+2
            test_data, _, _ = transform(g_t2)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)
            data_size.append(len(train_data.edge_label)) # Set the data size as the size of training edge label (pos and neg edges)

        elif task_type == 'NC':
            train_list.append(g_t0)
            val_list.append(g_t1)
            test_list.append(g_t2)
            data_size.append(g_t0.num_nodes)
        else:
            print('E> Invalid task type specified. Options are {LP, NC}')
            exit(-1)
    
    return train_list, val_list, test_list, data_size

def get_gnn_clientdata(train_data, val_data, test_data, env_cfg, clients):
    """ First allocate subnodes to clients, then allocate TVT to each client according to subnodes """
    num_subgraphs = len(clients)
    train_subgraphs = metis_partition(train_data.edge_index, train_data.num_nodes, num_subgraphs, None)
    val_subgraphs = metis_partition(val_data.edge_index, val_data.num_nodes, num_subgraphs, train_subgraphs)
    test_subgraphs = metis_partition(test_data.edge_index, test_data.num_nodes, num_subgraphs, val_subgraphs)

    client_sizes = [] # the training data size of each client (used for weighted aggregation)
    client_train, client_val, client_test = [], [], []

    # According to LP or NC, we allocate the train, val and test to FLLPDataset or FLNCDataset
    for i in range(num_subgraphs): # for each client, allocate subgraph
        single_train = construct_single_client_data(train_data, train_subgraphs, i, clients, "train", env_cfg.mode)
        client_train.append(single_train)
        if env_cfg.mode == 'FLDGNN-LP':
            client_sizes.append(single_train.ellength)
        elif env_cfg.mode == 'FLDGNN-NC':
            client_sizes.append(len(single_train.subnodes))
        single_val = construct_single_client_data(val_data, val_subgraphs, i, clients, "val", env_cfg.mode)
        client_val.append(single_val)
        single_test = construct_single_client_data(test_data, test_subgraphs, i, clients, "test", env_cfg.mode)
        client_test.append(single_test)

    fed_train = FLFedDataset(client_train)
    fed_val = FLFedDataset(client_val)
    fed_test = FLFedDataset(client_test)

    return fed_train, fed_val, fed_test, client_sizes
            
def construct_single_client_data(data, subgraph_label, client_idx, clients, tvt_mode, task_type):
    node_mask = (subgraph_label == client_idx)
    subnodes = torch.arange(data.num_nodes)[node_mask]

    ei_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    subgraph_ei = data.edge_index[:, ei_mask]
    
    if tvt_mode == "train":
        clients[client_idx].compute_weights(subgraph_ei) # Compute Weights for each client
        clients[client_idx].prev_edge_index = subgraph_ei # Clients get new edge index in training by comparing with prev_edge_index

    if task_type == "FLDGNN-LP":
        # Also get Edge Label
        el_mask = node_mask[data.edge_label_index[0]] & node_mask[data.edge_label_index[1]]

        return FLLPDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], data.edge_label_index[:, el_mask], data.edge_label[el_mask], clients[client_idx].prev_edge_index, clients[client_idx])

    elif task_type == "FLDGNN-NC":
        return FLNCDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], clients[client_idx].prev_edge_index, data.y[node_mask], clients[client_idx])

def metis_partition(edge_index, num_nodes, num_parts, prev_partition=None):
    """ Stay consistent partition for TVT, so prev_partition is to record the partition of training data """
    # Convert graph to undirected for METIS partitioning
    undirected_ei = to_undirected(edge_index)

    # Build adjacency list
    adjacency_list = [[] for _ in range(num_nodes)]
    for src, dst in undirected_ei.t().tolist():
        adjacency_list[src].append(dst)
        adjacency_list[dst].append(src)

    # Run METIS for initial partitioning
    _, partitioning_labels = metis.part_graph(adjacency_list, num_parts)

    # If previous partition exists, maintain consistency
    if prev_partition is not None:
        for node in range(num_nodes):
            if prev_partition[node] is not None:
                partitioning_labels[node] = prev_partition[node]

    return torch.tensor(partitioning_labels)
