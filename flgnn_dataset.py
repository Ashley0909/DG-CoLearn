import os
import copy
from collections import defaultdict
import numpy as np
import torch
import metis
from torch_geometric import datasets as torchgeometric_datasets
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from fl_clients import EdgeDevice
from utils import process_data, tensor_difference
from graph_partition import our_gpa
from plot_graphs import draw_graph

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
        hidden_conv1, hidden_conv2 = 128, 128 #64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]),torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(data[0].num_nodes)])]
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
        hidden_conv1, hidden_conv2 = 128, 128 #64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
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

    return num_snapshots, train_list, val_list, test_list, data_size, {'last_embeddings': last_embeddings, 'num_nodes': data[0].num_nodes, 'y': label}

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

def get_cut_edges(node_assignment, coo_format):
    '''
    Takes as input:
    1) node_assignment where i th index refers to node i and node_assignment[i] is client it's assigned to
    2) coo_format = 2d list where first list is start edges and
    
    Output: Dictionary of lists, ith key is the start node and ith value is the list of cutting nodes connecting it
    '''
    ccn_dict = defaultdict(list)
    for start_node, end_node in zip(coo_format[0], coo_format[1]):
        if node_assignment[start_node] != node_assignment[end_node]:
            ccn_dict[start_node].append(end_node)
            # ccn_dict[end_node].append(start_node)
    return ccn_dict

def get_gnn_clientdata(train_data, val_data, test_data, env_cfg, clients, prev_nodes=0, prev_partition=None):
    '''A function that first partition the graph to clients, then allocate edges to each clients accordingly.
    
    Hightlighted Inputs:
    1. prev_nodes: number of node the test graph has recorded in the previous snapshot (Default=0)
    2. prev_partition: the node allocation recorded in the previous snapshot (Default=None)
    '''
    print("Total number of edges", train_data.edge_index.shape[1])
    draw_graph(train_data.edge_index, 'global')
    
    num_subgraphs = len(clients)
    train_subgraphs, train_nodes = graph_partition(train_data.edge_index, train_data.num_nodes, num_subgraphs, node_label=train_data.y, prev_nodes=prev_nodes, prev_partition=prev_partition)
    val_subgraphs, val_nodes = graph_partition(val_data.edge_index, val_data.num_nodes, num_subgraphs, node_label=val_data.y, prev_nodes=train_nodes, prev_partition=train_subgraphs)
    test_subgraphs, test_nodes = graph_partition(test_data.edge_index, test_data.num_nodes, num_subgraphs, node_label=test_data.y, prev_nodes=val_nodes, prev_partition=val_subgraphs)

    if env_cfg.mode == 'FLDGNN-LP': # For LP partition, split negative edges
        train_neg_edges = tensor_difference(train_data.edge_label_index, train_data.edge_index)
        val_neg_edges = tensor_difference(val_data.edge_label_index, val_data.edge_index)
        test_neg_edges = tensor_difference(test_data.edge_label_index, test_data.edge_index)

        neg_train_subgraphs, _ = graph_partition(train_neg_edges, train_data.num_nodes, num_subgraphs, node_label=train_data.y)
        neg_val_subgraphs, _ = graph_partition(val_neg_edges, val_data.num_nodes, num_subgraphs, node_label=val_data.y)
        neg_test_subgraphs, _ = graph_partition(test_neg_edges, test_data.num_nodes, num_subgraphs, node_label=test_data.y)

        for train_nodes in torch.unique(train_neg_edges):
            train_subgraphs[train_nodes] = neg_train_subgraphs[train_nodes]
        for val_nodes in torch.unique(val_neg_edges):
            val_subgraphs[val_nodes] = neg_val_subgraphs[val_nodes]
        for test_nodes in torch.unique(test_neg_edges):
            test_subgraphs[test_nodes] = neg_test_subgraphs[test_nodes]

    cc_edges_train = get_cut_edges(train_subgraphs.tolist(), train_data.edge_index.tolist())

    client_sizes = [] # the training data size of each client (used for weighted aggregation)
    client_train, client_val, client_test = [], [], []

    # According to LP or NC, we allocate the train, val and test to FLLPDataset or FLNCDataset
    for i in range(num_subgraphs): # for each client, allocate subgraph
        single_train = construct_single_client_data(train_data, train_subgraphs, i, clients, "train", env_cfg.mode)
        client_train.append(single_train)
        single_val = construct_single_client_data(val_data, val_subgraphs, i, clients, "val", env_cfg.mode)
        client_val.append(single_val)
        single_test = construct_single_client_data(test_data, test_subgraphs, i, clients, "test", env_cfg.mode)
        client_test.append(single_test)

        if env_cfg.mode == 'FLDGNN-LP':
            client_sizes.append(single_train.ellength)
        elif env_cfg.mode == 'FLDGNN-NC':
            client_sizes.append(len(single_train.subnodes))

    fed_train = FLFedDataset(client_train)
    fed_val = FLFedDataset(client_val)
    fed_test = FLFedDataset(client_test)

    return fed_train, fed_val, fed_test, client_sizes, cc_edges_train, test_subgraphs, test_nodes
            
def construct_single_client_data(data, subgraph_label, client_idx, clients, tvt_mode, task_type):
    node_mask = (subgraph_label == client_idx)
    subnodes = torch.arange(data.num_nodes)[node_mask]

    ei_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    subgraph_ei = data.edge_index[:, ei_mask]

    if task_type == "FLDGNN-LP":
        el_mask = node_mask[data.edge_label_index[0]] & node_mask[data.edge_label_index[1]]
        fed_data =  FLLPDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], data.edge_label_index[:, el_mask], data.edge_label[el_mask], clients[client_idx].prev_edge_index, clients[client_idx])

    elif task_type == "FLDGNN-NC":
        fed_data = FLNCDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], clients[client_idx].prev_edge_index, data.y[node_mask], clients[client_idx])
    
    if tvt_mode == "train":
        clients[client_idx].prev_edge_index = subgraph_ei # Clients get new edge index in training by comparing with prev_edge_index
        clients[client_idx].subnodes = subnodes
    
    return fed_data

def graph_partition(edge_index, num_nodes, num_parts, partition_type='Ours', node_label=None, prev_nodes=0, prev_partition=None):
    """ 
    Stay consistent partition for TVT, so prev_partition is to record the partition of testing data (the most recent snapshot)

    Inputs:
    1. edge_index: COO format of edges
    2. num_nodes: Number of nodes in the data
    3. num_parts: Number of desired subgraphs
    4. partition_type: Type of Partitioning Algorithm (Options={'Metis', 'Ours'}) (Default='Ours')
    5. node_label: Node Labels in NC problems to help our graph partitioning (Default=None)
    6. edge_label: Edge Labels in LP problems (Default=None)
    6. prev_partition: Partitioning Labels of training or validation for consistency if there exists (Default=None)

    Outputs:
    1. partitioning_labels: Tensor array of subgraph assignment of each node
    2. num_nodes: Number of nodes in the data
    """
    # If previous partition exists, maintain consistency
    if prev_partition is not None and num_nodes == prev_nodes:
        return prev_partition, num_nodes

    # Convert graph to undirected for partitioning
    undirected_ei = to_undirected(edge_index)

    # Build adjacency list
    adjacency_list = [set() for _ in range(num_nodes)]
    for src, dst in undirected_ei.t().tolist():
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    adjacency_list = [list(neigh) for neigh in adjacency_list]

    if partition_type == 'Metis':
        _, partitioning_labels = metis.part_graph(adjacency_list, num_parts)
    elif partition_type == 'Ours':
        partitioning_labels = our_gpa(adjacency_list, node_labels=node_label, K=num_parts)
    else:
        print('E> Invalid partitioning algorithm specified. Options are {Metis, Ours}')
        exit(-1)

    return torch.tensor(partitioning_labels), num_nodes
