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
from utils import process_data, generate_neg_edges
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
        task_cfg.num_classes = label
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
    train_list, val_list, test_list = partition_data(task_cfg.task_type, num_snapshots, data)

    return num_snapshots, train_list, val_list, test_list, {'last_embeddings': last_embeddings, 'num_nodes': data[0].num_nodes, 'y': label}

def partition_data(task_type, num_snapshots, data):
    """ Partition data in train using t, val using t+1 and test using t+2 """
    train_list, val_list, test_list = [], [], []

    for i in range(num_snapshots - 2): # There are num_snapshots rounds of training
        g_t0 = copy.deepcopy(data[i])
        g_t1 = copy.deepcopy(data[i+1])
        g_t2 = copy.deepcopy(data[i+2])

        if task_type == 'LP':
            g_t0.x = torch.Tensor([[1] for _ in range(g_t0.num_nodes)])
            g_t1.x = torch.Tensor([[1] for _ in range(g_t1.num_nodes)])
            g_t2.x = torch.Tensor([[1] for _ in range(g_t2.num_nodes)])

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for training in time t
            train_data, _, _ = transform(g_t0)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for validation in time t+1
            val_data, _, _ = transform(g_t1)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for test in time t+2
            test_data, _, _ = transform(g_t2)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)

        elif task_type == 'NC':
            train_list.append(g_t0)
            val_list.append(g_t1)
            test_list.append(g_t2)
        else:
            print('E> Invalid task type specified. Options are {LP, NC}')
            exit(-1)
    
    return train_list, val_list, test_list

def get_cut_edges(node_assignment, coo_format):
    '''
    Takes as input:
    1) node_assignment where i th index refers to node i and node_assignment[i] is client it's assigned to
    2) coo_format = 2d list where first list is start edges and
    
    Output: Dictionary of lists, ith key is the start node and ith value is the list of cutting nodes connecting it
    '''
    ccn_dict = defaultdict(list)
    for start_node, end_node in zip(coo_format[0], coo_format[1]):
        if (node_assignment[start_node] != node_assignment[end_node]) and (end_node not in ccn_dict[start_node]):
            ccn_dict[start_node].append(end_node)
    return ccn_dict

def get_gnn_clientdata(train_data, val_data, test_data, env_cfg, clients, prev_num_subgraphs=0, prev_partition=None):
    '''A function that first partition the graph to clients, then allocate edges to each clients accordingly.
    
    Hightlighted Inputs:
    1. prev_nodes: number of node the test graph has recorded in the previous snapshot (Default=0)
    2. prev_partition: the node allocation recorded in the previous snapshot (Default=None)
    '''
    num_subgraphs = gen_train_clients(train_data.edge_index.shape[1], len(clients))
    global_size = train_data.edge_index.shape[1]
    print(f"A total of {global_size} training edges")
    print(num_subgraphs, "clients are chosen to train")
    data_size = train_data.num_nodes # The total number of nodes in the global training graph

    train_subgraphs, train_clients = graph_partition(train_data.edge_index, train_data.num_nodes, num_subgraphs, node_label=train_data.y, prev_num_subgraphs=prev_num_subgraphs, prev_partition=prev_partition)
    val_subgraphs, val_clients = graph_partition(val_data.edge_index, val_data.num_nodes, num_subgraphs, node_label=val_data.y, prev_num_subgraphs=train_clients, prev_partition=train_subgraphs)
    test_subgraphs, test_clients = graph_partition(test_data.edge_index, test_data.num_nodes, num_subgraphs, node_label=test_data.y, prev_num_subgraphs=val_clients, prev_partition=val_subgraphs)

    cc_edges_train = get_cut_edges(train_subgraphs.tolist(), train_data.edge_index.tolist())
    print(f"Total number of cut edges: {sum(len(v) for v in cc_edges_train.values())}")
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
        print(f"Client {i} has {single_train.edge_index.shape[1]} positive training edges, {single_val.edge_index.shape[1]} positive val edges and {single_test.edge_index.shape[1]} positive test edges")

        client_sizes.append(len(single_train.subnodes)) # Client data size is the number of subnodes a client has

    fed_train = FLFedDataset(client_train)
    fed_val = FLFedDataset(client_val)
    fed_test = FLFedDataset(client_test)

    return fed_train, fed_val, fed_test, client_sizes, cc_edges_train, test_subgraphs, test_clients, data_size
       
def construct_single_client_data(data, subgraph_label, client_idx, clients, tvt_mode, task_type):
    node_mask = (subgraph_label == client_idx)
    subnodes = torch.arange(data.num_nodes)[node_mask]

    ei_mask = []
    for src, targ in zip(data.edge_index[0], data.edge_index[1]):
        if node_mask[src] == True and node_mask[targ] == True:
            ei_mask.append(True)
        else:
            ei_mask.append(False)

    subgraph_ei = data.edge_index[:, ei_mask]

    if task_type == "FLDGNN-LP":
        # Generate Negative Edges
        negative_edges = generate_neg_edges(data.edge_index[:, ei_mask], subnodes, data.edge_index[:, ei_mask].size(1))
        edge_label_index = torch.cat([data.edge_index[:, ei_mask], negative_edges], dim=1)
        edge_label = torch.concat([data.edge_label[ei_mask], torch.zeros(data.edge_index[:, ei_mask].size(1))])
        fed_data =  FLLPDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], edge_label_index, edge_label, clients[client_idx].prev_edge_index, clients[client_idx])

    elif task_type == "FLDGNN-NC":
        fed_data = FLNCDataset(data.x[node_mask], subnodes, data.edge_index[:, ei_mask], clients[client_idx].prev_edge_index, data.y[node_mask], clients[client_idx])
    
    if tvt_mode == "train":
        clients[client_idx].prev_edge_index = subgraph_ei # Clients get new edge index in training by comparing with prev_edge_index
        clients[client_idx].subnodes = subnodes
    
    return fed_data

def graph_partition(edge_index, num_nodes, num_parts, partition_type='Ours', node_label=None, prev_num_subgraphs=0, prev_partition=None):
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
    if prev_partition is not None and num_parts == prev_num_subgraphs:
        return prev_partition, num_parts

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
        print("Partition Graph")
        partitioning_labels = our_gpa(copy.deepcopy(adjacency_list), edge_index.shape[1], node_labels=node_label, K=num_parts)
    else:
        print('E> Invalid partitioning algorithm specified. Options are {Metis, Ours}')
        exit(-1)

    return torch.tensor(partitioning_labels), num_parts

def gen_train_clients(total_num_edges, max_num_clients, num_edge_per_clients=500):
    ''' Determine number of training clients in this snapshot based on the total number of edges in the global graph '''
    num_clients = total_num_edges // num_edge_per_clients
    
    return min(max(1,num_clients), max_num_clients)