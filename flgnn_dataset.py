import os
import copy
from collections import defaultdict
import numpy as np
import torch
import metis
import networkx as nx
from torch_geometric.data import Data
import community as community_louvain
from torch_geometric import datasets as torchgeometric_datasets

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from fl_clients import EdgeDevice
from utils import process_txt_data, process_csv_data, download_url, extract_gz, generate_neg_edges, compute_label_weights, label_dirichlet_partition
from graph_partition import our_gpa
# from plot_graphs import draw_graph

class FLLPDataset():
    def __init__(self, node_feature, subnodes, edge_index, edge_label_index, edge_label, previous_edge_index, client=None):
        self.node_feature = node_feature
        self.subnodes = subnodes
        self.edge_index = edge_index  # edge_indces of pos edges
        self.edge_label_index = edge_label_index # edge indices of pos and neg edges
        self.node_label = edge_label  # 1/0 labels of combined edges (pos and neg)
        self.previous_edge_index = previous_edge_index
        self.length = node_feature.shape[0]
        self.eilength = edge_index.shape[1]
        self.ellength = edge_label.shape[0]
        self.location = client

    def __len__(self):
        return len(self.node_feature)

    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, EdgeDevice)
        self.location = client

class FLNCDataset():
    def __init__(self, node_feature, subnodes, edge_index, previous_edge_index, node_label, class_weights, client=None):
        self.node_feature = node_feature
        self.subnodes = subnodes
        self.edge_index = edge_index
        self.node_label = node_label
        self.class_weights = class_weights
        self.previous_edge_index = previous_edge_index
        self.location = client

    def __len__(self):
        return len(self.node_feature)
    
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
            if not os.path.isdir("data/UCI"):
                path = download_url('http://snap.stanford.edu/data/CollegeMsg.txt.gz', task_cfg.path) # Download data if needed
                extract_gz(path, task_cfg.path)
                os.unlink(path)
            txt_path = os.path.join(task_cfg.path, "CollegeMsg.txt")
            data = process_txt_data(txt_path)
        elif task_cfg.dataset == 'bitcoinAlpha':
            if not os.path.isdir("data/bitcoinAlpha"):
                path = download_url('https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz', task_cfg.path)
                extract_gz(path, task_cfg.path)
                os.unlink(path)
            csv_path = os.path.join(task_cfg.path, "soc-sign-bitcoinalpha.csv")
            data = process_csv_data(csv_path)
        else:
            print('E> Invalid link prediction dataset specified. Options are {bitcoinOTC, UCI}')
            exit(-1)

        num_snapshots = len(data)
        label = 2 # positive or negative edges
        task_cfg.num_classes = label
        hidden_conv1, hidden_conv2 = 128, 128 #64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]),torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(data[0].num_nodes)])]
        task_cfg.out_dim = 1
        num_nodes = data[0].num_nodes

    elif task_cfg.task_type == 'NC':
        data = np.load('./data/{}.npz'.format(task_cfg.dataset))
        adjs = data['adjs']
        feature = data['attmats']
        label = data['labels']
        assert adjs.shape[1] == adjs.shape[2] == feature.shape[0] == label.shape[0] # number of nodes
        assert adjs.shape[0] == feature.shape[1]
        num_snapshots = adjs.shape[0]
        num_nodes = feature.shape[0]
        print("total number of nodes", num_nodes)
        hidden_conv1, hidden_conv2 = 128, 128 #64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
        task_cfg.in_dim = feature.shape[2]

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
        task_cfg.out_dim = max(label).item() + 1
        for graph, feat in zip(data, feature):
            graph.node_feature = feat
            graph.node_label = label

    """ Split each snapshot into train, val and test """
    train_list, val_list, test_list = partition_data(task_cfg, num_snapshots, data)

    return num_snapshots, train_list, val_list, test_list, {'last_embeddings': last_embeddings, 'num_nodes': num_nodes, 'node_label': label}

def partition_data(task_cfg, num_snapshots, data):
    """ Partition data in train using t, val using t+1 and test using t+2 """
    train_list, val_list, test_list = [], [], []

    for i in range(num_snapshots - 2): # There are num_snapshots rounds of training
        g_t0 = copy.deepcopy(data[i])
        g_t1 = copy.deepcopy(data[i+1])
        g_t2 = copy.deepcopy(data[i+2])

        if task_cfg.task_type == 'LP':
            g_t0.node_feature = torch.Tensor([[1 for _ in range(16)] for _ in range(g_t0.num_nodes)])
            g_t1.node_feature = torch.Tensor([[1 for _ in range(16)] for _ in range(g_t1.num_nodes)])
            g_t2.node_feature = torch.Tensor([[1 for _ in range(16)] for _ in range(g_t2.num_nodes)])

            task_cfg.in_dim = 16 # Set it to be the size of the input node feature

            g_t0.edge_feature = torch.Tensor([[1 for _ in range(128)] for _ in range(g_t0.edge_index.shape[1])])
            g_t1.edge_feature = torch.Tensor([[1 for _ in range(128)] for _ in range(g_t1.edge_index.shape[1])])
            g_t2.edge_feature = torch.Tensor([[1 for _ in range(128)] for _ in range(g_t2.edge_index.shape[1])])

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for training in time t
            train_data, _, _ = transform(g_t0)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for validation in time t+1
            val_data, _, _ = transform(g_t1)
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0, add_negative_train_samples=False)  # All for test in time t+2
            test_data, _, _ = transform(g_t2)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)

        elif task_cfg.task_type == 'NC':
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
    coo_ccn = [[], []]
    ccn_label = []
    ccn_dict = defaultdict(list)
    for start_node, end_node in zip(coo_format[0], coo_format[1]):
        if (node_assignment[start_node] != node_assignment[end_node]):
            ccn_dict[start_node].append(end_node)
            coo_ccn[0].append(start_node)
            coo_ccn[1].append(end_node)
            ccn_label.append(1)

    return ccn_dict, torch.tensor(coo_ccn), torch.tensor(ccn_label)

def get_gnn_clientdata(server, train_data, val_data, test_data, env_cfg, task_cfg, clients):
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
    # server.construct_global_adj_matrix(train_data.edge_index, data_size)
    server.record_num_nodes(data_size)

    train_subgraphs = graph_partition(server, train_data.edge_index, train_data.num_nodes, num_subgraphs, node_label=train_data.node_label if env_cfg.mode == 'FLDGNN-NC' else None, tvt_type='train')
    server.record_num_subgraphs(num_subgraphs)
    server.construct_client_adj_matrix(train_subgraphs)
    val_subgraphs = graph_partition(server, val_data.edge_index, val_data.num_nodes, num_subgraphs, node_label=val_data.node_label if env_cfg.mode == 'FLDGNN-NC' else None)
    test_subgraphs = graph_partition(server, test_data.edge_index, test_data.num_nodes, num_subgraphs, node_label=test_data.node_label if env_cfg.mode == 'FLDGNN-NC' else None)

    ''' Server gets cce and construct server-side test data '''
    cc_edges_train, _, _ = get_cut_edges(train_subgraphs.tolist(), train_data.edge_index.tolist())
    print(f"Total number of cut edges: {sum(len(v) for v in cc_edges_train.values())}")
    server.record_ccn(cc_edges_train)

    cce_test, server_ei, server_el = get_cut_edges(test_subgraphs.tolist(), test_data.edge_index.tolist())
    server.construct_ccn_test_data(task_cfg.in_dim, server_ei, server_el, cce_test.keys())

    client_sizes = [] # the training data size of each client (used for weighted aggregation)
    client_train, client_val, client_test = [], [], []
    # According to LP or NC, we allocate the train, val and test to FLLPDataset or FLNCDataset
    for i in range(num_subgraphs): # for each client, allocate subgraph
        single_train = construct_single_client_data(task_cfg, train_data, train_subgraphs, i, clients, "train", env_cfg.mode)
        client_train.append(single_train)
        single_val = construct_single_client_data(task_cfg, val_data, val_subgraphs, i, clients, "val", env_cfg.mode)
        client_val.append(single_val)
        single_test = construct_single_client_data(task_cfg, test_data, test_subgraphs, i, clients, "test", env_cfg.mode)
        client_test.append(single_test)
        print(f"Client {i} has {single_train.dataset.edge_index.shape[1]} positive training edges, {single_val.dataset.edge_index.shape[1]} positive val edges and {single_test.dataset.edge_index.shape[1]} positive test edges")

        # client_sizes.append(len(single_train.dataset.subnodes)) # Client data size is the number of subnodes a client has
        client_sizes.append(single_train.dataset.edge_index.shape[1]) # Client data size is the number of training edges a client has

    fed_train = FLFedDataset(client_train)
    fed_val = FLFedDataset(client_val)
    fed_test = FLFedDataset(client_test)

    return fed_train, fed_val, fed_test, client_sizes, global_size # changed data_size to total number of edges, not nodes
       
def construct_single_client_data(task_cfg, data, subgraph_label, client_idx, clients, tvt_mode, task_type):
    node_mask = (subgraph_label == client_idx)
    subnodes = torch.arange(data.num_nodes)[node_mask]

    ei_mask = []
    for src, targ in zip(data.edge_index[0], data.edge_index[1]):
        if node_mask[src] == True and node_mask[targ] == True:
            ei_mask.append(True)
        else:
            ei_mask.append(False)

    subgraph_ei = data.edge_index[:, ei_mask]
    indim = task_cfg.in_dim

    if task_type == "FLDGNN-LP":
        # Generate Negative Edges
        negative_edges = generate_neg_edges(data.edge_index[:, ei_mask], subnodes, data.edge_index[:, ei_mask].size(1))
        edge_label_index = torch.cat([data.edge_index[:, ei_mask], negative_edges], dim=1)
        edge_label = torch.concat([data.edge_label[ei_mask], torch.zeros(data.edge_index[:, ei_mask].size(1))])
        # fed_data =  FLLPDataset(data.node_feature[node_mask], subnodes, data.edge_index[:, ei_mask], edge_label_index, edge_label, clients[client_idx].prev_edge_index, clients[client_idx])
        fed_data = Data(node_feature=data.node_feature[node_mask], edge_label_index=edge_label_index, edge_label=edge_label, subnodes=subnodes, 
                        edge_feature=data.edge_feature[ei_mask], edge_index=data.edge_index[:, ei_mask],  previous_edge_index=clients[client_idx].prev_edge_index,
                        node_states=[torch.zeros((data.num_nodes, indim)), torch.zeros((data.num_nodes, indim))],
                        location=clients[client_idx], keep_ratio=0.2)
        fed_data_loader = DataLoader(fed_data, batch_size=1)

    elif task_type == "FLDGNN-NC":
        class_weights = compute_label_weights(data.node_label[node_mask])
        # fed_data = FLNCDataset(data.node_feature[node_mask], subnodes, data.edge_index[:, ei_mask], clients[client_idx].prev_edge_index, data.node_label[node_mask], class_weights, clients[client_idx])
        fed_data = Data(node_feature=data.node_feature[node_mask], node_label_index=subnodes, node_label=data.node_label[node_mask], subnodes=subnodes, 
                        edge_index=data.edge_index[:, ei_mask],  previous_edge_index=clients[client_idx].prev_edge_index, 
                        class_weights=class_weights, location=clients[client_idx],
                        node_states=[torch.zeros((data.num_nodes, indim)), torch.zeros((data.num_nodes, indim))], keep_ratio=0.2)

        fed_data_loader = DataLoader(fed_data, batch_size=1)
    if tvt_mode == "train":
        clients[client_idx].prev_edge_index = subgraph_ei # Clients get new edge index in training by comparing with prev_edge_index
        clients[client_idx].subnodes = subnodes
    
    return fed_data_loader

def graph_partition(server, edge_index, num_nodes, num_parts, partition_type='Ours', node_label=None, tvt_type='test'):
    """ 
    Stay consistent partition for TVT, so prev_partition is to record the partition of testing data (the most recent snapshot)
    Input server instance to store current partition and adj_list if needed

    Inputs:
    1. edge_index: COO format of edges
    2. num_nodes: Number of nodes in the data
    3. num_parts: Number of desired subgraphs
    4. partition_type: Type of Partitioning Algorithm (Options={'Metis', 'Louvain, 'Ours'}) (Default='Ours')
    5. node_label: Node Labels in NC problems to help our graph partitioning (Default=None)
    6. edge_label: Edge Labels in LP problems (Default=None)
    6. prev_partition: Partitioning Labels of training or validation for consistency if there exists (Default=None)

    Outputs:
    1. partitioning_labels: Tensor array of subgraph assignment of each node
    2. num_nodes: Number of nodes in the data
    """
    # If previous partition exists, maintain consistency
    if partition_type == 'Ours' and server.node_assignment is not None and num_parts == server.num_subgraphs:
        return server.node_assignment

    # Convert graph to undirected for partitioning
    undirected_ei = to_undirected(edge_index)

    # Build adjacency list
    adjacency_list = [set() for _ in range(num_nodes)]
    for src, dst in undirected_ei.t().tolist():
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    adjacency_list = [list(neigh) for neigh in adjacency_list]

    if tvt_type == 'train':
        server.construct_glob_adj_mtx(adjacency_list)

    if partition_type == 'Metis':
        _, partitioning_labels = metis.part_graph(adjacency_list, num_parts)
    elif partition_type == 'Louvain':
        G = nx.Graph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        partition = community_louvain.best_partition(G)
        partitioning_labels = [partition[i] for i in range(len(G.nodes))]
    elif partition_type == 'Dirichlet':
        partitioning_labels = label_dirichlet_partition(node_label, len(node_label), max(node_label), num_parts, beta=100)
    elif partition_type == 'Ours':
        print("Partition Graph")
        partitioning_labels = our_gpa(copy.deepcopy(adjacency_list), edge_index.shape[1], node_labels=node_label, K=num_parts)
    else:
        print('E> Invalid partitioning algorithm specified. Options are {Metis, Ours}')
        exit(-1)

    return torch.tensor(partitioning_labels)

def gen_train_clients(total_num_edges, max_num_clients, num_edge_per_clients=510):
    ''' Determine number of training clients in this snapshot based on the total number of edges in the global graph '''
    num_clients = total_num_edges // num_edge_per_clients
    
    return min(max(1,num_clients), max_num_clients)