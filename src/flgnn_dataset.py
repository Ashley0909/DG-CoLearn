import os
import time
import copy
from collections import defaultdict, Counter
import numpy as np
import torch
from torch_geometric import datasets as torchgeometric_datasets
from torch_geometric.data import Data
from src.other_splitters.louvainSplitter import LouvainSplitter
from data.as733_processing.load_as import load_generic_dataset

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from src.utils.utils import (
    process_txt_data, download_url, extract_gz, generate_neg_edges, compute_label_weights,
    count_label_occur, extract_tar_gz, get_exclusive_subgraph
)
# from other_partition import label_split, custom_metis
import partition

class FLFedDataset:
    def __init__(self, fbd_list):
        self.fbd_list = fbd_list 
        self.total_datasize = 0
        for fbd in self.fbd_list:
            self.total_datasize += len(fbd)  # mnist: train=60000, test=10000
        
    def __len__(self):
        return len(self.fbd_list)
    
    def __getitem__(self, item):
        return self.fbd_list[item]

def load_gnndata(task_cfg):
    if not os.path.isdir(task_cfg.path):
        os.makedirs(task_cfg.path)

    if task_cfg.task_type == 'LP':
        if task_cfg.dataset.lower() == 'bitcoinotc':
            data = torchgeometric_datasets.BitcoinOTC(task_cfg.path)
        elif task_cfg.dataset.lower() == 'uci':
            if not os.path.exists(task_cfg.path):
                path = download_url('http://snap.stanford.edu/data/CollegeMsg.txt.gz', task_cfg.path) # Download data if needed
                extract_gz(path)
                os.unlink(path)
            txt_path = os.path.join(task_cfg.path, "CollegeMsg.txt")
            data = process_txt_data(txt_path)
        elif task_cfg.dataset == 'as733':
            if not os.path.exists(task_cfg.path):
                tar_path = download_url('https://snap.stanford.edu/data/as-733.tar.gz', task_cfg.path) # Download data if needed
                extract_tar_gz(tar_path, task_cfg.path)
                os.unlink(tar_path)
                
            data = load_generic_dataset(task_cfg.path)
        else:
            print('E> Invalid link prediction dataset specified. Options are {bitcoinOTC, UCI, as733}')
            exit(-1)

        num_snapshots = len(data)
        label = 2 # positive or negative edges
        task_cfg.num_classes = label
        hidden_conv1, hidden_conv2 = 128, 128 #64, 32
        last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]),torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(data[0].num_nodes)])]
        task_cfg.in_dim = 32 # Set it to be the size of the input node feature
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
    train_list, val_list, test_list = generate_tvt(task_cfg, num_snapshots, data)

    return num_snapshots, train_list, val_list, test_list, {'last_embeddings': last_embeddings, 'num_nodes': num_nodes}

def generate_tvt(task_cfg, num_snapshots, data):
    """ Partition data in train using t, val using t+1 and test using t+2 """
    train_list, val_list, test_list = [], [], []

    for i in range(num_snapshots - 2): # There are num_snapshots rounds of training
        g_t0 = copy.deepcopy(data[i])
        g_t1 = copy.deepcopy(data[i+1])
        g_t2 = copy.deepcopy(data[i+2])

        if task_cfg.task_type == 'LP':
            hasattr(g_t0, 'node_feature') or setattr(g_t0, 'node_feature', torch.Tensor([[1 for _ in range(task_cfg.in_dim)] for _ in range(g_t0.num_nodes)]))
            hasattr(g_t1, 'node_feature') or setattr(g_t1, 'node_feature', torch.Tensor([[1 for _ in range(task_cfg.in_dim)] for _ in range(g_t1.num_nodes)]))
            hasattr(g_t2, 'node_feature') or setattr(g_t2, 'node_feature', torch.Tensor([[1 for _ in range(task_cfg.in_dim)] for _ in range(g_t2.num_nodes)]))

            hasattr(g_t0, 'edge_feature') or setattr(g_t0, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t0.edge_index.shape[1])]))
            hasattr(g_t1, 'edge_feature') or setattr(g_t1, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t1.edge_index.shape[1])]))
            hasattr(g_t2, 'edge_feature') or setattr(g_t2, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t2.edge_index.shape[1])]))

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
            hasattr(g_t0, 'edge_feature') or setattr(g_t0, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t0.edge_index.shape[1])]))
            hasattr(g_t1, 'edge_feature') or setattr(g_t1, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t1.edge_index.shape[1])]))
            hasattr(g_t2, 'edge_feature') or setattr(g_t2, 'edge_feature', torch.Tensor([[1 for _ in range(128)] for _ in range(g_t2.edge_index.shape[1])]))

            train_list.append(g_t0)
            val_list.append(g_t1)
            test_list.append(g_t2)
        else:
            print('E> Invalid task type specified. Options are {LP, NC}')
            exit(-1)
    
    return train_list, val_list, test_list

def get_gnn_clientdata(server, train_data, val_data, test_data, task_cfg, clients):
    ''' A function that first partition the graph to clients, then allocate edges to each clients accordingly. '''
    num_subgraphs = gen_train_clients(train_data.edge_index.shape[1], len(clients))
    global_size = train_data.edge_index.shape[1]
    print(f"A total of {global_size} training edges")
    print(num_subgraphs, "clients are chosen to train")
    data_size = train_data.num_nodes # The total number of nodes in the global training graph
    # server.construct_global_adj_matrix(train_data.edge_index, data_size)
    server.record_num_nodes(data_size)

    ''' Check if the previous number of subgraphs is equal to current estimated number of subgraphs '''
    # [Ablation Study] Comment out this part to run ablation study on partitioning algorithm
    if hasattr(server, 'num_prev_subgraphs'):
        if server.num_prev_subgraphs >= num_subgraphs:
            print(f"Reusing previous partitioning with {num_subgraphs} clients...")
            server.record_reuse_bool(True)
            num_subgraphs = server.num_prev_subgraphs
        else:
            server.record_reuse_bool(False)
    else:
        server.record_reuse_bool(False)

    train_subgraphs = graph_partition(server, train_data, num_subgraphs, task_cfg.task_type, partition_type='Ours', tvt_type='train')
    server.record_prev_num_subgraphs(num_subgraphs)
    val_subgraphs = graph_partition(server, val_data, num_subgraphs, task_cfg.task_type, partition_type='Ours', tvt_type='val')
    test_subgraphs = graph_partition(server, test_data, num_subgraphs, task_cfg.task_type, partition_type='Ours', tvt_type='test')
    # server.construct_client_adj_matrix(train_subgraphs) # Only for model answer simulation

    if task_cfg.task_type == 'NC':
        count_label_occur(train_subgraphs, train_data.node_label)
    
    ''' Server gets cce and construct server-side test data '''
    cc_edges_train, _, _ = get_cut_edges(server, train_subgraphs.tolist(), train_data.edge_index.tolist())
    print(f"Total number of cut edges: {int(sum(len(v) for v in cc_edges_train.values())//2)}")
    server.record_ccn(cc_edges_train)

    cce_test, server_ei, server_el = get_cut_edges(server, test_subgraphs.tolist(), test_data.edge_index.tolist(), tvt_type='test')
    server.construct_ccn_test_data(task_cfg.in_dim, task_cfg.edge_dim, server_ei, server_el, cce_test.keys())

    client_sizes = [] # the training data size of each client (used for weighted aggregation)
    client_train, client_val, client_test = [], [], []
    
    for i in range(num_subgraphs): # for each client, allocate subgraph
        single_train = construct_single_client_data(server, train_data, train_subgraphs, i, clients, "train", task_cfg.task_type)
        client_train.append(single_train)
        single_val = construct_single_client_data(server, val_data, val_subgraphs, i, clients, "val", task_cfg.task_type)
        client_val.append(single_val)
        single_test = construct_single_client_data(server, test_data, test_subgraphs, i, clients, "test", task_cfg.task_type)
        client_test.append(single_test)

        print(f"Client {i} has {single_train.dataset.edge_index.shape[1]} positive training edges, {single_val.dataset.edge_index.shape[1]} positive val edges and {single_test.dataset.edge_index.shape[1]} positive test edges")
        client_sizes.append(single_train.dataset.edge_index.shape[1]) # Client data size is the number of training edges a client has

    fed_train = FLFedDataset(client_train)
    fed_val = FLFedDataset(client_val)
    fed_test = FLFedDataset(client_test)

    return fed_train, fed_val, fed_test, client_sizes, global_size

def graph_partition(server, data, num_parts, task_type, partition_type='Ours', tvt_type='test'):
    """ 
    Stay consistent partition for TVT, so prev_partition is to record the partition of testing data (the most recent snapshot)
    Input server instance to store current partition and adj_list if needed

    Output: partitioning_labels: Tensor array of subgraph assignment of each node
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    node_label = data.node_label if task_type == 'NC' else None

    # If reuse previous partitioning
    if partition_type == 'Ours' and server.reuse_partition:
        return reuse_partition(num_parts, server, data, tvt_type)

    # Convert graph to undirected for partitioning
    undirected_ei = to_undirected(edge_index)

    # Build adjacency list
    adjacency_list = [set() for _ in range(num_nodes)]
    for src, dst in undirected_ei.t().tolist():
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    adjacency_list = [list(neigh) for neigh in adjacency_list]

    # if tvt_type == 'train': # Only for model answer simulation
    #     server.construct_glob_adj_mtx(adjacency_list)

    start_time = time.time()
    # if partition_type == 'Metis':
    #     partitioning_labels = custom_metis(adjacency_list, num_parts)
    # elif partition_type == 'Louvain':
    #     louvainSplitter = LouvainSplitter(num_parts)
    #     partitioning_labels = louvainSplitter(data)
    if partition_type == 'Ours':
        node_labels = [] if node_label is None else node_label.tolist()
        labels = partition.CoLearnPartition(copy.deepcopy(adjacency_list), edge_index.shape[1], node_labels=node_labels, K=num_parts)
        partitioning_labels = torch.tensor(labels)
    # elif partition_type == 'Label':
    #     partitioning_labels = label_split(data, num_parts, task_type=task_type)
    else:
        print('E> Invalid partitioning algorithm specified. Options are {Metis, Louvain, Dirichlet, Label, Ours}')
        exit(-1)

    end_time = time.time()
    print(f"Time taken to partition graph using {partition_type}: {end_time - start_time}")

    server.record_node_assignment(partitioning_labels, tvt_type) # For if reuse partition
    server.record_prev_edges(data.edge_index, tvt_type) # For if reuse partition

    return partitioning_labels

def gen_train_clients(total_num_edges, max_num_clients, num_edge_per_clients=510):
    ''' Determine number of training clients in this snapshot based on the total number of edges in the global graph '''
    num_clients = total_num_edges // num_edge_per_clients
    
    return min(max(1,num_clients), max_num_clients)

def get_cut_edges(server, node_assignment, coo_format, tvt_type='train'):
    '''
    Takes as input:
    1) node_assignment where i th index refers to node i and node_assignment[i] is client it's assigned to
    2) coo_format = 2d list where first list are the source nodes and second list are the target nodes
    
    Output: Dictionary of lists, ith key is the start node and ith value is the list of cutting nodes connecting it
    '''
    coo_ccn = [[], []]
    ccn_label = []
    ccn_dict = defaultdict(list)

    if (server.global_changed_edges is not None) and (tvt_type == 'train'):
        g = server.global_changed_edges
        g_src = g[0].tolist()
        g_dst = g[1].tolist()
        changed_set = set(zip(g_src, g_dst))
    for start_node, end_node in zip(coo_format[0], coo_format[1]):
        if (node_assignment[start_node] != node_assignment[end_node]):
            if (server.global_changed_edges is not None and tvt_type == 'train' and (start_node, end_node) in changed_set) or (server.global_changed_edges is None):
                ccn_dict[start_node].append(end_node)
                coo_ccn[0].append(start_node)
                coo_ccn[1].append(end_node)
                ccn_label.append(1)

    return ccn_dict, torch.tensor(coo_ccn), torch.tensor(ccn_label)

def construct_single_client_data(server, data, subgraph_label, client_idx, clients, tvt_mode, task_type):
    node_mask = (subgraph_label == client_idx)
    subnodes = torch.arange(data.num_nodes)[node_mask]

    ei_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    subgraph_ei = data.edge_index[:, ei_mask]
    subgraph_edge_feat = data.edge_feature[ei_mask]
    edge_label_mask = torch.ones(subgraph_ei.size(1), dtype=torch.bool)
    indim = 16

    # If there are global changed edges recorded, filter out edges that are not changed
    # [Ablation Study] Comment out this part to run ablation study on node embedding exchange
    if server.global_changed_edges is not None and tvt_mode == "train":
        subgraph_edge_t = subgraph_ei.t()
        changed_edges_t = server.global_changed_edges.t()

        changed_mask = (subgraph_edge_t.unsqueeze(1) == changed_edges_t.unsqueeze(0)).all(dim=2).any(dim=1)

        subgraph_ei = subgraph_ei[:, changed_mask]
        subgraph_edge_feat = subgraph_edge_feat[changed_mask]
        if task_type == "LP":
            edge_label_mask = changed_mask # Also restrict edge_label_index and edge_label

    if task_type == "LP":
        # Generate Negative Edges
        negative_edges = generate_neg_edges(subgraph_ei, subnodes, subgraph_ei.size(1))
        edge_label_index = torch.cat([subgraph_ei, negative_edges], dim=1)
        edge_label = torch.concat([data.edge_label[ei_mask][edge_label_mask], torch.zeros(subgraph_ei.size(1))])
        fed_data = Data(node_feature=data.node_feature[node_mask], edge_label_index=edge_label_index, edge_label=edge_label, subnodes=subnodes, 
                        edge_feature=subgraph_edge_feat, edge_index=subgraph_ei, node_states=[torch.zeros((data.num_nodes, indim)) for _ in range(2)], 
                        location=clients[client_idx], keep_ratio=0.4)
        fed_data_loader = DataLoader(fed_data, batch_size=1)

    elif task_type == "NC":
        class_weights = compute_label_weights(data.node_label[node_mask])
        fed_data = Data(node_feature=data.node_feature[node_mask], node_label_index=subnodes, node_label=data.node_label[node_mask], subnodes=subnodes, 
                        edge_index=subgraph_ei, edge_feature=subgraph_edge_feat, class_weights=class_weights,
                        node_states=[torch.zeros((data.num_nodes, indim)) for _ in range(2)], location=clients[client_idx], keep_ratio=0.8)
        fed_data_loader = DataLoader(fed_data, batch_size=1)

    if tvt_mode == "train":
        clients[client_idx].changed_edge_index = subgraph_ei
        clients[client_idx].subnodes = subnodes # Record list of nodes this client will have (for NE exchange)
    
    return fed_data_loader

def reuse_partition(num_parts, server, data, tvt_type):
    if num_parts == 1:
        # Partition is the same, since there is only 1 client
        return server.node_assignment[tvt_type]
    if tvt_type in server.previous_edge_index:
        start_time = time.time()
        global_changed_edges, new_nodes = get_exclusive_subgraph(data.edge_index, server.previous_edge_index[tvt_type])
        server.record_global_changed_edges(global_changed_edges)
        print(f"Number of changed edges in global graph: {global_changed_edges.shape[1]}, number of new nodes: {len(new_nodes)}")
        prev_snapshot_nodes = torch.unique(server.previous_edge_index[tvt_type])
        prev_node_set = set(prev_snapshot_nodes.tolist())
        node2client = server.node_assignment[tvt_type] # Shortcut to previous assignment
        for node, onehop_neigh in new_nodes.items():
            # Find already allocated 1-hop neighbours
            alr_allo_1hop = [neigh for neigh in onehop_neigh if neigh in prev_node_set]
            if alr_allo_1hop:
                counts = Counter(node2client[neigh] for neigh in alr_allo_1hop) # majority vote for clients of valid neighbours
                assigned_client = counts.most_common(1)[0][0]
                node2client[node] = int(assigned_client)
                continue

            # Else, Find already allocated 2-hop neighbours
            twohop_neigh = get_2hop_neigh(data.edge_index, onehop_neigh)
            alr_allo_2hop = [neigh for neigh in twohop_neigh if neigh in prev_node_set]
            if alr_allo_2hop:
                counts = Counter(node2client[neigh] for neigh in alr_allo_2hop) # majority vote for clients of valid neighbours
                assigned_client = counts.most_common(1)[0][0]
                node2client[node] = int(assigned_client)
                continue

            # If no neighbors exist in previous graph, assign randomly
            assigned_client = np.random.randint(0, num_parts)
            node2client[node] = int(assigned_client)
            
        server.record_node_assignment(node2client, tvt_type)
        end_time = time.time()
        print(f"Time taken to partition graph by reusing: {end_time - start_time}")
        return server.node_assignment[tvt_type]

def get_2hop_neigh(edge_index, onehop_node):
    if not onehop_node:
        return []
    
    onehope_node_t = torch.tensor(onehop_node)
    mask = torch.isin(edge_index[0], onehope_node_t) | torch.isin(edge_index[1], onehope_node_t)
    two_hop_edges = edge_index[:, mask]

    if two_hop_edges.nelement() == 0:
        return []
    
    return torch.unique(torch.cat([two_hop_edges[0], two_hop_edges[1]])).tolist()