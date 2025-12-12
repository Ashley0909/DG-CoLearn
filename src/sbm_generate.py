import networkx as nx
import copy
import random
import torch
from torch_geometric.utils import from_networkx

def generate_graph(task_cfg):
    ''' Generate a sequence of evolving graphs by SBM. Rather than gradually adjusting connection, first construct snapshot 0, then manually change edges.'''
    snapshots = []
    num_snapshots = 5
    for t in range(num_snapshots):
        num_nodes = 200
        num_blocks = 10
        feat_dim = 16
        task_cfg.in_dim = feat_dim
        task_cfg.out_dim = num_blocks

        snapshot_nx = generate_evolving_snapshot(num_snapshots, num_nodes, num_blocks, p_in=0.6, p_out=0.01)
        snapshots, last_embeddings = convert_nx_to_data(snapshot_nx, feat_dim)

    train_list, val_list, test_list = [], [], []
    for t in range(num_snapshots-2):
        train_list.append(snapshots[t])
        val_list.append(snapshots[t+1])
        test_list.append(snapshots[t+2])

    return num_snapshots, train_list, val_list, test_list,  {'last_embeddings': last_embeddings, 'num_nodes': num_nodes}

def generate_evolving_snapshot(num_snapshots, num_nodes, num_blocks, p_in, p_out, edge_change_per_step=20):
    #Initial Graph
    sizes = [num_nodes // num_blocks] * num_blocks
    probs = [[p_in if i == j else p_out for j in range(num_blocks)] for i in range(num_blocks)]
    G_prev = nx.stochastic_block_model(sizes, probs, seed=42)
    snapshots = [G_prev]

    for _ in range(1, num_snapshots):
        G_new = evolve_sbm_graph(snapshots[-1], edge_change_per_step)
        snapshots.append(G_new)

    return snapshots

def evolve_sbm_graph(prev_graph, edge_change_per_step):
    G_new = copy.deepcopy(prev_graph)
    nodes = list(G_new.nodes())

    # Edge Removal
    edges = list(G_new.edges())
    for _ in range(edge_change_per_step // 2):
        if edges:
            edge_to_remove = random.choice(edges)
            G_new.remove_edge(*edge_to_remove)
            edges.remove(edge_to_remove)

    # Edge Additional
    for _ in range(edge_change_per_step // 2):
        u, v = random.sample(nodes, 2)
        if not G_new.has_edge(u,v):
            G_new.add_edge(u,v)

    return G_new

def convert_nx_to_data(snapshots, feat_dim, hidden_conv=128):
    data_list = []
    num_nodes = snapshots[0].number_of_nodes()
    node_features = torch.randn((num_nodes, feat_dim))

    for G in snapshots:
        data = from_networkx(G)
        
        data.node_feature = node_features.clone()
        data.node_label = torch.tensor([G.nodes[n]['block'] for n in G.nodes()], dtype=torch.long) # Community labels
        data.edge_feature = torch.Tensor([[1 for _ in range(feat_dim)] for _ in range(data.edge_index.shape[1])])
        
        data_list.append(data)

    empty_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)]), 
                    torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)]), 
                    torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)])]

    return data_list, empty_embeddings