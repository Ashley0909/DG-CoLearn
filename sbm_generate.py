import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from plot_graphs import draw_graph


def generate_graph(task_cfg):
    # Generate a sequence of graphs with evolving probabilities
    snapshots = []
    num_snapshots = 5
    for t in range(num_snapshots):  # 5 time steps
        num_nodes = 200
        num_blocks = 10
        p_in = 0.6 - t * 0.05  # gradually decreasing intra-community connection
        p_out = 0.05 + t * 0.05  # increasing inter-community connection
        snapshot, labels, last_embeddings = generate_sbm_snapshot(task_cfg, num_nodes, num_blocks, p_in=p_in, p_out=p_out, seed=42 + t)
        snapshots.append(snapshot)

    train_list, val_list, test_list = [], [], []
    for t in range(num_snapshots-2):
        train_list.append(snapshots[t])
        val_list.append(snapshots[t+1])
        test_list.append(snapshots[t+2])

    return num_snapshots, train_list, val_list, test_list,  {'last_embeddings': last_embeddings, 'num_nodes': num_nodes, 'node_label': labels}


def generate_sbm_snapshot(task_cfg, num_nodes, num_blocks, p_in=0.6, p_out=0.1, seed=None, hidden_conv=128):
    sizes = [num_nodes // num_blocks] * num_blocks
    probs = [[p_in if i == j else p_out for j in range(num_blocks)] for i in range(num_blocks)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # Generate features and labels
    feat_dim = 16 # Set the input feature dimension as 16
    task_cfg.in_dim = feat_dim
    task_cfg.out_dim = num_blocks
    x = torch.randn((num_nodes, feat_dim))
    y = torch.tensor([G.nodes[i]['block'] for i in range(num_nodes)], dtype=torch.long) # Community labels

    empty_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)]), 
                        torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)]), 
                        torch.Tensor([[0 for _ in range(hidden_conv)] for _ in range(num_nodes)])]

    data = from_networkx(G)
    data.node_feature = x
    data.node_label = y
    data.edge_feature = torch.Tensor([[1 for _ in range(feat_dim)] for _ in range(data.edge_index.shape[1])])
    return data, y, empty_embeddings


