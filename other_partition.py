import numpy as np
import torch
from collections import defaultdict
import metis

from graph_partition import get_all_connected_components, connect_graphs

def label_split(graph_data, client_num, major_label=3, major_rate=0.8, sample_rate=1.0, task_type='NC'):
    """
    Adapt FedDGL's splitting method on Node Classification and Link Prediction (improved)
    Args:
        graph_data: A PyTorch Geometric Data object (or similar)
        client_num: Number of clients (subgraphs)
        major_label: Number of dominant labels each client holds
        major_rate: Proportion of client's nodes from major labels
        sample_rate: Portion of the entire graph to assign across all clients
    Returns:
        allocation: Tensor of shape [num_nodes], each value in [0, client_num-1]
    """
    x = graph_data.node_feature
    num_nodes = x.shape[0]
    
    if task_type == 'NC':
        y = graph_data.node_label.numpy()
    else:
        edge_index = graph_data.edge_index
        y = infer_edge_to_node(edge_index, num_nodes)
        major_label = 1

    num_classes = len(set(y))
    node_indices = np.arange(num_nodes)
    total_assignable = int(sample_rate * num_nodes)
    available_nodes = set(node_indices)

    # Allocation result (-1 means unassigned)
    allocation = -1 * np.ones(num_nodes, dtype=int)

    for cid in range(client_num):
        if len(available_nodes) == 0:
            break

        client_nodes = list(available_nodes)
        node_by_label = defaultdict(list)
        for node in client_nodes:
            node_by_label[y[node]].append(node)

        # Pick major labels for this client
        holding_labels = np.random.permutation(np.arange(num_classes))[:major_label]
        major_indices = []
        for label in holding_labels:
            major_indices += node_by_label[label]

        major_count = int(total_assignable * major_rate / client_num)
        major_selected = np.random.permutation(major_indices)[:min(major_count, len(major_indices))]

        # Fill the rest randomly
        remaining_count = int(total_assignable / client_num) - len(major_selected)
        remaining_pool = list(set(client_nodes) - set(major_selected))
        rest_selected = np.random.permutation(remaining_pool)[:remaining_count]

        assigned = list(major_selected) + list(rest_selected)

        # Assign nodes
        allocation[assigned] = cid
        available_nodes -= set(assigned)

    # If any nodes are still unassigned, assign them randomly
    unassigned = np.where(allocation == -1)[0]
    for node in unassigned:
        allocation[node] = np.random.randint(0, client_num)

    return torch.tensor(allocation)

def infer_edge_to_node(edge_index, num_nodes):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    node_labels = np.zeros(num_nodes, dtype=int)

    # Mark nodes involved in edges with label 1
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0][i], edge_index[1][i]
        node_labels[u] = 1
        node_labels[v] = 1

    return node_labels

def label_dirichlet_partition(
    labels: np.array, N: int, K: int, n_parties: int, beta: float
) -> list:
    """
    This function partitions data based on labels by using the Dirichlet distribution, to ensure even distribution of samples

    Arguments:
    labels: (NumPy array) - An array with labels or categories for each data point
    N: (int) - Total number of data points in the dataset
    K: (int) - Total number of unique labels
    n_parties: (int) - The number of groups into which the data should be partitioned
    beta: (float) - Dirichlet distribution parameter value

    Return:
    split_data_indexes (list) - list indices of data points assigned into groups

    """
    min_size = 0
    min_require_size = 10

    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            proportions = np.array(
                [
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])

    node_to_subgraph = [-1] * N
    
    # Assign subgraph index to each node
    for subgraph_id, nodes in enumerate(split_data_indexes):
        for node in nodes:
            node_to_subgraph[node] = subgraph_id
    return node_to_subgraph

def custom_metis(adj_list, num_parts):
    ''' METIS does not handle when num_parts=1. '''
    if num_parts > 1:
        # Identify all connected components
        connected_components, isolated_nodes = get_all_connected_components(adj_list)
        adj_list, synthetic_edges = connect_graphs(connected_components, adj_list)
        for iso_node in isolated_nodes:
            if iso_node != 0:
                adj_list[iso_node].append(0)
                adj_list[0].append(iso_node)
        _, partitioning_labels = metis.part_graph(adj_list, num_parts)
    
        return torch.tensor(partitioning_labels)
    else:
        return torch.zeros(len(adj_list), dtype=torch.long)

def prepare_metis_input_with_synthetic_edges(adj_list, K):
    # Ensure dict is full for METIS input
    num_nodes = len(adj_list)
    for i in range(num_nodes):
        adj_list.setdefault(i, set())
    
    # Step 1: Connect disconnected components
    components, isolated = get_all_connected_components(adj_list)
    
    # Step 2: Add synthetic edges
    synthetic_edges = []
    for i in range(len(components) - 1):
        u = next(iter(components[i]))
        v = next(iter(components[i + 1]))
        adj_list[u].add(v)
        adj_list[v].add(u)
        synthetic_edges.append((u, v))

    # Add isolated nodes to dummy connection (e.g., connect to node 0 if not already)
    for iso_node in isolated:
        if iso_node != 0:
            adj_list[iso_node].add(0)
            adj_list[0].add(iso_node)
            synthetic_edges.append((iso_node, 0))

    # Step 3: Convert to METIS format (list of neighbor lists)
    metis_input = [list(adj_list[i]) for i in range(num_nodes)]

    # Step 4: Partition with METIS
    edgecuts, part_labels = metis.part_graph(metis_input, nparts=K)

    return part_labels, synthetic_edges
    