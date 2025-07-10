import requests
import gzip
import shutil
import os

import numpy as np
import torch
import sys
import datetime
import random
from torch_geometric.data import Data
import torch_geometric
from collections import deque

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import coo_matrix


class Logger(object):
    def __init__(self, path):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M")

        log_dir = os.path.join("stats", path, date_str)
        os.makedirs(log_dir, exist_ok=True)

        filename = os.path.join(log_dir, f"{time_str}.txt")
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def isatty(self):
        return False

def download_url(url, save_path):
    response = requests.get(url, stream=True)
    file_path = os.path.join(save_path, url.split("/")[-1])
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path

def extract_gz(gz_path):
    extracted_path = gz_path.replace(".gz", "")
    with gzip.open(gz_path, "rb") as f_in:
        with open(extracted_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return extracted_path

def process_txt_data(txt_path):
    with open(txt_path, 'r') as f:
        lines = [[x for x in line.split(' ')]
                    for line in f.read().split('\n')[:-1]]
        
        edge_indices = [[int(line[0]), int(line[1])] for line in lines]
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_index = edge_index - edge_index.min()
        edge_index = edge_index.t().contiguous()
        num_nodes = int(edge_index.max()) + 1

        stamps = [
            datetime.datetime.fromtimestamp(int(float(line[2])))
            for line in lines
        ]

    # total_duration = (stamps[-1] - stamps[0]).days

    offset = datetime.timedelta(days=5.0) # 193 days, Results in 39 time steps
    graph_indices, factor = [], 1
    for t in stamps:
        factor = factor if t < stamps[0] + factor * offset else factor + 1
        graph_indices.append(factor - 1)
    graph_idx = torch.tensor(graph_indices, dtype=torch.long)

    data_list = []
    for i in range(int(graph_idx.max()) + 1):
        mask = (graph_idx > (i - 10)) & (graph_idx <= i)
        data = Data()
        data.edge_index = edge_index[:, mask]
        data.num_nodes = num_nodes
        data_list.append(data)

    return data_list

def get_exclusive_subgraph(current, prev):
    ''' Get the 2-hop neighbourhood of edges that are exclusive to the current subgraph (added and removed) '''
    # Convert to NumPy for efficient comparison
    current_T = current.cpu().numpy().T
    prev_T = prev.cpu().numpy().T

    # Find exclusive edges in current (not in prev)
    add_mask = np.all(current_T[:, None] == prev_T[None, :], axis=-1)
    exclusive_add_mask = ~np.any(add_mask, axis=1)
    exclusive_current = current[:, exclusive_add_mask]

    # Find exclusive edges in prev (not in current)
    remove_mask = np.all(prev_T[:, None] == current_T[None, :], axis=-1)
    exclusive_remove_mask = ~np.any(remove_mask, axis=1)
    exclusive_prev = prev[:, exclusive_remove_mask]

    # Combine both sets of exclusive edges
    exclusive_edges = torch.cat([exclusive_current, exclusive_prev], dim=1)

    # Get 1-hop neighbors
    new_nodes = torch.unique(exclusive_edges)
    mask_1hop = torch.isin(current[0], new_nodes) | torch.isin(current[1], new_nodes)
    one_hop_edges = current[:, mask_1hop]

    # Get 2-hop neighbors
    one_hop_nodes = torch.unique(one_hop_edges)
    mask_2hop = torch.isin(current[0], one_hop_nodes) | torch.isin(current[1], one_hop_nodes)
    two_hop_edges = current[:, mask_2hop]

    if two_hop_edges.nelement() == 0:
        print('>E No edges left to train')

    return two_hop_edges

def node_embedding_update_sum(start_node, ccn, k):
    '''
    Function to return the contribution of each neighbouring node to start node and its hop embedding
    Inputs:
    1) start_node -> node we wish to find contribution for next node embedding
    2) ccn -> defaultdict(list) of cross client nodes
    3) k -> Hop we wish to find embedding of start_node for

    Output:
    list of tuples corresponding to (node required, hop) for vector embedding update
    '''
    embeddings_required = []
    dq = deque([(start_node, k, {start_node})])
    while dq:
        node, hop, nodes_visited = dq.popleft()
        embeddings_required.append([node, hop])
        if hop > 1 and node == start_node:
            embeddings_required += [[node, 0]]  * len(ccn[node]) # count the times 1-hop ccn visits itself
        elif hop == 1:
            embeddings_required += [[node, 0]] # add 0-hop whenever it is visited

        for neigh in ccn[node]:
            if neigh not in nodes_visited and hop>0:
                dq.append((neigh, hop-1, nodes_visited|{neigh}))

    return embeddings_required

def get_global_embedding(embeddings, ccn, node_client_map, subnodes_union, first_parti_client):
    '''
    Function to return the global embedding to update the client's local embeddings, using the formula:
    1 hop NE of node i => NE1[i] + SUM(NE0[j]) for j in ccn[i]
    2 hop NE of node i => NE2[i] + SUM(NE1[j] + NE0[j] + NE0[i]) for j in ccn[i] + SUM(NE0[k]) for k in ccn[j]

    Inputs:
    1) embeddings -> defaultdict(Tensor) of 0-hop, 1-hop and 2-hop NE of each client
    2) ccn -> defaultdict(list) of cross client nodes
    3) node_client_map -> the client each node is assigned for training

    Output:
    list of 0-hop, 1-hop and 2-hop Global NE 
    '''
    if len(embeddings) == 1:
        return embeddings[0] # Only one client
    
    hop_embeddings = []
    for hop in range(3):
        hop_matrix = []
        for node in range(len(node_client_map)):
            node_embedding_sum = node_embedding_update_sum(node, ccn, hop)
            final_embedding = torch.zeros(embeddings[first_parti_client][0][0].shape).to("cuda:0")
            for update_node, k in node_embedding_sum:
                if update_node in subnodes_union:
                    final_embedding += embeddings[node_client_map[update_node]][k][update_node]
            hop_matrix.append(final_embedding)
        stack = torch.stack(hop_matrix)
        hop_embeddings.append(stack)

    return hop_embeddings

def lp_prediction(pred_score, true_l):
    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()

    true = true_l.cpu().numpy()
    acc = accuracy_score(true, pred)
    ap = average_precision_score(true, pred_score)

    return acc, ap

def nc_prediction(pred_score, true_l):
    pred = pred_score.argmax(dim=1).detach().cpu().numpy()
    true = true_l.cpu().numpy()

    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average='macro')
    micro_f1 = f1_score(true, pred, average='micro')

    return acc, macro_f1, micro_f1

def compute_mrr(pred_score, true_l, edge_label_index, do_softmax=True):
    ''' Using the same way how EvolveGCN evaluates mrr '''
    if do_softmax:
        probs = torch.softmax(pred_score, dim=0)
    else:
        probs = pred_score

    probs = probs.cpu().detach().numpy()
    true_l = true_l.cpu().detach().numpy()

    source_nodes = edge_label_index[0].cpu().detach().numpy()
    target_nodes = edge_label_index[1].cpu().detach().numpy()

    pred_matrix = coo_matrix((probs, (source_nodes, target_nodes))).toarray()
    true_matrix = coo_matrix((true_l, (source_nodes, target_nodes))).toarray()

    # Calculate mrr for each row where there are true edges
    row_mrrs = []
    for i, pred_row in enumerate(pred_matrix):
        # Check if there are any existing edges in the true_matrix for this row
        if np.isin(1, true_matrix[i]):  # 1 indicates an existing edge
            row_mrrs.append(get_row_mrr(pred_row, true_matrix[i]))

    avg_mrr = torch.tensor(row_mrrs).mean()  # Return the average mrr across all rows
    return avg_mrr.float().item()

def get_row_mrr(prob_score, true_l):
    prob_score = np.nan_to_num(prob_score)
    # Get the mask for the existing edges (true labels == 1)
    existing_mask = true_l == 1
    # Sort predictions in descending order (probabilities)
    ordered_indices = np.flip(prob_score.argsort())

    # Apply the ordered indices to the existing mask to find the rank of true edges
    ordered_existing_mask = existing_mask[ordered_indices]
    existing_ranks = np.arange(1, true_l.shape[0] + 1, dtype=np.cfloat)[ordered_existing_mask]

    if existing_ranks.shape[0] == 0: # No valid ranks, return 0 instead of NaN
        return 0.0

    # Calculate Mean Reciprocal Rank (mrr) for the row
    mrr = (1 / existing_ranks).sum() / existing_ranks.shape[0]
    return mrr

def generate_neg_edges(edge_index, node_range:torch.Tensor, num_neg_samples:int=None):
    ''' Generate `num_neg_samples` negative edges from pairs in `node_range` where they do not exist in `edge_index`'''
    existing_edges = set(map(tuple, edge_index.t().tolist()))  # Convert edge_index to set for fast lookup
    allowed_nodes = node_range.tolist()  # Ensure it's a list for indexing

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)  # Default: same number as positive edges

    neg_edges = set()

    while len(neg_edges) < num_neg_samples:
        src, dst = random.choice(allowed_nodes), random.choice(allowed_nodes)
        if src != dst and (src, dst) not in existing_edges:
            neg_edges.add((src, dst))

    neg_edge_index = torch.tensor(list(neg_edges)).t()  # Convert back to tensor shape (2, num_neg_samples)
    
    return neg_edge_index

def count_label_occur(node_assignment, node_labels):
    if node_labels == None:
        return
    
    pairs = torch.stack([node_assignment, node_labels], dim=1)

    # Get unique (subgraph, label) pairs and their counts
    unique_pairs, counts = torch.unique(pairs, return_counts=True, dim=0)

    # Convert results into a dictionary-like structure
    subgraph_label_counts = {}
    for (subgraph, label), count in zip(unique_pairs.tolist(), counts.tolist()):
        if subgraph not in subgraph_label_counts:
            subgraph_label_counts[subgraph] = {}
        subgraph_label_counts[subgraph][label] = count

    # Print results
    for subgraph, label_counts in subgraph_label_counts.items():
        print(f"Subgraph {subgraph}: {label_counts}")

def compute_label_weights(node_label):
    num_classes = torch.max(node_label)
    class_counts = torch.bincount(node_label, minlength=num_classes).float()

    class_weights = 1.0 / (class_counts + 1e-6) # Avoid division by zero
    class_weights /= class_weights.sum() # Normalise to sum to 1

    return class_weights

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

def find_common_nodes(all_nodes, exclusive_edge_index):
    ''' Get the nodes that are unchanged throughout the snapshots. '''
    changed_nodes = torch.unique(exclusive_edge_index).to('cpu')
    mask = ~torch.isin(all_nodes, changed_nodes)
    common_nodes = all_nodes[mask]

    return common_nodes

