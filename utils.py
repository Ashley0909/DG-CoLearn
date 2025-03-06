from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import numpy as np
import copy
import torch
from torch_geometric.data import Data
import sys
import datetime
import random
from collections import deque

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from functools import reduce

from plot_graphs import plot_h, plot_cluster

class Logger(object):
    def __init__(self, path):
        filename = "stats/{}/{}.txt".format(path, datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def process_data(txt_path):
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

def sample_dirichlet(split_points, y_samples, env_cfg, alpha, client_shard_sizes):
    classes = {} # list of index of the label {label: indicies}
    for idx, y in enumerate(y_samples):
        if y.item() in classes:
            classes[y.item()].append(idx)
        else:
            classes[y.item()] = [idx]

    for n in range(len(classes.keys())): # For each class
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_size = class_size * np.random.dirichlet(np.array(env_cfg.n_clients * [alpha])) # determines the size of sampled of this class to each clients
        sampled_size = sampled_size.astype(int)
        start_point = 0

        for c in range(env_cfg.n_clients): # For each client
            elem = range(start_point, start_point + sampled_size[c])
            start_point = start_point + sampled_size[c]
            split_points[c].extend(torch.tensor(classes[n])[elem])
            if len(client_shard_sizes) < env_cfg.n_clients: # Append a new list to begin a new client
                client_shard_sizes.append(sampled_size[c])
            else:
                client_shard_sizes[c] += sampled_size[c]

    return split_points, client_shard_sizes

def normalize(data, expt=None):
    """
    Normalize data
    :param data: data to normalize (in np.array)
    :param expt: a list of col(s) to keep original value
    :return: normalized data
    """
    if not expt:
        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    else:
        tmp = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))  # norm all cols
        tmp[:, expt] = data[:, expt]  # roll these cols back
        return tmp
    
def clustering(data, flag, e, server_round, y_label, pca_dim=2):
    """ Cluster the clients by their data (data is a tensor) """
    pca = PCA(n_components=pca_dim)
    reduced_data = pca.fit_transform(np.array(data))

    plot_h(matrix=reduced_data, path=str(pca_dim)+'dreduced', name='Reduced FCW')

    if flag == 0:
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(reduced_data)
        comb_C = kmeans.predict(reduced_data)

        centroids = kmeans.cluster_centers_
        centroids_assignment = kmeans.predict(centroids)
        unique_labels = np.unique(comb_C)
        # Compute intra-cluster distances
        intra_cluster = euclidean_distances(centroids[centroids_assignment == unique_labels[0]], centroids[centroids_assignment == unique_labels[1]])
        if intra_cluster[0][0] < 0.05:
            flag = 1

        # Get the maximum and average distance of a data point to its cluster
        max_dist = []
        avg_dist = []
        for l in unique_labels:
            distances = euclidean_distances(reduced_data[comb_C == l], centroids[centroids_assignment == l])
            avg_dist.append(np.average(distances))
            max_dist.append(np.max(distances))
        e = np.max(max_dist)

        # Identify Obvious clients (where they are close to their cluster centroid)
        distances = pairwise_distances_argmin_min(reduced_data, kmeans.cluster_centers_)
        obvious_clients = np.array([]).astype(int)
        for l in unique_labels:
            obvious = np.where((distances[0] == l) & (distances[1] <= (avg_dist[l] * 1.5)))[0]
            obvious_clients = np.concatenate((obvious_clients, obvious))

    else:
        mp = 2
        e -= 0.0025*(server_round/5)
        mp -= (server_round//20)
        db = DBSCAN(eps=max(e, 0.03), min_samples=max(3,mp)).fit(reduced_data)
        comb_C = db.labels_
        unique_labels = np.unique(comb_C)
        if len(unique_labels) < 3:
            pseudo_centroids = {}  # To store the pseudo-centroid for each cluster
            avg_distances = {}  # To store the average distance for each cluster

            # 1. Compute pseudo-centroids and average distances for each cluster
            for label in unique_labels:
                cluster_points = reduced_data[comb_C == label]
                
                # Compute pseudo-centroid (mean of all points in the cluster)
                pseudo_centroid = cluster_points.mean(axis=0)
                pseudo_centroids[label] = pseudo_centroid
                # Compute distances of all points in the cluster to the pseudo-centroid
                distances = pairwise_distances(cluster_points, [pseudo_centroid])
                # Compute average distance for this cluster
                avg_distances[label] = np.mean(distances)

            # 2. Identify Obvious Clients
            obvious_clients = []

            for i, label in enumerate(comb_C):
                point = reduced_data[i]
                pseudo_centroid = pseudo_centroids[label]
                
                # Compute distance to pseudo-centroid
                distance = np.linalg.norm(point - pseudo_centroid)
                if distance <= (avg_distances[label] * 1.5):
                    obvious_clients.append(i)
            obvious_clients = np.array(obvious_clients)
    
    plot_cluster(data=reduced_data, label=comb_C, dim=pca_dim, y_label=y_label)

    counts = Counter(comb_C)
    largest_class = max(counts, key=counts.get)

    if len(unique_labels) < 3:
        comb_C = np.where(comb_C == largest_class, 0, 1)
        #Need condition for DBSCAN as well
    else:
        two_largest_classes = counts.most_common(2) # [(top1class, #datapts), (top2class, #datapts)]
        second_largest_class = two_largest_classes[1][0]
        print("largest class", largest_class, "second largest class", second_largest_class)
        obvious_clients = np.where((comb_C == largest_class) | (comb_C == second_largest_class))[0]
        print("obvious clients", obvious_clients)

        comb_C = np.where(comb_C == largest_class, 0, np.where(comb_C == second_largest_class, 1, -1))
        print("comb_C for 3 or more clusters", comb_C)

    return comb_C, e, flag, obvious_clients

def merge_clients(data, comb_C, target_label, avg_1, avg_0):
    for a in np.unique(comb_C):
        layer = data[comb_C == a][:, target_label]
        target_layer = torch.tensor(layer).mean(dim=0)
        if abs(target_layer - avg_1[target_label]) > abs(target_layer - avg_0[target_label]):
            comb_C[comb_C == a] = 0
        else:
            comb_C[comb_C == a] = 1

    return comb_C

def check_ambiguous(ambiguous_ids, fcw, target_label, min_avg, maj_avg, assignment):
    for amb in ambiguous_ids:
        if abs(fcw[amb][target_label] - min_avg[target_label]) > abs(fcw[amb][target_label] - maj_avg[target_label]):
            assignment[amb] = 0
            print(amb, "being allocated to class", 0)
        else:
            assignment[amb] = 1
            print(amb, "being allocated to class", 1)

    return assignment

def weighted_aggregate(models, benign_clients, malicious_clients, evil_clients, client_shard_sizes, data_size, target_label, last_layer, sim_weight=13):
    client_weights_vec = np.array(client_shard_sizes) / data_size

    # Shape the global model using the first participating client
    print("benign_clients[0]", benign_clients[0])
    global_model = copy.deepcopy(models[benign_clients[0]])
    global_model_params = global_model.state_dict()
    for pname, param in global_model_params.items():
        global_model_params[pname] = 0.0
    
    # First aggregate the benign clients with normal FedAvg technique
    for id in benign_clients:
        for name, param in models[id].state_dict().items():
            global_model_params[name] += param.data * client_weights_vec[id]

    # Then Aggregate the malicious clients to another model with FedAvg
    if len(malicious_clients) != 0 or len(evil_clients) != 0:
        print("Malicious Clients exist")
        malicious_model = copy.deepcopy(models[malicious_clients[0]]) if len(malicious_clients) != 0 else copy.deepcopy(models[evil_clients[0]])
        malicious_model_params = malicious_model.state_dict()
        for pname, param in malicious_model_params.items():
            malicious_model_params[pname] = 0.0
            
        bad_clients = np.concatenate((malicious_clients, evil_clients))
        for id in bad_clients:
            for name, param in models[id].state_dict().items():
                malicious_model_params[name] += param.data * client_weights_vec[id]
        
        # Aggregate the malicious parameters according to the difference to benign clients (only aggregated fully connected layers)
        layer_name = last_layer.split('.')[0]
        for name, benign_param in global_model_params.items():
            if layer_name in name: # Identify fully connected layers
                dist = (benign_param - malicious_model_params[name]).abs()

                if len(benign_param.shape) == 2: # weights
                    neuron_dist = dist.sum(dim=1, keepdim=True)
                    weight = torch.exp(-neuron_dist * sim_weight)
                    weight = weight.expand_as(benign_param)
                elif len(benign_param.shape) == 1: # bias
                    weight = torch.exp(-dist * sim_weight)
                weight[target_label] = 0
                global_model_params[name] += (malicious_model_params[name] * weight)
                global_model_params[name] /= (1 + weight)

    # Load state dict
    global_model.load_state_dict(global_model_params)

    return global_model

def get_exclusive_edges(current, prev):
    # Get edges in current but not in prev
    current_T = current.cpu().numpy().T
    prev_T = prev.cpu().numpy().T

    mask = np.all(current_T[:, None] == prev_T[None, :], axis=-1)
    exclusive_mask = ~np.any(mask, axis=1)

    exclusive_current = current[:, exclusive_mask]

    # Now get all the edges inside the 2-hop neighbourhoods of newly added edges
    new_nodes = torch.unique(exclusive_current)

    mask_1hop = torch.isin(current[0], new_nodes) | torch.isin(current[1], new_nodes)
    one_hop_edges = current[:, mask_1hop]

    one_hop_nodes = torch.unique(one_hop_edges)
    mask_2hop = torch.isin(current[0], one_hop_nodes) | torch.isin(current[1], one_hop_nodes)
    two_hop_edges = current[:, mask_2hop]

    if two_hop_edges is None:
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
        elif hop > 1:
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
            node_embdedding_sum = node_embedding_update_sum(node, ccn, hop)
            final_embedding = torch.zeros(embeddings[first_parti_client][0][0].shape).to("cuda:0")
            for update_node, k in node_embdedding_sum:
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
    macro_f1 = f1_score(true, pred, average='macro')
    # macro_auc = roc_auc_score(true, pred_score, average='macro')
    # micro_auc = roc_auc_score(true, pred_score, average='micro')

    return acc, ap, macro_f1

def nc_prediction(pred_score, true_l):
    pred = pred_score.argmax(dim=1).detach().cpu().numpy()
    true = true_l.cpu().numpy()

    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average='macro')
    micro_f1 = f1_score(true, pred, average='micro')

    return acc, macro_f1, micro_f1

def compute_mrr(pred_score, true_l):
    sorted_indices = torch.argsort(pred_score, descending=True)
    sorted_labels = true_l[sorted_indices]

    true_edge_ranks = torch.where(sorted_labels == 1)[0]

    reciprocal_ranks = 1.0 / (true_edge_ranks + 1)

    mrr = reciprocal_ranks.mean().item()

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
