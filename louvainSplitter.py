from torch_geometric.utils import to_networkx
import community as community_louvain  # python-louvain
import networkx as nx
import torch
from torch_geometric.data import Data
from collections import defaultdict

class LouvainSplitter:
    def __init__(self, client_num, delta=20):
        self.client_num = client_num
        self.delta = delta

    def __call__(self, data: Data):
        data.index_orig = torch.arange(data.num_nodes)

        G = to_networkx(data, node_attrs=['node_feature'], to_undirected=True)
        nx.set_node_attributes(G, {i: i for i in range(data.num_nodes)}, name="index_orig")

        partition = community_louvain.best_partition(G) # Louvain does not fix number of communities, it only relies on modularity

        # Step 2: Organize nodes by cluster
        cluster_to_nodes = defaultdict(list)
        for node, cluster_id in partition.items():
            cluster_to_nodes[cluster_id].append(node)

        # Step 3: Reassign clusters into exactly num_clients buckets
        cluster_ids = list(cluster_to_nodes.keys())
        cluster_ids.sort(key=lambda c: len(cluster_to_nodes[c]), reverse=True)

        client_assignments = [[] for _ in range(self.client_num)]

        # Round-robin assign clusters to clients
        for i, cluster_id in enumerate(cluster_ids):
            client_assignments[i % self.client_num].extend(cluster_to_nodes[cluster_id])

        # Step 4: Create new label vector
        partitioning_labels = torch.empty(data.num_nodes, dtype=torch.long)
        for client_idx, nodes in enumerate(client_assignments):
            for node in nodes:
                partitioning_labels[node] = client_idx

        return partitioning_labels
