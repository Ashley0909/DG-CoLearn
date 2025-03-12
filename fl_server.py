import numpy as np
import scipy.sparse as sp
import torch

class Server:
    ''' A server class to record global_adj_list, number of subgraphs, node_assignment and ccn.'''
    def __init__(self):
        self.global_adj_mtx = None
        self.edge_index = None
        self.num_nodes = 0
        self.num_subgraphs = 0
        self.clients_adj_matrices = None
        self.ccn = None
        self.node_assignment = None

    def construct_global_adj_matrix(self, edge_index, num_nodes):
        ''' Save current snapshot's edge index and use it to construct adj_matrix for node embedding exchange. '''
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        
        row, col = edge_index
        adj_matrix = sp.coo_matrix((torch.ones_like(row), (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
        dense_adj_matrix = adj_matrix.toarray()  # Converts to dense numpy array
        self.global_adj_mtx = dense_adj_matrix

    def record_num_subgraphs(self, num_subgraphs):
        ''' Save number of subgraphs. '''
        self.num_subgraphs = num_subgraphs

    def construct_client_adj_matrix(self, node_assignment):
        ''' Save node assignment. '''
        self.node_assignment = node_assignment

        # Initialize empty adjacency matrices for each subgraph
        clients_adj_matrices = [np.zeros((self.num_nodes, self.num_nodes), dtype=int) for _ in range(self.num_subgraphs)]
        
        # Iterate through edges
        for i in range(self.edge_index.shape[1]):  # edge_index is (2, num_edges)
            src, dst = self.edge_index[:, i]
            src_subgraph = node_assignment[src].item()
            dst_subgraph = node_assignment[dst].item()
            
            # Only add edge if both nodes belong to the same subgraph
            if src_subgraph == dst_subgraph:
                clients_adj_matrices[src_subgraph][src, dst] = 1
                clients_adj_matrices[src_subgraph][dst, src] = 1  # Assuming undirected graph

        self.clients_adj_matrices =  np.array(clients_adj_matrices)

    def record_ccn(self, ccn):
        ''' Save cross client neighbours. '''
        self.ccn = ccn

    def get_node_embedding_needed(self, start_node, node_assignment, k):
        ''' Return all the (client, node, number of times needed to add) for each hop. '''
        if k == 1:
            ne_needed = [[] for _ in range(k)] # info needed for hop 0
            to_subtract = self.clients_adj_matrices[node_assignment[start_node]]
            adjustment_coefficient = self.global_adj_mtx[start_node] - to_subtract[start_node]
            for i, coe in enumerate(adjustment_coefficient):
                if coe > 0:
                    ne_needed[0].append((node_assignment[i], i, coe))

        elif k == 2:
            ne_needed = [[] for _ in range(k)] # info needed for hop 0, 1
            global_two_hop = np.linalg.matrix_power(self.global_adj_mtx, 2) # Corrected (** 2 is wrong)
            to_subtract = np.linalg.matrix_power(self.clients_adj_matrices[node_assignment[start_node]], 2)
            for neigh in self.ccn[start_node]:
                to_subtract[start_node] += self.clients_adj_matrices[node_assignment[neigh]][neigh] # Correct (have to specify which row)
                ne_needed[1].append((node_assignment[neigh], neigh, 1))

            adjustment_coefficient = global_two_hop[start_node] - to_subtract[start_node]
            for i, coe in enumerate(adjustment_coefficient):
                if coe > 0:
                    ne_needed[0].append((node_assignment[i], i, coe))
            
        return ne_needed # Corrected

    def fast_get_global_embedding(self, embeddings, subnodes_union):
        node_assign = self.node_assignment.tolist()
        hop_embeddings = []
        for hop in range(3):
            hop_matrix = []
            for node in range(len(node_assign)):
                if self.ccn[node] == [] or hop == 0:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                else:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                    # print("hop", hop, "node", node, "starting emb", final_embedding)
                    ne_needed = self.get_node_embedding_needed(node, node_assign, hop)
                    for hop_needed, tuples in enumerate(ne_needed):
                        for client, node, num_times in tuples:
                            if node in subnodes_union:
                                # print(f"hop {hop_needed}: ({client},{node},{num_times}) => {embeddings[client][hop_needed][node]}")
                                final_embedding += embeddings[client][hop_needed][node] * num_times

                hop_matrix.append(final_embedding)
            hop_embeddings.append(hop_matrix)

        return hop_embeddings
