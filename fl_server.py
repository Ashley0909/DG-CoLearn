import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from fl_models import MLPEncoder

class Server:
    ''' A server class to record global_adj_list, number of subgraphs, node_assignment and ccn.'''
    def __init__(self):
        self.global_adj_mtx = None
        self.adj_list = None
        self.num_nodes = 0
        self.num_subgraphs = 0
        self.clients_adj_matrices = None
        self.ccn = None
        self.node_assignment = None
        self.test_loader = None
        self.client_features = []

        self.global_adj_mtx_gpu = None
        self.clients_adj_matrices_gpu = None

    def record_num_subgraphs(self, num_subgraphs):
        ''' Save number of subgraphs. '''
        self.num_subgraphs = num_subgraphs

    def record_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes

    def construct_glob_adj_mtx(self, adj_list):
        '''Create global adjacency matrix as a sparse matrix.'''
        self.adj_list = adj_list

        global_adj_matrix = sp.lil_matrix((self.num_nodes, self.num_nodes), dtype=int)
        for src in range(self.num_nodes):
            for dst in adj_list[src]:
                global_adj_matrix[src, dst] = 1
        global_adj_matrix = global_adj_matrix.tocsr()  # Convert to CSR format for efficient operations
    
        self.global_adj_mtx = global_adj_matrix
        self.global_adj_mtx_gpu = self._sparse_to_torch_gpu(global_adj_matrix)

    def construct_client_adj_matrix(self, node_assignment):
        ''' Save node assignment. '''
        self.node_assignment = node_assignment

        # Create client adjacency matrices as sparse matrices
        clients_adj_matrix = []
        for client in range(self.num_subgraphs):
            client_adj_matrix = sp.lil_matrix((self.num_nodes, self.num_nodes), dtype=int)
            for src in range(self.num_nodes):
                if node_assignment[src] == client:
                    for dst in self.adj_list[src]:
                        if node_assignment[dst] == client:
                            client_adj_matrix[src, dst] = 1
            clients_adj_matrix.append(client_adj_matrix.tocsr())  # Convert to CSR format

        self.clients_adj_matrices = clients_adj_matrix
        self.clients_adj_matrices_gpu = [self._sparse_to_torch_gpu(mtx) for mtx in clients_adj_matrix]

    def _sparse_to_torch_gpu(self, sparse_matrix):
        '''Convert scipy.sparse.csr_matrix to torch.sparse tensor on GPU.'''
        sparse_matrix = sparse_matrix.tocoo()
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
        indices = torch.tensor(indices, dtype=torch.long, device='cuda')
        values = torch.tensor(sparse_matrix.data, dtype=torch.float32, device='cuda')
        shape = sparse_matrix.shape
        return torch.sparse_coo_tensor(indices, values, shape)

    def record_ccn(self, ccn):
        ''' Save cross client neighbours. '''
        self.ccn = ccn

    def construct_ccn_test_data(self, indim, edge_index, edge_label, subnodes):
        node_feature = torch.Tensor([[1 for _ in range(indim)] for _ in range(self.num_nodes)])
        edge_feature = torch.Tensor([[1 for _ in range(128)] for _ in range(edge_index.shape[1])])

        server_data = Data(node_feature=node_feature, edge_label_index=edge_index, edge_label=edge_label,
                        edge_feature=edge_feature, edge_index=edge_index, subnodes=subnodes,
                        node_states=[torch.zeros((self.num_nodes, indim//2)), torch.zeros((self.num_nodes, indim//2))], keep_ratio=0.0) # after dimension reduction to 16
        self.test_loader = DataLoader(server_data, batch_size=1)

    def get_node_embedding_needed(self, start_node, k):
        ''' Return all the (client, node, number of times needed to add) for each hop. '''
        ne_needed = [[] for _ in range(k)]
        node_assign = self.node_assignment.tolist()
        
        if k == 1:
            # Convert to dense for the specific row (since sparse slicing is inefficient for single rows)
            to_subtract = self.clients_adj_matrices[node_assign[start_node]][start_node].toarray()[0]
            adjustment_coefficient = self.global_adj_mtx[start_node].toarray()[0] - to_subtract
            indices = np.where(adjustment_coefficient > 0)[0]
            ne_needed[0] = [(node_assign[i], i, adjustment_coefficient[i]) for i in indices]
        
        elif k == 2:
            # Compute global_two_hop using sparse matrix multiplication
            global_two_hop = self.global_adj_mtx @ self.global_adj_mtx
            to_subtract = self.clients_adj_matrices[node_assign[start_node]] @ self.clients_adj_matrices[node_assign[start_node]]
            
            # Add contributions from neighbors
            start_subtract = to_subtract[start_node].toarray()[0]
            for neigh in self.ccn[start_node]:
                start_subtract += self.clients_adj_matrices[node_assign[neigh]][neigh].toarray()[0]
                ne_needed[1].append((node_assign[neigh], neigh, 1))
            
            # Compute adjustment coefficient
            adjustment_coefficient = global_two_hop[start_node].toarray()[0] - start_subtract
            indices = np.where(adjustment_coefficient > 0)[0]
            ne_needed[0] = [(node_assign[i], i, adjustment_coefficient[i]) for i in indices]
        
        return ne_needed

    def fast_get_global_embedding(self, embeddings, subnodes_union):
        num_nodes = len(self.adj_list)
        node_assign = self.node_assignment.tolist()
        
        hop_embeddings = []
        for hop in range(3):
            hop_matrix = []
            for node in range(num_nodes):
                if self.ccn[node] == [] or hop == 0:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                else:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                    ne_needed = self.get_node_embedding_needed(node, hop)
                    for hop_needed, tuples in enumerate(ne_needed):
                        for client, node_needed, num_times in tuples:
                            if node_needed in subnodes_union:
                                final_embedding += embeddings[client][hop_needed][node_needed] * num_times
                hop_matrix.append(final_embedding)
            stack = torch.stack(hop_matrix)
            hop_embeddings.append(stack)
        
        return hop_embeddings
    
    def get_node_embedding_needed_gpu(self, start_node, k):
        ''' Return all the (client, node, number of times needed to add) for each hop. '''
        ne_needed = [[] for _ in range(k)]
        node_assign = self.node_assignment.tolist()
        
        if k == 1:
            # Convert to dense for the specific row (since sparse slicing is inefficient for single rows)
            to_subtract = self.clients_adj_matrices_gpu[node_assign[start_node]][start_node].to_dense()
            adjustment_coefficient = self.global_adj_mtx_gpu[start_node].to_dense() - to_subtract
            indices = torch.where(adjustment_coefficient > 0)[0].cpu().numpy()
            ne_needed[0] = [(node_assign[i], i, adjustment_coefficient[i].item()) for i in indices]
        
        elif k == 2:
            # Compute global_two_hop using sparse matrix multiplication
            global_two_hop = torch.sparse.mm(self.global_adj_mtx_gpu, self.global_adj_mtx_gpu)
            to_subtract = torch.sparse.mm(self.clients_adj_matrices_gpu[node_assign[start_node]],self.clients_adj_matrices_gpu[node_assign[start_node]])
            
            # Add contributions from neighbors
            start_subtract = to_subtract[start_node].to_dense()
            for neigh in self.ccn[start_node]:
                start_subtract += self.clients_adj_matrices_gpu[node_assign[neigh]][neigh].to_dense()
                ne_needed[1].append((node_assign[neigh], neigh, 1))
            
            # Compute adjustment coefficient
            adjustment_coefficient = global_two_hop[start_node].to_dense() - start_subtract
            indices = torch.where(adjustment_coefficient > 0)[0].cpu().numpy()
            ne_needed[0] = [(node_assign[i], i, adjustment_coefficient[i].item()) for i in indices]
        
        return ne_needed

    def fast_get_global_embedding_gpu(self, embeddings, subnodes_union):
        num_nodes = len(self.adj_list)
        node_assign = self.node_assignment.tolist()
        
        hop_embeddings = []
        for hop in range(3):
            hop_matrix = []
            for node in range(num_nodes):
                if self.ccn[node] == [] or hop == 0:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                else:
                    final_embedding = embeddings[node_assign[node]][hop][node].clone()
                    ne_needed = self.get_node_embedding_needed_gpu(node, hop)
                    for hop_needed, tuples in enumerate(ne_needed):
                        for client, node_needed, num_times in tuples:
                            if node_needed in subnodes_union:
                                final_embedding += embeddings[client][hop_needed][node_needed] * num_times
                hop_matrix.append(final_embedding)
            stack = torch.stack(hop_matrix)
            hop_embeddings.append(stack)
        
        return hop_embeddings
    
    def get_global_node_states(self):
        encoder = MLPEncoder(self.client_features[0].shape[1])
        node_assign = self.node_assignment.tolist() # which client takes which node

        hop_embeddings = []
        for hop in range(1,3):
            hop_matrix = []
            for node in range(self.num_nodes):
                if node in self.ccn.keys():
                    node_feature = torch.zeros(self.client_features[0].shape[1]).to('cuda:0')
                    neighbours = torch.sparse.mm(self.global_adj_mtx_gpu, self.global_adj_mtx_gpu).to_dense()[node] if hop == 2 else self.global_adj_mtx.todense()[node].tolist()[0]
                    for neigh, value in enumerate(neighbours):
                        if value > 0:
                            node_feature += self.client_features[node_assign[neigh]][neigh].to('cuda:0') * value
                    hop_matrix.append(node_feature.tolist())
                else:
                    hop_matrix.append(torch.zeros(self.client_features[0].shape[1]).tolist())
            encoded_features = encoder(torch.tensor(hop_matrix))
            hop_embeddings.append(encoded_features)
        return hop_embeddings
            