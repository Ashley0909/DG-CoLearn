import torch
from torch_sparse import SparseTensor

def compute_neighborhood_features(edge_index, node_features, tot_num_nodes):
    """
    Computes 1-hop and 2-hop neighborhood summed features for each node.
    
    Args:
    - edge_index (Tensor): The edge index in COO format, shape (2, num_edges).
    - node_features (Tensor): The node features, shape (num_nodes, num_features).
    
    Returns:
    - 1-hop summed features (Tensor): Summed features of 1-hop neighbors, shape (num_nodes, num_features).
    - 2-hop summed features (Tensor): Summed features of 2-hop neighbors, shape (num_nodes, num_features).
    """    
    # Convert edge_index to SparseTensor for fast operations
    edge_index = edge_index.to(torch.long)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(tot_num_nodes, tot_num_nodes))
    
    # 1-hop neighbors (direct neighbors)
    one_hop_neighbors = adj.matmul(node_features)  # Compute one hop by Ax
    
    # 2-hop neighbors (neighbors of neighbors)
    two_hop_adj = adj.matmul(adj)
    sum_adj = two_hop_adj + adj
    two_hop_neighbors = sum_adj.matmul(node_features)  # Compute within 2 hop by (A+A^2) x
    
    return one_hop_neighbors, two_hop_neighbors

def average_feat_aggre(hop_features):
    ''' Take average of the hop_features.
    Input: hop_features => List of features submitted by clients 
    '''
    final_feature = hop_features[0]
    count = 1

    for i in range(1, len(hop_features)):
        final_feature += hop_features[i]
        count += 1
    
    return final_feature // count