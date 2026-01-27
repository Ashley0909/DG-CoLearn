from torch_geometric.utils import coalesce
import torch
from torch_sparse import SparseTensor

def compute_neighborhood_features(edge_index, node_features, subnodes, tot_num_nodes):
    """
    Computes 1-hop and 2-hop neighborhood summed features for each node.
    
    Args:
    - edge_index (Tensor): The edge index in COO format, shape (2, num_edges).
    - node_features (Tensor): The node features, shape (num_nodes, num_features).
    
    Returns:
    - 1-hop summed features (Tensor): Summed features of 1-hop neighbors, shape (num_nodes, num_features).
    - 2-hop summed features (Tensor): Summed features of 2-hop neighbors, shape (num_nodes, num_features).
    """    
    # Convert local node features to global
    x_global = node_features.new_zeros((tot_num_nodes, node_features.size(1)))
    x_global[subnodes] = node_features

    # Convert edge_index to SparseTensor for fast operations
    edge_index = edge_index.to(torch.long)
    edge_index, _ = coalesce(edge_index, None, tot_num_nodes, tot_num_nodes)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(tot_num_nodes, tot_num_nodes))
    
    # 1-hop neighbors (direct neighbors)
    one_hop_neighbors = adj.matmul(x_global)  # Compute one hop by Ax
    
    # 2-hop neighbors (neighbors of neighbors)
    two_hop_neighbors = one_hop_neighbors + adj.matmul(one_hop_neighbors)  # Add 1-hop features again to get total within 2-hop features
    
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