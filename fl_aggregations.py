import copy
import numpy as np
import torch

def gnn_aggregate(models, local_shards_sizes, data_size, client_id, global_model):
    '''Aggregate local GNN models to get global GNN model. `client_id` are the clients who participated in this snapshot, and we only aggregate them '''
    # Move model to CPU for state_dict operations
    device = next(global_model.parameters()).device
    global_model.to('cpu')
    
    # Zero out global model parameters
    global_model_params = global_model.state_dict()
    for name, param in global_model_params.items():
        global_model_params[name] = 0.0

    # Calculate weights based on data size
    client_weights_val = np.array(local_shards_sizes) / np.array(local_shards_sizes).sum()

    # Aggregate parameters from participating clients
    for lm in range(len(models)):  # For each client's parameters
        if lm in client_id:
            for name, param in models[lm].items():
                global_model_params[name] += param.data * client_weights_val[lm]

    global_model.load_state_dict(global_model_params)
    # Move model back to original device
    global_model.to(device)
    torch.cuda.synchronize()
    return global_model