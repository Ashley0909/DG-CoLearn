import copy
import numpy as np

def gnn_aggregate(models, local_shards_sizes, data_size, client_id):
    '''Aggregate local GNN models to get global GNN model. `client_id` are the clients who participated in this snapshot, and we only aggregate them '''
    # Construct an empty global model using a local model
    global_model = copy.deepcopy(models[client_id[0]])
    global_model_params = global_model.state_dict()  # All the parameters of each layer in the model

    for name, param in global_model_params.items():
        global_model_params[name] = 0.0

    client_weights_val = np.array(local_shards_sizes) / data_size

    for lm in range(len(models)):  # For each local model
        if lm in client_id:
            for name, param in models[lm].state_dict().items():
                global_model_params[name] += param.data * client_weights_val[lm]

    global_model.load_state_dict(global_model_params)
    return global_model