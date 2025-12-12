import copy
import numpy as np
import torch

def gnn_aggregate(models, local_shards_sizes, data_size, client_id):
    '''Aggregate local GNN models to get global GNN model. `client_id` are the clients who participated in this snapshot, and we only aggregate them '''
    # Construct an empty global model using a local model
    global_model = copy.deepcopy(models[client_id[0]])
    global_model_params = global_model.state_dict()  # All the parameters of each layer in the model

    for name, param in global_model_params.items():
        global_model_params[name] = 0.0

    client_weights_val = np.array(local_shards_sizes) / np.array(local_shards_sizes).sum()

    for lm in range(len(models)):  # For each local model
        if lm in client_id:
            for name, param in models[lm].state_dict().items():
                global_model_params[name] += param.data * client_weights_val[lm]

    global_model.load_state_dict(global_model_params)
    return global_model

def gnn_weighted_aggregate(models, local_shards_sizes, client_id, prev_global_model=None, agg_mode='fedavg', alpha=0.7, weight_mode='uniform'):
    """
    Aggregate local models.

    - agg_mode:
        'fedavg'   -> pure average of participating clients
        'partial'  -> blend previous global with average: (1-alpha)*prev + alpha*avg
    - weight_mode:
        'uniform'  -> equal client weights
        'sqrt'     -> sqrt(size) weights (damp heavy shards)
    """
    # template from first participating client
    global_model = copy.deepcopy(models[client_id[0]])
    avg_params = global_model.state_dict()

    # zero accumulators for float params
    for name, param in avg_params.items():
        if torch.is_floating_point(param) and 'running_' not in name:
            avg_params[name] = torch.zeros_like(param)

    # weights over participating clients only
    subset_sizes = np.array([local_shards_sizes[i] for i in client_id], dtype=float)
    if weight_mode == 'sqrt':
        w = np.sqrt(np.clip(subset_sizes, 1.0, None))
    else:
        w = np.ones(len(client_id), dtype=float)
    w = w / w.sum()
    id_to_w = {cid: float(wt) for cid, wt in zip(client_id, w)}

    # accumulate float params
    for lm in client_id:
        wlm = id_to_w[lm]
        state = models[lm].state_dict()
        for name, param in state.items():
            if torch.is_floating_point(param) and 'running_' not in name:
                avg_params[name] += param.detach() * wlm

    # copy non-floats and BN running stats from a reference client
    ref_sd = models[client_id[0]].state_dict()
    for name, param in ref_sd.items():
        if (not torch.is_floating_point(param)) or ('running_' in name):
            avg_params[name] = param.clone()

    # optional partial averaging for stability
    if agg_mode == 'partial' and prev_global_model is not None:
        prev_sd = prev_global_model.state_dict()
        for name, param in avg_params.items():
            if torch.is_floating_point(param) and 'running_' not in name:
                avg_params[name] = (1.0 - alpha) * prev_sd[name] + alpha * param
            else:
                # keep non-floats/BN running stats already copied from ref
                pass

    global_model.load_state_dict(avg_params)
    return global_model