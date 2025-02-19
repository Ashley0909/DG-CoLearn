import copy
import numpy as np
import torch
from itertools import islice

from utils import clustering, weighted_aggregate, merge_clients, check_ambiguous
from plot_graphs import plot_h
    

def safa_aggregate(models, local_shards_sizes, data_size):
    # Construct an empty global model using a local model
    global_model = copy.deepcopy(models[0])
    global_model_params = global_model.state_dict()  # All the parameters of each layer in the model

    for name, param in global_model_params.items():
        global_model_params[name] = 0.0

    client_weights_val = np.array(local_shards_sizes) / data_size

    for lm in range(len(models)):  # For each local model
        for name, param in models[lm].state_dict().items():
            global_model_params[name] += param.data * client_weights_val[lm]

    global_model.load_state_dict(global_model_params)
    return global_model

def gnn_aggregate(models, local_shards_sizes, data_size, client_id):
    # client_id are the clients who participated in this snapshot, we only aggregate them
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

def fedassets_aggregate(round, models, client_id, client_shard_sizes, data_size, mali_map):
    global malicious_record, e, benign_avg, malicious_avg, flag

    exist_malicious_clients = np.array([], dtype=int)
    if round == 0:
        # flag, e = 0, 0
        flag, e = 1, 0.5
        benign_avg, malicious_avg = None, None
        malicious_record = []

    # Get Fully Connected Weight for clustering
    fcw, y_label, actual = [], [], []
    printed = 0

    for id, lm in enumerate(models):
        if id not in client_id: # neglect non-participants
            continue
        if id in malicious_record: # neglect detected malicious clients
            if not isinstance(exist_malicious_clients, list):
                exist_malicious_clients = exist_malicious_clients.tolist()
            exist_malicious_clients.append(id) # .append only works for lists
            client_id.remove(id)
            continue
        local_model_params = lm.state_dict().items()
        second_last_name, second_last_param = list(islice(local_model_params, len(local_model_params) - 2, len(local_model_params) - 1))[0]

        if printed == 0:
            print("Collecting", second_last_name, ", Task has", second_last_param.shape[0], "classes")
            printed = 1

        summed_feature = torch.sum(second_last_param, dim=1)
        fcw.append(summed_feature.cpu())
        y_label.append(str(id)+'->'+str(mali_map[id]))
        actual.append(mali_map[id])

    # plot_h(matrix=fcw, path='fcw', name='Output Layer Weight', y_labels=y_label, anno=True)

    if isinstance(exist_malicious_clients, list):
        exist_malicious_clients = np.array(exist_malicious_clients)

    """ Clustering """
    # assignment, e, flag, obvious_clients = clustering(data=fcw, flag=flag, server_round=round, e=e, y_label=y_label, pca_dim=2)

    # # Extract the clients that are obvious and detect the target label
    # obvious_fcw = np.array(fcw)[obvious_clients]
    # obvious_assignment = assignment[obvious_clients]

    # # Get the client ids (Just for testing, can remove)
    # obvious_ids = np.array(client_id)[obvious_clients]
    # obv_1  = obvious_ids[obvious_assignment == 1]
    # obv_0  = obvious_ids[obvious_assignment == 0]
    # print("Only taking a look at", obvious_ids)
    # print("Class 1:", obv_1)
    # print("Class 0:", obv_0)

    # # Detect Target Label of the round
    # min_clients = obvious_fcw[obvious_assignment == 1]
    # maj_clients = obvious_fcw[obvious_assignment == 0]

    # min_clients_tensor = torch.tensor(min_clients).to('cuda')
    # maj_clients_tensor = torch.tensor(maj_clients).to('cuda')

    # min_average_per_class = min_clients_tensor.mean(dim=0)
    # maj_average_per_class = maj_clients_tensor.mean(dim=0)

    # difference = (min_average_per_class - maj_average_per_class)
    # target_label = torch.argmax(difference.abs())
    # top_values, top_indices = torch.topk(difference.abs(), k=2)

    # # Determine the status of ambiguous clients
    # ambiguous_clients = np.setdiff1d(np.arange(len(client_id)), obvious_clients)
    # assignment = check_ambiguous(ambiguous_clients, np.array(fcw), target_label, min_average_per_class, maj_average_per_class, assignment)

    # # Merge clients if there are more than 2 assignments
    # if len(np.unique(assignment)) > 2:
    #     assignment = merge_clients(np.array(fcw), assignment, target_label, min_average_per_class, maj_average_per_class)

    # # Determine which cluster is benign (if min>maj then min is malicious, vice versa)
    # sign = torch.sign(torch.argmax(difference))
    # if sign == 1:
    #     malicious_clients = np.array(client_id)[assignment == 1]
    #     benign_clients = np.array(client_id)[assignment == 0]
    # else:
    #     malicious_clients = np.array(client_id)[assignment == 0]
    #     benign_clients = np.array(client_id)[assignment == 1]

    """ Assume 100% accuracy """
    assignment = np.array(actual)
    print("assignment", assignment)
    benign_clients = np.array(client_id)[assignment == 0]
    malicious_clients = np.array(client_id)[assignment == 1]
    target_label = 9

    print("number of malicious clients", len(malicious_clients))
    print("number of benign clients", len(benign_clients))

    malicious_record.extend(malicious_clients)
    malicious_record = list(set(malicious_record))  # Avoid Duplication

    # Determine aggregation weight of each participating client
    global_model = weighted_aggregate(models, benign_clients, malicious_clients, exist_malicious_clients, client_shard_sizes, data_size, target_label, second_last_name)
    
    return global_model, benign_clients

        

    

        

