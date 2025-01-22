import copy
import sys
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import average_precision_score

from utils import get_exclusive_edges, lp_prediction, compute_mrr, nc_prediction
from plot_graphs import draw_graph, plot_h

class FLClient:
    def __init__(self, id):
        self.id = id

class FLBackdoorClient:
    def __init__(self, id, malicious):
        self.id = id
        self.malicious = malicious

class EdgeDevice:
    def __init__(self, id, prev_ne, weights, subnodes):
        self.id = id
        self.prev_ne = prev_ne
        self.curr_ne = prev_ne # same stucture as prev_ne
        self.prev_edge_index = None
        self.weights = weights
        self.subnodes = subnodes

    def send_embeddings(self): # send the current trained NE for sharing
        return self.curr_ne
    
    def send_weights(self):
        return self.weights
    
    def compute_weights(self, edge_index, device="cuda:0"):
        weight = self.weights.to(device)
        weight[torch.unique(edge_index)] = 1       # Include the node itself
        pred_count = torch.bincount(edge_index[1]) # Count the occurrance of the second row of edge_index (dst nodes)
        weight[:len(pred_count)] += pred_count         # bincount only shows from 0 to the max node in edge_index[1], so could be shorter than weights
        self.weights = weight
    
    def update_embeddings(self, shared_ne): # update the previous NE for GRU integration after sharing, or for next snapshot learning 
        self.prev_ne = shared_ne

class FogDevice:
    def __init__(self):
        self.shared_ne = None

    def share_embeddings(self, embeddings, weights):
        temp_embeddings = copy.deepcopy(embeddings)
        avg = torch.sum(weights, dim=0, keepdim=False)

        for i in range(len(weights)): # for each client
            indices = (weights[i] == 1).nonzero(as_tuple=True)[0]
            for l in range(2): # 2 conv layers
                for j in range(len(weights)):
                    if j != i:
                        mul = temp_embeddings[j][l][indices] * weights[j][indices][:, None]
                        embeddings[i][l][indices] += mul
                embeddings[i][l][indices] /= avg[indices][:, None] # Take average

        return embeddings

def distribute_models(global_model, local_models, client_ids):
    for id in client_ids:
        local_models[id] = copy.deepcopy(global_model)

def generate_clients_perf(env_cfg, from_file=False, s0=1e-2):
    """ Generate a series of client performance values following the specified distribution """
    if from_file:
        fname = 'gen/clients_perf_' + str(env_cfg.n_clients)
        return np.loadtxt(fname)
    
    n_clients = env_cfg.n_clients
    perf_vec = None

    # Case 1: Equal performance
    if env_cfg.perf_dist[0] == 'E':
        perf_vec = [1.0 for _ in range(n_clients)]
        while np.min(perf_vec) < s0:  # in case of super straggler
            perf_vec = [1.0 for _ in range(n_clients)]

    # Case 2: eXponential distribution of performance
    elif env_cfg.perf_dist[0] == 'X':  # ('X', None), lambda = 1/1, mean = 1
        perf_vec = [random.expovariate(1.0) for _ in range(n_clients)]
        while np.min(perf_vec) < s0:  # in case of super straggler
            perf_vec = [random.expovariate(1.0) for _ in range(n_clients)]

    # Case 3: Normal distribution of performance
    elif env_cfg.perf_dist[0] == 'N':  # ('N', rlt_sigma), mu = 1, sigma = rlt_sigma * mu
        perf_vec = [0.0 for _ in range(n_clients)]
        for i in range(n_clients):
            perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
            while perf_vec[i] <= s0:  # in case of super straggler
                perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
    else:
        print('Error> Invalid client performance distribution option')
        exit(0)

    return perf_vec

def generate_clients_crash_prob(env_cfg, n_clients=None):
    """ Generate a series of probability that the corresponding client will crash """
    prob_vec = None
    # Case 1: Equal prob
    if env_cfg.crash_dist[0] == 'E': 
        if n_clients == None:
            n_clients = env_cfg.n_clients
            prob_vec = [env_cfg.crash_dist[1] for _ in range(n_clients)]
        else:
            prob_vec = [env_cfg.crash_dist[1] for _ in range(n_clients)] + [1.0 for _ in range(env_cfg.n_clients - n_clients)]  # If there are less than env_cfg.n_clients clients allocated with data

    # Case 2: Uniform distribution of crashing prob
    elif env_cfg.crash_dist[0] == 'U':  # ('U', (low, high))
        low = env_cfg.crash_dist[1][0]
        high = env_cfg.crash_dist[1][1]
        # check
        if low < 0 or high < 0 or low > 1 or high > 1 or low >= high:
            print('Error> Invalid crash prob interval')
            exit(0)
        if n_clients == None:
            n_clients = env_cfg.n_clients
            prob_vec = [random.uniform(low, high) for _ in range(n_clients)]
        else:
            prob_vec = [random.uniform(low, high) for _ in range(n_clients)] + [1.0 for _ in range(env_cfg.n_clients - n_clients)]
    else:
        print('Error> Invalid crash prob distribution option')
        exit(0)

    return prob_vec

def generate_crash_trace(env_cfg, clients_crash_prob_vec):
    """ Generate a crash trace (length = #rounds) for simulation """
    crash_trace = []
    progress_trace = []
    for _ in range(env_cfg.n_rounds):
        crash_ids = []  # crashed ids this round
        progress = [1.0 for _ in range(env_cfg.n_clients)]  # 1.0 denotes well progressed
        for c_id in range(env_cfg.n_clients):
            rand = random.random()
            if rand <= clients_crash_prob_vec[c_id]:  # crash
                crash_ids.append(c_id)
                progress[c_id] = rand / clients_crash_prob_vec[c_id]  # progress made before crash

        crash_trace.append(crash_ids)
        progress_trace.append(progress)

    return crash_trace, progress_trace

def sort_ids_by_perf_desc(ids, clients_perf_val):
    cp_map = {}
    for id in ids:
        cp_map[id] = clients_perf_val[id]
    
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=True)
    sorted_idx = [sorted_map[i][0] for i in range(len(ids))]

    return sorted_idx

def version_filter(versions, client_ids, latest_ver, lag_tolerant=1):
    """ Apply filter to client ids by checking their model versions. If the version is lagged behind the latest version by more than lag_tolerant, it will be filtered out"""
    good_ids = []
    deprecated_ids = []
    for id in client_ids:
        if latest_ver - versions[id] <= lag_tolerant:
            good_ids.append(id)
        else:
            deprecated_ids.append(id)
    
    return good_ids, deprecated_ids

def select_clients_FCFM(make_ids, picked_ids, clients_perf_val, cross_rounders, quota):
    """ Select clients to aggregate their models according to Compensatory First-Come-First-Merge principle """
    # picked_ids are the clients that are picked in the previous round, being low priority in the current round

    picks = []
    in_time_make_ids = [m_id for m_id in make_ids if m_id not in cross_rounders]
    high_priority_ids = [h_id for h_id in in_time_make_ids if h_id not in picked_ids]
    low_priority_ids = [l_id for l_id in in_time_make_ids if l_id in picked_ids]

    # Case 0: clients finishing in time not enough for fraction C, just gather them all
    if len(in_time_make_ids) <= quota:
        return copy.deepcopy(in_time_make_ids)
    # Case 1: # of priority ids > quota
    if len(high_priority_ids) >= quota:
        sorted_priority_ids = sort_ids_by_perf_desc(high_priority_ids, clients_perf_val)
        picks = sorted_priority_ids[0:int(quota)]
    # Case 2: # of priority ids <= quota
    # the rest are picked by order of performance ("FCFM"), lowest batch overhead first
    else:
        picks += high_priority_ids  # they have priority
        sorted_low_priority_ids = sort_ids_by_perf_desc(low_priority_ids, clients_perf_val)  # FCFM
        for i in range(min(quota - len(picks), len(sorted_low_priority_ids))):
            picks.append(sorted_low_priority_ids[i])

    return picks

def select_clients_randomly(make_ids, quota):
    if quota > len(make_ids):
        quota = len(make_ids)
    selected_elements = random.sample(make_ids, quota)
    ids = sorted(selected_elements)
    return ids

def batch_sum_accuracy(predicted_y, labels, loss):
    """ Compute Accuracy = (TP+TN)/(TP+TN+FP+FN) """
    assert len(labels) == len(predicted_y)
    accuracy = torch.tensor(0.0)
    count = len(labels)

    if loss == 'mse':  # sum up (1 - relative error)
        labels = labels.view_as(predicted_y)
        predicted_y, labels = predicted_y.float(), labels.float()
        accuracy += sum(1.0 - abs((labels - predicted_y))/torch.max(predicted_y, labels)).item()
        accuracy = accuracy.detach().item()
    elif loss == 'nllLoss':
        pred = predicted_y.argmax(dim=1, keepdim=True) # output the class with highest prob
        accuracy += pred.eq(labels.view_as(pred)).sum().item()
        accuracy = accuracy.detach().item()
    elif loss == 'svmLoss':
        labels = labels.view_as(predicted_y)
        for res in labels * predicted_y:
            accuracy += torch.tensor(1.0) if res.item() > 0 else torch.tensor(0.0)
        accuracy = accuracy.detach().item()
    elif loss == 'bce':
        pred = torch.sigmoid(predicted_y).cpu().detach().numpy()
        label = labels.cpu().detach().numpy()
        accuracy = average_precision_score(label, pred)
        count = 1

    return accuracy, count

def train(models, client_ids, env_cfg, cm_map, fdl, task_cfg, last_loss_rep, round, epoch, snapshot=None, verbose=True):
    device = env_cfg.device
    if len(client_ids) == 0:
        return last_loss_rep

    num_models = env_cfg.n_clients
    client_train_loss = last_loss_rep
    for id in client_ids:
        client_train_loss[id] = 0.0
    
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # sys.stdout is all the python output statement (e.g. print statements). We write these to the null device => not printing them

    for m in range(num_models):
        models[m].train()  # Pytorch makes sure it is in training mode

    # Record Loss
    if task_cfg.loss == 'mse': # Regression
        loss_func = nn.MSELoss(reduction='mean')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss()
    elif task_cfg.loss == 'bce':
        loss_func = nn.BCEWithLogitsLoss()  # By default, it shows the average loss per data point (reduction='mean')
    elif task_cfg.loss == 'ce':
        loss_func = nn.CrossEntropyLoss()

    # One optimizer for each model (re-instantiate optimizers to clear any possible momentum
    optimizers = []
    for i in range(num_models):
        if task_cfg.optimizer == 'SGD':
            optimizers.append(optim.SGD(models[i].parameters(), lr=task_cfg.lr))
        elif task_cfg.optimizer == 'Adam':
            optimizers.append(optim.Adam(models[i].parameters(), lr=task_cfg.lr))
        else:
            print('Err> Invalid optimizer %s specified' % task_cfg.optimizer)

    # Begin an epoch of training
    if task_cfg.task_type in ['LP', 'NC']:
        for data in fdl.fbd_list: # Traverse the data of each client
            client = data.location
            model_id = cm_map[client.id]
            print("Client", client.id)
            if model_id not in client_ids: # neglect non-participants
                continue
            
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            if task_cfg.task_type == 'LP':
                edge_label_index, edge_label = data.edge_label_index.to(device), data.y.to(device)
            else:
                node_label, subnodes = data.y.to(device), data.subnodes.to(device)

            model = models[model_id]
            optimizer = optimizers[model_id]
            optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step

            # Train with only new/relevant edges
            if data.previous_edge_index != None:
                prev_edge_index = data.previous_edge_index.to(device)
                new_edge_index = get_exclusive_edges(edge_index, prev_edge_index)
            else:
                new_edge_index = edge_index

            # if (snapshot + round + epoch) == 0:  # Draw graphs (Too big to draw in NC for now)
            #     draw_graph(edge_index=edge_index, name='original', round=round, ss=snapshot, client=model_id)

            if round == 0 and epoch == 0:
                print("size of graph", new_edge_index.shape[1])
                print("Shrink in graph size", edge_index.shape[1] - new_edge_index.shape[1])

            if task_cfg.task_type == 'LP':
                predicted_y, client.curr_ne = model(x, new_edge_index, task_cfg.task_type, edge_label_index, client.prev_ne)
            else:
                predicted_y, client.curr_ne = model(x, new_edge_index, task_cfg.task_type, subnodes=subnodes, previous_embeddings=client.prev_ne)

            """ Compute the Weights for Combining Node Embedding (Only need to do it once in First Round First Epoch) """
            if (epoch + round) == 0:
                weight = client.weights.to(device)
                weight[torch.unique(new_edge_index)] = 1       # Include the node itself
                pred_count = torch.bincount(new_edge_index[1]) # Count the occurrance of the second row of edge_index (dst nodes)
                weight[:len(pred_count)] += pred_count         # bincount only shows from 0 to the max node in edge_index[1], so could be shorter than weights
                client.weights = weight

            # Train with complete data set
            # predicted_y, client.curr_ne = model(x, edge_index, task_cfg.task_type, edge_label_index, client.prev_ne)

            # Compute Loss
            if task_cfg.task_type == 'LP':
                loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
            else:
                loss = loss_func(predicted_y[subnodes], node_label)
            loss.backward(retain_graph=True)  # Use backpropagation to compute gradients
            optimizer.step() # Update weights based on computed gradients
            
            # client_train_loss[model_id] += loss.detach().item()
            client_train_loss[model_id] += loss
    else:
        for _, (inputs, labels, client) in enumerate(fdl):
            inputs, labels = inputs.to(device), labels.to(device)
            model_id = cm_map[client.id]
            if model_id not in client_ids: # neglect non-participants
                continue
            
            model = models[model_id]
            optimizer = optimizers[model_id]
            optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step
            predicted_y = model(inputs)

            # Compute Loss
            loss = loss_func(predicted_y, labels)
            loss.backward()  # Use backpropagation to compute gradients
            optimizer.step() # Update weights based on computed gradients

            client_train_loss[model_id] += loss.detach().item() * len(inputs)

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__

    return client_train_loss

def local_test(models, client_ids, task_cfg, env_cfg, cm_map, fdl, last_loss_rep, last_acc_rep, round):
    if not client_ids:
        return last_loss_rep, last_acc_rep, None
    
    device = env_cfg.device
    client_test_loss = last_loss_rep
    client_test_acc = last_acc_rep

    for id in client_ids:
        client_test_loss[id] = 0.0
        client_test_acc[id] = 0.0
    
    # Record Loss
    if task_cfg.loss == 'mse': # Regression
        loss_func = nn.MSELoss(reduction='sum')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
    elif task_cfg.loss == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif task_cfg.loss == 'ce':
        loss_func = nn.CrossEntropyLoss()

    for m in range(env_cfg.n_clients):
        models[m].eval()   # Pytorch makes sure it is in evaluation mode

    with torch.no_grad():  # Don't need to compute gradients bc testing don't require updating weights
        if task_cfg.task_type in ['LP', 'NC']:
            count = 0.0 # only for getting metrics, since each client only has one batch (so dont need count for accuracy)
            metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0,  'macro_auc': 0.0, 'micro_auc': 0.0, 'mrr': 0.0}
            for data in fdl.fbd_list:
                x, edge_index = data.x.to(device), data.edge_index.to(device)
                if task_cfg.task_type == 'LP':
                    edge_label_index, edge_label = data.edge_label_index.to(device), data.y.to(device)
                else:
                    node_label, subnodes = data.y.to(device), data.subnodes.to(device)
                model_id = cm_map[data.location.id]
                print("Client", data.location.id)
                if model_id not in client_ids: # neglect non-participants
                    continue

                model = models[model_id]
                if task_cfg.task_type == 'LP':
                    predicted_y, _ = model(x, edge_index, task_cfg.task_type, edge_label_index)
                    loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
                    acc, ap, macro_f1, macro_auc, micro_auc = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
                    mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y))
                    metrics['mrr'] += mrr
                    metrics['ap'], metrics['macro_f1'], metrics['macro_auc'], metrics['micro_auc'] = metrics['ap'] + ap, metrics['macro_f1'] + macro_f1, metrics['macro_auc'] + macro_auc, metrics['micro_auc'] + micro_auc
                else:
                    predicted_y, _ = model(x, edge_index, task_cfg.task_type, subnodes=subnodes)
                    loss = loss_func(predicted_y[subnodes], node_label)
                    acc, macro_f1, micro_f1 = nc_prediction(predicted_y[subnodes], node_label)
                    metrics['macro_f1'], metrics['micro_f1'] = metrics['macro_f1'] + macro_f1, metrics['macro_f1'] + micro_f1

                # Compute Loss and other metrics
                client_test_loss[model_id] += loss.detach().item()
                client_test_acc[model_id] += acc
                count += 1

            metrics = {key: value / count for key, value in metrics.items()}
            return client_test_loss, client_test_acc, metrics
        else:
            count = [0 for _ in range(env_cfg.n_clients)]
            for _, (inputs, labels, client) in enumerate(fdl):
                inputs, labels = inputs.to(device), labels.to(device)
                model_id = cm_map[client.id]

                if model_id not in client_ids: # neglect non-participants
                    count[model_id] = 1
                    continue

                model = models[model_id]
                predicted_y = model(inputs)

                # Compute Loss
                loss = loss_func(predicted_y, labels)
                client_test_loss[model_id] += loss.detach().item()

                # Compute Accuracy (= accuracy / count)
                b_acc, b_count = batch_sum_accuracy(predicted_y, labels, task_cfg.loss)
                client_test_acc[model_id] += b_acc
                count[model_id] += b_count
    
            return client_test_loss, [client_test_acc[x] / count[x] for x in range(len(client_test_acc))], None

def global_test(global_model, client_ids, task_cfg, env_cfg, cm_map, fdl, round):
    """ Testing the aggregated global model by averaging its error on each local data """
    device = env_cfg.device
    test_sum_loss = [0 for _ in range(env_cfg.n_clients)]
    
    # Define loss based on task
    if task_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
    elif task_cfg.loss == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif task_cfg.loss == 'ce':
        loss_func = nn.CrossEntropyLoss()

    # Initialize evaluation mode
    global_model.eval()

    # Local evaluation, batch-wise
    accuracy = 0.0
    count = 0
    metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'macro_auc': 0.0, 'micro_auc': 0.0, 'mrr': 0.0}

    if task_cfg.task_type in ['LP', 'NC']:
        for data in fdl.fbd_list:
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            if task_cfg.task_type == 'LP':
                edge_label_index, edge_label = data.edge_label_index.to(device), data.y.to(device)
            else:
                node_label, subnodes = data.y.to(device), data.subnodes.to(device)

            model_id = cm_map[data.location.id]
            if model_id not in client_ids: # neglect non-participants
                continue

            if task_cfg.task_type == 'LP':
                predicted_y, _ = global_model(x, edge_index, edge_label_index, task_cfg.task_type)
                loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
                acc, ap, macro_f1, macro_auc, micro_auc = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
                mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y))
                metrics['mrr'] += mrr
                accuracy, metrics['ap'], metrics['macro_f1'], metrics['macro_auc'], metrics['micro_auc'] = accuracy + acc, metrics['ap'] + ap, metrics['macro_f1'] + macro_f1, metrics['macro_auc'] + macro_auc, metrics['micro_auc'] + micro_auc
            else:
                predicted_y, _ = global_model(x, edge_index, task_cfg.task_type, subnodes=subnodes)
                loss = loss_func(predicted_y[subnodes], node_label)
                acc, macro_f1, micro_f1 = nc_prediction(predicted_y[subnodes], node_label)
                accuracy, metrics['macro_f1'], metrics['micro_f1'] = accuracy + acc, metrics['macro_f1'] + macro_f1, metrics['macro_f1'] + micro_f1

            # Compute Loss
            if not torch.isnan(loss).any():
                test_sum_loss[model_id] += loss.detach().item()

            count += 1
        metrics = {key: value / count for key, value in metrics.items()}
        
        return test_sum_loss, accuracy/count, metrics
    
    else:
        for _, (inputs, labels, client) in enumerate(fdl):
            inputs, labels = inputs.to(device), labels.to(device)
            model_id = cm_map[client.id]
            if model_id not in client_ids: # neglect non-participants
                continue

            predicted_y = global_model(inputs)
            # Compute Loss
            loss = loss_func(predicted_y, labels)
            test_sum_loss[model_id] += loss.detach().item()

            # Compute Accuracy
            b_acc, b_count = batch_sum_accuracy(predicted_y, labels, task_cfg.loss)
            accuracy += b_acc
            count += b_count

        return test_sum_loss, accuracy/count, metrics

def node_check(edge_index, new_changes, stack, edge, threshold=15):
    if len(new_changes) == 0:
        return []

    # root = stack[-1]
    
    score = 1.0
    while stack:
        current_node = stack.pop()
        neighbours = edge_index[1][edge_index[0] == current_node]
        if score == 1.0 and edge == 0:
            mask = ~(new_changes[0] == current_node)
            new_changes = new_changes[:, mask]

        for neighbour in neighbours:
            if neighbour.item() not in new_changes[0]:
                degree = torch.sum(edge_index[0] == neighbour.item())
                curr_score = score / degree
                # print("int(neighbour)", int(neighbour), "score", score, "number of difference", torch.sum(torch.logical_xor(feature_aggregation[root],feature_aggregation[neighbour.item()])))
                if curr_score >= threshold:
                    new = torch.tensor([[current_node],[neighbour.item()]]).to('cuda')
                    new_changes = torch.cat((new_changes, new),1)
                    stack.append(neighbour.item())
                else:
                    score = 1.0  #if the node exceed the threshold influence, we do not add it to the stack and end traversing the branch

        if neighbours.shape[0] == 1:  # the node is a leaf, reached the end of a branch
            score = 1.0
        
        score = score / 2
    
    return new_changes

def get_relevant_edges(edge_index, new_changes):
    if len(new_changes) == 0:
        return []

    stack = new_changes[1].tolist()  #List all target nodes to start with

    while stack:
        current_node = stack.pop()
        new_changes = node_check(edge_index, new_changes, [current_node], 1)

        while current_node in stack:
            stack.remove(current_node)

        # pair = new_changes[1][new_changes[0] == current_node]

        # for p in pair: # in case there are multiple pairs
        #     print("Pair", p)
        #     if p.item() == current_node:  # updated feature
        #         new_changes = node_check(edge_index, new_changes, [current_node], 0)
        #     else:
        #         new_changes = node_check(edge_index, new_changes, [current_node], 1)

    # Step 1: Combine the updated_edge into tuples and sort
    combined = list(zip(new_changes[0].tolist(), new_changes[1].tolist()))
    sorted_combined = sorted(combined, key=lambda x: (x[0], x[1]))

    # Step 2: Separate the sorted elements back into separate tensors
    sorted_edges = torch.tensor(list(zip(*sorted_combined))).to('cuda')
    
    return sorted_edges