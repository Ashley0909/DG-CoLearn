import copy
import sys
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import math

from sklearn.metrics import average_precision_score

from utils import get_exclusive_subgraph, lp_prediction, compute_mrr, nc_prediction
from plot_graphs import draw_graph, plot_h

class EdgeDevice:
    def __init__(self, id, prev_ne, subnodes):
        self.id = id
        self.prev_ne = prev_ne
        self.curr_ne = prev_ne # same stucture as prev_ne store 0-hop , 1-hop and 2-hop
        self.prev_edge_index = None
        self.subnodes = subnodes

    def send_embeddings(self): # send the current trained NE for sharing
        return self.curr_ne

    def update_embeddings(self, shared_ne): # update the previous NE for GRU integration after sharing, or for next snapshot learning 
        self.prev_ne = shared_ne

def distribute_models(global_model, local_models, client_ids):
    for id in client_ids:
        local_models[id] = copy.deepcopy(global_model)

def train(models, client_ids, env_cfg, cm_map, fdl, task_cfg, last_loss_rep, verbose=True):
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

    schedulers = []
    for i in range(num_models):
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], [10001], gamma=10**-2))

    # Begin an epoch of training
    for data in fdl.fbd_list: # Traverse the data of each client
        client = data.location
        model_id = cm_map[client.id]
        if model_id not in client_ids: # neglect non-participants
            continue
        
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if task_cfg.task_type == 'LP':
            edge_label_index, edge_label, train_nodes = data.edge_label_index.to(device), data.y.to(device), data.subnodes.to(device)
        else:
            node_label, train_nodes = data.y.to(device), data.subnodes.to(device) # data.subnode is the client's training nodes

        model = models[model_id]
        optimizer = optimizers[model_id]
        scheduler = schedulers[model_id]
        # optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step

        # ''' Extract 2 hop subgraph of changed pairs of nodes '''
        # if data.previous_edge_index != None:
        #     exclusive_edge_index = get_exclusive_subgraph(edge_index, data.previous_edge_index.to('cuda:0'))
        #     print("Shrink in graph", edge_index.shape[1] - exclusive_edge_index.shape[1])

        if task_cfg.task_type == 'LP':
            predicted_y, client.curr_ne = model(x, edge_index, task_cfg.task_type, edge_label_index, subnodes=train_nodes, previous_embeddings=client.prev_ne)
        else:
            predicted_y, client.curr_ne = model(x, edge_index, task_cfg.task_type, subnodes=train_nodes, previous_embeddings=client.prev_ne)

        # Compute Loss
        if task_cfg.task_type == 'LP':
            loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
        else:
            loss = loss_func(predicted_y[train_nodes], node_label)
        loss.backward(retain_graph=True)  # Use backpropagation to compute gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1) # Stop exploding gradients
        optimizer.step() # Update weights based on computed gradients
        optimizer.zero_grad()
        scheduler.step()

        schedulers[model_id] = scheduler
        optimizers[model_id] = optimizer
        # client_train_loss[model_id] += loss.detach().item()
        client_train_loss[model_id] += loss

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__

    return client_train_loss

def local_test(models, client_ids, task_cfg, env_cfg, cm_map, fdl, last_loss_rep, last_acc_rep):
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
        count = 0.0 # only for getting metrics, since each client only has one batch (so dont need count for accuracy)
        metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}
        for data in fdl.fbd_list:
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            if task_cfg.task_type == 'LP':
                edge_label_index, edge_label, val_nodes = data.edge_label_index.to(device), data.y.to(device), data.subnodes.to(device)
                if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no validation data
                    print("Ignore clients with no validation data")
                    continue
            else:
                node_label, val_nodes = data.y.to(device), data.subnodes.to(device)
            model_id = cm_map[data.location.id]
            if model_id not in client_ids: # neglect non-participants
                continue

            model = models[model_id]
            if task_cfg.task_type == 'LP':
                predicted_y, _ = model(x, edge_index, task_cfg.task_type, edge_label_index, subnodes=val_nodes)
                loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
                acc, ap, macro_f1 = lp_prediction(torch.sigmoid(predicted_y), edge_label.type_as(predicted_y))
                mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index) # need raw scores for mrr
                metrics['mrr'] += mrr
                metrics['ap'], metrics['macro_f1'] = metrics['ap'] + ap, metrics['macro_f1'] + macro_f1
            else:
                predicted_y, _ = model(x, edge_index, task_cfg.task_type, subnodes=val_nodes)
                loss = loss_func(predicted_y[val_nodes], node_label)
                acc, macro_f1, micro_f1 = nc_prediction(predicted_y[val_nodes], node_label)
                metrics['macro_f1'], metrics['micro_f1'] = metrics['macro_f1'] + macro_f1, metrics['macro_f1'] + micro_f1
            # Compute Loss and other metrics
            client_test_loss[model_id] += loss.detach().item()
            client_test_acc[model_id] += acc
            count += 1

        metrics = {key: value / count for key, value in metrics.items()}
        return client_test_loss, client_test_acc, metrics
        
def global_test(global_model, client_ids, task_cfg, env_cfg, cm_map, fdl):
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
    metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}

    for data in fdl.fbd_list:
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if task_cfg.task_type == 'LP':
            edge_label_index, edge_label, test_nodes = data.edge_label_index.to(device), data.y.to(device), data.subnodes.to(device)
            if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no testing data
                print("Ignore participants with no testing data")
                continue
        else:
            node_label, test_nodes = data.y.to(device), data.subnodes.to(device)

        model_id = cm_map[data.location.id]
        if model_id not in client_ids: # neglect non-participants
            continue

        if task_cfg.task_type == 'LP':
            predicted_y, _ = global_model(x, edge_index, task_cfg.task_type, edge_label_index, subnodes=test_nodes)
            predicted_score = torch.sigmoid(predicted_y)
            loss = loss_func(predicted_y, edge_label.type_as(predicted_y))
            acc, ap, macro_f1 = lp_prediction(predicted_score, edge_label.type_as(predicted_y))
            mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index)
            if not math.isnan(mrr):
                metrics['mrr'] += mrr
                accuracy, metrics['ap'], metrics['macro_f1'] = accuracy + acc, metrics['ap'] + ap, metrics['macro_f1'] + macro_f1
            else:
                count -= 1
        else:
            predicted_y, _ = global_model(x, edge_index, task_cfg.task_type, subnodes=test_nodes)
            loss = loss_func(predicted_y[test_nodes], node_label)
            acc, macro_f1, micro_f1 = nc_prediction(predicted_y[test_nodes], node_label)
            accuracy, metrics['macro_f1'], metrics['micro_f1'] = accuracy + acc, metrics['macro_f1'] + macro_f1, metrics['macro_f1'] + micro_f1

        # Compute Loss
        if not torch.isnan(loss).any():
            test_sum_loss[model_id] += loss.detach().item()

        count += 1
    metrics = {key: value / count for key, value in metrics.items()}
    
    return test_sum_loss, accuracy/count, metrics