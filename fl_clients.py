import copy
import sys
import os
import random
import numpy as np
import register
import torch
import torch.nn as nn
import torch.nn.functional as functional
import math

from utils import get_exclusive_subgraph, lp_prediction, compute_mrr, nc_prediction
from plot_graphs import draw_graph, plot_h
from fl_models import ReshapeH

class EdgeDevice:
    def __init__(self, id, prev_ne, subnodes):
        self.id = id
        self.prev_ne = prev_ne
        self.curr_ne = prev_ne # same stucture as prev_ne store 0-hop , 1-hop and 2-hop
        self.prev_edge_index = []
        self.subnodes = subnodes

    def send_embeddings(self): # send the current trained NE for sharing
        return self.curr_ne

    def update_embeddings(self, shared_ne): # update the previous NE for GRU integration after sharing, or for next snapshot learning 
        self.prev_ne = shared_ne

    def upload_features(self, node_feature, total_num_nodes, encoder):
        reshape = ReshapeH(total_num_nodes)
        expanded_node_feature = reshape.reshape_to_fill(node_feature, self.subnodes)
        encoded_features = encoder(expanded_node_feature)
        return encoded_features


def distribute_models(global_model, local_models, client_ids):
    for id in client_ids:
        local_models[id] = copy.deepcopy(global_model)

def compute_loss(task_cfg, pred, true):
    '''

    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    bce_loss = nn.BCEWithLogitsLoss(size_average=True)
    mse_loss = nn.MSELoss(size_average=True)
    ce_loss = nn.CrossEntropyLoss(size_average=True)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # if multi task binary classification, treat as flatten binary
    if true.ndim > 1 and task_cfg.loss == 'ce':
        pred, true = torch.flatten(pred), torch.flatten(true)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if task_cfg.loss == 'ce':
        # multiclass
        if pred.ndim > 1:
            return ce_loss(pred, true), pred
            # pred = functional.log_softmax(pred, dim=-1)
            # return functional.nll_loss(pred, true), pred
        # binary
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif task_cfg.loss == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.
                         format(task_cfg.loss))

def train(env_cfg, task_cfg, models, optimizers, schedulers, client_ids, cm_map, fdl, last_loss_rep, rd, epoch, verbose=True):
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

    # Begin an epoch of training
    for data in fdl.fbd_list: # Traverse the data of each client
        client = data.dataset.location
        model_id = cm_map[client.id]
        if model_id not in client_ids: # neglect non-participants
            continue
        
        x, edge_index = data.dataset.node_feature.to(device), data.dataset.edge_index.to(device)
        if task_cfg.task_type == 'LP':
            edge_label = data.dataset.edge_label.to(device)

        model = models[model_id]
        optimizer = optimizers[model_id]
        scheduler = schedulers[model_id]
        optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step

        ''' Extract 2 hop subgraph of changed pairs of nodes '''
        if rd == 0 and epoch == 0 and data.dataset.previous_edge_index != []:
            exclusive_edge_index = get_exclusive_subgraph(edge_index, data.dataset.previous_edge_index.to('cuda:0'))
            print("Shrink in graph", edge_index.shape[1] - exclusive_edge_index.shape[1])

        if task_cfg.task_type == 'LP':
            if client.prev_ne is not None:
                for i in range(len(data.dataset.node_states)):
                    data.dataset.node_states[i] = client.prev_ne[i] * (1 - 0.6) + data.dataset.node_states[i] * 0.6
            predicted_y, true, client.curr_ne = model(copy.deepcopy(data.dataset))
        else:
            if client.prev_ne is not None:
                for i in range(len(data.dataset.node_states)):
                    data.dataset.node_states[i] = client.prev_ne[i] * (1 - 0.2) + data.dataset.node_states[i] * 0.2
            predicted_y, true, client.curr_ne = model(copy.deepcopy(data.dataset))

        # Compute Loss
        if task_cfg.task_type == 'LP':
            loss, _ = compute_loss(task_cfg, predicted_y, edge_label.type_as(predicted_y))
        else:
            loss, _ = compute_loss(task_cfg, predicted_y, true)
        loss.backward(retain_graph=True)  # Need retain_graph=True for temporal updates
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no gradient!")
        #     else:
        #         print(f"{name} has grad {param.grad}")
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Stop exploding gradients
        optimizer.step() # Update weights based on computed gradients
        scheduler.step()

        optimizers[model_id] = optimizer
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

    for m in range(env_cfg.n_clients):
        models[m].eval()   # Pytorch makes sure it is in evaluation mode

    with torch.no_grad():  # Don't need to compute gradients bc testing don't require updating weights
        count = 0.0 # only for getting metrics, since each client only has one batch (so dont need count for accuracy)
        metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}
        for data in fdl.fbd_list:
            if task_cfg.task_type == 'LP':
                edge_label_index, edge_label, val_nodes = data.dataset.edge_label_index.to(device), data.dataset.edge_label.to(device), data.dataset.subnodes.to(device)
                if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no validation data
                    print("Ignore clients with no validation data")
                    continue

            model_id = cm_map[data.dataset.location.id]
            if model_id not in client_ids: # neglect non-participants
                continue

            model = models[model_id]
            if task_cfg.task_type == 'LP':
                predicted_y, true, _ = model(copy.copy(data.dataset))
                loss, _ = compute_loss(task_cfg, predicted_y, edge_label.type_as(predicted_y))
                acc, ap = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
                mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index) # need raw scores for mrr
                metrics['mrr'] += mrr
                metrics['ap'] += ap
            else:
                predicted_y, true, _ = model(copy.copy(data.dataset))
                loss, _ = compute_loss(task_cfg, predicted_y, true)
                acc, macro_f1, micro_f1 = nc_prediction(predicted_y, true)
                metrics['macro_f1'] += macro_f1
                metrics['micro_f1'] += micro_f1
            # Compute Loss and other metrics
            client_test_loss[model_id] += loss.detach().item()
            client_test_acc[model_id] += acc
            count += 1

        metrics = {key: value / count for key, value in metrics.items()}
        return client_test_loss, client_test_acc, metrics
        
def global_test(global_model, server, client_ids, task_cfg, env_cfg, cm_map, fdl):
    """ Testing the aggregated global model by averaging its error on each local data """
    device = env_cfg.device
    
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
        if task_cfg.task_type == 'LP':
            edge_label_index, edge_label = data.dataset.edge_label_index.to(device), data.dataset.edge_label.to(device)
            if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no testing data
                print("Ignore participants with no testing data")
                continue

        model_id = cm_map[data.dataset.location.id]
        if model_id not in client_ids: # neglect non-participants
            continue

        if task_cfg.task_type == 'LP':
            predicted_y, _, _ = global_model(copy.copy(data.dataset))
            acc, ap = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
            print("Test Accuracy is", acc, "by client with", data.dataset.edge_index.shape[1], "edges")
            mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index)
            if not math.isnan(mrr):
                metrics['mrr'] += mrr
                accuracy, metrics['ap'] = accuracy + acc, metrics['ap'] + ap
            else:
                count -= 1
        else:
            predicted_y, true_label, _ = global_model(copy.copy(data.dataset))
            acc, macro_f1, micro_f1 = nc_prediction(predicted_y, true_label)
            accuracy, metrics['macro_f1'], metrics['micro_f1'] = accuracy + acc, metrics['macro_f1'] + macro_f1, metrics['micro_f1'] + micro_f1

        count += 1

    """ Server test model using cross client edges """
    if task_cfg.task_type == 'LP':
        server_data = server.test_loader.dataset
        if server_data.edge_index.shape[1] > 1: # Test if there are any cce
            predicted_y, true_label, _ = global_model(copy.copy(server_data))
            ccn_acc, ccn_ap = lp_prediction(predicted_y, true_label)
            ccn_mrr = compute_mrr(predicted_y, true_label, server_data.edge_label_index)
            if not math.isnan(mrr):
                metrics['mrr'] += ccn_mrr
                accuracy, metrics['ap'] = accuracy + ccn_acc, metrics['ap'] + ccn_ap
            else:
                count -= 1
            print("Test Accuracy is", ccn_acc, "with MRR", ccn_mrr, "and AP", ccn_ap, "by server with", server_data.edge_index.shape[1], "edges")

            count += 1

    metrics = {key: value / count for key, value in metrics.items()}
    
    # return test_sum_loss, accuracy/count, metrics
    return 0, accuracy/count, metrics