import copy
import sys
import os
import time
import register
import torch
import torch.nn as nn
import math

from utils import get_exclusive_subgraph, lp_prediction, compute_mrr, nc_prediction, find_common_nodes
from plot_graphs import draw_graph, plot_h
from fl_models import ReshapeH

class EdgeDevice:
    def __init__(self, id, prev_ne, subnodes):
        self.id = id
        self.prev_ne = prev_ne
        self.curr_ne = prev_ne # same stucture as prev_ne store 1-hop and 2-hop
        self.prev_edge_index = []
        self.subnodes = subnodes
        self.h0 = None # Implementation to Paper (NE exchange)
        self.proj_1hop = None # Implementation to Paper (Set it later because of dynamic tensor size)

    # def send_embeddings(self): # send the current trained NE for sharing 
    #     return self.curr_ne

    def update_embeddings(self, shared_ne): # update the previous NE for GRU integration after sharing, or for next snapshot learning 
        self.prev_ne = shared_ne

    # def upload_features(self, node_feature, total_num_nodes, encoder):
    #     reshape = ReshapeH(total_num_nodes)
    #     expanded_node_feature = reshape.reshape_to_fill(node_feature, self.subnodes)
    #     encoded_features = encoder(expanded_node_feature)
    #     return encoded_features
    
    def send_ccn_embeddings(self, ccn): # Our Implementation of NE Exchange Scheme 
        sent = {}
        for node in self.subnodes:
            if node.item() in ccn: # ccn records bidirectional edges, if node is a key in ccn, it means that it will also be a value in ccn 
                sent[node] = {
                    0: self.h0[node],
                    1: self.curr_ne[0][node],
                }
        return sent
    
    def receive_from_server(self, messages):  # Our Implementation of NE Exchange Scheme  
        ''' Client receives messages from server and update their node embeddings.
        
        messages[node] = {
            '1hop': Tensor for additional NE to add to current 1hop NE,
            '2hop': Tensor for additional NE to add to current 2hop NE,
            'ccn_count': Int for number of cross client neighbours of node
        } 
        '''
        for node, msg in messages.items():
            h2_local_add = self.h0[node] * msg['ccn_count']
            if self.proj_1hop is None:
                self.proj_1hop = nn.Linear(h2_local_add.shape[0], 16)
            local_0hop = self.proj_1hop(h2_local_add).detach().clone()
            if msg['1hop'] is not None:
                self.curr_ne[0][node] += msg['1hop']

            self.curr_ne[1][node] = self.curr_ne[1][node] + (local_0hop + msg['2hop']) if isinstance(msg['2hop'], torch.Tensor) else self.curr_ne[1][node] + local_0hop

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
        
        edge_index = data.dataset.edge_index
        if task_cfg.task_type == 'LP':
            edge_label = data.dataset.edge_label

        model = models[model_id]
        optimizer = optimizers[model_id]
        scheduler = schedulers[model_id]
        optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step

        ''' Extract 2 hop subgraph of changed pairs of nodes '''
        if rd == 0 and epoch == 0 and data.dataset.previous_edge_index != []:
            exclusive_edge_index = get_exclusive_subgraph(edge_index, data.dataset.previous_edge_index)
            print(f"Only learn {exclusive_edge_index.shape[1]} edges. Shrink in graph: {edge_index.shape[1] - exclusive_edge_index.shape[1]}")
            # Record k and \overline{k}
            data.dataset.edge_index = exclusive_edge_index.to('cpu')
            all_nodes = copy.deepcopy(data.dataset.subnodes)
            data.dataset.common = find_common_nodes(all_nodes, exclusive_edge_index)

        if len(data.dataset.edge_index[0]) != 0:
            ''' There might be a case where client does not have new graph to learn, if so, we do nothing. '''
            # Import previous state as node_states
            if client.prev_ne is not None:
                for i in range(len(data.dataset.node_states)):
                    data.dataset.node_states[i] = client.prev_ne[i]
            start_time = time.time()
            predicted_y, true, client.curr_ne, h_0 = model(copy.deepcopy(data.dataset))
            print(f"Time taken for Client {model_id} to train: {time.time() - start_time}")

            client.h0 = h_0.detach().clone() # Append h_0 to client.h0 (for NE exchange)

            # Compute Loss
            if task_cfg.task_type == 'LP':
                loss, _ = compute_loss(task_cfg, predicted_y, edge_label.type_as(predicted_y))
            else:
                loss, _ = compute_loss(task_cfg, predicted_y, true)
            loss.backward(retain_graph=True)  # Need retain_graph=True for temporal updates
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
                edge_label_index, edge_label, val_nodes = data.dataset.edge_label_index, data.dataset.edge_label, data.dataset.subnodes
                if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no validation data
                    print("Ignore clients with no validation data")
                    continue

            model_id = cm_map[data.dataset.location.id]
            if model_id not in client_ids: # neglect non-participants
                continue

            model = models[model_id]
            if task_cfg.task_type == 'LP':
                predicted_y, true, _, _ = model(copy.copy(data.dataset))
                loss, _ = compute_loss(task_cfg, predicted_y, edge_label.type_as(predicted_y))
                acc, ap = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
                mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index) # need raw scores for mrr
                metrics['mrr'] += mrr
                metrics['ap'] += ap
            else:
                predicted_y, true, _, _ = model(copy.copy(data.dataset))
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

    # Initialize evaluation mode
    global_model.eval()

    # Local evaluation, batch-wise
    accuracy = 0.0
    count = 0
    metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}

    for data in fdl.fbd_list:
        if task_cfg.task_type == 'LP':
            edge_label_index, edge_label = data.dataset.edge_label_index, data.dataset.edge_label
            if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no testing data
                print("Ignore participants with no testing data")
                continue

        model_id = cm_map[data.dataset.location.id]
        if model_id not in client_ids: # neglect non-participants
            continue

        if task_cfg.task_type == 'LP':
            predicted_y, _, _, _ = global_model(copy.copy(data.dataset))
            acc, ap = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
            print("Test Accuracy is", acc, "by client with", data.dataset.edge_index.shape[1], "edges")
            mrr = compute_mrr(predicted_y, edge_label.type_as(predicted_y), edge_label_index)
            if not math.isnan(mrr):
                metrics['mrr'] += mrr
                accuracy, metrics['ap'] = accuracy + acc, metrics['ap'] + ap
            else:
                count -= 1
        else:
            predicted_y, true_label, _, _ = global_model(copy.copy(data.dataset))
            acc, macro_f1, micro_f1 = nc_prediction(predicted_y, true_label)
            accuracy, metrics['macro_f1'], metrics['micro_f1'] = accuracy + acc, metrics['macro_f1'] + macro_f1, metrics['micro_f1'] + micro_f1

        count += 1

    """ Server test model using cross client edges """
    if task_cfg.task_type == 'LP':
        server_data = server.test_loader.dataset
        if server_data.edge_index.shape[1] > 1: # Test if there are any cce
            predicted_y, true_label, _, _ = global_model(copy.copy(server_data))
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

def catastrophic_forgetting_test(global_model, client_ids, task_cfg, env_cfg, cm_map, data_dict):
    device = env_cfg.device
    fdl = data_dict['data']
    original_metric = data_dict['metric']

    # Initialize evaluation mode
    global_model.eval()

    # Local evaluation, batch-wise
    full_f1, accuracy = 0.0, 0.0
    count = 0
    metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}
    forgetting_dict = {}

    for data in fdl.fbd_list:
        if task_cfg.task_type == 'LP':
            edge_label = data.dataset.edge_label
            if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no testing data
                print("Ignore participants with no testing data")
                continue

        model_id = cm_map[data.dataset.location.id]
        if model_id not in client_ids: # neglect non-participants
            continue

        if task_cfg.task_type == 'LP':
            predicted_y, _, _, _ = global_model(copy.copy(data.dataset))
            acc, ap = lp_prediction(predicted_y, edge_label.type_as(predicted_y))
            accuracy, metrics['ap'] = accuracy + acc, metrics['ap'] + ap
        else:
            predicted_y, true_label, _, _ = global_model(copy.copy(data.dataset))
            _, _, micro_f1 = nc_prediction(predicted_y, true_label)
            full_f1 += micro_f1
        count += 1

    if task_cfg.task_type == 'LP':
        current_acc = accuracy/count
        acc_forgetting = original_metric['best_acc'] - current_acc
        forgetting_dict['acc'] = acc_forgetting
        current_ap = metrics['ap']/count
        ap_forgetting = original_metric['best_ap'] - current_ap
        forgetting_dict['ap'] = ap_forgetting
    else:
        current_f1 = full_f1 / count
        forgetting = original_metric['best_f1'] - current_f1
        forgetting_dict['f1'] = forgetting 
    
    return forgetting_dict