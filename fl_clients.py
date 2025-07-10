import copy
import sys
import os
import time
import register
import torch
import torch.nn as nn
import torch.optim as optim
import math
import ray

from utils import get_exclusive_subgraph, lp_prediction, compute_mrr, nc_prediction, find_common_nodes
from plot_graphs import draw_graph, plot_h
from fl_models import ReshapeH

@ray.remote(num_gpus=1) #tag this class as ray actor
class EdgeDevice:
    def __init__(self, id, prev_ne, subnodes):
        self.id = id
        self.prev_ne = prev_ne
        self.curr_ne = prev_ne # same stucture as prev_ne store 0-hop , 1-hop and 2-hop
        self.prev_edge_index = []
        self.subnodes = subnodes
        #Rayed variable
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data = None

    # Ray getter setters
    # === GETTERS ===
    def get_id(self):
        return self.id
    def get_prev_ne(self):
        return self.prev_ne
    def get_curr_ne(self):
        return self.curr_ne
    def get_prev_edge_index(self):
        return self.prev_edge_index
    def get_subnodes(self):
        return self.subnodes
    
    def get_model(self):
        return self.model
    def get_model_parameters(self):
        self.model.to("cpu")
        return self.model.state_dict()
    
    def get_model_device(self):
        return next(self.model.parameters()).device

    def get_data(self):
        return self.data
    def get_optimizer(self):
        return self.optimizer
    def get_rayed_param(self):
        return (self.model,self.optimizer,self.scheduler,self.data)
    # === SETTERS ===
    def set_id(self, value):
        self.id = value
    def set_prev_ne(self, value):
        self.prev_ne = value
    def set_curr_ne(self, value):
        self.curr_ne = value
    def set_prev_edge_index(self, value):
        self.prev_edge_index = value
    def set_subnodes(self, value):
        self.subnodes = value
    
    def set_model(self,value):
        self.model = value
    def set_model_parameters(self, value):
        self.model.load_state_dict(value)
    def set_model_to_device(self, device):
        self.model.to(device)

    def set_data(self,value):
        self.data = value
    def set_optimizer(self,type,lr,weight_decay = None,betas = None):
        if type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999))
        else:
            self.optimizer = None
    def set_scheduler(self,T_max,eta_min):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, eta_min)
    # === Additional Func ===
    def set_train_mode(self):
        self.model.train()
    def set_eval_mode(self):
        self.model.eval()
    def ray_train(self,data,device,task_cfg,rd,epoch):
        import time
        
        print(f"[Client {self.id}] Device being used: {device}")
        print(f"[Client {self.id}] Model device: {next(self.model.parameters()).device}")
        
        # Synchronize CUDA before starting timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        self.optimizer.zero_grad() # Reset the gradients of all model parameters before performing a new optimization step
        edge_index = data.dataset.edge_index.to(device)
        
        if task_cfg.task_type== 'LP':
            edge_label = data.dataset.edge_label.to(device)

        ''' Extract 2 hop subgraph of changed pairs of nodes '''
        if rd == 0 and epoch == 0 and data.dataset.previous_edge_index != []:
            exclusive_edge_index = get_exclusive_subgraph(edge_index, data.dataset.previous_edge_index.to('cuda:0'))
            print(f"Only learn {exclusive_edge_index.shape[1]} edges. Shrink in graph: {edge_index.shape[1] - exclusive_edge_index.shape[1]}")
            # Record k and \overline{k}
            data.dataset.edge_index = exclusive_edge_index.to('cpu')
            all_nodes = copy.deepcopy(data.dataset.subnodes)
            data.dataset.common = find_common_nodes(all_nodes, exclusive_edge_index)
        

        if len(data.dataset.edge_index[0]) != 0:
            ''' There might be a case where client does not have new graph to learn, if so, we do nothing. '''
            # Import previous state as node_states
            if self.prev_ne is not None:
                for i in range(len(data.dataset.node_states)):
                    data.dataset.node_states[i] = self.prev_ne[i]
            
            # Move all tensors to GPU before training
            data.dataset.node_feature = data.dataset.node_feature.to(device)
            data.dataset.edge_feature = data.dataset.edge_feature.to(device)
            data.dataset.node_states = [state.to(device) for state in data.dataset.node_states]
            data.dataset.edge_index = data.dataset.edge_index.to(device)
            if task_cfg.task_type == 'LP':
                data.dataset.edge_label_index = data.dataset.edge_label_index.to(device)
            else:
                data.dataset.node_label = data.dataset.node_label.to(device)
                data.dataset.node_label_index = data.dataset.node_label_index.to(device)

            #Training
            predicted_y, true, self.curr_ne = self.model(copy.deepcopy(data.dataset))

            if task_cfg.task_type == 'LP':
                loss, _ = compute_loss(task_cfg, predicted_y, edge_label.type_as(predicted_y))
            else:
                loss, _ = compute_loss(task_cfg, predicted_y, true)
            
            loss.backward(retain_graph=True) # Need retain_graph=True for temporal updates
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Stop exploding gradients
            self.optimizer.step() # Update weights based on computed gradients
            self.scheduler.step()
            
            # Synchronize CUDA before ending timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"[Client {self.id}] Train function execution time: {total_time:.4f} seconds")
            
            return (loss, None)
        else:
            # Synchronize CUDA before ending timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"[Client {self.id}] Train function execution time: {total_time:.4f} seconds")
            
            # Return a zero loss when there are no edges to train on
            return (torch.tensor(0.0, device=device), None)
        
    def ray_test(self,data,device,task_cfg):
        import time
        
        print(f"[Client {self.id}] Testing on device: {device}")
        print(f"[Client {self.id}] Model device during test: {next(self.model.parameters()).device}")
        metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}
        model = self.model
        
        # Synchronize CUDA before starting timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Move all tensors to GPU before testing
            data.dataset.node_feature = data.dataset.node_feature.to(device)
            data.dataset.edge_feature = data.dataset.edge_feature.to(device)
            data.dataset.node_states = [state.to(device) for state in data.dataset.node_states]
            data.dataset.edge_index = data.dataset.edge_index.to(device)
            if task_cfg.task_type == 'LP':
                data.dataset.edge_label_index = data.dataset.edge_label_index.to(device)
                data.dataset.edge_label = data.dataset.edge_label.to(device)
            else:
                data.dataset.node_label = data.dataset.node_label.to(device)
                data.dataset.node_label_index = data.dataset.node_label_index.to(device)

            if task_cfg.task_type == 'LP':
                predicted_y, true, _ = model(copy.copy(data.dataset))
                loss, _ = compute_loss(task_cfg, predicted_y, data.dataset.edge_label.type_as(predicted_y))
                acc, ap = lp_prediction(predicted_y, data.dataset.edge_label.type_as(predicted_y))
                mrr = compute_mrr(predicted_y, data.dataset.edge_label.type_as(predicted_y), data.dataset.edge_label_index)
                metrics['mrr'] += mrr
                metrics['ap'] += ap
            else:
                predicted_y, true, _ = model(copy.copy(data.dataset))
                loss, _ = compute_loss(task_cfg, predicted_y, true)
                acc, macro_f1, micro_f1 = nc_prediction(predicted_y, true)
                metrics['macro_f1'] += macro_f1
                metrics['micro_f1'] += micro_f1
        
        # Synchronize CUDA before ending timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"[Client {self.id}] Test function execution time: {total_time:.4f} seconds")
        
        return (metrics,loss,acc)
    
    def send_embeddings(self): # send the current trained NE for sharing
        return self.curr_ne

    def update_embeddings(self, shared_ne): # update the previous NE for GRU integration after sharing, or for next snapshot learning 
        self.prev_ne = shared_ne

    def upload_features(self, node_feature, total_num_nodes, encoder):
        reshape = ReshapeH(total_num_nodes)
        expanded_node_feature = reshape.reshape_to_fill(node_feature, self.subnodes)
        encoded_features = encoder(expanded_node_feature)
        return encoded_features


def distribute_models(global_model, clients):
    global_model.to("cpu")
    global_param = global_model.state_dict()
    for client in clients:
        client.set_model_to_device.remote("cpu") #needed because transferred from another machine
        client.set_model_parameters.remote(global_param)
    ray.get([client.set_model_to_device.remote("cuda:0") for client in clients])
    global_model.to("cuda:0")
    # Add CUDA synchronization to ensure all GPU operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compute_loss(task_cfg, pred, true):
    '''

    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    ce_loss = nn.CrossEntropyLoss(reduction='mean')

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

def train(env_cfg, task_cfg, client_ids, cm_map, fdl, last_loss_rep, rd, epoch, clients, log_file, verbose=True):
    #legacy list
    #models
    #optimizers
    #schedulers
    device = env_cfg.device
    if len(client_ids) == 0:
        return last_loss_rep

    client_train_loss = last_loss_rep
    for id in client_ids:
        client_train_loss[id] = 0.0
    
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # sys.stdout is all the python output statement (e.g. print statements). We write these to the null device => not printing them

    for client in clients:
        client.set_train_mode.remote() # Pytorch makes sure it is in training mode

    # Begin an epoch of training
    for data in fdl.fbd_list: # Traverse the data of each client
        client = data.dataset.location
        model_id = cm_map[ray.get(client.get_id.remote())]
        if model_id not in client_ids: # neglect non-participants
            continue
        
        #dispatch to ray client for training
        local_loss,time_log = ray.get(client.ray_train.remote(data,device,task_cfg,rd,epoch))
        client_train_loss[model_id] += local_loss
        torch.cuda.synchronize()

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__

    return client_train_loss

def local_test(client_ids, task_cfg, env_cfg, cm_map, fdl, last_loss_rep, last_acc_rep, clients):
    if not client_ids:
        return last_loss_rep, last_acc_rep, None
    
    device = env_cfg.device
    client_test_loss = last_loss_rep
    client_test_acc = last_acc_rep

    for id in client_ids:
        client_test_loss[id] = 0.0
        client_test_acc[id] = 0.0

    for client in clients:
        client.set_eval_mode.remote()   # Pytorch makes sure it is in evaluation mode

    count = 0.0 # only for getting metrics, since each client only has one batch (so dont need count for accuracy)
    metrics = {'ap': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'mrr': 0.0}

    for data in fdl.fbd_list:
        client = data.dataset.location
        model_id = cm_map[ray.get(client.get_id.remote())]
        if model_id not in client_ids: # neglect non-participants
            continue

        if task_cfg.task_type == 'LP':
            edge_label = data.dataset.edge_label
            if len(edge_label) == 0 or edge_label.numel() == 0: # neglect participants with no validation data
                print("Ignore clients with no validation data")
                continue
            
        # Actual evaluation part
        local_metrics, local_loss, local_acc = ray.get(client.ray_test.remote(data,device,task_cfg))
        
        # Compute Loss and other metrics
        metrics['mrr'] += local_metrics['mrr']
        metrics['ap'] += local_metrics['ap']
        metrics['macro_f1'] += local_metrics['macro_f1']
        metrics['micro_f1'] += local_metrics['micro_f1']
        client_test_loss[model_id] += local_loss.detach().item()
        client_test_acc[model_id] += local_acc
        count += 1
        torch.cuda.synchronize()
    
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
            if len(data.dataset.edge_label) == 0 or data.dataset.edge_label.numel() == 0:
                print("Ignore participants with no testing data")
                continue

        model_id = cm_map[ray.get(data.dataset.location.get_id.remote())]
        if model_id not in client_ids:
            continue

        # Move all tensors to GPU before testing
        data.dataset.node_feature = data.dataset.node_feature.to(device)
        data.dataset.edge_feature = data.dataset.edge_feature.to(device)
        data.dataset.node_states = [state.to(device) for state in data.dataset.node_states]
        data.dataset.edge_index = data.dataset.edge_index.to(device)
        if task_cfg.task_type == 'LP':
            data.dataset.edge_label_index = data.dataset.edge_label_index.to(device)
            data.dataset.edge_label = data.dataset.edge_label.to(device)
        else:
            data.dataset.node_label = data.dataset.node_label.to(device)
            data.dataset.node_label_index = data.dataset.node_label_index.to(device)

        if task_cfg.task_type == 'LP':
            predicted_y, _, _ = global_model(copy.copy(data.dataset))
            acc, ap = lp_prediction(predicted_y, data.dataset.edge_label.type_as(predicted_y))
            print("Test Accuracy is", acc, "by client with", data.dataset.edge_index.shape[1], "edges")
            mrr = compute_mrr(predicted_y, data.dataset.edge_label.type_as(predicted_y), data.dataset.edge_label_index)
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

    # Add CUDA synchronization to ensure all GPU operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    """ Server test model using cross client edges """
    if task_cfg.task_type == 'LP':
        server_data = server.test_loader.dataset
        if server_data.edge_index.shape[1] > 1:
            # Move server data to GPU
            server_data.node_feature = server_data.node_feature.to(device)
            server_data.edge_feature = server_data.edge_feature.to(device)
            server_data.node_states = [state.to(device) for state in server_data.node_states]
            server_data.edge_index = server_data.edge_index.to(device)
            server_data.edge_label_index = server_data.edge_label_index.to(device)
            server_data.edge_label = server_data.edge_label.to(device)

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

    torch.cuda.synchronize()
    metrics = {key: value / count for key, value in metrics.items()}
    
    # return test_sum_loss, accuracy/count, metrics
    return 0, accuracy/count, metrics

def initialize_models(global_model, clients, device):
    for client in clients:
        client.set_model.remote(copy.deepcopy(global_model))
    ray.get([client.set_model_to_device.remote(device) for client in clients])
    torch.cuda.synchronize()