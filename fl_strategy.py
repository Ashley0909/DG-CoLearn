import numpy as np
import copy
import torch
import time
import ray
from collections import defaultdict
import torch.optim as optim

from utils import get_global_embedding
from fl_clients import distribute_models, train, local_test, global_test
from fl_aggregations import gnn_aggregate
from plot_graphs import configure_plotly
from fl_models import MLPEncoder
from sim_fedgcn import compute_neighborhood_features, average_feat_aggre # For simulating FedGCN 

def update_cloud_cache(cache, local_models, ids):
   """ Update each clients' local models in the cache """
   for id in ids:
      cache[id] = copy.deepcopy(local_models[id])

def sample_clients(data_list, cm_map):
   # Select the clients who has edges in this snapshot (even only with positive edges its fine)
   client_list = []
   for data in data_list:
      client = data.dataset.location
      if data.dataset.edge_index.shape[1] != 0:
         id = ray.get(client.get_id.remote())
         client_list.append(cm_map[id])
   return client_list

def run_dygl(env_cfg, task_cfg, server, global_mod, cm_map, fed_data_train, fed_data_val, fed_data_test, snapshot, client_shard_sizes, data_size, test_ap_fig, test_ap, tot_num_nodes, clients, log_file, max_i):
   # Initialise [DONE]
   t0 = time.perf_counter()
   global_model = global_mod
   cache = [None for _ in range(env_cfg.n_clients)] # stores the local model trained in each snapshot (only this snapshot)
   client_ids = list(range(env_cfg.n_clients))
   best_loss = float('inf')
   best_acc, best_ap, best_f1 = -1.0, -1.0, -1.0
   best_model = None
   train_loss = [0.0 for _ in range(env_cfg.n_clients)]
   val_loss = [0.0 for _ in range(env_cfg.n_clients)]
   val_acc = [0.0 for _ in range(env_cfg.n_clients)]
   t1 = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}] Time taken for dygl Initialisation: {t1 - t0}\n")

   #Distribute models
   t0 = time.perf_counter()
   distribute_models(global_model, clients) #Now distribute Global model to the local clients
   # Add CUDA synchronization to ensure all GPU operations are complete
   if torch.cuda.is_available():
       torch.cuda.synchronize()
   t1 = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, RAY] Time taken for Distributing Models: {t1 - t0}\n")
  
   # Assign all clients who has edges involved participate in all training rounds [DONE]
   t0 = time.perf_counter()
   client_ids = sample_clients(fed_data_train, cm_map)
   print("Participating Clients:", client_ids)
   t1 = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, RAY] Time taken for Sampling Clients: {t1 - t0}\n")

   # Configure Plotly [DONE]
   t0 = time.perf_counter()
   x_labels = []
   for rd in range(env_cfg.n_rounds):
      for ep in range(env_cfg.n_epochs):
         x_labels.append(f"Round {rd} Epoch {ep}")
   val_ap = []
   val_ap_fig = configure_plotly(x_labels, val_ap, 'Average Validation Precision (Area under PR Curve)', snapshot)
   t1 = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}] Time taken for Plotly Configuration: {t1 - t0}\n")

   # One optimizer for each model (re-instantiate optimizers to clear any possible momentum) [DONE]
   start_time_optimizers = time.perf_counter()   
   for client in clients:
      if task_cfg.optimizer == 'SGD':
         ray.get(client.set_optimizer.remote(type = "SGD",lr = task_cfg.lr, weight_decay=None , betas=None))
      elif task_cfg.optimizer == 'Adam':
         ray.get(client.set_optimizer.remote(type = "Adam",lr = task_cfg.lr, weight_decay=5e-4, betas=(0.9, 0.999)))
      else:
         print('Err> Invalid optimizer %s specified' % task_cfg.optimizer)
   end_time_optimizers = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, RAY] Time taken for Initializing Optimizers: {end_time_optimizers - start_time_optimizers}\n")

   # Set scheduler for each client [DONE]
   t0 = time.perf_counter()
   for client in clients:
      # schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[30,60,90]))
      ray.get(client.set_scheduler.remote(T_max=env_cfg.n_epochs, eta_min=1e-5))
      # schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i],mode='max',factor=0.5,patience=5,verbose=True))
   t1 = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, RAY] Time taken for Initializing Optimizers: {end_time_optimizers - start_time_optimizers}\n")

   # Pretraining Communication [DONE]
   start_time_pretrain = time.perf_counter()
   in_dim = fed_data_train[0].dataset.node_feature.shape[1]
   encoder = MLPEncoder(in_dim=in_dim, out_dim=16)
   for c in fed_data_train:
      client = c.dataset.location 
      feature = ray.get(client.upload_features.remote(c.dataset.node_feature, tot_num_nodes, encoder))
      server.client_features.append(feature) # Server collects the clients' features

   # Global feature Computation [DONE]
   start_time = time.perf_counter()
   global_states = server.get_global_node_states() # Server computes global features
   torch.cuda.synchronize()
   end_time = time.perf_counter()
   print(f"Time taken for Global Node State Computation: {end_time - start_time}")
   for c in fed_data_train: 
      c.dataset.node_states = copy.deepcopy([s.detach() for s in global_states]) # global features temporary stored in client's dataset
   end_time_pretrain = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, RAY] Time taken for Pretraining Communication: {end_time_pretrain - start_time_pretrain}\n")

   best_metrics = defaultdict()
   for rd in range(env_cfg.n_rounds):

      # Copy Global Model Parameters to cache and set metrics template [DONE]
      t0 = time.perf_counter()
      print("Round", rd)
      global_model.to("cpu")
      best_local_models = [copy.deepcopy(global_model.state_dict()) for _ in range(env_cfg.n_clients)] #global models param pre-training
      global_model.to("cuda:0")
      torch.cuda.synchronize()
      best_val_acc = [float('-inf') for _ in range(env_cfg.n_clients)]
      t1 = time.perf_counter()
      log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}] Compare metrics initialization time: {t1 - t0}\n")

      for epoch in range(env_cfg.n_epochs):
         # Train Local Model
         t0 = time.perf_counter()
         train_loss = train(env_cfg, task_cfg, client_ids, cm_map, fed_data_train, train_loss, rd, epoch, clients, log_file, verbose=True)
         t1 = time.perf_counter()
         log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}, Epoch {epoch}/{env_cfg.n_epochs}, RAY] Train time: {t1 - t0}\n")

         # Update metrics data
         t0 = time.perf_counter()
         val_loss, val_acc, val_metrics = local_test(client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc, clients)
         val_ap.append(val_metrics['ap'])
         val_ap_fig.data[0].y = val_ap  # Update node_label for Val AP Fig
         print('>   @Local> accuracy = ', val_acc) # Keep! for local client performance reference
         # Grab all the local client model
         for c in client_ids:
            if val_acc[c] > best_val_acc[c]:
               best_local_models[c] = copy.deepcopy(ray.get(clients[c].get_model_parameters.remote()))
               ray.get(clients[c].set_model_to_device.remote("cuda:0"))
               best_val_acc[c] = val_acc[c]
         t1 = time.perf_counter()
         log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}, Epoch {epoch}/{env_cfg.n_epochs}, RAY] Local test and update metrics time: {t1 - t0}\n")

      # print('>   @Local> Val Metrics = ', val_metrics) # Keep! for local client performance reference
      # Aggregate Local Models
      start_time_aggregate = time.perf_counter()
      update_cloud_cache(cache, best_local_models, client_ids)
      global_model = gnn_aggregate(cache, client_shard_sizes, data_size, client_ids, global_model)
      torch.cuda.synchronize()
      end_time_aggregate = time.perf_counter()    
      log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}] Aggregation time: {end_time_aggregate - start_time_aggregate}\n")
      print("Aggregated Model")

      #Global Test
      start_time_test = time.perf_counter()
      global_loss, global_acc, global_metrics = global_test(global_model, server, client_ids, task_cfg, env_cfg, cm_map, fed_data_test)
      end_time_test = time.perf_counter()
      log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}] Global Test time: {end_time_test - start_time_test}\n")

      # Update metrics
      t0 = time.perf_counter()
      overall_loss = np.array(global_loss)[np.array(global_loss) != 0.0].sum() / data_size
      global_f1 = global_metrics['micro_f1']
      global_ap = global_metrics['ap']
      global_mrr = global_metrics['mrr']
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', global_acc)
      print('>   @Cloud> Other Metrics = ', global_metrics)
      test_ap.append(global_metrics['ap']) # Update metrics data
      test_ap_fig.data[0].y = test_ap  # Update node_label for Test AP Fig
      t1 = time.perf_counter()
      log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}, Epoch {epoch}/{env_cfg.n_epochs}] Update metrics time: {t1 - t0}\n")

      # Record Best Readings      
      if (task_cfg.task_type == "NC" and global_f1 > best_f1) or (task_cfg.task_type == "LP" and global_acc > best_acc):
         t0 = time.perf_counter()
         best_model = global_model
         best_model = global_model
         best_metrics['best_loss'] = overall_loss
         best_metrics['best_acc'], best_acc = global_acc, global_acc
         best_metrics['best_ap'], best_ap = global_ap, global_ap
         best_metrics['best_f1'], best_f1 = global_f1, global_f1
         best_metrics['best_mrr'], best_mrr = global_mrr, global_mrr
         best_metrics['best_round'] = rd
         t1 = time.perf_counter()
         log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}, NC] Update best metrics time: {t1 - t0}\n")
      
      #update best model
      if env_cfg.keep_best:
         t0 = time.perf_counter()
         global_model = best_model
         overall_loss = best_loss
         global_acc = best_acc
         global_ap = best_ap
         global_mrr = best_mrr
         global_f1 = best_f1
         t1 = time.perf_counter()
         log_file.write(f"[SNAPSHOT {snapshot}/{max_i}, Round {rd}/{env_cfg.n_rounds}] Update best model time: {t1 - t0}\n")

   # Move model to CPU for saving
   device = next(global_model.parameters()).device
   global_model.to('cpu')
   
   start_time_save = time.perf_counter()
   checkpoint = {
      'snapshot': snapshot,
      'learning_rate': task_cfg.lr,
      'model_state_dict': global_model.state_dict()
   }
   torch.save(checkpoint, f'model_state/{task_cfg.dataset}/model_checkpoint_ss{snapshot}_lr{task_cfg.lr}.pth')
   end_time_save = time.perf_counter()
   log_file.write(f"[SNAPSHOT {snapshot}/{max_i}] Model save time: {end_time_save - start_time_save}\n")

   # Move model back to GPU
   global_model.to(device)
   best_model.to(device)
   torch.cuda.synchronize()

   return best_model, best_metrics, val_ap_fig, test_ap_fig, test_ap