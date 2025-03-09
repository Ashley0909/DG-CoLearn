import numpy as np
import math
import copy
import random
import torch
from collections import defaultdict
import os

from utils import get_global_embedding
from fl_models import ROLANDGNN
from fl_clients import distribute_models, select_clients_FCFM, select_clients_randomly, version_filter, train, local_test, global_test
from configurations import EventHandler
from fl_aggregations import safa_aggregate, fedassets_aggregate, gnn_aggregate
from plot_graphs import configure_plotly, time_cpu

def get_cross_rounders(clients_est_round_T_train, max_round_interval):
   # Cross rounders are the clients that run over the time limit of one round. Slow clients.
   cross_rounder_ids = []
   for c_id in range(len(clients_est_round_T_train)):
      if clients_est_round_T_train[c_id] > max_round_interval:
            cross_rounder_ids.append(c_id)
   return cross_rounder_ids

def update_cloud_cache(cache, local_models, ids):
   """ Update each clients' local models in the cache """
   for id in ids:
      cache[id] = copy.deepcopy(local_models[id])

def update_cloud_cache_deprecated(cache, global_model, deprecated_ids):
   """ Update entries of those clients lagging too much behind with the latest global model """
   for id in deprecated_ids:
      cache[id] = copy.deepcopy(global_model)

def get_versions(ids, versions):
   """ Show versions of specified clients, as a dict """
   cv_map = {}
   for id in ids:
      cv_map[id] = versions[id]

   return cv_map

def update_versions(versions, ids, rd):
   for id in ids:
      versions[id] = rd

def sample_clients(data_list, cm_map):
   # Select the clients who has edges in this snapshot (even only with positive edges its fine)
   client_list = []
   for data in data_list:
      client = data.location
      if data.edge_index.shape[1] != 0:
         client_list.append(cm_map[client.id])
   return client_list

def run_FL(env_cfg, task_cfg, global_mod, cm_map, data_size, fed_data_train, fed_data_test, client_shard_sizes, clients_perf_val, crash_trace, 
           progress_trace, clients_est_round_T_train, response_time_limit, lag_t, snapshot=None, mali_map={}):
   """
    Run FL
    :param env_cfg: environment config
    :param task_cfg: task config
    :param global_mod: global model
    :param cm_map: client-model mapping
    :param data_size: total data size
    :param fed_data_train: federated training set
    :param fed_data_test: federated test set
    :param client_shard_sizes: sizes of clients' shards (train_size, test_size)
    :param clients_perf_val: batch overhead values of clients
    :param clients_crash_prob: crash probs of clients
    :param crash_trace: simulated crash trace
    :param progress_trace: simulated progress trace
    :param response_time_limit: maximum round interval
    :param lag_t: tolerance of lag
    :return:
   """
   # Initialise 
   global_model = global_mod
   local_models = [None for _ in range(env_cfg.n_clients)]
   client_ids = list(range(env_cfg.n_clients))

   distribute_models(global_model, local_models, client_ids)  # Distribute the global model to each client as local models

   cache = None
   benign_clients = []

   # Traces
   reporting_train_loss = [0.0 for _ in range(env_cfg.n_clients)]
   reporting_test_loss = [0.0 for _ in range(env_cfg.n_clients)]
   reporting_test_acc = [0.0 for _ in range(env_cfg.n_clients)]
   versions = np.array([-1 for _ in range(env_cfg.n_clients)])
   pick_trace = []
   make_trace = []
   undrafted_trace = []
   deprecated_trace = []
   round_trace = []
   acc_trace = []

   # Global event handler
   event_handler = EventHandler(['time', 'T_dist'])

   # Local counters
   client_timers = [0.01 for _ in range(env_cfg.n_clients)]
   client_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]
   if task_cfg.task_type == 'LP': # Train in one batch
      clients_est_round_T_train = env_cfg.n_epochs / np.array(clients_perf_val)
   else:
      clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(clients_perf_val)
   cross_rounders = get_cross_rounders(clients_est_round_T_train, response_time_limit)
   picked_ids = []
   client_futile_timers = [0.0 for _ in range(env_cfg.n_clients)]  # futile = useless
   eu_count = 0.0  # effective updates count
   sync_count = 0.0  # synchronization count
   version_var = 0.0

   # Best global model loss
   best_round = -1
   best_loss = float('inf')
   best_acc = -1.0
   best_model = None

   """ Begin Training """
   for rd in range(env_cfg.n_rounds):
      print("Round", rd)

      # Reset Timers
      client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]
      client_round_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]
      picked_client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]
      client_round_train_timers = [0.0 for _ in range(env_cfg.n_clients)]

      # Randomly pick a set of clients to train
      quota = math.ceil(env_cfg.n_clients * env_cfg.pick_frac)
      selected_ids = random.sample(range(env_cfg.n_clients), quota)
      selected_ids.sort()
      print("Picked Clients", selected_ids)

      # Simulate Device Crash
      crash_ids = crash_trace[rd]
      make_ids = [c_id for c_id in range(env_cfg.n_clients) if c_id not in crash_ids]
      picked_ids = select_clients_FCFM(make_ids, picked_ids, clients_perf_val, cross_rounders, quota)
      undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
      
      # Tracing
      make_trace.append(make_ids)
      pick_trace.append(picked_ids)
      undrafted_trace.append(undrafted_ids)

      # Distributing Global Model
      print("Distributing Global Model")
      good_ids, deprecated_ids = version_filter(versions, client_ids, rd - 1, lag_tolerant=lag_t)
      latest_ids, _ = version_filter(versions, good_ids, rd - 1, lag_tolerant=0)

      distribute_models(global_model, local_models, deprecated_ids) # Deprecated clients are forced to sync
      update_cloud_cache_deprecated(cache, global_model, deprecated_ids)
      deprecated_trace.append(deprecated_ids)
      update_versions(versions, deprecated_ids, rd-1)

      distribute_models(global_model, local_models, latest_ids) # Also sync up-to-version clients
      sync_count += len(deprecated_ids) + len(latest_ids)

      # Local Training
      for epoch in range(env_cfg.n_epochs):
         if rd + epoch == 0: # 1st epoch of 1st round, all-in
            temp_make_ids = copy.deepcopy(selected_ids)
            selected_ids = list(range(env_cfg.n_clients)) # Because no client is crashed yet
         elif rd == 0 and epoch == 1:
            cache = copy.deepcopy(local_models) # Since the first epoch changes everything
            selected_ids = temp_make_ids

         reporting_train_loss = train(local_models, selected_ids, env_cfg, cm_map, fed_data_train, task_cfg, reporting_train_loss, rd, epoch, verbose=True)
         reporting_test_loss, reporting_test_acc, localtest_metrics = local_test(local_models, selected_ids, task_cfg, env_cfg, cm_map, fed_data_test, reporting_test_loss, reporting_test_acc)
         test_loss_average = np.mean(np.array(reporting_test_loss)[np.array(reporting_test_loss) != 0.0])

      print("Local Epoch", epoch, "training losses", reporting_train_loss, "test metrics", localtest_metrics)

      # Aggregation Step
      update_cloud_cache(cache, local_models, picked_ids)
      if env_cfg.mode == "FedAssets":
         global_model, benign = fedassets_aggregate(rd, cache, selected_ids, client_shard_sizes, data_size, mali_map)
         benign_clients.extend(benign)
      else:
         global_model = safa_aggregate(cache, client_shard_sizes, data_size)
      update_cloud_cache(cache, local_models, undrafted_ids)

      # Versioning
      effective_updates = len(picked_ids)
      eu_count += effective_updates
      version_var += 0.0 if effective_updates == 0 else np.var(versions[selected_ids])
      update_versions(versions, selected_ids, rd)

      # Reporting Phase
      if env_cfg.mode == 'FedAssets':
         selected_ids = select_clients_randomly(benign_clients, quota)
      post_aggre_loss, post_aggre_acc, post_aggre_metrics = global_test(global_model, selected_ids, task_cfg, env_cfg, cm_map, fed_data_test, rd)
      overall_loss = np.array(post_aggre_loss).sum() / (data_size * env_cfg.test_frac)
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', post_aggre_acc)
      print('>   @Cloud> Other Metrics = ', post_aggre_metrics)

      if overall_loss < best_loss:
         best_loss = overall_loss
         best_acc = post_aggre_acc
         best_model = global_model
         best_round = rd
      
      if env_cfg.keep_best:
         global_model = best_model
         overall_loss = best_loss
         post_aggre_acc = best_acc

      round_trace.append(overall_loss)
      acc_trace.append(post_aggre_acc)

      # Update Timers
      for c_id in range(env_cfg.n_clients):
         if c_id in make_ids:
            T_comm = 2 * task_cfg.model_size / env_cfg.bw_set[0]
            if task_cfg.task_type == 'LP': # Train in one batch
               T_train = env_cfg.n_epochs / clients_perf_val[c_id]
            else:
               T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_val[c_id]
            client_round_train_timers[c_id] = T_train
            
            client_round_timers[c_id] = min(response_time_limit, T_comm + T_train)  # including comm. and training

            client_round_comm_timers[c_id] = T_comm  # comm. is part of the run time
            client_timers[c_id] += client_round_timers[c_id]  # sum up
            client_comm_timers[c_id] += client_round_comm_timers[c_id]  # sum up
            if c_id in picked_ids:
               picked_client_round_timers[c_id] = client_round_timers[c_id]  # we need to await the picked
            if c_id in deprecated_ids:  # deprecated clients, forced to sync. at distributing step
               client_futile_timers[c_id] += progress_trace[rd][c_id] * client_round_timers[c_id]
               client_round_timers[c_id] = response_time_limit  # no response
      dist_time = task_cfg.model_size * sync_count / env_cfg.bw_set[1] # Distribution time

      # Event updates
      event_handler.add_parallel('time', picked_client_round_timers, reduce='max')
      event_handler.add_sequential('time', dist_time)
      event_handler.add_sequential('T_dist', dist_time)

      print('> Round client run time:', client_round_timers)  # round estimated finish time
      print('> Round client train time:', client_round_train_timers)

   # Statistics
   global_timer = event_handler.get_state('time')
   global_dist_timer = event_handler.get_state('T_dist')

   print("Total Time Consumption:", global_timer)
   print("Total distribution time:", global_dist_timer)
   
   return best_model, best_round, best_loss

def run_FedAssets(env_cfg, task_cfg, global_mod, cm_map, data_size, fed_data_train, fed_data_test, client_shard_sizes, mali_map):
   # Initialise 
   global_model = global_mod
   local_models = [None for _ in range(env_cfg.n_clients)]
   client_ids = list(range(env_cfg.n_clients))

   cache = copy.deepcopy(local_models)
   benign_clients = []

   # Traces
   reporting_train_loss = [0.0 for _ in range(env_cfg.n_clients)]
   reporting_test_loss = [0.0 for _ in range(env_cfg.n_clients)]
   reporting_test_acc = [0.0 for _ in range(env_cfg.n_clients)]

   # Best global model loss
   best_round = -1
   best_loss = float('inf')
   best_acc = -1.0
   best_model = None

   """ Begin Training """
   for rd in range(env_cfg.n_rounds):
      print("Round", rd)
      # Randomly pick a set of clients to train
      quota = math.ceil(env_cfg.n_clients * env_cfg.pick_frac)
      train_ids = random.sample(range(env_cfg.n_clients), quota)
      train_ids.sort()
      print("Picked Clients", train_ids)

      # Distributing Global Model
      print("Distributing Global Model")
      distribute_models(global_model, local_models, client_ids) # distributing the global model to all clients

      # Local Training
      for epoch in range(env_cfg.n_epochs):
         reporting_train_loss = train(local_models, train_ids, env_cfg, cm_map, fed_data_train, task_cfg, reporting_train_loss, rd, epoch, verbose=True)
         reporting_test_loss, reporting_test_acc, _ = local_test(local_models, train_ids, task_cfg, env_cfg, cm_map, fed_data_test, reporting_test_loss, reporting_test_acc)

      # print("Local Epoch", epoch, "training losses", reporting_train_loss, "test accuracy", reporting_test_acc)
      print("Local Epoch", epoch, "test accuracy", reporting_test_acc)

      # Aggregation Step
      update_cloud_cache(cache, local_models, train_ids)
      global_model, benign = fedassets_aggregate(rd, cache, train_ids, client_shard_sizes, data_size, mali_map)
      benign_clients.extend(benign)

      # Reporting Phase
      quota = math.ceil(env_cfg.n_clients * env_cfg.pick_frac) // 2
      evaluate_ids = select_clients_randomly(benign_clients, quota)
      print("evaluate id", evaluate_ids)
      post_aggre_loss, post_aggre_acc, post_aggre_metrics = global_test(global_model, evaluate_ids, task_cfg, env_cfg, cm_map, fed_data_test, rd)
      overall_loss = np.array(post_aggre_loss).sum() / (data_size * env_cfg.test_frac)
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', post_aggre_acc)
      print('>   @Cloud> Other Metrics = ', post_aggre_metrics)

      if overall_loss < best_loss:
         best_loss = overall_loss
         best_acc = post_aggre_acc
         best_model = global_model
         best_round = rd
      
      if env_cfg.keep_best:
         global_model = best_model
         overall_loss = best_loss
         post_aggre_acc = best_acc
   
   return best_model, best_round, best_loss

def run_dygl(env_cfg, task_cfg, global_mod, clients, cm_map, fed_data_train, fed_data_val, fed_data_test, snapshot, client_shard_sizes, data_size, test_ap_fig, test_ap, ccn_dict, node_assignment):
   # Initialise
   global_model = global_mod
   # if os.path.exists('model_state/model.pth'):
   #    global_model = 
   
   local_models = [None for _ in range(env_cfg.n_clients)]
   cache = [None for _ in range(env_cfg.n_clients)] # stores the local model trained in each snapshot (only this snapshot)
   client_ids = list(range(env_cfg.n_clients))

   distribute_models(global_model, local_models, client_ids)

   # Assign all clients who has edges involved participate in all training rounds
   client_ids = sample_clients(fed_data_train, cm_map)
   print("Participating Clients:", client_ids)

   best_round = -1
   best_loss = float('inf')
   best_acc, best_ap = -1.0, -1.0
   best_model = None
   train_loss = [0.0 for _ in range(env_cfg.n_clients)]
   val_loss = [0.0 for _ in range(env_cfg.n_clients)]
   val_acc = [0.0 for _ in range(env_cfg.n_clients)]

   # Configure Plotly
   x_labels = []
   for rd in range(env_cfg.n_rounds):
      for ep in range(env_cfg.n_epochs):
         x_labels.append(f"Round {rd} Epoch {ep}")
   val_ap = []

   val_ap_fig = configure_plotly(x_labels, val_ap, 'Average Validation Precision (Area under PR Curve)', snapshot)

   """ Begin Training """
   for rd in range(env_cfg.n_rounds):
      print("Round", rd)
      best_local_models = copy.deepcopy(local_models)
      best_val_acc = [float('-inf') for _ in range(len(client_ids))]

      for epoch in range(env_cfg.n_epochs // 2):
         if epoch == 1:
            """ Share Node Embedding after first epoch """
            trained_embeddings = defaultdict(torch.Tensor)
            subnodes_union = set()
            for c in client_ids:
               # plot_h(matrix=clients[c].curr_ne[1], path='ne1_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Trained Node Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
               trained_embeddings[c] = clients[c].send_embeddings()
               subnodes_union = subnodes_union.union(clients[c].subnodes.tolist())

            print("Share Embeddings")
            shared_embeddings = get_global_embedding(trained_embeddings, ccn_dict, node_assignment.tolist(), subnodes_union, client_ids[0])
            for c in range(len(client_ids)):
               clients[c].update_embeddings(shared_embeddings)
               # plot_h(matrix=clients[c].prev_ne[1], path='newprev_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Updated Prev Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
               
         train_loss = train(local_models, client_ids, env_cfg, cm_map, fed_data_train, task_cfg, train_loss, rd, epoch, verbose=True)
         val_loss, val_acc, val_metrics = local_test(local_models, client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc)
         # Update metrics data
         val_ap.append(val_metrics['ap'])
         val_ap_fig.data[0].y = val_ap  # Update y for Val AP Fig
         # print('>   @Local> accuracy = ', val_acc)
         # print('>   @Local> Other Metrics = ', val_metrics)

      """ Share Node Embedding after training for the first half of epochs """
      trained_embeddings = defaultdict(torch.Tensor)
      subnodes_union = set()
      for c in client_ids:
         # plot_h(matrix=clients[c].curr_ne[1], path='ne1_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Trained Node Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
         trained_embeddings[c] = clients[c].send_embeddings()
         subnodes_union = subnodes_union.union(clients[c].subnodes.tolist())

      print("Share Embeddings")
      shared_embeddings = get_global_embedding(trained_embeddings, ccn_dict, node_assignment.tolist(), subnodes_union, client_ids[0])
      for c in range(len(client_ids)):
         clients[c].update_embeddings(shared_embeddings)
         # plot_h(matrix=clients[c].prev_ne[1], path='newprev_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Updated Prev Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)

      for epoch in range(env_cfg.n_epochs // 2, env_cfg.n_epochs):
         train_loss = train(local_models, client_ids, env_cfg, cm_map, fed_data_train, task_cfg, train_loss, rd, epoch, verbose=True)
         val_loss, val_acc, val_metrics = local_test(local_models, client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc)
         val_ap.append(val_metrics['ap']) # Update metric data
         val_ap_fig.data[0].y = val_ap  # Update y for Val AP Fig
         # print('>   @Local> accuracy = ', val_acc)
         # print('>   @Local> Other Metrics = ', val_metrics)
         for c in client_ids:
            if val_acc[c] > best_val_acc[c]:
               best_local_models[c] = copy.deepcopy(local_models[c])
               best_val_acc[c] = val_acc[c]

      print('>   @Local> Best accuracies = ', best_val_acc)
      # Aggregate Local Models
      update_cloud_cache(cache, best_local_models, client_ids)
      global_model = gnn_aggregate(cache, client_shard_sizes, data_size, client_ids)
      print("Aggregated Model")
      # Global Test
      global_loss, global_acc, global_metrics = global_test(global_model, client_ids, task_cfg, env_cfg, cm_map, fed_data_test, rd)
      overall_loss = np.array(global_loss)[np.array(global_loss) != 0.0].sum() / data_size
      global_ap = global_metrics['ap']
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', global_acc)
      print('>   @Cloud> Other Metrics = ', global_metrics)
      test_ap.append(global_metrics['ap']) # Update metrics data
      test_ap_fig.data[0].y = test_ap  # Update y for Test AP Fig

      # Record Best Readings
      # if overall_loss < best_loss:
      if (env_cfg.mode == "FLDGNN-NC" and global_acc > best_acc) or (env_cfg.mode == "FLDGNN-LP" and global_ap > best_ap):
         best_loss = overall_loss
         best_acc = global_acc
         best_ap = global_ap
         best_model = global_model
         best_round = rd
      
      if env_cfg.keep_best:
         global_model = best_model
         overall_loss = best_loss
         global_acc = best_acc
         global_ap = best_ap

      # Save model's state
      # torch.save(global_model.state_dict(), "model_state/model.pth")
   if env_cfg.mode == "FLDGNN-LP":
      return best_model, best_round, best_ap, val_ap_fig, test_ap_fig, test_ap
   else:
      return best_model, best_round, best_acc, val_ap_fig, test_ap_fig, test_ap