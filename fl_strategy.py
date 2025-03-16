import numpy as np
import copy
import torch
import torch.profiler
import time
from collections import defaultdict
import torch.optim as optim

from utils import get_global_embedding
from fl_clients import distribute_models, train, local_test, global_test
from fl_aggregations import gnn_aggregate
from plot_graphs import configure_plotly

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
         client_list.append(cm_map[client.id])
   return client_list

def run_dygl(env_cfg, task_cfg, server, global_mod, clients, cm_map, fed_data_train, fed_data_val, fed_data_test, snapshot, client_shard_sizes, data_size, test_ap_fig, test_ap):
   # Initialise
   global_model = global_mod   
   local_models = [None for _ in range(env_cfg.n_clients)]
   cache = [None for _ in range(env_cfg.n_clients)] # stores the local model trained in each snapshot (only this snapshot)
   client_ids = list(range(env_cfg.n_clients))

   distribute_models(global_model, local_models, client_ids)

   # Assign all clients who has edges involved participate in all training rounds
   client_ids = sample_clients(fed_data_train, cm_map)
   print("Participating Clients:", client_ids)

   best_round = -1
   best_loss = float('inf')
   best_acc, best_ap, best_f1 = -1.0, -1.0, -1.0
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

   # One optimizer for each model (re-instantiate optimizers to clear any possible momentum
   optimizers = []
   for i in client_ids:
      if task_cfg.optimizer == 'SGD':
         optimizers.append(optim.SGD(local_models[i].parameters(), lr=task_cfg.lr))
      elif task_cfg.optimizer == 'Adam':
         optimizers.append(optim.Adam(local_models[i].parameters(), lr=task_cfg.lr, weight_decay=5e-4, betas=(0.9, 0.999)))
      else:
         print('Err> Invalid optimizer %s specified' % task_cfg.optimizer)

   schedulers = []
   for i in client_ids:
      # schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[30,60,90]))
      schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max=env_cfg.n_epochs, eta_min=1e-5))
      # schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i],mode='max',factor=0.5,patience=5,verbose=True))

   """ Begin Training """
   for rd in range(env_cfg.n_rounds):
      print("Round", rd)
      best_local_models = copy.deepcopy(local_models)
      best_val_acc = [float('-inf') for _ in range(len(client_ids))]

      for epoch in range(env_cfg.n_epochs // 2):
         # if epoch == 1:
         #    """ Share Node Embedding after first epoch """
         #    trained_embeddings = defaultdict(torch.Tensor)
         #    subnodes_union = set()
         #    for c in client_ids:
         #       # plot_h(matrix=clients[c].curr_ne[1], path='ne1_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Trained Node Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
         #       trained_embeddings[c] = clients[c].send_embeddings()
         #       subnodes_union = subnodes_union.union(clients[c].subnodes.tolist())

         #    shared_embeddings = server.fast_get_global_embedding_gpu(trained_embeddings, subnodes_union)

         #    for c in range(len(client_ids)):
         #       clients[c].update_embeddings(shared_embeddings)
         #       # plot_h(matrix=clients[c].prev_ne[1], path='newprev_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Updated Prev Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
               
         train_loss = train(env_cfg, task_cfg, local_models, optimizers, schedulers, client_ids, cm_map, fed_data_train, train_loss, rd, epoch, verbose=True)
         val_loss, val_acc, val_metrics = local_test(local_models, client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc)
         # Update metrics data
         val_ap.append(val_metrics['ap'])
         val_ap_fig.data[0].y = val_ap  # Update node_label for Val AP Fig
         print('>   @Local> accuracy = ', val_acc)
         # print('>   @Local> Other Metrics = ', val_metrics)

      # """ Share Node Embedding after training for the first half of epochs """
      # trained_embeddings = defaultdict(torch.Tensor)
      # subnodes_union = set()
      # for c in client_ids:
      #    # plot_h(matrix=clients[c].curr_ne[1], path='ne1_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Trained Node Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)
      #    trained_embeddings[c] = clients[c].send_embeddings()
      #    subnodes_union = subnodes_union.union(clients[c].subnodes.tolist())

      # print("Share Embeddings")
      # shared_embeddings = server.fast_get_global_embedding_gpu(trained_embeddings, subnodes_union)
      # for c in range(len(client_ids)):
      #    clients[c].update_embeddings(shared_embeddings)
         # plot_h(matrix=clients[c].prev_ne[1], path='newprev_client'+str(c)+'ep'+str(epoch)+'rd', name=f'Updated Prev Embeddings of Client {c}', round=rd, vmin=-0.5, vmax=0.3)

      for epoch in range(env_cfg.n_epochs // 2, env_cfg.n_epochs):
         train_loss = train(env_cfg, task_cfg, local_models, optimizers, schedulers, client_ids, cm_map, fed_data_train, train_loss, rd, epoch, verbose=True)
         val_loss, val_acc, val_metrics = local_test(local_models, client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc)
         val_ap.append(val_metrics['ap']) # Update metric data
         val_ap_fig.data[0].y = val_ap  # Update node_label for Val AP Fig
         # print('>   @Local> accuracy = ', val_acc)
         # print('>   @Local> Other Metrics = ', val_metrics)
         for c in client_ids:
            if val_acc[c] > best_val_acc[c]:
               best_local_models[c] = copy.deepcopy(local_models[c])
               best_val_acc[c] = val_acc[c]

      # print('>   @Local> Best accuracies = ', best_val_acc)
      print('>   @Local> Val F1 = ', val_metrics)
      # Aggregate Local Models
      update_cloud_cache(cache, best_local_models, client_ids)
      global_model = gnn_aggregate(cache, client_shard_sizes, data_size, client_ids)
      print("Aggregated Model")
      global_loss, global_acc, global_metrics = global_test(global_model, server, client_ids, task_cfg, env_cfg, cm_map, fed_data_test)
      overall_loss = np.array(global_loss)[np.array(global_loss) != 0.0].sum() / data_size
      global_f1 = global_metrics['macro_f1']
      global_ap = global_metrics['ap']
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', global_acc)
      print('>   @Cloud> Other Metrics = ', global_metrics)
      test_ap.append(global_metrics['ap']) # Update metrics data
      test_ap_fig.data[0].y = test_ap  # Update node_label for Test AP Fig

      # Record Best Readings
      # if overall_loss < best_loss:
      if (env_cfg.mode == "FLDGNN-NC" and global_f1 > best_f1) or (env_cfg.mode == "FLDGNN-LP" and global_ap > best_ap):
         best_loss = overall_loss
         best_acc = global_acc
         best_ap = global_ap
         best_f1 = global_f1
         best_model = global_model
         best_round = rd
      
      if env_cfg.keep_best:
         global_model = best_model
         overall_loss = best_loss
         global_acc = best_acc
         global_ap = best_ap
         global_f1 = best_f1

   # Save Model State and Optimizer State
   checkpoint = {
      'snapshot': snapshot,
      'learning_rate': task_cfg.lr,
      'model_state_dict': global_model.state_dict()
   }
   torch.save(checkpoint, f'model_state/{task_cfg.dataset}/model_checkpoint_ss{snapshot}_lr{task_cfg.lr}.pth')

   if env_cfg.mode == "FLDGNN-LP":
      return global_acc, global_metrics['mrr'], best_model, best_round, best_ap, val_ap_fig, test_ap_fig, test_ap
   elif env_cfg.mode == "FLDGNN-NC":
      return global_acc, global_metrics['macro_f1'], best_model, best_round, best_f1, val_ap_fig, test_ap_fig, test_ap