import numpy as np
import copy
import torch
import time
from collections import defaultdict
import torch.optim as optim

from fl_clients import distribute_models, train, local_test, global_test, catastrophic_forgetting_test
from fl_aggregations import gnn_aggregate
from plot_graphs import configure_plotly
# from utils import get_global_embedding
# from fl_models import MLPEncoder
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
      if data.dataset.edge_index.shape[1] > 1:
         client_list.append(cm_map[client.id])
   return client_list

def run_dygl(env_cfg, task_cfg, server, clients, global_mod, cm_map, fed_data_train, fed_data_val, fed_data_test, snapshot, client_shard_sizes, data_size, test_ap_fig, test_ap, past_test_dict, tot_num_nodes):
   # Initialise
   global_model = global_mod   
   local_models = [None for _ in range(env_cfg.n_clients)]
   cache = [None for _ in range(env_cfg.n_clients)] # stores the local model trained in each snapshot (only this snapshot)
   client_ids = list(range(env_cfg.n_clients))

   distribute_models(global_model, local_models, client_ids)

   # Assign all clients who has edges involved participate in all training rounds
   client_ids = sample_clients(fed_data_train, cm_map)
   print("Participating Clients:", client_ids)

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

   # One optimizer for each model (re-instantiate optimizers to clear any possible momentum)
   optimizers = {}
   for i in client_ids:
      if task_cfg.optimizer == 'SGD':
         optimizers[i] = optim.SGD(local_models[i].parameters(), lr=task_cfg.lr)
      elif task_cfg.optimizer == 'Adam':
         optimizers[i] = optim.Adam(local_models[i].parameters(), lr=task_cfg.lr, weight_decay=5e-4, betas=(0.9, 0.999))
      else:
         print('Err> Invalid optimizer %s specified' % task_cfg.optimizer)

   schedulers = {}
   for i in client_ids:
      # schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[30,60,90]))
      schedulers[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max=env_cfg.n_epochs, eta_min=1e-5)
      # schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i],mode='max',factor=0.5,patience=5,verbose=True))

   ''' Pretraining Communication (Model Answer for Node Embeddings) '''
   # print("Pretraining Communication Starts")
   # in_dim = fed_data_train[0].dataset.node_feature.shape[1]
   # encoder = MLPEncoder(in_dim=in_dim, out_dim=16)
   # for c in fed_data_train:
   #    client = c.dataset.location 
   #    feature = client.upload_features(c.dataset.node_feature, tot_num_nodes, encoder)
   #    server.client_features.append(feature) # Server collects the clients' features
   # print("Clients finished uploading embeddings, server computing global embeddings...")
   # start_time = time.time()
   # global_states = server.get_global_node_states() # Server computes global features
   # end_time = time.time()
   # print(f"Time taken for Global Node State Computation: {end_time - start_time}")
   # for c in fed_data_train:
   #    c.dataset.node_states = copy.deepcopy([s.detach() for s in global_states]) # Clients collects the global features

   ''' Simulating FedGCN Pretain Communication '''
   # start_fedgcn = time.time()
   # features, subnodes = [], []
   # # client compute features
   # for c in fed_data_train:
   #    data = c.dataset
   #    one_hop_feat, two_hop_feat = compute_neighborhood_features(data.edge_index, data.node_feature, tot_num_nodes)
   #    features.append(two_hop_feat)
   #    subnodes.append(data.subnodes)
   # # clients send them to server and server computes average
   # start_comp = time.time()
   # final_feat = average_feat_aggre(features)
   # end_comp = time.time()
   # # server redistribute to clients
   # for i, c in enumerate(fed_data_train):
   #    c.dataset.node_feature = final_feat
   # end_fedgcn = time.time()
   # print(f"Time taken for Full Node Embedding Exchange: {end_fedgcn - start_fedgcn}")
   # print(f"Time taken for Server to compute NE: {end_comp - start_comp}")   
   # print(f"Time taken for Communication: {(end_fedgcn - start_fedgcn) - (end_comp - start_comp)}")  

   best_metrics = defaultdict()
   """ Begin Training """
   for rd in range(env_cfg.n_rounds):
      print("Round", rd)
      best_local_models = copy.deepcopy(local_models)
      best_val_acc = [float('-inf') for _ in range(env_cfg.n_clients)]

      for epoch in range(env_cfg.n_epochs):
         train_loss = train(env_cfg, task_cfg, local_models, optimizers, schedulers, client_ids, cm_map, fed_data_train, train_loss, rd, epoch, verbose=True)
         val_loss, val_acc, val_metrics = local_test(local_models, client_ids, task_cfg, env_cfg, cm_map, fed_data_val, val_loss, val_acc)
         # Update metrics data
         val_ap.append(val_metrics['ap'])
         val_ap_fig.data[0].y = val_ap  # Update node_label for Val AP Fig
         print('>   @Local> accuracy = ', val_acc) # Keep! for local client performance reference

         for c in client_ids:
            if val_acc[c] > best_val_acc[c]:
               best_local_models[c] = copy.deepcopy(local_models[c])
               best_val_acc[c] = val_acc[c]

         # Node Embedding Exchange Scheme
         if epoch == (env_cfg.n_epochs - 1) and rd == 0:
            node_embeds = []
            ccn = server.ccn
            ne_start_time = time.time()
            for c in client_ids:
               node_embeds.append(clients[c].send_ccn_embeddings(ccn))
            compute_start = time.time()
            messages = server.aggregate_and_send(node_embeds)
            compute_end = time.time()
            for c in client_ids:
               clients[c].receive_from_server(messages)
            ne_end_time = time.time()
            print(f"Time taken for Full Node Embedding Exchange: {ne_end_time - ne_start_time}")   
            print(f"Time taken for Server to compute NE: {compute_end - compute_start}")   
            print(f"Time taken for Communication: {(ne_end_time - ne_start_time) - (compute_end - compute_start)}")   

      print('>   @Local> Val Metrics = ', val_metrics) # Keep! for local client performance reference
      # Aggregate Local Models
      update_cloud_cache(cache, best_local_models, client_ids)
      global_model = gnn_aggregate(cache, client_shard_sizes, data_size, client_ids)
      print("Aggregated Model")
      global_loss, global_acc, global_metrics = global_test(global_model, server, client_ids, task_cfg, env_cfg, cm_map, fed_data_test)
      # Measure catastrophic forgetting
      if past_test_dict['data'] is not None:
         catstro_dict = catastrophic_forgetting_test(global_model, client_ids, task_cfg, env_cfg, cm_map, past_test_dict)
         print('>   @Cloud> Forgetting = ', catstro_dict)
      overall_loss = np.array(global_loss)[np.array(global_loss) != 0.0].sum() / data_size
      global_f1 = global_metrics['micro_f1']
      global_ap = global_metrics['ap']
      print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
      print('>   @Cloud> accuracy = ', global_acc)
      print('>   @Cloud> Other Metrics = ', global_metrics)
      test_ap.append(global_metrics['ap']) # Update metrics data
      test_ap_fig.data[0].y = test_ap  # Update node_label for Test AP Fig

      # Record Best Readings      
      if (task_cfg.task_type == "NC" and global_f1 > best_f1) or (task_cfg.task_type == "LP" and global_acc > best_acc):
         best_model = global_model
         best_metrics['best_loss'] = overall_loss
         best_metrics['best_acc'], best_acc = global_acc, global_acc
         best_metrics['best_ap'], best_ap = global_ap, global_ap
         best_metrics['best_f1'], best_f1 = global_f1, global_f1
         best_metrics['best_round'] = rd
      
      if env_cfg.keep_best:
         global_model = best_model
         overall_loss = best_loss
         global_acc = best_acc
         global_ap = best_ap
         global_f1 = best_f1

   # Save Model State and Optimizer State (if needed)
   # checkpoint = {
   #    'snapshot': snapshot,
   #    'learning_rate': task_cfg.lr,
   #    'model_state_dict': global_model.state_dict()
   # }
   # torch.save(checkpoint, f'model_state/{task_cfg.dataset}/model_checkpoint_ss{snapshot}_lr{task_cfg.lr}.pth')

   return best_model, best_metrics, val_ap_fig, test_ap_fig, test_ap, fed_data_test
