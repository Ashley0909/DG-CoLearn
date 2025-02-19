import os
import sys
import torch
import time
import numpy as np
from datetime import datetime

from fl_strategy import run_FL,run_FedAssets,run_dygl
from configurations import init_config, init_FL_clients, init_global_model, init_FLBackdoor_clients, init_GNN_clients
from utils import Logger
from flgnn_dataset import load_gnndata, get_gnn_clientdata
from fl_dataset import FLDataLoader, get_clientdata, load_data
from fl_clients import generate_clients_perf, generate_clients_crash_prob, generate_crash_trace
from plot_graphs import configure_plotly, time_gpu

torch.autograd.set_detect_anomaly(True)

def main():
    # Set Configuration
    crash_prob = float(sys.argv[1])
    lag_tol = int(sys.argv[2])
    pick_C = float(sys.argv[3])
    task2run = str(sys.argv[4])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    task_mode = str(sys.argv[5])

    bw_set = (0.175, 1250) # (client throughput, bandwidth_server) in MB/s

    env_cfg, task_cfg = init_config(task2run, pick_C, crash_prob, bw_set)

    env_cfg.mode = task_mode

    # Load Data
    if task_mode in ["FLDGNN-LP", "FLDGNN-NC"]:
        num_snapshots, train_list, val_list, test_list, data_size, arg = load_gnndata(task_cfg)
    else:
        train_x, train_y, test_x, test_y, data_size = load_data(task_cfg, env_cfg)
    
    # Create a list of information per snapshots in FLDGNN
    if task_mode in ["FLDGNN-LP", "FLDGNN-NC"]:
        sys.stdout = Logger('FLDGNN')
        print(f"Running {task_mode}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")

        clients, cindexmap = init_GNN_clients(env_cfg.n_clients, arg['last_embeddings'], arg['weights']) # Stay the same for all snapshots
        glob_model = init_global_model(env_cfg, task_cfg)

        # Configure Plot to plot global model performance
        x_labels = []
        test_ap = []
        for ss in range(num_snapshots):
            for rd in range(env_cfg.n_rounds):
                x_labels.append(f"Snapshot {ss} Round {rd}")
        test_ap_fig = configure_plotly(x_labels, test_ap, 'Average Tested Precision (Area under PR Curve)', "")
        # directory = "val_ap_plots/{}".format(datetime.now().strftime("%Y%m%d_%H%M"))
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        for i in range(num_snapshots-2): # only (num_snapshots - 2) training rounds because of TVT split
            print("Snapshot", i)
            fed_data_train, fed_data_val, fed_data_test, client_shard_sizes = get_gnn_clientdata(train_list[i], val_list[i], test_list[i], env_cfg, clients)
            glob_model, _, _, val_fig, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, glob_model, clients, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                       i, client_shard_sizes, data_size[i], test_ap_fig, test_ap)

            for c in clients: # Pass the curr_ne to prev_ne for training in the upcoming round
                c.update_embeddings(c.curr_ne)

            # Plot result
            # val_fig.show()
            # file_path = os.path.join(directory, "{}_SS{}.png".format(datetime.now().strftime("%Y%m%d_%H%M"), i))
            # val_fig.write_image(file_path)
            if i > 20: # Stop after snapshot 6
                # test_ap_fig.show()
                # test_ap_fig.write_image("test_ap_plots/{}.png".format(datetime.now().strftime("%Y%m%d_%H%M")))
                exit(-1)
    else:
        # Create Clients
        if task_mode == 'FedAssets':
            sys.stdout = Logger('FedAssets')
            clients, cindexmap, mali_map = init_FLBackdoor_clients(env_cfg.n_clients, env_cfg.benign_ratio)
        elif task_mode == 'Semi-Async':
            sys.stdout = Logger('Semi_Async')
            clients, cindexmap = init_FL_clients(env_cfg.n_clients)
        print(f"Running {task_mode}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")

        fed_data_train, fed_data_test, client_shard_sizes = get_clientdata(train_x, train_y, test_x, test_y, env_cfg, task_cfg, clients)
        fed_loader_train = FLDataLoader(fed_data_train, cindexmap, env_cfg.batch_size, env_cfg.shuffle)
        fed_loader_test = FLDataLoader(fed_data_test, cindexmap, env_cfg.batch_size, env_cfg.shuffle)

        glob_model = init_global_model(env_cfg, task_cfg)

        # Run FL Strategy
        if task_mode == 'FedAssets':
            _, _, _ = run_FedAssets(env_cfg, task_cfg, glob_model, cindexmap, data_size, fed_loader_train, fed_loader_test, client_shard_sizes, mali_map)
        else:
            # Prepare Simulation
            clients_perf_vec = generate_clients_perf(env_cfg, from_file=True)

            # Maximum waiting time for client response in a round setting
            clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(clients_perf_vec)
            response_time_limit = env_cfg.max_T if env_cfg.max_T else max(clients_est_round_T_train) + 2 * task_cfg.model_size / bw_set[0]

            # Generate client crash probability
            clients_crash_prob_vec = generate_clients_crash_prob(env_cfg)
            # Crash trace simulation
            crash_trace, progress_trace = generate_crash_trace(env_cfg, clients_crash_prob_vec)

            _, _, _ = run_FL(env_cfg, task_cfg, glob_model, cindexmap, data_size, fed_loader_train, fed_loader_test, client_shard_sizes, clients_perf_vec, 
                                                crash_trace, progress_trace,clients_est_round_T_train, response_time_limit, lag_tol)

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()