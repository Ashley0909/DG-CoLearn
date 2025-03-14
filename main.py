import os
import sys
import torch
import time
import numpy as np
from datetime import datetime
import warnings

from fl_strategy import run_dygl
from fl_server import Server
from configurations import init_config, init_global_model, init_GNN_clients
from utils import Logger
from flgnn_dataset import load_gnndata, get_gnn_clientdata
from plot_graphs import configure_plotly, time_gpu

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

def main():
    # Set Configuration
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    task_mode = str(sys.argv[2])

    bw_set = (0.175, 1250) # (client throughput, bandwidth_server) in MB/s

    env_cfg, task_cfg = init_config(dataset, bw_set)

    env_cfg.mode = task_mode

    # Load Data
    num_snapshots, train_list, val_list, test_list, arg = load_gnndata(task_cfg)
    
    # Create a list of information per snapshots in FLDGNN
    # sys.stdout = Logger('fl_nc')
    print(f"Running {task_mode}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")

    clients, cindexmap = init_GNN_clients(env_cfg.n_clients, arg['last_embeddings']) # Stay the same for all snapshots
    glob_model = init_global_model(env_cfg, task_cfg)
    server = Server()

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
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], env_cfg, clients)
        # glob_model, best_round, best_metric, val_fig, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, glob_model, clients, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
        #                                                            i, client_shard_sizes, data_size, test_ap_fig, test_ap, ccn_dict, node_assignment)
        global_snapshot_acc, global_snapshot_mrr, glob_model, best_round, best_metric, _, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, server, glob_model, clients, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                    i, client_shard_sizes, data_size, test_ap_fig, test_ap)
        print("Snapshot Ends. Best Round:", best_round, "Best AP:", best_metric, "Best Accuracy:", global_snapshot_acc, "Best MRR:", global_snapshot_mrr)
        print("=============")
        for c in clients: # Pass the curr_ne to prev_ne for training in the upcoming round
            c.update_embeddings(c.curr_ne)

        # file_path = os.path.join(directory, "{}_SS{}.png".format(datetime.now().strftime("%Y%m%d_%H%M"), i))
        # val_fig.write_image(file_path)
        # test_ap_fig.write_image("test_ap_plots/{}.png".format(datetime.now().strftime("%Y%m%d_%H%M")))

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()