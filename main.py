import sys
import torch
import warnings

from fl_strategy import run_dygl
from fl_server import Server
from configurations import init_config, init_global_model, init_GNN_clients
from utils import Logger
from flgnn_dataset import load_gnndata, get_gnn_clientdata
from sbm_generate import generate_graph
from plot_graphs import configure_plotly

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

def main():
    # Set Configuration
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}

    bw_set = (0.175, 1250) # (client throughput, bandwidth_server) in MB/s

    env_cfg, task_cfg = init_config(dataset, bw_set)

    # Load Data
    if dataset == "SBM":
        num_snapshots, train_list, val_list, test_list, arg = generate_graph(task_cfg)
    else:
        num_snapshots, train_list, val_list, test_list, arg = load_gnndata(task_cfg)
    
    # Create a list of information per snapshots in FLDGNN
    sys.stdout = Logger('fl_sbm')
    print(f"Running {task_cfg.task_type}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")
    print("Only Learn New Graph")

    clients, cindexmap = init_GNN_clients(env_cfg.n_clients, last_ne=None) # Stay the same for all snapshots
    glob_model = init_global_model(env_cfg, task_cfg, arg)
    server = Server()

    # Configure Plot to plot global model performance
    x_labels = []
    test_ap = []
    for ss in range(num_snapshots):
        for rd in range(env_cfg.n_rounds):
            x_labels.append(f"Snapshot {ss} Round {rd}")
    test_ap_fig = configure_plotly(x_labels, test_ap, 'Average Tested Precision (Area under PR Curve)', "")

    for i in range(num_snapshots-2): # only (num_snapshots - 2) training rounds because of TVT split
        print("Snapshot", i)
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], task_cfg, clients)       
        glob_model, best_metrics, _, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, server, glob_model, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                    i, client_shard_sizes, data_size, test_ap_fig, test_ap, arg['num_nodes'])
        print("Snapshot Ends. Best Round:", best_metrics['best_round'], "Best Metrics:", best_metrics)
        print("=============")
        for c in clients: # Pass the curr_ne to prev_ne for training in the upcoming round
            c.update_embeddings(c.curr_ne)

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()