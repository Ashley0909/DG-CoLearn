import sys
import torch
import warnings
import time
warnings.filterwarnings("ignore")

from src.fl_strategy import run_dygl
from src.fl_server import Server
from src.configurations import init_config, init_global_model, init_GNN_clients
from src.utils.utils import Logger
from src.flgnn_dataset import load_gnndata, get_gnn_clientdata
from src.sbm_generate import generate_graph
from src.plotting.plot_graphs import configure_plotly
from graphgym.config import cfg

torch.autograd.set_detect_anomaly(True)

def main():
    """ Wall-to-wall total training time """
    pipeline_start_time = time.time()
    # Set Configuration
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    
    # Parse optional arguments
    incremental_learning = True  # default to True
    cfg_file = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--incremental_learning':
            if i + 1 < len(sys.argv):
                incremental_learning = sys.argv[i + 1].lower() in ('true', '1', 'yes')
                i += 2
            else:
                i += 1
        else:
            cfg_file = str(sys.argv[i])  # Assume it's the config file for as-733
            i += 1
    
    if cfg_file:  # Case when dataset is as-733
        cfg.merge_from_file(cfg_file) # Update the config on the graphgym side

    bw_set = (0.175, 1250) # (client throughput, bandwidth_server) in MB/s

    env_cfg, task_cfg = init_config(dataset, bw_set)
    task_cfg.incremental_learning = incremental_learning  # Add incremental learning flag to config

    # Load Data
    data_loading_start = time.time()
    if dataset == "SBM":
        num_snapshots, train_list, val_list, test_list, arg = generate_graph(task_cfg)
    else:
        num_snapshots, train_list, val_list, test_list, arg = load_gnndata(task_cfg)
    data_loading_time = time.time() - data_loading_start
    
    # Create a list of information per snapshots in FLDGNN
    sys.stdout = Logger('fast_gpa') # Log the print statements to a text file
    print(f"Data Loading Time: {data_loading_time:.2f} seconds")
    print(f"Running {task_cfg.task_type}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")
    incremental_mode_str = "Incremental Learning (Only Learn New Edges)" if incremental_learning else "Full Graph Learning"
    print(f"Mode: {incremental_mode_str}")

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

    past_test_data, best_metrics = None, None # For measuring catastrophic forgetting
    snapshot_times = []
    for i in range(num_snapshots-2): # only (num_snapshots - 2) training rounds because of TVT split
        snapshot_start_time = time.time()
        print("Snapshot", i)
        server.server_round = i
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], task_cfg, clients)       
        glob_model, best_metrics, _, test_ap_fig, test_ap, past_test_data = run_dygl(env_cfg, task_cfg, server, clients, glob_model, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                    i, client_shard_sizes, data_size, test_ap_fig, test_ap, {'data': past_test_data, 'metric': best_metrics}, arg["num_nodes"])
        snapshot_time = time.time() - snapshot_start_time
        snapshot_times.append(snapshot_time)
        print("Snapshot Ends. Best Round:", best_metrics['best_round'], "Best Metrics:", best_metrics)
        print(f"Snapshot {i} Training Time: {snapshot_time:.2f} seconds")
        print("=============")
    
    # Print total training time statistics
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Data Loading Time: {data_loading_time:.2f} seconds")
    if snapshot_times:
        print(f"Training Time per Snapshot: {snapshot_times}")
        print(f"Total Training Time (all snapshots): {sum(snapshot_times):.2f} seconds")
        print(f"Average Training Time per Snapshot: {sum(snapshot_times)/len(snapshot_times):.2f} seconds")
    print(f"Wall-to-Wall Total Pipeline Time: {total_time:.2f} seconds")
    print("="*60 + "\n")

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()