import sys
import torch
import warnings
import ray
import logging
import os
import time
from datetime import datetime
from tqdm import tqdm

from fl_strategy import run_dygl
from fl_server import Server
from configurations import init_config, init_global_model, init_GNN_clients
from utils import Logger
from flgnn_dataset import load_gnndata, get_gnn_clientdata
from sbm_generate import generate_graph
from plot_graphs import configure_plotly
from fl_clients import initialize_models

# Create Logs directory if it doesn't exist
if not os.path.exists("Logs"):
    os.makedirs("Logs")

# Log the file and prior settings
timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
#log_file = open(f"Logs/log_{timestamp}_{sys.argv[1]}.txt", 'w')
log_file = open(f"Logs/{sys.argv[1]}_round:{sys.argv[2]}_epoch:{sys.argv[3]}_Mode:{sys.argv[4]}log_{timestamp}_.txt", 'w')

if not hasattr(sys.stdout, "isatty"):
    sys.stdout.isatty = lambda: False
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*size_average and reduce args will be deprecated.*")

#initialise ray cluster 
ray.init(runtime_env={"working_dir": "/workspace/DG-CoLearn-ray"})
head_node_ip = '172.25.28.144'#hard code

def main():
    # Initialise ray node [DONE]
    t0 = time.perf_counter()
    nodes = ray.nodes()
    active_nodes = [node for node in nodes if node['Alive']] #only want the alive nodes
    head_node = None #head node
    client_nodes = [] #client node list
    for node in active_nodes:
        if node['NodeManagerAddress'] == head_node_ip: 
            head_node = node
        else:
            client_nodes.append(node)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN, RAY] Ray cluster activated time: {t1 - t0} seconds\n")
    
    print("-"*40)
    print(f" Ray cluster activated: {len(active_nodes)} alive!")
    print(f" Head node IP: {head_node['NodeManagerAddress']}")
    print(f" Number of worker: {len(client_nodes)}")
    print("-"*40)

    # Data Prep [DONE]
    print("-"*40)
    print("Data Prep")

    # Set Configuration
    t0 = time.perf_counter()
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    env_cfg, task_cfg = init_config(dataset,len(client_nodes),int(sys.argv[2]),int(sys.argv[3]))
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Config initialisation time: {t1 - t0} seconds\n")
    print("-"*40)   
     
    # Load Data
    t0 = time.perf_counter()
    if dataset == "SBM":
        num_snapshots, train_list, val_list, test_list, arg = generate_graph(task_cfg)
    else:
        num_snapshots, train_list, val_list, test_list, arg = load_gnndata(task_cfg)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Data loading time: {t1 - t0} seconds\n")

    
    # Create a list of information per snapshots in FLDGNN [DONE]
    t0 = time.perf_counter()
    sys.stdout = Logger('fl_sbm')
    print(f"Running {task_cfg.task_type}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")
    print("Only Learn New Graph")
    print("Data Prep Complete")
    print("-"*40)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Logger initialisation time: {t1 - t0} seconds\n")

    # Create server, client and cindexmap [DONE]
    t0 = time.perf_counter()
    print("-"*40)
    print("Creating Client/Server Assigning Ray Actor")
    clients, cindexmap = init_GNN_clients(last_ne=None, client_nodes=client_nodes) # Stay the same for all snapshots
    glob_model = init_global_model(env_cfg, task_cfg, arg) # Global model on local machine
    server = Server()
    print("Client/Server Creation Complete")
    print("-"*40)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN, RAY] Client/Server creation time: {t1 - t0} seconds\n")

    # Configure Plot to plot global model performance [DONE]
    t0 = time.perf_counter()
    print("-"*40)
    print("Configure Plot to plot global model performance")
    x_labels = []
    test_ap = []
    for ss in range(num_snapshots):
        for rd in range(env_cfg.n_rounds):
            x_labels.append(f"Snapshot {ss} Round {rd}")
    test_ap_fig = configure_plotly(x_labels, test_ap, 'Average Tested Precision (Area under PR Curve)', "")   
    print("Complete") 
    print("-"*40)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Plot configuration time: {t1 - t0} seconds\n")

    # Local model initialization [DONE]
    t0 = time.perf_counter()
    glob_model.to("cpu")
    initialize_models(glob_model,clients,env_cfg.device)
    torch.cuda.synchronize()
    glob_model.to(env_cfg.device)
    # Add CUDA synchronization after moving model to device
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    log_file.write(f"[MAIN, RAY] local model initialization time: {t1 - t0} seconds\n")
    print("prepared")
    # Training Loop
    print("-"*50)
    print("Training Loop Start")
    
    # Create progress bar for overall training
    pbar = tqdm(total=num_snapshots-2, desc="Overall Progress", position=0)
    acc = []
    f1 = []
    map = []
    mrr = []
    # Snapshot loop
    for i in range(num_snapshots - 2): # only (num_snapshots - 2) training rounds because of TVT split
        print("-"*60)
        print("Snapshot", i)

        # Data Split Server Side
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], task_cfg, clients, log_file, i, num_snapshots-2)

        # Distribute Learning Client Size
        glob_model, best_metrics, _, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, server, glob_model, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                    i, client_shard_sizes, data_size, test_ap_fig, test_ap, arg['num_nodes'], clients, log_file, num_snapshots-2)

        print("Snapshot Ends. Best Round:", best_metrics['best_round'], "Best Metrics:", best_metrics)
        print("=============")
        map.append(best_metrics['best_ap'])
        mrr.append(best_metrics['best_mrr'])
        acc.append(best_metrics['best_acc'])
        f1.append(best_metrics['best_f1'])
        # Update prev_ne
        t0 = time.perf_counter()
        # Add Ray synchronization using ray.get() on list of futures
        ray.get([c.update_embeddings.remote(ray.get(c.get_curr_ne.remote())) for c in clients])
        t1 = time.perf_counter()
        log_file.write(f"[MAIN, RAY] Prev_ne update time: {t1 - t0} seconds\n")
        
        # Update progress bar
        pbar.update(1)
        best_metric = best_metrics.get('best_metric', 'N/A')
        if isinstance(best_metric, (int, float)):
            best_metric = f"{best_metric:.4f}"
        pbar.set_postfix({
            'Best Round': best_metrics['best_round'],
            'Best Metrics': best_metric
        })
        print("-"*60)
    
    # Close progress bar
    pbar.close()
    print("-"*50)
    log_file.write(f"Accuracy per snap: {acc}\n")
    log_file.write(f"MAP per snap: {map}\n")
    log_file.write(f"MRR per snap: {mrr}\n")
    log_file.write(f"Avg Accuracy: {sum(acc)/len(acc)}\n")
    log_file.write(f"Avg F1: {sum(f1)/len(f1)}\n")
    log_file.write(f"Avg MAP: {sum(map)/len(map)}\n")
    log_file.write(f"Avg MRR: {sum(mrr)/len(mrr)}\n")
    log_file.flush()
    log_file.flush()
    log_file.close()

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()

