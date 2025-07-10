import sys
import torch
import warnings
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


# Create Logs directory if it doesn't exist
if not os.path.exists("Logs"):
    os.makedirs("Logs")

# Create Heatmap directory if it doesn't exist
if not os.path.exists("Logs/Heatmap"):
    os.makedirs("Logs/Heatmap")

# Check command line arguments
if len(sys.argv) < 5:
    print("Usage: python main.py <dataset> <n_rounds> <n_epochs> <mode>")
    print("Example: python main.py SBM 10 5 Heatmap")
    print("Available datasets: boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit, SBM")
    sys.exit(1)

#Log the file - will be opened in main() to ensure unique files per run
log_file = None

if not hasattr(sys.stdout, "isatty"):
    sys.stdout.isatty = lambda: False
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*size_average and reduce args will be deprecated.*")

def main():
    global log_file
    #Log the file - create unique file for this run
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    
    # Create directory structure for log file
    log_dir = f"Logs/{sys.argv[4]}/{sys.argv[1]}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = open(f"{log_dir}/{sys.argv[1]}_round:{sys.argv[2]}_epoch:{sys.argv[3]}_Mode:{sys.argv[4]}log_{timestamp}_.txt", 'w')
    
    # Set Configuration
    print("-"*40)
    print("Data Prep")
    t0 = time.perf_counter()
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    bw_set = (0.175, 1250) # (client throughput, bandwidth_server) in MB/s
    env_cfg, task_cfg = init_config(dataset, bw_set, int(sys.argv[2]), int(sys.argv[3]))
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
    
    # Create a list of information per snapshots in FLDGNN
    t0 = time.perf_counter()
    sys.stdout = Logger('fl_sbm')
    print(f"Running {task_cfg.task_type}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")
    print("Only Learn New Graph")
    print("Data Prep Complete")
    print("-"*40)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Logger initialisation time: {t1 - t0} seconds\n")

    # Create server and clients
    t0 = time.perf_counter()
    print("-"*40)
    print("Creating Client/Server")
    clients, cindexmap = init_GNN_clients(env_cfg.n_clients, last_ne=None) # Stay the same for all snapshots
    glob_model = init_global_model(env_cfg, task_cfg, arg)
    server = Server()
    print("Client/Server Creation Complete")
    print("-"*40)
    t1 = time.perf_counter()
    log_file.write(f"[MAIN] Client/Server creation time: {t1 - t0} seconds\n")

    # Configure Plot to plot global model performance
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

    # Training Loop
    print("-"*50)
    print("Training Loop Start")
    
    # Create progress bar for overall training
    pbar = tqdm(total=num_snapshots-2, desc="Overall Progress", position=0)
    acc = []
    f1 = []
    map = []
    mrr = []
    for i in range(num_snapshots-2): # only (num_snapshots - 2) training rounds because of TVT split
        print("-"*60)
        print("Snapshot", i)
        
        # Data Split Server Side
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], task_cfg, clients, log_file, i, num_snapshots-2, sys.argv[4])
        
        # Distribute Learning
        torch.cuda.synchronize()
        glob_model, best_metrics, _, test_ap_fig, test_ap = run_dygl(env_cfg, task_cfg, server, glob_model, cindexmap, fed_data_train, fed_data_val, fed_data_test, 
                                                                    i, client_shard_sizes, data_size, test_ap_fig, test_ap, arg['num_nodes'], log_file, num_snapshots-2)
        torch.cuda.synchronize()

        print("Snapshot Ends. Best Round:", best_metrics['best_round'], "Best Metrics:", best_metrics)
        print("=============")
        map.append(best_metrics['best_ap'])
        mrr.append(best_metrics['best_mrr'])
        acc.append(best_metrics['best_acc'])
        f1.append(best_metrics['best_f1'])
        
        # Update prev_ne
        t0 = time.perf_counter()
        for c in clients: # Pass the curr_ne to prev_ne for training in the upcoming round
            c.update_embeddings(c.curr_ne)    
        t1 = time.perf_counter()
        log_file.write(f"[SNAPSHOT {i}] Prev_ne update time: {t1 - t0} seconds\n")
        
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
    print(acc)
    print(f"Log file object: {log_file}")
    print(f"Log file closed: {log_file.closed}")
    try:
        log_file.write(f"Accuracy per snap: {acc}\n")
        log_file.write(f"MAP per snap: {map}\n")
        log_file.write(f"MRR per snap: {mrr}\n")
        log_file.write(f"Avg Accuracy: {sum(acc)/len(acc)}\n")
        log_file.write(f"Avg F1: {sum(f1)/len(f1)}\n")
        log_file.write(f"Avg MAP: {sum(map)/len(map)}\n")
        log_file.write(f"Avg MRR: {sum(mrr)/len(mrr)}\n")
        log_file.flush()
        print("Successfully wrote to log file")
    except Exception as e:
        print(f"Error writing to log file: {e}")
    finally:
        log_file.close()

if __name__ == "__main__":
    main()