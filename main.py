import sys
import os
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
import logging
logging.basicConfig(level=logging.INFO)

torch.autograd.set_detect_anomaly(True)


def _configure_torch_threads_from_env():
    """Thread tuning for reproducible CPU performance."""
    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
    torch_interop_threads = int(os.getenv("TORCH_INTEROP_THREADS", "1"))

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_interop_threads)


def main():
    """ Wall-to-wall total training time """

    logging.info("Starting pipeline")
    pipeline_start_time = time.time()
    _configure_torch_threads_from_env()
    # Set Configuration
    dataset = str(sys.argv[1])  # string: options={boston, mnist, cifar10, cifar100, bitcoinOTC, DBLP, Reddit}
    
    # Parse optional arguments
    incremental_learning = True  # default to True
    mode = 'snapshot'  # 'snapshot' or 'ctdg'
    patch_size = 100000  # edges per patch in CTDG mode
    fl_strategy = 'dgcolearn'  # 'dgcolearn' or 'feddgl'
    cfg_file = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--incremental_learning':
            if i + 1 < len(sys.argv):
                incremental_learning = sys.argv[i + 1].lower() in ('true', '1', 'yes')
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--mode':
            if i + 1 < len(sys.argv):
                mode = sys.argv[i + 1].lower()
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--patch_size':
            if i + 1 < len(sys.argv):
                patch_size = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--fl_strategy':
            if i + 1 < len(sys.argv):
                fl_strategy = sys.argv[i + 1].lower()
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
    task_cfg.mode = mode
    task_cfg.patch_size = patch_size
    task_cfg.fl_strategy = fl_strategy

    # Load Data
    data_loading_start = time.time()
    if mode == 'ctdg':
        from src.ctdg_dataset import load_ctdg_data
        num_snapshots, train_list, val_list, test_list, arg = load_ctdg_data(task_cfg, patch_size)
    elif dataset == "SBM":
        num_snapshots, train_list, val_list, test_list, arg = generate_graph(task_cfg)
    else:
        num_snapshots, train_list, val_list, test_list, arg = load_gnndata(task_cfg)
    data_loading_time = time.time() - data_loading_start
    
    # Create a list of information per snapshots in FLDGNN
    # sys.stdout = Logger('fast_gpa') # Log the logging.info statements to a text file
    logging.info(f"Torch CPU threads: intra_op={torch.get_num_threads()}, inter_op={torch.get_num_interop_threads()}")
    logging.info(f"Data Loading Time: {data_loading_time:.2f} seconds")
    logging.info(f"Running {task_cfg.task_type}: n_client={env_cfg.n_clients}, n_epochs={env_cfg.n_epochs}, dataset={task_cfg.dataset}")
    incremental_mode_str = "Incremental Learning (Only Learn New Edges)" if incremental_learning else "Full Graph Learning"
    logging.info(f"Mode: {incremental_mode_str}")
    if mode == 'ctdg':
        graph_str = "cumulative" if not incremental_learning else "incremental"
        logging.info(f"CTDG Mode ({graph_str}): patch_size={patch_size}, {num_snapshots} patches")

    clients, cindexmap = init_GNN_clients(env_cfg.n_clients, last_ne=None) # Stay the same for all snapshots
    glob_model = init_global_model(env_cfg, task_cfg, arg)
    server = Server()

    # For CTDG mode: seed the server with the precomputed partition so that
    # per-patch graph_partition calls take the fast reuse_partition path
    if mode == 'ctdg' and 'precomputed_partition' in arg:
        precomputed = arg['precomputed_partition']
        precomputed_ns = arg['precomputed_num_subgraphs']
        precomputed_ei = arg['precomputed_edge_index']
        server.record_prev_num_subgraphs(precomputed_ns)
        # Use empty previous edges so get_exclusive_subgraph treats all
        # patch edges as "new" (avoids O(n*m) comparison against 5M+ edges)
        empty_ei = torch.zeros((2, 0), dtype=torch.long)
        for tvt in ('train', 'val', 'test'):
            server.record_node_assignment(precomputed.clone(), tvt)
            server.record_prev_edges(empty_ei, tvt)
        server.record_reuse_bool(True)
        logging.info(f"CTDG: seeded server with precomputed partition ({precomputed_ns} clients)")

    # Configure Plot to plot global model performance
    x_labels = []
    test_ap = []
    for ss in range(num_snapshots):
        for rd in range(env_cfg.n_rounds):
            x_labels.append(f"Snapshot {ss} Round {rd}")
    test_ap_fig = configure_plotly(x_labels, test_ap, 'Average Tested Precision (Area under PR Curve)', "")

    # Initialize FedDGL state if using FedDGL strategy
    feddgl_state = None
    if fl_strategy == 'feddgl':
        from src.feddgl import FedDGLState
        proto_dim = task_cfg.in_dim  # matches EvolveGCN hidden dim (cfg.gnn.dim_inner set in init_global_model)
        feddgl_state = FedDGLState(
            num_classes=task_cfg.num_classes, proto_dim=proto_dim,
            gamma=1.0, q=0.5, topk=50
        )
        logging.info(f"FedDGL: initialized state with {task_cfg.num_classes} classes, proto_dim={proto_dim}")

    past_test_data, best_metrics = None, None # For measuring catastrophic forgetting
    snapshot_times = []
    for i in range(num_snapshots-2): # only (num_snapshots - 2) training rounds because of TVT split
        snapshot_start_time = time.time()
        logging.info("Snapshot %d", i)
        server.server_round = i
        fed_data_train, fed_data_val, fed_data_test, client_shard_sizes, data_size = get_gnn_clientdata(server, train_list[i], val_list[i], test_list[i], task_cfg, clients)
        if fl_strategy == 'feddgl':
            from src.feddgl import run_feddgl
            glob_model, best_metrics, _, test_ap_fig, test_ap, past_test_data = run_feddgl(
                env_cfg, task_cfg, server, clients, glob_model, cindexmap,
                fed_data_train, fed_data_val, fed_data_test,
                i, client_shard_sizes, data_size, test_ap_fig, test_ap,
                {'data': past_test_data, 'metric': best_metrics}, arg["num_nodes"],
                feddgl_state=feddgl_state)
        else:
            glob_model, best_metrics, _, test_ap_fig, test_ap, past_test_data = run_dygl(env_cfg, task_cfg, server, clients, glob_model, cindexmap, fed_data_train, fed_data_val, fed_data_test,
                                                                        i, client_shard_sizes, data_size, test_ap_fig, test_ap, {'data': past_test_data, 'metric': best_metrics}, arg["num_nodes"])
        snapshot_time = time.time() - snapshot_start_time
        snapshot_times.append(snapshot_time)
        logging.info(f"Snapshot Ends. Best Round: {best_metrics.get('best_round', 'N/A')}, Best Metrics: {best_metrics}")
        logging.info(f"Snapshot {i} Training Time: {snapshot_time:.2f} seconds")
        logging.info("=============")
    
    # logging.info total training time statistics
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    logging.info("\n" + "="*60)
    logging.info("PIPELINE SUMMARY")
    logging.info("="*60)
    logging.info(f"Data Loading Time: {data_loading_time:.2f} seconds")
    if snapshot_times:
        logging.info(f"Training Time per Snapshot: {snapshot_times}")
        logging.info(f"Total Training Time (all snapshots): {sum(snapshot_times):.2f} seconds")
        logging.info(f"Average Training Time per Snapshot: {sum(snapshot_times)/len(snapshot_times):.2f} seconds")
    logging.info(f"Wall-to-Wall Total Pipeline Time: {total_time:.2f} seconds")
    logging.info("="*60 + "\n")

if __name__ == '__main__':  # If the file is run directly (python3 main.py), __name__ will be set to __main__ and will run the function main()
    main()