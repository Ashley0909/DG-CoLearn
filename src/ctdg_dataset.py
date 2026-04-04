"""
CTDG (Continuous-Time Dynamic Graph) data pipeline.

Converts raw TemporalData (edge events with timestamps) into fixed-size
patch graphs compatible with the existing snapshot-based FL pipeline.
Each patch groups a fixed number of consecutive edge events into a single
PyG Data object, which the recurrent GNN processes as a "snapshot".
"""

import copy
import logging
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, coalesce
from src.flgnn_dataset import load_tgbl, gen_train_clients
import partition

logging.basicConfig(level=logging.INFO)


def build_patches(temporal_data, num_nodes, in_dim, patch_size=100000):
    """
    Group consecutive edge events into fixed-size patch graphs,
    with RandomLinkSplit already applied for LP compatibility.

    Args:
        temporal_data: TGB TemporalData with fields src, dst, t, msg.
        num_nodes: Total number of nodes in the full graph.
        in_dim: Node feature dimension.
        patch_size: Number of edge events per patch.

    Returns:
        List of PyG Data objects (post-RandomLinkSplit), one per patch.
    """
    src = temporal_data.src
    dst = temporal_data.dst
    t = temporal_data.t
    msg = getattr(temporal_data, 'msg', None)

    num_events = src.shape[0]
    patches = []

    # Shared node feature tensor (all patches reference the same data)
    shared_node_feature = torch.ones(num_nodes, in_dim)

    transform = RandomLinkSplit(
        num_val=0.0, num_test=0.0, add_negative_train_samples=False
    )

    for start in range(0, num_events, patch_size):
        end = min(start + patch_size, num_events)

        patch_src = src[start:end]
        patch_dst = dst[start:end]

        # Build edge_index (directed first, then make undirected)
        edge_index = torch.stack([patch_src, patch_dst], dim=0).long()
        edge_index = to_undirected(edge_index)
        edge_index = coalesce(edge_index)

        # Edge features: use msg if available, else default 128-dim ones
        num_edges = edge_index.shape[1]
        if msg is not None:
            edge_feature = torch.ones(num_edges, msg.shape[1])
            orig_count = end - start
            if orig_count <= num_edges:
                edge_feature[:orig_count] = msg[start:end].float()
        else:
            edge_feature = torch.ones(num_edges, 128)

        g = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_feature=shared_node_feature,
            edge_feature=edge_feature,
            t_start=t[start].item(),
            t_end=t[end - 1].item(),
        )

        # Apply RandomLinkSplit in-place (generates edge_label_index, edge_label)
        split_data, _, _ = transform(g)
        patches.append(split_data)

    logging.info(f"Built {len(patches)} patches of size {patch_size} from {num_events} events")
    return patches


def build_cumulative_patches(temporal_data, num_nodes, in_dim, edge_dim, patch_size=100000):
    """
    Build cumulative patch graphs where each patch contains ALL edges
    from the beginning up to the current time window.

    Patch i = edges[0 : (i+1)*patch_size], deduplicated and undirected.
    This gives full structural context at each time step.

    To manage memory, edge features use a scalar 1.0 per edge (edge_dim=1)
    instead of full 128-dim vectors, since cumulative graphs can reach 16M+ edges.
    """
    src = temporal_data.src
    dst = temporal_data.dst
    t = temporal_data.t

    num_events = src.shape[0]
    patches = []

    shared_node_feature = torch.ones(num_nodes, in_dim)
    transform = RandomLinkSplit(
        num_val=0.0, num_test=0.0, add_negative_train_samples=False
    )

    for start in range(0, num_events, patch_size):
        # Cumulative: always start from 0
        end = min(start + patch_size, num_events)

        cum_src = src[:end]
        cum_dst = dst[:end]

        edge_index = torch.stack([cum_src, cum_dst], dim=0).long()
        edge_index = to_undirected(edge_index)
        edge_index = coalesce(edge_index)

        num_edges = edge_index.shape[1]
        edge_feature = torch.ones(num_edges, edge_dim)

        g = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_feature=shared_node_feature,
            edge_feature=edge_feature,
            t_start=t[0].item(),
            t_end=t[end - 1].item(),
        )

        split_data, _, _ = transform(g)
        patches.append(split_data)

    logging.info(f"Built {len(patches)} cumulative patches (patch_size={patch_size}) from {num_events} events, "
                 f"last patch has {patches[-1].edge_index.shape[1]} edges")
    return patches


def load_ctdg_data(task_cfg, patch_size=100000):
    """
    Load TGBL dataset and convert to patch-based format.

    TGB already provides chronological train/val/test splits, so we use
    those directly (no sliding-window TVT). Each split's patches become
    its own list, aligned by index for the main training loop.

    Returns the same tuple as load_gnndata():
        (num_snapshots, train_list, val_list, test_list, arg)
    """
    data, train_data, val_data, test_data = load_tgbl(task_cfg.path, task_cfg.dataset.lower())

    num_nodes = data.num_nodes
    logging.info(f"CTDG mode: {num_nodes} nodes, patch_size={patch_size}")

    # Set config fields before building patches (needed for node_feature dim)
    task_cfg.num_classes = 2
    task_cfg.in_dim = 32
    task_cfg.out_dim = 1

    msg = getattr(train_data, 'msg', None)
    if msg is not None:
        task_cfg.edge_dim = msg.shape[1]
    else:
        task_cfg.edge_dim = 128

    from graphgym.config import cfg
    cfg.dataset.edge_dim = task_cfg.edge_dim

    # Build patches from each TGB split independently
    incremental = getattr(task_cfg, 'incremental_learning', True)
    if not incremental:
        logging.info("CTDG cumulative mode: each patch includes all prior edges")
        build_fn = lambda td: build_cumulative_patches(td, num_nodes, task_cfg.in_dim, task_cfg.edge_dim, patch_size)
    else:
        build_fn = lambda td: build_patches(td, num_nodes, task_cfg.in_dim, patch_size)

    train_list = build_fn(train_data)
    val_list = build_fn(val_data)
    test_list = build_fn(test_data)

    # Align lists: main.py iterates for i in range(num_snapshots - 2)
    # and indexes train_list[i], val_list[i], test_list[i].
    # Use the shortest split length to determine iteration count.
    # Pad val/test by cycling if they have fewer patches than train.
    num_snapshots = len(train_list)
    if len(val_list) < num_snapshots:
        val_list = [val_list[i % len(val_list)] for i in range(num_snapshots)]
    if len(test_list) < num_snapshots:
        test_list = [test_list[i % len(test_list)] for i in range(num_snapshots)]

    # num_snapshots needs +2 because main.py loops range(num_snapshots - 2)
    num_snapshots = len(train_list) + 2

    # Build arg dict (same format as load_gnndata)
    hidden_conv1, hidden_conv2 = 128, 128
    last_embeddings = [
        torch.zeros(num_nodes, hidden_conv1),
        torch.zeros(num_nodes, hidden_conv1),
        torch.zeros(num_nodes, hidden_conv2),
    ]
    arg = {'last_embeddings': last_embeddings, 'num_nodes': num_nodes}

    # Pre-compute partition from ALL training edges so the per-patch
    # graph_partition calls can reuse it (avoids 638K-node partitioning each patch)
    all_train_edges = torch.cat([p.edge_index for p in train_list], dim=1)
    all_train_edges = coalesce(to_undirected(all_train_edges))
    total_train_edge_count = all_train_edges.shape[1]

    num_subgraphs = gen_train_clients(total_train_edge_count, 10)  # max 10 clients
    logging.info(f"CTDG: pre-computing partition with {num_subgraphs} clients on {total_train_edge_count} total training edges")

    # Build adjacency list for the combined training graph
    adjacency_list = [set() for _ in range(num_nodes)]
    for src_n, dst_n in all_train_edges.t().tolist():
        adjacency_list[src_n].add(dst_n)
        adjacency_list[dst_n].add(src_n)
    adjacency_list = [list(neigh) for neigh in adjacency_list]

    labels = partition.CoLearnPartition(
        copy.deepcopy(adjacency_list), total_train_edge_count,
        node_labels=[], K=num_subgraphs
    )
    precomputed_partition = torch.tensor(labels)
    arg['precomputed_partition'] = precomputed_partition
    arg['precomputed_num_subgraphs'] = num_subgraphs
    arg['precomputed_edge_index'] = all_train_edges

    logging.info(f"CTDG: {len(train_list)} training patches, "
                 f"{len(val_list)} val patches, {len(test_list)} test patches")
    return num_snapshots, train_list, val_list, test_list, arg
