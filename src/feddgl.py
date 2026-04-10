"""
FedDGL: Federated Dynamic Graph Learning for Temporal Evolution and Data Heterogeneity
(Xie et al., ACML 2024)

Implements three components:
1. Temporal Evolution Capture via Global Knowledge Distillation (Eq 11)
2. Global Prototype-Based Regularization (Eq 12-15)
3. Prototype Similarity-Based Personalized Aggregation (Eq 18-19)
"""
import copy
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from src.fl_clients import distribute_models, local_test, global_test, catastrophic_forgetting_test, compute_loss
from src.fl_aggregations import gnn_weighted_aggregate
from src.fl_strategy import sample_clients, update_cloud_cache
from src.plotting.plot_graphs import configure_plotly
from src.utils.utils import nc_prediction

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 1a. Server-side Global Prototype Network  (Eq 13)
# ---------------------------------------------------------------------------
class GlobalPrototypeNet(nn.Module):
    """Neural network H that refines local prototypes into global prototypes."""

    def __init__(self, proto_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU(),
            nn.Linear(proto_dim, proto_dim),
        )
        self.num_classes = num_classes

    def forward(self, local_protos):
        # local_protos: [num_classes, proto_dim]
        return self.net(local_protos)


# ---------------------------------------------------------------------------
# 1b. FedDGL State (persists across snapshots)
# ---------------------------------------------------------------------------
class FedDGLState:
    def __init__(self, num_classes, proto_dim, gamma=1.0, q=0.5, topk=50, proto_lr=1e-3):
        self.prev_global_model = None        # w_g^(t-1)
        self.global_prototypes = None        # P̄ [num_classes, proto_dim]
        self.proto_net = GlobalPrototypeNet(proto_dim, num_classes)
        self.proto_optimizer = optim.Adam(self.proto_net.parameters(), lr=proto_lr)
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.gamma = gamma
        self.q = q
        self.topk = topk

    def update_prev_global(self, model):
        self.prev_global_model = copy.deepcopy(model)
        self.prev_global_model.eval()


# ---------------------------------------------------------------------------
# 1c. Sensitive Node Selection (Eq 4-10, simplified)
# ---------------------------------------------------------------------------
def select_sensitive_nodes(prev_model, data, num_nodes, topk=50, q=0.5):
    """Select nodes most sensitive to temporal evolution.

    Simplified version: evaluate per-node loss under the previous model on the
    current graph.  Nodes with loss below q * mean_loss are candidates; return
    top-k by loss magnitude (the most "forgotten" ones).

    If prev_model is None (first snapshot), returns all node indices.
    """
    if prev_model is None:
        return torch.arange(num_nodes)

    prev_model.eval()
    with torch.no_grad():
        pred, true, _, _ = prev_model(copy.deepcopy(data))

    if true.dim() == 2:
        # soft labels — per-node CE
        log_probs = F.log_softmax(pred, dim=-1)
        per_node_loss = -(true * log_probs).sum(dim=-1)
    else:
        per_node_loss = F.cross_entropy(pred, true, reduction='none')

    mean_loss = per_node_loss.mean()
    # Candidate: nodes with loss < q * mean (well-learned before but potentially
    # disrupted by evolution)
    candidate_mask = per_node_loss < (q * mean_loss)
    if candidate_mask.sum() == 0:
        candidate_mask = torch.ones(per_node_loss.shape[0], dtype=torch.bool)

    candidate_losses = per_node_loss.clone()
    candidate_losses[~candidate_mask] = -float('inf')

    # Select top-k by highest loss among candidates (most sensitive)
    k = min(topk, int(candidate_mask.sum().item()))
    if k == 0:
        return torch.arange(num_nodes)
    _, indices = torch.topk(candidate_losses, k)
    return indices


# ---------------------------------------------------------------------------
# 1d. Knowledge Distillation Loss (Eq 11)
# ---------------------------------------------------------------------------
def compute_kd_loss(h_local, prev_global_model, data, sensitive_idx, gamma):
    """L_KD = gamma * sum ||z_local - z_global||_2 over sensitive nodes.

    h_local: current model's intermediate features [num_nodes, dim]
    prev_global_model: previous round's global model (detached forward pass)
    sensitive_idx: indices of sensitive nodes
    """
    if prev_global_model is None or len(sensitive_idx) == 0:
        return torch.tensor(0.0, requires_grad=True)

    prev_global_model.eval()
    with torch.no_grad():
        _, _, _, h_global = prev_global_model(copy.deepcopy(data))

    z_local = h_local[sensitive_idx]
    z_global = h_global[sensitive_idx].detach()

    kd = torch.norm(z_local - z_global, p=2, dim=-1).mean()
    return gamma * kd


# ---------------------------------------------------------------------------
# 1e. Local Prototype Computation (Eq 12)
# ---------------------------------------------------------------------------
def compute_local_prototypes(h_0, node_labels, num_classes):
    """Compute per-class mean embeddings as local prototypes.

    h_0: [num_nodes, feature_dim] intermediate features
    node_labels: [num_nodes] (1D) or [num_nodes, num_classes] (2D soft)
    Returns: [num_classes, feature_dim] prototype tensor
    """
    feature_dim = h_0.shape[1]
    protos = torch.zeros(num_classes, feature_dim, device=h_0.device)

    if node_labels.dim() == 2:
        labels = node_labels.argmax(dim=1)
    else:
        labels = node_labels

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            protos[c] = h_0[mask].mean(dim=0)

    return protos


# ---------------------------------------------------------------------------
# 1f. Train Global Prototypes via Contrastive Learning (Eq 13-14)
# ---------------------------------------------------------------------------
def train_global_prototypes(state, local_protos_list):
    """Server trains global prototypes using contrastive loss.

    local_protos_list: list of [num_classes, proto_dim] tensors from each client
    Updates state.global_prototypes in-place.
    """
    if not local_protos_list:
        return

    # Average local prototypes as input to prototype net
    stacked = torch.stack(local_protos_list)  # [K, C, D]
    avg_local = stacked.mean(dim=0)  # [C, D]

    state.proto_net.train()
    state.proto_optimizer.zero_grad()

    global_protos = state.proto_net(avg_local)  # [C, D]

    # Contrastive loss (Eq 14): for each client k and class c,
    # positive = e^{-cos(P_{k,c}, P̄_c)}, negative = e^{-cos(P_{k,c}, P̄_{c'})}
    loss = torch.tensor(0.0)
    num_clients = len(local_protos_list)
    num_classes = global_protos.shape[0]

    for k in range(num_clients):
        for c in range(num_classes):
            p_kc = local_protos_list[k][c]
            if p_kc.norm() < 1e-8:
                continue  # skip empty prototypes

            # cos similarities to all global prototypes
            cos_sims = F.cosine_similarity(
                p_kc.unsqueeze(0), global_protos, dim=1
            )  # [C]
            # d(k,c,c',t) = e^{-cos}
            d_vals = torch.exp(-cos_sims)
            positive = d_vals[c]
            denominator = d_vals.sum()
            if denominator > 0:
                loss = loss - torch.log(positive / denominator + 1e-8)

    if num_clients > 0 and loss.requires_grad:
        loss = loss / (num_clients * max(num_classes, 1))
        loss.backward()
        state.proto_optimizer.step()

    state.global_prototypes = global_protos.detach().clone()


# ---------------------------------------------------------------------------
# 1g. Prototype Regularization Loss (Eq 15)
# ---------------------------------------------------------------------------
def compute_prototype_reg_loss(local_protos, global_protos):
    """L_P = (1/|C_k|) * sum cos(P_{k,c}, P̄_c) for classes present in client.

    We want to MAXIMIZE cosine similarity, so loss = -L_P (minimize negative cos).
    """
    if global_protos is None:
        return torch.tensor(0.0, requires_grad=True)

    # Only consider classes where local prototype is non-zero
    norms = local_protos.norm(dim=1)
    active = norms > 1e-8
    if active.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    cos_sims = F.cosine_similarity(
        local_protos[active], global_protos[active], dim=1
    )
    # Minimize negative cosine similarity
    return -cos_sims.mean()


# ---------------------------------------------------------------------------
# 1h. Personalized Aggregation (Eq 18-19)
# ---------------------------------------------------------------------------
def feddgl_personalized_aggregate(models, local_protos_list, global_protos, client_ids):
    """Prototype-similarity-based personalized aggregation.

    Returns: personalized models (one per client), plus the FedAvg global model.
    """
    if not client_ids or global_protos is None:
        return models, models[client_ids[0]] if client_ids else None

    # Compute cosine similarity between each client's prototype and global
    cos_scores = {}
    flat_global = global_protos.flatten()
    for i, cid in enumerate(client_ids):
        if i < len(local_protos_list) and local_protos_list[i] is not None:
            flat_local = local_protos_list[i].flatten()
            cos_scores[cid] = F.cosine_similarity(
                flat_local.unsqueeze(0), flat_global.unsqueeze(0)
            ).item()
        else:
            cos_scores[cid] = 0.0

    # Normalize to get lambda_k (Eq 19)
    total_cos = sum(max(v, 0.0) for v in cos_scores.values()) + 1e-8
    lambdas = {cid: max(cos_scores[cid], 0.0) / total_cos for cid in client_ids}

    # Compute FedAvg model
    avg_model = copy.deepcopy(models[client_ids[0]])
    avg_sd = avg_model.state_dict()
    for name, param in avg_sd.items():
        if torch.is_floating_point(param) and 'running_' not in name:
            avg_sd[name] = torch.zeros_like(param)

    w = 1.0 / len(client_ids)
    for cid in client_ids:
        sd = models[cid].state_dict()
        for name, param in sd.items():
            if torch.is_floating_point(param) and 'running_' not in name:
                avg_sd[name] += param.detach() * w

    ref_sd = models[client_ids[0]].state_dict()
    for name, param in ref_sd.items():
        if not torch.is_floating_point(param) or 'running_' in name:
            avg_sd[name] = param.clone()
    avg_model.load_state_dict(avg_sd)

    # Personalized blend: w_k = lambda_k * w_k + (1-lambda_k) * w_avg
    for cid in client_ids:
        lam = lambdas[cid]
        local_sd = models[cid].state_dict()
        blended_sd = copy.deepcopy(avg_sd)
        for name, param in blended_sd.items():
            if torch.is_floating_point(param) and 'running_' not in name:
                blended_sd[name] = lam * local_sd[name] + (1.0 - lam) * avg_sd[name]
        models[cid].load_state_dict(blended_sd)

    return models, avg_model


# ---------------------------------------------------------------------------
# 1i. FedDGL Training Loop (drop-in replacement for run_dygl)
# ---------------------------------------------------------------------------
def run_feddgl(env_cfg, task_cfg, server, clients, global_mod, cm_map,
               fed_data_train, fed_data_val, fed_data_test,
               snapshot, client_shard_sizes, data_size,
               test_ap_fig, test_ap, past_test_dict, tot_num_nodes,
               feddgl_state=None):
    """FedDGL training loop — same interface as run_dygl plus feddgl_state."""

    global_model = global_mod
    local_models = [None for _ in range(env_cfg.n_clients)]
    cache = [None for _ in range(env_cfg.n_clients)]
    client_ids = list(range(env_cfg.n_clients))

    distribute_models(global_model, local_models, client_ids)

    client_ids = sample_clients(fed_data_train, cm_map)
    logging.info(f"[FedDGL] Participating Clients: {client_ids}")

    best_loss = float('inf')
    best_acc, best_f1 = -1.0, -1.0
    best_model = None
    train_loss = [0.0 for _ in range(env_cfg.n_clients)]
    val_loss = [0.0 for _ in range(env_cfg.n_clients)]
    val_acc = [0.0 for _ in range(env_cfg.n_clients)]
    val_metrics = {}

    x_labels = []
    for rd in range(env_cfg.n_rounds):
        for ep in range(env_cfg.n_epochs):
            x_labels.append(f"Round {rd} Epoch {ep}")
    val_ap = []
    val_ap_fig = configure_plotly(x_labels, val_ap, 'FedDGL Val AP', snapshot)

    # Optimizers
    optimizers = {}
    for i in client_ids:
        optimizers[i] = optim.Adam(
            local_models[i].parameters(), lr=task_cfg.lr, weight_decay=5e-4
        )
    schedulers = {}
    for i in client_ids:
        schedulers[i] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[i], T_max=env_cfg.n_epochs * 3, eta_min=1e-5
        )

    best_metrics = defaultdict()

    for rd in range(env_cfg.n_rounds):
        logging.info("[FedDGL] Round %d", rd)

        if not client_ids:
            logging.info("[FedDGL] No participating clients, skipping round")
            val_ap.append(0.0)
            continue

        best_local_models = copy.deepcopy(local_models)
        best_val_acc = [float('-inf') for _ in range(env_cfg.n_clients)]
        round_local_protos = [None] * env_cfg.n_clients

        # ----- Per-client local training -----
        for data in fed_data_train.fbd_list:
            client = data.dataset.location
            model_id = cm_map[client.id]
            if model_id not in client_ids:
                continue

            edge_index = data.dataset.edge_index
            if len(edge_index[0]) == 0:
                continue

            model = local_models[model_id]
            optimizer = optimizers[model_id]
            scheduler = schedulers[model_id]

            # Select sensitive nodes (Eq 4-10)
            sensitive_idx = select_sensitive_nodes(
                feddgl_state.prev_global_model, data.dataset,
                tot_num_nodes, topk=feddgl_state.topk, q=feddgl_state.q
            )

            for epoch in range(env_cfg.n_epochs):
                model.train()
                optimizer.zero_grad()

                # Import previous state as node_states
                client_obj = clients[model_id]
                if client_obj.prev_ne is not None:
                    for i_ns in range(len(data.dataset.node_states)):
                        data.dataset.node_states[i_ns] = client_obj.prev_ne[i_ns]

                predicted_y, true, client_obj.curr_ne, h_0 = model(
                    copy.deepcopy(data.dataset)
                )
                client_obj.h0 = h_0.detach().clone()

                # L_CE
                loss_ce, _ = compute_loss(task_cfg, predicted_y, true)

                # L_KD (Eq 11)
                loss_kd = compute_kd_loss(
                    h_0, feddgl_state.prev_global_model,
                    data.dataset, sensitive_idx, feddgl_state.gamma
                )

                # L_P (Eq 15) — prototype regularization
                # h_0 is [num_nodes, dim] but true is [num_subnodes] (client scope)
                subnodes = data.dataset.subnodes
                local_protos = compute_local_prototypes(
                    h_0[subnodes].detach(), true, feddgl_state.num_classes
                )
                loss_proto = compute_prototype_reg_loss(
                    local_protos, feddgl_state.global_prototypes
                )

                # Total loss (Eq 16)
                loss = loss_ce + loss_kd + loss_proto

                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # After local training: compute final prototypes for aggregation
            model.eval()
            with torch.no_grad():
                _, true_final, _, h_0_final = model(copy.deepcopy(data.dataset))
            round_local_protos[model_id] = compute_local_prototypes(
                h_0_final[data.dataset.subnodes], true_final, feddgl_state.num_classes
            )

        # ----- Validation -----
        val_loss, val_acc, val_metrics = local_test(
            local_models, client_ids, task_cfg, env_cfg, cm_map,
            fed_data_val, val_loss, val_acc, val_metrics
        )
        val_ap.append(val_metrics.get('ap', 0.0))
        logging.info(f'[FedDGL]   @Local Val Metrics = {val_metrics}')

        # Track best local models
        for c in client_ids:
            if val_acc[c] > best_val_acc[c]:
                best_local_models[c] = copy.deepcopy(local_models[c])
                best_val_acc[c] = val_acc[c]

        # ----- Server: Update Global Prototypes (Eq 13-14) -----
        active_protos = [
            round_local_protos[cid] for cid in client_ids
            if round_local_protos[cid] is not None
        ]
        train_global_prototypes(feddgl_state, active_protos)

        # ----- Personalized Aggregation (Eq 18-19) -----
        update_cloud_cache(cache, best_local_models, client_ids)
        personalized, global_model = feddgl_personalized_aggregate(
            cache, active_protos, feddgl_state.global_prototypes, client_ids
        )
        # Write personalized models back only for participating clients
        for cid in client_ids:
            if personalized[cid] is not None:
                local_models[cid] = personalized[cid]

        if global_model is None:
            global_model = global_mod

        logging.info("[FedDGL] Aggregated (personalized)")

        # ----- Global Test -----
        global_loss, global_acc, global_metrics = global_test(
            global_model, server, client_ids, task_cfg, env_cfg, cm_map,
            fed_data_test
        )

        if past_test_dict['data'] is not None:
            catstro_dict = catastrophic_forgetting_test(
                global_model, client_ids, task_cfg, env_cfg, cm_map, past_test_dict
            )
            logging.info(f'[FedDGL]   @Cloud Forgetting = {catstro_dict}')

        nonzero_loss = np.array(global_loss)[np.array(global_loss) != 0.0]
        overall_loss = nonzero_loss.sum() / data_size if nonzero_loss.size > 0 else 0.0
        global_f1 = global_metrics.get('micro_f1', 0.0)
        global_ap = global_metrics.get('ap', 0.0)
        logging.info(f'[FedDGL]   @Cloud accuracy = {global_acc}')
        logging.info(f'[FedDGL]   @Cloud Metrics = {global_metrics}')
        test_ap.append(global_ap)
        test_ap_fig.data[0].y = test_ap

        if (task_cfg.task_type == "NC" and global_f1 > best_f1) or \
           (task_cfg.task_type == "LP" and global_acc > best_acc):
            best_model = global_model
            best_metrics['best_loss'] = overall_loss
            best_metrics['best_acc'], best_acc = global_acc, global_acc
            best_metrics['best_ap'] = global_ap
            best_metrics['best_f1'], best_f1 = global_f1, global_f1
            best_metrics['best_round'] = rd

        if env_cfg.keep_best and best_model is not None:
            global_model = best_model

    # Update FedDGL state for next snapshot
    feddgl_state.update_prev_global(global_model)

    # Save checkpoint
    checkpoint = {
        'snapshot': snapshot,
        'learning_rate': task_cfg.lr,
        'model_state_dict': global_model.state_dict(),
    }
    save_path = Path(f'model_state/{task_cfg.dataset}')
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, f"{save_path}/feddgl_checkpoint_ss{snapshot}.pth")

    # Always return a valid model (fall back to global_model if best was never set)
    if best_model is None:
        best_model = global_model

    return best_model, best_metrics, val_ap_fig, test_ap_fig, test_ap, fed_data_test
