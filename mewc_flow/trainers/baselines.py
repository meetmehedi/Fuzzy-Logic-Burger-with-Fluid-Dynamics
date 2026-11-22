"""
baselines.py
============
Complete, Matched-Budget Continual Learning Baselines for Q1 Journal Evaluation.

All methods share the EXACT same backbone capacity, optimizer (Adam),
CosineAnnealingLR (eta_min=0.01*lr), gradient clipping (max_norm=1.0), epochs,
batch size, and deterministic random seeds:

1. Fine-Tuning (Lower Bound)
2. Joint Training (Upper Bound — all regimes trained simultaneously)
3. Static EWC (Kirkpatrick et al., 2017)
4. Online EWC / OFU (Schwarz et al., 2018)
5. Experience Replay (ER — Rolnick et al., 2018, matched episodic buffer)
6. Replay-TS (Hemati et al., 2026 — temporal-slicing compact multi-step window
   replay with autoregressive 2-step supervision, matched total snapshot budget!)
7. SLE-FNO (Elhadidy et al., 2026 — Single-Layer Extension: frozen backbone after
   Task 0 + lightweight per-task extension adapter + prototype/Re routing)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, List, Optional

from mewc_flow.trainers.base_trainer import evaluate_all_tasks_row, train_one_task
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import EpisodicMemory
from mewc_flow.utils.metrics import summarise_results


def run_finetuning(
    model: nn.Module,
    tasks: List[Dict],
    heldout_tasks: Optional[List[Dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    model = model.to(device)
    t0 = time.time()
    r_init_mse, r_init_rel = evaluate_all_tasks_row(model, tasks, device=device)

    R_mse, R_rel = [], []
    for k, task in enumerate(tasks):
        train_one_task(model, task, epochs=epochs, lr=lr, device=device, verbose=verbose)
        m_row, r_row = evaluate_all_tasks_row(model, tasks, device=device)
        R_mse.append(m_row)
        R_rel.append(r_row)

    heldout_mse, heldout_rel = {}, {}
    if heldout_tasks:
        for ht in heldout_tasks:
            m_row, r_row = evaluate_all_tasks_row(model, [ht], device=device)
            heldout_mse[ht["label"]] = m_row[0]
            heldout_rel[ht["label"]] = r_row[0]

    s_mse = summarise_results(R_mse, r_init=r_init_mse)
    s_rel = summarise_results(R_rel, r_init=r_init_rel)
    mem = model.memory_breakdown_bytes() if hasattr(model, "memory_breakdown_bytes") else {}
    return {
        "eval_matrix": R_mse,
        "eval_matrix_rel_l2": R_rel,
        "r_init_mse": r_init_mse,
        "avg_forgetting": s_mse["avg_forgetting"],
        "avg_forgetting_rel_l2": s_rel["avg_forgetting"],
        "bwt": s_mse["bwt"],
        "fwt": s_mse["fwt"],
        "avg_final_mse": s_mse["avg_final_mse"],
        "avg_final_rel_l2": s_rel["avg_final_mse"],
        "task0_degradation_pct": s_mse["task0_degradation_pct"],
        "heldout_mse": heldout_mse,
        "heldout_rel_l2": heldout_rel,
        "wall_clock_sec": time.time() - t0,
        "memory_bytes": {**mem, "consolidation_bytes": 0, "replay_buffer_bytes": 0, "extra_cl_bytes": 0, "extra_cl_mb": 0.0},
    }


def run_joint_training(
    model: nn.Module,
    tasks: List[Dict],
    heldout_tasks: Optional[List[Dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """Multi-task oracle upper bound trained jointly on all T regimes simultaneously."""
    model = model.to(device)
    t0 = time.time()
    all_train_ds = ConcatDataset([t["train_ds"] for t in tasks])
    g = torch.Generator()
    g.manual_seed(seed)
    joint_loader = DataLoader(all_train_ds, batch_size=batch_size, shuffle=True, generator=g)

    optimizer = Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for _ in range(epochs):
        model.train()
        for u_in, re, u_out in joint_loader:
            u_in, re, u_out = u_in.to(device), re.to(device), u_out.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(u_in, re), u_out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        sched.step()

    final_mse, final_rel = evaluate_all_tasks_row(model, tasks, device=device)
    # Build pseudo eval matrix with identical final row (forgetting = 0 by definition)
    T = len(tasks)
    R_mse = [final_mse for _ in range(T)]
    R_rel = [final_rel for _ in range(T)]

    heldout_mse, heldout_rel = {}, {}
    if heldout_tasks:
        for ht in heldout_tasks:
            m_row, r_row = evaluate_all_tasks_row(model, [ht], device=device)
            heldout_mse[ht["label"]] = m_row[0]
            heldout_rel[ht["label"]] = r_row[0]

    full_data_bytes = sum(
        (t["train_ds"].u_in.numel() + t["train_ds"].u_out.numel()) * 4 for t in tasks
    )
    mem = model.memory_breakdown_bytes() if hasattr(model, "memory_breakdown_bytes") else {}
    return {
        "eval_matrix": R_mse,
        "eval_matrix_rel_l2": R_rel,
        "avg_forgetting": 0.0,
        "avg_forgetting_rel_l2": 0.0,
        "bwt": 0.0,
        "fwt": 0.0,
        "avg_final_mse": float(np.mean(final_mse)),
        "avg_final_rel_l2": float(np.mean(final_rel)),
        "task0_degradation_pct": 0.0,
        "heldout_mse": heldout_mse,
        "heldout_rel_l2": heldout_rel,
        "wall_clock_sec": time.time() - t0,
        "memory_bytes": {
            **mem,
            "consolidation_bytes": 0,
            "replay_buffer_bytes": full_data_bytes,
            "extra_cl_bytes": full_data_bytes,
            "extra_cl_mb": full_data_bytes / 1e6,
        },
    }


def run_experience_replay(
    model: nn.Module,
    tasks: List[Dict],
    heldout_tasks: Optional[List[Dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    memory_per_task: int = 16,
    beta_replay: float = 1.0,
    device: str = "cpu",
) -> Dict:
    """Standard Experience Replay (Rolnick et al., 2018) on matched episodic budget."""
    model = model.to(device)
    t0 = time.time()
    r_init_mse, r_init_rel = evaluate_all_tasks_row(model, tasks, device=device)
    memory = EpisodicMemory(capacity_per_task=memory_per_task)

    R_mse, R_rel = [], []
    for k, task in enumerate(tasks):
        optimizer = Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        for _ in range(epochs):
            model.train()
            for u_in, re, u_out in task["train_loader"]:
                u_in, re, u_out = u_in.to(device), re.to(device), u_out.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(u_in, re), u_out)
                if len(memory) > 0:
                    mu, mre, mn, _ = memory.sample(u_in.shape[0], device=device)
                    loss = loss + beta_replay * F.mse_loss(model(mu, mre), mn)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            sched.step()

        memory.add_task_dataset(k, task["train_ds"])
        m_row, r_row = evaluate_all_tasks_row(model, tasks, device=device)
        R_mse.append(m_row)
        R_rel.append(r_row)

    heldout_mse, heldout_rel = {}, {}
    if heldout_tasks:
        for ht in heldout_tasks:
            m_row, r_row = evaluate_all_tasks_row(model, [ht], device=device)
            heldout_mse[ht["label"]] = m_row[0]
            heldout_rel[ht["label"]] = r_row[0]

    s_mse = summarise_results(R_mse, r_init=r_init_mse)
    s_rel = summarise_results(R_rel, r_init=r_init_rel)
    mem = model.memory_breakdown_bytes() if hasattr(model, "memory_breakdown_bytes") else {}
    rep_b = memory.memory_bytes()
    return {
        "eval_matrix": R_mse,
        "eval_matrix_rel_l2": R_rel,
        "r_init_mse": r_init_mse,
        "avg_forgetting": s_mse["avg_forgetting"],
        "avg_forgetting_rel_l2": s_rel["avg_forgetting"],
        "bwt": s_mse["bwt"],
        "fwt": s_mse["fwt"],
        "avg_final_mse": s_mse["avg_final_mse"],
        "avg_final_rel_l2": s_rel["avg_final_mse"],
        "task0_degradation_pct": s_mse["task0_degradation_pct"],
        "heldout_mse": heldout_mse,
        "heldout_rel_l2": heldout_rel,
        "wall_clock_sec": time.time() - t0,
        "memory_bytes": {**mem, "consolidation_bytes": 0, "replay_buffer_bytes": rep_b, "extra_cl_bytes": rep_b, "extra_cl_mb": rep_b / 1e6},
    }


class TemporalSlicingBuffer:
    """
    Replay-TS Buffer (Hemati et al., 2026):
    Stores compact contiguous 3-frame autoregressive temporal windows (u_t, u_{t+1}, u_{t+2})
    sampled across past trajectories under a matched total frame storage budget.
    """

    def __init__(self, frame_budget_per_task: int = 32, window_len: int = 3):
        # Match ER's `memory_per_task * 2` stored frames (u_in + u_out = 2 frames/exemplar)
        self.n_windows = max(2, frame_budget_per_task // window_len)
        self.window_len = window_len
        self._wins: List[torch.Tensor] = []
        self._res: List[torch.Tensor] = []
        self._cached = None

    def add_task_dataset(self, dataset, re_norm: float):
        candidates = []
        for seg in dataset.segments:
            for start in range(0, len(seg) - self.window_len + 1, 2):
                candidates.append(seg[start : start + self.window_len])
        if not candidates:
            return
        take = min(self.n_windows, len(candidates))
        idxs = np.linspace(0, len(candidates) - 1, take, dtype=int)
        chosen = np.stack([candidates[i] for i in idxs], axis=0).astype(np.float32)  # (take, W, ...)
        self._wins.append(torch.from_numpy(chosen))
        self._res.append(torch.full((take, 1), float(re_norm), dtype=torch.float32))
        self._cached = (torch.cat(self._wins, dim=0), torch.cat(self._res, dim=0))

    def __len__(self) -> int:
        return 0 if self._cached is None else int(self._cached[0].shape[0])

    def sample(self, batch_size: int, device: str = "cpu"):
        if self._cached is None:
            return None
        wins, res = self._cached
        k = min(batch_size, wins.shape[0])
        idx = torch.randperm(wins.shape[0])[:k]
        return wins[idx].to(device), res[idx].to(device)

    def memory_bytes(self) -> int:
        if self._cached is None:
            return 0
        return sum(t.numel() * t.element_size() for t in self._cached)


def run_replay_ts(
    model: nn.Module,
    tasks: List[Dict],
    heldout_tasks: Optional[List[Dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    memory_per_task: int = 16,
    beta_replay: float = 1.0,
    device: str = "cpu",
) -> Dict:
    """
    Replay-TS baseline (Hemati et al., 2026):
    Stores compact 3-step contiguous temporal windows matched to the exact same
    frame storage budget as ER (`memory_per_task * 2` frames/task) and applies
    autoregressive 2-step rollout replay loss to combat temporal drift.
    """
    model = model.to(device)
    t0 = time.time()
    r_init_mse, r_init_rel = evaluate_all_tasks_row(model, tasks, device=device)
    buf = TemporalSlicingBuffer(frame_budget_per_task=memory_per_task * 2, window_len=3)

    R_mse, R_rel = [], []
    for k, task in enumerate(tasks):
        optimizer = Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        for _ in range(epochs):
            model.train()
            for u_in, re, u_out in task["train_loader"]:
                u_in, re, u_out = u_in.to(device), re.to(device), u_out.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(u_in, re), u_out)

                if len(buf) > 0:
                    w_seq, w_re = buf.sample(u_in.shape[0], device=device)
                    # 2-step autoregressive temporal-slicing replay supervision
                    p1 = model(w_seq[:, 0], w_re)
                    p2 = model(p1, w_re)
                    l_ts = 0.5 * (F.mse_loss(p1, w_seq[:, 1]) + F.mse_loss(p2, w_seq[:, 2]))
                    loss = loss + beta_replay * l_ts

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            sched.step()

        buf.add_task_dataset(task["train_ds"], task["re_norm"])
        m_row, r_row = evaluate_all_tasks_row(model, tasks, device=device)
        R_mse.append(m_row)
        R_rel.append(r_row)

    heldout_mse, heldout_rel = {}, {}
    if heldout_tasks:
        for ht in heldout_tasks:
            m_row, r_row = evaluate_all_tasks_row(model, [ht], device=device)
            heldout_mse[ht["label"]] = m_row[0]
            heldout_rel[ht["label"]] = r_row[0]

    s_mse = summarise_results(R_mse, r_init=r_init_mse)
    s_rel = summarise_results(R_rel, r_init=r_init_rel)
    mem = model.memory_breakdown_bytes() if hasattr(model, "memory_breakdown_bytes") else {}
    rep_b = buf.memory_bytes()
    return {
        "eval_matrix": R_mse,
        "eval_matrix_rel_l2": R_rel,
        "r_init_mse": r_init_mse,
        "avg_forgetting": s_mse["avg_forgetting"],
        "avg_forgetting_rel_l2": s_rel["avg_forgetting"],
        "bwt": s_mse["bwt"],
        "fwt": s_mse["fwt"],
        "avg_final_mse": s_mse["avg_final_mse"],
        "avg_final_rel_l2": s_rel["avg_final_mse"],
        "task0_degradation_pct": s_mse["task0_degradation_pct"],
        "heldout_mse": heldout_mse,
        "heldout_rel_l2": heldout_rel,
        "wall_clock_sec": time.time() - t0,
        "memory_bytes": {**mem, "consolidation_bytes": 0, "replay_buffer_bytes": rep_b, "extra_cl_bytes": rep_b, "extra_cl_mb": rep_b / 1e6},
    }


def run_sle_fno(
    model: nn.Module,
    tasks: List[Dict],
    heldout_tasks: Optional[List[Dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict:
    """
    SLE-FNO baseline (Elhadidy et al., 2026):
    Trains the shared FNO backbone + adapter 0 on Task 0, then FREEZES the
    shared backbone completely and trains ONLY a new single-layer extension
    adapter per subsequent task (zero replay, zero EWC). Uses soft Re-proximity
    routing at inference time for task-agnostic evaluation.
    """
    model = model.to(device)
    t0 = time.time()
    r_init_mse, r_init_rel = evaluate_all_tasks_row(model, tasks, device=device)

    R_mse, R_rel = [], []
    for k, task in enumerate(tasks):
        model.add_regime_adapter(device=torch.device(device))
        # After Task 0, freeze shared backbone completely (Single-Layer Extension)
        if k > 0:
            for p in model.backbone.parameters():
                p.requires_grad = False
        model.freeze_past_adapters(active_task_id=k)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        for _ in range(epochs):
            model.train()
            for u_in, re, u_out in task["train_loader"]:
                u_in, re, u_out = u_in.to(device), re.to(device), u_out.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(u_in, re, task_id=k)
                loss = F.mse_loss(pred, u_out)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
            sched.step()

        # Register task Re for soft routing (SLE-FNO has no contrastive encoder)
        dummy = torch.zeros(1, getattr(model, "latent_dim", 32))
        model.register_prototype(k, dummy, re_norm=task.get("re_norm"))

        m_row, r_row = evaluate_all_tasks_row(model, tasks, device=device, routing_mode="soft")
        R_mse.append(m_row)
        R_rel.append(r_row)

    heldout_mse, heldout_rel = {}, {}
    if heldout_tasks:
        for ht in heldout_tasks:
            m_row, r_row = evaluate_all_tasks_row(model, [ht], device=device, routing_mode="soft")
            heldout_mse[ht["label"]] = m_row[0]
            heldout_rel[ht["label"]] = r_row[0]

    s_mse = summarise_results(R_mse, r_init=r_init_mse)
    s_rel = summarise_results(R_rel, r_init=r_init_rel)
    mem = model.memory_breakdown_bytes()
    extra_b = mem["adapter_bytes"] + mem["prototype_bytes"]
    return {
        "eval_matrix": R_mse,
        "eval_matrix_rel_l2": R_rel,
        "r_init_mse": r_init_mse,
        "avg_forgetting": s_mse["avg_forgetting"],
        "avg_forgetting_rel_l2": s_rel["avg_forgetting"],
        "bwt": s_mse["bwt"],
        "fwt": s_mse["fwt"],
        "avg_final_mse": s_mse["avg_final_mse"],
        "avg_final_rel_l2": s_rel["avg_final_mse"],
        "task0_degradation_pct": s_mse["task0_degradation_pct"],
        "heldout_mse": heldout_mse,
        "heldout_rel_l2": heldout_rel,
        "wall_clock_sec": time.time() - t0,
        "memory_bytes": {**mem, "consolidation_bytes": 0, "replay_buffer_bytes": 0, "extra_cl_bytes": extra_b, "extra_cl_mb": extra_b / 1e6},
    }
