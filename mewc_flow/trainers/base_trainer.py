"""
base_trainer.py
===============
Shared deterministic seeding, fair optimization helpers, full T x T evaluation
(including pre-task zero-shot evaluation for exact Forward Transfer FWT), and
vanilla Fine-Tuning lower bound.

All trainers in MEWC-Flow share identical optimization primitives:
- Optimizer : Adam(lr=lr)
- Scheduler : CosineAnnealingLR(T_max=epochs, eta_min=lr * 0.01)
- Stability : clip_grad_norm_(model.parameters(), max_norm=1.0)
"""

import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from mewc_flow.utils.metrics import evaluate_model_on_loader, summarise_results


def set_all_seeds(seed: int):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on_task(
    model: nn.Module,
    task: Dict,
    device: str = "cpu",
    routing_mode: str = "soft",
    task_id: Optional[int] = None,
) -> float:
    """Return sample-weighted validation MSE on a task (task-agnostic by default)."""
    res = evaluate_model_on_loader(
        model,
        task["val_loader"],
        device=device,
        routing_mode=routing_mode,
        task_id=task_id,
        dy=task.get("dy", 1.0),
        dx=task.get("dx", 1.0),
    )
    return res["mse"]


def evaluate_on_task_full(
    model: nn.Module,
    task: Dict,
    device: str = "cpu",
    routing_mode: str = "soft",
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """Return both MSE and Rel-L2 on a task (task-agnostic by default)."""
    return evaluate_model_on_loader(
        model,
        task["val_loader"],
        device=device,
        routing_mode=routing_mode,
        task_id=task_id,
        dy=task.get("dy", 1.0),
        dx=task.get("dx", 1.0),
    )


def evaluate_all_tasks_row(
    model: nn.Module,
    tasks: List[Dict],
    device: str = "cpu",
    routing_mode: str = "soft",
) -> tuple[List[float], List[float]]:
    """Evaluate model across ALL T tasks (both past and unseen future tasks for FWT)."""
    mse_row, rel_row = [], []
    for t in tasks:
        res = evaluate_on_task_full(model, t, device=device, routing_mode=routing_mode)
        mse_row.append(res["mse"])
        rel_row.append(res["rel_l2"])
    return mse_row, rel_row


def train_one_task(
    model: nn.Module,
    task: Dict,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 10,
    scheduler: bool = True,
    max_grad_norm: float = 1.0,
    verbose: bool = False,
) -> Dict:
    """Vanilla sequential fine-tuning on a single task (fair lower bound)."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        if scheduler
        else None
    )

    train_loader: DataLoader = task["train_loader"]
    val_loader: DataLoader = task["val_loader"]
    label: str = task["label"]

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for u_in, re, u_out in train_loader:
            u_in = u_in.to(device)
            re = re.to(device)
            u_out = u_out.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(u_in, re)
            loss = nn.functional.mse_loss(pred, u_out)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if sched is not None:
            sched.step()

        tr_l = epoch_loss / max(1, n_batches)
        val_l = evaluate_on_task(model, task, device=device)
        train_losses.append(tr_l)
        val_losses.append(val_l)

        if verbose and (epoch % log_every == 0 or epoch == epochs):
            print(f"  [{label}] Ep {epoch:2d}/{epochs} | train={tr_l:.6f} val={val_l:.6f} ({time.time()-t0:.1f}s)")

    return {"train_loss": train_losses, "val_loss": val_losses, "label": label}


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    return evaluate_model_on_loader(model, loader, device=device)["mse"]
