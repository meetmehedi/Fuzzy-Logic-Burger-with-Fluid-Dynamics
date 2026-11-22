"""
metrics.py
==========
Standardised continual-learning and field-accuracy metrics for MEWC-Flow.

Sign Conventions (unambiguous across the entire codebase & manuscript):
-----------------------------------------------------------------------
In MSE space (where LOWER is better):
- avg_forgetting = (1/(T-1)) * sum_{j=0}^{T-2} [ R[T-1, j] - R[j, j] ]
      > 0  : catastrophic forgetting (error increased after later tasks)
      = 0  : zero forgetting
      < 0  : positive backward transfer (later tasks reduced error on earlier tasks)
- bwt = -avg_forgetting
      > 0  : positive backward transfer
      < 0  : catastrophic forgetting
- fwt = (1/(T-1)) * sum_{j=1}^{T-1} [ R_init[j] - R[j-1, j] ]
      > 0  : positive forward transfer relative to untrained initialisation
- avg_final_mse = (1/T) * sum_{j=0}^{T-1} R[T-1, j]   (lower is better)
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-sample relative L2 error: ||pred - target||_2 / (||target||_2 + eps)."""
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1)
    targ = target.reshape(B, -1)
    return (diff.norm(dim=-1) / (targ.norm(dim=-1) + eps)).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


@torch.no_grad()
def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
    routing_mode: str = "soft",
    task_id: Optional[int] = None,
    dy: float = 1.0,
    dx: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader returning exact sample-weighted MSE and Rel-L2.
    Task-agnostic by default (task_id=None, routing_mode='soft').
    """
    model.eval()
    total_sq = 0.0
    total_elems = 0
    rel_l2_sum = 0.0
    n_samples = 0

    is_modular = hasattr(model, "adapters")
    for u_in, re, u_out in loader:
        u_in = u_in.to(device)
        re = re.to(device)
        u_out = u_out.to(device)

        if is_modular:
            pred = model(u_in, re, task_id=task_id, routing_mode=routing_mode, dy=dy, dx=dx)
        else:
            pred = model(u_in, re)

        total_sq += torch.nn.functional.mse_loss(pred, u_out, reduction="sum").item()
        total_elems += u_out.numel()

        B = u_in.shape[0]
        diff = (pred - u_out).reshape(B, -1).norm(dim=-1)
        denom = u_out.reshape(B, -1).norm(dim=-1) + 1e-8
        rel_l2_sum += (diff / denom).sum().item()
        n_samples += B

    return {
        "mse": total_sq / max(1, total_elems),
        "rel_l2": rel_l2_sum / max(1, n_samples),
    }


def compute_forgetting(R: List[List[Optional[float]]]) -> float:
    """
    Average Forgetting in error (MSE / Rel-L2) space.
    Positive = forgetting (degradation); Negative = backward improvement.
    """
    T = len(R)
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        r_final = R[T - 1][j]
        r_diag = R[j][j]
        if r_final is not None and r_diag is not None and np.isfinite(r_final) and np.isfinite(r_diag):
            vals.append(float(r_final - r_diag))
    return float(np.mean(vals)) if vals else 0.0


def compute_bwt(R: List[List[Optional[float]]]) -> float:
    """Backward Transfer: BWT = -Average Forgetting (positive = improvement)."""
    return -compute_forgetting(R)


def compute_fwt(
    R: List[List[Optional[float]]],
    r_init: Optional[List[float]] = None,
) -> float:
    """
    Forward Transfer:
      If r_init (untrained baseline error per task) is provided:
          FWT = mean_{j=1..T-1} (r_init[j] - R[j-1][j])   (positive = helpful transfer)
      Otherwise returns mean zero-shot error R[j-1][j].
    """
    T = len(R)
    if T < 2:
        return 0.0
    vals = []
    for j in range(1, T):
        r_prev = R[j - 1][j]
        if r_prev is not None and np.isfinite(r_prev):
            if r_init is not None and j < len(r_init) and np.isfinite(r_init[j]):
                vals.append(float(r_init[j] - r_prev))
            else:
                vals.append(float(r_prev))
    return float(np.mean(vals)) if vals else 0.0


def compute_avg_acc(R: List[List[Optional[float]]]) -> float:
    """Average final error across all tasks after training the last task (lower is better)."""
    T = len(R)
    row = [float(R[T - 1][j]) for j in range(T) if R[T - 1][j] is not None and np.isfinite(R[T - 1][j])]
    return float(np.mean(row)) if row else 0.0


def compute_task0_degradation_pct(R: List[List[Optional[float]]]) -> float:
    """Percentage degradation on Task 0 from right after Task 0 to after final task."""
    T = len(R)
    r0 = R[0][0]
    rf = R[T - 1][0]
    if r0 is None or rf is None or r0 <= 0:
        return 0.0
    return float(100.0 * (rf - r0) / r0)


def summarise_results(
    R: List[List[Optional[float]]],
    r_init: Optional[List[float]] = None,
) -> Dict[str, Any]:
    forgetting = compute_forgetting(R)
    return {
        "avg_forgetting": forgetting,
        "bwt": -forgetting,
        "fwt": compute_fwt(R, r_init=r_init),
        "avg_final_mse": compute_avg_acc(R),
        "avg_acc": compute_avg_acc(R),
        "task0_degradation_pct": compute_task0_degradation_pct(R),
        "n_tasks": len(R),
    }


def print_eval_matrix(R: List[List[Optional[float]]], task_labels: Optional[List[str]] = None):
    T = len(R)
    labels = task_labels or [f"T{i}" for i in range(T)]
    header = "After\\On  " + "  ".join(f"{l[:10]:>10}" for l in labels)
    print(header)
    for i in range(T):
        row_str = f"Task-{i:>2}    "
        for j in range(T):
            val = R[i][j]
            if val is None or not np.isfinite(val):
                row_str += f"{'--':>10}  "
            else:
                row_str += f"{val:10.6f}  "
        print(row_str)
    s = summarise_results(R)
    print(f"  AvgForgetting={s['avg_forgetting']:+.6f} | BWT={s['bwt']:+.6f} | "
          f"AvgFinalMSE={s['avg_final_mse']:.6f} | T0_Degrad={s['task0_degradation_pct']:+.1f}%")
