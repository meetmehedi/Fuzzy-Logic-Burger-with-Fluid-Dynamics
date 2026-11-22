"""
ewc_trainer.py
==============
Fair sequential multi-task trainer for Static EWC (Kirkpatrick et al., 2017)
and Online EWC / OFU (Schwarz et al., 2018) on complex-parameter neural operators.

Matches exact optimizer, CosineAnnealingLR (eta_min=0.01*lr), gradient clipping
(max_norm=1.0), full T x T evaluation matrix (for FWT + BWT + Forgetting), and
exact byte-level memory accounting.
"""

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Optional

from mewc_flow.models.ewc import EWC
from mewc_flow.trainers.base_trainer import evaluate_all_tasks_row, evaluate_on_task
from mewc_flow.utils.metrics import summarise_results


class EWCTrainer:
    """
    Sequential trainer with Static EWC (online_decay=None) or
    Online EWC (online_decay in (0, 1)).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 200.0,
        lr: float = 1e-3,
        epochs: int = 20,
        n_fisher: int = 160,
        online_decay: Optional[float] = None,
        fisher_noise: float = 0.0,
        max_grad_norm: float = 1.0,
        log_every: int = 10,
        verbose: bool = False,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.fisher_noise = fisher_noise
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.verbose = verbose
        self.ewc = EWC(
            model,
            lambda_ewc=lambda_ewc,
            n_samples=n_fisher,
            online_decay=online_decay,
        )

    def run_all_tasks(
        self,
        tasks: List[Dict],
        heldout_tasks: Optional[List[Dict]] = None,
        device: str = "cpu",
    ) -> Dict:
        self.model = self.model.to(device)
        t_start = time.time()

        # Baseline untrained error for exact FWT computation
        r_init_mse, r_init_rel = evaluate_all_tasks_row(self.model, tasks, device=device)

        R_mse: List[List[float]] = []
        R_rel: List[List[float]] = []
        per_task_history = []

        for task_idx, task in enumerate(tasks):
            hist = self._train_task(task, task_idx, device)
            per_task_history.append(hist)

            mse_row, rel_row = evaluate_all_tasks_row(self.model, tasks, device=device)
            R_mse.append(mse_row)
            R_rel.append(rel_row)

            # Consolidate diagonal Fisher after task completes
            self.ewc.consolidate(
                task["train_loader"],
                device=device,
                fisher_noise=self.fisher_noise,
            )

        heldout_mse, heldout_rel = {}, {}
        if heldout_tasks:
            for ht in heldout_tasks:
                res_ht = evaluate_all_tasks_row(self.model, [ht], device=device)
                heldout_mse[ht["label"]] = res_ht[0][0]
                heldout_rel[ht["label"]] = res_ht[1][0]

        elapsed = time.time() - t_start
        summary_mse = summarise_results(R_mse, r_init=r_init_mse)
        summary_rel = summarise_results(R_rel, r_init=r_init_rel)

        mem_model = self.model.memory_breakdown_bytes() if hasattr(self.model, "memory_breakdown_bytes") else {}
        ewc_bytes = self.ewc.memory_bytes()

        return {
            "eval_matrix": R_mse,
            "eval_matrix_rel_l2": R_rel,
            "r_init_mse": r_init_mse,
            "r_init_rel_l2": r_init_rel,
            "avg_forgetting": summary_mse["avg_forgetting"],
            "avg_forgetting_rel_l2": summary_rel["avg_forgetting"],
            "bwt": summary_mse["bwt"],
            "fwt": summary_mse["fwt"],
            "avg_final_mse": summary_mse["avg_final_mse"],
            "avg_final_rel_l2": summary_rel["avg_final_mse"],
            "avg_acc": summary_mse["avg_final_mse"],
            "task0_degradation_pct": summary_mse["task0_degradation_pct"],
            "heldout_mse": heldout_mse,
            "heldout_rel_l2": heldout_rel,
            "per_task_history": per_task_history,
            "wall_clock_sec": elapsed,
            "memory_bytes": {
                **mem_model,
                "consolidation_bytes": ewc_bytes,
                "replay_buffer_bytes": 0,
                "extra_cl_bytes": ewc_bytes,
                "extra_cl_mb": ewc_bytes / 1e6,
            },
        }

    def _train_task(self, task: Dict, task_idx: int, device: str) -> Dict:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        train_loader = task["train_loader"]
        train_losses, val_losses, ewc_penalties = [], [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            ep_mse, ep_ewc, n_b = 0.0, 0.0, 0
            for u_in, re, u_out in train_loader:
                u_in = u_in.to(device)
                re = re.to(device)
                u_out = u_out.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model(u_in, re)
                mse_loss = nn.functional.mse_loss(pred, u_out)
                ewc_pen = self.ewc.penalty(self.model)
                loss = mse_loss + ewc_pen

                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()

                ep_mse += mse_loss.item()
                ep_ewc += float(ewc_pen.item())
                n_b += 1

            sched.step()
            train_losses.append(ep_mse / max(1, n_b))
            ewc_penalties.append(ep_ewc / max(1, n_b))
            if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs):
                v = evaluate_on_task(self.model, task, device=device)
                val_losses.append(v)
                print(f"  [EWC T{task_idx}] Ep {epoch:2d}/{self.epochs} | MSE={train_losses[-1]:.6f} EWC={ewc_penalties[-1]:.5f} val={v:.6f}")

        return {
            "label": task["label"],
            "task_idx": task_idx,
            "train_loss": train_losses,
            "ewc_penalty": ewc_penalties,
        }
