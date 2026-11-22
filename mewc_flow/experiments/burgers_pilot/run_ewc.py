"""
run_ewc.py
==========
Benchmark 4: 1-D Viscous Burgers Shock-Formation Benchmark & RQ4 Study
(Physics-Residual Replay vs. Stored-Window Replay).

Task Sequence (100x Reynolds-number span: smooth diffusive -> steep shock front):
  Task 0: nu = 0.0500 (Re = 20)
  Task 1: nu = 0.0100 (Re = 100)
  Task 2: nu = 0.0020 (Re = 500)
  Task 3: nu = 0.0005 (Re = 2000)
  Held-out: nu = 0.0200 (Re = 50), nu = 0.0050 (Re = 200)

Compares 9 methods across 3 seeds (loads config from burgers_pilot.yaml when available):
  1. Fine-Tuning (Lower bound)
  2. Static EWC
  3. Online EWC (OFU)
  4. ER (Stored Exemplar Replay)
  5. Replay-TS (Temporal-Slicing Window Replay)
  6. SLE-FNO (Single-Layer Extension)
  7. MEWC-Flow (PDE-Residual Replay, 0 stored data — answers RQ4!)
  8. MEWC-Flow (Full: Stored Replay + PDE Residual + A-EWC + Modular Contrastive)
  9. Joint Training (Upper bound)

Evaluates exact 1-D Burgers PDE residual, 20-step shock rollout Rel-L2,
forgetting, zero-shot interpolation, and statistical significance.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.burgers_dataset import make_burgers_tasks, burgers_pde_residual_torch
from mewc_flow.models.surrogate import build_surrogate
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import (
    run_finetuning, run_joint_training, run_experience_replay, run_replay_ts, run_sle_fno
)
from mewc_flow.utils.stats import summarize_seeds, paired_significance
from mewc_flow.utils.results_io import save_json, save_checkpoint


METHODS = [
    "Fine-Tuning",
    "EWC",
    "Online-EWC",
    "ER",
    "Replay-TS",
    "SLE-FNO",
    "MEWC-PDE-Replay",
    "MEWC-Flow",
    "Joint",
]


@torch.no_grad()
def eval_burgers_rollout_and_pde(model, task, n_steps: int = 20, device: str = "cpu") -> Dict:
    model.eval()
    is_mod = hasattr(model, "adapters")
    # Exact 1-step PDE residual over val_loader
    res_sq, n_el = 0.0, 0
    for u_in, re_enc, _ in task["val_loader"]:
        u_in_d, re_d = u_in.to(device), re_enc.to(device)
        pred = model(u_in_d, re_d, routing_mode="soft") if is_mod else model(u_in_d, re_d)
        r = burgers_pde_residual_torch(u_in_d, pred, task["nu"])
        res_sq += r.pow(2).sum().item()
        n_el += r.numel()
    pde_rmse = float(np.sqrt(res_sq / max(1, n_el)))

    # Multi-IC 20-step autoregressive rollout
    wins = task["val_ds"].get_rollout_windows(n_steps=n_steps)
    arr = np.stack(wins, axis=0)  # (N_ic, n_steps+1, nx)
    u_cur = torch.from_numpy(arr[:, 0]).to(device)
    re_t = torch.full((len(arr), 1), float(task["re_norm"]), dtype=torch.float32, device=device)

    preds = []
    for _ in range(n_steps):
        u_nxt = model(u_cur, re_t, routing_mode="soft") if is_mod else model(u_cur, re_t)
        preds.append(u_nxt.cpu().numpy())
        u_cur = u_nxt
    pred_tr = np.stack(preds, axis=1)  # (N_ic, n_steps, nx)
    true_tr = arr[:, 1 : n_steps + 1]

    diff = np.linalg.norm(pred_tr - true_tr, axis=-1)
    denom = np.linalg.norm(true_tr, axis=-1) + 1e-8
    curve = (diff / denom).mean(axis=0)
    return {
        "pde_residual_rmse": pde_rmse,
        "rollout_curve": curve.tolist(),
        "step10_rel_l2": float(curve[min(9, len(curve) - 1)]),
        "step20_rel_l2": float(curve[-1]),
    }


def _run_job(job: Dict) -> Dict:
    torch.set_num_threads(job["num_threads"])
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]
    mem = job["memory_per_task"]
    device = "cpu"

    set_all_seeds(seed)
    tasks, heldout = make_burgers_tasks(
        nx=128, nt=40, n_train=24, n_val=8, batch_size=64, order="ascending", include_heldout=True, seed=seed
    )

    if method == "Fine-Tuning":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = run_finetuning(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "Joint":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = run_joint_training(model, tasks, heldout, epochs=epochs, lr=1e-3, batch_size=64, device=device, seed=seed)
    elif method == "EWC":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=256, online_decay=None).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "Online-EWC":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=256, online_decay=0.85).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "ER":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = run_experience_replay(model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    elif method == "Replay-TS":
        model = build_surrogate(nx=128, width=32, modes=16, device=device)
        res = run_replay_ts(model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    elif method == "SLE-FNO":
        model = build_surrogate(nx=128, width=32, modes=16, device=device, modular=True, use_contrastive_encoder=False)
        res = run_sle_fno(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "MEWC-PDE-Replay":
        # Zero-stored-data physics-residual replay (answers RQ4!)
        model = build_surrogate(nx=128, width=32, modes=16, device=device, modular=True, use_contrastive_encoder=False)
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="adaptive",
            lambda_min=0.05, lambda_max=1.0, n_fisher=256,
            use_contrastive=False, use_replay=False, memory_per_task=0,
            use_physics_loss=True, gamma_physics=0.02,
            use_pde_residual_replay=True, gamma_pde_replay=1e-6,
            is_1d=True,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)
    elif method == "MEWC-Flow":
        model = build_surrogate(nx=128, width=32, modes=16, device=device, modular=True, use_contrastive_encoder=True)
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="adaptive",
            lambda_min=0.01, lambda_max=0.15, n_fisher=256,
            use_contrastive=True, alpha_contrastive=0.20, temperature=0.10,
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.02,
            use_pde_residual_replay=True, gamma_pde_replay=5e-7,
            is_1d=True,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)
    else:
        raise ValueError(method)

    evals = [eval_burgers_rollout_and_pde(model, t, n_steps=20, device=device) for t in tasks]
    if job["save_ckpt"]:
        save_checkpoint(model, job["out_dir"], filename=f"model_{method.lower().replace('-', '_')}.pt")

    res.pop("per_task_history", None)
    res.pop("histories", None)
    res["method"] = method
    res["seed"] = seed
    res["mean_pde_residual"] = float(np.mean([e["pde_residual_rmse"] for e in evals]))
    res["mean_rollout_step10"] = float(np.mean([e["step10_rel_l2"] for e in evals]))
    res["mean_rollout_step20"] = float(np.mean([e["step20_rel_l2"] for e in evals]))
    res["task0_rollout_curve"] = evals[0]["rollout_curve"]
    res["task0_rollout_step20"] = evals[0]["step20_rel_l2"]
    res["heldout_mean_mse"] = float(np.mean(list(res["heldout_mse"].values())))

    print(f"  [Burgers][{method:<15}][s={seed}] FinalMSE={res['avg_final_mse']:.6f} | "
          f"Forget={res['avg_forgetting']:+.6f} | PDE={res['mean_pde_residual']:.3f} | "
          f"R20={res['mean_rollout_step20']:.4f} | HO={res['heldout_mean_mse']:.6f}", flush=True)
    return res


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "burgers_pilot.yaml")
    default_epochs = 20
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            y = yaml.safe_load(f) or {}
            default_epochs = int(y.get("training", {}).get("epochs", default_epochs))

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=default_epochs)
    p.add_argument("--memory_per_task", type=int, default=32)
    p.add_argument("--seeds", type=str, default="42,102,202")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [
        {
            "method": m, "seed": s, "epochs": args.epochs, "memory_per_task": args.memory_per_task,
            "out_dir": args.out_dir, "save_ckpt": (s == seeds[0]),
            "num_threads": max(1, (os.cpu_count() or 4) // args.workers),
        }
        for s in seeds for m in METHODS
    ]

    by_m: Dict[str, List[Dict]] = {m: [] for m in METHODS}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(_run_job, j) for j in jobs]):
            r = f.result()
            by_m[r["method"]].append(r)

    scalar_keys = [
        "avg_forgetting", "bwt", "fwt", "avg_final_mse", "avg_final_rel_l2",
        "task0_degradation_pct", "heldout_mean_mse", "mean_pde_residual",
        "mean_rollout_step10", "mean_rollout_step20", "task0_rollout_step20", "wall_clock_sec",
    ]
    summary = {}
    for m, runs in by_m.items():
        rs = sorted(runs, key=lambda r: r["seed"])
        ms = {k: summarize_seeds([r[k] for r in rs]) for k in scalar_keys}
        ms["eval_matrix_mean"] = np.mean([r["eval_matrix"] for r in rs], axis=0).tolist()
        ms["task0_rollout_curve_mean"] = np.mean([r["task0_rollout_curve"] for r in rs], axis=0).tolist()
        ms["memory_bytes"] = rs[0]["memory_bytes"]
        summary[m] = ms

    sig = {}
    mewc_rs = sorted(by_m["MEWC-Flow"], key=lambda r: r["seed"])
    for b in METHODS:
        if b == "MEWC-Flow":
            continue
        b_rs = sorted(by_m[b], key=lambda r: r["seed"])
        sig[f"MEWC-Flow_vs_{b}"] = {
            k: paired_significance([r[k] for r in mewc_rs], [r[k] for r in b_rs])
            for k in ["avg_forgetting", "avg_final_mse", "mean_pde_residual", "mean_rollout_step20", "heldout_mean_mse"]
        }

    out = {
        "dataset": "1-D Viscous Burgers Shock-Formation Benchmark (nu: 0.05 -> 0.01 -> 0.002 -> 0.0005)",
        "seeds": seeds,
        "epochs_per_task": args.epochs,
        "memory_per_task": args.memory_per_task,
        "methods": summary,
        "significance_tests": sig,
    }
    out_path = os.path.join(args.out_dir, "burgers_benchmark_summary.json")
    save_json(out, out_path)
    print(f"\nSaved Burgers Q1 + RQ4 PDE-Residual Replay benchmark -> {out_path}")


if __name__ == "__main__":
    main()
