"""
run_ablations.py
================
Comprehensive Multi-Seed Ablation, Routing Mode, Task-Order Sensitivity (RQ5),
and Memory-Budget Sweep on Real Experimental PIV Cylinder Wake Data.

Four Rigorous Experimental Studies (all multi-seed, zero data leakage, strict JSON):
1. Systematic Component Ablation (8 variants x 3 seeds):
   - Full MEWC-Flow (Adapters + Contrastive Router + A-EWC/OFU + Replay + Vorticity Physics)
   - w/o Modular Adapters (monolithic FNO)
   - w/o Contrastive Router (Re-proximity / non-physics routing)
   - w/o Adaptive EWC (replaced with static EWC)
   - w/o EWC (lambda = 0)
   - w/o Episodic Replay (STRICT 0-byte memory — no buffer access anywhere!)
   - w/o Physics Consistency Loss (gamma_phys = 0)
   - Replay-Only (ER baseline)
   Evaluated on Forgetting, Final MSE, Multi-Window Rollout (T=10, 20), Vorticity RMSE,
   AND Zero-Shot Held-Out Regimes (Re=910, 1890)!
2. Routing Mode Ablation (on trained Full MEWC-Flow models):
   - Soft Prototype Routing vs Hard Top-1 Prototype vs Oracle Task ID vs Uniform Averaging
3. Task-Order Sensitivity (RQ5, 3 orders x 3 seeds):
   - Ascending (240->650->1280->2520) vs Descending (2520->1280->650->240) vs Shuffled (1280->240->2520->650)
   - Compares Fine-Tuning, ER, and MEWC-Flow
4. Memory-Budget Sensitivity Sweep (4 budgets x 3 seeds):
   - memory_per_task in {4, 8, 16, 32} comparing ER, Replay-TS, and MEWC-Flow
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
import torch

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.piv_cylinder_dataset import make_piv_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.trainers.base_trainer import set_all_seeds, evaluate_all_tasks_row
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import run_finetuning, run_experience_replay, run_replay_ts
from mewc_flow.utils.physics_metrics import multi_window_rollout_eval
from mewc_flow.utils.stats import summarize_seeds
from mewc_flow.utils.results_io import save_json


ABLATION_CONFIGS = [
    {
        "name": "Full MEWC-Flow",
        "use_adapters": True, "use_contrastive": True, "cons": "adaptive",
        "use_replay": True, "use_physics": True,
    },
    {
        "name": "w/o Modular Adapters",
        "use_adapters": False, "use_contrastive": True, "cons": "adaptive",
        "use_replay": True, "use_physics": True,
    },
    {
        "name": "w/o Contrastive Router",
        "use_adapters": True, "use_contrastive": False, "cons": "adaptive",
        "use_replay": True, "use_physics": True,
    },
    {
        "name": "w/o Adaptive EWC (Static)",
        "use_adapters": True, "use_contrastive": True, "cons": "static",
        "use_replay": True, "use_physics": True,
    },
    {
        "name": "w/o EWC",
        "use_adapters": True, "use_contrastive": True, "cons": "none",
        "use_replay": True, "use_physics": True,
    },
    {
        "name": "w/o Episodic Replay (0 MB)",
        "use_adapters": True, "use_contrastive": True, "cons": "adaptive",
        "use_replay": False, "use_physics": True,
    },
    {
        "name": "w/o Physics Loss",
        "use_adapters": True, "use_contrastive": True, "cons": "adaptive",
        "use_replay": True, "use_physics": False,
    },
    {
        "name": "Replay-Only (ER)",
        "use_adapters": False, "use_contrastive": False, "cons": "none",
        "use_replay": True, "use_physics": False,
    },
]


def _run_ablation_job(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job["num_threads"])
    cfg = job["cfg"]
    seed = job["seed"]
    epochs = job["epochs"]
    mem = job["memory_per_task"] if cfg["use_replay"] else 0

    set_all_seeds(seed)
    tasks, heldout = make_piv_tasks(data_root=job["data_dir"], batch_size=16, order="ascending", include_heldout=True, seed=seed)

    model = build_fno2d(
        modes=8, width=24, device=device, modular=True, latent_dim=32, adapter_bottleneck=16,
        use_adapters=cfg["use_adapters"], use_contrastive_encoder=cfg["use_contrastive"], route_temperature=0.14,
    )
    trainer = MEWCFlowTrainer(
        model,
        lr=1e-3,
        epochs=epochs,
        consolidation_mode=cfg["cons"],
        lambda_min=0.01,
        lambda_max=0.15,
        lambda_static=0.08,
        ema_beta=0.9,
        fisher_decay=0.85,
        n_fisher=148,
        use_contrastive=cfg["use_contrastive"],
        alpha_contrastive=0.25,
        temperature=0.10,
        routing_mode="soft",
        use_replay=cfg["use_replay"],
        memory_per_task=mem,
        beta_replay=1.0,
        use_physics_loss=cfg["use_physics"],
        gamma_physics=0.05,
    )
    res = trainer.run_all_tasks(tasks, heldout_tasks=heldout, device=device)

    t0 = tasks[0]
    r0 = multi_window_rollout_eval(
        model, t0["val_ds"], re_norm=t0["re_norm"], n_steps=20, stride=3,
        device=device, routing_mode="soft", dy=t0["dy"], dx=t0["dx"], max_windows=10
    )

    out = {
        "kind": "ablation",
        "name": cfg["name"],
        "seed": seed,
        "avg_forgetting": res["avg_forgetting"],
        "avg_final_mse": res["avg_final_mse"],
        "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
        "heldout_re910_mse": float(res["heldout_mse"][heldout[0]["label"]]),
        "heldout_re1890_mse": float(res["heldout_mse"][heldout[1]["label"]]),
        "rollout_step10": r0["step10_rel_l2"],
        "rollout_step20": r0["step_final_rel_l2"],
        "vorticity_rmse": r0["vorticity_rmse"],
        "vph_025": r0["vph_025"],
        "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
        "replay_buffer_bytes": res["memory_bytes"]["replay_buffer_bytes"],
    }

    # On Full MEWC-Flow, also evaluate all 4 routing modes (zero extra training!)
    if cfg["name"] == "Full MEWC-Flow":
        routing_eval = {}
        for rmode in ["soft", "hard", "uniform"]:
            m_seen, _ = evaluate_all_tasks_row(model, tasks, device=device, routing_mode=rmode)
            m_ho, _ = evaluate_all_tasks_row(model, heldout, device=device, routing_mode=rmode)
            r_t0 = multi_window_rollout_eval(
                model, t0["val_ds"], re_norm=t0["re_norm"], n_steps=20, stride=3,
                device=device, routing_mode=rmode, dy=t0["dy"], dx=t0["dx"], max_windows=10
            )
            routing_eval[rmode] = {
                "avg_seen_mse": float(np.mean(m_seen)),
                "heldout_mean_mse": float(np.mean(m_ho)),
                "heldout_re910_mse": float(m_ho[0]),
                "heldout_re1890_mse": float(m_ho[1]),
                "rollout_step10": r_t0["step10_rel_l2"],
                "vorticity_rmse": r_t0["vorticity_rmse"],
            }
        # Oracle routing on seen tasks (ground-truth task ID)
        oracle_seen = []
        for tid, t in enumerate(tasks):
            from mewc_flow.trainers.base_trainer import evaluate_on_task
            oracle_seen.append(evaluate_on_task(model, t, device=device, task_id=tid))
        r_or = multi_window_rollout_eval(
            model, t0["val_ds"], re_norm=t0["re_norm"], n_steps=20, stride=3,
            device=device, task_id=0, dy=t0["dy"], dx=t0["dx"], max_windows=10
        )
        routing_eval["oracle"] = {
            "avg_seen_mse": float(np.mean(oracle_seen)),
            "heldout_mean_mse": None,
            "rollout_step10": r_or["step10_rel_l2"],
            "vorticity_rmse": r_or["vorticity_rmse"],
        }
        out["routing_ablation"] = routing_eval

    print(f"  [Ablation][{cfg['name']:<26}][s={seed}] Forget={out['avg_forgetting']:+.5f} | "
          f"Final={out['avg_final_mse']:.5f} | R10={out['rollout_step10']:.4f} | "
          f"Vort={out['vorticity_rmse']:.4f} | HO={out['heldout_mean_mse']:.5f}", flush=True)
    return out


def _run_order_job(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job["num_threads"])
    order = job["order"]
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]

    set_all_seeds(seed)
    tasks, heldout = make_piv_tasks(data_root=job["data_dir"], batch_size=16, order=order, include_heldout=True, seed=seed)

    if method == "Fine-Tuning":
        m = build_fno2d(modes=8, width=24, device=device)
        res = run_finetuning(m, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "ER":
        m = build_fno2d(modes=8, width=24, device=device)
        res = run_experience_replay(m, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=16, device=device)
    else:
        m = build_fno2d(modes=8, width=24, device=device, modular=True, latent_dim=32, adapter_bottleneck=16, route_temperature=0.14)
        res = MEWCFlowTrainer(
            m, lr=1e-3, epochs=epochs, consolidation_mode="adaptive",
            lambda_min=0.01, lambda_max=0.15, use_contrastive=True, alpha_contrastive=0.25,
            use_replay=True, memory_per_task=16, use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

    print(f"  [Order][{order:<10}][{method:<11}][s={seed}] Forget={res['avg_forgetting']:+.5f} | "
          f"Final={res['avg_final_mse']:.5f} | HO={np.mean(list(res['heldout_mse'].values())):.5f}", flush=True)
    return {
        "kind": "order",
        "order": order,
        "method": method,
        "seed": seed,
        "avg_forgetting": res["avg_forgetting"],
        "avg_final_mse": res["avg_final_mse"],
        "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
    }


def _run_budget_job(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job["num_threads"])
    mem = job["memory_per_task"]
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]

    set_all_seeds(seed)
    tasks, heldout = make_piv_tasks(data_root=job["data_dir"], batch_size=16, order="ascending", include_heldout=True, seed=seed)

    if method == "ER":
        m = build_fno2d(modes=8, width=24, device=device)
        res = run_experience_replay(m, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    elif method == "Replay-TS":
        m = build_fno2d(modes=8, width=24, device=device)
        res = run_replay_ts(m, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    else:
        m = build_fno2d(modes=8, width=24, device=device, modular=True, latent_dim=32, adapter_bottleneck=16, route_temperature=0.14)
        res = MEWCFlowTrainer(
            m, lr=1e-3, epochs=epochs, consolidation_mode="adaptive",
            lambda_min=0.01, lambda_max=0.15, use_contrastive=True, alpha_contrastive=0.25,
            use_replay=True, memory_per_task=mem, use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

    t0 = tasks[0]
    r0 = multi_window_rollout_eval(
        m, t0["val_ds"], re_norm=t0["re_norm"], n_steps=20, stride=3,
        device=device, routing_mode="soft", dy=t0["dy"], dx=t0["dx"], max_windows=10
    )
    print(f"  [Budget][M={mem:2d}][{method:<10}][s={seed}] Forget={res['avg_forgetting']:+.5f} | "
          f"Final={res['avg_final_mse']:.5f} | R10={r0['step10_rel_l2']:.4f} | Mem={res['memory_bytes']['extra_cl_mb']:.2f}MB", flush=True)
    return {
        "kind": "budget",
        "memory_per_task": mem,
        "method": method,
        "seed": seed,
        "avg_forgetting": res["avg_forgetting"],
        "avg_final_mse": res["avg_final_mse"],
        "rollout_step10": r0["step10_rel_l2"],
        "vorticity_rmse": r0["vorticity_rmse"],
        "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
        "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seeds", type=str, default="42,102,202")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="./data/piv")
    p.add_argument("--out_dir", type=str, default="./mewc_flow/experiments/piv_experiments/results")
    args = p.parse_args()

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    workers = 1 if device != "cpu" else args.workers
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    n_threads = max(1, (os.cpu_count() or 4) // max(1, workers))

    print(f"=== PIV Multi-Seed Ablations, Routing, Order Sensitivity & Memory Sweep (device={device}, seeds={seeds}) ===", flush=True)
    abl_jobs, ord_jobs, bud_jobs = [], [], []

    for s in seeds:
        for cfg in ABLATION_CONFIGS:
            abl_jobs.append({"cfg": cfg, "seed": s, "epochs": args.epochs, "memory_per_task": 16, "data_dir": args.data_dir, "device": device, "num_threads": n_threads})
        for order in ["ascending", "descending", "shuffled"]:
            for m in ["Fine-Tuning", "ER", "MEWC-Flow"]:
                ord_jobs.append({"order": order, "method": m, "seed": s, "epochs": args.epochs, "data_dir": args.data_dir, "device": device, "num_threads": n_threads})
        for mem in [4, 8, 16, 32]:
            for m in ["ER", "Replay-TS", "MEWC-Flow"]:
                bud_jobs.append({"memory_per_task": mem, "method": m, "seed": s, "epochs": args.epochs, "data_dir": args.data_dir, "device": device, "num_threads": n_threads})

    abl_raw, ord_raw, bud_raw = [], [], []
    if workers == 1:
        for j in abl_jobs:
            abl_raw.append(_run_ablation_job(j))
        for j in ord_jobs:
            ord_raw.append(_run_order_job(j))
        for j in bud_jobs:
            bud_raw.append(_run_budget_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = (
                [ex.submit(_run_ablation_job, j) for j in abl_jobs]
                + [ex.submit(_run_order_job, j) for j in ord_jobs]
                + [ex.submit(_run_budget_job, j) for j in bud_jobs]
            )
            for f in as_completed(futs):
                r = f.result()
                if r["kind"] == "ablation":
                    abl_raw.append(r)
                elif r["kind"] == "order":
                    ord_raw.append(r)
                else:
                    bud_raw.append(r)

    # Aggregate ablations
    abl_summary = {}
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        rows = [r for r in abl_raw if r["name"] == name]
        abl_summary[name] = {
            k: summarize_seeds([r[k] for r in rows])
            for k in ["avg_forgetting", "avg_final_mse", "heldout_mean_mse", "heldout_re910_mse",
                      "heldout_re1890_mse", "rollout_step10", "rollout_step20", "vorticity_rmse", "vph_025"]
        }
        abl_summary[name]["extra_cl_mb"] = rows[0]["extra_cl_mb"]
        abl_summary[name]["replay_buffer_bytes"] = rows[0]["replay_buffer_bytes"]

    # Aggregate routing modes
    full_rows = [r for r in abl_raw if r["name"] == "Full MEWC-Flow"]
    routing_summary = {}
    for rmode in ["soft", "hard", "uniform", "oracle"]:
        routing_summary[rmode] = {
            "avg_seen_mse": summarize_seeds([r["routing_ablation"][rmode]["avg_seen_mse"] for r in full_rows]),
            "rollout_step10": summarize_seeds([r["routing_ablation"][rmode]["rollout_step10"] for r in full_rows]),
            "vorticity_rmse": summarize_seeds([r["routing_ablation"][rmode]["vorticity_rmse"] for r in full_rows]),
        }
        if rmode != "oracle":
            routing_summary[rmode]["heldout_mean_mse"] = summarize_seeds(
                [r["routing_ablation"][rmode]["heldout_mean_mse"] for r in full_rows]
            )

    # Aggregate task orders
    order_summary = {}
    for order in ["ascending", "descending", "shuffled"]:
        order_summary[order] = {}
        for m in ["Fine-Tuning", "ER", "MEWC-Flow"]:
            rows = [r for r in ord_raw if r["order"] == order and r["method"] == m]
            order_summary[order][m] = {
                k: summarize_seeds([r[k] for r in rows])
                for k in ["avg_forgetting", "avg_final_mse", "heldout_mean_mse"]
            }

    # Aggregate memory budgets
    budget_summary = {}
    for mem in [4, 8, 16, 32]:
        budget_summary[str(mem)] = {}
        for m in ["ER", "Replay-TS", "MEWC-Flow"]:
            rows = [r for r in bud_raw if r["memory_per_task"] == mem and r["method"] == m]
            budget_summary[str(mem)][m] = {
                k: summarize_seeds([r[k] for r in rows])
                for k in ["avg_forgetting", "avg_final_mse", "rollout_step10", "vorticity_rmse", "heldout_mean_mse"]
            }
            budget_summary[str(mem)][m]["extra_cl_mb"] = rows[0]["extra_cl_mb"]

    full_out = {
        "seeds": seeds,
        "epochs_per_task": args.epochs,
        "ablations": abl_summary,
        "routing_modes": routing_summary,
        "task_order_sensitivity": order_summary,
        "memory_budget_sweep": budget_summary,
    }
    out_path = os.path.join(args.out_dir, "ablation_study_summary.json")
    save_json(full_out, out_path)
    print(f"\nSaved full PIV ablation & sensitivity summary -> {out_path}")


if __name__ == "__main__":
    main()
