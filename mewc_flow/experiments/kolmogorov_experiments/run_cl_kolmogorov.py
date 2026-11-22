"""
run_cl_kolmogorov.py
====================
Benchmark 2: 2D Incompressible Forced Navier-Stokes (Kolmogorov Flow)
Crossing the Laminar-to-Chaotic Hydrodynamic Bifurcation:
  Re = 20 (Steady Laminar) -> Re = 45 (Bifurcation Onset)
  -> Re = 120 (Transitional Vortex) -> Re = 400 (Chaotic 2D Turbulence)
  Held-out zero-shot interpolation: Re = 75, Re = 240.

Evaluates ALL 8 CL methods across 3 seeds on:
- Catastrophic forgetting across the hydrodynamic bifurcation (G1 / RQ2)
- Exact 2D Navier-Stokes vorticity-transport PDE residual
- Exact incompressibility residual ||div(u)||_rms
- Multi-window 20-step rollout stability & 2D Energy Spectrum E(k)
- Task-agnostic physics-only regime identification accuracy
- Paired statistical tests (MEWC-Flow vs all baselines)
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.kolmogorov_ns_dataset import make_kolmogorov_tasks, kolmogorov_pde_residual
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import (
    run_finetuning, run_joint_training, run_experience_replay, run_replay_ts, run_sle_fno
)
from mewc_flow.utils.physics_metrics import multi_window_rollout_eval
from mewc_flow.utils.stats import summarize_seeds, paired_significance
from mewc_flow.utils.results_io import save_json, save_checkpoint


METHODS = ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"]


@torch.no_grad()
def _eval_ns_pde_residual(model, task, device="cpu", routing_mode="soft") -> float:
    model.eval()
    is_mod = hasattr(model, "adapters")
    u_ins, u_preds = [], []
    for u_in, re_enc, _ in task["val_loader"]:
        u_in_d = u_in.to(device)
        re_d = re_enc.to(device)
        pred = model(u_in_d, re_d, routing_mode=routing_mode, dy=task["dy"], dx=task["dx"]) if is_mod else model(u_in_d, re_d)
        u_ins.append(u_in.numpy())
        u_preds.append(pred.cpu().numpy())
    return kolmogorov_pde_residual(np.concatenate(u_ins, axis=0), np.concatenate(u_preds, axis=0), re=task["re"])


def _run_job(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job["num_threads"])
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]
    mem = job["memory_per_task"]

    set_all_seeds(seed)
    tasks, heldout = make_kolmogorov_tasks(
        data_root=job["data_dir"], batch_size=16, order="ascending", auto_generate=False, seed=seed
    )

    if method == "Fine-Tuning":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_finetuning(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "Joint":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_joint_training(model, tasks, heldout, epochs=epochs, lr=1e-3, batch_size=16, device=device, seed=seed)
    elif method == "EWC":
        model = build_fno2d(modes=8, width=24, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=160, online_decay=None).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "Online-EWC":
        model = build_fno2d(modes=8, width=24, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=160, online_decay=0.85).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "ER":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_experience_replay(model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    elif method == "Replay-TS":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_replay_ts(model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem, device=device)
    elif method == "SLE-FNO":
        model = build_fno2d(modes=8, width=24, device=device, modular=True, use_adapters=True, use_contrastive_encoder=False)
        res = run_sle_fno(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "MEWC-Flow":
        model = build_fno2d(
            modes=8, width=24, device=device, modular=True, latent_dim=32,
            adapter_bottleneck=16, use_adapters=True, use_contrastive_encoder=True, route_temperature=0.14,
        )
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="adaptive",
            lambda_min=0.01, lambda_max=0.15, ema_beta=0.9, fisher_decay=0.85, n_fisher=160,
            use_contrastive=True, alpha_contrastive=0.25, temperature=0.10, routing_mode="soft",
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)
    else:
        raise ValueError(method)

    # Physical verification across all 4 regimes (including laminar Re=20 & post-bifurcation Re=45!)
    pde_residuals, div_rms_list, vort_rmse_list, r10_list, spec_list = [], [], [], [], []
    per_regime_rollout = {}
    for t in tasks:
        pde_r = _eval_ns_pde_residual(model, t, device=device)
        r_ev = multi_window_rollout_eval(
            model, t["val_ds"], re_norm=t["re_norm"], n_steps=20, stride=5,
            device=device, routing_mode="soft", dy=t["dy"], dx=t["dx"], periodic=True, max_windows=8,
        )
        pde_residuals.append(pde_r)
        div_rms_list.append(r_ev["divergence_pred_rms"])
        vort_rmse_list.append(r_ev["vorticity_rmse"])
        r10_list.append(r_ev["step10_rel_l2"])
        spec_list.append(r_ev["spectral"]["log_spec_rmse_all"])
        per_regime_rollout[f"Re={t['re']}"] = {
            "pde_residual": pde_r,
            "div_rms": r_ev["divergence_pred_rms"],
            "vorticity_rmse": r_ev["vorticity_rmse"],
            "step10_rel_l2": r_ev["step10_rel_l2"],
            "step20_rel_l2": r_ev["step_final_rel_l2"],
            "vph_025": r_ev["vph_025"],
            "rel_l2_mean": r_ev["rel_l2_mean"],
            "log_spec_rmse_all": r_ev["spectral"]["log_spec_rmse_all"],
        }

    # Physics-only regime ID for MEWC-Flow
    regime_acc, sil = None, None
    if method == "MEWC-Flow":
        model.eval()
        protos = model.prototypes.cpu().numpy()
        zs_all, yt, yp = [], [], []
        for tid, t in enumerate(tasks):
            with torch.no_grad():
                for u_in, _, _ in t["val_loader"]:
                    z = model.encode_regime(u_in.to(device), dy=t["dy"], dx=t["dx"]).cpu().numpy()
                    zs_all.append(z)
                    yt.extend([tid] * len(z))
                    yp.extend(np.argmax(z @ protos.T, axis=1).tolist())
        Z = np.concatenate(zs_all, axis=0)
        regime_acc = float(np.mean(np.array(yt) == np.array(yp)))
        sil = float(silhouette_score(Z, np.array(yt)))

    if job["save_ckpt"]:
        save_checkpoint(model, job["out_dir"], filename=f"model_{method.lower().replace('-', '_')}.pt")

    res.pop("per_task_history", None)
    res.pop("histories", None)
    res["method"] = method
    res["seed"] = seed
    res["mean_pde_residual"] = float(np.mean(pde_residuals))
    res["mean_div_rms"] = float(np.mean(div_rms_list))
    res["mean_vorticity_rmse"] = float(np.mean(vort_rmse_list))
    res["mean_rollout_step10"] = float(np.mean(r10_list))
    res["mean_log_spec_rmse"] = float(np.mean(spec_list))
    res["task0_rollout_step10"] = float(r10_list[0])
    res["task1_rollout_step10"] = float(r10_list[1])
    res["heldout_mean_mse"] = float(np.mean(list(res["heldout_mse"].values())))
    res["per_regime_rollout"] = per_regime_rollout
    if regime_acc is not None:
        res["regime_id_accuracy"] = regime_acc
        res["silhouette_score"] = sil

    print(f"  [NS-Bifurc][{method:<11}][s={seed}] FinalMSE={res['avg_final_mse']:.5f} | "
          f"Forget={res['avg_forgetting']:+.5f} | PDE_Res={res['mean_pde_residual']:.3f} | "
          f"Div={res['mean_div_rms']:.4f} | R10={res['mean_rollout_step10']:.4f} | HO={res['heldout_mean_mse']:.5f}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--memory_per_task", type=int, default=16)
    p.add_argument("--seeds", type=str, default="42,102,202")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="./data/kolmogorov")
    p.add_argument("--out_dir", type=str, default="./mewc_flow/experiments/kolmogorov_experiments/results")
    args = p.parse_args()

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    workers = 1 if device != "cpu" else args.workers
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    make_kolmogorov_tasks(data_root=args.data_dir, auto_generate=True)
    print(f"=== Kolmogorov 2D NS Bifurcation Benchmark (device={device}, seeds={seeds}, epochs={args.epochs}) ===", flush=True)

    jobs = [
        {
            "method": m, "seed": s, "epochs": args.epochs, "memory_per_task": args.memory_per_task,
            "data_dir": args.data_dir, "out_dir": args.out_dir, "device": device, "save_ckpt": (s == seeds[0]),
            "num_threads": max(1, (os.cpu_count() or 4) // max(1, workers)),
        }
        for s in seeds for m in METHODS
    ]

    by_m: Dict[str, List[Dict]] = {m: [] for m in METHODS}
    if workers == 1:
        for j in jobs:
            r = _run_job(j)
            by_m[r["method"]].append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(_run_job, j) for j in jobs]):
                r = f.result()
                by_m[r["method"]].append(r)

    scalar_keys = [
        "avg_forgetting", "bwt", "fwt", "avg_final_mse", "avg_final_rel_l2",
        "task0_degradation_pct", "heldout_mean_mse", "mean_pde_residual",
        "mean_div_rms", "mean_vorticity_rmse", "mean_rollout_step10",
        "task0_rollout_step10", "task1_rollout_step10", "mean_log_spec_rmse", "wall_clock_sec",
    ]
    summary = {}
    for m, runs in by_m.items():
        rs = sorted(runs, key=lambda r: r["seed"])
        ms = {k: summarize_seeds([r[k] for r in rs]) for k in scalar_keys}
        ms["eval_matrix_mean"] = np.mean([r["eval_matrix"] for r in rs], axis=0).tolist()
        ms["memory_bytes"] = rs[0]["memory_bytes"]
        ms["per_regime_seed0"] = rs[0]["per_regime_rollout"]
        if "regime_id_accuracy" in rs[0]:
            ms["regime_id_accuracy"] = summarize_seeds([r["regime_id_accuracy"] for r in rs])
            ms["silhouette_score"] = summarize_seeds([r["silhouette_score"] for r in rs])
        summary[m] = ms

    sig = {}
    mewc_rs = sorted(by_m["MEWC-Flow"], key=lambda r: r["seed"])
    for b in METHODS:
        if b == "MEWC-Flow":
            continue
        b_rs = sorted(by_m[b], key=lambda r: r["seed"])
        sig[f"MEWC-Flow_vs_{b}"] = {
            k: paired_significance([r[k] for r in mewc_rs], [r[k] for r in b_rs])
            for k in ["avg_forgetting", "avg_final_mse", "mean_pde_residual", "mean_rollout_step10", "heldout_mean_mse"]
        }

    out = {
        "dataset": "2D Incompressible Forced Navier-Stokes (Kolmogorov Bifurcation Re=20->45->120->400)",
        "seeds": seeds,
        "epochs_per_task": args.epochs,
        "memory_per_task": args.memory_per_task,
        "methods": summary,
        "significance_tests": sig,
    }
    out_path = os.path.join(args.out_dir, "kolmogorov_benchmark_summary.json")
    save_json(out, out_path)
    print(f"\nSaved Kolmogorov NS Bifurcation benchmark -> {out_path}")


if __name__ == "__main__":
    main()
