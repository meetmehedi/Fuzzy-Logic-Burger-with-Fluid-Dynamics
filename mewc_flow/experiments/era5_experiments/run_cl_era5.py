"""
run_cl_era5.py
==============
Benchmark 3: Real-World Planetary Atmospheric Wind Fields (ECMWF ERA5 / WeatherBench 2).

Fixes over previous run:
1. Zero-leakage fixed physical wind scaling (U_REF = 10.0 m/s) preserving true
   hemispheric storm-track kinetic energy shifts.
2. Proper convergence (25 epochs/task) so catastrophic forgetting on seasonal
   circulation regimes is clearly measurable.
3. Evaluates ALL 8 CL methods across 3 fixed seeds + saves seed-0 checkpoints.
4. Evaluates 100% of held-out 2019 frames (123 transitions each, not 24!).
5. Multi-window 12-step (72-hour) rollouts, vorticity RMSE, 2D energy spectrum,
   AND hemispheric slice-based spatial error analysis (NH Storm Track vs Tropics vs SH).
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

from mewc_flow.data.era5_dataset import make_era5_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import (
    run_finetuning, run_joint_training, run_experience_replay, run_replay_ts, run_sle_fno
)
from mewc_flow.utils.physics_metrics import multi_window_rollout_eval, slice_error_analysis_era5
from mewc_flow.utils.stats import summarize_seeds, paired_significance
from mewc_flow.utils.results_io import save_json, save_checkpoint


METHODS = ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"]


def _run_job(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job["num_threads"])
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]
    mem = job["memory_per_task"]

    set_all_seeds(seed)
    tasks, heldout = make_era5_tasks(data_root=job["data_dir"], batch_size=16, order="ascending", seed=seed)

    if method == "Fine-Tuning":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_finetuning(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "Joint":
        model = build_fno2d(modes=8, width=24, device=device)
        res = run_joint_training(model, tasks, heldout, epochs=epochs, lr=1e-3, batch_size=16, device=device, seed=seed)
    elif method == "EWC":
        model = build_fno2d(modes=8, width=24, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=112, online_decay=None).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "Online-EWC":
        model = build_fno2d(modes=8, width=24, device=device)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=112, online_decay=0.85).run_all_tasks(
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
            lambda_min=0.01, lambda_max=0.15, ema_beta=0.9, fisher_decay=0.85, n_fisher=112,
            use_contrastive=True, alpha_contrastive=0.25, temperature=0.10, routing_mode="soft",
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)
    else:
        raise ValueError(method)

    # Multi-window 12-step (72-hour) rollouts on Task 0 (Summer) + all 4 tasks + hemispheric slices
    r_all = {}
    for t in tasks:
        r_ev = multi_window_rollout_eval(
            model, t["val_ds"], re_norm=t["re_norm"], n_steps=12, stride=2,
            device=device, routing_mode="soft", dy=t["dy"], dx=t["dx"], max_windows=10,
        )
        r_all[t["name"]] = {
            "rel_l2_mean": r_ev["rel_l2_mean"],
            "step4_rel_l2": float(r_ev["rel_l2_mean"][min(3, len(r_ev["rel_l2_mean"]) - 1)]),
            "step12_rel_l2": r_ev["step_final_rel_l2"],
            "vorticity_rmse": r_ev["vorticity_rmse"],
            "log_spec_rmse_all": r_ev["spectral"]["log_spec_rmse_all"],
            "hemispheric_slices": slice_error_analysis_era5(r_ev["spatial_rmse_map"]),
        }

    # Physics-only regime ID on ERA5
    regime_acc, sil = None, None
    if method == "MEWC-Flow":
        model.eval()
        protos = model.prototypes.cpu().numpy()
        zs_all, yt, yp = [], [], []
        for tid, t in enumerate(tasks):
            with torch.no_grad():
                for u_in, _, _ in t["val_loader"]:
                    z = model.encode_regime(u_in.to(device)).cpu().numpy()
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
    res["heldout_summer_mse"] = float(res["heldout_mse"][heldout[0]["label"]])
    res["heldout_winter_mse"] = float(res["heldout_mse"][heldout[1]["label"]])
    res["heldout_mean_mse"] = float(np.mean(list(res["heldout_mse"].values())))
    res["mean_rollout_72h"] = float(np.mean([v["step12_rel_l2"] for v in r_all.values()]))
    res["summer_rollout_72h"] = float(r_all["summer"]["step12_rel_l2"])
    res["mean_vorticity_rmse"] = float(np.mean([v["vorticity_rmse"] for v in r_all.values()]))
    res["nh_storm_rmse"] = float(np.mean([v["hemispheric_slices"]["nh_storm_track_rmse"] for v in r_all.values()]))
    res["tropics_rmse"] = float(np.mean([v["hemispheric_slices"]["tropics_rmse"] for v in r_all.values()]))
    res["sh_midlat_rmse"] = float(np.mean([v["hemispheric_slices"]["sh_midlat_rmse"] for v in r_all.values()]))
    res["rollout_by_season"] = r_all
    if regime_acc is not None:
        res["regime_id_accuracy"] = regime_acc
        res["silhouette_score"] = sil

    print(f"  [ERA5][{method:<11}][s={seed}] FinalMSE={res['avg_final_mse']:.5f} | "
          f"Forget={res['avg_forgetting']:+.5f} | R72h={res['mean_rollout_72h']:.4f} | "
          f"HO_Sum={res['heldout_summer_mse']:.5f} | HO_Win={res['heldout_winter_mse']:.5f}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--memory_per_task", type=int, default=16)
    p.add_argument("--seeds", type=str, default="42,102,202")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="./data/era5")
    p.add_argument("--out_dir", type=str, default="./mewc_flow/experiments/era5_experiments/results")
    args = p.parse_args()

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    workers = 1 if device != "cpu" else args.workers
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== ERA5 Planetary Wind Benchmark (device={device}, seeds={seeds}, epochs={args.epochs}) ===", flush=True)

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
        "task0_degradation_pct", "heldout_summer_mse", "heldout_winter_mse", "heldout_mean_mse",
        "mean_rollout_72h", "summer_rollout_72h", "mean_vorticity_rmse",
        "nh_storm_rmse", "tropics_rmse", "sh_midlat_rmse", "wall_clock_sec",
    ]
    summary = {}
    for m, runs in by_m.items():
        rs = sorted(runs, key=lambda r: r["seed"])
        ms = {k: summarize_seeds([r[k] for r in rs]) for k in scalar_keys}
        ms["eval_matrix_mean"] = np.mean([r["eval_matrix"] for r in rs], axis=0).tolist()
        ms["memory_bytes"] = rs[0]["memory_bytes"]
        ms["summer_rollout_curve_mean"] = np.mean([r["rollout_by_season"]["summer"]["rel_l2_mean"] for r in rs], axis=0).tolist()
        ms["summer_rollout_curve_std"] = np.std([r["rollout_by_season"]["summer"]["rel_l2_mean"] for r in rs], axis=0, ddof=1).tolist() if len(rs) > 1 else [0.0] * 12
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
            for k in ["avg_forgetting", "avg_final_mse", "mean_rollout_72h", "heldout_mean_mse"]
        }

    out = {
        "dataset": "ECMWF ERA5 Real Planetary Atmospheric Wind Reanalysis (WeatherBench 2)",
        "seeds": seeds,
        "epochs_per_task": args.epochs,
        "memory_per_task": args.memory_per_task,
        "methods": summary,
        "significance_tests": sig,
    }
    out_path = os.path.join(args.out_dir, "era5_benchmark_summary.json")
    save_json(out, out_path)
    print(f"\nSaved ERA5 planetary Q1 benchmark -> {out_path}")


if __name__ == "__main__":
    main()
