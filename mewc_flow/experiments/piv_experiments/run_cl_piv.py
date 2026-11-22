"""
run_cl_piv.py
=============
PhD / Q1-Grade Multi-Seed Continual Learning Benchmark on Real Experimental
PIV Cylinder Wake Data (CaltechDATA / Scientific Data, DOI: 10.22002/g2z1f-nex65).

Evaluates 8 methods under matched architecture, optimizer, gradient clipping,
and fixed deterministic seeds across 4 sequential Reynolds regimes
(Re = 240 -> 650 -> 1280 -> 2520) + 2 held-out zero-shot interpolation regimes
(Re = 910, 1890):
  1. Fine-Tuning (Lower bound)
  2. Joint Training (Upper bound)
  3. Static EWC (Kirkpatrick et al., 2017)
  4. Online EWC / OFU (Schwarz et al., 2018)
  5. Experience Replay (ER — Rolnick et al., 2018)
  6. Replay-TS (Hemati et al., 2026)
  7. SLE-FNO (Elhadidy et al., 2026)
  8. MEWC-Flow (Proposed: Modular Adapters + Physics-Only Contrastive Router +
                Complex A-EWC/OFU + Compact Replay + Vorticity/Div Consistency)

Computes:
- Mean +/- Std, 95% Bootstrap CIs, Paired t-test & Wilcoxon p-values, Cohen's d
- Full T x T MSE & Rel-L2 matrices (Forgetting, BWT, FWT, Avg Final MSE)
- Multi-window 20-step rollouts across ALL regimes + Valid Prediction Horizon (VPH)
- Physical diagnostics: Anisotropic Vorticity RMSE, Divergence RMSE, KE/Enstrophy error,
  2D radially-averaged Energy Spectrum E(k) band errors, Wake shedding frequency error,
  Spatial slice error (Near-Wake / Far-Wake / Free-Stream)
- Task-agnostic physics-only regime classification accuracy & Silhouette score
- Exact byte-level memory budget (MB) and wall-clock training time (s)
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import silhouette_score, confusion_matrix

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.piv_cylinder_dataset import make_piv_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import (
    run_finetuning, run_joint_training, run_experience_replay, run_replay_ts, run_sle_fno
)
from mewc_flow.utils.physics_metrics import multi_window_rollout_eval, slice_error_analysis_piv
from mewc_flow.utils.stats import summarize_seeds, paired_significance
from mewc_flow.utils.results_io import save_json, save_checkpoint


METHODS = [
    "Fine-Tuning",
    "EWC",
    "Online-EWC",
    "ER",
    "Replay-TS",
    "SLE-FNO",
    "MEWC-Flow",
    "Joint",
]


@torch.no_grad()
def evaluate_regime_identification(model, tasks, heldout_tasks, device="cpu") -> Dict:
    """Strictly task-agnostic, zero-Re-leakage regime classification & manifold geometry."""
    if not hasattr(model, "encode_regime") or model.regime_encoder is None:
        return {}
    model.eval()
    protos = model.prototypes.to(device)  # (K, d)
    all_z, all_true, all_pred = [], [], []
    routing_weights_by_regime = {}

    for tid, t in enumerate(tasks):
        zs, ws = [], []
        for u_in, re_enc, _ in t["val_loader"]:
            u_in = u_in.to(device)
            re_enc = re_enc.to(device)
            z = model.encode_regime(u_in, dy=t["dy"], dx=t["dx"])
            w = model.compute_routing_weights(z, re_enc, len(model.adapters), routing_mode="soft")
            zs.append(z.cpu().numpy())
            ws.append(w.cpu().numpy())
        Z_t = np.concatenate(zs, axis=0)
        W_t = np.concatenate(ws, axis=0)
        sims = Z_t @ protos.cpu().numpy().T
        preds = np.argmax(sims, axis=1)
        all_z.append(Z_t)
        all_true.extend([tid] * len(Z_t))
        all_pred.extend(preds.tolist())
        routing_weights_by_regime[f"Re={t['re']}"] = W_t.mean(axis=0).tolist()

    for ht in heldout_tasks:
        ws = []
        for u_in, re_enc, _ in ht["val_loader"]:
            u_in = u_in.to(device)
            re_enc = re_enc.to(device)
            z = model.encode_regime(u_in, dy=ht["dy"], dx=ht["dx"])
            w = model.compute_routing_weights(z, re_enc, len(model.adapters), routing_mode="soft")
            ws.append(w.cpu().numpy())
        W_h = np.concatenate(ws, axis=0)
        routing_weights_by_regime[f"Re={ht['re']} (Held-out)"] = W_h.mean(axis=0).tolist()

    Z_all = np.concatenate(all_z, axis=0)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc = float(np.mean(y_true == y_pred))
    sil = float(silhouette_score(Z_all, y_true)) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(tasks)))).tolist()
    return {
        "regime_id_accuracy": acc,
        "silhouette_score": sil,
        "confusion_matrix": cm,
        "routing_weights_mean": routing_weights_by_regime,
    }


def _run_one_seed_method(job: Dict) -> Dict:
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(job.get("num_threads", 2))
    method = job["method"]
    seed = job["seed"]
    epochs = job["epochs"]
    mem_per_task = job["memory_per_task"]
    data_dir = job["data_dir"]
    out_dir = job["out_dir"]
    save_ckpt = job["save_ckpt"]

    set_all_seeds(seed)
    tasks, heldout = make_piv_tasks(
        data_root=data_dir, batch_size=16, order="ascending", include_heldout=True, seed=seed
    )

    if method == "Fine-Tuning":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = run_finetuning(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "Joint":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = run_joint_training(model, tasks, heldout, epochs=epochs, lr=1e-3, batch_size=16, device=device, seed=seed)
    elif method == "EWC":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=148, online_decay=None).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "Online-EWC":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = EWCTrainer(model, lambda_ewc=2.0, lr=1e-3, epochs=epochs, n_fisher=148, online_decay=0.85).run_all_tasks(
            tasks, heldout_tasks=heldout, device=device
        )
    elif method == "ER":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = run_experience_replay(
            model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem_per_task, beta_replay=1.0, device=device
        )
    elif method == "Replay-TS":
        model = build_fno2d(modes=8, width=24, device=device, modular=False)
        res = run_replay_ts(
            model, tasks, heldout, epochs=epochs, lr=1e-3, memory_per_task=mem_per_task, beta_replay=1.0, device=device
        )
    elif method == "SLE-FNO":
        model = build_fno2d(modes=8, width=24, device=device, modular=True, use_adapters=True, use_contrastive_encoder=False)
        res = run_sle_fno(model, tasks, heldout, epochs=epochs, lr=1e-3, device=device)
    elif method == "MEWC-Flow":
        model = build_fno2d(
            modes=8, width=24, device=device, modular=True, latent_dim=32,
            adapter_bottleneck=16, use_adapters=True, use_contrastive_encoder=True, route_temperature=0.14,
        )
        trainer = MEWCFlowTrainer(
            model,
            lr=1e-3,
            epochs=epochs,
            consolidation_mode="adaptive",
            lambda_min=0.01,
            lambda_max=0.15,
            ema_beta=0.9,
            fisher_decay=0.85,
            n_fisher=148,
            use_contrastive=True,
            alpha_contrastive=0.25,
            temperature=0.10,
            routing_mode="soft",
            use_replay=True,
            memory_per_task=mem_per_task,
            beta_replay=1.0,
            use_physics_loss=True,
            gamma_physics=0.05,
        )
        res = trainer.run_all_tasks(tasks, heldout_tasks=heldout, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Multi-window 20-step rollouts & physical verification on ALL 4 seen regimes + Task 0 detailed
    rollout_by_task = {}
    for tid, t in enumerate(tasks):
        r_eval = multi_window_rollout_eval(
            model, t["val_ds"], re_norm=t["re_norm"], n_steps=20, stride=3,
            device=device, routing_mode="soft", dy=t["dy"], dx=t["dx"], periodic=False, max_windows=10
        )
        slices = slice_error_analysis_piv(r_eval["spatial_rmse_map"])
        rollout_by_task[f"Re={t['re']}"] = {
            "n_windows": r_eval["n_windows"],
            "rel_l2_mean": r_eval["rel_l2_mean"],
            "rel_l2_std": r_eval["rel_l2_std"],
            "step1_rel_l2": r_eval["step1_rel_l2"],
            "step5_rel_l2": r_eval["step5_rel_l2"],
            "step10_rel_l2": r_eval["step10_rel_l2"],
            "step20_rel_l2": r_eval["step_final_rel_l2"],
            "vph_025": r_eval["vph_025"],
            "vph_050": r_eval["vph_050"],
            "vorticity_rmse": r_eval["vorticity_rmse"],
            "divergence_rmse": r_eval["divergence_rmse"],
            "ke_rel_err": r_eval["ke_rel_err"],
            "enstrophy_rel_err": r_eval["enstrophy_rel_err"],
            "log_spec_rmse_all": r_eval["spectral"]["log_spec_rmse_all"],
            "log_spec_rmse_low": r_eval["spectral"]["log_spec_rmse_low"],
            "log_spec_rmse_mid": r_eval["spectral"]["log_spec_rmse_mid"],
            "log_spec_rmse_high": r_eval["spectral"]["log_spec_rmse_high"],
            "dominant_freq_rel_err": r_eval["temporal_freq"]["dominant_freq_rel_err"],
            "temporal_spectrum_corr": r_eval["temporal_freq"]["temporal_spectrum_corr"],
            "spatial_slices": slices,
        }

    # Aggregate rollout metrics across all 4 seen regimes
    t0_key = f"Re={tasks[0]['re']}"
    res["rollout_task0"] = rollout_by_task[t0_key]
    res["rollout_all_tasks"] = rollout_by_task
    res["mean_rollout_step10"] = float(np.mean([v["step10_rel_l2"] for v in rollout_by_task.values()]))
    res["mean_rollout_step20"] = float(np.mean([v["step20_rel_l2"] for v in rollout_by_task.values()]))
    res["mean_vorticity_rmse"] = float(np.mean([v["vorticity_rmse"] for v in rollout_by_task.values()]))
    res["mean_divergence_rmse"] = float(np.mean([v["divergence_rmse"] for v in rollout_by_task.values()]))
    res["mean_vph_025"] = float(np.mean([v["vph_025"] for v in rollout_by_task.values()]))
    res["mean_log_spec_rmse"] = float(np.mean([v["log_spec_rmse_all"] for v in rollout_by_task.values()]))

    # Held-out mean MSE
    res["heldout_mean_mse"] = float(np.mean(list(res["heldout_mse"].values())))
    res["heldout_re910_mse"] = float(res["heldout_mse"][heldout[0]["label"]])
    res["heldout_re1890_mse"] = float(res["heldout_mse"][heldout[1]["label"]])

    # Physics-only regime identification (for MEWC-Flow)
    if method == "MEWC-Flow":
        res["regime_id"] = evaluate_regime_identification(model, tasks, heldout, device=device)

    # Save canonical checkpoint for seed 0 (used for high-res field contour & manifold plots)
    if save_ckpt:
        tag = method.lower().replace("-", "_")
        save_checkpoint(model, out_dir, filename=f"model_{tag}.pt")

    # Strip bulky per-epoch histories before returning over IPC
    res.pop("per_task_history", None)
    res.pop("histories", None)
    res["method"] = method
    res["seed"] = seed
    print(f"  [PIV][{method:<11}][seed={seed}] FinalMSE={res['avg_final_mse']:.6f} | "
          f"Forget={res['avg_forgetting']:+.6f} | T0_r10={res['rollout_task0']['step10_rel_l2']:.4f} | "
          f"Vort={res['rollout_task0']['vorticity_rmse']:.4f} | HO910={res['heldout_re910_mse']:.5f} | "
          f"Mem={res['memory_bytes']['extra_cl_mb']:.2f}MB ({res['wall_clock_sec']:.1f}s)", flush=True)
    return res


def aggregate_benchmark(runs_by_method: Dict[str, List[Dict]], seeds: List[int]) -> Dict:
    scalar_keys = [
        "avg_forgetting", "bwt", "fwt", "avg_final_mse", "avg_final_rel_l2",
        "task0_degradation_pct", "heldout_mean_mse", "heldout_re910_mse", "heldout_re1890_mse",
        "mean_rollout_step10", "mean_rollout_step20", "mean_vorticity_rmse",
        "mean_divergence_rmse", "mean_vph_025", "mean_log_spec_rmse", "wall_clock_sec",
    ]
    t0_keys = [
        "step1_rel_l2", "step5_rel_l2", "step10_rel_l2", "step20_rel_l2",
        "vph_025", "vph_050", "vorticity_rmse", "divergence_rmse",
        "ke_rel_err", "enstrophy_rel_err", "log_spec_rmse_all",
        "log_spec_rmse_low", "log_spec_rmse_mid", "log_spec_rmse_high",
        "dominant_freq_rel_err", "temporal_spectrum_corr",
    ]

    summary = {}
    for m, runs in runs_by_method.items():
        runs_sorted = sorted(runs, key=lambda r: r["seed"])
        m_sum = {k: summarize_seeds([r[k] for r in runs_sorted]) for k in scalar_keys}
        m_sum["task0_rollout"] = {
            k: summarize_seeds([r["rollout_task0"][k] for r in runs_sorted]) for k in t0_keys
        }
        # Mean +/- std rollout curve on Task 0
        curves = np.array([r["rollout_task0"]["rel_l2_mean"] for r in runs_sorted])
        m_sum["task0_rollout_curve_mean"] = curves.mean(axis=0).tolist()
        m_sum["task0_rollout_curve_std"] = curves.std(axis=0, ddof=1).tolist() if len(curves) > 1 else np.zeros(curves.shape[1]).tolist()

        # Mean evaluation matrix across seeds
        mats = np.array([r["eval_matrix"] for r in runs_sorted])
        m_sum["eval_matrix_mean"] = mats.mean(axis=0).tolist()
        m_sum["eval_matrix_std"] = mats.std(axis=0, ddof=1).tolist() if len(mats) > 1 else np.zeros_like(mats[0]).tolist()

        # Spatial slices on Task 0
        for sk in ["near_wake_rmse", "far_wake_rmse", "free_stream_rmse"]:
            m_sum[f"slice_{sk}"] = summarize_seeds([r["rollout_task0"]["spatial_slices"][sk] for r in runs_sorted])

        m_sum["memory_bytes"] = runs_sorted[0]["memory_bytes"]
        if "regime_id" in runs_sorted[0]:
            m_sum["regime_id_accuracy"] = summarize_seeds([r["regime_id"]["regime_id_accuracy"] for r in runs_sorted])
            m_sum["silhouette_score"] = summarize_seeds([r["regime_id"]["silhouette_score"] for r in runs_sorted])
            m_sum["confusion_matrix_seed0"] = runs_sorted[0]["regime_id"]["confusion_matrix"]
            m_sum["routing_weights_seed0"] = runs_sorted[0]["regime_id"]["routing_weights_mean"]
        if "aewc_trace" in runs_sorted[0] and runs_sorted[0]["aewc_trace"]:
            m_sum["aewc_trace_seed0"] = runs_sorted[0]["aewc_trace"]
        summary[m] = m_sum

    # Paired statistical significance: MEWC-Flow vs every baseline across matched seeds
    sig_tests = {}
    if "MEWC-Flow" in runs_by_method:
        mewc_runs = sorted(runs_by_method["MEWC-Flow"], key=lambda r: r["seed"])
        for base in METHODS:
            if base == "MEWC-Flow" or base not in runs_by_method:
                continue
            b_runs = sorted(runs_by_method[base], key=lambda r: r["seed"])
            sig_tests[f"MEWC-Flow_vs_{base}"] = {
                "avg_forgetting": paired_significance([r["avg_forgetting"] for r in mewc_runs], [r["avg_forgetting"] for r in b_runs]),
                "avg_final_mse": paired_significance([r["avg_final_mse"] for r in mewc_runs], [r["avg_final_mse"] for r in b_runs]),
                "task0_step10_rel_l2": paired_significance([r["rollout_task0"]["step10_rel_l2"] for r in mewc_runs], [r["rollout_task0"]["step10_rel_l2"] for r in b_runs]),
                "task0_vorticity_rmse": paired_significance([r["rollout_task0"]["vorticity_rmse"] for r in mewc_runs], [r["rollout_task0"]["vorticity_rmse"] for r in b_runs]),
                "heldout_mean_mse": paired_significance([r["heldout_mean_mse"] for r in mewc_runs], [r["heldout_mean_mse"] for r in b_runs]),
            }

    return {"seeds": seeds, "methods": summary, "significance_tests": sig_tests}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--memory_per_task", type=int, default=16)
    p.add_argument("--seeds", type=str, default="42,102,202,302,402")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--data_dir", type=str, default="./data/piv")
    p.add_argument("--out_dir", type=str, default="./mewc_flow/experiments/piv_experiments/results")
    args = p.parse_args()

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    workers = 1 if device != "cpu" else args.workers
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== PIV Q1 Multi-Seed Benchmark: {len(METHODS)} methods x {len(seeds)} seeds (device={device}, epochs={args.epochs}, mem={args.memory_per_task}) ===", flush=True)

    jobs = []
    for seed in seeds:
        for m in METHODS:
            jobs.append({
                "method": m,
                "seed": seed,
                "epochs": args.epochs,
                "memory_per_task": args.memory_per_task,
                "data_dir": args.data_dir,
                "out_dir": args.out_dir,
                "device": device,
                "save_ckpt": (seed == seeds[0]),
                "num_threads": max(1, (os.cpu_count() or 4) // max(1, workers)),
            })

    runs_by_method: Dict[str, List[Dict]] = {m: [] for m in METHODS}
    if workers == 1:
        for j in jobs:
            res = _run_one_seed_method(j)
            runs_by_method[res["method"]].append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one_seed_method, j) for j in jobs]
            for f in as_completed(futs):
                res = f.result()
                runs_by_method[res["method"]].append(res)

    agg = aggregate_benchmark(runs_by_method, seeds)
    agg["dataset"] = "Real Experimental PIV Cylinder Wake (CaltechDATA, DOI: 10.22002/g2z1f-nex65)"
    agg["epochs_per_task"] = args.epochs
    agg["memory_per_task"] = args.memory_per_task

    out_path = os.path.join(args.out_dir, "piv_benchmark_summary.json")
    save_json(agg, out_path)
    print(f"\nSaved full PIV Q1 multi-seed benchmark -> {out_path}")

    print("\n=== SUMMARY (Mean +/- Std across seeds) ===")
    print(f"{'Method':<12} | {'Forgetting':<19} | {'Final MSE':<19} | {'T0 Roll10':<15} | {'T0 VortRMSE':<15} | {'Heldout MSE':<15} | {'Extra MB'}")
    print("-" * 116)
    for m in METHODS:
        d = agg["methods"][m]
        print(
            f"{m:<12} | {d['avg_forgetting']['mean']:+.5f}+/-{d['avg_forgetting']['std']:.5f} | "
            f"{d['avg_final_mse']['mean']:.5f}+/-{d['avg_final_mse']['std']:.5f} | "
            f"{d['task0_rollout']['step10_rel_l2']['mean']:.4f}+/-{d['task0_rollout']['step10_rel_l2']['std']:.4f} | "
            f"{d['task0_rollout']['vorticity_rmse']['mean']:.4f}+/-{d['task0_rollout']['vorticity_rmse']['std']:.4f} | "
            f"{d['heldout_mean_mse']['mean']:.5f}+/-{d['heldout_mean_mse']['std']:.5f} | "
            f"{d['memory_bytes']['extra_cl_mb']:6.2f}"
        )


if __name__ == "__main__":
    main()
