"""
run_fuzzy_benchmark.py
======================
Runs the Neuro-Fuzzy MEWC-Flow (Fuzzy-MEWC-Flow) benchmark across all 4 datasets:
  1. Experimental Caltech PIV Cylinder Wakes (5 seeds: 42,102,202,302,402; 15 epochs; M=16)
  2. 2D Kolmogorov Navier-Stokes Bifurcation (3 seeds: 42,102,202; 12 epochs; M=16)
  3. Real ERA5 Global 10-m Planetary Winds (3 seeds: 42,102,202; 18 epochs; M=16)
  4. 1D Viscous Burgers Shock Sequence (3 seeds: 42,102,202; 20 epochs; M=32)

Compares Fuzzy-MEWC-Flow (Takagi-Sugeno Physics-Only Regime Router + 9-Rule Sugeno
Fuzzy A-EWC Controller) head-to-head against saved neural MEWC-Flow and baselines,
computes paired t-tests & Cohen's d, exports human-readable IF-THEN physical rules,
saves strict JSON -> mewc_flow/experiments/fuzzy_results/fuzzy_mewc_summary.json,
and renders 300-DPI Figure 8 -> fig8_neuro_fuzzy_interpretability.png.
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, silhouette_score

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from mewc_flow.data.piv_cylinder_dataset import make_piv_tasks
from mewc_flow.data.kolmogorov_ns_dataset import make_kolmogorov_tasks, kolmogorov_pde_residual
from mewc_flow.data.era5_dataset import make_era5_tasks
from mewc_flow.data.burgers_dataset import make_burgers_tasks, burgers_pde_residual_torch
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.models.surrogate import build_surrogate
from mewc_flow.models.fuzzy_logic import SugenoFuzzyLambdaController
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.utils.physics_metrics import multi_window_rollout_eval, slice_error_analysis_era5
from mewc_flow.utils.stats import summarize_seeds, paired_significance
from mewc_flow.utils.results_io import save_json


def _load_json(rel_path: str) -> Dict:
    p = os.path.join(ROOT, rel_path)
    return json.load(open(p)) if os.path.exists(p) else {}


@torch.no_grad()
def eval_fuzzy_piv_routing(model, tasks, heldout, device="cpu") -> Dict:
    model.eval()
    all_z, all_true, all_pred = [], [], []
    routing_means = {}
    for tid, t in enumerate(tasks):
        zs, ws = [], []
        for u_in, _, _ in t["val_loader"]:
            u_d = u_in.to(device)
            feats = model.fuzzy_router._standardize(
                model.fuzzy_router.extract_invariants(u_d, dy=t["dy"], dx=t["dx"])
            )
            w = model.fuzzy_router(u_d, dy=t["dy"], dx=t["dx"], n_active=len(model.adapters), routing_mode="soft")
            zs.append(feats.cpu().numpy())
            ws.append(w.cpu().numpy())
        Z_t = np.concatenate(zs, axis=0)
        W_t = np.concatenate(ws, axis=0)
        preds = np.argmax(W_t, axis=1)
        all_z.append(Z_t)
        all_true.extend([tid] * len(Z_t))
        all_pred.extend(preds.tolist())
        routing_means[f"Re={t['re']}"] = W_t.mean(axis=0).tolist()

    for ht in heldout:
        ws = []
        for u_in, _, _ in ht["val_loader"]:
            u_d = u_in.to(device)
            w = model.fuzzy_router(u_d, dy=ht["dy"], dx=ht["dx"], n_active=len(model.adapters), routing_mode="soft")
            ws.append(w.cpu().numpy())
        routing_means[f"Re={ht['re']}*"] = np.concatenate(ws, axis=0).mean(axis=0).tolist()

    Z_all = np.concatenate(all_z, axis=0)
    yt = np.array(all_true)
    yp = np.array(all_pred)
    return {
        "regime_id_accuracy": float(np.mean(yt == yp)),
        "silhouette_score": float(silhouette_score(Z_all, yt)) if len(np.unique(yt)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(len(tasks)))).tolist(),
        "routing_weights_mean": routing_means,
    }


def run_piv_fuzzy(seeds: List[int], epochs: int, mem: int, device: str) -> Dict:
    print(f"=== [1/4] Fuzzy-MEWC-Flow on PIV (seeds={seeds}, epochs={epochs}, M={mem}, dev={device}) ===", flush=True)
    raw = []
    seed0_trace = {}
    seed0_rules = []
    seed0_routing = {}
    for s in seeds:
        set_all_seeds(s)
        tasks, heldout = make_piv_tasks(
            data_root="./data/piv", batch_size=16, order="ascending", include_heldout=True, seed=s
        )
        model = build_fno2d(
            modes=8, width=24, device=device, modular=True,
            adapter_bottleneck=16, use_adapters=True, router_type="fuzzy", tau_fuzzy=1.0,
        )
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="fuzzy_adaptive",
            lambda_min=0.01, lambda_max=0.15, ema_beta=0.9, fisher_decay=0.85, n_fisher=148,
            use_contrastive=False, alpha_contrastive=0.25, routing_mode="soft",
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

        r_eval0 = multi_window_rollout_eval(
            model, tasks[0]["val_ds"], re_norm=tasks[0]["re_norm"], n_steps=20, stride=3,
            device=device, routing_mode="soft", dy=tasks[0]["dy"], dx=tasks[0]["dx"], periodic=False, max_windows=10,
        )
        rid = eval_fuzzy_piv_routing(model, tasks, heldout, device=device)
        ho_mean = float(np.mean(list(res["heldout_mse"].values())))
        entry = {
            "seed": s,
            "avg_forgetting": res["avg_forgetting"],
            "avg_final_mse": res["avg_final_mse"],
            "task0_degradation_pct": res["task0_degradation_pct"],
            "task0_step10_rel_l2": r_eval0["step10_rel_l2"],
            "task0_vorticity_rmse": r_eval0["vorticity_rmse"],
            "task0_vph_025": r_eval0["vph_025"],
            "heldout_mean_mse": ho_mean,
            "regime_id_accuracy": rid["regime_id_accuracy"],
            "silhouette_score": rid["silhouette_score"],
            "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
            "wall_clock_sec": res["wall_clock_sec"],
        }
        raw.append(entry)
        if s == seeds[0]:
            seed0_trace = res["aewc_trace"]
            seed0_rules = model.fuzzy_router.export_linguistic_rules(
                [f"Re={t['re']}" for t in tasks]
            )
            seed0_routing = rid
        print(
            f"  [PIV][Fuzzy-MEWC-Flow][s={s}] Forget={entry['avg_forgetting']:+.2e} | "
            f"Final={entry['avg_final_mse']:.3e} | R10={entry['task0_step10_rel_l2']:.4f} | "
            f"Vort={entry['task0_vorticity_rmse']:.4f} | RegAcc={entry['regime_id_accuracy']*100:.1f}%",
            flush=True,
        )

    piv_ref = _load_json("mewc_flow/experiments/piv_experiments/results/piv_benchmark_summary.json")
    sig = {}
    if piv_ref and "methods" in piv_ref:
        for base in ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"]:
            if base not in piv_ref["methods"]:
                continue
            bm = piv_ref["methods"][base]
            sig[f"Fuzzy-MEWC-Flow_vs_{base}"] = {
                "avg_forgetting": paired_significance([r["avg_forgetting"] for r in raw], bm["avg_forgetting"]["values"]),
                "avg_final_mse": paired_significance([r["avg_final_mse"] for r in raw], bm["avg_final_mse"]["values"]),
                "task0_step10_rel_l2": paired_significance([r["task0_step10_rel_l2"] for r in raw], bm["task0_rollout"]["step10_rel_l2"]["values"]),
                "task0_vorticity_rmse": paired_significance([r["task0_vorticity_rmse"] for r in raw], bm["task0_rollout"]["vorticity_rmse"]["values"]),
                "heldout_mean_mse": paired_significance([r["heldout_mean_mse"] for r in raw], bm["heldout_mean_mse"]["values"]),
            }

    return {
        "seeds": seeds,
        "epochs_per_task": epochs,
        "memory_per_task": mem,
        "avg_forgetting": summarize_seeds([r["avg_forgetting"] for r in raw]),
        "avg_final_mse": summarize_seeds([r["avg_final_mse"] for r in raw]),
        "task0_degradation_pct": summarize_seeds([r["task0_degradation_pct"] for r in raw]),
        "task0_step10_rel_l2": summarize_seeds([r["task0_step10_rel_l2"] for r in raw]),
        "task0_vorticity_rmse": summarize_seeds([r["task0_vorticity_rmse"] for r in raw]),
        "task0_vph_025": summarize_seeds([r["task0_vph_025"] for r in raw]),
        "heldout_mean_mse": summarize_seeds([r["heldout_mean_mse"] for r in raw]),
        "regime_id_accuracy": summarize_seeds([r["regime_id_accuracy"] for r in raw]),
        "silhouette_score": summarize_seeds([r["silhouette_score"] for r in raw]),
        "extra_cl_mb": float(np.mean([r["extra_cl_mb"] for r in raw])),
        "seed0_rules": seed0_rules,
        "seed0_routing": seed0_routing,
        "seed0_trace": seed0_trace,
        "significance_tests": sig,
    }


def run_kolmogorov_fuzzy(seeds: List[int], epochs: int, mem: int, device: str) -> Dict:
    print(f"=== [2/4] Fuzzy-MEWC-Flow on Kolmogorov 2D NS (seeds={seeds}, epochs={epochs}, M={mem}, dev={device}) ===", flush=True)
    raw = []
    for s in seeds:
        set_all_seeds(s)
        tasks, heldout = make_kolmogorov_tasks(
            data_root="./data/kolmogorov", batch_size=16, order="ascending", auto_generate=False, seed=s
        )
        model = build_fno2d(
            modes=8, width=24, device=device, modular=True,
            adapter_bottleneck=16, use_adapters=True, router_type="fuzzy", tau_fuzzy=1.0,
        )
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="fuzzy_adaptive",
            lambda_min=0.01, lambda_max=0.15, ema_beta=0.9, fisher_decay=0.85, n_fisher=160,
            use_contrastive=False, alpha_contrastive=0.25, routing_mode="soft",
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

        model.eval()
        pde_list, div_list, r10_list = [], [], []
        for t in tasks:
            u_ins, u_preds = [], []
            with torch.no_grad():
                for u_in, re_enc, _ in t["val_loader"]:
                    pred = model(u_in.to(device), re_enc.to(device), routing_mode="soft", dy=t["dy"], dx=t["dx"])
                    u_ins.append(u_in.numpy())
                    u_preds.append(pred.cpu().numpy())
            pde_list.append(kolmogorov_pde_residual(np.concatenate(u_ins, 0), np.concatenate(u_preds, 0), re=t["re"]))
            r_ev = multi_window_rollout_eval(
                model, t["val_ds"], re_norm=t["re_norm"], n_steps=20, stride=5,
                device=device, routing_mode="soft", dy=t["dy"], dx=t["dx"], periodic=True, max_windows=8,
            )
            div_list.append(r_ev["divergence_pred_rms"])
            r10_list.append(r_ev["step10_rel_l2"])

        entry = {
            "seed": s,
            "avg_forgetting": res["avg_forgetting"],
            "avg_final_mse": res["avg_final_mse"],
            "mean_pde_residual": float(np.mean(pde_list)),
            "mean_div_rms": float(np.mean(div_list)),
            "mean_rollout_step10": float(np.mean(r10_list)),
            "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
            "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
        }
        raw.append(entry)
        print(
            f"  [Kolmogorov][Fuzzy-MEWC-Flow][s={s}] Forget={entry['avg_forgetting']:+.2e} | "
            f"Final={entry['avg_final_mse']:.3e} | PDE={entry['mean_pde_residual']:.3f} | R10={entry['mean_rollout_step10']:.4f}",
            flush=True,
        )

    kol_ref = _load_json("mewc_flow/experiments/kolmogorov_experiments/results/kolmogorov_benchmark_summary.json")
    sig = {}
    if kol_ref and "methods" in kol_ref:
        for base in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"]:
            if base not in kol_ref["methods"]:
                continue
            bm = kol_ref["methods"][base]
            sig[f"Fuzzy-MEWC-Flow_vs_{base}"] = {
                "avg_forgetting": paired_significance([r["avg_forgetting"] for r in raw], bm["avg_forgetting"]["values"]),
                "avg_final_mse": paired_significance([r["avg_final_mse"] for r in raw], bm["avg_final_mse"]["values"]),
                "mean_pde_residual": paired_significance([r["mean_pde_residual"] for r in raw], bm["mean_pde_residual"]["values"]),
                "mean_rollout_step10": paired_significance([r["mean_rollout_step10"] for r in raw], bm["mean_rollout_step10"]["values"]),
                "heldout_mean_mse": paired_significance([r["heldout_mean_mse"] for r in raw], bm["heldout_mean_mse"]["values"]),
            }

    return {
        "seeds": seeds,
        "epochs_per_task": epochs,
        "memory_per_task": mem,
        "avg_forgetting": summarize_seeds([r["avg_forgetting"] for r in raw]),
        "avg_final_mse": summarize_seeds([r["avg_final_mse"] for r in raw]),
        "mean_pde_residual": summarize_seeds([r["mean_pde_residual"] for r in raw]),
        "mean_div_rms": summarize_seeds([r["mean_div_rms"] for r in raw]),
        "mean_rollout_step10": summarize_seeds([r["mean_rollout_step10"] for r in raw]),
        "heldout_mean_mse": summarize_seeds([r["heldout_mean_mse"] for r in raw]),
        "extra_cl_mb": float(np.mean([r["extra_cl_mb"] for r in raw])),
        "significance_tests": sig,
    }


def run_era5_fuzzy(seeds: List[int], epochs: int, mem: int, device: str) -> Dict:
    print(f"=== [3/4] Fuzzy-MEWC-Flow on ERA5 Winds (seeds={seeds}, epochs={epochs}, M={mem}, dev={device}) ===", flush=True)
    raw = []
    for s in seeds:
        set_all_seeds(s)
        tasks, heldout = make_era5_tasks(data_root="./data/era5", batch_size=16, order="ascending", seed=s)
        model = build_fno2d(
            modes=8, width=24, device=device, modular=True,
            adapter_bottleneck=16, use_adapters=True, router_type="fuzzy", tau_fuzzy=1.0,
        )
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="fuzzy_adaptive",
            lambda_min=0.01, lambda_max=0.15, ema_beta=0.9, fisher_decay=0.85, n_fisher=112,
            use_contrastive=False, alpha_contrastive=0.25, routing_mode="soft",
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.05,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

        r72_list, nh_list = [], []
        for t in tasks:
            r_ev = multi_window_rollout_eval(
                model, t["val_ds"], re_norm=t["re_norm"], n_steps=12, stride=2,
                device=device, routing_mode="soft", dy=t["dy"], dx=t["dx"], max_windows=10,
            )
            r72_list.append(r_ev["step_final_rel_l2"])
            nh_list.append(slice_error_analysis_era5(r_ev["spatial_rmse_map"])["nh_storm_track_rmse"])

        entry = {
            "seed": s,
            "avg_forgetting": res["avg_forgetting"],
            "avg_final_mse": res["avg_final_mse"],
            "mean_rollout_72h": float(np.mean(r72_list)),
            "heldout_summer_mse": float(res["heldout_mse"][heldout[0]["label"]]),
            "heldout_winter_mse": float(res["heldout_mse"][heldout[1]["label"]]),
            "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
            "nh_storm_rmse": float(np.mean(nh_list)),
            "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
        }
        raw.append(entry)
        print(
            f"  [ERA5][Fuzzy-MEWC-Flow][s={s}] Forget={entry['avg_forgetting']:+.2e} | "
            f"Final={entry['avg_final_mse']:.3e} | R72h={entry['mean_rollout_72h']:.4f} | NH={entry['nh_storm_rmse']:.4f}",
            flush=True,
        )

    era_ref = _load_json("mewc_flow/experiments/era5_experiments/results/era5_benchmark_summary.json")
    sig = {}
    if era_ref and "methods" in era_ref:
        for base in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"]:
            if base not in era_ref["methods"]:
                continue
            bm = era_ref["methods"][base]
            sig[f"Fuzzy-MEWC-Flow_vs_{base}"] = {
                "avg_forgetting": paired_significance([r["avg_forgetting"] for r in raw], bm["avg_forgetting"]["values"]),
                "avg_final_mse": paired_significance([r["avg_final_mse"] for r in raw], bm["avg_final_mse"]["values"]),
                "mean_rollout_72h": paired_significance([r["mean_rollout_72h"] for r in raw], bm["mean_rollout_72h"]["values"]),
                "heldout_mean_mse": paired_significance([r["heldout_mean_mse"] for r in raw], bm["heldout_mean_mse"]["values"]),
            }

    return {
        "seeds": seeds,
        "epochs_per_task": epochs,
        "memory_per_task": mem,
        "avg_forgetting": summarize_seeds([r["avg_forgetting"] for r in raw]),
        "avg_final_mse": summarize_seeds([r["avg_final_mse"] for r in raw]),
        "mean_rollout_72h": summarize_seeds([r["mean_rollout_72h"] for r in raw]),
        "heldout_summer_mse": summarize_seeds([r["heldout_summer_mse"] for r in raw]),
        "heldout_winter_mse": summarize_seeds([r["heldout_winter_mse"] for r in raw]),
        "heldout_mean_mse": summarize_seeds([r["heldout_mean_mse"] for r in raw]),
        "nh_storm_rmse": summarize_seeds([r["nh_storm_rmse"] for r in raw]),
        "extra_cl_mb": float(np.mean([r["extra_cl_mb"] for r in raw])),
        "significance_tests": sig,
    }


@torch.no_grad()
def _eval_burgers_rollout_pde(model, task, n_steps=20, device="cpu") -> Dict:
    model.eval()
    res_sq, n_el = 0.0, 0
    for u_in, re_enc, _ in task["val_loader"]:
        u_d, r_d = u_in.to(device), re_enc.to(device)
        pred = model(u_d, r_d, routing_mode="soft")
        r = burgers_pde_residual_torch(u_d, pred, task["nu"])
        res_sq += r.pow(2).sum().item()
        n_el += r.numel()
    pde_rmse = float(np.sqrt(res_sq / max(1, n_el)))

    wins = task["val_ds"].get_rollout_windows(n_steps=n_steps)
    arr = np.stack(wins, axis=0)
    u_cur = torch.from_numpy(arr[:, 0]).to(device)
    re_t = torch.full((len(arr), 1), float(task["re_norm"]), dtype=torch.float32, device=device)
    preds = []
    for _ in range(n_steps):
        u_nxt = model(u_cur, re_t, routing_mode="soft")
        preds.append(u_nxt.cpu().numpy())
        u_cur = u_nxt
    pred_tr = np.stack(preds, axis=1)
    true_tr = arr[:, 1 : n_steps + 1]
    curve = (np.linalg.norm(pred_tr - true_tr, axis=-1) / (np.linalg.norm(true_tr, axis=-1) + 1e-8)).mean(axis=0)
    return {"pde_residual_rmse": pde_rmse, "step20_rel_l2": float(curve[-1])}


def run_burgers_fuzzy(seeds: List[int], epochs: int, mem: int) -> Dict:
    print(f"=== [4/4] Fuzzy-MEWC-Flow on 1D Burgers (seeds={seeds}, epochs={epochs}, M={mem}, dev=cpu) ===", flush=True)
    device = "cpu"
    raw = []
    for s in seeds:
        set_all_seeds(s)
        tasks, heldout = make_burgers_tasks(
            nx=128, nt=40, n_train=24, n_val=8, batch_size=64, order="ascending", include_heldout=True, seed=s
        )
        model = build_surrogate(
            nx=128, width=32, modes=16, device=device, modular=True,
            router_type="fuzzy", tau_fuzzy=1.0,
        )
        res = MEWCFlowTrainer(
            model, lr=1e-3, epochs=epochs, consolidation_mode="fuzzy_adaptive",
            lambda_min=0.01, lambda_max=0.15, n_fisher=256,
            use_contrastive=False, alpha_contrastive=0.20,
            use_replay=True, memory_per_task=mem, beta_replay=1.0,
            use_physics_loss=True, gamma_physics=0.02,
            use_pde_residual_replay=True, gamma_pde_replay=5e-7,
            is_1d=True,
        ).run_all_tasks(tasks, heldout_tasks=heldout, device=device)

        evals = [_eval_burgers_rollout_pde(model, t, n_steps=20, device=device) for t in tasks]
        entry = {
            "seed": s,
            "avg_forgetting": res["avg_forgetting"],
            "avg_final_mse": res["avg_final_mse"],
            "mean_pde_residual": float(np.mean([e["pde_residual_rmse"] for e in evals])),
            "mean_rollout_step20": float(np.mean([e["step20_rel_l2"] for e in evals])),
            "heldout_mean_mse": float(np.mean(list(res["heldout_mse"].values()))),
            "extra_cl_mb": res["memory_bytes"]["extra_cl_mb"],
        }
        raw.append(entry)
        print(
            f"  [Burgers][Fuzzy-MEWC-Flow][s={s}] Forget={entry['avg_forgetting']:+.2e} | "
            f"Final={entry['avg_final_mse']:.6f} | PDE={entry['mean_pde_residual']:.3f} | R20={entry['mean_rollout_step20']:.4f}",
            flush=True,
        )

    bg_ref = _load_json("mewc_flow/experiments/burgers_pilot/results/burgers_benchmark_summary.json")
    sig = {}
    if bg_ref and "methods" in bg_ref:
        for base in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-PDE-Replay", "MEWC-Flow", "Joint"]:
            if base not in bg_ref["methods"]:
                continue
            bm = bg_ref["methods"][base]
            sig[f"Fuzzy-MEWC-Flow_vs_{base}"] = {
                "avg_forgetting": paired_significance([r["avg_forgetting"] for r in raw], bm["avg_forgetting"]["values"]),
                "avg_final_mse": paired_significance([r["avg_final_mse"] for r in raw], bm["avg_final_mse"]["values"]),
                "mean_pde_residual": paired_significance([r["mean_pde_residual"] for r in raw], bm["mean_pde_residual"]["values"]),
                "mean_rollout_step20": paired_significance([r["mean_rollout_step20"] for r in raw], bm["mean_rollout_step20"]["values"]),
                "heldout_mean_mse": paired_significance([r["heldout_mean_mse"] for r in raw], bm["heldout_mean_mse"]["values"]),
            }

    return {
        "seeds": seeds,
        "epochs_per_task": epochs,
        "memory_per_task": mem,
        "avg_forgetting": summarize_seeds([r["avg_forgetting"] for r in raw]),
        "avg_final_mse": summarize_seeds([r["avg_final_mse"] for r in raw]),
        "mean_pde_residual": summarize_seeds([r["mean_pde_residual"] for r in raw]),
        "mean_rollout_step20": summarize_seeds([r["mean_rollout_step20"] for r in raw]),
        "heldout_mean_mse": summarize_seeds([r["heldout_mean_mse"] for r in raw]),
        "extra_cl_mb": float(np.mean([r["extra_cl_mb"] for r in raw])),
        "significance_tests": sig,
    }


def plot_fig8_neuro_fuzzy(summary: Dict, out_path: str):
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig = plt.figure(figsize=(15.5, 10.0), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    piv_fz = summary["benchmarks"]["PIV"]
    rules = piv_fz["seed0_rules"]
    routing = piv_fz["seed0_routing"]["routing_weights_mean"]
    trace = piv_fz["seed0_trace"]

    # Panel (a): Learned Gaussian Memberships on log_Enstrophy (standardized)
    ax_a = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2.5, 8.0, 400)
    rule_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, r in enumerate(rules):
        c = r["std_centers"]["log_Enstrophy"]
        sig = r["std_sigmas"]["log_Enstrophy"]
        mu = np.exp(-((xs - c) ** 2) / (2.0 * sig ** 2))
        ax_a.plot(xs, mu, color=rule_colors[idx % 4], linewidth=2.2,
                  label=f"{r['regime']} ($c={c:+.2f}, \\sigma={sig:.2f}$)")
        ax_a.fill_between(xs, 0, mu, color=rule_colors[idx % 4], alpha=0.10)
    ax_a.set_xlabel("Standardized Flow Enstrophy ($\\tilde{\\phi}_{\\text{log\\_Enstrophy}}$)")
    ax_a.set_ylabel("Gaussian Membership $\\mu_k(\\phi)$")
    ax_a.set_title("(a) Takagi-Sugeno Gaussian Rules on PIV Enstrophy", fontweight="bold")
    ax_a.set_ylim(0, 1.08)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(fontsize=8.0, loc="upper right")

    # Panel (b): Zero-Leakage Takagi-Sugeno Soft Firing Heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    row_labels = ["Re=240", "Re=650", "Re=910*", "Re=1280", "Re=1890*", "Re=2520"]
    col_labels = ["Rule 240", "Rule 650", "Rule 1280", "Rule 2520"]
    W = np.array([routing[lbl] for lbl in row_labels])
    im = ax_b.imshow(W, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax_b.set_xticks(np.arange(4)); ax_b.set_xticklabels(col_labels)
    ax_b.set_yticks(np.arange(6)); ax_b.set_yticklabels(row_labels)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = W[i, j]
            ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                      color="white" if val > 0.55 else "black", fontsize=8.5, fontweight="bold")
    ax_b.set_title("(b) Physics-Only Fuzzy Rule Firing Matrix (* Held-Out)", fontweight="bold")
    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    # Panel (c): 9-Rule Sugeno Control Surface lambda_t(alpha_t, rho_t)
    ax_c = fig.add_subplot(gs[1, 0])
    surf = SugenoFuzzyLambdaController.control_surface_grid(0.01, 0.15, n=41)
    A = np.array(surf["alphas"])
    D = np.array(surf["drifts"])
    Z = np.array(surf["lambda_surface"])
    cs = ax_c.contourf(A, D, Z, levels=20, cmap="viridis")
    plt.colorbar(cs, ax=ax_c, fraction=0.046, pad=0.04, label="Consolidation $\\lambda_t$")
    ah = np.array(trace.get("alignment_history", []))
    dh = np.array(trace.get("drift_history", []))
    if len(ah) > 0 and len(dh) > 0:
        ax_c.scatter(ah[::2], dh[::2], c="white", s=8, alpha=0.65, edgecolors="black", linewidth=0.3,
                     label="Live PIV Trajectory ($\\alpha_t, \\rho_t$)")
        ax_c.legend(fontsize=8.0, loc="lower left")
    ax_c.set_xlabel("Gradient Cosine Alignment $\\alpha_t$")
    ax_c.set_ylabel("Normalized Loss Drift $\\rho_t$")
    ax_c.set_title("(c) 9-Rule Sugeno FIS Control Surface $\\lambda_t(\\alpha_t, \\rho_t)$", fontweight="bold")

    # Panel (d): 4-Benchmark Head-to-Head Normalized Summary (MEWC-Flow vs Fuzzy-MEWC-Flow vs ER vs FT)
    ax_d = fig.add_subplot(gs[1, 1])
    piv_ref = _load_json("mewc_flow/experiments/piv_experiments/results/piv_benchmark_summary.json")
    kol_ref = _load_json("mewc_flow/experiments/kolmogorov_experiments/results/kolmogorov_benchmark_summary.json")
    era_ref = _load_json("mewc_flow/experiments/era5_experiments/results/era5_benchmark_summary.json")
    bg_ref = _load_json("mewc_flow/experiments/burgers_pilot/results/burgers_benchmark_summary.json")

    cats = ["PIV\nRollout $t=10$", "Kolmogorov\nNS PDE Res.", "ERA5\n72-h Rollout", "Burgers\nRollout $t=20$"]
    comp_methods = ["Fine-Tuning", "ER", "MEWC-Flow", "Fuzzy-MEWC-Flow"]
    colors = {"Fine-Tuning": "#d62728", "ER": "#1f77b4", "MEWC-Flow": "#2ca02c", "Fuzzy-MEWC-Flow": "#9467bd"}

    def _get_vals(m):
        if m == "Fuzzy-MEWC-Flow":
            return [
                summary["benchmarks"]["PIV"]["task0_step10_rel_l2"]["mean"],
                summary["benchmarks"]["Kolmogorov"]["mean_pde_residual"]["mean"],
                summary["benchmarks"]["ERA5"]["mean_rollout_72h"]["mean"],
                summary["benchmarks"]["Burgers"]["mean_rollout_step20"]["mean"],
            ]
        return [
            piv_ref["methods"][m]["task0_rollout"]["step10_rel_l2"]["mean"],
            kol_ref["methods"][m]["mean_pde_residual"]["mean"],
            era_ref["methods"][m]["mean_rollout_72h"]["mean"],
            bg_ref["methods"][m]["mean_rollout_step20"]["mean"],
        ]

    # Normalize each benchmark column relative to MEWC-Flow = 1.00 so all 4 fit cleanly on one axis
    base_vals = np.array(_get_vals("MEWC-Flow"))
    x = np.arange(len(cats))
    w = 0.19
    for idx, m in enumerate(comp_methods):
        vals = np.array(_get_vals(m))
        rel = vals / base_vals
        bars = ax_d.bar(x + (idx - 1.5) * w, rel, w, label=m, color=colors[m], edgecolor="black", linewidth=0.5)
        for bi, b in enumerate(bars):
            raw_v = vals[bi]
            lbl = f"{raw_v:.3f}" if raw_v < 2.0 else f"{raw_v:.2f}"
            ax_d.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02, lbl,
                      ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax_d.axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
    ax_d.set_xticks(x); ax_d.set_xticklabels(cats)
    ax_d.set_ylabel("Error Relative to MEWC-Flow (lower = better)")
    ax_d.set_ylim(0, max(1.6, ax_d.get_ylim()[1] * 1.22))
    ax_d.set_title("(d) 4-Benchmark Head-to-Head Comparison (Raw Annotated)", fontweight="bold")
    ax_d.grid(axis="y", linestyle=":", alpha=0.5)
    ax_d.legend(fontsize=8.0, loc="upper left", ncol=2)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    out_dir = os.path.join(ROOT, "mewc_flow", "experiments", "fuzzy_results")
    os.makedirs(out_dir, exist_ok=True)

    piv = run_piv_fuzzy([42, 102, 202, 302, 402], epochs=15, mem=16, device=args.device)
    kol = run_kolmogorov_fuzzy([42, 102, 202], epochs=12, mem=16, device=args.device)
    era = run_era5_fuzzy([42, 102, 202], epochs=18, mem=16, device=args.device)
    bg = run_burgers_fuzzy([42, 102, 202], epochs=20, mem=32)

    summary = {
        "method": "Fuzzy-MEWC-Flow",
        "description": "Takagi-Sugeno ANFIS Physics-Only Regime Router + 9-Rule Sugeno Fuzzy A-EWC Controller",
        "benchmarks": {
            "PIV": piv,
            "Kolmogorov": kol,
            "ERA5": era,
            "Burgers": bg,
        },
    }
    json_path = os.path.join(out_dir, "fuzzy_mewc_summary.json")
    save_json(summary, json_path)
    print(f"Saved strict JSON -> {json_path}", flush=True)

    fig_path = os.path.join(out_dir, "fig8_neuro_fuzzy_interpretability.png")
    plot_fig8_neuro_fuzzy(summary, fig_path)


if __name__ == "__main__":
    main()
