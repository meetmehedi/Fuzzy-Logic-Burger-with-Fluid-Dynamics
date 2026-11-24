"""
evaluate_rollout_and_physics.py
===============================
Generates Publication-Grade Q1 Figures & Deep Error Analysis for PIV Cylinder Wake:
  - Fig 1: Multi-Seed Benchmark Performance, Forgetting Matrix & Memory-Accuracy Pareto
  - Fig 2: Shared-Colorbar Velocity, Vorticity Contours, Spatial Error Heatmaps,
           Multi-Window Rollout Curves (mean +/- std), and 2D Radial Energy Spectrum E(k)
  - Fig 4: Multi-Seed Component Ablations, Routing Modes, Task-Order Sensitivity (RQ5),
           Memory-Budget Sweep, and Complex A-EWC Adaptive Lambda Dynamics!

Fixes all earlier visualization flaws:
- Shared vmin/vmax colorbar scale across velocity panels (no misleading per-panel scaling).
- Full un-truncated titles ("Vorticity omega (Ground Truth)", not "(Ground)").
- Multi-window rollout trajectories with +/- 1 std shading across 5 seeds.
- 2D isotropic power spectral density E(k) showing wavenumber-resolved forgetting.
"""

import json
import os
import sys
from typing import Dict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.piv_cylinder_dataset import make_piv_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.utils.physics_metrics import (
    multi_window_rollout_eval,
    compute_vorticity_and_divergence_2d,
    radial_energy_spectrum_2d,
)
from mewc_flow.utils.results_io import save_json


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


COLORS = {
    "Fine-Tuning": "#d62728",
    "EWC":         "#ff7f0e",
    "Online-EWC":  "#bcbd22",
    "ER":          "#1f77b4",
    "Replay-TS":   "#9467bd",
    "SLE-FNO":     "#8c564b",
    "MEWC-Flow":   "#2ca02c",
    "Joint":       "#222222",
}


def _load_model(method: str, out_dir: str):
    tag = method.lower().replace("-", "_")
    path = os.path.join(out_dir, f"model_{tag}.pt")
    if not os.path.exists(path):
        return None
    if method == "MEWC-Flow":
        m = build_fno2d(modes=8, width=24, modular=True, latent_dim=32, adapter_bottleneck=16,
                        use_adapters=True, use_contrastive_encoder=True, route_temperature=0.14)
        for _ in range(4):
            m.add_regime_adapter()
        m.register_buffer("prototypes", torch.zeros(4, 32))
    elif method == "SLE-FNO":
        m = build_fno2d(modes=8, width=24, modular=True, latent_dim=32, adapter_bottleneck=16,
                        use_adapters=True, use_contrastive_encoder=False)
        for _ in range(4):
            m.add_regime_adapter()
        m.register_buffer("prototypes", torch.zeros(4, 32))
        m.prototype_re = [240/2520, 650/2520, 1280/2520, 1.0]
    else:
        m = build_fno2d(modes=8, width=24, modular=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def plot_fig1_benchmark(bench: Dict, out_dir: str):
    methods = [m for m in ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"] if m in bench["methods"]]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=300)

    # Panel A: Average Final MSE & Average Forgetting
    ax = axes[0]
    x = np.arange(len(methods))
    means = [bench["methods"][m]["avg_final_mse"]["mean"] for m in methods]
    stds = [bench["methods"][m]["avg_final_mse"]["std"] for m in methods]
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=[COLORS[m] for m in methods], alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Average Final Val MSE (lower = better)")
    ax.set_title("(a) Final Accuracy Across All PIV Regimes")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Average Forgetting (delta MSE)
    ax = axes[1]
    cl_methods = [m for m in methods if m != "Joint"]
    x2 = np.arange(len(cl_methods))
    f_means = [bench["methods"][m]["avg_forgetting"]["mean"] for m in cl_methods]
    f_stds = [bench["methods"][m]["avg_forgetting"]["std"] for m in cl_methods]
    ax.bar(x2, f_means, yerr=f_stds, capsize=3, color=[COLORS[m] for m in cl_methods], alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x2)
    ax.set_xticklabels(cl_methods, rotation=30, ha="right")
    ax.set_ylabel("Average Forgetting $\\Delta$MSE (lower = better)")
    ax.set_title("(b) Catastrophic Forgetting (5 Seeds Mean $\\pm$ SD)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel C: Memory Budget vs. Task-0 Rollout Step-10 Error (Pareto)
    ax = axes[2]
    for m in cl_methods:
        mem_mb = bench["methods"][m]["memory_bytes"]["extra_cl_mb"]
        r10 = bench["methods"][m]["task0_rollout"]["step10_rel_l2"]["mean"]
        r10_sd = bench["methods"][m]["task0_rollout"]["step10_rel_l2"]["std"]
        ax.errorbar(mem_mb, r10, yerr=r10_sd, fmt="o", markersize=9, capsize=3, color=COLORS[m], label=m, markeredgecolor="black")
        ax.annotate(m, (mem_mb, r10), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Additional CL Memory Overhead (MB)")
    ax.set_ylabel("Task-0 (Re=240) 10-Step Rollout Rel. $L_2$")
    ax.set_title("(c) Memory Budget vs. Rollout Stability")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig1_piv_benchmark_and_forgetting.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_fig2_physics_rollout_spectrum(bench: Dict, tasks, out_dir: str):
    t0 = tasks[0]
    show_methods = ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow"]
    models = {m: _load_model(m, out_dir) for m in show_methods}
    evals = {}
    for m, mod in models.items():
        if mod is not None:
            evals[m] = multi_window_rollout_eval(
                mod, t0["val_ds"], re_norm=t0["re_norm"], n_steps=20, stride=3,
                device="cpu", routing_mode="soft", dy=t0["dy"], dx=t0["dx"], max_windows=10
            )

    t_plot = 9  # step 10
    panel_methods = ["Fine-Tuning", "ER", "MEWC-Flow"]
    true_u = evals["MEWC-Flow"]["sample_true"]  # (20, 2, 64, 64)

    fig = plt.figure(figsize=(16.5, 11.5), dpi=300)

    # Row 1: Shared-Colorbar Velocity Magnitude |u| / U_inf
    vmag_true = np.sqrt(true_u[t_plot, 0]**2 + true_u[t_plot, 1]**2)
    vmags = [vmag_true] + [np.sqrt(evals[m]["sample_pred"][t_plot, 0]**2 + evals[m]["sample_pred"][t_plot, 1]**2) for m in panel_methods]
    vmax_u = max(v.max() for v in vmags)
    vmin_u = min(v.min() for v in vmags)
    titles_r1 = ["Ground Truth PIV (Re=240)", "Fine-Tuning (t=10)", "Experience Replay (t=10)", "MEWC-Flow (t=10)"]
    for i, (arr, ttl) in enumerate(zip(vmags, titles_r1)):
        ax = plt.subplot(3, 4, i + 1)
        im = ax.imshow(arr, cmap="viridis", origin="lower", vmin=vmin_u, vmax=vmax_u)
        ax.set_title(ttl, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Shared-Colorbar Anisotropic Vorticity + Spatial Error Heatmaps
    w_true, _ = compute_vorticity_and_divergence_2d(true_u[t_plot], dy=t0["dy"], dx=t0["dx"])
    w_list = [w_true] + [compute_vorticity_and_divergence_2d(evals[m]["sample_pred"][t_plot], dy=t0["dy"], dx=t0["dx"])[0] for m in panel_methods]
    w_lim = max(np.abs(w_true).max(), 1e-4)
    titles_r2 = ["Vorticity $\\omega$ (Ground Truth)", "Vorticity $\\omega$ (Fine-Tuning)", "Vorticity $\\omega$ (ER)", "Vorticity $\\omega$ (MEWC-Flow)"]
    for i, (arr, ttl) in enumerate(zip(w_list, titles_r2)):
        ax = plt.subplot(3, 4, i + 5)
        im = ax.imshow(arr, cmap="RdBu_r", origin="lower", vmin=-w_lim, vmax=w_lim)
        ax.set_title(ttl)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3 Left: Multi-Seed Multi-Window Rollout Error Growth (t=1..20)
    ax_r = plt.subplot(3, 2, 5)
    steps = np.arange(1, 21)
    for m in show_methods:
        if m not in bench["methods"]:
            continue
        mu = np.array(bench["methods"][m]["task0_rollout_curve_mean"])
        sd = np.array(bench["methods"][m]["task0_rollout_curve_std"])
        lw = 2.6 if m == "MEWC-Flow" else 1.6
        ax_r.plot(steps, mu, label=m, color=COLORS[m], linewidth=lw)
        ax_r.fill_between(steps, np.maximum(0, mu - sd), mu + sd, color=COLORS[m], alpha=0.14)
    ax_r.set_xlabel("Autoregressive Rollout Step ($t$)")
    ax_r.set_ylabel("Relative $L_2$ Error $\\epsilon(t)$")
    ax_r.set_title("(a) Multi-Window Rollout Stability on Past Regime (Re=240, 5 Seeds Mean $\\pm$ SD)", fontweight="bold")
    ax_r.grid(True, linestyle=":", alpha=0.5)
    ax_r.legend(ncol=2)

    # Row 3 Right: 2D Radially-Averaged Energy Spectrum E(k) at Step 10
    ax_s = plt.subplot(3, 2, 6)
    k_bins, e_true = radial_energy_spectrum_2d(true_u[t_plot])
    ax_s.loglog(k_bins, e_true, "k-", linewidth=2.5, label="Ground Truth PIV")
    for m in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "MEWC-Flow"]:
        if m in evals:
            _, e_p = radial_energy_spectrum_2d(evals[m]["sample_pred"][t_plot])
            lw = 2.4 if m == "MEWC-Flow" else 1.5
            ls = "-" if m == "MEWC-Flow" else "--"
            ax_s.loglog(k_bins, e_p, linestyle=ls, linewidth=lw, color=COLORS[m], label=m)
    ax_s.axvline(8, color="gray", linestyle=":", linewidth=1.2, label="FNO Spectral Cutoff ($k_{\\max}=8$)")
    ax_s.set_xlabel("Radial Wavenumber $k$")
    ax_s.set_ylabel("Kinetic Energy Spectrum $E(k)$")
    ax_s.set_title("(b) 2D Radially-Averaged Energy Spectrum $E(k)$ at $t=10$ (Re=240)", fontweight="bold")
    ax_s.grid(True, which="both", linestyle=":", alpha=0.45)
    ax_s.legend(ncol=2)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig2_piv_physics_rollout_spectrum.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # Also save structured physics summary JSON
    phys_out = {
        m: {
            "step1_rel_l2": evals[m]["step1_rel_l2"],
            "step10_rel_l2": evals[m]["step10_rel_l2"],
            "step20_rel_l2": evals[m]["step_final_rel_l2"],
            "vph_025": evals[m]["vph_025"],
            "vorticity_rmse": evals[m]["vorticity_rmse"],
            "divergence_rmse": evals[m]["divergence_rmse"],
            "ke_rel_err": evals[m]["ke_rel_err"],
            "enstrophy_rel_err": evals[m]["enstrophy_rel_err"],
            "log_spec_rmse_all": evals[m]["spectral"]["log_spec_rmse_all"],
            "log_spec_rmse_low": evals[m]["spectral"]["log_spec_rmse_low"],
            "log_spec_rmse_mid": evals[m]["spectral"]["log_spec_rmse_mid"],
            "log_spec_rmse_high": evals[m]["spectral"]["log_spec_rmse_high"],
        }
        for m in evals
    }
    save_json(phys_out, os.path.join(out_dir, "physics_and_rollout_summary.json"))


def plot_fig4_ablations_and_sensitivity(bench: Dict, abl: Dict, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=300)

    # Panel A: Component Ablations (Final MSE & Rollout Step-10)
    ax = axes[0, 0]
    names = list(abl["ablations"].keys())
    x = np.arange(len(names))
    r10_m = [abl["ablations"][n]["rollout_step10"]["mean"] for n in names]
    r10_s = [abl["ablations"][n]["rollout_step10"]["std"] for n in names]
    cols = ["#2ca02c" if n == "Full MEWC-Flow" else "#4c72b0" for n in names]
    ax.barh(x, r10_m, xerr=r10_s, capsize=3, color=cols, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Task-0 Rollout Step-10 Rel. $L_2$ (lower = better)")
    ax.set_title("(a) Systematic Component Ablation (3 Seeds Mean $\\pm$ SD)", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel B: Task-Order Sensitivity (RQ5)
    ax = axes[0, 1]
    orders = ["ascending", "descending", "shuffled"]
    ox = np.arange(len(orders))
    w = 0.25
    for idx, m in enumerate(["Fine-Tuning", "ER", "MEWC-Flow"]):
        ms = [abl["task_order_sensitivity"][o][m]["avg_forgetting"]["mean"] for o in orders]
        ss = [abl["task_order_sensitivity"][o][m]["avg_forgetting"]["std"] for o in orders]
        ax.bar(ox + (idx - 1) * w, ms, w, yerr=ss, capsize=3, label=m, color=COLORS[m], edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(ox)
    ax.set_xticklabels(["Ascending\n(240->2520)", "Descending\n(2520->240)", "Shuffled\n(1280->240->2520->650)"])
    ax.set_ylabel("Average Forgetting $\\Delta$MSE")
    ax.set_title("(b) Task-Order Sensitivity (RQ5)", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    # Panel C: Memory-Budget Sweep
    ax = axes[1, 0]
    mems = [4, 8, 16, 32]
    for m in ["ER", "Replay-TS", "MEWC-Flow"]:
        ys = [abl["memory_budget_sweep"][str(b)][m]["rollout_step10"]["mean"] for b in mems]
        es = [abl["memory_budget_sweep"][str(b)][m]["rollout_step10"]["std"] for b in mems]
        ax.errorbar(mems, ys, yerr=es, marker="o", linewidth=2.2, capsize=3, label=m, color=COLORS[m])
    ax.set_xticks(mems)
    ax.set_xlabel("Episodic Exemplars Stored per Task ($M$)")
    ax.set_ylabel("Task-0 Rollout Step-10 Rel. $L_2$")
    ax.set_title("(c) Memory-Budget Scaling Sweep", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    # Panel D: Complex Adaptive EWC (A-EWC) Alignment & Dynamic Lambda Trajectory
    ax = axes[1, 1]
    trace = bench["methods"]["MEWC-Flow"].get("aewc_trace_seed0", {})
    if trace and trace.get("lambda_history"):
        lam = np.array(trace["lambda_history"])
        alp = np.array(trace["alignment_history"])
        # Smooth with moving average window=15 for crisp visualization
        win = min(15, max(1, len(lam) // 20))
        kern = np.ones(win) / win
        lam_s = np.convolve(lam, kern, mode="valid")
        alp_s = np.convolve(alp, kern, mode="valid")
        steps = np.arange(len(lam_s))

        ax.plot(steps, lam_s, color="#2ca02c", linewidth=2.0, label="Adaptive $\\lambda_t$ (A-EWC)")
        ax.set_xlabel("Optimization Step Across Sequential Regimes")
        ax.set_ylabel("Consolidation Strength $\\lambda_t$", color="#2ca02c")
        ax.tick_params(axis="y", labelcolor="#2ca02c")

        ax2 = ax.twinx()
        ax2.plot(steps, alp_s, color="#1f77b4", alpha=0.75, linewidth=1.4, linestyle="--", label="Grad Cosine $\\alpha_t$")
        ax2.axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
        ax2.set_ylabel("Complex Gradient Alignment $\\alpha_t$", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_title("(d) Complex A-EWC Dynamic Stability-Plasticity Regulation", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "A-EWC trace unavailable", ha="center", va="center")

    plt.tight_layout()
    p = os.path.join(out_dir, "fig4_piv_ablations_and_sensitivity.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def main():
    out_dir = "./mewc_flow/experiments/piv_experiments/results"
    bench_path = os.path.join(out_dir, "piv_benchmark_summary.json")
    abl_path = os.path.join(out_dir, "ablation_study_summary.json")

    with open(bench_path) as f:
        bench = json.load(f)
    tasks, _ = make_piv_tasks(data_root="./data/piv", include_heldout=True)

    plot_fig1_benchmark(bench, out_dir)
    plot_fig2_physics_rollout_spectrum(bench, tasks, out_dir)
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl = json.load(f)
        if isinstance(abl, dict) and "ablations" in abl:
            plot_fig4_ablations_and_sensitivity(bench, abl, out_dir)


if __name__ == "__main__":
    main()
