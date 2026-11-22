"""
generate_extra_figures.py
=========================
Generates publication-ready 300-DPI Q1 figures for Benchmarks 2, 3, and 4:
  - Fig 5: 2D Kolmogorov Navier-Stokes Bifurcation (Re=20->45->120->400)
           Vorticity contours across bifurcation, PDE residual, divergence, & rollout curves
  - Fig 6: Real ECMWF ERA5 Planetary Atmospheric Winds
           Seasonal CL comparison, 72-hr rollout curves, & Hemispheric spatial error slices
  - Fig 7: 1-D Viscous Burgers Shock-Formation & RQ4 (Physics-Residual Replay vs Stored Replay)
           Method comparison, shock profiles, PDE residual, & 20-step rollout error
"""

import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mewc_flow.data.kolmogorov_ns_dataset import make_kolmogorov_tasks
from mewc_flow.data.burgers_dataset import make_burgers_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.models.surrogate import build_surrogate
from mewc_flow.utils.physics_metrics import compute_vorticity_and_divergence_2d


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
})

COLORS = {
    "Fine-Tuning":     "#d62728",
    "EWC":             "#ff7f0e",
    "Online-EWC":      "#bcbd22",
    "ER":              "#1f77b4",
    "Replay-TS":       "#9467bd",
    "SLE-FNO":         "#8c564b",
    "MEWC-PDE-Replay": "#17becf",
    "MEWC-Flow":       "#2ca02c",
    "Joint":           "#222222",
}


def plot_fig5_kolmogorov():
    out_dir = "./mewc_flow/experiments/kolmogorov_experiments/results"
    path = os.path.join(out_dir, "kolmogorov_benchmark_summary.json")
    if not os.path.exists(path):
        print(f"Skipping Fig 5 ({path} missing)")
        return
    with open(path) as f:
        data = json.load(f)

    tasks, _ = make_kolmogorov_tasks(data_root="./data/kolmogorov", auto_generate=False)
    methods = [m for m in ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"] if m in data["methods"]]
    cl_methods = [m for m in methods if m != "Joint"]

    fig = plt.figure(figsize=(16.5, 9.5), dpi=300)

    # Row 1: Ground-truth vorticity snapshots across the 4 bifurcation regimes (Re=20, 45, 120, 400)
    for i, t in enumerate(tasks):
        ax = plt.subplot(2, 4, i + 1)
        uv = t["val_ds"].segments[0][10]  # (2, 64, 64)
        w, _ = compute_vorticity_and_divergence_2d(uv, dy=t["dy"], dx=t["dx"], periodic=True)
        lim = max(np.abs(w).max(), 1e-4)
        im = ax.imshow(w, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
        ax.set_title(t["label"], fontweight="bold", fontsize=9.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2 Col 1: Average Forgetting
    ax1 = plt.subplot(2, 3, 4)
    x = np.arange(len(cl_methods))
    fm = [data["methods"][m]["avg_forgetting"]["mean"] for m in cl_methods]
    fs = [data["methods"][m]["avg_forgetting"]["std"] for m in cl_methods]
    ax1.bar(x, fm, yerr=fs, capsize=3, color=[COLORS[m] for m in cl_methods], edgecolor="black", linewidth=0.5)
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(cl_methods, rotation=30, ha="right")
    ax1.set_ylabel("Avg. Forgetting $\\Delta$MSE")
    ax1.set_title("(a) Forgetting Across Bifurcation (Re=20->400)", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Row 2 Col 2: Exact Navier-Stokes PDE Residual
    ax2 = plt.subplot(2, 3, 5)
    x_all = np.arange(len(methods))
    pm = [data["methods"][m]["mean_pde_residual"]["mean"] for m in methods]
    ps = [data["methods"][m]["mean_pde_residual"]["std"] for m in methods]
    ax2.bar(x_all, pm, yerr=ps, capsize=3, color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_all); ax2.set_xticklabels(methods, rotation=30, ha="right")
    ax2.set_ylabel("Mean NS Vorticity PDE Residual (RMS)")
    ax2.set_title("(b) Exact 2D Navier-Stokes PDE Residual", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Row 2 Col 3: Rollout curves on Re=45 (Post-Bifurcation Regime)
    ax3 = plt.subplot(2, 3, 6)
    steps = np.arange(1, 21)
    for m in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow"]:
        if m in data["methods"]:
            curve = data["methods"][m]["per_regime_seed0"]["Re=45"]["rel_l2_mean"]
            lw = 2.5 if m == "MEWC-Flow" else 1.5
            ax3.plot(steps, curve, label=m, color=COLORS[m], linewidth=lw)
    ax3.set_xlabel("Autoregressive Step ($t$)")
    ax3.set_ylabel("Relative $L_2$ Error")
    ax3.set_title("(c) Post-Bifurcation Rollout Stability (Re=45)", fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(ncol=2)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig5_kolmogorov_bifurcation.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_fig6_era5():
    out_dir = "./mewc_flow/experiments/era5_experiments/results"
    path = os.path.join(out_dir, "era5_benchmark_summary.json")
    if not os.path.exists(path):
        print(f"Skipping Fig 6 ({path} missing)")
        return
    with open(path) as f:
        data = json.load(f)

    methods = [m for m in ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow", "Joint"] if m in data["methods"]]
    cl_methods = [m for m in methods if m != "Joint"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=300)

    # Panel A: Final MSE & Held-out 2019 MSE
    ax = axes[0]
    x = np.arange(len(methods))
    w = 0.38
    m_seen = [data["methods"][m]["avg_final_mse"]["mean"] for m in methods]
    s_seen = [data["methods"][m]["avg_final_mse"]["std"] for m in methods]
    m_ho = [data["methods"][m]["heldout_mean_mse"]["mean"] for m in methods]
    s_ho = [data["methods"][m]["heldout_mean_mse"]["std"] for m in methods]
    ax.bar(x - w/2, m_seen, w, yerr=s_seen, capsize=2, label="Seen 2020 Seasons", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, m_ho, w, yerr=s_ho, capsize=2, label="Held-out 2019 Year", color="#55a868", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Validation MSE (lower = better)")
    ax.set_title("(a) ERA5 Seen & Unseen Year Accuracy", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    # Panel B: 72-Hour (12-Step) Summer Rollout Curve
    ax = axes[1]
    hours = np.arange(1, 13) * 6
    for m in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-Flow"]:
        if m in data["methods"]:
            mu = np.array(data["methods"][m]["summer_rollout_curve_mean"])
            sd = np.array(data["methods"][m]["summer_rollout_curve_std"])
            lw = 2.5 if m == "MEWC-Flow" else 1.5
            ax.plot(hours, mu, label=m, color=COLORS[m], linewidth=lw)
            ax.fill_between(hours, np.maximum(0, mu - sd), mu + sd, color=COLORS[m], alpha=0.14)
    ax.set_xlabel("Forecast Lead Time (Hours)")
    ax.set_ylabel("Relative $L_2$ Error")
    ax.set_title("(b) 72-Hr Rollout Retention on Summer 2020", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(ncol=2)

    # Panel C: Hemispheric Slice-Based Spatial Error Analysis
    ax = axes[2]
    show = ["Fine-Tuning", "EWC", "ER", "Replay-TS", "MEWC-Flow"]
    sx = np.arange(len(show))
    bw = 0.25
    nh = [data["methods"][m]["nh_storm_rmse"]["mean"] for m in show]
    tr = [data["methods"][m]["tropics_rmse"]["mean"] for m in show]
    sh = [data["methods"][m]["sh_midlat_rmse"]["mean"] for m in show]
    ax.bar(sx - bw, nh, bw, label="NH Storm Track", color="#d62728", alpha=0.85)
    ax.bar(sx,      tr, bw, label="Tropics",        color="#ff7f0e", alpha=0.85)
    ax.bar(sx + bw, sh, bw, label="SH Mid-Lat",     color="#1f77b4", alpha=0.85)
    ax.set_xticks(sx); ax.set_xticklabels(show, rotation=22, ha="right")
    ax.set_ylabel("Spatial Rollout RMSE")
    ax.set_title("(c) Hemispheric Spatial Error Slices", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    p = os.path.join(out_dir, "fig6_era5_planetary_analysis.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_fig7_burgers():
    out_dir = "./mewc_flow/experiments/burgers_pilot/results"
    path = os.path.join(out_dir, "burgers_benchmark_summary.json")
    if not os.path.exists(path):
        print(f"Skipping Fig 7 ({path} missing)")
        return
    with open(path) as f:
        data = json.load(f)

    methods = [m for m in ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-PDE-Replay", "MEWC-Flow", "Joint"] if m in data["methods"]]
    cl_methods = [m for m in methods if m != "Joint"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), dpi=300)

    # Panel A: RQ4 — Forgetting (Stored Replay vs PDE-Residual Replay vs Full MEWC-Flow)
    ax = axes[0]
    x = np.arange(len(cl_methods))
    fm = [data["methods"][m]["avg_forgetting"]["mean"] for m in cl_methods]
    fs = [data["methods"][m]["avg_forgetting"]["std"] for m in cl_methods]
    ax.bar(x, fm, yerr=fs, capsize=2.5, color=[COLORS[m] for m in cl_methods], edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cl_methods, rotation=32, ha="right")
    ax.set_ylabel("Avg. Forgetting $\\Delta$MSE")
    ax.set_title("(a) RQ4: Stored vs. Zero-Data PDE-Residual Replay", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Exact 1-D Burgers PDE Residual
    ax = axes[1]
    x_all = np.arange(len(methods))
    pm = [data["methods"][m]["mean_pde_residual"]["mean"] for m in methods]
    ps = [data["methods"][m]["mean_pde_residual"]["std"] for m in methods]
    ax.bar(x_all, pm, yerr=ps, capsize=2.5, color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_all); ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_ylabel("Exact Burgers PDE Residual (RMS)")
    ax.set_title("(b) Governing PDE Residual Verification", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel C: 20-Step Autoregressive Rollout on Task 0 (nu=0.05)
    ax = axes[2]
    steps = np.arange(1, 21)
    for m in ["Fine-Tuning", "EWC", "ER", "Replay-TS", "MEWC-PDE-Replay", "MEWC-Flow"]:
        if m in data["methods"]:
            lw = 2.5 if m == "MEWC-Flow" else 1.6
            ax.plot(steps, data["methods"][m]["task0_rollout_curve_mean"], label=m, color=COLORS[m], linewidth=lw)
    ax.set_xlabel("Autoregressive Rollout Step ($t$)")
    ax.set_ylabel("Relative $L_2$ Error")
    ax.set_title("(c) 20-Step Rollout Retention on Diffusive Task 0", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(ncol=2)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig7_burgers_pde_residual_replay.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    plot_fig5_kolmogorov()
    plot_fig6_era5()
    plot_fig7_burgers()
