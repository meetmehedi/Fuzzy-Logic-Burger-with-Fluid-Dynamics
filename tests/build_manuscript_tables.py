"""
build_manuscript_tables.py
==========================
Generates every numerical table of the manuscript DIRECTLY from the saved
strict-JSON results (zero manual transcription). Output: manuscript_tables.md
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = lambda p: os.path.join(ROOT, p)


def load(p):
    p = R(p)
    return json.load(open(p)) if os.path.exists(p) else None


def ms(d, k=5, sci=False):
    if d is None:
        return "–"
    if sci:
        return f"{d['mean']:.2e} ± {d['std']:.1e}"
    return f"{d['mean']:.{k}f} ± {d['std']:.{k}f}"


def sig(tests, pair, key):
    t = tests.get(pair, {}).get(key)
    if not t:
        return ""
    p = t["p_ttest"]
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


ORDER = ["Fine-Tuning", "EWC", "Online-EWC", "ER", "Replay-TS", "SLE-FNO", "MEWC-PDE-Replay", "MEWC-Flow", "Joint"]
out = ["# Auto-generated manuscript tables (from saved JSON results)\n",
       "Significance markers compare MEWC-Flow against each baseline with a paired two-sided t-test across matched seeds: * p<0.05, ** p<0.01. Lower is better for every metric.\n"]

# ---------------- PIV ----------------
piv = load("mewc_flow/experiments/piv_experiments/results/piv_benchmark_summary.json")
if piv:
    sg = piv.get("significance_tests", {})
    out.append(f"\n## Table 2. PIV cylinder wake ({len(piv['seeds'])} seeds, {piv['epochs_per_task']} epochs/task, M={piv['memory_per_task']})\n")
    out.append("| Method | Avg. forgetting (ΔMSE) | Final MSE (4 regimes) | Re=240 rollout rel-L2 (t=10) | Re=240 vorticity RMSE | Held-out MSE (Re=910, 1890) | Extra memory (MB) |")
    out.append("|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in piv["methods"]:
            continue
        d = piv["methods"][m]
        pr = f"MEWC-Flow_vs_{m}"
        out.append(
            f"| {m} | {ms(d['avg_forgetting'], sci=True)}{sig(sg, pr, 'avg_forgetting')} | "
            f"{ms(d['avg_final_mse'], sci=True)}{sig(sg, pr, 'avg_final_mse')} | "
            f"{ms(d['task0_rollout']['step10_rel_l2'], 4)}{sig(sg, pr, 'task0_step10_rel_l2')} | "
            f"{ms(d['task0_rollout']['vorticity_rmse'], 4)}{sig(sg, pr, 'task0_vorticity_rmse')} | "
            f"{ms(d['heldout_mean_mse'], sci=True)}{sig(sg, pr, 'heldout_mean_mse')} | "
            f"{d['memory_bytes']['extra_cl_mb']:.2f} |"
        )
    out.append("\n### Table 3. Paired tests, MEWC-Flow minus baseline (PIV)\n")
    out.append("| Comparison | ΔFinal MSE (p) | ΔForgetting (p) | ΔRollout t=10 (p) | ΔHeld-out MSE (p) | Cohen's d (Final MSE) |")
    out.append("|---|---|---|---|---|---|")
    for pr, t in sg.items():
        out.append(
            f"| {pr.replace('_vs_', ' vs ')} | {t['avg_final_mse']['mean_diff']:+.2e} ({t['avg_final_mse']['p_ttest']:.3f}) | "
            f"{t['avg_forgetting']['mean_diff']:+.2e} ({t['avg_forgetting']['p_ttest']:.3f}) | "
            f"{t['task0_step10_rel_l2']['mean_diff']:+.4f} ({t['task0_step10_rel_l2']['p_ttest']:.3f}) | "
            f"{t['heldout_mean_mse']['mean_diff']:+.2e} ({t['heldout_mean_mse']['p_ttest']:.3f}) | "
            f"{t['avg_final_mse']['cohens_d']:+.2f} |"
        )
    mf = piv["methods"].get("MEWC-Flow", {})
    if "regime_id_accuracy" in mf:
        out.append(f"\nPhysics-only regime identification (no Re input): accuracy {ms(mf['regime_id_accuracy'], 3)}, silhouette {ms(mf['silhouette_score'], 3)}.\n")

man = load("mewc_flow/experiments/piv_experiments/results/latent_manifold_summary.json")
if man:
    out.append("\n### Table 4. Soft routing weights of the seed-0 MEWC-Flow model (rows: input regime, * = held-out)\n")
    out.append("| Input | Adapter Re=240 | Adapter Re=650 | Adapter Re=1280 | Adapter Re=2520 |")
    out.append("|---|---|---|---|---|")
    for lbl, row in zip(man["routing_labels"], man["routing_weights_matrix"]):
        out.append(f"| {lbl} | " + " | ".join(f"{w:.2f}" for w in row) + " |")
    out.append(f"\nSeed-0 nearest-prototype accuracy {man['regime_id_accuracy_task_agnostic']*100:.1f}%, silhouette {man['silhouette_score_seen_physics_only']:.3f}. Confusion matrix (rows true, cols predicted): {man['confusion_matrix']}.\n")

# ---------------- Ablations ----------------
abl = load("mewc_flow/experiments/piv_experiments/results/ablation_study_summary.json")
if abl and isinstance(abl, dict) and "ablations" in abl:
    out.append(f"\n## Table 5. PIV component ablation ({len(abl['seeds'])} seeds)\n")
    out.append("| Variant | Forgetting | Final MSE | Rollout t=10 | Vorticity RMSE | Held-out MSE | Extra MB |")
    out.append("|---|---|---|---|---|---|---|")
    for n, d in abl["ablations"].items():
        out.append(f"| {n} | {ms(d['avg_forgetting'], sci=True)} | {ms(d['avg_final_mse'], sci=True)} | {ms(d['rollout_step10'], 4)} | "
                   f"{ms(d['vorticity_rmse'], 4)} | {ms(d['heldout_mean_mse'], sci=True)} | {d['extra_cl_mb']:.2f} |")
    out.append("\n### Table 6. Routing modes (Full MEWC-Flow)\n")
    out.append("| Routing | Seen MSE | Held-out MSE | Rollout t=10 |")
    out.append("|---|---|---|---|")
    for rm, d in abl["routing_modes"].items():
        out.append(f"| {rm} | {ms(d['avg_seen_mse'], sci=True)} | {ms(d.get('heldout_mean_mse'), sci=True) if d.get('heldout_mean_mse') else '–'} | {ms(d['rollout_step10'], 4)} |")
    out.append("\n### Table 7. Task-order sensitivity (forgetting / final MSE)\n")
    out.append("| Order | Fine-Tuning | ER | MEWC-Flow |")
    out.append("|---|---|---|---|")
    for o, md in abl["task_order_sensitivity"].items():
        out.append(f"| {o} | " + " | ".join(f"{ms(md[m]['avg_forgetting'], sci=True)} / {md[m]['avg_final_mse']['mean']:.2e}" for m in ["Fine-Tuning", "ER", "MEWC-Flow"]) + " |")
    out.append("\n### Table 8. Memory-budget sweep (rollout t=10 / extra MB)\n")
    out.append("| M per task | ER | Replay-TS | MEWC-Flow |")
    out.append("|---|---|---|---|")
    for b, md in abl["memory_budget_sweep"].items():
        out.append(f"| {b} | " + " | ".join(f"{md[m]['rollout_step10']['mean']:.4f} / {md[m]['extra_cl_mb']:.2f}" for m in ["ER", "Replay-TS", "MEWC-Flow"]) + " |")

# ---------------- Kolmogorov ----------------
kol = load("mewc_flow/experiments/kolmogorov_experiments/results/kolmogorov_benchmark_summary.json")
if kol:
    sg = kol.get("significance_tests", {})
    out.append(f"\n## Table 9. 2D Kolmogorov Navier–Stokes bifurcation, Re=20→45→120→400 ({len(kol['seeds'])} seeds)\n")
    out.append("| Method | Forgetting | Final MSE | NS PDE residual | ‖∇·u‖ RMS | Rollout t=10 (mean of 4) | Held-out MSE (Re=75, 240) | Extra MB |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in kol["methods"]:
            continue
        d = kol["methods"][m]
        pr = f"MEWC-Flow_vs_{m}"
        out.append(f"| {m} | {ms(d['avg_forgetting'], sci=True)}{sig(sg, pr, 'avg_forgetting')} | {ms(d['avg_final_mse'], sci=True)}{sig(sg, pr, 'avg_final_mse')} | "
                   f"{ms(d['mean_pde_residual'], 3)}{sig(sg, pr, 'mean_pde_residual')} | {ms(d['mean_div_rms'], 4)} | "
                   f"{ms(d['mean_rollout_step10'], 4)}{sig(sg, pr, 'mean_rollout_step10')} | {ms(d['heldout_mean_mse'], sci=True)}{sig(sg, pr, 'heldout_mean_mse')} | "
                   f"{d['memory_bytes']['extra_cl_mb']:.2f} |")

# ---------------- ERA5 ----------------
era = load("mewc_flow/experiments/era5_experiments/results/era5_benchmark_summary.json")
if era and "seeds" in era:
    sg = era.get("significance_tests", {})
    out.append(f"\n## Table 10. ERA5 10-m winds, seasonal sequence ({len(era['seeds'])} seeds)\n")
    out.append("| Method | Forgetting | Final MSE | 72-h rollout rel-L2 | Held-out summer 2019 | Held-out winter 2019 | NH storm-track RMSE | Extra MB |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in era["methods"]:
            continue
        d = era["methods"][m]
        pr = f"MEWC-Flow_vs_{m}"
        out.append(f"| {m} | {ms(d['avg_forgetting'], sci=True)}{sig(sg, pr, 'avg_forgetting')} | {ms(d['avg_final_mse'], sci=True)}{sig(sg, pr, 'avg_final_mse')} | "
                   f"{ms(d['mean_rollout_72h'], 4)}{sig(sg, pr, 'mean_rollout_72h')} | {ms(d['heldout_summer_mse'], sci=True)} | {ms(d['heldout_winter_mse'], sci=True)} | "
                   f"{ms(d['nh_storm_rmse'], 4)} | {d['memory_bytes']['extra_cl_mb']:.2f} |")

# ---------------- Burgers ----------------
bg = load("mewc_flow/experiments/burgers_pilot/results/burgers_benchmark_summary.json")
if bg:
    sg = bg.get("significance_tests", {})
    out.append(f"\n## Table 11. 1-D Burgers shock sequence ν=0.05→0.0005 and RQ4 ({len(bg['seeds'])} seeds)\n")
    out.append("| Method | Forgetting | Final MSE | Burgers PDE residual | Rollout t=20 (mean of 4) | Held-out MSE | Extra MB |")
    out.append("|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in bg["methods"]:
            continue
        d = bg["methods"][m]
        pr = f"MEWC-Flow_vs_{m}"
        out.append(f"| {m} | {ms(d['avg_forgetting'], sci=True)}{sig(sg, pr, 'avg_forgetting')} | {ms(d['avg_final_mse'], sci=True)}{sig(sg, pr, 'avg_final_mse')} | "
                   f"{ms(d['mean_pde_residual'], 3)} | {ms(d['mean_rollout_step20'], 4)}{sig(sg, pr, 'mean_rollout_step20')} | "
                   f"{ms(d['heldout_mean_mse'], sci=True)} | {d['memory_bytes']['extra_cl_mb']:.3f} |")

# ---------------- Neuro-Fuzzy (Fuzzy-MEWC-Flow) ----------------
fz = load("mewc_flow/experiments/fuzzy_results/fuzzy_mewc_summary.json")
if fz and "benchmarks" in fz:
    bms = fz["benchmarks"]
    out.append("\n## Table 12. Neuro-Fuzzy MEWC-Flow (Fuzzy-MEWC-Flow) vs Neural MEWC-Flow across all 4 benchmarks\n")
    out.append("| Benchmark | Variant | Forgetting | Final MSE | Key Physics / Rollout Metric | Held-out / RegID | Extra MB |")
    out.append("|---|---|---|---|---|---|---|")
    p_fz = bms["PIV"]
    p_nn = piv["methods"]["MEWC-Flow"] if piv else None
    if p_nn:
        out.append(f"| PIV (5 seeds) | MEWC-Flow (Neural) | {ms(p_nn['avg_forgetting'], sci=True)} | {ms(p_nn['avg_final_mse'], sci=True)} | R10={ms(p_nn['task0_rollout']['step10_rel_l2'], 4)} / Vort={ms(p_nn['task0_rollout']['vorticity_rmse'], 4)} | RegAcc={ms(p_nn['regime_id_accuracy'], 3)} | {p_nn['memory_bytes']['extra_cl_mb']:.2f} |")
    out.append(f"| PIV (5 seeds) | Fuzzy-MEWC-Flow | {ms(p_fz['avg_forgetting'], sci=True)} | {ms(p_fz['avg_final_mse'], sci=True)} | R10={ms(p_fz['task0_step10_rel_l2'], 4)} / Vort={ms(p_fz['task0_vorticity_rmse'], 4)} | RegAcc={ms(p_fz['regime_id_accuracy'], 3)} | {p_fz['extra_cl_mb']:.2f} |")

    k_fz = bms["Kolmogorov"]
    k_nn = kol["methods"]["MEWC-Flow"] if kol else None
    if k_nn:
        out.append(f"| Kolmogorov NS (3 seeds) | MEWC-Flow (Neural) | {ms(k_nn['avg_forgetting'], sci=True)} | {ms(k_nn['avg_final_mse'], sci=True)} | NS_PDE={ms(k_nn['mean_pde_residual'], 3)} / R10={ms(k_nn['mean_rollout_step10'], 4)} | HO={ms(k_nn['heldout_mean_mse'], sci=True)} | {k_nn['memory_bytes']['extra_cl_mb']:.2f} |")
    out.append(f"| Kolmogorov NS (3 seeds) | Fuzzy-MEWC-Flow | {ms(k_fz['avg_forgetting'], sci=True)} | {ms(k_fz['avg_final_mse'], sci=True)} | NS_PDE={ms(k_fz['mean_pde_residual'], 3)} / R10={ms(k_fz['mean_rollout_step10'], 4)} | HO={ms(k_fz['heldout_mean_mse'], sci=True)} | {k_fz['extra_cl_mb']:.2f} |")

    e_fz = bms["ERA5"]
    e_nn = era["methods"]["MEWC-Flow"] if era else None
    if e_nn:
        out.append(f"| ERA5 Winds (3 seeds) | MEWC-Flow (Neural) | {ms(e_nn['avg_forgetting'], sci=True)} | {ms(e_nn['avg_final_mse'], sci=True)} | R72h={ms(e_nn['mean_rollout_72h'], 4)} / NH={ms(e_nn['nh_storm_rmse'], 4)} | HO_Sum={ms(e_nn['heldout_summer_mse'], sci=True)} | {e_nn['memory_bytes']['extra_cl_mb']:.2f} |")
    out.append(f"| ERA5 Winds (3 seeds) | Fuzzy-MEWC-Flow | {ms(e_fz['avg_forgetting'], sci=True)} | {ms(e_fz['avg_final_mse'], sci=True)} | R72h={ms(e_fz['mean_rollout_72h'], 4)} / NH={ms(e_fz['nh_storm_rmse'], 4)} | HO_Sum={ms(e_fz['heldout_summer_mse'], sci=True)} | {e_fz['extra_cl_mb']:.2f} |")

    b_fz = bms["Burgers"]
    b_nn = bg["methods"]["MEWC-Flow"] if bg else None
    if b_nn:
        out.append(f"| Burgers 1D (3 seeds) | MEWC-Flow (Neural) | {ms(b_nn['avg_forgetting'], sci=True)} | {ms(b_nn['avg_final_mse'], sci=True)} | PDE={ms(b_nn['mean_pde_residual'], 3)} / R20={ms(b_nn['mean_rollout_step20'], 4)} | HO={ms(b_nn['heldout_mean_mse'], sci=True)} | {b_nn['memory_bytes']['extra_cl_mb']:.3f} |")
    out.append(f"| Burgers 1D (3 seeds) | Fuzzy-MEWC-Flow | {ms(b_fz['avg_forgetting'], sci=True)} | {ms(b_fz['avg_final_mse'], sci=True)} | PDE={ms(b_fz['mean_pde_residual'], 3)} / R20={ms(b_fz['mean_rollout_step20'], 4)} | HO={ms(b_fz['heldout_mean_mse'], sci=True)} | {b_fz['extra_cl_mb']:.3f} |")

    out.append(f"\nFuzzy-MEWC-Flow PIV seed-0 accuracy: {p_fz['seed0_routing']['regime_id_accuracy']*100:.1f}%, silhouette {p_fz['seed0_routing']['silhouette_score']:.3f}. Confusion matrix: {p_fz['seed0_routing']['confusion_matrix']}.\n")
    out.append("\n### Extracted PIV Linguistic IF-THEN Rules (Seed-0)\n")
    for r in p_fz["seed0_rules"]:
        out.append(f"- **{r['regime']}:** {r['if_then']}")

path = R("manuscript_tables.md")
open(path, "w").write("\n".join(out) + "\n")
print(f"Wrote {path}")
