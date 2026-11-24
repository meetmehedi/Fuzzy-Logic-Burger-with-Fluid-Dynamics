"""
analyze_latent_manifold.py
==========================
Analyzes the STRICTLY PHYSICS-ONLY (ZERO Re-label leakage!) MEWC-Flow
contrastive latent representation space z = e_psi(u_t) in S^{31}:
1. Extracts embeddings across all 4 seen regimes (Re=240, 650, 1280, 2520)
   and 2 held-out interpolation regimes (Re=910, 1890) using ONLY velocity fields.
2. Evaluates task-agnostic nearest-prototype classification accuracy, confusion
   matrix, and Silhouette clustering score (RQ3).
3. Computes soft prototype routing weights across seen AND unseen held-out regimes
   to verify smooth physical interpolation across Reynolds regimes.
4. Saves publication-quality figure `fig3_piv_latent_manifold_and_routing.png`
   (3 panels: Cosine KernelPCA manifold, Task-Agnostic Confusion Matrix, Soft Routing Heatmap).
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.piv_cylinder_dataset import make_piv_tasks
from mewc_flow.models.fno2d import build_fno2d
from mewc_flow.utils.results_io import save_json


def main():
    out_dir = "./mewc_flow/experiments/piv_experiments/results"
    mewc_path = os.path.join(out_dir, "model_mewc_flow.pt")
    if not os.path.exists(mewc_path):
        raise FileNotFoundError(f"Missing {mewc_path}. Run run_cl_piv.py first.")

    tasks, heldout = make_piv_tasks(data_root="./data/piv", include_heldout=True)

    model = build_fno2d(
        modes=8, width=24, modular=True, latent_dim=32, adapter_bottleneck=16,
        use_adapters=True, use_contrastive_encoder=True, route_temperature=0.14,
    )
    for _ in range(4):
        model.add_regime_adapter()
    model.register_buffer("prototypes", torch.zeros(4, 32))
    model.load_state_dict(torch.load(mewc_path, map_location="cpu", weights_only=True))
    model.prototype_re = [t["re_norm"] for t in tasks]
    model.eval()

    protos = model.prototypes.numpy()  # (4, 32)
    all_z, all_re, all_is_ho = [], [], []
    y_true_seen, y_pred_seen = [], []
    routing_rows, routing_labels = [], []

    ordered_eval = [
        (tasks[0], False, 0),
        (tasks[1], False, 1),
        (heldout[0], True, -1),
        (tasks[2], False, 2),
        (heldout[1], True, -1),
        (tasks[3], False, 3),
    ]

    with torch.no_grad():
        for t_dict, is_ho, tid in ordered_eval:
            zs, ws = [], []
            for u_in, re_enc, _ in t_dict["val_loader"]:
                z = model.encode_regime(u_in, dy=t_dict["dy"], dx=t_dict["dx"])
                w = model.compute_routing_weights(z, re_enc, len(model.adapters), routing_mode="soft")
                zs.append(z.numpy())
                ws.append(w.numpy())
            Z_t = np.concatenate(zs, axis=0)
            W_t = np.concatenate(ws, axis=0)

            all_z.append(Z_t)
            all_re.extend([t_dict["re"]] * len(Z_t))
            all_is_ho.extend([is_ho] * len(Z_t))
            routing_rows.append(W_t.mean(axis=0))
            routing_labels.append(f"Re={t_dict['re']}" + ("*" if is_ho else ""))

            if not is_ho:
                preds = np.argmax(Z_t @ protos.T, axis=1)
                y_true_seen.extend([tid] * len(Z_t))
                y_pred_seen.extend(preds.tolist())

    Z = np.concatenate(all_z, axis=0)
    Re_vals = np.array(all_re)
    is_ho_arr = np.array(all_is_ho)

    seen_mask = ~is_ho_arr
    sil_seen = float(silhouette_score(Z[seen_mask], Re_vals[seen_mask]))
    acc_seen = float(np.mean(np.array(y_true_seen) == np.array(y_pred_seen)))
    cm = confusion_matrix(y_true_seen, y_pred_seen, labels=[0, 1, 2, 3])
    routing_mat = np.stack(routing_rows, axis=0)  # (6, 4)

    pca = PCA(n_components=2).fit(Z)
    var_exp = pca.explained_variance_ratio_
    kpca = KernelPCA(n_components=2, kernel="cosine")
    Z_kpca = kpca.fit_transform(Z)

    # Publication 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300)

    palette = {240: "#1f77b4", 650: "#2ca02c", 910: "#ff7f0e", 1280: "#d62728", 1890: "#9467bd", 2520: "#8c564b"}
    markers = {240: "o", 650: "s", 910: "*", 1280: "^", 1890: "D", 2520: "v"}

    # Panel A: Physics-Only Cosine KernelPCA Manifold
    ax = axes[0]
    for re_v in [240, 650, 910, 1280, 1890, 2520]:
        m = (Re_vals == re_v)
        ho = is_ho_arr[m][0]
        ax.scatter(
            Z_kpca[m, 0], Z_kpca[m, 1],
            c=palette[re_v], marker=markers[re_v],
            s=95 if ho else 45, alpha=0.92 if ho else 0.72,
            edgecolors="black" if ho else "none", linewidths=0.9,
            label=f"Re={re_v}" + (" (Held-out)" if ho else ""),
        )
    ax.set_title(f"(a) Physics-Only Contrastive Manifold (Zero Re Input)\nSilhouette = {sil_seen:.3f} | PCA Top-2 Var = {sum(var_exp)*100:.1f}%", fontweight="bold")
    ax.set_xlabel("Cosine KernelPCA Component 1")
    ax.set_ylabel("Cosine KernelPCA Component 2")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")

    # Panel B: Task-Agnostic Regime Confusion Matrix
    ax = axes[1]
    cm_norm = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    im1 = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    re_seen_lbls = ["Re=240", "Re=650", "Re=1280", "Re=2520"]
    ax.set_xticks(range(4)); ax.set_xticklabels(re_seen_lbls)
    ax.set_yticks(range(4)); ax.set_yticklabels(re_seen_lbls)
    ax.set_xlabel("Predicted Regime Prototype")
    ax.set_ylabel("True Experimental Regime")
    ax.set_title(f"(b) Task-Agnostic Regime ID (Acc = {acc_seen*100:.1f}%)", fontweight="bold")
    for i in range(4):
        for j in range(4):
            txt_c = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)", ha="center", va="center", color=txt_c, fontsize=8.5)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Panel C: Soft Prototype Routing Weight Matrix (Seen + Unseen Interpolation!)
    ax = axes[2]
    im2 = ax.imshow(routing_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Adapter 0\n(Re=240)", "Adapter 1\n(Re=650)", "Adapter 2\n(Re=1280)", "Adapter 3\n(Re=2520)"])
    ax.set_yticks(range(6)); ax.set_yticklabels(routing_labels)
    ax.set_xlabel("Modular Regime Adapter Weight $w_k(u_t)$")
    ax.set_ylabel("Input Flow Regime (* = Unseen Held-Out)")
    ax.set_title("(c) Soft Prototype Routing Weights Across Regimes", fontweight="bold")
    for i in range(6):
        for j in range(4):
            txt_c = "white" if routing_mat[i, j] > 0.55 else "black"
            ax.text(j, i, f"{routing_mat[i, j]:.2f}", ha="center", va="center", color=txt_c, fontsize=8.5)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig3_piv_latent_manifold_and_routing.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    save_json({
        "silhouette_score_seen_physics_only": sil_seen,
        "regime_id_accuracy_task_agnostic": acc_seen,
        "pca_variance_explained": var_exp.tolist(),
        "confusion_matrix": cm.tolist(),
        "routing_labels": routing_labels,
        "routing_weights_matrix": routing_mat.tolist(),
        "total_samples": int(len(Z)),
    }, os.path.join(out_dir, "latent_manifold_summary.json"))


if __name__ == "__main__":
    main()
