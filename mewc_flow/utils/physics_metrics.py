"""
physics_metrics.py
==================
Comprehensive Fluid-Dynamics Physical Verification, Spectral Error Analysis,
Multi-Window Autoregressive Rollouts, and Slice-Based Spatial Error Diagnostics.

Provides:
- Differentiable PyTorch vorticity & divergence consistency loss for training
- Anisotropic 2D vorticity & divergence computation (exact aspect-ratio dy, dx)
- 2D radially-averaged Kinetic Energy Spectrum E(k) & Low/Mid/High-k spectral error
- Wake shedding peak frequency (Strouhal proxy) & temporal FFT correlation
- Multi-window autoregressive rollouts (mean +/- std across multiple ICs + VPH)
- Slice-based spatial error maps (PIV near-wake/far-wake/free-stream; ERA5 NH/Tropics/SH)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Differentiable PyTorch spatial-derivative physics loss (for training)
# ---------------------------------------------------------------------------

def derivative_consistency_loss_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    dy: float = 1.0,
    dx: float = 1.0,
    periodic: bool = False,
) -> torch.Tensor:
    """
    Differentiable vorticity + divergence matching loss:
        L_phys = MSE(omega(pred), omega(target)) + 0.5 * MSE(div(pred), div(target))
    Matching BOTH planar vorticity and planar divergence preserves physical curl
    dynamics while remaining valid for 3-D turbulent PIV slices (where div_2D = -dw/dz != 0)
    AND exact 2-D incompressible Navier-Stokes (where div_2D == 0).
    """
    if periodic:
        du_dx_p = (torch.roll(pred[:, 0], -1, dims=-1) - torch.roll(pred[:, 0], 1, dims=-1)) / (2.0 * dx)
        du_dy_p = (torch.roll(pred[:, 0], -1, dims=-2) - torch.roll(pred[:, 0], 1, dims=-2)) / (2.0 * dy)
        dv_dx_p = (torch.roll(pred[:, 1], -1, dims=-1) - torch.roll(pred[:, 1], 1, dims=-1)) / (2.0 * dx)
        dv_dy_p = (torch.roll(pred[:, 1], -1, dims=-2) - torch.roll(pred[:, 1], 1, dims=-2)) / (2.0 * dy)

        du_dx_t = (torch.roll(target[:, 0], -1, dims=-1) - torch.roll(target[:, 0], 1, dims=-1)) / (2.0 * dx)
        du_dy_t = (torch.roll(target[:, 0], -1, dims=-2) - torch.roll(target[:, 0], 1, dims=-2)) / (2.0 * dy)
        dv_dx_t = (torch.roll(target[:, 1], -1, dims=-1) - torch.roll(target[:, 1], 1, dims=-1)) / (2.0 * dx)
        dv_dy_t = (torch.roll(target[:, 1], -1, dims=-2) - torch.roll(target[:, 1], 1, dims=-2)) / (2.0 * dy)
    else:
        # Interior central differences
        du_dx_p = (pred[:, 0, 1:-1, 2:] - pred[:, 0, 1:-1, :-2]) / (2.0 * dx)
        du_dy_p = (pred[:, 0, 2:, 1:-1] - pred[:, 0, :-2, 1:-1]) / (2.0 * dy)
        dv_dx_p = (pred[:, 1, 1:-1, 2:] - pred[:, 1, 1:-1, :-2]) / (2.0 * dx)
        dv_dy_p = (pred[:, 1, 2:, 1:-1] - pred[:, 1, :-2, 1:-1]) / (2.0 * dy)

        du_dx_t = (target[:, 0, 1:-1, 2:] - target[:, 0, 1:-1, :-2]) / (2.0 * dx)
        du_dy_t = (target[:, 0, 2:, 1:-1] - target[:, 0, :-2, 1:-1]) / (2.0 * dy)
        dv_dx_t = (target[:, 1, 1:-1, 2:] - target[:, 1, 1:-1, :-2]) / (2.0 * dx)
        dv_dy_t = (target[:, 1, 2:, 1:-1] - target[:, 1, :-2, 1:-1]) / (2.0 * dy)

    vort_p = dv_dx_p - du_dy_p
    vort_t = dv_dx_t - du_dy_t
    div_p = du_dx_p + dv_dy_p
    div_t = du_dx_t + dv_dy_t

    return F.mse_loss(vort_p, vort_t) + 0.5 * F.mse_loss(div_p, div_t)


# ---------------------------------------------------------------------------
# NumPy Physical & Spectral Diagnostics (for Evaluation & Verification)
# ---------------------------------------------------------------------------

def compute_vorticity_and_divergence_2d(
    uv: np.ndarray,
    dy: float = 1.0,
    dx: float = 1.0,
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D planar vorticity (dv/dx - du/dy) and divergence (du/dx + dv/dy)
    respecting exact anisotropic grid spacing (dy, dx).

    uv : (..., 2, H, W)
    Returns (vort, div) each of shape (..., H, W).
    """
    u = uv[..., 0, :, :]
    v = uv[..., 1, :, :]
    if periodic:
        du_dx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dx)
        du_dy = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2.0 * dy)
        dv_dx = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dx)
        dv_dy = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2.0 * dy)
    else:
        du_dy = np.gradient(u, dy, axis=-2)
        du_dx = np.gradient(u, dx, axis=-1)
        dv_dy = np.gradient(v, dy, axis=-2)
        dv_dx = np.gradient(v, dx, axis=-1)
    return (dv_dx - du_dy), (du_dx + dv_dy)


def radial_energy_spectrum_2d(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic radially-averaged 2D kinetic energy spectrum E(k).

    uv : (..., 2, H, W)
    Returns:
      k_bins : integer radial wavenumbers [1 .. k_max]
      E_k    : mean spectral kinetic energy at each shell k
    """
    arr = uv.reshape(-1, 2, uv.shape[-2], uv.shape[-1])
    _, _, H, W = arr.shape
    u_hat = np.fft.fft2(arr[:, 0], norm="ortho")
    v_hat = np.fft.fft2(arr[:, 1], norm="ortho")
    e2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2).mean(axis=0)  # (H, W)

    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky)
    kr = np.round(np.sqrt(KX**2 + KY**2)).astype(int)

    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    E_k = np.array([float(e2d[kr == k].sum()) for k in k_bins], dtype=np.float64)
    return k_bins, E_k


def spectral_band_errors(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute log-spectral RMSE across All, Low (k<=4), Mid (5<=k<=12), High (k>12) bands."""
    k_bins, e_true = radial_energy_spectrum_2d(true)
    _, e_pred = radial_energy_spectrum_2d(pred)

    log_diff = np.log10(e_pred + 1e-12) - np.log10(e_true + 1e-12)
    low = k_bins <= 4
    mid = (k_bins >= 5) & (k_bins <= 12)
    high = k_bins > 12

    return {
        "k_bins": k_bins.tolist(),
        "E_true": e_true.tolist(),
        "E_pred": e_pred.tolist(),
        "log_spec_rmse_all": float(np.sqrt(np.mean(log_diff**2))),
        "log_spec_rmse_low": float(np.sqrt(np.mean(log_diff[low]**2))) if low.any() else 0.0,
        "log_spec_rmse_mid": float(np.sqrt(np.mean(log_diff[mid]**2))) if mid.any() else 0.0,
        "log_spec_rmse_high": float(np.sqrt(np.mean(log_diff[high]**2))) if high.any() else 0.0,
    }


def wake_shedding_frequency_metrics(pred_trajs: np.ndarray, true_trajs: np.ndarray) -> Dict[str, float]:
    """
    Evaluate vortex-shedding temporal dynamics from cross-stream velocity v(t)
    sampled across wake centerline probes (rows H//4..3H//4, cols W//3..2W//3).

    pred_trajs, true_trajs : (N_win, T, 2, H, W)
    """
    _, T, _, H, W = true_trajs.shape
    v_true = true_trajs[:, :, 1, H // 4 : 3 * H // 4, W // 3 : 2 * W // 3].mean(axis=(-2, -1))
    v_pred = pred_trajs[:, :, 1, H // 4 : 3 * H // 4, W // 3 : 2 * W // 3].mean(axis=(-2, -1))

    # Remove DC mean along time axis
    vt = v_true - v_true.mean(axis=1, keepdims=True)
    vp = v_pred - v_pred.mean(axis=1, keepdims=True)

    # Spectral power along time
    pt = np.abs(np.fft.rfft(vt, axis=1))[:, 1:]  # exclude DC
    pp = np.abs(np.fft.rfft(vp, axis=1))[:, 1:]
    freqs = np.fft.rfftfreq(T, d=1.0)[1:]

    # Dominant frequency error (Strouhal-number proxy since St = f * D / U_inf)
    f_true = freqs[np.argmax(pt, axis=1)]
    f_pred = freqs[np.argmax(pp, axis=1)]
    rel_freq_err = np.mean(np.abs(f_pred - f_true) / (f_true + 1e-8))

    # Temporal power spectrum correlation
    corrs = []
    for i in range(len(pt)):
        if pt[i].std() > 1e-8 and pp[i].std() > 1e-8:
            corrs.append(float(np.corrcoef(pt[i], pp[i])[0, 1]))
    return {
        "dominant_freq_rel_err": float(rel_freq_err),
        "temporal_spectrum_corr": float(np.mean(corrs)) if corrs else 0.0,
    }


@torch.no_grad()
def multi_window_rollout_eval(
    model: torch.nn.Module,
    dataset,
    re_norm: float,
    n_steps: int = 20,
    stride: int = 3,
    device: str = "cpu",
    routing_mode: str = "soft",
    task_id: Optional[int] = None,
    dy: float = 1.0,
    dx: float = 1.0,
    periodic: bool = False,
    max_windows: int = 12,
) -> Dict:
    """
    Run autoregressive rollouts across multiple distinct initial conditions
    (NOT just 1 IC!), returning:
    - per-step mean and std relative L2 trajectory (t = 1..n_steps)
    - Valid Prediction Horizon (VPH) at thresholds 0.25 and 0.50
    - Vorticity RMSE, Divergence RMSE, Kinetic Energy & Enstrophy relative errors
    - 2D Radially-averaged energy spectrum E(k) & Low/Mid/High-k errors
    - Spatial RMSE map (H, W) for slice-based regional error analysis
    - Sample predicted/true trajectories for plotting
    """
    model.eval()
    model.to(device)
    is_modular = hasattr(model, "adapters")

    windows = dataset.get_rollout_windows(n_steps=n_steps, stride=stride)
    if not windows:
        # Fall back to longest available segment
        seg = dataset.segments[0]
        n_steps = min(n_steps, len(seg) - 1)
        windows = [seg[: n_steps + 1]]
    windows = windows[:max_windows]

    # Stack initial states: (N_win, 2, H, W)
    win_arr = np.stack(windows, axis=0)  # (N_win, n_steps+1, 2, H, W)
    N_win = win_arr.shape[0]
    u_curr = torch.from_numpy(win_arr[:, 0]).to(device)
    re_t = torch.full((N_win, 1), float(re_norm), dtype=torch.float32, device=device)

    preds = []
    for _ in range(n_steps):
        if is_modular:
            u_next = model(u_curr, re_t, task_id=task_id, routing_mode=routing_mode, dy=dy, dx=dx)
        else:
            u_next = model(u_curr, re_t)
        preds.append(u_next.cpu().numpy())
        u_curr = u_next

    pred_trajs = np.stack(preds, axis=1)        # (N_win, n_steps, 2, H, W)
    true_trajs = win_arr[:, 1 : n_steps + 1]    # (N_win, n_steps, 2, H, W)

    # Step-wise relative L2 error per window -> (N_win, n_steps)
    diff_norms = np.linalg.norm((pred_trajs - true_trajs).reshape(N_win, n_steps, -1), axis=-1)
    true_norms = np.linalg.norm(true_trajs.reshape(N_win, n_steps, -1), axis=-1) + 1e-8
    rel_l2_mat = diff_norms / true_norms

    mean_curve = rel_l2_mat.mean(axis=0)
    std_curve = rel_l2_mat.std(axis=0)

    def vph(thresh: float) -> float:
        horizons = []
        for w in range(N_win):
            exceed = np.where(rel_l2_mat[w] > thresh)[0]
            horizons.append(float(exceed[0]) if len(exceed) > 0 else float(n_steps))
        return float(np.mean(horizons))

    # Physical derivative diagnostics across all rollout steps & windows
    vort_true, div_true = compute_vorticity_and_divergence_2d(true_trajs, dy=dy, dx=dx, periodic=periodic)
    vort_pred, div_pred = compute_vorticity_and_divergence_2d(pred_trajs, dy=dy, dx=dx, periodic=periodic)

    vort_rmse = float(np.sqrt(np.mean((vort_pred - vort_true)**2)))
    div_rmse = float(np.sqrt(np.mean((div_pred - div_true)**2)))
    div_pred_rms = float(np.sqrt(np.mean(div_pred**2)))

    # Kinetic energy & enstrophy conservation error
    ke_true = 0.5 * (true_trajs[:, :, 0]**2 + true_trajs[:, :, 1]**2).mean(axis=(-2, -1))
    ke_pred = 0.5 * (pred_trajs[:, :, 0]**2 + pred_trajs[:, :, 1]**2).mean(axis=(-2, -1))
    ke_rel_err = float(np.mean(np.abs(ke_pred - ke_true) / (ke_true + 1e-8)))

    ens_true = 0.5 * (vort_true**2).mean(axis=(-2, -1))
    ens_pred = 0.5 * (vort_pred**2).mean(axis=(-2, -1))
    ens_rel_err = float(np.mean(np.abs(ens_pred - ens_true) / (ens_true + 1e-8)))

    # Spectral energy E(k) error at mid-rollout (step min(9, n_steps-1)) and full rollout
    t_mid = min(9, n_steps - 1)
    spec_metrics = spectral_band_errors(pred_trajs[:, t_mid], true_trajs[:, t_mid])

    # Shedding frequency / temporal FFT diagnostics
    freq_metrics = wake_shedding_frequency_metrics(pred_trajs, true_trajs)

    # Spatial error map averaged over windows and steps: (H, W)
    spatial_rmse_map = np.sqrt(np.mean((pred_trajs - true_trajs)**2, axis=(0, 1, 2)))

    return {
        "n_windows": int(N_win),
        "n_steps": int(n_steps),
        "rel_l2_mean": mean_curve.tolist(),
        "rel_l2_std": std_curve.tolist(),
        "step1_rel_l2": float(mean_curve[0]),
        "step5_rel_l2": float(mean_curve[min(4, n_steps - 1)]),
        "step10_rel_l2": float(mean_curve[min(9, n_steps - 1)]),
        "step_final_rel_l2": float(mean_curve[-1]),
        "vph_025": vph(0.25),
        "vph_050": vph(0.50),
        "vorticity_rmse": vort_rmse,
        "divergence_rmse": div_rmse,
        "divergence_pred_rms": div_pred_rms,
        "ke_rel_err": ke_rel_err,
        "enstrophy_rel_err": ens_rel_err,
        "spectral": spec_metrics,
        "temporal_freq": freq_metrics,
        "spatial_rmse_map": spatial_rmse_map,
        "sample_pred": pred_trajs[0],
        "sample_true": true_trajs[0],
    }


def slice_error_analysis_piv(spatial_rmse_map: np.ndarray) -> Dict[str, float]:
    """Slice PIV spatial error into Near-Wake Core, Far-Wake, and Free-Stream zones."""
    H, W = spatial_rmse_map.shape
    wake_y = slice(H // 4, 3 * H // 4)
    near_wake = spatial_rmse_map[wake_y, : W // 2]
    far_wake = spatial_rmse_map[wake_y, W // 2 :]
    free_stream = np.concatenate([spatial_rmse_map[: H // 4, :], spatial_rmse_map[3 * H // 4 :, :]], axis=0)
    return {
        "near_wake_rmse": float(near_wake.mean()),
        "far_wake_rmse": float(far_wake.mean()),
        "free_stream_rmse": float(free_stream.mean()),
    }


def slice_error_analysis_era5(spatial_rmse_map: np.ndarray) -> Dict[str, float]:
    """Slice global ERA5 error (32 lat x 64 lon) into SH Mid-Lat, Tropics, and NH Storm Track."""
    H, _ = spatial_rmse_map.shape
    # WeatherBench 2 lat runs -90 (row 0) -> +90 (row 31)
    sh_mid = spatial_rmse_map[2:12, :]
    tropics = spatial_rmse_map[12:20, :]
    nh_storm = spatial_rmse_map[20:30, :]
    return {
        "sh_midlat_rmse": float(sh_mid.mean()),
        "tropics_rmse": float(tropics.mean()),
        "nh_storm_track_rmse": float(nh_storm.mean()),
    }
