"""
fuzzy_logic.py
==============
Principled Neuro-Fuzzy Mechanisms for MEWC-Flow (Fuzzy-MEWC-Flow):

1. Zero-Leakage Physics-Only Invariant Extractors (2D & 1D):
   Extracts strictly flow-derived dimensionless spatial statistics from the
   velocity field u_t alone (ZERO Re or task-ID input!):
   - 2D: [KE, TKE_var, log1p(enstrophy), log1p(vort_std), log1p(div_energy)]
   - 1D: [u_energy, log1p(grad_energy), log1p(max_|du/dx|), log1p(curv_energy)]

2. Takagi-Sugeno (ANFIS) Physical Regime Router (TakagiSugenoRegimeRouter):
   - Uses Task-0 fitted standardization stats (zero future-regime leakage!).
   - Assigns each flow regime k an interpretable multi-dimensional Gaussian
     membership rule:
         mu_{k,m}(phi_m) = exp( -(phi_tilde_m - c_{k,m})^2 / (2 * sigma_{k,m}^2) )
   - Product T-norm rule firing strength (in numerically stable log-space):
         f_k(u_t) = prod_{m=1}^M mu_{k,m}(phi_m)
   - Normalized firing weights (partition of unity, sum_k w_k = 1):
         w_bar_k(u_t) = softmax( log f_k(u_t) / tau_fuzzy )
   - Exports human-readable linguistic IF-THEN physical rules.

3. 9-Rule Zero-Order Sugeno Fuzzy Stability-Plasticity Controller
   (SugenoFuzzyLambdaController):
   - Maps gradient cosine alignment alpha_t in [-1, 1] and normalized EMA
     loss drift rho_t in [-1, 1] via 3x3 Gaussian antecedents and crisp Sugeno
     consequent weights into adaptive consolidation strength:
         lambda_t in [lambda_min, lambda_max].
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


INVARIANT_NAMES_2D: Tuple[str, ...] = (
    "KE",
    "TKE_var",
    "log_Enstrophy",
    "log_Vort_std",
    "log_Div_energy",
)

INVARIANT_NAMES_1D: Tuple[str, ...] = (
    "u_energy",
    "log_Grad_energy",
    "log_Max_slope",
    "log_Curv_energy",
)


def extract_physical_invariants_2d(
    u: torch.Tensor, dy: float = 1.0, dx: float = 1.0
) -> torch.Tensor:
    """
    Compute 5 strictly flow-derived scalar physical invariants from u: (B, C, H, W).
    ZERO Reynolds-number or task-ID input enters this function.
    """
    ux = u[:, 0]
    uy = u[:, 1]
    ux_p = F.pad(ux.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    uy_p = F.pad(uy.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)

    du_dx = (ux_p[:, 1:-1, 2:] - ux_p[:, 1:-1, :-2]) / (2.0 * dx)
    du_dy = (ux_p[:, 2:, 1:-1] - ux_p[:, :-2, 1:-1]) / (2.0 * dy)
    dv_dx = (uy_p[:, 1:-1, 2:] - uy_p[:, 1:-1, :-2]) / (2.0 * dx)
    dv_dy = (uy_p[:, 2:, 1:-1] - uy_p[:, :-2, 1:-1]) / (2.0 * dy)

    vort = dv_dx - du_dy
    div = du_dx + dv_dy

    ke = 0.5 * (ux.pow(2) + uy.pow(2)).mean(dim=(-2, -1))
    tke_var = ux.var(dim=(-2, -1), unbiased=False) + uy.var(dim=(-2, -1), unbiased=False)
    log_enstrophy = torch.log1p(0.5 * vort.pow(2).mean(dim=(-2, -1)))
    log_vort_std = torch.log1p(vort.std(dim=(-2, -1), unbiased=False))
    log_div_e = torch.log1p(div.pow(2).mean(dim=(-2, -1)))

    return torch.stack([ke, tke_var, log_enstrophy, log_vort_std, log_div_e], dim=-1)


def extract_physical_invariants_1d(u: torch.Tensor) -> torch.Tensor:
    """
    Compute 4 strictly flow-derived scalar physical invariants from u: (B, Nx).
    ZERO viscosity or Reynolds-number input enters this function.
    """
    if u.dim() == 3:
        u = u.squeeze(1)
    nx = u.shape[-1]
    dx = 1.0 / float(nx)
    du_dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)
    d2u_dx2 = (torch.roll(u, -1, dims=-1) - 2.0 * u + torch.roll(u, 1, dims=-1)) / (dx * dx)

    u_energy = u.pow(2).mean(dim=-1)
    log_grad_e = torch.log1p(du_dx.pow(2).mean(dim=-1))
    log_max_slope = torch.log1p(du_dx.abs().amax(dim=-1))
    log_curv_e = torch.log1p(d2u_dx2.pow(2).mean(dim=-1))

    return torch.stack([u_energy, log_grad_e, log_max_slope, log_curv_e], dim=-1)


class TakagiSugenoRegimeRouter(nn.Module):
    """
    Interpretable ANFIS / Takagi-Sugeno Physical Regime Router.

    Each learned regime k in {0, ..., K-1} has M Gaussian membership functions
    defined over standardized physical flow invariants phi_tilde in R^M:
        mu_{k,m}(phi_tilde_m) = exp( - (phi_tilde_m - c_{k,m})^2 / (2 * sigma_{k,m}^2) )
    Product T-norm firing strengths in log-space yield normalized fuzzy adapter
    weights sum_k w_bar_k = 1 without any Re label or task-ID input.
    """

    def __init__(
        self,
        is_1d: bool = False,
        n_features: Optional[int] = None,
        tau_fuzzy: float = 1.0,
        sigma_min: float = 0.20,
    ):
        super().__init__()
        self.is_1d = bool(is_1d)
        self.n_features = int(n_features if n_features is not None else (4 if is_1d else 5))
        self.tau_fuzzy = float(tau_fuzzy)
        self.sigma_min = float(sigma_min)
        self.feature_names = INVARIANT_NAMES_1D if self.is_1d else INVARIANT_NAMES_2D

        # Standardization stats fitted ONLY on Task-0 (zero future-regime leakage!)
        self.register_buffer("feat_mean", torch.zeros(self.n_features))
        self.register_buffer("feat_std", torch.ones(self.n_features))
        self.register_buffer("stats_initialized", torch.tensor(False, dtype=torch.bool))

        # Dynamic ANFIS membership parameters: (K, M)
        self.centers = nn.Parameter(torch.zeros(0, self.n_features))
        self.log_sigmas = nn.Parameter(torch.zeros(0, self.n_features))
        self.rule_re: List[float] = []

    def extract_invariants(
        self, u: torch.Tensor, dy: float = 1.0, dx: float = 1.0
    ) -> torch.Tensor:
        if self.is_1d:
            return extract_physical_invariants_1d(u)
        return extract_physical_invariants_2d(u, dy=dy, dx=dx)

    @torch.no_grad()
    def fit_task0_normalization(self, raw_feats: torch.Tensor):
        """Fit baseline physical feature standardization strictly on Task-0 samples."""
        if bool(self.stats_initialized.item()) or raw_feats.shape[0] == 0:
            return
        dev = self.feat_mean.device
        rf = raw_feats.to(dev).float()
        mu = rf.mean(dim=0)
        sd = rf.std(dim=0, unbiased=False).clamp_min(1e-4)
        self.feat_mean.copy_(mu)
        self.feat_std.copy_(sd)
        self.stats_initialized.fill_(True)

    def _standardize(self, raw_feats: torch.Tensor) -> torch.Tensor:
        return (raw_feats - self.feat_mean.to(raw_feats.device)) / self.feat_std.to(
            raw_feats.device
        ).clamp_min(1e-4)

    def _sigmas(self) -> torch.Tensor:
        return F.softplus(self.log_sigmas) + self.sigma_min

    @torch.no_grad()
    def register_regime_rule(
        self,
        task_id: int,
        raw_feats: torch.Tensor,
        re_norm: Optional[float] = None,
    ):
        """
        Initialize or update rule k's Gaussian membership centers & spreads from
        training physical invariants of task k.
        """
        if not bool(self.stats_initialized.item()) and task_id == 0:
            self.fit_task0_normalization(raw_feats)

        dev = self.feat_mean.device
        std_feats = self._standardize(raw_feats.to(dev).float())
        mu_k = std_feats.mean(dim=0)
        sig_k = std_feats.std(dim=0, unbiased=False).clamp_min(max(self.sigma_min + 0.05, 0.35))
        # Invert softplus(x) + sigma_min = sig_k -> log(exp(y) - 1)
        y = (sig_k - self.sigma_min).clamp_min(1e-3)
        raw_log_sig = torch.log(torch.expm1(y).clamp_min(1e-4))

        K = self.centers.shape[0]
        if task_id < K:
            c_data = self.centers.data.clone()
            s_data = self.log_sigmas.data.clone()
            c_data[task_id] = mu_k
            s_data[task_id] = raw_log_sig
            self.centers = nn.Parameter(c_data.to(dev))
            self.log_sigmas = nn.Parameter(s_data.to(dev))
            if re_norm is not None and task_id < len(self.rule_re):
                self.rule_re[task_id] = float(re_norm)
        else:
            # Grow rules up to task_id
            while self.centers.shape[0] <= task_id:
                new_c = torch.cat([self.centers.data, mu_k.unsqueeze(0)], dim=0)
                new_s = torch.cat([self.log_sigmas.data, raw_log_sig.unsqueeze(0)], dim=0)
                self.centers = nn.Parameter(new_c.to(dev))
                self.log_sigmas = nn.Parameter(new_s.to(dev))
                self.rule_re.append(float(re_norm) if re_norm is not None else float(task_id))

    def forward(
        self,
        u: torch.Tensor,
        dy: float = 1.0,
        dx: float = 1.0,
        n_active: Optional[int] = None,
        routing_mode: str = "soft",
        return_memberships: bool = False,
    ):
        """
        Compute normalized Takagi-Sugeno firing weights w_bar: (B, K).
        Operates STRICTLY on physical flow invariants (zero Re input).
        """
        B = u.shape[0]
        dev = u.device
        K = self.centers.shape[0] if n_active is None else min(int(n_active), self.centers.shape[0])

        if K == 0:
            w = torch.zeros(B, 0, device=dev)
            return (w, torch.zeros(B, 0, self.n_features, device=dev)) if return_memberships else w

        raw_feats = self.extract_invariants(u, dy=dy, dx=dx)
        z = self._standardize(raw_feats)  # (B, M)

        c = self.centers[:K].to(dev)  # (K, M)
        sig = self._sigmas()[:K].to(dev)  # (K, M)

        # Scaled squared distance: (B, K, M)
        diff_sq = (z.unsqueeze(1) - c.unsqueeze(0)).pow(2) / (2.0 * sig.pow(2).unsqueeze(0))
        log_mu = -diff_sq  # (B, K, M)
        log_firing = log_mu.sum(dim=-1)  # (B, K) product T-norm in log-space

        if routing_mode == "hard":
            top_idx = log_firing.argmax(dim=-1, keepdim=True)
            w = torch.zeros(B, K, device=dev)
            w.scatter_(1, top_idx, 1.0)
        elif routing_mode == "uniform":
            w = torch.full((B, K), 1.0 / float(K), device=dev)
        else:
            w = F.softmax(log_firing / max(self.tau_fuzzy, 1e-3), dim=-1)

        if return_memberships:
            return w, torch.exp(log_mu.clamp_min(-40.0))
        return w

    def rule_classification_loss(
        self,
        u: torch.Tensor,
        task_ids: torch.Tensor,
        dy: float = 1.0,
        dx: float = 1.0,
    ) -> torch.Tensor:
        """
        End-to-end ANFIS cross-entropy loss sharpening Gaussian antecedents over
        replay-mixed multi-regime batches.
        """
        K = self.centers.shape[0]
        if K <= 1 or u.shape[0] <= 1:
            return torch.tensor(0.0, device=u.device)

        raw_feats = self.extract_invariants(u, dy=dy, dx=dx)
        z = self._standardize(raw_feats)
        c = self.centers[:K].to(u.device)
        sig = self._sigmas()[:K].to(u.device)
        log_firing = -((z.unsqueeze(1) - c.unsqueeze(0)).pow(2) / (2.0 * sig.pow(2).unsqueeze(0))).sum(dim=-1)
        logits = log_firing / max(self.tau_fuzzy, 1e-3)
        return F.cross_entropy(logits, task_ids.long().to(u.device))

    @torch.no_grad()
    def export_linguistic_rules(
        self, regime_labels: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Export learned ANFIS Gaussian rules with both standardized centers/sigmas
        and human-readable linguistic terms (VeryLow / Low / Medium / High / VeryHigh).
        """
        K = self.centers.shape[0]
        if K == 0:
            return []
        c_std = self.centers.detach().cpu().numpy()
        s_std = self._sigmas().detach().cpu().numpy()
        f_mu = self.feat_mean.detach().cpu().numpy()
        f_sd = self.feat_std.detach().cpu().numpy()

        # Assign linguistic labels relative to min..max span across rules per feature
        rules = []
        terms = ["Low", "Med-Low", "Medium", "High", "VeryHigh"]
        for k in range(K):
            label = regime_labels[k] if regime_labels and k < len(regime_labels) else f"Regime-{k}"
            antecedents = []
            phys_centers = {}
            std_centers = {}
            std_sigmas = {}
            for m, fname in enumerate(self.feature_names):
                col = c_std[:, m]
                span = max(float(col.max() - col.min()), 1e-6)
                rel = float((c_std[k, m] - col.min()) / span)
                t_idx = min(len(terms) - 1, max(0, int(round(rel * (len(terms) - 1)))))
                c_phys = float(c_std[k, m] * f_sd[m] + f_mu[m])
                s_phys = float(s_std[k, m] * f_sd[m])
                antecedents.append(
                    f"{fname} is {terms[t_idx]} (c={c_std[k, m]:+.2f}, sigma={s_std[k, m]:.2f})"
                )
                phys_centers[fname] = {"center": c_phys, "sigma": s_phys}
                std_centers[fname] = float(c_std[k, m])
                std_sigmas[fname] = float(s_std[k, m])

            rules.append(
                {
                    "rule_id": int(k),
                    "regime": label,
                    "if_then": f"IF {' AND '.join(antecedents)} THEN activate Adapter({label})",
                    "std_centers": std_centers,
                    "std_sigmas": std_sigmas,
                    "phys_centers": phys_centers,
                }
            )
        return rules


class SugenoFuzzyLambdaController:
    """
    9-Rule Zero-Order Sugeno Fuzzy Controller for Dynamic EWC Consolidation.

    Inputs:
      1. alpha_t in [-1, 1]   : EMA task-gradient cosine alignment
         Membership sets: Conflict (c=-1.0), Neutral (c=0.0), Aligned (c=+1.0), sigma=0.55
      2. drift_rho in [-1, 1] : Normalized relative loss increase vs. running baseline
         Membership sets: LowDrift (c=-0.6), ModDrift (c=0.0), HighDrift (c=+0.6), sigma=0.45

    Consequent Matrix u_ij in [0, 1] (rows: Conflict, Neutral, Aligned; cols: High, Mod, Low drift):
         Conflict  + High/Mod/Low -> [1.00, 0.85, 0.65]
         Neutral   + High/Mod/Low -> [0.75, 0.45, 0.25]
         Aligned   + High/Mod/Low -> [0.40, 0.15, 0.00]

    Output:
         lambda_t = lambda_min + u_norm * (lambda_max - lambda_min)
    """

    ALPHA_CENTERS = (-1.0, 0.0, 1.0)
    ALPHA_SIGMA = 0.55
    DRIFT_CENTERS = (-0.6, 0.0, 0.6)  # LowDrift, ModDrift, HighDrift
    DRIFT_SIGMA = 0.45

    # Rows = [Conflict, Neutral, Aligned]; Cols = [LowDrift, ModDrift, HighDrift]
    CONSEQUENTS = (
        (0.65, 0.85, 1.00),
        (0.25, 0.45, 0.75),
        (0.00, 0.15, 0.40),
    )

    @classmethod
    def infer(
        cls,
        alpha_t: float,
        drift_rho: float,
        lambda_min: float,
        lambda_max: float,
    ) -> Tuple[float, float, np.ndarray]:
        a = float(max(-1.0, min(1.0, alpha_t)))
        r = float(max(-1.0, min(1.0, drift_rho)))

        mu_a = [
            math.exp(-((a - c) ** 2) / (2.0 * cls.ALPHA_SIGMA ** 2))
            for c in cls.ALPHA_CENTERS
        ]
        mu_r = [
            math.exp(-((r - c) ** 2) / (2.0 * cls.DRIFT_SIGMA ** 2))
            for c in cls.DRIFT_CENTERS
        ]

        firing = np.zeros((3, 3), dtype=np.float64)
        num = 0.0
        den = 0.0
        for i in range(3):
            for j in range(3):
                w = mu_a[i] * mu_r[j]
                firing[i, j] = w
                num += w * cls.CONSEQUENTS[i][j]
                den += w

        u_norm = float(num / max(den, 1e-12))
        lam = float(lambda_min + u_norm * (lambda_max - lambda_min))
        return lam, u_norm, firing

    @classmethod
    def control_surface_grid(
        cls, lambda_min: float, lambda_max: float, n: int = 41
    ) -> Dict[str, List]:
        alphas = np.linspace(-1.0, 1.0, n)
        drifts = np.linspace(-1.0, 1.0, n)
        Z = np.zeros((n, n), dtype=np.float64)
        for ii, r in enumerate(drifts):
            for jj, a in enumerate(alphas):
                lam, _, _ = cls.infer(float(a), float(r), lambda_min, lambda_max)
                Z[ii, jj] = lam
        return {
            "alphas": alphas.tolist(),
            "drifts": drifts.tolist(),
            "lambda_surface": Z.tolist(),
        }
