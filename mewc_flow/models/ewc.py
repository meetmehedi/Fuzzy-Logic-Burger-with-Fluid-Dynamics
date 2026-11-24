"""
ewc.py
======
Complex-Parameter Elastic Weight Consolidation (Static EWC, Online EWC / OFU,
and Gradient-Alignment Adaptive EWC from MEWC-RL++).

Mathematical Formulation (extending MEWC-RL++ Eq. 1.8-1.14 to complex operators):
--------------------------------------------------------------------------------
1. Diagonal Complex Fisher Information Matrix (strictly real float32):
       F_i = E[ |dL / d(theta_i)|^2 ]   (using |g|^2 = Re(g)^2 + Im(g)^2 for complex theta)
2. Online Fisher Updating (OFU) with exponential decay gamma_F in (0, 1]:
       F^{(k)} = gamma_F * F^{(k-1)} + (1 - gamma_F) * F_hat^{(k)}
   (Constant O(|theta|) memory independent of the number of tasks; setting
    online_decay=None reverts to standard additive accumulation.)
3. Gradient Memory & Cosine Alignment (Adaptive EWC / A-EWC):
       g_prev^{(t)} = beta_g * g_prev^{(t-1)} + (1 - beta_g) * g_t
       alpha_t      = Re(<g_t, g_prev>) / (||g_t||_2 * ||g_prev||_2 + eps)  in [-1, 1]
       alpha_tilde  = (1 + alpha_t) / 2                                      in [0, 1]
       lambda_t     = lambda_min + (1 - alpha_tilde) * (lambda_max - lambda_min)
   When task gradients conflict (alpha_t < 0), lambda_t -> lambda_max (stronger stability);
   when task gradients align (alpha_t > 0), lambda_t -> lambda_min (higher plasticity).
4. Strictly Real Quadratic Penalty over Protected Parameters:
       L_EWC = (lambda_t / 2) * sum_i F_i * |theta_i - theta*_i|^2

Fixes over previous implementation:
-----------------------------------
- Exact batch counting (fixes n_seen // u_in.size(0) bug on partial final batches).
- Device-consistent zero tensor when unconsolidated.
- Optional parameter-name allowlist so modular MEWC-Flow protects ONLY shared
  backbone parameters while regime-specific adapters train freely.
- Exact byte-level memory accounting for Q1 matched-budget reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Iterable
import math
from mewc_flow.models.fuzzy_logic import SugenoFuzzyLambdaController


class EWC:
    """
    Static or Online (decayed) Diagonal Complex-Parameter EWC.

    Parameters
    ----------
    model         : nn.Module
    lambda_ewc    : float          — static consolidation coefficient
    n_samples     : int            — max samples for Fisher estimation
    online_decay  : Optional[float] — if None, additive EWC; if float in (0,1),
                                     Online Fisher Updating (OFU): F = decay*F + (1-decay)*F_hat
    param_names   : Optional[List[str]] — subset of parameter names to protect (None = all)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 500.0,
        n_samples: int = 200,
        online_decay: Optional[float] = None,
        param_names: Optional[Iterable[str]] = None,
    ):
        self.model = model
        self.lambda_ewc = float(lambda_ewc)
        self.n_samples = int(n_samples)
        self.online_decay = online_decay
        self.param_allowlist = set(param_names) if param_names is not None else None

        self._params_star: Optional[Dict[str, torch.Tensor]] = None
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._loss_scale: float = 1.0
        self._n_protected_params: int = 1
        self._step_scale_sq: float = 1e-6  # (r_0)^2 with r_0 = 1e-3 (natural step radius)
        self._consolidated = False

    def _is_protected(self, name: str, param: nn.Parameter) -> bool:
        if not param.requires_grad:
            return False
        if self.param_allowlist is not None and name not in self.param_allowlist:
            return False
        return True

    def consolidate(
        self,
        loader: DataLoader,
        device: str = "cpu",
        fisher_noise: float = 0.0,
        task_id: Optional[int] = None,
    ):
        """Estimate diagonal Fisher and update stored reference weights theta*."""
        was_training = self.model.training
        self.model.eval()

        params_star = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if self._is_protected(n, p)
        }
        fisher = {
            n: torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for n, p in self.model.named_parameters()
            if self._is_protected(n, p)
        }

        self.model.zero_grad(set_to_none=True)
        n_seen = 0
        n_batches = 0
        loss_sum = 0.0

        for u_in, re, u_out in loader:
            if n_seen >= self.n_samples:
                break
            u_in = u_in.to(device)
            re = re.to(device)
            u_out = u_out.to(device)

            if fisher_noise > 0.0:
                u_out = u_out + fisher_noise * torch.randn_like(u_out)

            if task_id is not None and hasattr(self.model, "adapters"):
                pred = self.model(u_in, re, task_id=task_id)
            else:
                pred = self.model(u_in, re)

            loss = nn.functional.mse_loss(pred, u_out)
            loss_sum += float(loss.item())
            loss.backward()

            for n, p in self.model.named_parameters():
                if n in fisher and p.grad is not None:
                    g = p.grad.detach()
                    term = (g.real.pow(2) + g.imag.pow(2)) if p.is_complex() else g.pow(2)
                    fisher[n] += term.float()

            self.model.zero_grad(set_to_none=True)
            n_seen += u_in.size(0)
            n_batches += 1

        denom = float(max(1, n_batches))
        task_mse = max(loss_sum / denom, 1e-8)
        total_sum = 0.0
        total_elems = 0
        for n in fisher:
            fisher[n] /= denom
            total_sum += float(fisher[n].sum().item())
            total_elems += fisher[n].numel()

        # Empirical-Bayes scale-invariant Fisher normalization: E[F_tilde] = 1.0
        mean_f = total_sum / max(1, total_elems)
        if mean_f > 1e-18:
            for n in fisher:
                fisher[n] /= mean_f

        self._n_protected_params = max(1, total_elems)
        if self._fisher is None:
            self._fisher = fisher
            self._loss_scale = task_mse
        elif self.online_decay is not None:
            g = float(self.online_decay)
            for n in fisher:
                self._fisher[n] = g * self._fisher[n] + (1.0 - g) * fisher[n]
            self._loss_scale = g * self._loss_scale + (1.0 - g) * task_mse
        else:
            for n in fisher:
                self._fisher[n] = self._fisher[n] + fisher[n]
            self._loss_scale = 0.5 * self._loss_scale + 0.5 * task_mse

        self._params_star = params_star
        self._consolidated = True
        if was_training:
            self.model.train()

    def update_fisher(self, loader: DataLoader, device: str = "cpu", fisher_noise: float = 0.0, task_id: Optional[int] = None):
        return self.consolidate(loader, device=device, fisher_noise=fisher_noise, task_id=task_id)

    def current_lambda(self) -> float:
        return self.lambda_ewc

    def penalty(self, model: nn.Module) -> torch.Tensor:
        dev = next(model.parameters()).device
        if not self._consolidated or self.current_lambda() <= 0.0:
            return torch.tensor(0.0, device=dev)

        loss = torch.tensor(0.0, device=dev)
        for n, p in model.named_parameters():
            if n in self._fisher and p.requires_grad:
                omega = self._fisher[n]
                p_star = self._params_star[n]
                if omega.device != dev:
                    omega = omega.to(dev)
                    self._fisher[n] = omega
                if p_star.device != dev:
                    p_star = p_star.to(dev)
                    self._params_star[n] = p_star
                diff_raw = p - p_star
                diff_sq = (
                    diff_raw.real.pow(2) + diff_raw.imag.pow(2)
                    if p.is_complex()
                    else diff_raw.pow(2)
                )
                loss = loss + (omega * diff_sq).sum()
        mean_drift = loss / (float(self._n_protected_params) * self._step_scale_sq)
        return 0.5 * self.current_lambda() * self._loss_scale * mean_drift

    def memory_bytes(self) -> int:
        """Exact memory footprint of stored theta* and diagonal Fisher F (in bytes)."""
        if not self._consolidated:
            return 0
        total = 0
        for t in self._params_star.values():
            total += t.numel() * t.element_size()
        for t in self._fisher.values():
            total += t.numel() * t.element_size()
        return total


class AdaptiveEWC(EWC):
    """
    Gradient-Alignment Adaptive EWC (A-EWC) + Online Fisher Updating (OFU)
    directly extending MEWC-RL++ (Eq. 1.8 - 1.14) to complex neural operators.

    Maintains a single constant-memory EMA gradient vector g_prev over the
    protected shared backbone parameters and dynamically modulates lambda_t
    between [lambda_min, lambda_max] based on real complex-gradient alignment.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_min: float = 10.0,
        lambda_max: float = 250.0,
        ema_beta: float = 0.9,
        fisher_decay: float = 0.85,
        n_samples: int = 200,
        param_names: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            model=model,
            lambda_ewc=0.5 * (lambda_min + lambda_max),
            n_samples=n_samples,
            online_decay=fisher_decay,
            param_names=param_names,
        )
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.ema_beta = float(ema_beta)

        self._grad_ema: Optional[Dict[str, torch.Tensor]] = None
        self._active_lambda: float = 0.5 * (lambda_min + lambda_max)
        self.alignment_history: List[float] = []
        self.lambda_history: List[float] = []

    def current_lambda(self) -> float:
        return self._active_lambda

    @torch.no_grad()
    def update_from_task_gradient(self, model: nn.Module, current_loss: Optional[float] = None) -> float:
        """
        Read current task-loss gradients on protected parameters, compute cosine
        alignment alpha with g_prev, adapt lambda_t, and update EMA g_prev.
        Must be called after task-loss backward() and BEFORE adding EWC gradient
        (or using detached task-loss gradients).
        """
        grads = {}
        for n, p in model.named_parameters():
            if self._is_protected(n, p) and p.grad is not None:
                grads[n] = p.grad.detach().clone()

        if not grads:
            return 0.0

        if self._grad_ema is None:
            self._grad_ema = {n: g.clone() for n, g in grads.items()}
            self._active_lambda = self.lambda_min
            return 1.0

        # Real inner product over real & complex parameters on-device (single GPU sync!)
        dev = next(iter(grads.values())).device
        dot = torch.tensor(0.0, device=dev)
        norm_curr_sq = torch.tensor(0.0, device=dev)
        norm_prev_sq = torch.tensor(0.0, device=dev)
        b = self.ema_beta
        for n, g in grads.items():
            if n not in self._grad_ema:
                self._grad_ema[n] = g.clone()
                continue
            gp = self._grad_ema[n]
            if gp.device != dev:
                gp = gp.to(dev)
            if g.is_complex():
                dot = dot + (g.real * gp.real + g.imag * gp.imag).sum()
                norm_curr_sq = norm_curr_sq + (g.real.pow(2) + g.imag.pow(2)).sum()
                norm_prev_sq = norm_prev_sq + (gp.real.pow(2) + gp.imag.pow(2)).sum()
            else:
                dot = dot + (g * gp).sum()
                norm_curr_sq = norm_curr_sq + g.pow(2).sum()
                norm_prev_sq = norm_prev_sq + gp.pow(2).sum()
            # Update EMA gradient memory (Eq. 1.9) on-device
            self._grad_ema[n] = b * gp + (1.0 - b) * g

        alpha_t = dot / (norm_curr_sq.sqrt() * norm_prev_sq.sqrt() + 1e-12)
        alpha = max(-1.0, min(1.0, float(alpha_t.item())))
        self._apply_lambda_rule(alpha, current_loss=current_loss)
        return alpha

    def _apply_lambda_rule(self, alpha: float, current_loss: Optional[float] = None):
        alpha_tilde = 0.5 * (1.0 + alpha)
        # Eq. 1.11 from MEWC-RL++: conflicting gradients (alpha < 0) -> larger lambda
        self._active_lambda = self.lambda_min + (1.0 - alpha_tilde) * (self.lambda_max - self.lambda_min)
        self.alignment_history.append(alpha)
        self.lambda_history.append(self._active_lambda)

    def memory_bytes(self) -> int:
        base = super().memory_bytes()
        ema_b = (
            sum(t.numel() * t.element_size() for t in self._grad_ema.values())
            if self._grad_ema is not None
            else 0
        )
        return base + ema_b


class FuzzyAdaptiveEWC(AdaptiveEWC):
    """
    Neuro-Fuzzy Stability-Plasticity Controller for Complex A-EWC.
    Replaces linear cosine-to-lambda mapping with a 9-rule zero-order Sugeno FIS
    taking EMA gradient alignment alpha_t and normalized loss drift rho_t in [-1, 1].
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_min: float = 0.01,
        lambda_max: float = 0.15,
        ema_beta: float = 0.9,
        fisher_decay: float = 0.85,
        n_samples: int = 200,
        param_names: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            model=model,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            ema_beta=ema_beta,
            fisher_decay=fisher_decay,
            n_samples=n_samples,
            param_names=param_names,
        )
        self._loss_ema: Optional[float] = None
        self.drift_history: List[float] = []

    def _apply_lambda_rule(self, alpha: float, current_loss: Optional[float] = None):
        if current_loss is not None:
            curr = max(float(current_loss), 1e-10)
            if self._loss_ema is None:
                self._loss_ema = curr
                rho_t = 0.0
            else:
                rel_diff = (curr - self._loss_ema) / max(self._loss_ema, 1e-8)
                rho_t = float(math.tanh(2.0 * rel_diff))
                self._loss_ema = self.ema_beta * self._loss_ema + (1.0 - self.ema_beta) * curr
        else:
            rho_t = 0.0

        lam, _, _ = SugenoFuzzyLambdaController.infer(
            alpha_t=alpha,
            drift_rho=rho_t,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
        )
        self._active_lambda = lam
        self.alignment_history.append(alpha)
        self.drift_history.append(rho_t)
        self.lambda_history.append(lam)
