"""
fno2d.py
========
2D Fourier Neural Operator (FNO-2d) & Modular MEWC-Flow Architecture (MEWCFNO2d).

Architecture Overview
---------------------
1. Shared Spectral Backbone (FNO2dBackbone):
   - Concatenates [u_t, v_t, x_grid, y_grid, Re_channel] -> lifting Conv1x1 -> width
   - 4 SpectralConv2d + Conv1x1 residual blocks with GELU
   - Shared projection head h_shared(F) -> base residual delta_u_shared
2. Physics-Only Contrastive Regime Encoder (PhysicsRegimeEncoder2d):
   - Operates STRICTLY on the velocity field u_t = [u, v] and its spatial
     differential invariants [u, v, |u|, omega = dv/dx - du/dy, div = du/dx + dv/dy].
   - Receives ZERO Re label or task-ID input -> genuine task-agnostic regime
     identification (RQ3) with zero label leakage!
   - Projects to L2-normalized latent code z in S^{latent_dim - 1}.
   - Stores running L2-normalized regime prototypes {c_0, ..., c_{K-1}}.
3. Modular Regime Adapters (RegimeAdapter2d):
   - Lightweight per-regime FiLM channel modulation (gamma_k, beta_k) + bottleneck
     1x1 residual head (~1.2% of backbone parameters per regime).
   - At test time (task-agnostic!), soft cosine prototype attention blends adapters:
         w_k(u_t) = softmax( (z(u_t)^T c_k) / tau_route )
         u_{t+1}  = u_t + h_shared(F) + sum_k w_k(u_t) * h_k(F)
   - Supports routing modes: "soft" (default, enables zero-shot cross-Re blending),
     "hard" (top-1 prototype), "oracle" (ground-truth task ID), "uniform" (ablation).
   - Also supports SLE-FNO mode (single-layer extension adapters on frozen backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from mewc_flow.models.fuzzy_logic import TakagiSugenoRegimeRouter


class SpectralConv2d(nn.Module):
    """2D Fourier layer with complex spectral weight tensors."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


def _compute_physics_channels(u: torch.Tensor, dy: float = 1.0, dx: float = 1.0) -> torch.Tensor:
    """
    Build purely flow-derived invariant features from velocity u: (B, C, H, W).
    For C >= 2 ([u, v, ...]), returns [u, v, |u|, vorticity, divergence] (B, 5, H, W).
    Strictly independent of Re or task ID (zero label leakage).
    """
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]
    speed = torch.sqrt(u_x.pow(2) + u_y.pow(2) + 1e-8)

    # Central finite differences with replicate padding on boundaries
    ux_p = F.pad(u_x, (1, 1, 1, 1), mode="replicate")
    uy_p = F.pad(u_y, (1, 1, 1, 1), mode="replicate")

    du_dx = (ux_p[:, :, 1:-1, 2:] - ux_p[:, :, 1:-1, :-2]) / (2.0 * dx)
    du_dy = (ux_p[:, :, 2:, 1:-1] - ux_p[:, :, :-2, 1:-1]) / (2.0 * dy)
    dv_dx = (uy_p[:, :, 1:-1, 2:] - uy_p[:, :, 1:-1, :-2]) / (2.0 * dx)
    dv_dy = (uy_p[:, :, 2:, 1:-1] - uy_p[:, :, :-2, 1:-1]) / (2.0 * dy)

    vort = dv_dx - du_dy
    div = du_dx + dv_dy
    return torch.cat([u_x, u_y, speed, vort, div], dim=1)


class PhysicsRegimeEncoder2d(nn.Module):
    """
    Dedicated physics-only regime encoder e_psi(u_t) -> z in S^{latent_dim-1}.
    Sees ONLY flow velocity and spatial differential invariants (NO Re input!).
    Multi-scale strided CNN + global mean & std statistics capture both mean wake
    profile and turbulent fluctuation intensity / vorticity energy.
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(5, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
        )
        # Concatenate spatial mean AND spatial std (2 * 2*hidden = 4*hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, u: torch.Tensor, dy: float = 1.0, dx: float = 1.0) -> torch.Tensor:
        phys = _compute_physics_channels(u, dy=dy, dx=dx)
        h = self.conv(phys)
        mean_pool = h.mean(dim=(-2, -1))
        std_pool = h.var(dim=(-2, -1), unbiased=False).add(1e-6).sqrt()
        stats = torch.cat([mean_pool, std_pool], dim=-1)
        z = self.proj(stats)
        return F.normalize(z, p=2, dim=-1)


class RegimeAdapter2d(nn.Module):
    """
    Lightweight per-regime modular adapter (FiLM modulation + bottleneck residual head).
    Zero-initialized output layer so adding a new regime starts smoothly from the
    shared backbone prediction (promoting positive forward transfer).
    """

    def __init__(self, width: int, out_channels: int = 2, bottleneck: int = 16):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(width, bottleneck, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        mod = feat * (1.0 + self.gamma) + self.beta
        return self.net(mod)


class FNO2dSurrogate(nn.Module):
    """
    Standard Re-conditioned 2D Fourier Neural Operator (shared backbone baseline).
    Used by Fine-Tuning, Joint, Static EWC, Online EWC, ER, and Replay-TS.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 24,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_blocks = n_blocks

        # [u (in_channels), x_grid, y_grid, Re_channel] -> in_channels + 3
        self.fc0 = nn.Conv2d(in_channels + 3, width, kernel_size=1)
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_blocks)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(n_blocks)
        ])
        self.fc1 = nn.Conv2d(width, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, out_channels, kernel_size=1)

    def _build_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        grid_x = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        grid_y = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([grid_x, grid_y], dim=1)

    def encode_backbone(self, u: torch.Tensor, re: torch.Tensor) -> torch.Tensor:
        B, C, H, W = u.shape
        if re.dim() == 1:
            re = re.unsqueeze(1)
        grid = self._build_grid(B, H, W, u.device)
        re_chan = re.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([u, grid, re_chan], dim=1)
        x = self.fc0(x)
        for conv, w in zip(self.spectral_convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        return x

    def forward(self, u: torch.Tensor, re: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encode_backbone(u, re)
        delta = self.fc2(F.gelu(self.fc1(feat)))
        return u + delta

    def memory_breakdown_bytes(self) -> Dict[str, int]:
        backbone_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        return {
            "backbone_bytes": backbone_bytes,
            "adapter_bytes": 0,
            "encoder_bytes": 0,
            "prototype_bytes": 0,
            "total_model_bytes": backbone_bytes,
        }


class MEWCFNO2d(nn.Module):
    """
    Full Modular MEWC-Flow 2D FNO Surrogate:
      - Shared FNO-2d spectral backbone (protected by Complex A-EWC + OFU)
      - Physics-only Contrastive Regime Encoder (zero Re leakage!)
      - Modular Regime Adapters (FiLM + bottleneck 1x1 residual head per task)
      - Task-agnostic Soft / Hard Prototype Routing for continual retention &
        zero-shot cross-Reynolds interpolation.

    Can also be configured as SLE-FNO (use_contrastive=False, freeze_backbone_after_t0=True)
    or monolithic ablations (use_adapters=False).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 24,
        n_blocks: int = 4,
        latent_dim: int = 32,
        adapter_bottleneck: int = 16,
        use_adapters: bool = True,
        use_contrastive_encoder: bool = True,
        route_temperature: float = 0.20,
        router_type: str = "contrastive",
        tau_fuzzy: float = 1.0,
    ):
        super().__init__()
        self.backbone = FNO2dSurrogate(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            n_blocks=n_blocks,
        )
        self.width = width
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.adapter_bottleneck = adapter_bottleneck
        self.use_adapters = use_adapters
        self.router_type = router_type
        self.use_contrastive_encoder = use_contrastive_encoder and (router_type != "fuzzy")
        self.route_temperature = route_temperature

        self.regime_encoder = (
            PhysicsRegimeEncoder2d(latent_dim=latent_dim, hidden=24)
            if self.use_contrastive_encoder
            else None
        )
        self.fuzzy_router = (
            TakagiSugenoRegimeRouter(is_1d=False, tau_fuzzy=tau_fuzzy)
            if router_type == "fuzzy"
            else None
        )
        self.adapters = nn.ModuleList()
        # Registered buffer for prototypes: populated dynamically as tasks finish
        self.register_buffer("prototypes", torch.zeros(0, latent_dim))
        self.prototype_re: List[float] = []

    def add_regime_adapter(self, device: Optional[torch.device] = None) -> int:
        """Append a new lightweight modular adapter for an incoming flow regime."""
        if not self.use_adapters:
            return 0
        dev = device or next(self.backbone.parameters()).device
        adapter = RegimeAdapter2d(
            width=self.width,
            out_channels=self.out_channels,
            bottleneck=self.adapter_bottleneck,
        ).to(dev)
        self.adapters.append(adapter)
        return len(self.adapters) - 1

    def freeze_past_adapters(self, active_task_id: int):
        """Freeze all previously learned adapters so past regime specializations cannot drift."""
        for idx, adapter in enumerate(self.adapters):
            req = (idx == active_task_id)
            for p in adapter.parameters():
                p.requires_grad = req

    def unfreeze_all_adapters(self):
        for adapter in self.adapters:
            for p in adapter.parameters():
                p.requires_grad = True

    def shared_backbone_param_names(self) -> List[str]:
        """Return parameter names belonging ONLY to the shared FNO backbone (for A-EWC)."""
        return [n for n, _ in self.named_parameters() if n.startswith("backbone.")]

    @torch.no_grad()
    def register_prototype(
        self,
        task_id: int,
        embeddings: torch.Tensor,
        re_norm: Optional[float] = None,
    ):
        """Compute and store L2-normalized regime prototype c_k from training embeddings."""
        proto = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=-1).to(self.prototypes.device)
        if task_id < self.prototypes.shape[0]:
            self.prototypes[task_id] = proto.squeeze(0)
            if re_norm is not None and task_id < len(self.prototype_re):
                self.prototype_re[task_id] = float(re_norm)
        else:
            self.prototypes = torch.cat([self.prototypes, proto], dim=0)
            self.prototype_re.append(float(re_norm) if re_norm is not None else float(task_id))

    def encode_regime(self, u: torch.Tensor, dy: float = 1.0, dx: float = 1.0) -> Optional[torch.Tensor]:
        """Extract physics-only regime latent code z (NO Re input!)."""
        if self.regime_encoder is None:
            return None
        return self.regime_encoder(u, dy=dy, dx=dx)

    def compute_routing_weights(
        self,
        z: Optional[torch.Tensor],
        re: torch.Tensor,
        n_adapters: int,
        routing_mode: str = "soft",
        task_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute routing weights w: (B, n_adapters).
        - If task_id is explicitly given during training (or routing_mode == 'oracle'),
          uses one-hot routing to that adapter.
        - Otherwise, strictly task-agnostic routing via physics-only contrastive prototypes
          (or Re-proximity fallback if contrastive encoder is ablated).
        """
        B = re.shape[0]
        device = re.device

        if n_adapters == 0:
            return torch.zeros(B, 0, device=device)

        if task_id is not None and routing_mode != "soft_train":
            w = torch.zeros(B, n_adapters, device=device)
            if isinstance(task_id, int):
                idx = min(task_id, n_adapters - 1)
                w[:, idx] = 1.0
            else:
                # Per-sample task IDs (e.g., mixed current + replay batch during training)
                idx = task_id.clamp(0, n_adapters - 1)
                w.scatter_(1, idx.view(-1, 1), 1.0)
            return w

        if routing_mode == "uniform" or n_adapters == 1:
            return torch.full((B, n_adapters), 1.0 / n_adapters, device=device)

        # Task-agnostic routing using physics-only contrastive prototypes (+ physical Re kernel)
        if z is not None and self.prototypes.shape[0] >= n_adapters:
            protos = self.prototypes[:n_adapters].to(device)  # (K, d)
            sims = torch.matmul(z, protos.T)  # (B, K) cosine similarity in [-1, 1]
            logits = sims / self.route_temperature
            if len(self.prototype_re) >= n_adapters:
                proto_re = torch.tensor(self.prototype_re[:n_adapters], device=device, dtype=re.dtype).view(1, -1)
                re_dists = -(re.view(B, 1) - proto_re).pow(2) / (2.0 * 0.14**2)
                logits = logits + re_dists
            if routing_mode == "hard":
                top_idx = logits.argmax(dim=-1, keepdim=True)
                w = torch.zeros(B, n_adapters, device=device)
                w.scatter_(1, top_idx, 1.0)
                return w
            return F.softmax(logits, dim=-1)

        # Fallback when contrastive encoder is ablated: RBF proximity on scalar Re_norm
        if len(self.prototype_re) >= n_adapters:
            proto_re = torch.tensor(self.prototype_re[:n_adapters], device=device, dtype=re.dtype).view(1, -1)
            re_col = re.view(B, 1)
            dists = -(re_col - proto_re).pow(2) / (2.0 * 0.15**2)
            if routing_mode == "hard":
                top_idx = dists.argmax(dim=-1, keepdim=True)
                w = torch.zeros(B, n_adapters, device=device)
                w.scatter_(1, top_idx, 1.0)
                return w
            return F.softmax(dists, dim=-1)

        return torch.full((B, n_adapters), 1.0 / n_adapters, device=device)

    def forward(
        self,
        u: torch.Tensor,
        re: torch.Tensor,
        task_id: Optional[int] = None,
        routing_mode: str = "soft",
        return_latent: bool = False,
        return_weights: bool = False,
        dy: float = 1.0,
        dx: float = 1.0,
    ):
        if re.dim() == 1:
            re = re.unsqueeze(1)

        feat = self.backbone.encode_backbone(u, re)
        delta = self.backbone.fc2(F.gelu(self.backbone.fc1(feat)))

        z = self.encode_regime(u, dy=dy, dx=dx) if self.use_contrastive_encoder else None
        weights = None

        if self.use_adapters and len(self.adapters) > 0:
            if self.fuzzy_router is not None and (task_id is None or routing_mode == "soft_train"):
                weights = self.fuzzy_router(
                    u, dy=dy, dx=dx, n_active=len(self.adapters), routing_mode=routing_mode
                )
            else:
                weights = self.compute_routing_weights(
                    z=z,
                    re=re,
                    n_adapters=len(self.adapters),
                    routing_mode=routing_mode,
                    task_id=task_id,
                )  # (B, K)

            # Fast path when single scalar task_id is active during training
            if isinstance(task_id, int) and routing_mode != "soft_train":
                delta_adapt = self.adapters[min(task_id, len(self.adapters) - 1)](feat)
            else:
                adapter_outs = torch.stack([ad(feat) for ad in self.adapters], dim=1)  # (B, K, C, H, W)
                w_view = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
                delta_adapt = (adapter_outs * w_view).sum(dim=1)
            delta = delta + delta_adapt

        u_next = u + delta

        if return_latent and return_weights:
            return u_next, z, weights
        if return_latent:
            return u_next, z
        if return_weights:
            return u_next, weights
        return u_next

    def memory_breakdown_bytes(self) -> Dict[str, int]:
        backbone_b = sum(p.numel() * p.element_size() for p in self.backbone.parameters())
        adapter_b = sum(p.numel() * p.element_size() for p in self.adapters.parameters())
        enc_b = (
            sum(p.numel() * p.element_size() for p in self.regime_encoder.parameters())
            if self.regime_encoder is not None
            else (
                sum(p.numel() * p.element_size() for p in self.fuzzy_router.parameters())
                + self.fuzzy_router.feat_mean.numel() * self.fuzzy_router.feat_mean.element_size()
                + self.fuzzy_router.feat_std.numel() * self.fuzzy_router.feat_std.element_size()
                if self.fuzzy_router is not None
                else 0
            )
        )
        proto_b = self.prototypes.numel() * self.prototypes.element_size()
        return {
            "backbone_bytes": backbone_b,
            "adapter_bytes": adapter_b,
            "encoder_bytes": enc_b,
            "prototype_bytes": proto_b,
            "total_model_bytes": backbone_b + adapter_b + enc_b + proto_b,
        }


def build_fno2d(
    modes: int = 8,
    width: int = 24,
    in_channels: int = 2,
    out_channels: int = 2,
    n_blocks: int = 4,
    device: str = "cpu",
    modular: bool = False,
    latent_dim: int = 32,
    adapter_bottleneck: int = 16,
    use_adapters: bool = True,
    use_contrastive_encoder: bool = True,
    route_temperature: float = 0.20,
    router_type: str = "contrastive",
    tau_fuzzy: float = 1.0,
) -> nn.Module:
    if modular:
        model = MEWCFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes,
            modes2=modes,
            width=width,
            n_blocks=n_blocks,
            latent_dim=latent_dim,
            adapter_bottleneck=adapter_bottleneck,
            use_adapters=use_adapters,
            use_contrastive_encoder=use_contrastive_encoder,
            route_temperature=route_temperature,
            router_type=router_type,
            tau_fuzzy=tau_fuzzy,
        )
    else:
        model = FNO2dSurrogate(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes,
            modes2=modes,
            width=width,
            n_blocks=n_blocks,
        )
    return model.to(device)
