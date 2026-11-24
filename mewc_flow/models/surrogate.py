"""
surrogate.py
============
1-D Re-conditioned Spectral/Residual Surrogates for the Burgers Benchmark.

Includes:
1. BurgersSurrogate (standard 1-D Re-conditioned FNO-1d backbone for fair baselines)
2. MEWCBurgersSurrogate (Modular FNO-1d + Physics-only 1D Regime Encoder +
   Lightweight FiLM/Residual Regime Adapters for MEWC-Flow & SLE-FNO on 1D Burgers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from mewc_flow.models.fuzzy_logic import TakagiSugenoRegimeRouter


class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution layer with complex weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, nx)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = min(self.modes, N // 2 + 1)
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=N, norm="ortho")


class BurgersSurrogate(nn.Module):
    """Re-conditioned 1-D FNO surrogate mapping (u_t, re_norm) -> u_{t+1}."""

    def __init__(self, nx: int = 128, width: int = 32, modes: int = 16, n_blocks: int = 3, **kwargs):
        super().__init__()
        self.nx = nx
        self.width = width
        self.modes = modes
        # Input channels: [u, x_grid, re_channel] -> 3
        self.fc0 = nn.Conv1d(3, width, kernel_size=1)
        self.spectral_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_blocks)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(n_blocks)])
        self.fc1 = nn.Conv1d(width, 64, kernel_size=1)
        self.fc2 = nn.Conv1d(64, 1, kernel_size=1)

    def encode_backbone(self, u: torch.Tensor, re: torch.Tensor) -> torch.Tensor:
        # u: (B, nx), re: (B, 1)
        if re.dim() == 1:
            re = re.unsqueeze(1)
        B, N = u.shape
        grid = torch.linspace(0.0, 1.0, N, device=u.device).view(1, 1, N).expand(B, 1, N)
        re_chan = re.view(B, 1, 1).expand(B, 1, N)
        x = torch.cat([u.unsqueeze(1), grid, re_chan], dim=1)
        x = self.fc0(x)
        for conv, w in zip(self.spectral_convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        return x

    def forward(self, u: torch.Tensor, re: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encode_backbone(u, re)
        delta = self.fc2(F.gelu(self.fc1(feat))).squeeze(1)
        return u + delta

    def memory_breakdown_bytes(self) -> Dict[str, int]:
        b = sum(p.numel() * p.element_size() for p in self.parameters())
        return {
            "backbone_bytes": b,
            "adapter_bytes": 0,
            "encoder_bytes": 0,
            "prototype_bytes": 0,
            "total_model_bytes": b,
        }


class PhysicsRegimeEncoder1d(nn.Module):
    """Physics-only 1-D regime encoder operating on [u, du/dx, d^2u/dx^2] (NO Re input)."""

    def __init__(self, latent_dim: int = 32, hidden: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        N = u.shape[-1]
        dx = 1.0 / float(N)
        du_dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)
        d2u = (torch.roll(u, -1, dims=-1) - 2.0 * u + torch.roll(u, 1, dims=-1)) / (dx * dx)
        # Log-compress steep shock derivatives for numerical stability
        du_s = torch.sign(du_dx) * torch.log1p(du_dx.abs())
        d2u_s = torch.sign(d2u) * torch.log1p(d2u.abs())
        inp = torch.stack([u, du_s, d2u_s], dim=1)
        h = self.conv(inp)
        stats = torch.cat([h.mean(dim=-1), h.var(dim=-1, unbiased=False).add(1e-6).sqrt()], dim=-1)
        return F.normalize(self.proj(stats), p=2, dim=-1)


class RegimeAdapter1d(nn.Module):
    """Lightweight per-regime 1-D FiLM + residual adapter."""

    def __init__(self, width: int, bottleneck: int = 16):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, width, 1))
        self.beta = nn.Parameter(torch.zeros(1, width, 1))
        self.net = nn.Sequential(
            nn.Conv1d(width, bottleneck, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat * (1.0 + self.gamma) + self.beta).squeeze(1)


class MEWCBurgersSurrogate(nn.Module):
    """Modular MEWC-Flow 1-D FNO for Burgers shock-formation continual learning."""

    def __init__(
        self,
        nx: int = 128,
        width: int = 32,
        modes: int = 16,
        n_blocks: int = 3,
        latent_dim: int = 32,
        adapter_bottleneck: int = 16,
        use_adapters: bool = True,
        use_contrastive_encoder: bool = True,
        route_temperature: float = 0.20,
        router_type: str = "contrastive",
        tau_fuzzy: float = 1.0,
    ):
        super().__init__()
        self.backbone = BurgersSurrogate(nx=nx, width=width, modes=modes, n_blocks=n_blocks)
        self.width = width
        self.latent_dim = latent_dim
        self.adapter_bottleneck = adapter_bottleneck
        self.use_adapters = use_adapters
        self.router_type = router_type
        self.use_contrastive_encoder = use_contrastive_encoder and (router_type != "fuzzy")
        self.route_temperature = route_temperature

        self.regime_encoder = PhysicsRegimeEncoder1d(latent_dim=latent_dim) if self.use_contrastive_encoder else None
        self.fuzzy_router = TakagiSugenoRegimeRouter(is_1d=True, tau_fuzzy=tau_fuzzy) if router_type == "fuzzy" else None
        self.adapters = nn.ModuleList()
        self.register_buffer("prototypes", torch.zeros(0, latent_dim))
        self.prototype_re: List[float] = []

    def add_regime_adapter(self, device: Optional[torch.device] = None) -> int:
        if not self.use_adapters:
            return 0
        dev = device or next(self.backbone.parameters()).device
        self.adapters.append(RegimeAdapter1d(self.width, self.adapter_bottleneck).to(dev))
        return len(self.adapters) - 1

    def freeze_past_adapters(self, active_task_id: int):
        for idx, ad in enumerate(self.adapters):
            for p in ad.parameters():
                p.requires_grad = (idx == active_task_id)

    def shared_backbone_param_names(self) -> List[str]:
        return [n for n, _ in self.named_parameters() if n.startswith("backbone.")]

    @torch.no_grad()
    def register_prototype(self, task_id: int, embeddings: torch.Tensor, re_norm: Optional[float] = None):
        proto = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=-1).to(self.prototypes.device)
        if task_id < self.prototypes.shape[0]:
            self.prototypes[task_id] = proto.squeeze(0)
            if re_norm is not None and task_id < len(self.prototype_re):
                self.prototype_re[task_id] = float(re_norm)
        else:
            self.prototypes = torch.cat([self.prototypes, proto], dim=0)
            self.prototype_re.append(float(re_norm) if re_norm is not None else float(task_id))

    def encode_regime(self, u: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        if self.regime_encoder is None:
            return None
        return self.regime_encoder(u)

    def compute_routing_weights(self, z, re, n_adapters, routing_mode="soft", task_id=None):
        B = re.shape[0]
        device = re.device
        if n_adapters == 0:
            return torch.zeros(B, 0, device=device)
        if task_id is not None and routing_mode != "soft_train":
            w = torch.zeros(B, n_adapters, device=device)
            if isinstance(task_id, int):
                w[:, min(task_id, n_adapters - 1)] = 1.0
            else:
                w.scatter_(1, task_id.clamp(0, n_adapters - 1).view(-1, 1), 1.0)
            return w
        if routing_mode == "uniform" or n_adapters == 1:
            return torch.full((B, n_adapters), 1.0 / n_adapters, device=device)
        if z is not None and self.prototypes.shape[0] >= n_adapters:
            sims = torch.matmul(z, self.prototypes[:n_adapters].to(device).T)
            logits = sims / self.route_temperature
            if len(self.prototype_re) >= n_adapters:
                pre = torch.tensor(self.prototype_re[:n_adapters], device=device, dtype=re.dtype).view(1, -1)
                logits = logits - (re.view(B, 1) - pre).pow(2) / (2.0 * 0.12**2)
            if routing_mode == "hard":
                w = torch.zeros(B, n_adapters, device=device)
                w.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
                return w
            return F.softmax(logits, dim=-1)
        if len(self.prototype_re) >= n_adapters:
            pre = torch.tensor(self.prototype_re[:n_adapters], device=device, dtype=re.dtype).view(1, -1)
            d = -(re.view(B, 1) - pre).pow(2) / (2.0 * 0.15**2)
            return F.softmax(d, dim=-1)
        return torch.full((B, n_adapters), 1.0 / n_adapters, device=device)

    def forward(
        self,
        u: torch.Tensor,
        re: torch.Tensor,
        task_id: Optional[int] = None,
        routing_mode: str = "soft",
        return_latent: bool = False,
        return_weights: bool = False,
        **kwargs,
    ):
        if re.dim() == 1:
            re = re.unsqueeze(1)
        feat = self.backbone.encode_backbone(u, re)
        delta = self.backbone.fc2(F.gelu(self.backbone.fc1(feat))).squeeze(1)
        z = self.encode_regime(u) if self.use_contrastive_encoder else None
        weights = None

        if self.use_adapters and len(self.adapters) > 0:
            if self.fuzzy_router is not None and (task_id is None or routing_mode == "soft_train"):
                weights = self.fuzzy_router(
                    u, n_active=len(self.adapters), routing_mode=routing_mode
                )
            else:
                weights = self.compute_routing_weights(z, re, len(self.adapters), routing_mode, task_id)
            if isinstance(task_id, int) and routing_mode != "soft_train":
                delta = delta + self.adapters[min(task_id, len(self.adapters) - 1)](feat)
            else:
                outs = torch.stack([ad(feat) for ad in self.adapters], dim=1)  # (B, K, nx)
                delta = delta + (outs * weights.unsqueeze(-1)).sum(dim=1)

        u_next = u + delta
        if return_latent and return_weights:
            return u_next, z, weights
        if return_latent:
            return u_next, z
        if return_weights:
            return u_next, weights
        return u_next

    def memory_breakdown_bytes(self) -> Dict[str, int]:
        bb = sum(p.numel() * p.element_size() for p in self.backbone.parameters())
        ad = sum(p.numel() * p.element_size() for p in self.adapters.parameters())
        enc = (
            sum(p.numel() * p.element_size() for p in self.regime_encoder.parameters())
            if self.regime_encoder
            else (
                sum(p.numel() * p.element_size() for p in self.fuzzy_router.parameters())
                + self.fuzzy_router.feat_mean.numel() * self.fuzzy_router.feat_mean.element_size()
                + self.fuzzy_router.feat_std.numel() * self.fuzzy_router.feat_std.element_size()
                if self.fuzzy_router is not None
                else 0
            )
        )
        pr = self.prototypes.numel() * self.prototypes.element_size()
        return {
            "backbone_bytes": bb,
            "adapter_bytes": ad,
            "encoder_bytes": enc,
            "prototype_bytes": pr,
            "total_model_bytes": bb + ad + enc + pr,
        }


def build_surrogate(
    nx: int = 128,
    width: int = 32,
    modes: int = 16,
    n_blocks: int = 3,
    device: str = "cpu",
    modular: bool = False,
    latent_dim: int = 32,
    use_adapters: bool = True,
    use_contrastive_encoder: bool = True,
    router_type: str = "contrastive",
    tau_fuzzy: float = 1.0,
    **kwargs,
) -> nn.Module:
    if modular:
        model = MEWCBurgersSurrogate(
            nx=nx, width=width, modes=modes, n_blocks=n_blocks,
            latent_dim=latent_dim, use_adapters=use_adapters,
            use_contrastive_encoder=use_contrastive_encoder,
            router_type=router_type,
            tau_fuzzy=tau_fuzzy,
        )
    else:
        model = BurgersSurrogate(nx=nx, width=width, modes=modes, n_blocks=n_blocks)
    return model.to(device)
