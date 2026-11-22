# MEWC-Flow: Modular Adaptive Continual Learning for Neural Flow Surrogates Across Reynolds Regimes

Official research repository for **MEWC-Flow**, extending **MEWC-RL++** (Modular shared-bottom architecture + Gradient-Alignment Adaptive EWC + Online Fisher Updating) to 2D/1D Fourier Neural Operator (FNO) fluid-flow surrogates with a physics-only contrastive regime router and derivative physics regularization.

## Key Components

- **Shared Spectral Operator Backbone (`FNO2dSurrogate` / `BurgersSurrogate`):** Re-conditioned residual Fourier Neural Operator with complex spectral convolution layers.
- **Modular Regime Adapters (`RegimeAdapter2d` / `RegimeAdapter1d`):** Lightweight per-regime FiLM channel modulation + bottleneck $1\times 1$ residual heads (~1.2% of backbone parameters per regime), frozen after task completion to prevent adapter drift.
- **Physics-Only Contrastive Regime Encoder (`PhysicsRegimeEncoder2d` / `1d`):** Trained with Supervised Contrastive Loss (`SupConLoss`) + prototype anchoring strictly on flow velocity and spatial differential invariants ($[u, v, |\mathbf{u}|, \omega, \nabla\cdot\mathbf{u}]$) with **zero $Re$ or task-ID input** — enabling leak-free, task-agnostic soft prototype routing and cross-Reynolds interpolation.
- **Complex-Parameter Adaptive EWC + Online Fisher Updating (`AdaptiveEWC`):** Generalizes MEWC-RL++ Eqs. (1.8)–(1.14) to complex Fourier weights using real inner-product gradient cosine alignment $\alpha_t$, dynamic consolidation strength $\lambda_t \in [\lambda_{\min}, \lambda_{\max}]$, and exponentially decayed diagonal Fisher $F^{(k)} = \gamma_F F^{(k-1)} + (1-\gamma_F)\hat{F}^{(k)}$.
- **Matched-Budget Baselines:** Fair side-by-side implementations of `Fine-Tuning`, `Joint Training`, `Static EWC`, `Online-EWC`, `Experience Replay (ER)`, `Replay-TS` (Hemati et al., 2026), and `SLE-FNO` (Elhadidy et al., 2026), all sharing identical backbone capacity, optimizer, LR schedule, gradient clipping, and deterministic seeds.

## Project Layout

```text
.
├── MEWC-Flow Continual Learning of Flow Surrogates across Reynolds Regimes.md  # Full Q1 Manuscript
├── MEWC-RL++.pdf                                                               # Prior MEWC-RL++ Chapter
├── requirements.txt
├── tests/test_verification.py                                                  # Automated verification suite
├── data/                                                                       # PIV, Kolmogorov NS, ERA5 caches
├── relatedwork/                                                                # Literature tables & surveys
└── mewc_flow/
    ├── configs/burgers_pilot.yaml
    ├── data/                # Leak-free dataset loaders (PIV, Kolmogorov NS, ERA5, Burgers, JHTDB)
    ├── models/              # FNO2d / MEWCFNO2d, 1D Burgers FNO, Static/Online/Adaptive Complex EWC
    ├── trainers/            # Fair unified trainers & all 8 CL baselines
    ├── utils/               # CL metrics, physics/spectral/rollout/slice diagnostics, paired stats, strict JSON
    └── experiments/         # Multi-seed runners & 300-DPI figure generators for all 4 benchmarks
```

## Quickstart & Verification

```bash
pip install -r requirements.txt
python3 tests/test_verification.py
```

See [mewc_flow/data/REAL_DATA_SETUP.md](mewc_flow/data/REAL_DATA_SETUP.md) for full data setup and experiment reproduction commands.
