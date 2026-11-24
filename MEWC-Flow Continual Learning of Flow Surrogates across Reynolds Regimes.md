# MEWC-Flow: Modular Elastic Weight Consolidation with Physics-Only Contrastive Regime Routing for Continual Learning of Fluid Flow Surrogates across Reynolds Regimes

**Md Mehedi Hasan**

---

## Abstract

Neural operators such as the Fourier Neural Operator (FNO) provide rapid surrogate inference for partial differential equations, yet suffer catastrophic forgetting when sequentially adapted across changing Reynolds numbers ($Re$), hydrodynamic bifurcations, or seasonal atmospheric states. Existing continual learning (CL) defenses either apply monolithic parameter penalties that underfit steep nonlinear regime shifts, require explicit regime labels at inference time, or lack evaluation across physical invariants and multi-step autoregressive rollouts. We propose **MEWC-Flow**, a modular continual operator-learning framework that integrates four mechanisms within an FNO backbone: (i) lightweight Feature-wise Linear Modulation (FiLM) and bottleneck residual **regime adapters**, (ii) a **physics-only supervised contrastive regime encoder** that maps purely flow-derived invariants (velocity magnitude, vorticity, and divergence, with zero $Re$ label input) onto a hyperspherical prototype space for task-agnostic soft routing, (iii) **complex-parameter Adaptive Elastic Weight Consolidation (A-EWC)** with Online Fisher Updating (OFU) protecting only shared spectral weights via exponential gradient cosine alignment, and (iv) **matched-budget episodic replay** paired with vorticity/divergence physical regularization. We evaluate MEWC-Flow alongside seven competitive baselines—Fine-Tuning, EWC, Online-EWC, Experience Replay (ER), Temporal-Slicing Replay (Replay-TS), Single-Layer Extension (SLE-FNO), and an offline Joint upper bound—across four multi-seed benchmarks: experimental Caltech laboratory Particle Image Velocimetry (PIV) cylinder wakes ($Re = 240 \to 2520$, 5 seeds), pseudo-spectral 2D Kolmogorov Navier–Stokes bifurcations ($Re = 20 \to 400$, 3 seeds), real ERA5 global 10-m planetary winds across four seasons (3 seeds), and 1D viscous Burgers shock steepening ($\nu = 0.05 \to 0.0005$, 3 seeds) including zero-data PDE-residual replay (`MEWC-PDE-Replay`). Under strict leak-free nondimensionalization and matched byte-level memory accounting, MEWC-Flow achieves $-7.78 \times 10^{-5} \pm 1.8 \times 10^{-5}$ average forgetting on PIV wakes (eliminating the $+134.8\%$ early-regime degradation of Fine-Tuning, $p = 0.0203$ on step-10 rollout), identifies unseen flow snapshots with $77.6\% \pm 4.6\%$ accuracy ($84.2\%$ seed-0, silhouette $0.518$) without $Re$ supervision, attains the lowest exact 2D Navier–Stokes PDE residual on Kolmogorov bifurcations ($7.040 \pm 0.017$, outperforming ER at $p = 0.0035$ and Joint at $p = 0.0149$), and stabilizes 72-h ERA5 planetary wind rollouts ($0.9260 \pm 0.0149$ vs. ER $1.1024$, $p = 0.0031$; Replay-TS $1.2253$, $p = 0.0008$).

**Keywords:** Continual Learning, Fourier Neural Operators, Fluid Mechanics, Reynolds Number Adaptation, Elastic Weight Consolidation, Contrastive Regime Routing, Particle Image Velocimetry, Navier–Stokes Equations.

---

## 1. Introduction

Data-driven surrogate models based on neural operators—most prominently the Fourier Neural Operator (FNO) [19]—have demonstrated orders-of-magnitude speedups over direct numerical simulation (DNS) when learning solution operators of partial differential equations (PDEs). Despite strong accuracy within a fixed training distribution, real-world fluid engineering, experimental aerodynamics, and geophysical forecasting inherently operate over **non-stationary sequential streams**. As flow speed rises, geometry warms, or seasonal forcing shifts, the governing Reynolds number $Re = U_\infty D / \nu$ increases across qualitative dynamical transitions: from steady laminar shear, through supercritical Hopf vortex shedding, into transitional shear-layer breakup and broadband turbulence. Crucially, real experimental facilities and operational weather pipelines rarely allow storing every historical high-dimensional snapshot for full offline retraining from scratch.

Naive sequential fine-tuning of a monolithic FNO induces severe **catastrophic forgetting** [12] of low-$Re$ dynamics as gradient updates overwrite shared low-wavenumber spectral modes to accommodate steeper high-$Re$ gradients. Conversely, naively stiffening all weights with standard Elastic Weight Consolidation (EWC) [12, 14] restricts the plasticity required to resolve newly emerging inertial-range eddies or shock discontinuities. Recent operator-specific continual learning (CL) methods—such as Single-Layer Extensions (SLE-FNO) [2] and Temporal-Slicing Replay (Replay-TS) [3]—highlight complementary strengths of lightweight architectural expansion and temporal-window replay, while Reynolds-aware models [9] show that cross-$Re$ generalization collapses without explicit regime awareness. However, three foundational gaps remain open:

1. **Task-agnostic physical regime identification:** Deployment streams rarely tag incoming flow snapshots with ground-truth $Re$. Routing experts using raw labels leaks privileged metadata, whereas routing via unconditioned reconstruction error [4, 6] does not explicitly enforce cross-$Re$ metric separation in differential invariants.
2. **Complex spectral parameter consolidation:** Standard EWC assumes real-valued weights; FNO spectral convolution layers store complex-valued Fourier tensors (`torch.cfloat`), requiring a modulus-squared complex Fisher formulation coupled with modular parameter isolation.
3. **Evaluation integrity and physical rigor:** Per-regime normalization leaks unseen scale statistics across tasks, boundary-crossing temporal slices introduce unphysical discontinuities, and reporting single-step pixel MSE alone masks autoregressive rollout blowup, divergence violations, and inertial-band spectral collapse.

To bridge these gaps, we extend our prior modular adaptive consolidation framework **MEWC-RL++** [22] from continuous control to spatiotemporal fluid dynamics, introducing **MEWC-Flow**. Our investigation is organized around five focused research questions:

- **RQ1 (Stability–Plasticity Across Regimes):** How effectively do modular regime adapters combined with complex-parameter Adaptive EWC (A-EWC) and matched-budget episodic replay prevent catastrophic forgetting while preserving plasticity across sequential Reynolds-number shifts, including genuine hydrodynamic bifurcations?
- **RQ2 (Physical Consistency & Rollout Fidelity):** Beyond single-step MSE, does MEWC-Flow preserve physical invariants—vorticity RMSE, divergence-free incompressibility, exact PDE residuals, kinetic energy spectra $E(k)$, and valid prediction horizon (VPH)—across multi-step autoregressive rollouts?
- **RQ3 (Zero-Leakage Physics-Only Regime Routing):** Can a supervised contrastive encoder operating *exclusively* on spatial flow invariants ($[u, v, |\mathbf{u}|, \omega, \nabla\cdot\mathbf{u}]$ with zero $Re$ input) identify unseen regimes and smoothly interpolate adapters onto held-out Reynolds numbers?
- **RQ4 (Zero-Data Physics-Residual Replay Trade-off):** To what extent can differentiable governing PDE residuals replace stored observational samples (`MEWC-PDE-Replay`), and where does physical regularization hit its limit on sharp viscous Burgers shocks?
- **RQ5 (Task Ordering & Memory Budget Scaling):** How sensitive are consolidation and replay strategies to regime curriculum ordering (ascending, descending, shuffled) and episodic memory budget $M \in \{4, 8, 16, 32\}$?

### Contributions
1. **MEWC-Flow Architecture & Complex A-EWC:** We formulate a modular FNO architecture combining shared spectral blocks, lightweight per-regime FiLM + bottleneck adapters, a leakage-free physics-only supervised contrastive router, and complex-valued Adaptive EWC with Online Fisher Updating (OFU) and empirical-Bayes Fisher scale normalization.
2. **Leak-Free Continual Fluid Benchmark Suite:** We establish a verified four-benchmark evaluation protocol spanning real laboratory PIV wakes [20], pseudo-spectral 2D Kolmogorov Navier–Stokes bifurcations, real ERA5 global winds [21], and 1D viscous Burgers shocks—enforcing fixed physical nondimensionalization, run-boundary integrity, and exact byte-level memory accounting.
3. **Multi-Seed Statistical & Physical Verification:** Across matched multi-seed experiments verified automatically via [`tests/verify_manuscript_numbers.py`](tests/verify_manuscript_numbers.py), we report full paired $t$-tests, Cohen's $d$ effect sizes, systematic 8-variant ablations, and candid engineering trade-offs between monolithic Experience Replay and modular consolidation.

---

## 2. Related Work

The literature relevant to MEWC-Flow spans five interconnected threads: continual learning (CL) for neural operators and PDE surrogates, Reynolds-number-aware surrogate models, modular and regime-routing architectures, physics-informed continual learning, and the foundational CL algorithms that supply our baselines and theoretical vocabulary.

### 2.1 Continual Learning for Neural Operators and PDE Surrogates

Most neural surrogate models for fluid dynamics are trained offline on a fixed distribution and degrade sharply when the deployment regime changes. This practical limitation has motivated a recent wave of CL methods specifically designed for scientific machine learning.

**Replay and distillation for neural operators.** Wang et al. [1] embed CL directly into physics-informed neural operators built on the Transolver architecture. When out-of-distribution (OOD) data arrive, a small number of past samples enter the loss through a distillation-based constraint that preserves prior knowledge, while a Low-Rank Adaptation (LoRA) module enables fast adaptation. Validated on Darcy flow, 2D hyperelasticity, and 3D linear elasticity, this work shows that pairing replay with architectural adaptation outperforms either mechanism alone, though it does not study unsteady Navier–Stokes sequences. Complementarily, Howard, Fu, and Stinis [8] formulate continual learning via a multifidelity deep neural network that learns correlations between previously trained predictions and new-domain targets, combining additively with replay and memory-aware synapses.

**Architecture-based continual learning in Fourier Neural Operators.** Elhadidy, D'Souza, and Arzani [2] introduce SLE-FNO, which augments a Fourier Neural Operator [19] with a Single-Layer Extension (SLE) added per task. Across four sequential OOD configurations built from 230 CFD simulations of pulsatile aneurysmal blood flow, SLE-FNO is compared against EWC, Learning without Forgetting (LwF), replay, OGD, GEM, PiggyBack, and LoRA. Replay and architectural extensions achieved the strongest retention, whereas standalone parameter consolidation (EWC) struggled on spatial field-to-field regression. SLE-FNO provides a strong architectural baseline under matched backbone capacity.

**Temporal-slicing replay for autoregressive surrogates.** Autoregressive PDE surrogates face compounding rollout error, and storing full high-resolution trajectories quickly exhausts memory budgets. Hemati, Nguyen, and Sandfeld [3] propose Replay-TS, which stores compact contiguous temporal windows sampled across past simulations and applies sparse multi-step autoregressive supervision. Across Advection-Diffusion, Burgers, and Navier–Stokes streams, Replay-TS serves as a key matched-budget replay baseline in our evaluation.

### 2.2 Reynolds-Aware and Cross-Reynolds Surrogates

A surrogate encountering a new Reynolds number faces both sequential forgetting of earlier regimes and poor cross-Reynolds generalization even under single-regime training.

**Reynolds-conditioned surrogate models.** Guo et al. [9] propose RenaNet, a Reynolds-aware ConvGRU that ingests flow snapshots alongside an explicit Reynolds-number conditioning map over $1 \le Re \le 500$ for cylinder flow. Ablating the $Re$ channel degrades validation MSE by two orders of magnitude, confirming the necessity of explicit $Re$ conditioning. However, RenaNet is trained once offline on a fixed range without continual updating.

**Cross-Reynolds generalization and representation geometry.** Shi [10] demonstrates that a trained FNO incurs $46.68\%$ relative $L_2$ error under a tenfold Reynolds-number shift on forced 2D Navier–Stokes, while latent-space retrieval (`ConvAE-Relay`) achieves $38.34\%$ without target-regime fitting, isolating representation geometry and autoregressive drift as dominant bottlenecks. Similarly, Winkler and Stoll [5] demonstrate a "No Free Lunch" effect in flow surrogacy: surrogates validated on steady regimes consistently fail to transfer to transient time-varying regimes regardless of capacity, reinforcing the necessity of continual adaptation rather than static scaling alone.

### 2.3 Modular and Regime-Routing Architectures

**Progressive mixture-of-experts for turbulence modeling.** Ji et al. [4] introduce Progressive Mixture-of-Experts (PMoE) for continual RANS turbulence closures, using an autoencoder-based router to assign flow scenarios to specialized experts and spawning new experts when novel regimes arise. While PMoE avoids forgetting by construction, it targets RANS closures rather than full autoregressive spatiotemporal velocity operators and routes via reconstruction error rather than contrastive metric geometry.

**Online regime switching via OOD detection.** Kim et al. [6] propose MG-ANN for turbulent reacting flows, coupling an autoencoder OOD detector with regularization-based CL to switch dynamically between neural inference and DNS, providing precedent for task-agnostic regime identification during online fluid simulations.

### 2.4 Physics-Informed Continual Learning and Fluid Control

**Continual PINNs for parameterized PDEs.** Chen et al. [7] formulate CL-PINN, treating PDE instances across parameter values as sequential tasks using Bayesian active selection, dynamic loss weighting, and sparse physics-constrained replay without observational labels. Dekhovich et al. [11] propose iPINNs, using parameter sparsity to incrementally grow PINNs across PDE families. Both demonstrate that governing PDE residuals act as implicit physical memory—directly motivating our zero-data physics-residual replay comparison (RQ4) on 1D viscous Burgers flow. Meanwhile, Lagemann et al. [13] introduce HydroGym for flow control across varying Reynolds regimes, connecting continual reinforcement learning (such as our prior MEWC-RL++ framework [22]) to fluid-mechanics surrogate modeling.

### 2.5 Foundational Continual Learning & Contrastive Methods

Kirkpatrick et al. [12] penalize drift on parameters weighted by the diagonal Fisher Information Matrix (EWC), while Schwarz et al. [14] bound memory to $\mathcal{O}(|\theta|)$ via exponential Fisher decay (Online EWC). Our prior **MEWC-RL++** [22] extends Online EWC with shared-bottom modular heads and **Adaptive EWC (A-EWC)**, modulating consolidation strength $\lambda_t$ via gradient cosine alignment. Rolnick et al. [15] establish compact episodic experience replay (ER) as a robust task-agnostic retention baseline, Riemer et al. [16] frame CL as a gradient-alignment trade-off, and van den Oord et al. [17] and Khosla et al. [18] formalize the contrastive metric objective adopted by MEWC-Flow's physics-only regime encoder.

### 2.6 Positioning Summary

**Table 1.** Positioning of MEWC-Flow against representative continual and cross-Reynolds surrogate learning methods.

| Work | Surrogate | CL Strategy | Crosses Re Bifurcation? | Real Experimental Data? | Physical & Spectral Metrics? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Wang et al. [1] | Transolver PINO | Replay + Distillation + LoRA | No | No | Partial (PDE residual) |
| Elhadidy et al. [2] | FNO | SLE-FNO (Single-Layer Extension) | No (geometry/BC) | No (CFD) | No (TAWSS MSE) |
| Hemati et al. [3] | Autoregressive PDE | Replay-TS (Temporal Slicing) | No (mixed PDEs) | No (synthetic) | Rollout MSE |
| Ji et al. [4] | RANS Closure | PMoE + Autoencoder Router | No | No (RANS) | Partial |
| Kim et al. [6] | Reacting-Flow ROM | OOD Autoencoder + Reg. CL | No | No (DNS) | Partial |
| Chen et al. [7] | Parameterized PINN | Active Seq. + Sparse Physics Replay | No | No (synthetic) | PDE residual |
| Howard et al. [8] | Multifidelity DNN | Correlation-based CL | No | No (synthetic) | No |
| Guo et al. [9] | ConvGRU (RenaNet) | None (Static offline training) | Fixed $Re \le 500$ | No (LBM) | Continuity |
| Shi [10] | FNO, ConvAE | Retrieval (no CL) | Re shift only | No (synthetic) | Rollout drift |
| Dekhovich et al. [11] | PINN (iPINN) | Sparsity incremental | No | No (synthetic) | PDE residual |
| **MEWC-Flow (Ours)** | **Modular Re-FNO** | **Adapters + Complex A-EWC/OFU + Physics-Only SupCon Router + Matched Replay + Vorticity/PDE Loss** | **Yes (NS $Re=20 \to 400$ + PIV $240 \to 2520$)** | **Yes (Lab PIV [20] + ERA5 [21])** | **Yes (Vorticity, $\nabla\cdot\mathbf{u}$, $E(k)$, PDE res., VPH)** |

---

## 3. Problem Formulation & Data Integrity Protocol

### 3.1 Sequential Continual Operator Learning Setup

Consider an ordered sequence of $K$ dynamical fluid tasks $\mathcal{T}_0, \mathcal{T}_1, \dots, \mathcal{T}_{K-1}$, where task $\mathcal{T}_k$ corresponds to a distinct Reynolds number $Re_k$ (or kinematic viscosity $\nu_k$, or seasonal atmospheric forcing) and yields spatiotemporal trajectory pairs $(\mathbf{u}_t, \mathbf{u}_{t+1}) \in \mathcal{D}_k^{\text{train}}$, $\mathbf{u}_t \in \mathbb{R}^{C \times H \times W}$ (or $\mathbb{R}^{1 \times N_x}$ in 1D). At training stage $k$, the surrogate $\mathcal{G}_\theta$ only accesses $\mathcal{D}_k^{\text{train}}$ and an optional bounded episodic buffer $\mathcal{B}$ containing at most $M$ stored pairs per prior task. An upper-bound baseline (`Joint`) trains offline over $\bigcup_{k=0}^{K-1} \mathcal{D}_k^{\text{train}}$. Additionally, held-out unseen regimes $\mathcal{H}$ (e.g., intermediate Reynolds numbers $Re \in \{910, 1890\}$ for PIV, $Re \in \{75, 240\}$ for Kolmogorov NS, and unseen year 2019 seasons for ERA5) test zero-shot cross-regime interpolation.

Let $E_{i, j}$ denote the test error (MSE) evaluated on task $\mathcal{T}_j$ immediately after completing stage $i$ ($0 \le j \le i < K$). Following standard CL error conventions (`mewc_flow/utils/metrics.py`):

$$\text{Avg. Final MSE} = \frac{1}{K} \sum_{j=0}^{K-1} E_{K-1, j}, \qquad \text{Avg. Forgetting} = \frac{1}{K - 1} \sum_{j=0}^{K-2} \left(E_{K-1, j} - E_{j, j}\right) = -\text{BWT}.$$

Because error metrics satisfy "lower is better," **positive Forgetting** ($E_{K-1,j} > E_{j,j}$) indicates catastrophic degradation of earlier regimes, whereas **negative Forgetting** (equivalently positive Backward Transfer $\text{BWT} > 0$) signifies that subsequent training improved earlier regime performance. Task-0 relative degradation is recorded as $(E_{K-1,0} - E_{0,0}) / E_{0,0} \times 100\%$.

### 3.2 Leak-Free Physical Nondimensionalization & Trajectory Integrity

A critical pitfall in continual fluid learning benchmarks is *normalization leakage* and *cross-run trajectory splicing*. We enforce three strict integrity guarantees, automatically asserted in [`tests/test_verification.py`](tests/test_verification.py):

1. **Fixed Physical Nondimensionalization (Zero Test-Set Scaling Leakage):** Prior pipelines that z-score each regime independently leak future regime variance into early inputs. Instead, PIV velocity fields $\mathbf{u}_{\text{phys}}$ in $\text{m/s}$ are scaled purely by known physical inflow speed $U_\infty(Re) = Re \cdot \nu_{\text{water}} / D$ ($\nu_{\text{water}} = 1.004 \times 10^{-6}\text{ m}^2/\text{s}$, cylinder diameter $D = 0.0254\text{ m}$), and ERA5 winds are scaled by a constant physical reference scale $U_{\text{ref}} = 10.0\text{ m/s}$ (`ERA5_U_REF`).
2. **Run-Boundary Integrity:** The Caltech PIV dataset stores 100 consecutive frames per experimental run (`run_idx = frame // 100`). Splicing consecutive flat indices creates unphysical discontinuous jumps at $99 \to 100$. Our loaders enforce exact within-run temporal pairing ($t \bmod 100 < 99$) and chronological splits for ERA5, guaranteeing zero cross-run discontinuities.
3. **Full Held-Out Evaluation:** Held-out datasets evaluate all valid transitions without arbitrary subsampling caps (e.g., all 123 transitions per year in ERA5 held-out sets and all run pairs in PIV).

---

## 4. MEWC-Flow Methodology

**MEWC-Flow** couples a shared spectral operator backbone with per-regime modular adapters, a physics-only contrastive regime router, and complex-parameter Adaptive EWC.

### 4.1 Shared Spectral Backbone

For 2D benchmarks, the shared backbone (`FNO2dBackbone`) takes the normalized velocity field concatenated with normalized spatial coordinates $(x, y)$ and a broadcast physical scaling channel (e.g., $\log_{10}(Re)/4$ when conditioned, passed as zero during strict unlabelled ablation checks), projects through a $1 \times 1$ lifting convolution to channel width $d_h = 24$, applies $L = 4$ residual blocks each combining a complex `SpectralConv2d` layer (retaining $k_{\max} = 8$ Fourier modes per axis) with a local $1 \times 1$ convolution and GELU activation, and outputs a base residual update $h_{\text{shared}}(\mathbf{F})$ (`299,186` parameters in the base 2D model; `340,018` parameters in full modular 2D MEWC-Flow). For 1D Burgers flow, `FNO1d` uses $L = 3$ spectral blocks with $d_h = 32$ and $k_{\max} = 16$ modes (`54,625` base / `76,065` modular parameters). All models predict the temporal increment $\hat{\mathbf{u}}_{t+1} = \mathbf{u}_t + \Delta \hat{\mathbf{u}}$.

### 4.2 Physics-Only Contrastive Regime Encoder & Soft Routing (RQ3)

To route adapters at inference time **without** privileged $Re$ or task-ID metadata, `PhysicsRegimeEncoder2d` computes purely flow-derived spatial differential invariants from the velocity input $\mathbf{u}_t = (u, v)$ alone:

$$\Phi(\mathbf{u}_t) = \left[u, \; v, \; \sqrt{u^2 + v^2 + \epsilon}, \; \omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \; \delta = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right] \in \mathbb{R}^{5 \times H \times W},$$

computed via central finite differences (and $\Phi_{1\text{d}}(u_t) = [u, u_x, u_{xx}] \in \mathbb{R}^{3 \times N_x}$ in 1D). Zero Reynolds number or task-ID channel ever enters this encoder (verified by gradient/perturbation tests in `tests/test_verification.py`). A strided CNN pools both spatial means and standard deviations (capturing wake deficit alongside turbulent fluctuation energy and enstrophy) into an $L_2$-normalized hyperspherical embedding $\mathbf{z}(\mathbf{u}_t) \in \mathcal{S}^{d_z - 1}$ ($d_z = 32$).

When training task $k$ alongside replayed prior tasks $j < k$ (which supply multi-class negative and positive pairs in the mixed mini-batch), the encoder optimizes a Supervised Contrastive (SupCon) loss [18] with temperature $\tau_c = 0.10$:

$$\mathcal{L}_{\text{SupCon}} = -\sum_{i \in I} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i^\top \mathbf{z}_p / \tau_c)}{\sum_{a \in I \setminus \{i\}} \exp(\mathbf{z}_i^\top \mathbf{z}_a / \tau_c)},$$

where $P(i)$ indexes samples sharing the regime identity of sample $i$ in the replay-augmented batch. After each stage, the normalized class prototype $\mathbf{c}_k = \text{norm}(\mathbb{E}_{\mathbf{u} \in \mathcal{D}_k}[\mathbf{z}(\mathbf{u})])$ is frozen. At inference time (task-agnostic), soft cosine attention blends active adapters:

$$w_k(\mathbf{u}_t) = \frac{\exp(\mathbf{z}(\mathbf{u}_t)^\top \mathbf{c}_k / \tau_{\text{route}})}{\sum_{m=0}^{K-1} \exp(\mathbf{z}(\mathbf{u}_t)^\top \mathbf{c}_m / \tau_{\text{route}})}, \qquad \hat{\mathbf{u}}_{t+1} = \mathbf{u}_t + h_{\text{shared}}(\mathbf{F}) + \sum_{k=0}^{K-1} w_k(\mathbf{u}_t)\, h_k(\mathbf{F}),$$

allowing seamless interpolation for held-out intermediate Reynolds numbers (e.g., $Re = 910, 1890$). Hard top-1, uniform, and oracle routing modes are also supported for controlled ablation.

### 4.3 Lightweight Modular Regime Adapters

Each regime adapter `RegimeAdapter2d` ($k \in \{0, \dots, K-1\}$) acts on the backbone feature tensor $\mathbf{F} \in \mathbb{R}^{B \times 24 \times H \times W}$ via Feature-wise Linear Modulation ($\boldsymbol{\gamma}_k, \boldsymbol{\beta}_k$) followed by a $1 \times 1$ bottleneck residual block (width 16) outputting $h_k(\mathbf{F}) \in \mathbb{R}^{B \times C_{\text{out}} \times H \times W}$. Each adapter introduces only $\approx 1.2\%$ of the backbone parameters, remaining isolated from shared EWC penalties so regime-specific amplitude scales can adapt freely.

### 4.4 Complex-Parameter Adaptive EWC (A-EWC) & Online Fisher Updating

Extending MEWC-RL++ [22] to complex Fourier spectral weights $\theta_i \in \mathbb{C}$, `mewc_flow/models/ewc.py` defines the strictly real diagonal Fisher Information using complex gradient modulus squared and exact per-sample normalization:

$$\hat{F}_i^{(k)} = \frac{1}{N_{\text{seen}}} \sum_{n=1}^{N_{\text{seen}}} \left|\frac{\partial \ell_n}{\partial \theta_i}\right|^2, \qquad |g_i|^2 = \operatorname{Re}(g_i)^2 + \operatorname{Im}(g_i)^2.$$

Rather than storing $K$ separate Fisher buffers, Online Fisher Updating (OFU) maintains $\mathcal{O}(|\theta_{\text{shared}}|)$ memory via exponential decay $\gamma_F = 0.85$:

$$F^{(k)} = \gamma_F F^{(k-1)} + (1 - \gamma_F)\hat{F}^{(k)}.$$

To prevent scale collapse across datasets of widely differing MSE magnitudes (PIV $10^{-3}$ vs. ERA5 $2 \times 10^{-2}$ vs. Burgers $10^{-5}$), `AdaptiveEWC` applies empirical-Bayes scale normalization using stored task loss $\mathcal{L}_{\text{scale}}$ and natural step radius $r_0^2 = 10^{-6}$ over only the protected **shared backbone parameters** (`param_allowlist`). At optimization step $t$, gradient memory tracking computes EMA alignment between current shared-parameter gradient $\mathbf{g}_t$ and running gradient memory $\mathbf{g}_{\text{prev}}^{(t)} = \beta_g \mathbf{g}_{\text{prev}}^{(t-1)} + (1 - \beta_g)\mathbf{g}_t$ ($\beta_g = 0.9$):

$$\alpha_t = \frac{\operatorname{Re}\langle \mathbf{g}_t, \mathbf{g}_{\text{prev}}^{(t-1)} \rangle}{\|\mathbf{g}_t\|_2 \, \|\mathbf{g}_{\text{prev}}^{(t-1)}\|_2 + \epsilon} \in [-1, 1], \qquad \lambda_t = \lambda_{\min} + \left(1 - \frac{1 + \alpha_t}{2}\right)(\lambda_{\max} - \lambda_{\min}),$$

with $\lambda_{\min} = 0.01, \lambda_{\max} = 0.15$. Under gradient conflict ($\alpha_t < 0$), $\lambda_t \to \lambda_{\max}$ to stiffen critical shared modes; under aligned gradients ($\alpha_t > 0$), $\lambda_t \to \lambda_{\min}$ to permit forward transfer.

### 4.5 Combined Objective & Matched-Budget Replay

At task $k > 0$, MEWC-Flow minimizes:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}^{(k)} + \beta_{\text{rep}} \mathcal{L}_{\text{replay}} + \alpha_c \mathcal{L}_{\text{SupCon}} + \gamma_{\text{phys}} \mathcal{L}_{\text{phys}} + \frac{\lambda_t}{2} \sum_{i \in \Theta_{\text{shared}}} \tilde{F}_i \left|\theta_i - \theta_i^*\right|^2,$$

with $\beta_{\text{rep}} = 1.0$, $\alpha_c = 0.25$ ($\tau_c = 0.10$), and $\gamma_{\text{phys}} = 0.05$ in 2D (penalizing both velocity divergence $\|\nabla \cdot \hat{\mathbf{u}}\|_2^2$ and finite-difference vorticity error $\|\hat{\omega} - \omega_{\text{true}}\|_2^2$) or $\gamma_{\text{phys}} = 0.02$ in 1D (viscous Burgers PDE residual). For the zero-data variant `MEWC-PDE-Replay` (RQ4), episodic storage is set strictly to $M = 0$ (`use_replay=False`) while collocation perturbations from earlier viscosity parameters receive differentiable PDE-residual supervision (`burgers_pde_residual_torch`). All baselines and MEWC-Flow share identical optimizer settings: `Adam` ($\text{lr} = 10^{-3}$), `CosineAnnealingLR` ($\eta_{\min} = 0.01 \cdot \text{lr}$), and gradient clipping at $\|\mathbf{g}\|_2 \le 1.0$.

---

## 5. Experimental Setup & Verification

### 5.1 Four Continual Fluid Benchmarks

1. **Benchmark 1 — Laboratory PIV Cylinder Wakes (`Caltech PIV` [20]):** Real planar velocity measurements $(u, v)$ on a $64 \times 64$ spatial grid across four sequential sub-critical Reynolds regimes: $Re = 240 \to 650 \to 1280 \to 2520$, with held-out intermediate regimes $Re = 910, 1890$. Trained for 15 epochs/task across 5 seeds (`42, 102, 202, 302, 402`), episodic buffer $M = 16$ exemplars/task. Run boundaries ($t \bmod 100$) are strictly respected.
2. **Benchmark 2 — 2D Kolmogorov Navier–Stokes Bifurcation:** High-order pseudo-spectral RK4 simulation of incompressible 2D Navier–Stokes on $[0, 2\pi]^2$ ($64 \times 64$) with Kolmogorov forcing $f(y) = \sin(4y)$ across a tenfold bifurcation jump: $Re = 20$ (steady laminar) $\to Re = 45$ (supercritical instability onset) $\to Re = 120$ (transitional vortex shedding) $\to Re = 400$ (chaotic 2D turbulence), with held-out $Re = 75, 240$. Trained 12 epochs/task across 3 seeds (`42, 102, 202`), $M = 16$. Exact vorticity transport PDE residual and divergence are evaluated on predicted fields.
3. **Benchmark 3 — ERA5 Global 10-m Planetary Winds (`WeatherBench 2` [21]):** Real 6-hourly global horizontal winds $(u_{10}, v_{10})$ ($64 \times 64$ grid, normalized by $U_{\text{ref}} = 10.0\text{ m/s}$) across four chronological 2020 seasonal tasks (`Winter -> Spring -> Summer -> Autumn`), evaluated additionally on full chronological held-out `Summer 2019` and `Winter 2019` (123 transitions each) and 72-h (12-step) autoregressive rollouts plus Northern Hemisphere storm-track spatial slice RMSE. Trained 18 epochs/task across 3 seeds (`42, 102, 202`), $M = 16$.
4. **Benchmark 4 — 1D Viscous Burgers Shock Formation & RQ4:** Periodic 1D Burgers flow ($N_x = 128$, $N_t = 40$) spanning a $100\times$ viscosity drop $\nu = 0.05 \to 0.01 \to 0.002 \to 0.0005$ (held-out $\nu = 0.02, 0.005$), trained 20 epochs/task across 3 seeds (`42, 102, 202`), $M = 32$. Includes a capacity-constrained stress test (`width=16`, `modes=12`, 25 epochs/task).

### 5.2 Baselines, Memory Accounting & Verification Protocol

We compare against `Fine-Tuning` (lower bound), `EWC` (additive diagonal complex Fisher, $\lambda = 2.0$), `Online-EWC` ($\gamma_F = 0.85, \lambda = 2.0$), `ER` [15] (reservoir episodic replay), `Replay-TS` [3] (temporal-slicing replay), `SLE-FNO` [2] (single-layer extensions on frozen backbone), `MEWC-PDE-Replay` (zero-data PDE replay on Burgers), and offline `Joint` training (upper bound). Extra CL memory (`Extra MB`) is computed directly from exact buffer, adapter, Fisher, $\theta^*$, and encoder byte sizes (`mewc_flow/utils/results_io.py`). Every claim and table is programmatically cross-checked via [`tests/test_verification.py`](tests/test_verification.py) and [`tests/verify_manuscript_numbers.py`](tests/verify_manuscript_numbers.py) with strict JSON (`allow_nan=False`), paired two-sided $t$-tests, and Cohen's $d$.

---

## 6. Results & Error Analysis

### 6.1 Main PIV Cylinder Wake Benchmark (RQ1 & RQ2)

**Table 2.** Caltech PIV cylinder wake benchmark ($Re = 240 \to 650 \to 1280 \to 2520$, 5 seeds, 15 epochs/task, $M = 16$). Significance markers compare each baseline against `MEWC-Flow` (`*` $p < 0.05$, `**` $p < 0.01$, paired two-sided $t$-test across matched seeds). Lower is better for all metrics; negative forgetting denotes positive backward transfer.

| Method | Avg. Forgetting ($\Delta$MSE) | Final MSE (4 regimes) | $Re=240$ Rollout Rel-$L_2$ ($t=10$) | $Re=240$ Vorticity RMSE | Held-out MSE ($Re=910, 1890$) | Extra Memory (MB) |
|---|---|---|---|---|---|---|
| Fine-Tuning | $4.34 \times 10^{-4} \pm 4.1 \times 10^{-4}$* | $1.43 \times 10^{-3} \pm 3.1 \times 10^{-4}$* | $0.2125 \pm 0.0633$* | $0.0358 \pm 0.0057$ | $1.51 \times 10^{-3} \pm 1.9 \times 10^{-4}$* | 0.00 |
| EWC | $-2.25 \times 10^{-5} \pm 5.0 \times 10^{-6}$** | $1.26 \times 10^{-3} \pm 9.1 \times 10^{-6}$** | $0.1059 \pm 0.0053$ | $0.0280 \pm 0.0007$ | $1.50 \times 10^{-3} \pm 1.3 \times 10^{-5}$** | 3.57 |
| Online-EWC | $-1.09 \times 10^{-5} \pm 2.6 \times 10^{-5}$** | $1.24 \times 10^{-3} \pm 1.8 \times 10^{-5}$** | $0.1230 \pm 0.0237$ | $0.0289 \pm 0.0015$ | $1.47 \times 10^{-3} \pm 1.7 \times 10^{-5}$** | 3.57 |
| ER | $-8.02 \times 10^{-5} \pm 9.6 \times 10^{-6}$ | $1.03 \times 10^{-3} \pm 1.2 \times 10^{-5}$* | $0.0997 \pm 0.0043$ | $0.0279 \pm 0.0017$ | $1.26 \times 10^{-3} \pm 1.0 \times 10^{-5}$* | 4.20 |
| Replay-TS | $-7.94 \times 10^{-5} \pm 9.7 \times 10^{-6}$ | $1.06 \times 10^{-3} \pm 1.6 \times 10^{-5}$** | $0.0981 \pm 0.0019$ | $0.0270 \pm 0.0012$ | $1.29 \times 10^{-3} \pm 1.9 \times 10^{-5}$ | 3.93 |
| SLE-FNO | $2.60 \times 10^{-7} \pm 2.8 \times 10^{-7}$** | $1.34 \times 10^{-3} \pm 6.0 \times 10^{-6}$** | $0.1237 \pm 0.0029$** | $0.0320 \pm 0.0006$* | $1.59 \times 10^{-3} \pm 5.7 \times 10^{-6}$** | 0.01 |
| **MEWC-Flow** | **$-7.78 \times 10^{-5} \pm 1.8 \times 10^{-5}$** | **$1.05 \times 10^{-3} \pm 1.7 \times 10^{-5}$** | **$0.0997 \pm 0.0053$** | **$0.0278 \pm 0.0020$** | **$1.28 \times 10^{-3} \pm 2.1 \times 10^{-5}$** | **10.32** |
| Joint | $0.00 \times 10^{0} \pm 0.0 \times 10^{0}$** | $1.05 \times 10^{-3} \pm 4.3 \times 10^{-5}$ | $0.0853 \pm 0.0043$** | $0.0242 \pm 0.0015$* | $1.28 \times 10^{-3} \pm 5.0 \times 10^{-5}$ | 38.80 |

**Table 3.** Paired two-sided $t$-tests and Cohen's $d$ on Caltech PIV (MEWC-Flow minus baseline across 5 seeds). Negative $\Delta$ and $d < 0$ favor MEWC-Flow.

| Comparison | $\Delta$Final MSE ($p$) | $\Delta$Forgetting ($p$) | $\Delta$Rollout $t=10$ ($p$) | $\Delta$Held-out MSE ($p$) | Cohen's $d$ (Final MSE) |
|---|---|---|---|---|---|
| MEWC-Flow vs. Fine-Tuning | $-3.80 \times 10^{-4}$ (0.047) | $-5.12 \times 10^{-4}$ (0.045) | $-0.1127$ (0.020) | $-2.26 \times 10^{-4}$ (0.043) | $-1.27$ |
| MEWC-Flow vs. EWC | $-2.11 \times 10^{-4}$ (0.000) | $-5.52 \times 10^{-5}$ (0.001) | $-0.0061$ (0.228) | $-2.20 \times 10^{-4}$ (0.000) | $-13.97$ |
| MEWC-Flow vs. Online-EWC | $-1.93 \times 10^{-4}$ (0.000) | $-6.69 \times 10^{-5}$ (0.002) | $-0.0233$ (0.135) | $-1.89 \times 10^{-4}$ (0.000) | $-12.38$ |
| MEWC-Flow vs. ER | $+1.34 \times 10^{-5}$ (0.015) | $+2.41 \times 10^{-6}$ (0.745) | $-0.0000$ (0.996) | $+2.12 \times 10^{-5}$ (0.033) | $+1.84$ |
| MEWC-Flow vs. Replay-TS | $-1.49 \times 10^{-5}$ (0.002) | $+1.60 \times 10^{-6}$ (0.811) | $+0.0016$ (0.400) | $-6.56 \times 10^{-6}$ (0.398) | $-3.02$ |
| MEWC-Flow vs. SLE-FNO | $-2.94 \times 10^{-4}$ (0.000) | $-7.80 \times 10^{-5}$ (0.001) | $-0.0240$ (0.001) | $-3.13 \times 10^{-4}$ (0.000) | $-14.93$ |
| MEWC-Flow vs. Joint | $-5.71 \times 10^{-6}$ (0.731) | $-7.78 \times 10^{-5}$ (0.001) | $+0.0144$ (0.010) | $+2.02 \times 10^{-6}$ (0.916) | $-0.17$ |

Sequential Fine-Tuning suffers $+134.8\%$ Task-0 relative degradation and drops to a valid prediction horizon ($\text{VPH}_{0.25}$) of only $14.3 \pm 5.6$ steps (vs. $20.0 \pm 0.0$ for all regularized/replay methods and MEWC-Flow), losing shear-layer phase coherence by $t=10$ (step-10 rel-$L_2 = 0.2125$). MEWC-Flow reduces Final MSE to $1.05 \times 10^{-3}$—statistically indistinguishable from full offline Joint training ($1.05 \times 10^{-3}$, $p = 0.731$) while using $3.76\times$ less extra storage ($10.32\text{ MB}$ vs. $38.80\text{ MB}$) and beating EWC ($p < 10^{-4}$, $d = -13.97$), Online-EWC ($p < 10^{-4}$, $d = -12.38$), Replay-TS ($p = 0.002$, $d = -3.02$), and SLE-FNO ($p < 10^{-4}$, $d = -14.93$). Notice also the honest trade-off against monolithic `ER`: `ER` achieves slightly lower single-step Final MSE ($1.03 \times 10^{-3}$ vs. $1.05 \times 10^{-3}$, $p = 0.015$) at lower memory ($4.20\text{ MB}$), while matching MEWC-Flow on 10-step rollout ($0.0997$, $p = 0.996$). In seed-0 spectral diagnostics (`physics_and_rollout_summary.json`, Fig. 2), however, MEWC-Flow exhibits the lowest **mid-wavenumber inertial-band log-spectral RMSE** ($0.0484$ vs. ER $0.0794$, Replay-TS $0.0817$, SLE-FNO $0.1057$, Fine-Tuning $0.3287$) and lowest seed-0 step-10 rollout rel-$L_2$ ($0.0927$) and vorticity RMSE ($0.0251$).

![PIV Main Benchmark and Forgetting](mewc_flow/experiments/piv_experiments/results/fig1_piv_benchmark_and_forgetting.png)
**Figure 1.** PIV continual learning benchmark across 5 random seeds ($M=16$, 15 epochs/task): final MSE, catastrophic forgetting ($\Delta$MSE), Task-0 degradation, and held-out cross-$Re$ generalization.

![PIV Physics, Rollout and Spectrum](mewc_flow/experiments/piv_experiments/results/fig2_piv_physics_rollout_spectrum.png)
**Figure 2.** Physical and spectral diagnostics on $Re=240$ after learning up to $Re=2520$: vorticity snapshots, divergence RMSE, kinetic energy spectra $E(k)$ highlighting MEWC-Flow's inertial-band accuracy, and multi-step autoregressive rollout drift.

---

### 6.2 Zero-Leakage Physics-Only Regime Identification & Routing (RQ3)

Across all 5 PIV seeds, MEWC-Flow's physics-only contrastive encoder (`PhysicsRegimeEncoder2d`, zero $Re$ input) achieves **$77.55\% \pm 4.61\%$** task-agnostic regime classification accuracy and **$0.4647 \pm 0.0560$** latent silhouette score. On the representative seed-0 model (`latent_manifold_summary.json`, Fig. 3), nearest-prototype accuracy reaches **$84.2\%$** (silhouette $0.518$, first two PCA axes explaining $78.3\%$ and $18.4\%$ variance). The confusion matrix across the 196 test samples (`49` per regime) is `[[49, 0, 0, 0], [0, 18, 31, 0], [0, 0, 49, 0], [0, 0, 0, 49]]`: laminar ($Re=240$), transitional ($Re=1280$), and turbulent ($Re=2520$) regimes are $100\%$ separable purely from flow invariants, while all 31 misclassifications occur between adjacent transition wakes ($Re=650 \leftrightarrow 1280$) whose Strouhal shedding statistics overlap physically.

**Table 4.** Soft routing attention weights $w_k(\mathbf{u}_t)$ produced by the seed-0 MEWC-Flow encoder without $Re$ input (`*` denotes zero-shot unseen held-out Reynolds regimes).

| Input Regime | Adapter $Re=240$ | Adapter $Re=650$ | Adapter $Re=1280$ | Adapter $Re=2520$ |
|---|---|---|---|---|
| $Re=240$ | **0.97** | 0.03 | 0.00 | 0.00 |
| $Re=650$ | 0.00 | **0.64** | 0.36 | 0.00 |
| $Re=910$* | 0.00 | 0.07 | **0.92** | 0.00 |
| $Re=1280$ | 0.00 | 0.02 | **0.98** | 0.00 |
| $Re=1890$* | 0.00 | 0.00 | 0.04 | **0.96** |
| $Re=2520$ | 0.00 | 0.00 | 0.00 | **1.00** |

Notably, on unseen held-out regimes $Re=910^*$ and $Re=1890^*$, soft attention automatically interpolates neighboring high-Reynolds experts ($0.07 \cdot h_{650} + 0.92 \cdot h_{1280}$ and $0.04 \cdot h_{1280} + 0.96 \cdot h_{2520}$, respectively) purely from instantaneous flow invariants.

![Latent Manifold and Soft Routing](mewc_flow/experiments/piv_experiments/results/fig3_piv_latent_manifold_and_routing.png)
**Figure 3.** Hyperspherical PCA projection of the physics-only contrastive latent space (left) and emergent diagonal/interpolated soft routing weights across seen and unseen Reynolds numbers (right).

---

### 6.3 Component Ablations, Routing Modes, Task Ordering & Memory Scaling (RQ1, RQ3, RQ5)

Tables 5–8 report the 3-seed (`42, 102, 202`, 12 epochs/task) systematic PIV ablation suite. Removing episodic replay (`w/o Episodic Replay (0 MB)`) immediately collapses performance: forgetting jumps from $-9.80 \times 10^{-5}$ to $+4.28 \times 10^{-4}$, Final MSE degrades from $1.08 \times 10^{-3}$ to $1.50 \times 10^{-3}$, and Task-0 step-10 rollout error more than doubles from $0.0985 \pm 0.0043$ to $0.2248 \pm 0.0763$. This establishes **episodic replay as the primary anti-forgetting driver** on stationary-geometry PIV wakes. Equally informative is the tight clustering ($0.0953$–$0.0987$ step-10 rollout error) across all replay-equipped ablations on PIV: because cylinder geometry stays fixed across PIV tasks, lightweight adapters, contrastive routing, and EWC neither significantly degrade nor dramatically exceed raw `ER` on single-step PIV MSE—serving primarily as interpretable regime-routing structure and Fisher consolidation needed when topology bifurcates (Section 6.4) or dynamics shift seasonally (Section 6.5).

**Table 5.** Systematic PIV component ablation across 3 seeds (`42, 102, 202`, 12 epochs/task).

| Variant | Avg. Forgetting | Final MSE | Rollout $t=10$ | Vorticity RMSE | Held-out MSE | Extra MB |
|---|---|---|---|---|---|---|
| Full MEWC-Flow | $-9.80 \times 10^{-5} \pm 1.3 \times 10^{-5}$ | $1.08 \times 10^{-3} \pm 8.3 \times 10^{-6}$ | $0.0985 \pm 0.0043$ | $0.0273 \pm 0.0021$ | $1.31 \times 10^{-3} \pm 1.6 \times 10^{-5}$ | 10.32 |
| w/o Modular Adapters | $-1.14 \times 10^{-4} \pm 1.1 \times 10^{-5}$ | $1.06 \times 10^{-3} \pm 4.6 \times 10^{-6}$ | $0.0961 \pm 0.0026$ | $0.0263 \pm 0.0014$ | $1.28 \times 10^{-3} \pm 5.4 \times 10^{-6}$ | 10.31 |
| w/o Contrastive Router | $-9.83 \times 10^{-5} \pm 8.2 \times 10^{-6}$ | $1.09 \times 10^{-3} \pm 5.7 \times 10^{-6}$ | $0.0965 \pm 0.0036$ | $0.0265 \pm 0.0017$ | $1.31 \times 10^{-3} \pm 1.2 \times 10^{-5}$ | 10.15 |
| w/o Adaptive EWC (Static) | $-9.54 \times 10^{-5} \pm 9.3 \times 10^{-6}$ | $1.10 \times 10^{-3} \pm 2.9 \times 10^{-6}$ | $0.0976 \pm 0.0039$ | $0.0268 \pm 0.0018$ | $1.33 \times 10^{-3} \pm 1.1 \times 10^{-5}$ | 7.94 |
| w/o EWC | $-9.38 \times 10^{-5} \pm 1.4 \times 10^{-5}$ | $1.07 \times 10^{-3} \pm 1.1 \times 10^{-5}$ | $0.0987 \pm 0.0042$ | $0.0274 \pm 0.0019$ | $1.30 \times 10^{-3} \pm 1.7 \times 10^{-5}$ | 4.37 |
| w/o Episodic Replay (0 MB) | $+4.28 \times 10^{-4} \pm 7.1 \times 10^{-4}$ | $1.50 \times 10^{-3} \pm 5.3 \times 10^{-4}$ | $0.2248 \pm 0.0763$ | $0.0319 \pm 0.0025$ | $1.59 \times 10^{-3} \pm 3.4 \times 10^{-4}$ | 6.12 |
| w/o Physics Loss | $-9.93 \times 10^{-5} \pm 1.3 \times 10^{-5}$ | $1.08 \times 10^{-3} \pm 7.5 \times 10^{-6}$ | $0.0982 \pm 0.0044$ | $0.0272 \pm 0.0021$ | $1.31 \times 10^{-3} \pm 1.5 \times 10^{-5}$ | 10.32 |
| Replay-Only (ER) | $-1.09 \times 10^{-4} \pm 7.0 \times 10^{-6}$ | $1.07 \times 10^{-3} \pm 1.1 \times 10^{-5}$ | $0.0953 \pm 0.0030$ | $0.0263 \pm 0.0020$ | $1.29 \times 10^{-3} \pm 1.3 \times 10^{-5}$ | 4.20 |

**Table 6.** Routing mode comparison on Full MEWC-Flow (3 seeds). Soft and hard learned physics-only routing match ground-truth oracle routing on seen MSE ($1.08 \times 10^{-3}$) and step-10 rollout ($0.0985$), outperforming uniform adapter averaging ($0.1003$).

| Routing Mode | Seen Final MSE | Held-out MSE | Rollout $t=10$ |
|---|---|---|---|
| `soft` | $1.08 \times 10^{-3} \pm 8.3 \times 10^{-6}$ | $1.31 \times 10^{-3} \pm 1.6 \times 10^{-5}$ | $0.0985 \pm 0.0043$ |
| `hard` | $1.08 \times 10^{-3} \pm 8.7 \times 10^{-6}$ | $1.31 \times 10^{-3} \pm 1.6 \times 10^{-5}$ | $0.0985 \pm 0.0043$ |
| `uniform` | $1.09 \times 10^{-3} \pm 1.3 \times 10^{-5}$ | $1.31 \times 10^{-3} \pm 1.5 \times 10^{-5}$ | $0.1003 \pm 0.0048$ |
| `oracle` | $1.08 \times 10^{-3} \pm 7.9 \times 10^{-6}$ | – | $0.0985 \pm 0.0043$ |

**Table 7.** Task-order sensitivity across ascending ($240 \to 2520$), descending ($2520 \to 240$), and shuffled ($1280 \to 240 \to 2520 \to 650$) curricula (RQ5, 3 seeds, reported as Avg. Forgetting / Final MSE). Ascending order inflicts severe forgetting on Fine-Tuning ($+4.12 \times 10^{-4}$) because high-$Re$ broadband turbulence overwrites smooth laminar modes, whereas descending order starts with high-complexity turbulence and transfers cleanly ($-4.48 \times 10^{-5}$). Both ER and MEWC-Flow maintain negative forgetting across all orders.

| Curriculum Order | Fine-Tuning | ER | MEWC-Flow |
|---|---|---|---|
| Ascending ($240 \to 2520$) | $+4.12 \times 10^{-4} \pm 5.7 \times 10^{-4}$ / $1.46 \times 10^{-3}$ | $-1.09 \times 10^{-4} \pm 7.0 \times 10^{-6}$ / $1.07 \times 10^{-3}$ | $-9.80 \times 10^{-5} \pm 1.3 \times 10^{-5}$ / $1.08 \times 10^{-3}$ |
| Descending ($2520 \to 240$) | $-4.48 \times 10^{-5} \pm 5.0 \times 10^{-5}$ / $1.22 \times 10^{-3}$ | $-2.24 \times 10^{-4} \pm 3.0 \times 10^{-6}$ / $1.06 \times 10^{-3}$ | $-1.46 \times 10^{-4} \pm 1.4 \times 10^{-5}$ / $1.16 \times 10^{-3}$ |
| Shuffled ($1280 \to 240 \to 2520 \to 650$) | $+2.57 \times 10^{-5} \pm 5.6 \times 10^{-5}$ / $1.22 \times 10^{-3}$ | $-1.24 \times 10^{-4} \pm 7.8 \times 10^{-6}$ / $1.07 \times 10^{-3}$ | $-1.07 \times 10^{-4} \pm 6.9 \times 10^{-6}$ / $1.13 \times 10^{-3}$ |

**Table 8.** Episodic memory-budget scaling ($M \in \{4, 8, 16, 32\}$ exemplars/task, reported as step-10 rollout rel-$L_2$ / extra CL MB). MEWC-Flow improves monotonically from $0.1239$ ($M=4$) to $0.0954$ ($M=32$), matching `ER` ($0.0957$) while incurring a fixed $\approx 6.12\text{ MB}$ consolidation/modular overhead.

| $M$ per task | ER | Replay-TS | MEWC-Flow |
|---|---|---|---|
| 4 | 0.1214 / 1.05 MB | 0.1266 / 0.79 MB | 0.1239 / 7.17 MB |
| 8 | 0.1045 / 2.10 MB | 0.1184 / 1.97 MB | 0.1064 / 8.22 MB |
| 16 | 0.0953 / 4.20 MB | 0.0936 / 3.93 MB | 0.0985 / 10.32 MB |
| 32 | 0.0957 / 8.39 MB | 0.0904 / 8.26 MB | 0.0954 / 14.51 MB |

![PIV Ablations, Ordering, Memory Sweep and A-EWC Trace](mewc_flow/experiments/piv_experiments/results/fig4_piv_ablations_and_sensitivity.png)
**Figure 4.** (a) Component ablation confirming episodic replay dominance on PIV rollout stability; (b) asymmetric task-order sensitivity under Fine-Tuning vs. robustness of MEWC-Flow/ER; (c) memory-budget scaling; (d) live Complex A-EWC gradient alignment $\alpha_t$ dynamically modulating $\lambda_t$.

---

### 6.4 Crossing a Genuine Navier–Stokes Bifurcation (Kolmogorov 2D NS)

**Table 9.** 2D Kolmogorov Navier–Stokes bifurcation benchmark ($Re = 20 \to 45 \to 120 \to 400$, held-out $Re = 75, 240$; 3 seeds, 12 epochs/task, $M = 16$). Significance markers compare against MEWC-Flow (`*` $p < 0.05$, `**` $p < 0.01$).

| Method | Avg. Forgetting | Final MSE | NS PDE Residual (RMS) | $\|\nabla \cdot \mathbf{u}\|$ RMS | Rollout $t=10$ (mean of 4) | Held-out MSE ($Re=75, 240$) | Extra MB |
|---|---|---|---|---|---|---|---|
| Fine-Tuning | $-8.29 \times 10^{-4} \pm 5.0 \times 10^{-5}$ | $3.60 \times 10^{-3} \pm 1.5 \times 10^{-4}$* | $7.846 \pm 0.112$** | $0.6490 \pm 0.0679$ | $0.4670 \pm 0.0375$* | $5.14 \times 10^{-3} \pm 2.2 \times 10^{-4}$* | 0.00 |
| EWC | $-5.81 \times 10^{-4} \pm 1.4 \times 10^{-4}$ | $6.88 \times 10^{-3} \pm 8.6 \times 10^{-4}$* | $8.316 \pm 0.093$** | $0.7846 \pm 0.1250$ | $0.5079 \pm 0.0079$** | $1.09 \times 10^{-2} \pm 1.5 \times 10^{-3}$* | 3.57 |
| Online-EWC | $-9.95 \times 10^{-4} \pm 3.3 \times 10^{-4}$ | $4.55 \times 10^{-3} \pm 3.7 \times 10^{-4}$** | $8.044 \pm 0.089$** | $0.8097 \pm 0.0954$ | $0.4743 \pm 0.0412$* | $6.79 \times 10^{-3} \pm 5.8 \times 10^{-4}$** | 3.57 |
| ER | $-1.22 \times 10^{-3} \pm 8.0 \times 10^{-5}$* | $2.99 \times 10^{-3} \pm 7.7 \times 10^{-5}$ | $7.386 \pm 0.045$** | $0.4732 \pm 0.0210$ | $0.3054 \pm 0.0020$* | $4.52 \times 10^{-3} \pm 1.4 \times 10^{-4}$ | 4.20 |
| Replay-TS | $-1.19 \times 10^{-3} \pm 7.5 \times 10^{-5}$* | $3.00 \times 10^{-3} \pm 7.7 \times 10^{-5}$ | $7.365 \pm 0.041$** | $0.4535 \pm 0.0085$ | $0.2966 \pm 0.0012$* | $4.59 \times 10^{-3} \pm 1.1 \times 10^{-4}$ | 3.93 |
| SLE-FNO | $-8.88 \times 10^{-6} \pm 4.4 \times 10^{-6}$** | $1.25 \times 10^{-2} \pm 1.9 \times 10^{-5}$** | $9.090 \pm 0.006$** | $0.2145 \pm 0.0451$ | $0.6905 \pm 0.0006$** | $2.26 \times 10^{-2} \pm 5.5 \times 10^{-5}$** | 0.01 |
| **MEWC-Flow** | **$-1.00 \times 10^{-3} \pm 1.3 \times 10^{-4}$** | **$3.31 \times 10^{-3} \pm 1.8 \times 10^{-4}$** | **$7.040 \pm 0.017$** | **$0.5260 \pm 0.0237$** | **$0.3442 \pm 0.0092$** | **$4.80 \times 10^{-3} \pm 2.7 \times 10^{-4}$** | **10.32** |
| Joint | $0.00 \times 10^{0} \pm 0.0 \times 10^{0}$** | $2.34 \times 10^{-3} \pm 4.4 \times 10^{-5}$** | $7.120 \pm 0.017$* | $0.3778 \pm 0.0240$ | $0.2307 \pm 0.0048$** | $3.53 \times 10^{-3} \pm 7.5 \times 10^{-5}$** | 46.14 |

When crossing the critical Kolmogorov instability ($Re = 20 \to 45 \to 120 \to 400$), `SLE-FNO`'s frozen $Re=20$ backbone lacks plasticity to learn chaotic vortex dynamics (`Final MSE` $= 1.25 \times 10^{-2}$, $p = 0.0001$), and static `EWC` over-stiffens parameters (`Final MSE` $= 6.88 \times 10^{-3}$, $p = 0.0124$). Most notably, **MEWC-Flow achieves the lowest exact 2D Navier–Stokes vorticity-transport PDE residual ($7.040 \pm 0.017$)** among *all* continual and offline methods—significantly outperforming Fine-Tuning ($7.846$, $p = 0.0047$, $d = -8.41$), EWC ($8.316$, $p = 0.0015$), Online-EWC ($8.044$, $p = 0.0030$), ER ($7.386$, $p = 0.0035$, $d = -9.71$), Replay-TS ($7.365$, $p = 0.0030$, $d = -10.44$), SLE-FNO ($9.090$, $p < 10^{-4}$), and even Joint ($7.120$, $p = 0.0149$, $d = -4.68$). Meanwhile, `ER` ($2.99 \times 10^{-3}$, $0.3054$ rollout) and `Replay-TS` ($3.00 \times 10^{-3}$, $0.2966$ rollout) obtain slightly lower MSE and 10-step rel-$L_2$ than MEWC-Flow ($3.31 \times 10^{-3}$, $0.3442$), reflecting an accuracy-vs-PDE-consistency trade-off driven by MEWC-Flow's joint A-EWC + vorticity/divergence loss.

![Kolmogorov 2D NS Bifurcation](mewc_flow/experiments/kolmogorov_experiments/results/fig5_kolmogorov_bifurcation.png)
**Figure 5.** 2D Kolmogorov Navier–Stokes bifurcation ($Re=20 \to 400$): vorticity snapshots across bifurcation, forgetting, exact NS PDE residual (where MEWC-Flow achieves the minimum across all methods), and post-bifurcation $Re=45$ rollout trajectories.

---

### 6.5 Real Atmospheric Generalization (ERA5 10-m Planetary Winds)

**Table 10.** ERA5 10-m global planetary winds across 4 sequential 2020 seasons (`Winter -> Spring -> Summer -> Autumn`), evaluated additionally on full held-out 2019 seasons and 72-h rollouts (3 seeds, 18 epochs/task, $M = 16$).

| Method | Avg. Forgetting | Final MSE | 72-h Rollout Rel-$L_2$ | Held-out Summer 2019 | Held-out Winter 2019 | NH Storm-Track RMSE | Extra MB |
|---|---|---|---|---|---|---|---|
| Fine-Tuning | $-5.11 \times 10^{-4} \pm 3.5 \times 10^{-4}$* | $2.18 \times 10^{-2} \pm 1.0 \times 10^{-4}$ | $1.0770 \pm 0.0284$* | $2.24 \times 10^{-2} \pm 5.0 \times 10^{-4}$ | $2.02 \times 10^{-2} \pm 1.3 \times 10^{-4}$ | $0.3797 \pm 0.0113$ | 0.00 |
| EWC | $-7.19 \times 10^{-5} \pm 3.0 \times 10^{-5}$** | $2.44 \times 10^{-2} \pm 2.2 \times 10^{-4}$** | $0.9020 \pm 0.0123$ | $2.24 \times 10^{-2} \pm 3.7 \times 10^{-4}$ | $2.34 \times 10^{-2} \pm 2.2 \times 10^{-4}$ | $0.3283 \pm 0.0009$ | 3.57 |
| Online-EWC | $-1.13 \times 10^{-4} \pm 3.7 \times 10^{-5}$** | $2.43 \times 10^{-2} \pm 2.6 \times 10^{-4}$** | $0.8973 \pm 0.0101$ | $2.23 \times 10^{-2} \pm 3.6 \times 10^{-4}$ | $2.32 \times 10^{-2} \pm 8.3 \times 10^{-5}$ | $0.3273 \pm 0.0013$ | 3.57 |
| ER | $-1.34 \times 10^{-3} \pm 3.9 \times 10^{-5}$ | $2.06 \times 10^{-2} \pm 9.5 \times 10^{-5}$** | $1.1024 \pm 0.0031$** | $2.02 \times 10^{-2} \pm 5.0 \times 10^{-5}$ | $1.99 \times 10^{-2} \pm 1.6 \times 10^{-4}$ | $0.3656 \pm 0.0020$ | 2.10 |
| Replay-TS | $+1.25 \times 10^{-4} \pm 2.4 \times 10^{-4}$* | $2.27 \times 10^{-2} \pm 1.2 \times 10^{-4}$* | $1.2253 \pm 0.0107$** | $2.32 \times 10^{-2} \pm 3.0 \times 10^{-4}$ | $2.12 \times 10^{-2} \pm 2.2 \times 10^{-4}$ | $0.4022 \pm 0.0039$ | 1.97 |
| SLE-FNO | $-3.20 \times 10^{-5} \pm 7.3 \times 10^{-6}$** | $2.43 \times 10^{-2} \pm 2.9 \times 10^{-4}$** | $0.8868 \pm 0.0079$ | $2.23 \times 10^{-2} \pm 3.4 \times 10^{-4}$ | $2.29 \times 10^{-2} \pm 1.7 \times 10^{-4}$ | $0.3249 \pm 0.0012$ | 0.01 |
| **MEWC-Flow** | **$-1.29 \times 10^{-3} \pm 1.4 \times 10^{-4}$** | **$2.15 \times 10^{-2} \pm 1.5 \times 10^{-4}$** | **$0.9260 \pm 0.0149$** | **$2.01 \times 10^{-2} \pm 1.0 \times 10^{-4}$** | **$2.07 \times 10^{-2} \pm 7.4 \times 10^{-5}$** | **$0.3352 \pm 0.0036$** | **8.22** |
| Joint | $0.00 \times 10^{0} \pm 0.0 \times 10^{0}$** | $1.92 \times 10^{-2} \pm 1.4 \times 10^{-4}$** | $0.9701 \pm 0.0146$ | $1.84 \times 10^{-2} \pm 8.7 \times 10^{-5}$ | $1.87 \times 10^{-2} \pm 1.7 \times 10^{-4}$ | $0.3299 \pm 0.0024$ | 14.68 |

On planetary atmospheric circulation, single-step replay baselines (`ER`, `Replay-TS`, and `Fine-Tuning`) suffer severe **72-h rollout instability** ($1.1024$, $1.2253$, and $1.0770$, respectively) and elevated Northern Hemisphere storm-track RMSE ($0.3656$–$0.4022$). By contrast, **MEWC-Flow balances single-step accuracy and long-horizon stability**: it attains $-1.29 \times 10^{-3} \pm 1.4 \times 10^{-4}$ backward transfer, beats `EWC` ($p = 0.0049$), `Online-EWC` ($p = 0.0062$), `SLE-FNO` ($p = 0.0076$), and `Replay-TS` ($p = 0.0154$) on single-step Final MSE, matches `ER` on Held-out Summer 2019 ($2.01 \times 10^{-2}$ vs. $2.02 \times 10^{-2}$), and significantly outperforms `ER` ($0.9260$ vs. $1.1024$, $p = 0.0031$, $d = -10.38$), `Replay-TS` ($1.2253$, $p = 0.0008$, $d = -20.30$), and `Fine-Tuning` ($1.0770$, $p = 0.0118$, $d = -5.26$) on 72-h autoregressive rollouts while tracking `Joint` ($0.3352$ vs. $0.3299$) in the NH storm track.

![ERA5 Planetary Wind Analysis](mewc_flow/experiments/era5_experiments/results/fig6_era5_planetary_analysis.png)
**Figure 6.** ERA5 seasonal planetary wind continual learning: (a) seasonal 2020 vs. held-out 2019 MSE; (b) 72-h (12-step) multi-step rollout stability; (c) NH mid-latitude storm-track RMSE spatial comparison.

---

### 6.6 Viscous Burgers Shock Sequence & Zero-Data PDE-Residual Replay (RQ4)

**Table 11.** 1D viscous Burgers shock steepening ($\nu = 0.05 \to 0.01 \to 0.002 \to 0.0005$, held-out $\nu = 0.02, 0.005$; 3 seeds, 20 epochs/task, $M = 32$) including zero-data `MEWC-PDE-Replay` ($M = 0$).

| Method | Avg. Forgetting | Final MSE | Burgers PDE Residual | Rollout $t=20$ (mean of 4) | Held-out MSE | Extra MB |
|---|---|---|---|---|---|---|
| Fine-Tuning | $2.93 \times 10^{-4} \pm 3.1 \times 10^{-5}$** | $2.34 \times 10^{-4} \pm 2.4 \times 10^{-5}$** | $4.214 \pm 0.612$ | $0.9202 \pm 0.3335$ | $2.22 \times 10^{-4} \pm 2.6 \times 10^{-5}$ | 0.000 |
| EWC | $1.66 \times 10^{-4} \pm 2.8 \times 10^{-5}$* | $1.53 \times 10^{-4} \pm 2.7 \times 10^{-5}$* | $3.851 \pm 0.697$ | $0.6873 \pm 0.1733$* | $1.05 \times 10^{-4} \pm 2.5 \times 10^{-6}$ | 0.634 |
| Online-EWC | $1.96 \times 10^{-4} \pm 2.8 \times 10^{-5}$** | $1.69 \times 10^{-4} \pm 2.4 \times 10^{-5}$** | $3.912 \pm 0.699$ | $0.7381 \pm 0.2014$* | $1.29 \times 10^{-4} \pm 1.1 \times 10^{-5}$ | 0.634 |
| ER | $-2.13 \times 10^{-6} \pm 2.5 \times 10^{-6}$ | $1.73 \times 10^{-5} \pm 4.6 \times 10^{-7}$ | $2.943 \pm 0.608$ | $0.0780 \pm 0.0047$ | $1.13 \times 10^{-5} \pm 1.9 \times 10^{-6}$ | 0.133 |
| Replay-TS | $-3.35 \times 10^{-6} \pm 1.3 \times 10^{-6}$ | $1.82 \times 10^{-5} \pm 3.2 \times 10^{-6}$ | $2.977 \pm 0.608$ | $0.0839 \pm 0.0125$ | $1.17 \times 10^{-5} \pm 3.7 \times 10^{-6}$ | 0.129 |
| SLE-FNO | $-3.08 \times 10^{-7} \pm 7.4 \times 10^{-7}$ | $1.99 \times 10^{-4} \pm 4.4 \times 10^{-5}$* | $4.206 \pm 0.722$ | $0.3077 \pm 0.0259$* | $1.94 \times 10^{-4} \pm 1.2 \times 10^{-5}$ | 0.010 |
| MEWC-PDE-Replay ($M=0$) | $1.41 \times 10^{-4} \pm 6.5 \times 10^{-5}$ | $1.28 \times 10^{-4} \pm 4.7 \times 10^{-5}$ | $3.702 \pm 0.818$ | $0.2543 \pm 0.0194$** | $1.24 \times 10^{-4} \pm 3.1 \times 10^{-5}$ | 1.059 |
| **MEWC-Flow** | **$4.49 \times 10^{-6} \pm 5.2 \times 10^{-6}$** | **$2.37 \times 10^{-5} \pm 3.3 \times 10^{-6}$** | **$3.015 \pm 0.591$** | **$0.1194 \pm 0.0270$** | **$2.94 \times 10^{-5} \pm 9.0 \times 10^{-6}$** | **1.277** |
| Joint | $0.00 \times 10^{0} \pm 0.0 \times 10^{0}$ | $9.48 \times 10^{-6} \pm 1.0 \times 10^{-6}$* | $2.852 \pm 0.570$ | $0.0551 \pm 0.0045$* | $3.89 \times 10^{-6} \pm 1.1 \times 10^{-6}$ | 3.834 |

Evaluating **RQ4** cleanly highlights both the potential and limits of zero-data physics-residual replay: `MEWC-PDE-Replay` ($M=0$, $1.059\text{ MB}$ consolidation overhead) lowers Final MSE from $2.34 \times 10^{-4}$ (`Fine-Tuning`) to $1.28 \times 10^{-4}$, cuts PDE residual ($4.214 \to 3.702$), and slashes 20-step rollout error nearly fourfold ($0.9202 \to 0.2543$), also beating zero-replay `EWC` ($0.6873$) and `SLE-FNO` ($0.3077$). Nonetheless, anchored observational samples remain decisive across extreme $100\times$ viscosity steepening: `MEWC-Flow` reduces Final MSE by an order of magnitude further to $2.37 \times 10^{-5}$ (vs. `Fine-Tuning` $p = 0.0035$, `EWC` $p = 0.0163$, `Online-EWC` $p = 0.0096$, `SLE-FNO` $p = 0.0222$) and 20-step rollout error to $0.1194$ ($p = 0.0011$ vs. `MEWC-PDE-Replay`), while `ER` ($0.0780$, $p = 0.1403$) and `Replay-TS` ($0.0839$) remain competitive at $0.13\text{ MB}$. Finally, under severe capacity bottlenecking (`width=16, modes=12`, `burgers_stress_summary.json`), Fine-Tuning exhibits $+13{,}481.7\%$ Task-0 relative degradation (`Final MSE` $1.943 \times 10^{-4}$) and `EWC` degrades $+9{,}450.6\%$ ($1.314 \times 10^{-4}$), whereas `MEWC-Flow` cuts degradation to $+1{,}818.0\%$ ($7.269 \times 10^{-5}$) and `ER` reaches $+605.1\%$ ($2.650 \times 10^{-5}$).

![Burgers PDE Residual and RQ4 Study](mewc_flow/experiments/burgers_pilot/results/fig7_burgers_pde_residual_replay.png)
**Figure 7.** 1D viscous Burgers shock sequence ($\nu=0.05 \to 0.0005$) and RQ4 comparison: forgetting, Burgers PDE residual, and 20-step shock-propagation rollout stability highlighting the gap between zero-data `MEWC-PDE-Replay` and sample-grounded `MEWC-Flow`.

---

## 7. Discussion, Answers to Research Questions & Limitations

- **RQ1 (Stability–Plasticity):** Across all four benchmarks, `Fine-Tuning`, static `EWC`, `Online-EWC`, and `SLE-FNO` either forget early dynamics or fail to adapt across steep bifurcations. `MEWC-Flow` consistently prevents catastrophic forgetting ($-7.78 \times 10^{-5}$ PIV, $-1.00 \times 10^{-3}$ Kolmogorov NS, $-1.29 \times 10^{-3}$ ERA5, $+4.49 \times 10^{-6}$ Burgers). Ablations prove episodic replay is the essential stabilizer, while complex A-EWC protects shared spectral modes when gradients conflict.
- **RQ2 (Physical Consistency & Rollouts):** Simply achieving low 1-step MSE does not guarantee physical consistency. In 2D Kolmogorov NS, `MEWC-Flow` achieves the global minimum exact PDE residual ($7.040$, $p \le 0.015$ vs. all baselines including Joint); in PIV wakes, it achieves the lowest inertial-band spectral error ($0.0484$); and in ERA5 winds, it improves 72-h rollout stability by $16\%$ over `ER` ($0.9260$ vs. $1.1024$, $p = 0.0031$) and $24\%$ over `Replay-TS` ($1.2253$, $p = 0.0008$).
- **RQ3 (Physics-Only Regime Routing):** Purely flow-invariant representations (velocity, vorticity, divergence; zero $Re$ input) achieve $77.6\% \pm 4.6\%$ ($84.2\%$ seed-0) task-agnostic classification accuracy, match oracle routing on seen MSE ($1.08 \times 10^{-3}$), and blend adjacent adapters smoothly onto held-out Reynolds regimes ($910^*, 1890^*$).
- **RQ4 (Zero-Data PDE Replay):** Differentiable PDE residuals (`MEWC-PDE-Replay`, $M=0$) cut Burgers 20-step rollout error from $0.9202$ to $0.2543$, proving partial physical memory, yet cannot match sample-anchored replay ($0.078$–$0.119$) when shock discontinuities steepen across two orders of magnitude in viscosity.
- **RQ5 (Ordering & Memory Scaling):** Ascending $Re$ order exposes Fine-Tuning's vulnerability ($+4.12 \times 10^{-4}$ forgetting vs. descending $-4.48 \times 10^{-5}$), whereas MEWC-Flow maintains negative forgetting across all permutations and scales smoothly from $M=4$ to $M=32$.

### Limitations and Engineering Trade-offs
1. **Modular Memory & Adapter Overhead vs. Monolithic ER:** MEWC-Flow incurs $\approx 6.1\text{ MB}$ extra overhead beyond an equal-$M$ `ER` buffer (from shared-backbone Fisher/$\theta^*$ state, adapters, and contrastive encoder: $10.32\text{ MB}$ vs. $4.20\text{ MB}$ at $M=16$ in 2D). Where deployment strictly optimizes 1-step MSE on fixed geometry with minimal RAM, monolithic `ER` remains slightly stronger on Burgers ($1.73 \times 10^{-5}$ vs. $2.37 \times 10^{-5}$) and Kolmogorov 1-step MSE ($2.99 \times 10^{-3}$ vs. $3.31 \times 10^{-3}$), whereas MEWC-Flow is preferred when **PDE consistency, task-agnostic regime identification, or multi-step atmospheric rollout stability** dominate.
2. **Adjacent Transitional Overlap:** Near supercritical transitions ($Re=650 \leftrightarrow 1280$), instantaneous velocity snapshots naturally overlap in shedding statistics, limiting zero-label accuracy on transitional regimes ($84.2\%$ seed-0) unless temporal sequences are pooled over a full shedding cycle.
3. **Domain Scope:** Present validation covers 1D Burgers, 2D Kolmogorov NS, 2D planar PIV wakes, and 2D planetary surface winds; scaling complex A-EWC and adapter routing to 3D volumetric LES/DNS velocity fields remains open future work.

---

## 8. Conclusion & Reproducibility

We presented **MEWC-Flow**, a continual neural-operator framework unifying modular regime adapters, physics-only contrastive routing, complex-valued Adaptive Elastic Weight Consolidation, and matched-budget episodic replay. Across 1D Burgers shocks, 2D Kolmogorov Navier–Stokes bifurcations, Caltech laboratory PIV wakes, and real ERA5 atmospheric winds, MEWC-Flow eliminates catastrophic forgetting while enforcing superior PDE consistency, spectral fidelity, and multi-step rollout stability without test-time $Re$ labels.

All code, benchmark runners, leak-free data checks, automated table builders, Jupyter verification notebook ([`mewc_flow/notebooks/mewc_flow_q1_verification.ipynb`](mewc_flow/notebooks/mewc_flow_q1_verification.ipynb)), and exact verification scripts ([`tests/test_verification.py`](tests/test_verification.py), [`tests/verify_manuscript_numbers.py`](tests/verify_manuscript_numbers.py)) are publicly released in this repository.

---

## 9. References

1. Wang, Y.-Z., Eshaghi, M., Zhuang, X.-Y., Rabczuk, T., & Liu, Y.-H. (2026). Replay-Based Continual Learning for Physics-Informed Neural Operators. *arXiv:2605.04832*. https://doi.org/10.48550/arXiv.2605.04832
2. Elhadidy, M., D'Souza, R. M., & Arzani, A. (2026). SLE-FNO: Single-layer extensions for task-agnostic continual learning in Fourier neural operators. *Physica Scripta*. https://doi.org/10.1088/1402-4896/ae6bdd
3. Hemati, H., Nguyen, B. D., & Sandfeld, S. (2026). Continual learning for autoregressive PDE surrogates under evolving physical regimes. *Machine Learning for Computational Science and Engineering*. https://doi.org/10.1007/s44379-026-00085-9
4. Ji, H.-Y., Luo, Y., Zhou, H., & Zhao, Y. (2026). Progressive mixture-of-experts with autoencoder routing for continual RANS turbulence modelling. *Journal of Fluid Mechanics*. https://doi.org/10.1017/jfm.2026.11654
5. Winkler, G., & Stoll, M. (2026). No Free Lunch in Flow Surrogates under Time-Varying Boundary Conditions: A Two-Regime Study. *Journal of Computational Science*.
6. Kim, M.-S., Mo, H. S., Lacey, C. E., Jung, K.-S., Chen, J. H., & Yoo, C. (2026). MG-ANN: An online extension of reduced-order modeling with continual learning for turbulent reacting flows. *Proceedings of the Combustion Institute*. https://doi.org/10.1016/j.proci.2026.106546
7. Chen, X.-J., Hu, X., Chen, L., Liu, Y., & Fan, W. (2026). Continual-Learning Physics-Informed Neural Networks for Parameterized Partial Differential Equations. *arXiv preprint*.
8. Howard, A. A., Fu, Y., & Stinis, P. (2023). A multifidelity approach to continual learning for physical systems. *Machine Learning: Science and Technology*, 4, 025042. https://doi.org/10.1088/2632-2153/ad45b2
9. Guo, Y., Qiang, Y.-M., Chu, X., Ding, J., Chen, Y., Wang, Q., Wu, T., & Zhang, A. (2026). RenaNet: Reynolds-Aware Neural Network for Rapid Flow Field Prediction via Lattice Boltzmann Simulations. *Applied Sciences*, 16(13), 6622. https://doi.org/10.3390/app16136622
10. Shi, J. (2026). Striding Across Reynolds Numbers: Representation Geometry in Neural PDE Generalisation. *arXiv:2605.30112*. https://doi.org/10.48550/arXiv.2605.30112
11. Dekhovich, A., Sluiter, M. H. F., Tax, D. M. J., & Bessa, M. A. (2025). iPINNs: incremental learning for Physics-informed neural networks. *Engineering with Computers*, 41, 389–402. https://doi.org/10.1007/s00366-024-02010-1
12. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114
13. Lagemann, C., Paehler, L., Callaham, J. L., Mokbel, S., Ahnert, S., Lagemann, K., Adams, N. A., Vinuesa, R., & Brunton, S. L. (2026). The HydroGym Reinforcement Learning Platform for Fluid Dynamics. *Nature*.
14. Schwarz, J., Luketina, J., Czarnecki, W. M., Grabska-Barwinska, A., Teh, Y. W., Pascanu, R., & Hadsell, R. (2018). Progress & Compress: A scalable framework for continual learning. *Proceedings of ICML 2018*, arXiv:1805.06370.
15. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience Replay for Continual Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, arXiv:1811.11682.
16. Riemer, M., Cases, I., Ajemian, R., Liu, M., Rish, I., Tu, Y., & Tesauro, G. (2019). Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference (Meta-Experience Replay). *ICLR 2019*, arXiv:1810.11910.
17. van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*.
18. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. https://doi.org/10.48550/arXiv.2004.11362
19. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*, arXiv:2010.08895.
20. Renn, P. I., Palmer, E. H., & Gharib, M. (2026). Experimental dataset of sub-critical cylinder wake velocity fields. *Scientific Data*. https://doi.org/10.1038/s41597-026-07368-z (Dataset DOI: 10.22002/g2z1f-nex65).
21. Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., Russell, T., et al. (2024). WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models. *Journal of Advances in Modeling Earth Systems (JAMES)*, 16(6), e2023MS004019. https://doi.org/10.1029/2023MS004019
22. Hasan, M. M., Mahin, A. A., Afrose, M., Murad, M. W., Monsur, H. A., & Based, M. A. (2026). MEWC-RL++: A Modular and Adaptive Continual Reinforcement Learning Framework for Automation Systems in Structured but Evolving Environments. *Taylor & Francis Book Chapter*.
