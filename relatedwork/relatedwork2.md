# Related Work

## 2. Related Work

The literature relevant to MEWC-Flow spans five interconnected threads: continual learning (CL) for neural operators and PDE surrogates, Reynolds-number-aware surrogate models, modular and regime-routing architectures, physics-informed continual learning, and the foundational CL algorithms that supply the baselines and theoretical vocabulary.

---

### 2.1 Continual Learning for Neural Operators and PDE Surrogates

Most neural surrogate models for fluid dynamics are trained offline on a fixed distribution and degrade sharply when the deployment regime changes. This practical limitation has motivated a recent wave of CL methods specifically designed for scientific machine learning.

**Replay and distillation for neural operators.** Wang et al. [1] embed CL directly into physics-informed neural operators built on the Transolver architecture. When out-of-distribution (OOD) data arrive, a small number of past samples enter the loss through a distillation-based constraint that preserves prior knowledge, while a Low-Rank Adaptation (LoRA) module enables fast adaptation. Validated on Darcy flow, 2D hyperelasticity, and 3D linear elasticity, this work shows that pairing replay with architectural adaptation outperforms either mechanism alone, though it does not study unsteady Navier-Stokes sequences. Complementarily, Howard, Fu, and Stinis [8] formulate continual learning via a multifidelity deep neural network that learns correlations between previously trained predictions and new-domain targets, combining additively with replay and memory-aware synapses.

**Architecture-based continual learning in Fourier Neural Operators.** Elhadidy, D'Souza, and Arzani [2] introduce SLE-FNO, which augments a Fourier Neural Operator [19] with a Single-Layer Extension (SLE) added per task. Across four sequential OOD configurations built from 230 CFD simulations of pulsatile aneurysmal blood flow, SLE-FNO is compared against EWC, Learning without Forgetting (LwF), replay, OGD, GEM, PiggyBack, and LoRA. Replay and architectural extensions (PiggyBack, LoRA, SLE-FNO) achieved the strongest retention, whereas standalone parameter consolidation (EWC) struggled on spatial field-to-field regression. SLE-FNO provides a strong architectural baseline under matched backbone capacity.

**Temporal-slicing replay for autoregressive surrogates.** Autoregressive PDE surrogates face compounding rollout error, and storing full high-resolution trajectories quickly exhausts memory budgets. Hemati, Nguyen, and Sandfeld [3] propose Replay-TS, which stores compact contiguous temporal windows sampled across past simulations and applies sparse multi-step autoregressive supervision. Across Advection-Diffusion, Burgers, and Navier-Stokes streams, Replay-TS reduces MSE by over 30% relative to standard experience replay under a fixed memory budget. Replay-TS serves as a key matched-budget replay baseline in our evaluation.

---

### 2.2 Reynolds-Aware and Cross-Reynolds Surrogates

A surrogate encountering a new Reynolds number faces both sequential forgetting of earlier regimes and poor cross-Reynolds generalization even under single-regime training.

**Reynolds-conditioned surrogate models.** Guo et al. [9] propose RenaNet, a Reynolds-aware ConvGRU that ingests flow snapshots alongside an explicit Reynolds-number conditioning map over $1 \le Re \le 500$ for cylinder flow. Ablating the $Re$ channel degrades validation MSE by two orders of magnitude, confirming the necessity of explicit $Re$ conditioning. However, RenaNet is trained once offline on a fixed range without continual updating.

**Cross-Reynolds generalization and representation geometry.** Shi [10] demonstrates that a trained FNO incurs 46.68% relative $L_2$ error under a tenfold Reynolds-number shift on forced 2D Navier-Stokes, while latent-space retrieval (ConvAE-Relay) achieves 38.34% without target-regime fitting, isolating representation geometry and autoregressive drift (~12 percentage points) as dominant bottlenecks. Similarly, Winkler and Stoll [5] demonstrate a "No Free Lunch" effect in flow surrogacy: surrogates validated on steady regimes consistently fail to transfer to transient time-varying regimes regardless of capacity, reinforcing the necessity of continual adaptation rather than static scaling alone.

---

### 2.3 Modular and Regime-Routing Architectures

**Progressive mixture-of-experts for turbulence modeling.** Ji et al. [4] introduce Progressive Mixture-of-Experts (PMoE) for continual RANS turbulence closures, using an autoencoder-based router to assign flow scenarios to specialized experts and spawning new experts when novel regimes arise. While PMoE avoids forgetting by construction, it targets RANS closures rather than full autoregressive spatiotemporal velocity operators and routes via reconstruction error rather than contrastive metric geometry.

**Online regime switching via OOD detection.** Kim et al. [6] propose MG-ANN for turbulent reacting flows, coupling an autoencoder OOD detector with regularization-based CL to switch dynamically between neural inference and DNS. This provides precedent for task-agnostic regime identification during online fluid simulations.

---

### 2.4 Physics-Informed Continual Learning and Fluid Control

**Continual PINNs for parameterized PDEs.** Chen et al. [7] formulate CL-PINN, treating PDE instances across parameter values as sequential tasks using Bayesian active selection, dynamic loss weighting, and sparse physics-constrained replay without observational labels. Dekhovich et al. [11] propose iPINNs, using parameter sparsity to incrementally grow PINNs across PDE families. Both demonstrate that governing PDE residuals act as implicit physical memory—directly motivating our zero-data physics-residual replay comparison (RQ4) on 1D viscous Burgers flow.

**Fluid dynamics RL platforms.** Lagemann et al. [13] introduce HydroGym, a solver-independent RL benchmark for active flow control across varying Reynolds regimes, highlighting the growing intersection between continual RL (such as our prior MEWC-RL++ framework [22]) and fluid-mechanics surrogate modeling.

---

### 2.5 Foundational Continual Learning & Contrastive Methods

- **Elastic Weight Consolidation (EWC) & Online EWC:** Kirkpatrick et al. [12] penalize drift on parameters weighted by the diagonal Fisher Information Matrix; Schwarz et al. [14] bound memory to $\mathcal{O}(|\theta|)$ via exponential Fisher decay (Online EWC). Our prior **MEWC-RL++** [22] extends Online EWC with modular shared-bottom heads and **Adaptive EWC (A-EWC)**, modulating consolidation strength $\lambda_t$ dynamically via EMA gradient cosine alignment.
- **Experience Replay & Meta-Experience Replay:** Rolnick et al. [15] establish compact episodic replay as a robust task-agnostic retention baseline, while Riemer et al. [16] frame continual learning as a gradient-alignment trade-off between transfer and interference.
- **Contrastive Representation Learning:** van den Oord et al. [17] (InfoNCE) and Khosla et al. [18] (Supervised Contrastive Loss) provide the hyperspherical metric learning objective used in MEWC-Flow's physics-only regime encoder.

---

### 2.6 Summary: Position of MEWC-Flow

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
| **MEWC-Flow (Ours)** | **Modular Re-FNO** | **Adapters + Complex A-EWC/OFU + Physics-Only SupCon Router + Matched Replay + Vorticity/PDE Loss** | **Yes (NS $Re=20 \to 400$ + PIV $240 \to 2520$)** | **Yes (Lab PIV [20] + ERA5 [21])** | **Yes (Vorticity, $\nabla\cdot\mathbf{u}$, $E(k)$, PDE res., VPH, slices)** |

---

### References

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
