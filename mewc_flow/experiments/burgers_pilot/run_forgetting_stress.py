"""
run_forgetting_stress.py
========================
Capacity-constrained catastrophic forgetting stress test on 1-D Burgers
(narrow FNO width=16, extreme 100x viscosity jump nu: 0.05 -> 0.0005).
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mewc_flow.data.burgers_dataset import make_burgers_tasks
from mewc_flow.models.surrogate import build_surrogate
from mewc_flow.trainers.base_trainer import set_all_seeds
from mewc_flow.trainers.ewc_trainer import EWCTrainer
from mewc_flow.trainers.mewc_trainer import MEWCFlowTrainer
from mewc_flow.trainers.baselines import run_finetuning, run_experience_replay
from mewc_flow.utils.metrics import print_eval_matrix
from mewc_flow.utils.results_io import save_json


def main():
    seed = 42
    epochs = 25
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    set_all_seeds(seed)
    tasks, heldout = make_burgers_tasks(nx=128, nt=40, n_train=16, n_val=8, batch_size=32, include_heldout=True, seed=seed)

    results = {}
    for name, runner in [
        ("Fine-Tuning", lambda: run_finetuning(build_surrogate(nx=128, width=16, modes=12), tasks, heldout, epochs=epochs)),
        ("EWC", lambda: EWCTrainer(build_surrogate(nx=128, width=16, modes=12), lambda_ewc=2.0, epochs=epochs).run_all_tasks(tasks, heldout)),
        ("ER", lambda: run_experience_replay(build_surrogate(nx=128, width=16, modes=12), tasks, heldout, epochs=epochs, memory_per_task=24)),
        ("MEWC-Flow", lambda: MEWCFlowTrainer(
            build_surrogate(nx=128, width=16, modes=12, modular=True),
            epochs=epochs, consolidation_mode="adaptive", lambda_min=0.01, lambda_max=0.15,
            use_contrastive=True, use_replay=True, memory_per_task=24, use_physics_loss=True, gamma_physics=0.02, is_1d=True
        ).run_all_tasks(tasks, heldout)),
    ]:
        set_all_seeds(seed)
        res = runner()
        print(f"\n=== Stress Test: {name} ===")
        print_eval_matrix(res["eval_matrix"], [t["label"] for t in tasks])
        res.pop("per_task_history", None)
        res.pop("histories", None)
        results[name] = res

    save_json(results, os.path.join(out_dir, "burgers_stress_summary.json"))


if __name__ == "__main__":
    main()
