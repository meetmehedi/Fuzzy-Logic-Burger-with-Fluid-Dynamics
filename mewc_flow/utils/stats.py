"""
stats.py
========
Statistical Validation & Multi-Seed Hypothesis Testing for Q1 Journal Rigor.

Computes:
- Mean +/- Std across independent random seeds
- 95% Bootstrap Confidence Intervals
- Paired two-sided Student's t-test and Wilcoxon signed-rank p-values
- Cohen's d paired effect size
- Strict JSON sanitization (zero bare NaN/Inf values)
"""

import math
import numpy as np
from scipy import stats as sp_stats
from typing import List, Dict, Any, Sequence


def summarize_seeds(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if len(arr) >= 2:
        rng = np.random.default_rng(seed)
        boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    else:
        ci_low, ci_high = mean, mean

    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def paired_significance(
    proposed: Sequence[float],
    baseline: Sequence[float],
) -> Dict[str, Any]:
    """
    Paired two-sided t-test, Wilcoxon signed-rank test, and Cohen's d
    comparing `proposed` vs `baseline` across matched random seeds.
    (Lower metric = better; negative diff means proposed improves over baseline.)
    """
    x = np.asarray(proposed, dtype=np.float64)
    y = np.asarray(baseline, dtype=np.float64)
    assert len(x) == len(y), "Paired arrays must have equal length"
    diff = x - y
    n = len(diff)

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohens_d = float(mean_diff / (std_diff + 1e-12)) if n > 1 else 0.0

    if n >= 2 and std_diff > 1e-15:
        t_stat, p_ttest = sp_stats.ttest_rel(x, y)
        t_stat = float(t_stat)
        p_ttest = float(p_ttest)
    else:
        t_stat, p_ttest = 0.0, 1.0

    if n >= 3 and np.any(np.abs(diff) > 1e-15):
        try:
            _, p_wilcoxon = sp_stats.wilcoxon(x, y)
            p_wilcoxon = float(p_wilcoxon)
        except Exception:
            p_wilcoxon = p_ttest
    else:
        p_wilcoxon = p_ttest

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "cohens_d": cohens_d,
        "significant_005": bool(p_ttest < 0.05),
        "significant_001": bool(p_ttest < 0.01),
    }


def sanitize_json(obj: Any) -> Any:
    """Recursively convert numpy types and replace NaN/Inf with None for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
