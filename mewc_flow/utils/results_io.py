"""
results_io.py
=============
Utilities for saving and loading strict-JSON experiment results and checkpoints.
"""

import json
import os
import time
import torch
from typing import Dict, Any

from mewc_flow.utils.stats import sanitize_json


def save_json(data: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clean = sanitize_json(data)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    return path


def save_results(results: Dict[str, Any], out_dir: str, tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_{ts}" if tag else ts
    json_path = os.path.join(out_dir, f"{prefix}_results.json")
    save_json(results, json_path)
    print(f"  [IO] Results saved -> {json_path}")
    return json_path


def save_checkpoint(model, out_dir: str, tag: str = "", filename: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    if filename:
        name = filename
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"{tag}_{ts}_model.pt" if tag else f"{ts}_model.pt"
    path = os.path.join(out_dir, name)
    torch.save(model.state_dict(), path)
    return path
