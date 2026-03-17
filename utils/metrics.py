# utils/metrics.py
"""
Common STR evaluation metrics: accuracy, NED, edit distance
"""

import editdistance
import torch
import numpy as np


def accuracy(preds: list[str], gts: list[str]) -> float:
    """Exact match accuracy"""
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    return correct / len(gts) if gts else 0.0


def ned(preds: list[str], gts: list[str]) -> float:
    """Normalized Edit Distance (lower is better)"""
    total_dist = 0.0
    for p, g in zip(preds, gts):
        if not g:  # avoid div by zero
            total_dist += 0 if not p else 1.0
        else:
            dist = editdistance.eval(p, g)
            total_dist += dist / len(g)
    return total_dist / len(gts) if gts else 0.0


def edit_distance_stats(preds: list[str], gts: list[str]):
    """Detailed edit distance breakdown"""
    distances = [editdistance.eval(p, g) for p, g in zip(preds, gts)]
    return {
        "mean_ed": np.mean(distances),
        "median_ed": np.median(distances),
        "max_ed": np.max(distances),
        "zero_ed": sum(d == 0 for d in distances),
        "total_samples": len(distances)
    }


def print_metrics(preds, gts, name="Eval"):
    acc = accuracy(preds, gts)
    ned_val = ned(preds, gts)
    print(f"[{name}]")
    print(f"  Accuracy: {acc:.4f}  ({sum(p == g for p,g in zip(preds,gts))}/{len(gts)})")
    print(f"  NED:      {ned_val:.4f}")