"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) Torch RNGs for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_seeds(seeds_str, fallback_seed: int):
    """Parse a comma/space-separated seed string into a list of ints."""
    import re
    if not seeds_str:
        return [int(fallback_seed)]
    parts = re.split(r"[,\s]+", str(seeds_str).strip())
    seeds = []
    for p in parts:
        if not p:
            continue
        try:
            seeds.append(int(p))
        except ValueError:
            continue
    return seeds or [int(fallback_seed)]
