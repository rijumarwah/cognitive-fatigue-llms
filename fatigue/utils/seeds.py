"""Reproducibility helpers. Paper Appendix C.1 uses seeds {123, 2027}."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
