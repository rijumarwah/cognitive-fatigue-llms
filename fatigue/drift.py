"""
Embedding drift signal, D_t (paper Eq. 3, Section 3.1).

    D_t = || h_t - h_0 ||_2

Euclidean distance between the top-layer hidden state at decoding step t
and the top-layer hidden state of the final prompt token (h_0), which
serves as the representational reference/anchor for the whole generation.
"""

from typing import Optional

import torch


def get_embedding_drift(
    current_hidden: Optional[torch.Tensor], ref_hidden: Optional[torch.Tensor]
) -> float:
    """L2 distance between the current top-layer hidden state and the anchor."""
    if current_hidden is None or ref_hidden is None:
        return 0.0
    return float(torch.norm(current_hidden - ref_hidden, p=2).item())
