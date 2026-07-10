"""
Prompt-attention decay signal, A_t (paper Eq. 2, Section 3.1).

    A_t = (1/H) * sum_h mean_{j in prompt_slice} Attn_h(x_t, x_j)

Mean last-layer attention mass from the current decoding position to a fixed
slice of the initial prompt tokens (size K = config.prompt_slice_k).
"""

from typing import List, Optional

import numpy as np

from .config import DEFAULT_CONFIG, FatigueConfig


def prompt_span(base_len: int, config: FatigueConfig = DEFAULT_CONFIG) -> List[int]:
    """Fixed prompt-token index slice [0, K) used for the attention probe."""
    k = min(config.prompt_slice_k, base_len)
    return list(range(k))


def get_prompt_attention(attn_np: np.ndarray, span: List[int]) -> float:
    """
    attn_np: numpy array (H, S, S) -- last-layer attention weights for one
    forward pass (heads, query positions, key positions).

    Returns the mean attention mass from the last query position to the
    token indices in `span`. Returns 0.0 if inputs are degenerate.
    """
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        return 0.0
    if not span:
        return 0.0
    _, seq_len, _ = attn_np.shape
    valid = [i for i in span if 0 <= i < seq_len - 1]
    if not valid:
        return 0.0
    return float(np.nanmean(attn_np[:, -1, valid]))


def get_evidence_attention(attn_np: np.ndarray, evidence_span: List[int]) -> float:
    """Mean attention from the last query position to an evidence token span."""
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        return 0.0
    if not evidence_span:
        return 0.0
    _, seq_len, _ = attn_np.shape
    valid = [i for i in evidence_span if 0 <= i < seq_len]
    if not valid:
        return 0.0
    return float(np.nanmean(attn_np[:, -1, valid]))
