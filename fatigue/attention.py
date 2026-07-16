"""Attention-to-prompt signal (A_t).

A primary long-horizon failure mode is loss of instruction adherence: as
decoding proceeds the model conditions increasingly on its own recent output
rather than the original prompt. We expose this through the last-layer
attention weights, measuring how much mass the current (last) query position
places on a fixed prompt slice.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def prompt_attention(attn: np.ndarray, prompt_span: Sequence[int]) -> float:
    """Mean last-layer attention from the current token to the prompt slice.

    Parameters
    ----------
    attn:
        Last-layer attention array of shape ``(H, S, S)`` (heads, keys, queries
        collapsed to the last query row). Typically
        ``outputs.attentions[-1][0].float().cpu().numpy()``.
    prompt_span:
        Indices of the prompt tokens to attend over (e.g. ``range(K)``).

    Returns
    -------
    float
        Mean attention mass from the last query position to ``prompt_span``.
        Returns ``0.0`` on malformed input rather than raising.
    """
    if attn is None or getattr(attn, "ndim", 0) != 3:
        return 0.0
    if prompt_span is None or len(prompt_span) == 0:
        return 0.0
    _, s, _ = attn.shape
    valid = [i for i in prompt_span if 0 <= i < s - 1]
    if not valid:
        return 0.0
    return float(np.nanmean(attn[:, -1, valid]))


def evidence_attention(attn: np.ndarray, evidence_span: Sequence[int]) -> float:
    """Mean last-layer attention from the current token to an evidence span.

    Used by the positional-sensitivity analysis to measure how attention to a
    fixed piece of evidence changes as its position in the context varies.
    """
    if attn is None or getattr(attn, "ndim", 0) != 3:
        return 0.0
    if evidence_span is None or len(evidence_span) == 0:
        return 0.0
    _, s, _ = attn.shape
    valid = [i for i in evidence_span if 0 <= i < s]
    if not valid:
        return 0.0
    return float(np.nanmean(attn[:, -1, valid]))


def prompt_span(prompt_len: int, slice_k: int) -> List[int]:
    """Fixed prompt slice ``[0, min(K, prompt_len))`` used for A_t."""
    return list(range(min(int(slice_k), int(prompt_len))))

