"""Embedding-drift signal (D_t).

Long-horizon decoding repeatedly updates a shared residual stream, letting
small perturbations accumulate. The top-layer hidden state gradually drifts
away from the representational subspace induced by the prompt, often before any
surface-level incoherence appears. We measure this as the Euclidean distance
between the current hidden state and the final prompt hidden state h_0:

    D_t = || h_t - h_0 ||_2
"""

from __future__ import annotations


def embedding_drift(current_hidden, reference_hidden) -> float:
    """L2 distance between the current and reference (prompt) hidden states.

    Parameters
    ----------
    current_hidden:
        Top-layer hidden state at the current step, ``h_t`` (1-D torch tensor).
    reference_hidden:
        Top-layer hidden state of the final prompt token, ``h_0`` (1-D tensor).

    Returns
    -------
    float
        ``|| h_t - h_0 ||_2``; ``0.0`` if either input is missing.
    """
    if current_hidden is None or reference_hidden is None:
        return 0.0
    import torch  # lazy: torch only needed when operating on real tensors
    diff = current_hidden - reference_hidden
    return float(torch.linalg.vector_norm(diff.float(), ord=2).item())

