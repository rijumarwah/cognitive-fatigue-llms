"""Entropy signal (E_t).

Next-token Shannon entropy is a direct, inference-time view of the model's
predictive calibration. Entropy inside a healthy band indicates stable
generation; persistently low entropy signals overconfident, repetitive
degeneration, while abnormally high entropy signals indecision.
"""

from __future__ import annotations


def token_entropy(logits) -> float:
    """Shannon entropy (in nats) of the next-token distribution.

    Parameters
    ----------
    logits:
        Model logits of shape ``(B, S, V)``. The distribution of the final
        position of the first batch element is used.

    Returns
    -------
    float
        Entropy in nats; ``0.0`` on malformed or non-finite input.
    """
    if logits is None or getattr(logits, "ndim", 0) < 3:
        return 0.0
    import torch  # lazy: torch only needed when operating on real tensors
    import torch.nn.functional as F
    last = logits[0, -1, :]
    if last.dtype in (torch.float16, torch.bfloat16):
        last = last.float()
    log_probs = F.log_softmax(last, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    if not torch.isfinite(entropy):
        return 0.0
    return float(entropy.item())
