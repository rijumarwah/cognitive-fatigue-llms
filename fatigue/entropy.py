"""
Output entropy signal, E_t (paper Section 3.1, Section 6.1).

Shannon entropy (in nats) of the next-token predictive distribution
P(x_{t+1} | x_<=t), computed from the model's raw logits at the last
decoding position.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def get_entropy(logits: Optional[torch.Tensor]) -> float:
    """
    logits: (batch, seq_len, vocab) raw model logits for one forward pass.
    Returns the Shannon entropy (nats) of softmax(logits[0, -1, :]).
    """
    if logits is None or logits.ndim < 3:
        return 0.0
    last_logits = logits[0, -1, :]
    if last_logits.dtype in (torch.float16, torch.bfloat16):
        last_logits = last_logits.float()
    log_probs = F.log_softmax(last_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    if not torch.isfinite(entropy):
        return 0.0
    return float(entropy.item())
