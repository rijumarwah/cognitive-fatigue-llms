import numpy as np
import torch

from fatigue.attention import get_evidence_attention, get_prompt_attention, prompt_span
from fatigue.drift import get_embedding_drift
from fatigue.entropy import get_entropy


def test_prompt_span_respects_k_and_base_len():
    assert prompt_span(base_len=100, config=__import__("fatigue").DEFAULT_CONFIG) == list(range(64))
    assert prompt_span(base_len=10, config=__import__("fatigue").DEFAULT_CONFIG) == list(range(10))


def test_get_prompt_attention_uniform_attention():
    H, S = 4, 6
    attn = np.full((H, S, S), 1.0 / S, dtype=np.float32)
    score = get_prompt_attention(attn, span=[0, 1, 2])
    assert abs(score - (1.0 / S)) < 1e-6


def test_get_evidence_attention_empty_span_is_zero():
    attn = np.random.rand(2, 5, 5).astype(np.float32)
    assert get_evidence_attention(attn, []) == 0.0


def test_get_embedding_drift_matches_l2_norm():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    assert get_embedding_drift(a, b) == 0.0
    c = torch.tensor([4.0, 6.0, 3.0])
    # ||c - a|| = ||(3,4,0)|| = 5
    assert abs(get_embedding_drift(c, a) - 5.0) < 1e-5


def test_get_entropy_uniform_vs_peaked():
    vocab = 100
    uniform_logits = torch.zeros(1, 1, vocab)
    peaked_logits = torch.full((1, 1, vocab), -1e4)
    peaked_logits[0, 0, 0] = 1e4

    h_uniform = get_entropy(uniform_logits)
    h_peaked = get_entropy(peaked_logits)

    assert h_uniform > h_peaked
    assert abs(h_uniform - float(np.log(vocab))) < 1e-2
    assert h_peaked < 1e-3
