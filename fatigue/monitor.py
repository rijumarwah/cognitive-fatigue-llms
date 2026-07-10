"""
FatigueMonitor: drives a HuggingFace decoder-only model through generation
while computing the Fatigue Index online (paper Section 6.1, "Estimation of
the Fatigue Index").

This is the single canonical implementation of the generation loop that
`experiments/rq2.py` and `experiments/helper.py` previously duplicated (with
diverging, per-run-adaptive normalization). Use this module going forward;
the experiment scripts should call into it rather than redefining the loop.

Note on cost: to obtain attentions and hidden states at every probed step,
this monitor re-runs a full forward pass over the growing sequence at each
step (`use_cache=False`). This matches what produced the paper's reported
results, but it is O(n^2) in the number of generated tokens and will be slow
for long generations or large sweeps. If you need faster generation and can
tolerate probing less precisely, consider passing `use_cache=True` variants
that only reconstruct attentions on probed steps.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .attention import get_evidence_attention, get_prompt_attention, prompt_span
from .config import DEFAULT_CONFIG, FatigueConfig
from .drift import get_embedding_drift
from .entropy import get_entropy
from .index import compute_fi_series
from .utils.metrics import find_sublist, repetition_ratio


@dataclass
class FatigueTrace:
    """Raw + derived signals for one generated sequence."""

    attention: List[float] = field(default_factory=list)
    drift: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    evidence_attention: List[float] = field(default_factory=list)
    evidence_distance: List[float] = field(default_factory=list)
    fi_series: List[float] = field(default_factory=list)
    pred: str = ""
    repetition3: float = 0.0
    latency_s: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "attention": self.attention,
            "drift": self.drift,
            "entropy": self.entropy,
            "evidence_attention": self.evidence_attention,
            "evidence_distance": self.evidence_distance,
            "fi_series": self.fi_series,
            "pred": self.pred,
            "repetition3": self.repetition3,
            "latency_s": self.latency_s,
        }


def _sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int
) -> torch.Tensor:
    last_logits = logits[:, -1, :]
    if temperature is None or temperature <= 0:
        return torch.argmax(last_logits, dim=-1, keepdim=True)
    if temperature != 1.0:
        last_logits = last_logits / temperature
    probs = torch.softmax(last_logits, dim=-1)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), probs.shape[-1])
        topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_probs)
    if top_p is None or top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)
    if top_p <= 0.0:
        return torch.argmax(probs, dim=-1, keepdim=True)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    next_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, next_idx)


class FatigueMonitor:
    """Runs generation with a HF causal LM and records the FI trajectory."""

    def __init__(self, model, tokenizer, config: FatigueConfig = DEFAULT_CONFIG):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def run(
        self,
        prompt: str,
        evidence: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> FatigueTrace:
        import time

        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.max_new_tokens

        if not prompt or not isinstance(prompt, str):
            return FatigueTrace()

        device = next(self.model.parameters()).device
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        base_ids = inputs["input_ids"][0].tolist()
        ev_ids = (
            self.tokenizer(evidence, return_tensors="pt")["input_ids"][0].tolist()
            if evidence
            else []
        )
        ev_start = find_sublist(base_ids, ev_ids) if ev_ids else None
        ev_span = list(range(ev_start, ev_start + len(ev_ids))) if ev_start is not None else []

        max_positions = getattr(self.model.config, "max_position_embeddings", cfg.max_context_tokens)
        base_len = int(inputs["input_ids"].shape[1])
        steps = max(0, min(int(max_new_tokens), int(max_positions) - base_len))
        span = prompt_span(base_len, cfg)

        with torch.no_grad():
            out0 = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        init_hidden = out0.hidden_states[-1][0, -1]
        attn0 = out0.attentions[-1][0].detach().float().cpu().numpy()

        trace = FatigueTrace()
        trace.attention.append(get_prompt_attention(attn0, span))
        trace.drift.append(get_embedding_drift(out0.hidden_states[-1][0, -1], init_hidden))
        trace.entropy.append(get_entropy(out0.logits))
        trace.evidence_attention.append(get_evidence_attention(attn0, ev_span))
        trace.evidence_distance.append(
            float((base_len - 1) - (sum(ev_span) / len(ev_span) if ev_span else (base_len - 1)))
        )

        input_ids = inputs["input_ids"]
        for step in range(1, steps + 1):
            with torch.no_grad():
                out = self.model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            logits = out.logits
            attn = out.attentions[-1][0].detach().float().cpu().numpy()
            hidden = out.hidden_states[-1][0, -1]

            if step % cfg.probe_every == 0:
                trace.attention.append(get_prompt_attention(attn, span))
                trace.drift.append(get_embedding_drift(hidden, init_hidden))
                trace.entropy.append(get_entropy(logits))
                trace.evidence_attention.append(get_evidence_attention(attn, ev_span))
                seq_len = int(input_ids.shape[1])
                trace.evidence_distance.append(
                    float((seq_len - 1) - (sum(ev_span) / len(ev_span) if ev_span else (seq_len - 1)))
                )

            next_id = _sample_next_token(
                logits, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k
            )
            input_ids = torch.cat([input_ids, next_id], dim=-1)

            if step % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        gen_ids = input_ids[0, base_len:].tolist() if input_ids.shape[1] > base_len else []
        pred = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""

        trace.pred = pred
        trace.repetition3 = repetition_ratio(pred, n=3)
        trace.latency_s = time.perf_counter() - t0
        trace.fi_series = compute_fi_series(trace.attention, trace.drift, trace.entropy, cfg)
        return trace
