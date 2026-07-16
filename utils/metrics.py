"""Evaluation metrics used to *validate* FI (never to compute it).

EM / token-F1 follow the standard SQuAD-style normalization; repetition ratio
is the fraction of generated n-grams that are exact repeats and serves as a
behavioral degeneration proxy.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _norm_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.split()


def em_f1(pred: str, gold: str) -> Tuple[int, float]:
    """Exact-match (0/1) and token-level F1 between a prediction and a gold string."""
    p, g = _norm_tokens(pred), _norm_tokens(gold)
    em = int(p == g)
    if len(p) == 0 or len(g) == 0:
        return em, 0.0
    common = set(p) & set(g)
    match = sum(min(p.count(w), g.count(w)) for w in common)
    if match == 0:
        return em, 0.0
    prec, rec = match / len(p), match / len(g)
    return em, (2 * prec * rec / (prec + rec))


def safe_em_f1(pred: str, gold) -> Tuple[Optional[int], Optional[float]]:
    """EM/F1 that tolerates ``None`` and multi-answer (list) gold; takes the best."""
    if gold is None:
        return None, None
    if isinstance(gold, str):
        return em_f1(pred, gold) if gold.strip() else (None, None)
    if isinstance(gold, (list, tuple, set)):
        candidates = []
        for g in gold:
            if g is None:
                continue
            s = g if isinstance(g, str) else str(g)
            if s.strip():
                candidates.append(s)
        if not candidates:
            return None, None
        ems, f1s = zip(*(em_f1(pred, g) for g in candidates))
        return max(ems), max(f1s)
    s = str(gold)
    return em_f1(pred, s) if s.strip() else (None, None)


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of generated ``n``-grams that are exact repeats."""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    seen, repeats, total = set(), 0, 0
    for i in range(len(toks) - n + 1):
        ng = " ".join(toks[i:i + n])
        total += 1
        if ng in seen:
            repeats += 1
        else:
            seen.add(ng)
    return repeats / total if total > 0 else 0.0
