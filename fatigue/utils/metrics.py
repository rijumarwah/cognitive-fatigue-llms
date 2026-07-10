"""
Task-level evaluation helpers used alongside FI: exact-match/F1 (SQuAD-style)
and n-gram repetition ratio, the two failure proxies used in Section 7
(Table 2, Table 3) to validate FI against.
"""

import re
from typing import List, Optional, Tuple


def _norm_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.split()


def em_f1(pred: str, gold: str) -> Tuple[int, float]:
    """Exact-match (0/1) and token-overlap F1 between a prediction and gold answer."""
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


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams in `text` that are exact repeats of an earlier n-gram."""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    seen, repeats, total = set(), 0, 0
    for i in range(len(toks) - n + 1):
        ng = " ".join(toks[i : i + n])
        total += 1
        if ng in seen:
            repeats += 1
        else:
            seen.add(ng)
    return repeats / total if total > 0 else 0.0


def find_sublist(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Index of the first occurrence of `needle` inside `haystack`, or None."""
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None
