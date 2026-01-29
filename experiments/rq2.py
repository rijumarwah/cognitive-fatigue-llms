import os
import sys
import re
import json
import time
import random
from typing import List, Dict, Optional, Tuple

# Minor allocator help
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'max_split_size_mb:128')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")  # comment out if you want interactive windows
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
plt.rcParams.update({
    "figure.figsize": (7.6, 4.2),
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
})

# ---------------- Config ----------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEFAULT_SAMPLE_SIZE = 0  # 0 => full dataset
DEFAULT_DATASETS = "hotpot,triviaqa,squad,natural_questions"
DATA_PATH = "data/hotpot_full.jsonl"
OUTDIR = "rq2_outputs"
DEFAULT_HF_REVISION_ENV = os.environ.get("HF_DATASET_REVISION")
DEFAULT_HF_REVISION = None
DEFAULT_HF_TOKEN = (os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    or os.environ.get("HF_AUTH_TOKEN"))
DEFAULT_HF_FORCE_DOWNLOAD = os.environ.get("HF_FORCE_DOWNLOAD") == "1"

DEFAULT_MAX_NEW = 120
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 123
ENTROPY_BAND_LOW = 3.8
ENTROPY_BAND_HIGH = 5.0
FI_WEIGHT_ATTN = 0.40
FI_WEIGHT_DRIFT = 0.25
FI_WEIGHT_ENT = 0.35
ENTROPY_LAST_K = 30
PROMPT_SLICE_K = 64
PROBE_EVERY = 2
FI_SMOOTH_WINDOW = 5
FI_HYSTERESIS_HIGH = 0.50
FI_HYSTERESIS_LOW = 0.40
MAX_CONTEXT_TOKENS = 2048
USABLE_CONTEXT = MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW
CONTEXT_SHORT = int(0.10 * USABLE_CONTEXT)  # ~90
CONTEXT_MEDIUM = int(0.50 * USABLE_CONTEXT)  # ~452
CONTEXT_LONG = int(0.75 * USABLE_CONTEXT)  # ~678

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)

random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

# ---------------- Small helpers ----------------
def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(seeds_str: Optional[str], fallback_seed: int) -> List[int]:
    if not seeds_str:
        return [int(fallback_seed)]
    parts = re.split(r"[,\s]+", str(seeds_str).strip())
    seeds = []
    for p in parts:
        if not p:
            continue
        try:
            seeds.append(int(p))
        except ValueError:
            continue
    return seeds or [int(fallback_seed)]


def _norm_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.split()


def _filler_target_words(filler_len_tokens: int) -> int:
    if not filler_len_tokens or filler_len_tokens <= 0:
        return 0
    return max(1, int(filler_len_tokens / 1.3))


def _filler_from_context(context: Optional[str], filler_token: str, target_words: int,
                         offset_words: int = 0) -> str:
    if not context or not isinstance(context, str):
        return " ".join([filler_token] * target_words)
    words = context.split()
    if not words:
        return " ".join([filler_token] * target_words)
    start = offset_words % len(words)
    if len(words) >= target_words and (start + target_words) <= len(words):
        return " ".join(words[start:start + target_words])
    return " ".join([words[(start + i) % len(words)] for i in range(target_words)])


def em_f1(pred: str, gold: str) -> Tuple[int, float]:
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
    if gold is None:
        return None, None
    if isinstance(gold, str):
        if not gold.strip():
            return None, None
        return em_f1(pred, gold)
    if isinstance(gold, (list, tuple, set)):
        candidates = []
        for g in gold:
            if g is None:
                continue
            if isinstance(g, str):
                if g.strip():
                    candidates.append(g)
            else:
                s = str(g)
                if s.strip():
                    candidates.append(s)
        if not candidates:
            return None, None
        ems, f1s = zip(*(em_f1(pred, g) for g in candidates))
        return max(ems), max(f1s)
    s = str(gold)
    if not s.strip():
        return None, None
    return em_f1(pred, s)


def repetition_ratio(text: str, n: int = 3) -> float:
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


def find_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle:
        return None
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return None


PLOT_METRICS = {
    "attention": {"title": "Attention to Prompt", "ylabel": "Mean attention", "color": "#1f77b4"},
    "drift": {"title": "Embedding Drift", "ylabel": "L2 distance", "color": "#ff7f0e"},
    "entropy": {"title": "Entropy", "ylabel": "Nats", "color": "#2ca02c"},
    "evidence_attention": {"title": "Evidence Attention", "ylabel": "Mean attention", "color": "#d62728"},
    "evidence_distance": {"title": "Evidence Distance", "ylabel": "Token distance", "color": "#9467bd"},
}


def _save_plot(fig, path: str, show_plot: bool = False) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        if show_plot:
            plt.show()
    finally:
        plt.close(fig)


def _plot_metric_series(series: List[float], metric: str, out_path: str, show_plot: bool = False) -> None:
    if not isinstance(series, list) or not series:
        return
    meta = PLOT_METRICS.get(metric, {"title": metric, "ylabel": metric, "color": "#1f77b4"})
    x = np.arange(len(series))
    fig, ax = plt.subplots()
    ax.plot(x, series, color=meta["color"])
    ax.set_title(meta["title"])
    ax.set_xlabel("Step")
    ax.set_ylabel(meta["ylabel"])
    ax.margins(x=0.02)
    _save_plot(fig, out_path, show_plot=show_plot)


def _aggregate_series(series_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([]), np.array([])
    max_len = max(len(s) for s in series_list)
    if max_len == 0:
        return np.array([]), np.array([])
    arr = np.full((len(series_list), max_len), np.nan, dtype=np.float32)
    for i, s in enumerate(series_list):
        if not isinstance(s, list) or not s:
            continue
        arr[i, :len(s)] = np.array(s, dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def _plot_aggregate_groups(group_series: Dict[str, List[List[float]]],
                           metric: str,
                           title: str,
                           out_path: str,
                           show_plot: bool = False) -> None:
    if not group_series:
        return
    meta = PLOT_METRICS.get(metric, {"title": metric, "ylabel": metric, "color": "#1f77b4"})
    fig, ax = plt.subplots()
    for label in sorted(group_series.keys(), key=str):
        series_list = group_series[label]
        mean, std = _aggregate_series(series_list)
        if mean.size == 0:
            continue
        x = np.arange(len(mean))
        ax.plot(x, mean, label=str(label))
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(title or meta["title"])
    ax.set_xlabel("Step")
    ax.set_ylabel(meta["ylabel"])
    ax.margins(x=0.02)
    if len(group_series) > 1:
        ax.legend()
    _save_plot(fig, out_path, show_plot=show_plot)


def _plot_aggregate_groups_on_ax(ax, group_series: Dict[str, List[List[float]]],
                                 metric: str,
                                 title: str = "") -> None:
    if not group_series:
        ax.set_title(title or metric)
        ax.set_xlabel("Step")
        return
    meta = PLOT_METRICS.get(metric, {"title": metric, "ylabel": metric, "color": "#1f77b4"})
    for label in sorted(group_series.keys(), key=str):
        series_list = group_series[label]
        mean, std = _aggregate_series(series_list)
        if mean.size == 0:
            continue
        x = np.arange(len(mean))
        ax.plot(x, mean, label=str(label))
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(title or meta["title"])
    ax.set_xlabel("Step")
    ax.set_ylabel(meta["ylabel"])
    ax.margins(x=0.02)
    if len(group_series) > 1:
        ax.legend()


# --- Metrics from your snippet ---
def get_prompt_attention(attn_np: np.ndarray, prompt_span: List[int]) -> float:
    """
    attn_np: numpy (H, S, S) from the last layer.
    Score = mean attention from the last query position to a fixed prompt span.
    """
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        print("messed up: bad attention tensor for get_prompt_attention")
        return 0.0
    if not prompt_span:
        return 0.0
    _, S, _ = attn_np.shape
    valid = [i for i in prompt_span if 0 <= i < S - 1]
    if not valid:
        return 0.0
    return float(np.nanmean(attn_np[:, -1, valid]))


def get_evidence_attention(attn_np: np.ndarray, ev_span: List[int]) -> float:
    """
    Mean attention from last query to the provided evidence span indices.
    """
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        return 0.0
    if not ev_span:
        return 0.0
    _, S, _ = attn_np.shape
    valid = [i for i in ev_span if 0 <= i < S]
    if not valid:
        return 0.0
    return float(np.nanmean(attn_np[:, -1, valid]))


def get_embedding_drift(current_hidden: torch.Tensor, ref_hidden: torch.Tensor) -> float:
    if current_hidden is None or ref_hidden is None:
        print("messed up: hidden state missing for get_embedding_drift")
        return 0.0
    return float(torch.norm(current_hidden - ref_hidden, p=2).item())


def get_entropy(logits: torch.Tensor) -> float:
    if logits is None or logits.ndim < 3:
        print("messed up: logits missing for get_entropy")
        return 0.0
    last_logits = logits[0, -1, :]
    if last_logits.dtype in (torch.float16, torch.bfloat16):
        last_logits = last_logits.float()
    log_probs = F.log_softmax(last_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    if not torch.isfinite(entropy):
        print("messed up: non-finite entropy")
        return 0.0
    return float(entropy.item())


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
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
    next_id = sorted_idx.gather(-1, next_idx)
    return next_id


def _gen_series(series: List[float]) -> List[float]:
    if not isinstance(series, list) or not series:
        return []
    return series[1:] if len(series) > 1 else []


def _series_slope(series: List[float], step_size: int = 1) -> float:
    if not isinstance(series, list) or len(series) < 2:
        return 0.0
    y = np.array(series, dtype=np.float32)
    step = max(1, int(step_size))
    x = np.arange(len(y), dtype=np.float32) * float(step)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _normalize_series(series: List[float], invert: bool = False) -> List[float]:
    if not isinstance(series, list) or not series:
        return []
    arr = np.array(series, dtype=np.float32)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    rng = max_v - min_v
    if not np.isfinite(rng) or rng <= 1e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - min_v) / rng
    if invert:
        norm = 1.0 - norm
    return [float(x) for x in norm.tolist()]


def _smooth_series(series: List[float], window: int) -> List[float]:
    if not isinstance(series, list) or not series:
        return []
    w = max(1, int(window))
    if w == 1:
        return series[:]
    out = []
    for i in range(len(series)):
        start = max(0, i - w + 1)
        out.append(float(np.mean(series[start:i + 1])))
    return out


def _phi_attention(attn_series: List[float]) -> List[float]:
    if not isinstance(attn_series, list) or not attn_series:
        return []
    early_n = min(FI_SMOOTH_WINDOW, max(0, len(attn_series) - 1))
    early = attn_series[:1 + early_n]
    baseline = float(np.mean(early))
    drops = [baseline - v for v in attn_series]
    early_drops = drops[:1 + early_n]
    min_d = float(min(early_drops))
    max_d = float(max(early_drops))
    rng = max_d - min_d
    if rng <= 1e-8:
        return [0.0 for _ in drops]
    out = [(d - min_d) / rng for d in drops]
    return [min(1.0, max(0.0, float(v))) for v in out]


def _phi_entropy(entropy_series: List[float]) -> List[float]:
    if not isinstance(entropy_series, list) or not entropy_series:
        return []
    band = max(1e-6, float(ENTROPY_BAND_HIGH - ENTROPY_BAND_LOW))
    out = []
    for e in entropy_series:
        if e < ENTROPY_BAND_LOW:
            dev = ENTROPY_BAND_LOW - e
        elif e > ENTROPY_BAND_HIGH:
            dev = e - ENTROPY_BAND_HIGH
        else:
            dev = 0.0
        out.append(min(1.0, max(0.0, float(dev / band))))
    return out


def _phi_drift(drift_series: List[float]) -> List[float]:
    if not isinstance(drift_series, list) or not drift_series:
        return []
    max_d = float(max(drift_series))
    if max_d <= 1e-8:
        return [0.0 for _ in drift_series]
    return [min(1.0, max(0.0, float(d / max_d))) for d in drift_series]


def _apply_hysteresis(series: List[float],
                      high: float = FI_HYSTERESIS_HIGH,
                      low: float = FI_HYSTERESIS_LOW) -> List[int]:
    if not isinstance(series, list) or not series:
        return []
    state = 0
    out = []
    for v in series:
        if state == 0 and v >= high:
            state = 1
        elif state == 1 and v <= low:
            state = 0
        out.append(state)
    return out


def _entropy_in_band_pct(entropy_series: List[float],
                         low: float = ENTROPY_BAND_LOW,
                         high: float = ENTROPY_BAND_HIGH) -> float:
    series = _gen_series(entropy_series)
    if not series:
        return 0.0
    in_band = [v for v in series if (v >= low and v <= high)]
    return float(100.0 * len(in_band) / len(series))


def _mean_entropy_last_k(entropy_series: List[float], k: int = ENTROPY_LAST_K) -> float:
    series = _gen_series(entropy_series)
    if not series:
        return 0.0
    k_tokens = max(1, int(k))
    k_probes = max(1, int(np.ceil(k_tokens / max(1, PROBE_EVERY))))
    tail = series[-k_probes:] if len(series) >= k_probes else series
    return float(np.mean(tail))


def _fatigue_index_series(attn_series: List[float],
                          drift_series: List[float],
                          entropy_series: List[float]) -> List[float]:
    a = _gen_series(_phi_attention(attn_series))
    d = _gen_series(_phi_drift(drift_series))
    e = _gen_series(_phi_entropy(entropy_series))
    n = min(len(a), len(d), len(e))
    if n == 0:
        return []
    fi = [
        (FI_WEIGHT_ATTN * a[i]) + (FI_WEIGHT_DRIFT * d[i]) + (FI_WEIGHT_ENT * e[i])
        for i in range(n)
    ]
    return _smooth_series(fi, FI_SMOOTH_WINDOW)


def _mean(series: List[float]) -> float:
    if not isinstance(series, list) or not series:
        return 0.0
    return float(np.mean(series))


def _split_sentences(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _dedupe_preserve(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items or []:
        if not item:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _ensure_str_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, (list, tuple, set)):
        out = []
        for v in val:
            out.extend(_ensure_str_list(v))
        return out
    if isinstance(val, dict):
        out = []
        for key in ["text", "value", "answer", "answers", "aliases", "alias"]:
            if key in val:
                out.extend(_ensure_str_list(val.get(key)))
        return out
    return []


def _join_text(val, max_chars: int = 2000) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        text = val
    elif isinstance(val, (list, tuple)):
        text = " ".join([v for v in val if isinstance(v, str)])
    else:
        return ""
    text = " ".join(text.split())
    return text[:max_chars]


def _select_evidence(context: str, answers: List[str]) -> str:
    if context and isinstance(context, str):
        sents = _split_sentences(context)
        if answers:
            for ans in answers:
                if not ans:
                    continue
                ans_l = ans.lower()
                for s in sents:
                    if ans_l in s.lower():
                        return s
        if sents:
            return sents[0]
    if answers:
        return answers[0]
    return ""


def _extract_question_text(val) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for key in ["text", "question", "value"]:
            if key in val and isinstance(val.get(key), str):
                return val.get(key).strip()
        toks = val.get("tokens")
        if isinstance(toks, list):
            toks = [t for t in toks if isinstance(t, str)]
            if toks:
                return " ".join(toks)
    return ""


def _extract_from_doc_list(docs, fields: List[str],
                           max_docs: int = 3,
                           max_chars: int = 2000) -> str:
    if not docs:
        return ""
    if isinstance(docs, dict):
        docs = list(docs.values())
    if not isinstance(docs, (list, tuple)):
        return ""
    parts = []
    for doc in docs:
        if len(parts) >= max_docs:
            break
        text = ""
        if isinstance(doc, str):
            text = _join_text(doc, max_chars=max_chars)
        elif isinstance(doc, dict):
            for key in fields:
                if key in doc:
                    text = _join_text(doc.get(key), max_chars=max_chars)
                    if text:
                        break
            if not text:
                for v in doc.values():
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
        if text:
            parts.append(text)
    text = " ".join(parts)
    text = " ".join(text.split())
    return text[:max_chars]


def _extract_triviaqa_context(item: Dict, max_chars: int = 2000) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ["context", "search_context", "wiki_context"]:
        val = item.get(key)
        text = _join_text(val, max_chars=max_chars)
        if text:
            return text
    ctx = _extract_from_doc_list(item.get("entity_pages"),
                                 fields=["wiki_context", "context", "text", "content"],
                                 max_docs=3,
                                 max_chars=max_chars)
    if ctx:
        return ctx
    ctx = _extract_from_doc_list(item.get("search_results"),
                                 fields=["search_context", "context", "snippet", "text"],
                                 max_docs=3,
                                 max_chars=max_chars)
    return ctx


def _looks_like_commit_hash(rev: Optional[str]) -> bool:
    if not rev or not isinstance(rev, str):
        return False
    return re.fullmatch(r"[0-9a-f]{40}", rev.strip().lower()) is not None


def _sanitize_revision(rev: Optional[str]) -> Optional[str]:
    if not rev:
        return None
    if _looks_like_commit_hash(rev):
        print("Note: ignoring HF_DATASET_REVISION pinned hash; using default revision.")
        # Clear env pins so datasets/hub don't keep using the stale hash.
        os.environ.pop("HF_DATASET_REVISION", None)
        os.environ.pop("HF_HUB_REVISION", None)
        return None
    return rev


def _load_hf_dataset(path: str,
                     name: Optional[str] = None,
                     revision: Optional[str] = None,
                     token: Optional[str] = None,
                     force_download: bool = False):
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if force_download:
        kwargs["download_mode"] = "force_redownload"
    try:
        return load_dataset(path, name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            tok = kwargs.pop("token")
            kwargs["use_auth_token"] = tok
        return load_dataset(path, name, **kwargs)
    except Exception as e:
        msg = str(e)
        if revision and ("revision/" in msg or "revision" in msg) and "404" in msg:
            print("Note: revision failed, retrying without revision.")
            return _load_hf_dataset(path, name, revision=None, token=token, force_download=force_download)
        if not force_download:
            print(f"Note: load_dataset failed, retrying with force_redownload — {e}")
            return _load_hf_dataset(path, name, revision=revision, token=token, force_download=True)
        raise


def _is_4bit_model(model) -> bool:
    if getattr(model, "is_loaded_in_4bit", False):
        return True
    for _, module in model.named_modules():
        name = module.__class__.__name__
        if name in ("Linear4bit", "LinearFP4"):
            return True
    return False


def _select_indices(dsplit, n: int, seed: int):
    total = len(dsplit)
    if not n or int(n) <= 0 or int(n) >= total:
        return range(total)
    rng = random.Random(seed)
    return rng.sample(list(range(total)), int(n))


# ---------------- Data (HF samples -> jsonl) ----------------
def create_hotpot_sample_from_hf(out_path=DATA_PATH, n=DEFAULT_SAMPLE_SIZE, seed=42,
                                 hf_revision: Optional[str] = None, hf_token: Optional[str] = None,
                                 hf_force_download: bool = False):
    try:
        revision = hf_revision if hf_revision is not None else DEFAULT_HF_REVISION
        token = hf_token if hf_token is not None else DEFAULT_HF_TOKEN
        force = bool(hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)
        ds = _load_hf_dataset("hotpot_qa", "fullwiki", revision=revision, token=token,
                              force_download=force)
    except Exception as e:
        print("messed up: could not load HotpotQA from HF:", e)
        return out_path
    dsplit = ds["validation"] if "validation" in ds else ds[list(ds.keys())[0]]
    idxs = _select_indices(dsplit, n, seed)
    out = []
    for ii, i in enumerate(idxs):
        item = dsplit[int(i)]
        q = item.get("question") or item.get("query") or ""
        ans = item.get("answer") or ""
        ctx = ""
        if "context" in item and item["context"]:
            try:
                if isinstance(item["context"], list):
                    ctx = " ".join([" ".join(x) if isinstance(x, (list, tuple)) else str(x) for x in item["context"]])
                else:
                    ctx = str(item["context"])
            except Exception:
                ctx = str(item["context"])
        evidence = ""
        if "supporting_facts" in item and item["supporting_facts"]:
            try:
                evidence = " ".join([" ".join(x) if isinstance(x, (list, tuple)) else str(x)
                                     for x in item["supporting_facts"]])
            except Exception:
                evidence = str(item["supporting_facts"])
        if not evidence and ctx and ans:
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ctx) if s.strip()]
            found = ""
            for s in sents:
                if any(t.lower() in s.lower() for t in ans.split()):
                    found = s
                    break
            evidence = found or (sents[0] if sents else ctx[:200])
        rec = {"id": str(ii), "question": q, "context": ctx, "evidence": evidence, "answer": ans}
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created sample {out_path} with {len(out)} examples.")
    return out_path


def create_triviaqa_sample_from_hf(out_path: str, n: int = DEFAULT_SAMPLE_SIZE, seed: int = 42,
                                   config: str = "unfiltered",
                                   hf_revision: Optional[str] = None, hf_token: Optional[str] = None,
                                   hf_force_download: bool = False):
    try:
        revision = hf_revision if hf_revision is not None else DEFAULT_HF_REVISION
        token = hf_token if hf_token is not None else DEFAULT_HF_TOKEN
        force = bool(hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)
        ds = _load_hf_dataset("trivia_qa", config, revision=revision, token=token,
                              force_download=force)
    except Exception as e:
        try:
            revision = hf_revision if hf_revision is not None else DEFAULT_HF_REVISION
            token = hf_token if hf_token is not None else DEFAULT_HF_TOKEN
            force = bool(hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)
            ds = _load_hf_dataset("trivia_qa", "rc", revision=revision, token=token,
                                  force_download=force)
        except Exception as e2:
            print("messed up: could not load TriviaQA from HF:", e)
            print("messed up: fallback 'rc' failed:", e2)
            return out_path
    dsplit = ds["validation"] if "validation" in ds else ds[list(ds.keys())[0]]
    idxs = _select_indices(dsplit, n, seed)
    out = []
    for ii, i in enumerate(idxs):
        item = dsplit[int(i)]
        q = item.get("question") or item.get("query") or ""
        if not isinstance(q, str):
            q = str(q)
        ans_field = item.get("answer") or item.get("answers") or ""
        answers = _dedupe_preserve(_ensure_str_list(ans_field))
        ctx = _extract_triviaqa_context(item)
        evidence = _select_evidence(ctx, answers)
        rec = {"id": str(ii), "question": q, "context": ctx, "evidence": evidence, "answer": answers}
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created sample {out_path} with {len(out)} examples.")
    return out_path


def create_squad_sample_from_hf(out_path: str, n: int = DEFAULT_SAMPLE_SIZE, seed: int = 42,
                                hf_revision: Optional[str] = None, hf_token: Optional[str] = None,
                                hf_force_download: bool = False):
    try:
        revision = hf_revision if hf_revision is not None else DEFAULT_HF_REVISION
        token = hf_token if hf_token is not None else DEFAULT_HF_TOKEN
        force = bool(hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)
        ds = _load_hf_dataset("squad", None, revision=revision, token=token,
                              force_download=force)
    except Exception as e:
        print("messed up: could not load SQuAD from HF:", e)
        return out_path
    dsplit = ds["validation"] if "validation" in ds else ds[list(ds.keys())[0]]
    idxs = _select_indices(dsplit, n, seed)
    out = []
    for ii, i in enumerate(idxs):
        item = dsplit[int(i)]
        q = item.get("question") or ""
        if not isinstance(q, str):
            q = str(q)
        ctx = item.get("context") or ""
        if not isinstance(ctx, str):
            ctx = str(ctx)
        ans_field = item.get("answers") or {}
        if isinstance(ans_field, dict):
            answers = _dedupe_preserve(_ensure_str_list(ans_field.get("text")))
        else:
            answers = _dedupe_preserve(_ensure_str_list(ans_field))
        evidence = _select_evidence(ctx, answers)
        rec = {"id": str(ii), "question": q, "context": ctx, "evidence": evidence, "answer": answers}
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created sample {out_path} with {len(out)} examples.")
    return out_path


def create_natural_questions_sample_from_hf(out_path: str, n: int = DEFAULT_SAMPLE_SIZE, seed: int = 42,
                                            hf_revision: Optional[str] = None, hf_token: Optional[str] = None,
                                            hf_force_download: bool = False):
    ds = None
    ds_name = ""
    for name in ["natural_questions", "natural_questions_open", "nq_open"]:
        try:
            revision = hf_revision if hf_revision is not None else DEFAULT_HF_REVISION
            token = hf_token if hf_token is not None else DEFAULT_HF_TOKEN
            force = bool(hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)
            ds = _load_hf_dataset(name, None, revision=revision, token=token,
                                  force_download=force)
            ds_name = name
            break
        except Exception:
            continue
    if ds is None:
        print("messed up: could not load Natural Questions from HF.")
        return out_path
    dsplit = ds["validation"] if "validation" in ds else ds[list(ds.keys())[0]]
    idxs = _select_indices(dsplit, n, seed)
    out = []
    for ii, i in enumerate(idxs):
        item = dsplit[int(i)]
        q = _extract_question_text(item.get("question") or item.get("question_text") or item.get("query") or "")
        if not q and isinstance(item.get("question"), str):
            q = item.get("question")
        ans_field = item.get("answer") or item.get("answers") or item.get("short_answers") or ""
        answers = _dedupe_preserve(_ensure_str_list(ans_field))
        if not answers and isinstance(item.get("annotations"), list):
            for ann in item.get("annotations"):
                answers.extend(_ensure_str_list(ann.get("short_answers") or ann.get("short_answer")
                                                or ann.get("answers") or ann.get("answer")))
                yn = ann.get("yes_no_answer")
                if isinstance(yn, str) and yn.upper() in ("YES", "NO"):
                    answers.append(yn.upper())
            answers = _dedupe_preserve(answers)
        context = _join_text(item.get("context") or item.get("document_text") or item.get("document"),
                             max_chars=2000)
        if not context and ds_name == "natural_questions":
            context = _join_text(item.get("document_text") or item.get("document"), max_chars=2000)
        evidence = _select_evidence(context, answers)
        rec = {"id": str(ii), "question": q, "context": context, "evidence": evidence, "answer": answers}
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created sample {out_path} with {len(out)} examples.")
    return out_path


# ---------------- Model loader(s) ----------------
def load_4bit(model_name: str, require_4bit: bool = True):
    use_cuda = torch.cuda.is_available()
    try:
        if not use_cuda:
            raise RuntimeError("CUDA not available for 4-bit.")
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise RuntimeError("bitsandbytes not available") from e
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="eager"
        )
        if not _is_4bit_model(mdl):
            raise RuntimeError("model did not load in 4-bit")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"
            if tok.pad_token_id is None:
                tok.add_special_tokens({"pad_token": tok.pad_token})
                mdl.resize_token_embeddings(len(tok))
        mdl.eval()
        print("Loaded 4-bit model successfully.")
        return mdl, tok
    except Exception as e:
        msg = f"messed up: 4-bit load failed: {e}"
        if require_4bit:
            print(msg)
            raise
        print(msg + " — falling back to FP16/FP32")
    # fallback
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"attn_implementation": "eager"}
    if torch.cuda.is_available():
        kwargs.update({"dtype": torch.float16, "device_map": "auto"})
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": tok.pad_token})
            mdl.resize_token_embeddings(len(tok))
    mdl.eval()
    return mdl, tok


def load_fp(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"attn_implementation": "eager"}
    if torch.cuda.is_available():
        kwargs.update({"dtype": torch.float16, "device_map": "auto"})
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": tok.pad_token})
            mdl.resize_token_embeddings(len(tok))
    mdl.eval()
    return mdl, tok


# ---------------- Prompt builder ----------------
def build_prompt_with_filler(question: str, evidence: str, filler_token: str,
                             filler_len_tokens: int, evidence_pos: str,
                             context: Optional[str] = None,
                             use_context_filler: bool = False) -> str:
    if not question or not isinstance(question, str):
        print("messed up: empty question in build_prompt_with_filler")
        return ""
    evidence = evidence or ""
    if not use_context_filler:
        context = None

    if evidence_pos == "front":
        target_words = _filler_target_words(filler_len_tokens)
        filler = _filler_from_context(context, filler_token, target_words, offset_words=0)
        return f"{evidence}\n\n{filler}\n\nQuestion: {question}\nAnswer:"
    elif evidence_pos == "middle":
        left_tokens = max(0, int(filler_len_tokens) // 2)
        right_tokens = max(0, int(filler_len_tokens) - left_tokens)
        left_words = _filler_target_words(left_tokens)
        right_words = _filler_target_words(right_tokens)
        filler = _filler_from_context(context, filler_token, left_words, offset_words=0)
        filler2 = _filler_from_context(context, filler_token, right_words, offset_words=left_words)
        return f"{filler}\n\n{evidence}\n\n{filler2}\n\nQuestion: {question}\nAnswer:"
    elif evidence_pos == "end":
        target_words = _filler_target_words(filler_len_tokens)
        filler = _filler_from_context(context, filler_token, target_words, offset_words=0)
        return f"{filler}\n\nQuestion: {question}\n\n{evidence}\n\nAnswer:"
    else:
        print("messed up: unknown evidence_pos", evidence_pos)
        return f"Question: {question}\nAnswer:"


# ---------------- Core: fatigue run with evidence tracking ----------------
def run_fatigue_experiment(model, tokenizer, prompt: str, evidence: str,
                           max_new_tokens: int = DEFAULT_MAX_NEW,
                           top_p: float = DEFAULT_TOP_P,
                           top_k: int = DEFAULT_TOP_K,
                           temperature: float = DEFAULT_TEMPERATURE,
                           save_prefix: Optional[str] = None,
                           show_plot: bool = False) -> Dict:
    """
    - One initial forward (attentions + hidden states) to set references.
    - Sampling loop up to max_new_tokens, recording:
        attention-to-prompt, embedding drift, entropy,
        evidence_attention (mean attn to evidence span),
        evidence_distance (last idx - mean(evidence idx))
    - Decodes generated suffix; returns prediction, repetition, latency and series.
    """
    if not prompt or not isinstance(prompt, str):
        print("messed up: empty prompt in run_fatigue_experiment")
        return {"attention": [], "drift": [], "entropy": [], "evidence_attention": [], "evidence_distance": [],
                "pred": "", "em": None, "f1": None, "repetition3": 0.0, "latency_s": 0.0}

    device = next(model.parameters()).device
    t0 = time.perf_counter()

    # Tokenize prompt for the actual forward
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Evidence indices relative to this tokenization
    base_ids = inputs["input_ids"][0].tolist()
    ev_ids = tokenizer(evidence or "", return_tensors="pt")["input_ids"][0].tolist() if (evidence and isinstance(evidence, str)) else []
    ev_start = find_sublist(base_ids, ev_ids) if ev_ids else None
    ev_span = list(range(ev_start, ev_start + len(ev_ids))) if ev_start is not None else []

    # Respect model context window
    max_positions = getattr(model.config, "max_position_embeddings", 2048)
    base_len = int(inputs["input_ids"].shape[1])
    steps = max(0, min(int(max_new_tokens), int(max_positions) - base_len))

    prompt_span = list(range(min(PROMPT_SLICE_K, base_len)))

    # Initial forward (step 0)
    with torch.no_grad():
        out0 = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

    init_hidden = out0.hidden_states[-1][0, -1]
    logits0 = out0.logits
    # bfloat16 can't be converted to numpy directly; cast to float32 first
    attn0 = out0.attentions[-1][0].detach().float().cpu().numpy()  # (H,S,S)
    hidden0 = out0.hidden_states[-1][0, -1]

    attention_list = [get_prompt_attention(attn0, prompt_span)]
    drift_list = [get_embedding_drift(hidden0, init_hidden)]
    entropy_list = [get_entropy(logits0)]
    evid_attn_list = [get_evidence_attention(attn0, ev_span)]
    evid_dist_list = [float((base_len - 1) - (np.mean(ev_span) if ev_span else (base_len - 1)))]

    # Sampling generation loop
    input_ids = inputs["input_ids"]
    for step in range(1, steps + 1):
        with torch.no_grad():
            out = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )

        logits = out.logits
        attn = out.attentions[-1][0].detach().float().cpu().numpy()
        hidden = out.hidden_states[-1][0, -1]

        if step % PROBE_EVERY == 0:
            attention_list.append(get_prompt_attention(attn, prompt_span))
            drift_list.append(get_embedding_drift(hidden, init_hidden))
            entropy_list.append(get_entropy(logits))
            evid_attn_list.append(get_evidence_attention(attn, ev_span))

            S = int(input_ids.shape[1])  # current length
            evid_dist_list.append(float((S - 1) - (np.mean(ev_span) if ev_span else (S - 1))))

        # Sample next token (top-p by default)
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

        if step % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Decode prediction (generated suffix)
    gen_ids = input_ids[0, base_len:].tolist() if input_ids.shape[1] > base_len else []
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""
    if not pred and gen_ids:
        pred = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    latency_s = float(time.perf_counter() - t0)
    rep3 = repetition_ratio(pred, n=3)

    result = {
        "attention": attention_list,
        "drift": drift_list,
        "entropy": entropy_list,
        "evidence_attention": evid_attn_list,
        "evidence_distance": evid_dist_list,
        "pred": pred,
        "repetition3": rep3,
        "latency_s": latency_s
    }
    if save_prefix:
        try:
            for metric, series in [
                ("attention", attention_list),
                ("drift", drift_list),
                ("entropy", entropy_list),
                ("evidence_attention", evid_attn_list),
                ("evidence_distance", evid_dist_list),
            ]:
                out_path = f"{save_prefix}_{metric}.png"
                _plot_metric_series(series, metric, out_path, show_plot=show_plot)
        except Exception as e:
            print("messed up: plot failed:", e)
    return result


# ---------------- Experiments ----------------
def context_length_stress(model, tokenizer, ex: Dict, lengths_tokens: List[int],
                          max_new_tokens: int, top_p: float, top_k: int,
                          temperature: float) -> List[Dict]:
    if not ex or not ex.get("question"):
        print("messed up: bad example in context_length_stress")
        return []
    results = []
    for L in lengths_tokens:
        print(f"[ContextLen] target={L} tokens")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(L), "middle",
                                          context=ex.get("context", ""), use_context_filler=False)
        if not prompt:
            print("messed up: prompt build failed")
            continue
        # run fatigue
        save_prefix = os.path.join(OUTDIR, f"context_len{int(L)}_id{ex.get('id', 'x')}")
        fat = run_fatigue_experiment(model, tokenizer, prompt, ex.get("evidence", ""),
                                     max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                                     temperature=temperature,
                                     save_prefix=save_prefix)
        em_val, f1_val = safe_em_f1(fat["pred"], ex.get("answer"))
        fi_series = _fatigue_index_series(fat["attention"], fat["drift"], fat["entropy"])
        rec = {
            "length_tokens": int(L),
            "fatigue": {k: fat[k] for k in ["attention", "drift", "entropy"]},
            "evidence_attention": fat["evidence_attention"],
            "evidence_distance": fat["evidence_distance"],
            "pred": fat["pred"],
            "latency_s": fat["latency_s"],
            "repetition3": fat["repetition3"],
            "attention_slope": _series_slope(_gen_series(fat["attention"]), step_size=PROBE_EVERY),
            "drift_slope": _series_slope(_gen_series(fat["drift"]), step_size=PROBE_EVERY),
            "entropy_in_band_pct": _entropy_in_band_pct(fat["entropy"]),
            "mean_fi": _mean(fi_series),
            "em": em_val,
            "f1": f1_val,
            "gold": ex.get("answer", "")
        }
        results.append(rec)
        cleanup()
    return results


def positional_sensitivity_profile(model, tokenizer, ex: Dict, filler_len_tokens: int,
                                   max_new_tokens: int, top_p: float, top_k: int,
                                   temperature: float) -> List[Dict]:
    if not ex or not ex.get("question"):
        print("messed up: bad example in positional_sensitivity_profile")
        return []
    results = []
    for pos in ["front", "middle", "end"]:
        print(f"[Position] {pos}")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(filler_len_tokens), pos,
                                          context=ex.get("context", ""), use_context_filler=False)
        if not prompt:
            print("messed up: prompt build failed")
            continue
        save_prefix = os.path.join(OUTDIR, f"position_{pos}_id{ex.get('id', 'x')}")
        fat = run_fatigue_experiment(model, tokenizer, prompt, ex.get("evidence", ""),
                                     max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                                     temperature=temperature,
                                     save_prefix=save_prefix)
        em_val, f1_val = safe_em_f1(fat["pred"], ex.get("answer"))
        fi_series = _fatigue_index_series(fat["attention"], fat["drift"], fat["entropy"])
        rec = {
            "position": pos,
            "fatigue": {k: fat[k] for k in ["attention", "drift", "entropy"]},
            "evidence_attention": fat["evidence_attention"],
            "evidence_distance": fat["evidence_distance"],
            "attention_to_evidence_mean": _mean(_gen_series(fat["evidence_attention"])),
            "pred": fat["pred"],
            "latency_s": fat["latency_s"],
            "repetition3": fat["repetition3"],
            "entropy_in_band_pct": _entropy_in_band_pct(fat["entropy"]),
            "mean_fi": _mean(fi_series),
            "em": em_val,
            "f1": f1_val,
            "gold": ex.get("answer", "")
        }
        results.append(rec)
        cleanup()
    return results


def precision_quantization_ablation(ex_list: List[Dict], max_new_tokens: int,
                                    top_p: float, top_k: int, temperature: float,
                                    seed: int, model_name: str = MODEL_NAME,
                                    require_4bit: bool = True) -> Dict:
    """
    FP16/FP32 vs 4-bit NF4 comparison.
    Top-p decoding; same prompts and seeds per precision; measures entropy collapse,
    repetition ratio, EM/F1, latency (plus signal curves).
    """
    out = {"fp": [], "4bit": []}
    # Build prompts once for both precisions (middle evidence, fixed filler)
    prompts = []
    for ex in ex_list:
        prompt = build_prompt_with_filler(ex.get("question", ""), ex.get("evidence", ""), "filler", 200, "middle",
                                          context=ex.get("context", ""), use_context_filler=False)
        prompts.append((ex, prompt))

    seeds = [int(seed) + i for i in range(len(prompts))]

    # FP model
    print("[Precision] Loading FP model...")
    fp_model, fp_tok = load_fp(model_name)
    for i, (ex, prompt) in enumerate(prompts):
        set_seed(seeds[i])
        print("[Precision FP] id=", ex.get("id"))
        fat = run_fatigue_experiment(fp_model, fp_tok, prompt, ex.get("evidence", ""),
                                     max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                                     temperature=temperature,
                                     save_prefix=os.path.join(OUTDIR, f"precision_fp_id{ex.get('id', 'x')}"))
        em_val, f1_val = safe_em_f1(fat["pred"], ex.get("answer"))
        fi_series = _fatigue_index_series(fat["attention"], fat["drift"], fat["entropy"])
        out["fp"].append({
            "id": ex.get("id"), "pred": fat["pred"], "gold": ex.get("answer", ""),
            "attention": fat["attention"], "drift": fat["drift"], "entropy": fat["entropy"],
            "repetition3": fat["repetition3"], "repetition_ratio": fat["repetition3"],
            "entropy_in_band_pct": _entropy_in_band_pct(fat["entropy"]),
            "mean_entropy_lastk": _mean_entropy_last_k(fat["entropy"]),
            "mean_fi": _mean(fi_series),
            "latency_s": fat["latency_s"],
            "em": em_val, "f1": f1_val
        })
        cleanup()
    del fp_model
    cleanup()

    # 4-bit model
    print("[Precision] Loading 4-bit model...")
    q_model, q_tok = load_4bit(model_name, require_4bit=require_4bit)
    for i, (ex, prompt) in enumerate(prompts):
        set_seed(seeds[i])
        print("[Precision 4bit] id=", ex.get("id"))
        fat = run_fatigue_experiment(q_model, q_tok, prompt, ex.get("evidence", ""),
                                     max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                                     temperature=temperature,
                                     save_prefix=os.path.join(OUTDIR, f"precision_4bit_id{ex.get('id', 'x')}"))
        em_val, f1_val = safe_em_f1(fat["pred"], ex.get("answer"))
        fi_series = _fatigue_index_series(fat["attention"], fat["drift"], fat["entropy"])
        out["4bit"].append({
            "id": ex.get("id"), "pred": fat["pred"], "gold": ex.get("answer", ""),
            "attention": fat["attention"], "drift": fat["drift"], "entropy": fat["entropy"],
            "repetition3": fat["repetition3"], "repetition_ratio": fat["repetition3"],
            "entropy_in_band_pct": _entropy_in_band_pct(fat["entropy"]),
            "mean_entropy_lastk": _mean_entropy_last_k(fat["entropy"]),
            "mean_fi": _mean(fi_series),
            "latency_s": fat["latency_s"],
            "em": em_val, "f1": f1_val
        })
        cleanup()
    del q_model
    cleanup()

    return out


def aggregate_context_length_plots(out_ctx: List[Dict], outdir: str) -> None:
    grouped: Dict[str, Dict[str, List[List[float]]]] = {}
    for ex in out_ctx:
        for res in ex.get("results", []) or []:
            length = res.get("length_tokens")
            label = f"len={length}"
            bucket = grouped.setdefault(label, {k: [] for k in PLOT_METRICS.keys()})
            fatigue = res.get("fatigue", {}) or {}
            for metric in ["attention", "drift", "entropy"]:
                series = fatigue.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
            for metric in ["evidence_attention", "evidence_distance"]:
                series = res.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
    for metric in PLOT_METRICS.keys():
        group_series = {label: data[metric] for label, data in grouped.items() if data.get(metric)}
        if not group_series:
            continue
        out_path = os.path.join(outdir, f"aggregate_context_length_{metric}.png")
        title = f"Context Length Aggregate - {PLOT_METRICS[metric]['title']}"
        _plot_aggregate_groups(group_series, metric, title, out_path)


def aggregate_context_length_summary(out_ctx: List[Dict], outdir: str) -> None:
    grouped: Dict[str, Dict[str, List[List[float]]]] = {}
    for ex in out_ctx:
        for res in ex.get("results", []) or []:
            length = res.get("length_tokens")
            label = f"len={length}"
            bucket = grouped.setdefault(label, {k: [] for k in PLOT_METRICS.keys()})
            fatigue = res.get("fatigue", {}) or {}
            for metric in ["attention", "drift", "entropy"]:
                series = fatigue.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
    metrics = ["attention", "drift", "entropy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 3.6))
    for ax, metric in zip(axes, metrics):
        group_series = {label: data[metric] for label, data in grouped.items() if data.get(metric)}
        _plot_aggregate_groups_on_ax(ax, group_series, metric, title=PLOT_METRICS[metric]["title"])
    _save_plot(fig, os.path.join(outdir, "aggregate_context_length_summary.png"))


def aggregate_positional_plots(out_pos: List[Dict], outdir: str) -> None:
    grouped: Dict[str, Dict[str, List[List[float]]]] = {}
    for ex in out_pos:
        for res in ex.get("results", []) or []:
            pos = res.get("position")
            label = f"pos={pos}"
            bucket = grouped.setdefault(label, {k: [] for k in PLOT_METRICS.keys()})
            fatigue = res.get("fatigue", {}) or {}
            for metric in ["attention", "drift", "entropy"]:
                series = fatigue.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
            for metric in ["evidence_attention", "evidence_distance"]:
                series = res.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
    for metric in PLOT_METRICS.keys():
        group_series = {label: data[metric] for label, data in grouped.items() if data.get(metric)}
        if not group_series:
            continue
        out_path = os.path.join(outdir, f"aggregate_positional_{metric}.png")
        title = f"Positional Aggregate - {PLOT_METRICS[metric]['title']}"
        _plot_aggregate_groups(group_series, metric, title, out_path)


def aggregate_positional_summary(out_pos: List[Dict], outdir: str) -> None:
    grouped: Dict[str, Dict[str, List[List[float]]]] = {}
    for ex in out_pos:
        for res in ex.get("results", []) or []:
            pos = res.get("position")
            label = f"pos={pos}"
            bucket = grouped.setdefault(label, {k: [] for k in PLOT_METRICS.keys()})
            for metric in ["evidence_attention", "evidence_distance"]:
                series = res.get(metric)
                if isinstance(series, list) and series:
                    bucket[metric].append(series)
    metrics = ["evidence_attention", "evidence_distance"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.2, 3.6))
    for ax, metric in zip(axes, metrics):
        group_series = {label: data[metric] for label, data in grouped.items() if data.get(metric)}
        _plot_aggregate_groups_on_ax(ax, group_series, metric, title=PLOT_METRICS[metric]["title"])
    _save_plot(fig, os.path.join(outdir, "aggregate_positional_summary.png"))


def aggregate_precision_plots(out_prec: Dict, outdir: str) -> None:
    for metric in ["attention", "drift", "entropy"]:
        grouped: Dict[str, List[List[float]]] = {}
        for label in ["fp", "4bit"]:
            runs = out_prec.get(label, []) or []
            series_list = []
            for rec in runs:
                series = rec.get(metric)
                if isinstance(series, list) and series:
                    series_list.append(series)
            if series_list:
                grouped[label] = series_list
        if not grouped:
            continue
        out_path = os.path.join(outdir, f"aggregate_precision_{metric}.png")
        title = f"Precision Aggregate - {PLOT_METRICS[metric]['title']}"
        _plot_aggregate_groups(grouped, metric, title, out_path)


def aggregate_precision_summary(out_prec: Dict, outdir: str) -> None:
    metrics = ["entropy", "attention", "drift"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 3.6))
    for ax, metric in zip(axes, metrics):
        grouped: Dict[str, List[List[float]]] = {}
        for label in ["fp", "4bit"]:
            runs = out_prec.get(label, []) or []
            series_list = []
            for rec in runs:
                series = rec.get(metric)
                if isinstance(series, list) and series:
                    series_list.append(series)
            if series_list:
                grouped[label] = series_list
        _plot_aggregate_groups_on_ax(ax, grouped, metric, title=PLOT_METRICS[metric]["title"])
    _save_plot(fig, os.path.join(outdir, "aggregate_precision_summary.png"))


# ---------------- Dataset registry ----------------
DATASET_BUILDERS = {
    "hotpot": create_hotpot_sample_from_hf,
    "triviaqa": create_triviaqa_sample_from_hf,
    "squad": create_squad_sample_from_hf,
    "natural_questions": create_natural_questions_sample_from_hf,
}

DATASET_ALIASES = {
    "hotpot": "hotpot",
    "hotpotqa": "hotpot",
    "triviaqa": "triviaqa",
    "triviaq": "triviaqa",
    "trivia_qa": "triviaqa",
    "squad": "squad",
    "squad11": "squad",
    "naturalquestions": "natural_questions",
    "naturalquestion": "natural_questions",
    "natural_questions": "natural_questions",
    "naturalquestionsopen": "natural_questions",
    "natural_questions_open": "natural_questions",
    "nqopen": "natural_questions",
    "naturalpositions": "natural_questions",
    "natural_positions": "natural_questions",
}


def _normalize_dataset_name(name: str) -> str:
    if not name:
        return ""
    key = re.sub(r"[^a-z0-9_]+", "", str(name).strip().lower())
    return DATASET_ALIASES.get(key, "")


def parse_datasets(datasets_str: Optional[str]) -> List[str]:
    if not datasets_str:
        return []
    parts = re.split(r"[,\s]+", str(datasets_str).strip())
    out = []
    for raw in parts:
        if not raw:
            continue
        raw_clean = str(raw).strip()
        if raw_clean in ("*", "all"):
            for name in DATASET_BUILDERS.keys():
                if name not in out:
                    out.append(name)
            continue
        key = re.sub(r"[^a-z0-9_]+", "", raw_clean.lower())
        if key == "all":
            for name in DATASET_BUILDERS.keys():
                if name not in out:
                    out.append(name)
            continue
        canon = _normalize_dataset_name(raw)
        if not canon:
            print(f"messed up: unknown dataset '{raw}'")
            continue
        if canon not in out:
            out.append(canon)
    return out


def dataset_sample_path(dataset_name: str, sample_size: int, data_override: Optional[str] = None) -> str:
    if dataset_name == "hotpot" and data_override:
        return data_override
    safe = re.sub(r"[^a-z0-9_]+", "_", dataset_name.lower())
    if not sample_size or int(sample_size) <= 0:
        return os.path.join("data", f"{safe}_full.jsonl")
    return os.path.join("data", f"{safe}_sample_{int(sample_size)}.jsonl")


def ensure_dataset_file(dataset_name: str, out_path: str, sample_size: int, seed: int,
                        hf_revision: Optional[str] = None, hf_token: Optional[str] = None,
                        hf_force_download: bool = False) -> str:
    builder = DATASET_BUILDERS.get(dataset_name)
    if not builder:
        print(f"messed up: missing builder for dataset {dataset_name}")
        return out_path
    if not os.path.exists(out_path):
        print(f"{out_path} not found — creating {dataset_name} sample...")
        builder(out_path=out_path, n=int(sample_size), seed=seed,
                hf_revision=hf_revision, hf_token=hf_token,
                hf_force_download=hf_force_download)
    return out_path


def load_examples_jsonl(path: str) -> List[Dict]:
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return examples


# ---------------- CLI ----------------
def main(argv=None):
    global OUTDIR
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--datasets", default=DEFAULT_DATASETS,
                   help="Comma/space-separated datasets or 'all' (hotpot,triviaqa,squad,natural_questions)")
    p.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE,
                   help="Sample size to create when dataset jsonl is missing (<=0 for full)")
    p.add_argument("--hf_revision", default=DEFAULT_HF_REVISION,
                   help="HF dataset revision/branch/commit (default: none; uses env HF_DATASET_REVISION unless it's a pinned hash)")
    p.add_argument("--hf_token", default=DEFAULT_HF_TOKEN,
                   help="HF token for gated datasets (default: env HF_TOKEN)")
    p.add_argument("--hf_force_download", action="store_true",
                   help="Force redownload of HF dataset snapshots")
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit for context/positional runs (precision ablation loads both anyway)")
    p.add_argument("--allow_4bit_fallback", action="store_true",
                   help="Allow FP fallback if 4-bit loading fails (default: fail fast)")
    p.add_argument("--subset_size", type=int, default=0,
                   help="Limit examples per dataset (<=0 for full)")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling p; <=0 for greedy")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k sampling (0 disables)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature; <=0 for greedy")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--seeds", default="123,2027", help="Comma/space-separated seeds; overrides --seed")
    p.add_argument("--run_context", action="store_true")
    p.add_argument("--run_positional", action="store_true")
    p.add_argument("--run_precision", action="store_true")
    if argv is None:
        argv = sys.argv[1:]
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print("Note: ignoring unknown args:", unknown)
    if not (args.run_context or args.run_positional or args.run_precision):
        args.run_context = True
        args.run_positional = True
        args.run_precision = True

    seeds = parse_seeds(args.seeds, args.seed)
    set_seed(seeds[0])
    os.makedirs(args.outdir, exist_ok=True)

    if args.hf_revision:
        hf_revision = args.hf_revision
    else:
        hf_revision = _sanitize_revision(DEFAULT_HF_REVISION_ENV)
    hf_token = args.hf_token if args.hf_token else None
    hf_force_download = bool(args.hf_force_download or DEFAULT_HF_FORCE_DOWNLOAD)

    datasets = parse_datasets(args.datasets)
    if not datasets:
        print("messed up: no valid datasets specified; aborting.")
        return

    # Load model for context/positional runs
    if args.run_context or args.run_positional:
        print(f"Loading {args.model} ({'4-bit' if args.use_4bit else 'FP'}) for context/positional...")
        if args.use_4bit:
            model, tokenizer = load_4bit(args.model, require_4bit=not args.allow_4bit_fallback)
        else:
            model, tokenizer = load_fp(args.model)
        print("Model on device:", next(model.parameters()).device)
    else:
        model = tokenizer = None

    for dataset_name in datasets:
        print(f"\n=== Dataset {dataset_name} ===")
        data_path = dataset_sample_path(dataset_name, args.sample_size, data_override=args.data)
        ensure_dataset_file(dataset_name, data_path, args.sample_size, seed=42,
                            hf_revision=hf_revision, hf_token=hf_token,
                            hf_force_download=hf_force_download)
        examples = load_examples_jsonl(data_path)
        if not examples:
            print(f"messed up: no examples found for {dataset_name}; skipping.")
            continue
        if not args.subset_size or int(args.subset_size) <= 0:
            subset = examples
        else:
            subset = examples[:max(1, int(args.subset_size))]

        dataset_outdir = os.path.join(args.outdir, dataset_name)
        os.makedirs(dataset_outdir, exist_ok=True)

        all_ctx = []
        all_pos = []
        all_prec = {"fp": [], "4bit": []}

        multi_seed = len(seeds) > 1
        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            set_seed(seed)
            seed_outdir = os.path.join(dataset_outdir, f"seed_{seed}") if multi_seed else dataset_outdir
            os.makedirs(seed_outdir, exist_ok=True)
            OUTDIR = seed_outdir

            # 1) Context-length stress
            if args.run_context:
                print("Running Context-Length Stress...")
                context_dir = os.path.join(seed_outdir, "context_length")
                os.makedirs(context_dir, exist_ok=True)
                OUTDIR = context_dir
                out_ctx = []
                for ex in subset:
                    res = context_length_stress(model, tokenizer, ex,
                                                lengths_tokens=[CONTEXT_SHORT, CONTEXT_MEDIUM, CONTEXT_LONG],
                                                max_new_tokens=args.max_new_tokens,
                                                top_p=args.top_p, top_k=args.top_k,
                                                temperature=args.temperature)
                    out_ctx.append({"id": ex.get("id"), "results": res})
                path = os.path.join(context_dir, "ablation_context_length.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out_ctx, f, indent=2)
                print("Saved", path)
                try:
                    aggregate_context_length_plots(out_ctx, context_dir)
                    aggregate_context_length_summary(out_ctx, context_dir)
                except Exception as e:
                    print("messed up: context aggregate plots failed:", e)
                all_ctx.extend(out_ctx)

            # 2) Positional sensitivity
            if args.run_positional:
                print("Running Positional Sensitivity...")
                positional_dir = os.path.join(seed_outdir, "positional")
                os.makedirs(positional_dir, exist_ok=True)
                OUTDIR = positional_dir
                out_pos = []
                for ex in subset:
                    res = positional_sensitivity_profile(model, tokenizer, ex,
                                                         filler_len_tokens=min(200, USABLE_CONTEXT // 2),
                                                         max_new_tokens=args.max_new_tokens,
                                                         top_p=args.top_p, top_k=args.top_k,
                                                         temperature=args.temperature)
                    out_pos.append({"id": ex.get("id"), "results": res})
                path = os.path.join(positional_dir, "ablation_positional.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out_pos, f, indent=2)
                print("Saved", path)
                try:
                    aggregate_positional_plots(out_pos, positional_dir)
                    aggregate_positional_summary(out_pos, positional_dir)
                except Exception as e:
                    print("messed up: positional aggregate plots failed:", e)
                all_pos.extend(out_pos)

            # 3) Precision/quantization ablation
            if args.run_precision:
                print("Running Precision/Quantization Ablation...")
                precision_dir = os.path.join(seed_outdir, "precision")
                os.makedirs(precision_dir, exist_ok=True)
                OUTDIR = precision_dir
                out_prec = precision_quantization_ablation(subset,
                                                           max_new_tokens=args.max_new_tokens,
                                                           top_p=args.top_p, top_k=args.top_k,
                                                           temperature=args.temperature,
                                                           seed=seed,
                                                           model_name=args.model,
                                                           require_4bit=not args.allow_4bit_fallback)
                path = os.path.join(precision_dir, "ablation_precision.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out_prec, f, indent=2)
                print("Saved", path)
                try:
                    aggregate_precision_plots(out_prec, precision_dir)
                    aggregate_precision_summary(out_prec, precision_dir)
                except Exception as e:
                    print("messed up: precision aggregate plots failed:", e)
                all_prec["fp"].extend(out_prec.get("fp", []) or [])
                all_prec["4bit"].extend(out_prec.get("4bit", []) or [])

        summary_dir = os.path.join(dataset_outdir, "summary") if multi_seed else dataset_outdir
        if multi_seed:
            os.makedirs(summary_dir, exist_ok=True)
        if args.run_context and all_ctx:
            aggregate_context_length_summary(all_ctx, summary_dir)
        if args.run_positional and all_pos:
            aggregate_positional_summary(all_pos, summary_dir)
        if args.run_precision and (all_prec["fp"] or all_prec["4bit"]):
            aggregate_precision_summary(all_prec, summary_dir)

    print("All done. Outputs are in", args.outdir)


if __name__ == "__main__":
    main()
