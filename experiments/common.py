"""Shared harness for the cognitive-fatigue experiments.

Data loading, model loading, prompt construction, per-token generation with
signal probes, and plotting. All Fatigue Index math is imported from the
``fatigue`` package -- this module never redefines it. Both the architectural
stress driver and the empirical validation driver build on this harness.
"""

import os
import sys
import re
import json
import time
import random
from typing import List, Dict, Optional, Tuple

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")  # comment out for interactive windows
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

# Make the repository root importable when a script imports this module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---- Fatigue Index package (single source of truth for all FI math) ----
from fatigue import DEFAULT_CONFIG as FI_CONFIG
from fatigue.attention import (
    prompt_attention as get_prompt_attention,
    evidence_attention as get_evidence_attention,
)
from fatigue.drift import embedding_drift as get_embedding_drift
from fatigue.entropy import token_entropy as get_entropy
from fatigue.index import fatigue_index_series as _fatigue_index_series  # noqa: F401 (re-exported)
from utils.seeds import set_seed, parse_seeds  # noqa: F401 (parse_seeds re-exported)
from utils.metrics import safe_em_f1, repetition_ratio  # noqa: F401 (safe_em_f1 re-exported)

# ---------------- Config ----------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_DATASETS = "hotpot,triviaqa,squad,natural_questions"
DATA_PATH = "data/hotpot_full.jsonl"
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
ENTROPY_BAND_LOW = FI_CONFIG.entropy_band_low
ENTROPY_BAND_HIGH = FI_CONFIG.entropy_band_high
FI_WEIGHT_ATTN = FI_CONFIG.w_attention
FI_WEIGHT_DRIFT = FI_CONFIG.w_drift
FI_WEIGHT_ENT = FI_CONFIG.w_entropy
ENTROPY_LAST_K = 30
PROMPT_SLICE_K = FI_CONFIG.prompt_slice_k
PROBE_EVERY = FI_CONFIG.probe_every
FI_SMOOTH_WINDOW = FI_CONFIG.smooth_window
FI_HYSTERESIS_HIGH = FI_CONFIG.hysteresis_high
FI_HYSTERESIS_LOW = FI_CONFIG.hysteresis_low
MAX_CONTEXT_TOKENS = 2048
USABLE_CONTEXT = MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW
CONTEXT_SHORT = int(0.10 * USABLE_CONTEXT)
CONTEXT_MEDIUM = int(0.50 * USABLE_CONTEXT)
CONTEXT_LONG = int(0.75 * USABLE_CONTEXT)

os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
set_seed(DEFAULT_SEED)

def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
        print("Warning: could not load HotpotQA from HF:", e)
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
            print("Warning: could not load TriviaQA from HF:", e)
            print("Warning: fallback 'rc' failed:", e2)
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
        print("Warning: could not load SQuAD from HF:", e)
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
        print("Warning: could not load Natural Questions from HF.")
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
        msg = f"Warning: 4-bit load failed: {e}"
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


def build_prompt_with_filler(question: str, evidence: str, filler_token: str,
                             filler_len_tokens: int, evidence_pos: str,
                             context: Optional[str] = None,
                             use_context_filler: bool = False) -> str:
    if not question or not isinstance(question, str):
        print("Warning: empty question in build_prompt_with_filler")
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
        print("Warning: unknown evidence_pos", evidence_pos)
        return f"Question: {question}\nAnswer:"


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
        print("Warning: empty prompt in run_fatigue_experiment")
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
            print("Warning: plot failed:", e)
    return result


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
            print(f"Warning: unknown dataset '{raw}'")
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
        print(f"Warning: missing builder for dataset {dataset_name}")
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
