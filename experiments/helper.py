#!/usr/bin/env python3
"""
DEPRECATED: superseded by experiments/rq2.py.

This was an earlier draft of the RQ2 ablation runner. It only records the
raw signals (attention, drift, entropy) and never aggregates them into the
Fatigue Index -- rq2.py is the complete, current implementation and is what
should be used going forward. Kept for reference only; do not add new
features here. See fatigue/ for the canonical FI implementation and
CODE_AUDIT_NOTES.md for background on this consolidation.

RQ2 Ablations (simple, faithful to the fatigue snippet):

Implements three experiments:
1) Context-length stress:
   - Same task/evidence, different neutral filler lengths (short/medium/long).
   - Measure Attention/Drift/Entropy per step, plus EM/F1, repetition ratio, latency.

2) Positional sensitivity:
   - Move evidence to front/middle/end with identical content otherwise.
   - Measure EM/F1 vs position, attention-to-evidence vs distance, and the three signals.

3) Precision/quantization ablation:
   - Compare FP16 vs 4-bit NF4, same prompts/decoding (greedy).
   - Measure entropy collapse, repetition ratio, EM/F1, latency.

Inputs:
- HotpotQA sample (auto-created if missing)
- Falcon-7B-Instruct (4-bit if --use_4bit; FP16/FP32 otherwise; precision ablation loads both)

Outputs:
- JSON files in rq2_outputs/
- Per-run plots of Attention/Drift/Entropy curves (PNG) with save_prefix

Keep it simple:
- Minimal error handling; prints "messed up: ..." for missing inputs and continues.
"""

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

# ---------------- Config ----------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"
DATA_PATH  = "data/hotpot_sample_50.jsonl"
OUTDIR     = "rq2_outputs"

DEFAULT_MAX_NEW = 100
MAX_CONTEXT_TOKENS = 1024
USABLE_CONTEXT = MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW
CONTEXT_SHORT  = int(0.10 * USABLE_CONTEXT)  # ~90
CONTEXT_MEDIUM = int(0.50 * USABLE_CONTEXT)  # ~452
CONTEXT_LONG   = int(0.75 * USABLE_CONTEXT)  # ~678

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

# ---------------- Small helpers ----------------
def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _norm_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.split()

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

def repetition_ratio(text: str, n: int = 3) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    seen, repeats, total = set(), 0, 0
    for i in range(len(toks) - n + 1):
        ng = " ".join(toks[i:i+n])
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
        if hay[i:i+len(needle)] == needle:
            return i
    return None

# --- Metrics from your snippet ---
def get_attention_score(attn_np: np.ndarray) -> float:
    """
    attn_np: numpy (H, S, S) from the last layer.
    Score = mean attention from the last query position to previous tokens.
    """
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        print("messed up: bad attention tensor for get_attention_score")
        return 0.0
    H, S, _ = attn_np.shape
    if S <= 1:
        return 0.0
    return float(np.nanmean(attn_np[:, -1, :S-1]))

def get_evidence_attention(attn_np: np.ndarray, ev_span: List[int]) -> float:
    """
    Mean attention from last query to the provided evidence span indices.
    """
    if attn_np is None or getattr(attn_np, "ndim", 0) != 3:
        return 0.0
    if not ev_span:
        return 0.0
    H, S, _ = attn_np.shape
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
    dist = torch.distributions.Categorical(logits=last_logits)
    return float(dist.entropy().item())

# ---------------- Data (Hotpot sample like your other script) ----------------
def create_hotpot_sample_from_hf(out_path=DATA_PATH, n=50, seed=42):
    try:
        ds = load_dataset("hotpot_qa", "fullwiki")
    except Exception as e:
        print("messed up: could not load HotpotQA from HF:", e)
        return out_path
    dsplit = ds["validation"] if "validation" in ds else ds[list(ds.keys())[0]]
    rng = random.Random(seed)
    n = min(n, len(dsplit))
    idxs = rng.sample(list(range(len(dsplit))), n)
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
                    found = s; break
            evidence = found or (sents[0] if sents else ctx[:200])
        rec = {"id": str(ii), "question": q, "context": ctx, "evidence": evidence, "answer": ans}
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created sample {out_path} with {len(out)} examples.")
    return out_path

# ---------------- Model loader(s) ----------------
def load_4bit(model_name: str):
    use_cuda = torch.cuda.is_available()
    try:
        if not use_cuda:
            raise RuntimeError("CUDA not available for 4-bit.")
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
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"
            if tok.pad_token_id is None:
                tok.add_special_tokens({"pad_token": tok.pad_token})
                mdl.resize_token_embeddings(len(tok))
        mdl.eval()
        return mdl, tok
    except Exception as e:
        print("messed up: 4-bit load failed, falling back to FP16/FP32 —", e)
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
                             filler_len_tokens: int, evidence_pos: str) -> str:
    if not question or not isinstance(question, str):
        print("messed up: empty question in build_prompt_with_filler")
        return ""
    evidence = evidence or ""
    filler_words = max(1, int((filler_len_tokens or 0) / 1.3))
    filler = " ".join([filler_token] * filler_words)

    if evidence_pos == "front":
        return f"{evidence}\n\n{filler}\n\nQuestion: {question}\nAnswer:"
    elif evidence_pos == "middle":
        return f"{filler}\n\n{evidence}\n\n{filler}\n\nQuestion: {question}\nAnswer:"
    elif evidence_pos == "end":
        return f"{filler}\n\nQuestion: {question}\n\n{evidence}\n\nAnswer:"
    else:
        print("messed up: unknown evidence_pos", evidence_pos)
        return f"Question: {question}\nAnswer:"

# ---------------- Core: fatigue run with evidence tracking ----------------
def run_fatigue_experiment(model, tokenizer, prompt: str, evidence: str,
                           max_new_tokens: int = DEFAULT_MAX_NEW,
                           save_prefix: Optional[str] = None,
                           show_plot: bool = False) -> Dict:
    """
    - One initial forward (attentions + hidden states) to set references.
    - Greedy loop up to max_new_tokens, recording:
        attention_score, embedding_drift, entropy,
        evidence_attention (mean attn to evidence span),
        evidence_distance (last idx - mean(evidence idx))
    - Decodes generated suffix; returns EM/F1/repetition/latency and series.
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

    # Initial forward (step 0)
    with torch.no_grad():
        out0 = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    init_hidden = out0.hidden_states[-1][0, -1]
    logits0     = out0.logits
    attn0       = out0.attentions[-1][0].detach().cpu().numpy()  # (H,S,S)
    hidden0     = out0.hidden_states[-1][0, -1]

    attention_list = [get_attention_score(attn0)]
    drift_list     = [get_embedding_drift(hidden0, init_hidden)]
    entropy_list   = [get_entropy(logits0)]
    evid_attn_list = [get_evidence_attention(attn0, ev_span)]
    evid_dist_list = [float((base_len - 1) - (np.mean(ev_span) if ev_span else (base_len - 1)))]

    # Greedy generation loop
    input_ids = inputs["input_ids"]
    for step in range(1, steps + 1):
        with torch.no_grad():
            out = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        logits = out.logits
        attn   = out.attentions[-1][0].detach().cpu().numpy()
        hidden = out.hidden_states[-1][0, -1]

        attention_list.append(get_attention_score(attn))
        drift_list.append(get_embedding_drift(hidden, init_hidden))
        entropy_list.append(get_entropy(logits))
        evid_attn_list.append(get_evidence_attention(attn, ev_span))

        S = int(input_ids.shape[1])  # current length
        evid_dist_list.append(float((S - 1) - (np.mean(ev_span) if ev_span else (S - 1))))

        # Greedy next token
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B,1)
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
    return result

# ---------------- Experiments ----------------
def context_length_stress(model, tokenizer, ex: Dict, lengths_tokens: List[int], max_new_tokens: int) -> List[Dict]:
    if not ex or not ex.get("question"):
        print("messed up: bad example in context_length_stress")
        return []
    results = []
    for L in lengths_tokens:
        print(f"[ContextLen] target={L} tokens")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(L * 0.7), "middle")
        if not prompt:
            print("messed up: prompt build failed"); continue
        # run fatigue
        save_prefix = os.path.join(OUTDIR, f"context_len{int(L)}_id{ex.get('id','x')}")
        fat = run_fatigue_experiment(model, tokenizer, prompt, ex.get("evidence", ""), max_new_tokens=max_new_tokens, save_prefix=save_prefix)
        em_val = f1_val = None
        if isinstance(ex.get("answer"), str) and ex.get("answer").strip():
            em_val, f1_val = em_f1(fat["pred"], ex["answer"])
        rec = {
            "length_tokens": int(L),
            "fatigue": {k: fat[k] for k in ["attention", "drift", "entropy"]},
            "evidence_attention": fat["evidence_attention"],
            "evidence_distance": fat["evidence_distance"],
            "pred": fat["pred"],
            "latency_s": fat["latency_s"],
            "repetition3": fat["repetition3"],
            "em": em_val,
            "f1": f1_val,
            "gold": ex.get("answer", "")
        }
        results.append(rec)
        cleanup()
    return results

def positional_sensitivity_profile(model, tokenizer, ex: Dict, filler_len_tokens: int, max_new_tokens: int) -> List[Dict]:
    if not ex or not ex.get("question"):
        print("messed up: bad example in positional_sensitivity_profile")
        return []
    results = []
    for pos in ["front", "middle", "end"]:
        print(f"[Position] {pos}")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(filler_len_tokens), pos)
        if not prompt:
            print("messed up: prompt build failed"); continue
        save_prefix = os.path.join(OUTDIR, f"position_{pos}_id{ex.get('id','x')}")
        fat = run_fatigue_experiment(model, tokenizer, prompt, ex.get("evidence", ""), max_new_tokens=max_new_tokens, save_prefix=save_prefix)
        em_val = f1_val = None
        if isinstance(ex.get("answer"), str) and ex.get("answer").strip():
            em_val, f1_val = em_f1(fat["pred"], ex["answer"])
        rec = {
            "position": pos,
            "fatigue": {k: fat[k] for k in ["attention", "drift", "entropy"]},
            "evidence_attention": fat["evidence_attention"],
            "evidence_distance": fat["evidence_distance"],
            "pred": fat["pred"],
            "latency_s": fat["latency_s"],
            "repetition3": fat["repetition3"],
            "em": em_val,
            "f1": f1_val,
            "gold": ex.get("answer", "")
        }
        results.append(rec)
        cleanup()
    return results

def precision_quantization_ablation(ex_list: List[Dict], max_new_tokens: int, model_name: str = MODEL_NAME) -> Dict:
    """
    FP16/FP32 vs 4-bit NF4 comparison.
    Greedy decoding; same prompts per precision; measures entropy, repetition, EM/F1, latency.
    """
    out = {"fp": [], "4bit": []}
    # Build prompts once for both precisions (middle evidence, fixed filler)
    prompts = []
    for ex in ex_list:
        prompt = build_prompt_with_filler(ex.get("question", ""), ex.get("evidence", ""), "filler", 200, "middle")
        prompts.append((ex, prompt))

    # FP model
    print("[Precision] Loading FP model...")
    fp_model, fp_tok = load_fp(model_name)
    for ex, prompt in prompts:
        print("[Precision FP] id=", ex.get("id"))
        fat = run_fatigue_experiment(fp_model, fp_tok, prompt, ex.get("evidence", ""), max_new_tokens=max_new_tokens,
                                     save_prefix=os.path.join(OUTDIR, f"precision_fp_id{ex.get('id','x')}"))
        em_val = f1_val = None
        if isinstance(ex.get("answer"), str) and ex.get("answer").strip():
            em_val, f1_val = em_f1(fat["pred"], ex["answer"])
        out["fp"].append({
            "id": ex.get("id"), "pred": fat["pred"], "gold": ex.get("answer", ""),
            "entropy": fat["entropy"], "repetition3": fat["repetition3"], "latency_s": fat["latency_s"],
            "em": em_val, "f1": f1_val
        })
        cleanup()
    del fp_model; cleanup()

    # 4-bit model
    print("[Precision] Loading 4-bit model...")
    q_model, q_tok = load_4bit(model_name)
    for ex, prompt in prompts:
        print("[Precision 4bit] id=", ex.get("id"))
        fat = run_fatigue_experiment(q_model, q_tok, prompt, ex.get("evidence", ""), max_new_tokens=max_new_tokens,
                                     save_prefix=os.path.join(OUTDIR, f"precision_4bit_id{ex.get('id','x')}"))
        em_val = f1_val = None
        if isinstance(ex.get("answer"), str) and ex.get("answer").strip():
            em_val, f1_val = em_f1(fat["pred"], ex["answer"])
        out["4bit"].append({
            "id": ex.get("id"), "pred": fat["pred"], "gold": ex.get("answer", ""),
            "entropy": fat["entropy"], "repetition3": fat["repetition3"], "latency_s": fat["latency_s"],
            "em": em_val, "f1": f1_val
        })
        cleanup()
    del q_model; cleanup()

    return out

# ---------------- CLI ----------------
def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit for context/positional runs (precision ablation loads both anyway)")
    p.add_argument("--subset_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--run_context", action="store_true")
    p.add_argument("--run_positional", action="store_true")
    p.add_argument("--run_precision", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    # Ensure data
    if not os.path.exists(args.data):
        print(f"{args.data} not found — creating Hotpot sample...")
        create_hotpot_sample_from_hf(out_path=args.data, n=50, seed=42)

    # Load examples
    examples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    if not examples:
        print("messed up: no examples found; aborting.")
        return
    subset = examples[:max(1, int(args.subset_size))]

    # Load model for context/positional runs
    if args.run_context or args.run_positional:
        print(f"Loading {args.model} ({'4-bit' if args.use_4bit else 'FP'}) for context/positional...")
        if args.use_4bit:
            model, tokenizer = load_4bit(args.model)
        else:
            model, tokenizer = load_fp(args.model)
        print("Model on device:", next(model.parameters()).device)
    else:
        model = tokenizer = None

    # 1) Context-length stress
    if args.run_context:
        print("Running Context-Length Stress...")
        out_ctx = []
        for ex in subset:
            res = context_length_stress(model, tokenizer, ex,
                                        lengths_tokens=[CONTEXT_SHORT, CONTEXT_MEDIUM, CONTEXT_LONG],
                                        max_new_tokens=args.max_new_tokens)
            out_ctx.append({"id": ex.get("id"), "results": res})
        path = os.path.join(args.outdir, "ablation_context_length.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_ctx, f, indent=2)
        print("Saved", path)

    # 2) Positional sensitivity
    if args.run_positional:
        print("Running Positional Sensitivity...")
        out_pos = []
        for ex in subset:
            res = positional_sensitivity_profile(model, tokenizer, ex,
                                                 filler_len_tokens=min(200, USABLE_CONTEXT // 2),
                                                 max_new_tokens=args.max_new_tokens)
            out_pos.append({"id": ex.get("id"), "results": res})
        path = os.path.join(args.outdir, "ablation_positional.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_pos, f, indent=2)
        print("Saved", path)

    # 3) Precision/quantization ablation
    if args.run_precision:
        print("Running Precision/Quantization Ablation...")
        out_prec = precision_quantization_ablation(subset, max_new_tokens=args.max_new_tokens, model_name=args.model)
        path = os.path.join(args.outdir, "ablation_precision.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_prec, f, indent=2)
        print("Saved", path)

    print("All done. Outputs are in", args.outdir)

if __name__ == "__main__":
    main(sys.argv[1:])
