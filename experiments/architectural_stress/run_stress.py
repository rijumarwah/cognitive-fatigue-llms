"""Architectural stress experiments (paper Sec. 5.2 / RQ2).

Runs the context-length, positional-sensitivity, and precision (FP16 vs 4-bit
NF4) ablations. The data/model/generation harness and all Fatigue Index math
come from :mod:`experiments.common` and the :mod:`fatigue` package; this file
only adds the ablation orchestration, per-run reporting, and aggregate plots.

Run from the repository root::

    python experiments/architectural_stress/run_stress.py --datasets hotpot \\
        --run_context --run_positional --run_precision
"""

import os
import sys
import json

from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

# Make the repository root importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.common import *  # noqa: F401,F403  (shared harness + FI helpers)
from experiments.common import (
    PROBE_EVERY, ENTROPY_BAND_LOW, ENTROPY_BAND_HIGH, ENTROPY_LAST_K,
    MODEL_NAME, DATA_PATH, DEFAULT_DATASETS, DEFAULT_SAMPLE_SIZE,
    DEFAULT_MAX_NEW, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_TEMPERATURE, DEFAULT_SEED,
    DEFAULT_HF_REVISION, DEFAULT_HF_REVISION_ENV, DEFAULT_HF_TOKEN, DEFAULT_HF_FORCE_DOWNLOAD,
    CONTEXT_SHORT, CONTEXT_MEDIUM, CONTEXT_LONG, USABLE_CONTEXT,
    PLOT_METRICS, cleanup, set_seed, parse_seeds, safe_em_f1,
    build_prompt_with_filler, run_fatigue_experiment, load_fp, load_4bit,
    parse_datasets, dataset_sample_path, ensure_dataset_file, load_examples_jsonl,
    _fatigue_index_series, _save_plot, _plot_aggregate_groups, _plot_aggregate_groups_on_ax,
    _sanitize_revision,
)

OUTDIR = "results/architectural_stress"
os.makedirs(OUTDIR, exist_ok=True)


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


def _mean(series: List[float]) -> float:
    if not isinstance(series, list) or not series:
        return 0.0
    return float(np.mean(series))


def context_length_stress(model, tokenizer, ex: Dict, lengths_tokens: List[int],
                          max_new_tokens: int, top_p: float, top_k: int,
                          temperature: float) -> List[Dict]:
    if not ex or not ex.get("question"):
        print("Warning: bad example in context_length_stress")
        return []
    results = []
    for L in lengths_tokens:
        print(f"[ContextLen] target={L} tokens")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(L), "middle",
                                          context=ex.get("context", ""), use_context_filler=False)
        if not prompt:
            print("Warning: prompt build failed")
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
        print("Warning: bad example in positional_sensitivity_profile")
        return []
    results = []
    for pos in ["front", "middle", "end"]:
        print(f"[Position] {pos}")
        prompt = build_prompt_with_filler(ex["question"], ex.get("evidence", ""), "filler", int(filler_len_tokens), pos,
                                          context=ex.get("context", ""), use_context_filler=False)
        if not prompt:
            print("Warning: prompt build failed")
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
        print("Warning: no valid datasets specified; aborting.")
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
            print(f"Warning: no examples found for {dataset_name}; skipping.")
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
                    print("Warning: context aggregate plots failed:", e)
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
                    print("Warning: positional aggregate plots failed:", e)
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
                    print("Warning: precision aggregate plots failed:", e)
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
