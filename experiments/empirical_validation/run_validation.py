"""Empirical validation of the Fatigue Index (paper Sec. 7 / RQ3).

Runs four reliability analyses on natural (unstressed) generations, using the
paper-faithful FI from the :mod:`fatigue` package and the shared harness in
:mod:`experiments.common`:

  E1  Temporal trajectory -- mean FI over decoding steps (does FI rise?).
  E2  Predictive validity -- Spearman rho between mean FI and behavioral
      degeneration (repetition ratio) and task failure (1 - F1), computed over
      the full generation and over the first-K window.
  E3  Aggregation value -- AUROC of FI vs each single normalized signal at
      predicting item-level failure (F1 <= median), testing whether combining
      the three signals beats any one alone.
  E4  Hysteresis stability -- alert flips under two-threshold hysteresis vs a
      naive single threshold.

Spearman and AUROC are computed with small numpy implementations so the script
runs without SciPy/scikit-learn, though both are listed in requirements.

Run from the repository root::

    python experiments/empirical_validation/run_validation.py \\
        --datasets hotpot --sample_size 50 --precision fp
"""

import os
import sys
import json
import argparse

import numpy as np

# Make the repository root importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt

from experiments.common import (
    MODEL_NAME, DEFAULT_MAX_NEW, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_TEMPERATURE,
    DEFAULT_SEED, DEFAULT_SAMPLE_SIZE,
    cleanup, set_seed, parse_seeds, safe_em_f1, repetition_ratio,
    build_prompt_with_filler, run_fatigue_experiment, load_fp, load_4bit,
    parse_datasets, dataset_sample_path, ensure_dataset_file, load_examples_jsonl,
    _save_plot,
)
from fatigue import (
    DEFAULT_CONFIG as FI_CONFIG,
    fatigue_index_series, mean_fi,
    phi_attention_series, phi_entropy_series, phi_drift_series,
    apply_hysteresis, naive_alerts, count_flips,
)

OUTDIR = "results/empirical_validation"


# ---------------------------------------------------------------------------
# Small self-contained statistics (avoid a hard SciPy/sklearn dependency)
# ---------------------------------------------------------------------------
def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(a, b):
    """Spearman rank correlation; returns nan for degenerate input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[mask]), _rankdata(b[mask])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def auroc(scores, labels):
    """AUROC via the Mann-Whitney U statistic; nan if only one class present."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
def _drop_ref(series):
    return series[1:] if isinstance(series, list) and len(series) > 1 else (series or [])


def collect_run(model, tok, ex, args):
    """Run one example and return its per-step signals, FI series and outcomes."""
    prompt = build_prompt_with_filler(ex.get("question", ""), ex.get("evidence", ""),
                                      "filler", 0, "front", context=ex.get("context", ""))
    if not prompt:
        return None
    fat = run_fatigue_experiment(model, tok, prompt, ex.get("evidence", ""),
                                 max_new_tokens=args.max_new, top_p=args.top_p,
                                 top_k=args.top_k, temperature=args.temperature)
    fi = fatigue_index_series(fat["attention"], fat["drift"], fat["entropy"], FI_CONFIG)
    _, f1 = safe_em_f1(fat["pred"], ex.get("answer"))
    a, e, d = _drop_ref(fat["attention"]), _drop_ref(fat["entropy"]), _drop_ref(fat["drift"])
    return {
        "fi": fi,
        "phi_attention": phi_attention_series(a, FI_CONFIG),
        "phi_entropy": phi_entropy_series(e, FI_CONFIG),
        "phi_drift": phi_drift_series(d, FI_CONFIG),
        "repetition": fat.get("repetition3", repetition_ratio(fat.get("pred", ""))),
        "f1": f1 if f1 is not None else 0.0,
        "pred": fat.get("pred", ""),
    }


def _mean_first_k(series, k):
    if not series:
        return float("nan")
    seg = series[:k] if k and k > 0 else series
    return float(np.mean(seg)) if seg else float("nan")


def e1_temporal(runs, outdir):
    series = [r["fi"] for r in runs if r["fi"]]
    if not series:
        return {"n": 0}
    max_len = max(len(s) for s in series)
    arr = np.full((len(series), max_len), np.nan)
    for i, s in enumerate(series):
        arr[i, :len(s)] = s
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    x = np.arange(len(mean))
    fig, ax = plt.subplots()
    ax.plot(x, mean, color="#7048e8", label="mean FI")
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color="#7048e8")
    ax.axhline(FI_CONFIG.hysteresis_high, ls="--", c="#e8590c", label=r"$\theta$")
    ax.set_xlabel("Decoding step")
    ax.set_ylabel("Fatigue Index")
    ax.set_title("E1: Temporal FI trajectory")
    ax.legend()
    _save_plot(fig, os.path.join(outdir, "e1_temporal_fi.png"))
    slope = float(np.polyfit(x, mean, 1)[0]) if len(mean) > 1 else 0.0
    return {"n": len(series), "fi_start": float(mean[0]), "fi_end": float(mean[-1]), "slope": slope}


def e2_predictive(runs, first_k):
    mean_fi_full = [mean_fi(r["fi"]) for r in runs]
    mean_fi_k = [_mean_first_k(r["fi"], first_k) for r in runs]
    rep = [r["repetition"] for r in runs]
    fail = [1.0 - r["f1"] for r in runs]
    return {
        "spearman_fi_repetition_full": spearman(mean_fi_full, rep),
        "spearman_fi_repetition_first%d" % first_k: spearman(mean_fi_k, rep),
        "spearman_fi_failure_full": spearman(mean_fi_full, fail),
        "spearman_fi_failure_first%d" % first_k: spearman(mean_fi_k, fail),
    }


def e3_aggregation(runs):
    f1 = np.array([r["f1"] for r in runs], dtype=float)
    if len(f1) < 4:
        return {"note": "too few samples for AUROC"}
    labels = (f1 <= np.median(f1)).astype(int)  # failure = 1
    fi = [mean_fi(r["fi"]) for r in runs]
    a = [float(np.mean(r["phi_attention"])) if r["phi_attention"] else 0.0 for r in runs]
    e = [float(np.mean(r["phi_entropy"])) if r["phi_entropy"] else 0.0 for r in runs]
    d = [float(np.mean(r["phi_drift"])) if r["phi_drift"] else 0.0 for r in runs]
    return {
        "auroc_fi": auroc(fi, labels),
        "auroc_attention_only": auroc(a, labels),
        "auroc_entropy_only": auroc(e, labels),
        "auroc_drift_only": auroc(d, labels),
        "n_fail": int(labels.sum()), "n_ok": int((1 - labels).sum()),
    }


def e4_hysteresis(runs):
    theta = FI_CONFIG.hysteresis_high
    hyst_flips, naive_flips = [], []
    for r in runs:
        if not r["fi"]:
            continue
        hyst_flips.append(count_flips(apply_hysteresis(r["fi"], FI_CONFIG)))
        naive_flips.append(count_flips(naive_alerts(r["fi"], theta)))
    if not naive_flips:
        return {"note": "no series"}
    mh, mn = float(np.mean(hyst_flips)), float(np.mean(naive_flips))
    return {
        "mean_flips_hysteresis": mh,
        "mean_flips_naive": mn,
        "flip_reduction_pct": (100.0 * (mn - mh) / mn) if mn > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Empirical validation of the Fatigue Index (RQ3).")
    ap.add_argument("--datasets", default="hotpot")
    ap.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--precision", choices=["fp", "4bit"], default="fp")
    ap.add_argument("--max_new", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--first_k", type=int, default=20)
    ap.add_argument("--seeds", default=str(DEFAULT_SEED))
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seeds = parse_seeds(args.seeds, DEFAULT_SEED)
    datasets = parse_datasets(args.datasets) or ["hotpot"]

    print(f"[validation] model={args.model} precision={args.precision} "
          f"datasets={datasets} seeds={seeds}")
    model, tok = (load_fp(args.model) if args.precision == "fp"
                  else load_4bit(args.model, require_4bit=True))

    report = {"model": args.model, "precision": args.precision,
              "config": FI_CONFIG.to_dict(), "seeds": seeds, "datasets": {}}

    for ds in datasets:
        path = dataset_sample_path(ds, args.sample_size)
        ensure_dataset_file(ds, path, args.sample_size, seeds[0])
        examples = load_examples_jsonl(path)
        if not examples:
            print(f"[validation] no examples for {ds}; skipping")
            continue
        runs = []
        for seed in seeds:
            set_seed(seed)
            for ex in examples:
                r = collect_run(model, tok, ex, args)
                if r:
                    runs.append(r)
                cleanup()
        ds_out = os.path.join(args.outdir, ds)
        os.makedirs(ds_out, exist_ok=True)
        result = {
            "n_runs": len(runs),
            "E1_temporal": e1_temporal(runs, ds_out),
            "E2_predictive_validity": e2_predictive(runs, args.first_k),
            "E3_aggregation_auroc": e3_aggregation(runs),
            "E4_hysteresis_stability": e4_hysteresis(runs),
        }
        report["datasets"][ds] = result
        print(f"[validation] {ds}: {json.dumps(result, indent=2)}")

    with open(os.path.join(args.outdir, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[validation] wrote {os.path.join(args.outdir, 'validation_report.json')}")


if __name__ == "__main__":
    main()

