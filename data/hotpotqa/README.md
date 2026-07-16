# HotpotQA

Samples for this dataset are generated automatically — you do not need to
download anything by hand. Running either experiment driver (or
`experiments.common.ensure_dataset_file`) with `--datasets hotpot` will:

1. Download the `fullwiki` configuration of `hotpot_qa` from the Hugging Face
   Hub (`datasets.load_dataset("hotpot_qa", "fullwiki")`) on first use.
2. Sample `--sample_size` validation examples (seeded, default 50).
3. Extract `question`, `context`, `answer`, and a supporting-fact `evidence`
   sentence for each, and cache the result as JSONL at
   `data/hotpot_sample_<n>.jsonl`.

To force a fresh sample (e.g. after changing `--sample_size` or `--seed`),
delete the cached JSONL file and re-run; to force a re-download from the Hub,
pass `--hf_force_download` (see `experiments/common.py`).

Set `HF_TOKEN` in your environment if the dataset requires authentication, and
`HF_DATASET_REVISION` to pin a specific revision (commit-hash pins are
detected and ignored with a warning, since the Hub does not always resolve a
raw hash the same way as a named revision).
