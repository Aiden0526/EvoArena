# EvoMem-PersonaMem-Evo

This repository contains the cleaned code and a compact PersonaMem-Evo benchmark subset for reproducing robust and patch-mode memory experiments. The included benchmark is `data/personamem-evo-10p.csv`, a 10-person OOD subset with `chain_id` annotations for post-hoc chain accuracy. The 10p data is small enough to ship directly with the code, so no external dataset download is required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key for your backend. The scripts read environment variables directly; `.env` files are not loaded automatically. You can either export variables in your shell:

```bash
export OPENAI_API_KEY=your_key
# or, for OpenRouter:
export OPENROUTER_API_KEY=your_key
```

Or create `.env` from the template and source it before running:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## Run Experiments

Robust mode:

```bash
python test_persona_robust.py \
  --benchmark_file data/personamem-evo-10p.csv \
  --persona_root data \
  --model gpt-4o-mini \
  --backend openai \
  --size 32k \
  --retrieve_k 12 \
  --preference_aware_level full \
  --output results/robust_gpt-4o-mini_10p.csv
```

Patch mode:

```bash
python test_persona_patch.py \
  --benchmark_file data/personamem-evo-10p.csv \
  --persona_root data \
  --model gpt-4o-mini \
  --backend openai \
  --size 32k \
  --retrieve_k 12 \
  --patch_top_k 3 \
  --patch_usage always \
  --min_patch_similarity 0.5 \
  --preference_aware_level full \
  --output results/patch_gpt-4o-mini_10p.csv
```

For OpenRouter, use `--backend openrouter --model provider/model-name`; for a custom OpenAI-compatible endpoint, pass `--api_base` or set `OPENAI_BASE_URL`.

For a quick smoke test, add `--max_items 5`.

## Check Results

Each run writes a result CSV plus companion metric and usage files next to the output path. The main QA metric is in the generated `*_metrics.txt` file and in the `is_correct_mcq_32k` column of the result CSV.

To compute chain exact-match accuracy, run:

```bash
python scripts/evaluate_persona_chain_acc.py \
  results/robust_gpt-4o-mini_10p.csv \
  results/patch_gpt-4o-mini_10p.csv \
  --size 32k \
  --output results/chain_acc_summary.csv
```

Chain accuracy treats all rows with the same `chain_id` as one group; a chain is correct only when every QA in that group is correct.
