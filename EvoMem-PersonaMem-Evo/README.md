# EvoMem-PersonaMem-Evo

This folder contains the EvoMem implementation for PersonaMem-Evo experiments. It includes the memory layers, patch-store logic, baseline/patch runners, and a compact benchmark subset at `data/personamem-evo-10p.csv` for quick reproduction or smoke tests.

## Setup

```bash
cd EvoMem-PersonaMem-Evo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key for your backend. The Python runners read environment variables directly; they do not automatically load `.env` files.

```bash
export OPENAI_API_KEY=[your_api_key]
# or, for OpenRouter:
export OPENROUTER_API_KEY=[your_api_key]
```

You can also create a local `.env` file and source it before running direct Python commands:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## Backend and API Base

Both experiment runners support `--backend [openai/openrouter/ollama/sglang/vllm]` and `--model [your_model]`.

For common cases, the code already handles the base URL:

- `--backend openai`: uses the default OpenAI API base unless `--api_base` or `OPENAI_BASE_URL` is set.
- `--backend openrouter`: automatically uses `https://openrouter.ai/api/v1` if no `--api_base` is provided.
- Custom OpenAI-compatible endpoints: pass `--api_base [your_base_url]` or set `OPENAI_BASE_URL`.
- `sglang` and `vllm`: use `--sglang_host/--sglang_port` or `--api_base` as appropriate for your local server.

## Run Experiments Directly

Use placeholders for model/backend rather than copying one fixed setup. The important retrieval settings used in our current runs are `--retrieve_k 10` and, for patch mode, `--min_patch_similarity 0.4`.

Robust baseline:

```bash
python test_persona_robust.py \
  --benchmark_file data/personamem-evo-10p.csv \
  --persona_root data \
  --model [your_model] \
  --backend [openai/openrouter] \
  --size 32k \
  --retrieve_k 10 \
  --preference_aware_level none \
  --output results/robust_[your_model]_10p.csv
```

Patch / EvoMem mode:

```bash
python test_persona_patch.py \
  --benchmark_file data/personamem-evo-10p.csv \
  --persona_root data \
  --model [your_model] \
  --backend [openai/openrouter] \
  --size 32k \
  --retrieve_k 10 \
  --patch_top_k 3 \
  --patch_usage always \
  --min_patch_similarity 0.4 \
  --preference_aware_level full \
  --output results/patch_[your_model]_10p.csv
```

In the standard setting, the robust baseline uses `--preference_aware_level none`, and patch/EvoMem mode uses `--preference_aware_level full`.

For a custom endpoint, add:

```bash
--api_base [your_openai_compatible_base_url]
```

For a quick smoke test, add:

```bash
--max_items 5
```

For parallel execution, add `--batch [num_workers]`. Larger batches are faster but may hit provider rate limits.

## Run with the Helper Script

You can also run experiments through:

```bash
bash scripts/run_persona_baseline_patch.sh [robust|patch|both]
```

If no target is passed, the script uses `CONFIG_RUN_TARGET` from the top of `scripts/run_persona_baseline_patch.sh`.

The script has two ways to configure a run:

1. Edit the `CONFIG_*` variables near the top of the script.
2. Override them from the shell for a single run.

Important script parameters:

| Parameter | Meaning |
| --- | --- |
| `CONFIG_RUN_TARGET` / positional arg | `robust`, `patch`, or `both`. |
| `CONFIG_BACKEND` / `BACKEND` | Backend passed to the Python runner, e.g. `openai` or `openrouter`. |
| `CONFIG_API_KEY` / `API_KEY` | API key. The script exports it as both `OPENAI_API_KEY` and `OPENROUTER_API_KEY`. |
| `CONFIG_API_BASE` / `API_BASE` | Optional API base URL. Leave empty to use backend defaults; set it for custom endpoints. |
| `CONFIG_MODELS` / `MODEL_LIST` | Model or comma-separated model list. Use provider-style names for OpenRouter when needed. |
| `CONFIG_BENCHMARK_FILE` / `BENCHMARK_FILE` | Benchmark CSV path. |
| `CONFIG_PERSONA_ROOT` / `PERSONA_ROOT` | Optional Persona data root; can usually stay empty if auto-resolution works. |
| `CONFIG_SIZE` / `SIZE` | Chat-history size column, usually `32k`. |
| `CONFIG_BATCH` / `BATCH` | Number of worker processes. |
| `CONFIG_RETRIEVE_K` / `RETRIEVE_K` | Current-memory retrieval top-k. Current setting: `10`. |
| `CONFIG_PATCH_TOP_K` / `PATCH_TOP_K` | Historical patch retrieval top-k. Current setting: `3`. |
| `CONFIG_MIN_PATCH_SIMILARITY` / `MIN_PATCH_SIMILARITY` | Patch similarity threshold. Current setting: `0.4`. |
| `CONFIG_PATCH_USAGE` / `PATCH_USAGE` | `always` or `gated`. |
| `CONFIG_OUTPUT_DIR` / `OUTPUT_DIR` | Output directory for result CSVs. |
| `CONFIG_RESUME` / `RESUME` | `1` to reuse existing outputs/caches and resume. |
| `CONFIG_FORCE_REINGEST_PATCHES` / `FORCE_REINGEST_PATCHES` | `1` to rebuild patch caches; keep `0` when resuming. |
| `CONFIG_MAX_ITEMS` / `MAX_ITEMS` | Optional row limit for smoke tests. |
| `CONFIG_PERSONA_IDS` / `PERSONA_IDS` | Optional comma-separated persona-id filter. |

Example: run patch mode on OpenRouter without editing the script:

```bash
BACKEND=openrouter \
API_KEY=[your_api_key] \
MODEL_LIST=[your_model] \
BENCHMARK_FILE=data/personamem-evo-10p.csv \
RETRIEVE_K=10 \
PATCH_TOP_K=3 \
MIN_PATCH_SIMILARITY=0.4 \
BATCH=4 \
bash scripts/run_persona_baseline_patch.sh patch
```

Example: run both robust and patch modes with a custom OpenAI-compatible endpoint:

```bash
BACKEND=openai \
API_KEY=[your_api_key] \
API_BASE=[your_openai_compatible_base_url] \
MODEL_LIST=[your_model] \
BENCHMARK_FILE=data/personamem-evo-10p.csv \
RETRIEVE_K=10 \
MIN_PATCH_SIMILARITY=0.4 \
bash scripts/run_persona_baseline_patch.sh both
```

The helper script fixes preference-aware mode by run type: robust baseline uses `none`, and patch/EvoMem mode uses `full`. It also sources `.env` automatically if it exists in `EvoMem-PersonaMem-Evo/`. This is different from the direct Python commands, where you need to source `.env` yourself.

## Check Results

Each run writes a result CSV plus companion metric and usage files next to the output path. The main MCQ metric is in the generated `*_metrics.txt` file and in the `is_correct_mcq_32k` column of the result CSV.

To compute chain exact-match accuracy, run:

```bash
python scripts/evaluate_persona_chain_acc.py \
  results/robust_[your_model]_10p.csv \
  results/patch_[your_model]_10p.csv \
  --size 32k \
  --output results/chain_acc_summary.csv
```

Chain accuracy treats all rows with the same `chain_id` as one group; a chain is correct only when every QA in that group is correct.
