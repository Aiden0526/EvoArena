#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# =========================
# Editable experiment config
# =========================

# Which runner to launch by default if you do not pass [robust|patch|both].
CONFIG_RUN_TARGET="patch" # robust, patch, or both

# LLM backend/provider flag passed through to the Persona runner.
# Examples: openai, openrouter, vllm, sglang.
CONFIG_BACKEND="openrouter"

# API key for OpenAI-compatible backends.
# Leave empty here if you prefer exporting OPENAI_API_KEY or API_KEY in the shell.
CONFIG_API_KEY=""

# Optional API base URL.
# Examples:
#   https://api.openai.com/v1
#   https://openrouter.ai/api/v1
# Leave empty to use the backend default.
CONFIG_API_BASE="https://openrouter.ai/api/v1"

# OpenRouter prompt caching. Provider support varies; usage summaries record cache hits/writes.
CONFIG_OPENROUTER_PROMPT_CACHE="1"
CONFIG_OPENROUTER_PROMPT_CACHE_TTL=""

# Benchmark CSV to evaluate.
# Default is the 9-person OOD split.
CONFIG_BENCHMARK_FILE="data/PersonaMem-v2-enhanced-release/benchmark_v34/text/benchmark_9p_ood_v34.csv"

# Optional manual override for the PersonaMem root directory.
# Usually this can stay empty because the runner auto-resolves it from BENCHMARK_FILE.
CONFIG_PERSONA_ROOT=""

# Which chat-history size column to use from the benchmark.
# Common values depend on the CSV schema; 32k is the default used in the repo.
CONFIG_SIZE="32k"

# Number of worker processes launched by the runner.
# Higher is faster but increases rate-limit / timeout pressure.
CONFIG_BATCH="9"

# Number of current A-Mem memories to retrieve for each question.
CONFIG_RETRIEVE_K="10"

# Output directory for merged CSV results.
CONFIG_OUTPUT_DIR="results"

# LiteLLM log level.
# Set to WARNING or ERROR to suppress noisy provider-help banners.
# Set to DEBUG only when you need to inspect LiteLLM internals.
CONFIG_LITELLM_LOG="WARNING"

# Default model list. You can also override with MODEL_LIST=a,b,c.
CONFIG_MODELS=(
  "deepseek/deepseek-v4-pro"
  "z-ai/glm-5.1"
  "minimax/minimax-m2.7"
)

# -------------------------
# Common optional filters / knobs
# -------------------------

# Optional comma-separated persona-id filter.
# Example: "0,1,10". Leave empty to run all personas in the benchmark file.
CONFIG_PERSONA_IDS=""

# Optional limit on the number of benchmark rows to process.
# Leave empty to run the full benchmark split.
CONFIG_MAX_ITEMS=""

# Whether to write prompts, retrieval keywords, contexts, and metadata into the CSV.
# 1 = pass --save_debug_columns, 0 = keep output leaner.
CONFIG_INCLUDE_DEBUG_COLUMNS="0"

# Optional cache directory override.
# Leave empty to let the runner auto-name its cache directory.
CONFIG_CACHE_ROOT=""

# Legacy boolean switch for preference-aware prompting.
# 1 = enable old behavior, 0 = disable.
# If CONFIG_PREFERENCE_AWARE_LEVEL is not "none", that level takes precedence.
CONFIG_PREFERENCE_AWARE="1"

# Newer preference-aware control.
# none       = original prompts
# patch_only = preference-aware patch analysis/prompts with base graph prompts
# full       = preference-aware everywhere; equivalent to legacy --preference_aware
CONFIG_PREFERENCE_AWARE_LEVEL="none"

# Resume behavior.
# 1 = reuse stable output/cache paths and let the Python runner resume in place.
# 0 = refuse to reuse an existing output path.
CONFIG_RESUME="1"

# -------------------------
# Patch-only knobs
# -------------------------

# Number of historical patches to retrieve per query.
CONFIG_PATCH_TOP_K="3"

# How patch evidence is used.
# always = always inject retrieved patches into the answer context
# gated  = first ask the model whether patch detail is needed
CONFIG_PATCH_USAGE="always"

# Minimum cosine similarity threshold for retrieved patches.
# Larger values filter more aggressively.
CONFIG_MIN_PATCH_SIMILARITY="0.4"

# Whether to rebuild / reingest patch memories instead of reusing cached ones.
# 1 = force rebuild, 0 = reuse existing cache when available.
CONFIG_FORCE_REINGEST_PATCHES="0"

# Exclude revoke-type patches from retrieval/injection.
# Usually leave this at 0 unless you are doing an ablation.
CONFIG_EXCLUDE_REVOKE_PATCHES="0"

# Exclude additive patches from retrieval/injection.
# Usually leave this at 0 unless you are doing an ablation.
CONFIG_EXCLUDE_ADD_PATCHES="0"

# Require retrieved patches to indicate an explicit preference change.
# Mainly useful for controlled ablations.
CONFIG_REQUIRE_PREF_CHANGE="0"

# Ask an LLM to further filter candidate patches before use.
# Usually off unless you are testing that variant.
CONFIG_LLM_PATCH_FILTER="0"

# Enable gold-patch injection mode.
# This is for analysis / oracle-style evaluation, not the standard EvoMem run.
CONFIG_GT_PATCH="0"

# Optional path to a JSONL gold patch store.
# Leave empty to use the runner default when CONFIG_GT_PATCH=1.
CONFIG_GT_PATCH_FILE=""

# Number of gold patches to retrieve.
# Leave empty to fall back to CONFIG_PATCH_TOP_K.
CONFIG_GT_PATCH_TOP_K=""

# Similarity threshold for gold-patch retrieval.
# Leave empty to fall back to CONFIG_MIN_PATCH_SIMILARITY.
CONFIG_GT_PATCH_MIN_SIMILARITY=""

# How to choose gold patches when CONFIG_GT_PATCH=1.
# similarity = retrieve gold patches by embedding similarity
# oracle     = directly inject the row-aligned gold patch for that exact changed sample
#              This is only for upper-bound / debugging analysis and is not the normal patch setting.
CONFIG_GP_PATCH_RETRIEVAL="similarity"

usage() {
  cat <<'USAGE_EOF'
Usage:
  bash scripts/run_persona_baseline_patch.sh [robust|patch|both]

If no positional argument is passed, CONFIG_RUN_TARGET is used.
The script supports editable defaults at the top and env-var overrides.

Common overrides:
  BACKEND                LLM backend / provider flag passed to the runner
  API_KEY                API key for OpenAI-compatible backends
  API_BASE               Base URL or full chat/completions URL
  OPENROUTER_PROMPT_CACHE 1 to send OpenRouter prompt-cache control
  OPENROUTER_PROMPT_CACHE_TTL Optional cache TTL, e.g. 1h where supported
  MODEL_LIST             Comma-separated models; overrides CONFIG_MODELS
  BENCHMARK_FILE         Persona benchmark CSV
  PERSONA_ROOT           Optional Persona-release root override
  OUTPUT_DIR             Directory for merged outputs
  SIZE                   Persona chat-history size column, e.g. 32k
  BATCH                  Number of worker processes
  RETRIEVE_K             Current-memory retrieval top-k
  PERSONA_IDS            Optional comma-separated persona id filter
  MAX_ITEMS              Optional max benchmark rows
  INCLUDE_DEBUG_COLUMNS  1 to write prompts/contexts/metadata to output
  CACHE_ROOT             Optional cache directory override
  RESUME                 1 to automatically resume from existing outputs/caches
  PREFERENCE_AWARE       1 to use legacy preference-aware mode
  PREFERENCE_AWARE_LEVEL none|patch_only|full

Patch-only overrides:
  PATCH_TOP_K
  PATCH_USAGE
  MIN_PATCH_SIMILARITY
  FORCE_REINGEST_PATCHES 1 or 0 (set this to 0 for resume)
  EXCLUDE_REVOKE_PATCHES 1 or 0
  EXCLUDE_ADD_PATCHES    1 or 0
  REQUIRE_PREF_CHANGE    1 or 0
  LLM_PATCH_FILTER       1 or 0
  GT_PATCH               1 or 0
  GT_PATCH_FILE
  GT_PATCH_TOP_K
  GT_PATCH_MIN_SIMILARITY
  GP_PATCH_RETRIEVAL     similarity|oracle

Examples:
  bash scripts/run_persona_baseline_patch.sh patch
  BACKEND=openrouter API_KEY=... API_BASE=https://openrouter.ai/api/v1 \
    MODEL_LIST=moonshotai/kimi-k2.6 \
    bash scripts/run_persona_baseline_patch.sh robust
  MODEL_LIST=gpt-5.4-mini-2026-03-17,gpt-4.1-mini \
    PATCH_TOP_K=5 MIN_PATCH_SIMILARITY=0.3 \
    bash scripts/run_persona_baseline_patch.sh both
USAGE_EOF
}

RUN_TARGET="${1:-$CONFIG_RUN_TARGET}"
case "$RUN_TARGET" in
  robust|patch|both) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

BACKEND="${BACKEND:-$CONFIG_BACKEND}"
API_KEY="${OPENAI_API_KEY:-${API_KEY:-$CONFIG_API_KEY}}"
API_BASE="${OPENAI_BASE_URL:-${API_BASE:-$CONFIG_API_BASE}}"
OPENROUTER_PROMPT_CACHE="${OPENROUTER_PROMPT_CACHE:-$CONFIG_OPENROUTER_PROMPT_CACHE}"
OPENROUTER_PROMPT_CACHE_TTL="${OPENROUTER_PROMPT_CACHE_TTL:-$CONFIG_OPENROUTER_PROMPT_CACHE_TTL}"
BENCHMARK_FILE="${BENCHMARK_FILE:-$CONFIG_BENCHMARK_FILE}"
PERSONA_ROOT="${PERSONA_ROOT:-$CONFIG_PERSONA_ROOT}"
SIZE="${SIZE:-$CONFIG_SIZE}"
BATCH="${BATCH:-$CONFIG_BATCH}"
RETRIEVE_K="${RETRIEVE_K:-$CONFIG_RETRIEVE_K}"
OUTPUT_DIR="${OUTPUT_DIR:-$CONFIG_OUTPUT_DIR}"
LITELLM_LOG_LEVEL="${LITELLM_LOG:-$CONFIG_LITELLM_LOG}"
PERSONA_IDS="${PERSONA_IDS:-$CONFIG_PERSONA_IDS}"
MAX_ITEMS="${MAX_ITEMS:-$CONFIG_MAX_ITEMS}"
INCLUDE_DEBUG_COLUMNS="${INCLUDE_DEBUG_COLUMNS:-$CONFIG_INCLUDE_DEBUG_COLUMNS}"
CACHE_ROOT="${CACHE_ROOT:-$CONFIG_CACHE_ROOT}"
RESUME="${RESUME:-$CONFIG_RESUME}"
PREFERENCE_AWARE="${PREFERENCE_AWARE:-$CONFIG_PREFERENCE_AWARE}"
PREFERENCE_AWARE_LEVEL="${PREFERENCE_AWARE_LEVEL:-$CONFIG_PREFERENCE_AWARE_LEVEL}"

PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_PATCH_TOP_K}"
PATCH_USAGE="${PATCH_USAGE:-$CONFIG_PATCH_USAGE}"
MIN_PATCH_SIMILARITY="${MIN_PATCH_SIMILARITY:-$CONFIG_MIN_PATCH_SIMILARITY}"
FORCE_REINGEST_PATCHES="${FORCE_REINGEST_PATCHES:-$CONFIG_FORCE_REINGEST_PATCHES}"
EXCLUDE_REVOKE_PATCHES="${EXCLUDE_REVOKE_PATCHES:-$CONFIG_EXCLUDE_REVOKE_PATCHES}"
EXCLUDE_ADD_PATCHES="${EXCLUDE_ADD_PATCHES:-$CONFIG_EXCLUDE_ADD_PATCHES}"
REQUIRE_PREF_CHANGE="${REQUIRE_PREF_CHANGE:-$CONFIG_REQUIRE_PREF_CHANGE}"
LLM_PATCH_FILTER="${LLM_PATCH_FILTER:-$CONFIG_LLM_PATCH_FILTER}"
GT_PATCH="${GT_PATCH:-$CONFIG_GT_PATCH}"
GT_PATCH_FILE="${GT_PATCH_FILE:-$CONFIG_GT_PATCH_FILE}"
GT_PATCH_TOP_K="${GT_PATCH_TOP_K:-$CONFIG_GT_PATCH_TOP_K}"
GT_PATCH_MIN_SIMILARITY="${GT_PATCH_MIN_SIMILARITY:-$CONFIG_GT_PATCH_MIN_SIMILARITY}"
GP_PATCH_RETRIEVAL="${GP_PATCH_RETRIEVAL:-$CONFIG_GP_PATCH_RETRIEVAL}"

MODELS=("${CONFIG_MODELS[@]}")
if [[ -n "${MODEL_LIST:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODEL_LIST"
fi

NORMALIZED_MODELS=()
for model in "${MODELS[@]}"; do
  model="${model#${model%%[![:space:]]*}}"
  model="${model%${model##*[![:space:]]}}"
  model="${model%,}"
  if [[ -n "$model" ]]; then
    NORMALIZED_MODELS+=("$model")
  fi
done
MODELS=("${NORMALIZED_MODELS[@]}")

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "At least one model is required. Edit CONFIG_MODELS or set MODEL_LIST." >&2
  exit 1
fi

if [[ "${#MODELS[@]}" -eq 1 && -z "${MODELS[0]}" ]]; then
  echo "At least one non-empty model is required. Edit CONFIG_MODELS or set MODEL_LIST." >&2
  exit 1
fi

if [[ ! -f "$BENCHMARK_FILE" ]]; then
  echo "Benchmark file not found: $BENCHMARK_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Persona runners do not accept --api_key on the CLI.
# They read credentials from environment variables through memory_layer_robust.py.
if [[ -n "$API_KEY" ]]; then
  export OPENAI_API_KEY="$API_KEY"
  export OPENROUTER_API_KEY="$API_KEY"
fi
if [[ -n "$API_BASE" ]]; then
  export OPENAI_BASE_URL="$API_BASE"
fi
export LITELLM_LOG="$LITELLM_LOG_LEVEL"
export OPENROUTER_PROMPT_CACHE
export OPENROUTER_PROMPT_CACHE_TTL

ensure_resume_ready() {
  local output_path="$1"
  local label="$2"

  if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
    if [[ -f "$output_path" ]]; then
      echo "Resume enabled: reusing existing $label output $output_path"
    else
      echo "Resume enabled: starting fresh $label output $output_path"
    fi
    return 0
  fi

  if [[ -f "$output_path" ]]; then
    echo "$label output already exists and RESUME=0: $output_path" >&2
    echo "Either set RESUME=1 to continue, or change OUTPUT_DIR / model / benchmark target." >&2
    exit 1
  fi
}

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  value="${value//./_}"
  value="${value//-/_}"
  echo "$value"
}

persona_selection_suffix() {
  local suffix=""
  if [[ -n "$PERSONA_IDS" ]]; then
    local safe_ids="${PERSONA_IDS//,/ _}"
    safe_ids="${safe_ids// /_}"
    safe_ids="${safe_ids//[^A-Za-z0-9_]/_}"
    safe_ids="${safe_ids//__/_}"
    suffix="_personas_${safe_ids}"
  elif [[ -n "$MAX_ITEMS" ]]; then
    suffix="_maxitems_$(sanitize "$MAX_ITEMS")"
  fi
  printf '%s' "$suffix"
}

bool_enabled() {
  case "$1" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

set_common_args() {
  local model="$1"
  COMMON_ARGS=(
    --backend "$BACKEND"
    --model "$model"
    --benchmark_file "$BENCHMARK_FILE"
    --size "$SIZE"
    --retrieve_k "$RETRIEVE_K"
  )

  if [[ -n "$API_BASE" ]]; then
    COMMON_ARGS+=(--api_base "$API_BASE")
  fi
  if [[ -n "$PERSONA_ROOT" ]]; then
    COMMON_ARGS+=(--persona_root "$PERSONA_ROOT")
  fi
  if [[ -n "$PERSONA_IDS" ]]; then
    COMMON_ARGS+=(--persona_ids "$PERSONA_IDS")
  fi
  if [[ -n "$MAX_ITEMS" ]]; then
    COMMON_ARGS+=(--max_items "$MAX_ITEMS")
  fi
  if [[ -n "$BATCH" ]]; then
    COMMON_ARGS+=(--batch "$BATCH")
  fi
  if [[ -n "$CACHE_ROOT" ]]; then
    COMMON_ARGS+=(--cache_root "$CACHE_ROOT")
  fi

  if [[ "$PREFERENCE_AWARE_LEVEL" != "none" ]]; then
    COMMON_ARGS+=(--preference_aware_level "$PREFERENCE_AWARE_LEVEL")
  else
    if bool_enabled "$PREFERENCE_AWARE"; then
      COMMON_ARGS+=(--preference_aware)
    fi
  fi

  if bool_enabled "$INCLUDE_DEBUG_COLUMNS"; then
    COMMON_ARGS+=(--save_debug_columns)
  fi
}

set_patch_args() {
  PATCH_ARGS=(
    --patch_top_k "$PATCH_TOP_K"
    --patch_usage "$PATCH_USAGE"
    --min_patch_similarity "$MIN_PATCH_SIMILARITY"
    --gp_patch_retrieval "$GP_PATCH_RETRIEVAL"
  )

  if bool_enabled "$FORCE_REINGEST_PATCHES"; then
    PATCH_ARGS+=(--force_reingest_patches)
  fi
  if bool_enabled "$EXCLUDE_REVOKE_PATCHES"; then
    PATCH_ARGS+=(--exclude_revoke_patches)
  fi
  if bool_enabled "$EXCLUDE_ADD_PATCHES"; then
    PATCH_ARGS+=(--exclude_add_patches)
  fi
  if bool_enabled "$REQUIRE_PREF_CHANGE"; then
    PATCH_ARGS+=(--require_pref_change)
  fi
  if bool_enabled "$LLM_PATCH_FILTER"; then
    PATCH_ARGS+=(--llm_patch_filter)
  fi
  if bool_enabled "$GT_PATCH"; then
    PATCH_ARGS+=(--gt_patch)
  fi

  if [[ -n "$GT_PATCH_FILE" ]]; then
    PATCH_ARGS+=(--gt_patch_file "$GT_PATCH_FILE")
  fi
  if [[ -n "$GT_PATCH_TOP_K" ]]; then
    PATCH_ARGS+=(--gt_patch_top_k "$GT_PATCH_TOP_K")
  fi
  if [[ -n "$GT_PATCH_MIN_SIMILARITY" ]]; then
    PATCH_ARGS+=(--gt_patch_min_similarity "$GT_PATCH_MIN_SIMILARITY")
  fi
}

print_resolved_config() {
  echo "Run target: $RUN_TARGET"
  echo "Backend: $BACKEND"
  echo "API base: ${API_BASE:-NONE}"
  echo "Benchmark: $BENCHMARK_FILE"
  echo "OpenRouter prompt cache: ${OPENROUTER_PROMPT_CACHE:-0} | ttl: ${OPENROUTER_PROMPT_CACHE_TTL:-default}"
  echo "Persona root: ${PERSONA_ROOT:-AUTO}"
  echo "Size: $SIZE | batch: $BATCH | retrieve_k: $RETRIEVE_K"
  echo "persona_ids: ${PERSONA_IDS:-ALL} | max_items: ${MAX_ITEMS:-ALL}"
  echo "preference_aware_level: $PREFERENCE_AWARE_LEVEL | legacy_preference_aware: $PREFERENCE_AWARE"
  echo "output_dir: $OUTPUT_DIR | cache_root: ${CACHE_ROOT:-AUTO} | litellm_log: $LITELLM_LOG_LEVEL"
  echo "resume: $RESUME"
  if [[ "$RUN_TARGET" == "patch" || "$RUN_TARGET" == "both" ]]; then
    echo "patch_top_k: $PATCH_TOP_K | patch_usage: $PATCH_USAGE | min_patch_similarity: $MIN_PATCH_SIMILARITY"
    echo "force_reingest_patches: $FORCE_REINGEST_PATCHES | gt_patch: $GT_PATCH | gp_patch_retrieval: $GP_PATCH_RETRIEVAL"
  fi
  echo "Models: ${MODELS[*]}"
}

print_resolved_config

if [[ "$RUN_TARGET" == "patch" || "$RUN_TARGET" == "both" ]]; then
  if [[ ("$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE") && ("$FORCE_REINGEST_PATCHES" == "1" || "$FORCE_REINGEST_PATCHES" == "true" || "$FORCE_REINGEST_PATCHES" == "TRUE") ]]; then
    echo "RESUME=1 is incompatible with FORCE_REINGEST_PATCHES=1. Set FORCE_REINGEST_PATCHES=0 to resume Persona patch runs." >&2
    exit 1
  fi
fi

benchmark_tag="$(basename "$BENCHMARK_FILE" .csv)"
selection_suffix="$(persona_selection_suffix)"

for model in "${MODELS[@]}"; do
  safe_model="$(sanitize "$model")"
  COMMON_ARGS=()
  set_common_args "$model"

  echo "============================================================"
  echo "Model: $model"

  if [[ "$RUN_TARGET" == "robust" || "$RUN_TARGET" == "both" ]]; then
    robust_output="$OUTPUT_DIR/persona_robust_${safe_model}_${benchmark_tag}${selection_suffix}.csv"
    robust_cmd=(
      python test_persona_robust.py
      "${COMMON_ARGS[@]}"
      --output "$robust_output"
    )
    ensure_resume_ready "$robust_output" "Persona robust"
    echo "Running robust baseline -> $robust_output"
    "${robust_cmd[@]}"
  fi

  if [[ "$RUN_TARGET" == "patch" || "$RUN_TARGET" == "both" ]]; then
    patch_output="$OUTPUT_DIR/persona_patch_${safe_model}_${benchmark_tag}${selection_suffix}.csv"
    PATCH_ARGS=()
    set_patch_args
    patch_cmd=(
      python test_persona_patch.py
      "${COMMON_ARGS[@]}"
      "${PATCH_ARGS[@]}"
      --output "$patch_output"
    )
    ensure_resume_ready "$patch_output" "Persona patch"
    echo "Running patch variant -> $patch_output"
    "${patch_cmd[@]}"
  fi
done
