#!/usr/bin/env bash
# Source before launching. Validates the local LLM config file.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_fixed="${HARBOR_EVOMEM_LLM_ENV:-$_here/terminus2_llm.env}"
if [[ ! -f "$_fixed" ]]; then
  echo "error: required LLM env file missing: $_fixed" >&2
  echo "       copy $_here/terminus2_llm.env.example to $_here/terminus2_llm.env and fill it." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
# shellcheck disable=SC1091
source "$_fixed"
set +a

if [[ -z "${LLM_MODEL:-}" ]] || [[ -z "${LLM_API_KEY:-}" ]] || [[ -z "${LLM_BASE_URL:-}" ]]; then
  echo "error: $_fixed must set LLM_MODEL, LLM_API_KEY, and LLM_BASE_URL" >&2
  exit 1
fi
if [[ "$LLM_API_KEY" == REPLACE_WITH_* ]] || [[ "$LLM_MODEL" == REPLACE_WITH_* ]] || [[ "$LLM_BASE_URL" == REPLACE_WITH_* ]]; then
  echo "error: $_fixed still contains placeholder LLM values" >&2
  exit 1
fi
