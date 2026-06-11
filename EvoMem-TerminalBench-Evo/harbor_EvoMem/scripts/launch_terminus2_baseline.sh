#!/usr/bin/env bash
# scripts/launch_terminus2_baseline.sh — start the Terminus2 **baseline**
# runner over Terminal-shift chains (pure upstream Terminus2 via
# harbor_EvoMem.agents.terminus2_baseline.Terminus2Baseline —
# subclass with no overrides). Runs detached in tmux like launch_terminus2_evomem.sh.
#
# LLM triple is enforced by run_chain + scripts/terminus2_llm.env (see below).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/_ensure_terminus2_llm_env.sh"

REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATASET="${HARBOR_EVOMEM_DATASET:-$REPO_ROOT/Terminal-Bench-Evo}"
NO_SINGLETONS_ARG="--no-singletons"
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--include-singletons" ]; then
        NO_SINGLETONS_ARG=""
    else
        PASSTHROUGH+=("$arg")
    fi
done

EXEC_ARGS=(
    --variant terminus2_baseline
    --dataset "$DATASET"
    --parallel 6
    --max-chains 30
    --start-chain-index 1
    --tmux-session harbor-terminus2-baseline
    --agent-setup-timeout 900
)
[ -n "$NO_SINGLETONS_ARG" ] && EXEC_ARGS+=("$NO_SINGLETONS_ARG")
EXEC_ARGS+=("${PASSTHROUGH[@]+${PASSTHROUGH[@]}}")

exec "$HERE/launch_runs.sh" "${EXEC_ARGS[@]}"
