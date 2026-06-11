#!/usr/bin/env bash
# scripts/launch_runs.sh — start a detached tmux session that drives
# Harbor agents [+ EvoMem] across an entire Terminal-shift
# dataset.
#
# Sequential within a chain (so EvoMem accumulates across the
# prototype + variants), parallel across chains. Survives logout
# (everything lives inside ``tmux``). Killable cleanly via
# ``scripts/kill_runs.sh``.
#
# Usage:
#   scripts/launch_runs.sh --variant terminus2_evomem
#   scripts/launch_runs.sh --variant terminus2_baseline --parallel 3
#   scripts/launch_runs.sh --variant terminus2_evomem --chains 'bn-fit-modify adaptive-rejection-sampler'
#   scripts/launch_runs.sh --variant terminus2_evomem --no-singletons --parallel 4 --trials-dir /custom/path
#   scripts/launch_runs.sh --variant terminus2_evomem --max-chains 30
#
# Convenience wrappers exist for both Terminus2 variants.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --variant {terminus2_evomem|terminus2_baseline} [options]

Required:
  --variant {terminus2_evomem|terminus2_baseline}
                              which agent to use

Options:
  --parallel N                max concurrent chains (default: 3)
  --start-chain-index N       start from the N-th chain (1-based) in the
                              generated chain list (default: 1)
  --start-chain-id ID         start from the first occurrence of chain ID in
                              the generated chain list
  --max-chains N              keep only the first N chains after list generation
                              (default: all)
  --dataset PATH              Terminal-Bench-Evo task root (default: <repo>/Terminal-Bench-Evo)
  --trials-dir PATH           where Harbor writes trials (default: <repo>/runs/full-<variant>)
  --chains "a b c"            run only these chain ids (default: all auto-discovered)
  --chain-list FILE           read chain ids from FILE (one per line, # for comments)
  --no-singletons             skip chains that have only the prototype
  --tmux-session NAME         tmux session name (default: harbor-<variant>)
  --stop-on-failure           after a failing task, abort the rest of the chain
  --agent-setup-timeout SEC   Harbor agent setup timeout in seconds
                              (default: unset; Terminus2 wrappers set 900)
  --extra-run-chain-args STR  free-form string forwarded verbatim to run_chain.py
  --dry-run                   print what we would do, don't actually start tmux

Environment knobs:
  HARBOR_BIN                  path to harbor CLI (default: first harbor on PATH)
  PYTHON_BIN                  python that has harbor_EvoMem installed
                              (default: first python3 on PATH)
EOF
}

# ---------- defaults / arg parsing ----------------------------------
VARIANT=""
PARALLEL=3
START_CHAIN_INDEX=1
START_CHAIN_ID=""
MAX_CHAINS=0
DATASET=""
TRIALS_DIR=""
CHAINS_INLINE=""
CHAIN_LIST=""
NO_SINGLETONS=0
TMUX_SESSION=""
STOP_ON_FAILURE=""
AGENT_SETUP_TIMEOUT=""
EXTRA_RUN_CHAIN_ARGS=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --variant)              VARIANT="$2"; shift 2 ;;
        --parallel)             PARALLEL="$2"; shift 2 ;;
        --start-chain-index)    START_CHAIN_INDEX="$2"; shift 2 ;;
        --start-chain-id)       START_CHAIN_ID="$2"; shift 2 ;;
        --max-chains)           MAX_CHAINS="$2"; shift 2 ;;
        --dataset)              DATASET="$2"; shift 2 ;;
        --trials-dir)           TRIALS_DIR="$2"; shift 2 ;;
        --chains)               CHAINS_INLINE="$2"; shift 2 ;;
        --chain-list)           CHAIN_LIST="$2"; shift 2 ;;
        --no-singletons)        NO_SINGLETONS=1; shift ;;
        --tmux-session)         TMUX_SESSION="$2"; shift 2 ;;
        --stop-on-failure)      STOP_ON_FAILURE="1"; shift ;;
        --agent-setup-timeout)  AGENT_SETUP_TIMEOUT="$2"; shift 2 ;;
        --extra-run-chain-args) EXTRA_RUN_CHAIN_ARGS="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=1; shift ;;
        -h|--help)              usage; exit 0 ;;
        *)                      echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "error: --variant is required" >&2
    usage
    exit 2
fi
if ! [[ "$START_CHAIN_INDEX" =~ ^[0-9]+$ ]] || [ "$START_CHAIN_INDEX" -lt 1 ]; then
    echo "error: --start-chain-index must be an integer >= 1" >&2
    exit 2
fi
if [ -n "$START_CHAIN_ID" ] && [ "$START_CHAIN_INDEX" -ne 1 ]; then
    echo "error: use either --start-chain-index or --start-chain-id, not both" >&2
    exit 2
fi
if [ -n "$AGENT_SETUP_TIMEOUT" ]; then
    if ! [[ "$AGENT_SETUP_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "error: --agent-setup-timeout must be numeric seconds" >&2
        exit 2
    fi
fi
case "$VARIANT" in
    terminus2_evomem|terminus2_baseline) ;;
    *)
        echo "error: unsupported --variant '$VARIANT'" >&2
        exit 2
        ;;
esac

# ---------- resolve paths -------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
[ -z "$DATASET" ] && DATASET="$REPO_ROOT/Terminal-Bench-Evo"
HARBOR_BIN="${HARBOR_BIN:-$(command -v harbor || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

if [ -z "$HARBOR_BIN" ] || [ ! -x "$HARBOR_BIN" ]; then
    echo "error: HARBOR_BIN=${HARBOR_BIN:-<unset>} not executable; install Harbor or set HARBOR_BIN" >&2
    exit 2
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "error: PYTHON_BIN=${PYTHON_BIN:-<unset>} not executable; set PYTHON_BIN to the python used for pip install -e ." >&2
    exit 2
fi
if [ ! -d "$DATASET" ]; then
    echo "error: --dataset $DATASET not a directory" >&2
    exit 2
fi
if [ "$DRY_RUN" = "0" ] && ! command -v tmux >/dev/null 2>&1; then
    echo "error: tmux is not installed" >&2
    exit 2
fi

[ -z "$TRIALS_DIR" ]   && TRIALS_DIR="$PROJECT_ROOT/runs/full-$VARIANT"
if [ -z "$TMUX_SESSION" ]; then
    case "$VARIANT" in
        terminus2_evomem)    TMUX_SESSION=harbor-terminus2-evomem ;;
        terminus2_baseline)  TMUX_SESSION=harbor-terminus2-baseline ;;
    esac
fi
LOGS_DIR="$TRIALS_DIR/_logs/runner"
mkdir -p "$LOGS_DIR"

active_runner_rows() {
    ps -eo pid=,pgid=,args= \
        | awk -v variant="$VARIANT" -v trials="$TRIALS_DIR" '
            index($0, "scripts/run_chain.py") && index($0, "--variant " variant) && index($0, "--trials-dir " trials) {
                print
            }'
}

# ---------- build chain list ----------------------------------------
CHAIN_LIST_FILE="$LOGS_DIR/chains.txt"
if [ -n "$CHAINS_INLINE" ]; then
    : > "$CHAIN_LIST_FILE"
    for c in $CHAINS_INLINE; do echo "$c" >> "$CHAIN_LIST_FILE"; done
elif [ -n "$CHAIN_LIST" ]; then
    grep -vE '^\s*(#|$)' "$CHAIN_LIST" > "$CHAIN_LIST_FILE"
else
    # Auto-discover.
    LIST_ARGS=( --dataset "$DATASET" --format names )
    [ "$NO_SINGLETONS" = "1" ] && LIST_ARGS+=( --no-singletons )
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/list_chains.py" "${LIST_ARGS[@]}" \
        > "$CHAIN_LIST_FILE"
fi

N_CHAINS=$(wc -l < "$CHAIN_LIST_FILE" | awk '{print $1}')
if [ "$N_CHAINS" -eq 0 ]; then
    echo "error: no chains to run (chain list is empty)" >&2
    exit 2
fi

# Optional start offset: either by chain id marker or by 1-based index.
if [ -n "$START_CHAIN_ID" ]; then
    START_LINE=$(awk -v target="$START_CHAIN_ID" '$0==target {print NR; exit}' "$CHAIN_LIST_FILE")
    if [ -z "$START_LINE" ]; then
        echo "error: --start-chain-id '$START_CHAIN_ID' not found in chain list" >&2
        exit 2
    fi
    START_CHAIN_INDEX="$START_LINE"
fi

if [ "$START_CHAIN_INDEX" -gt "$N_CHAINS" ]; then
    echo "error: --start-chain-index $START_CHAIN_INDEX exceeds chain count $N_CHAINS" >&2
    exit 2
fi
if [ "$START_CHAIN_INDEX" -gt 1 ]; then
    tmp_file="${CHAIN_LIST_FILE}.tmp"
    sed -n "${START_CHAIN_INDEX},\$p" "$CHAIN_LIST_FILE" > "$tmp_file"
    mv "$tmp_file" "$CHAIN_LIST_FILE"
    N_CHAINS=$(wc -l < "$CHAIN_LIST_FILE" | awk '{print $1}')
fi

if [ "$MAX_CHAINS" -gt 0 ] && [ "$N_CHAINS" -gt "$MAX_CHAINS" ]; then
    tmp_file="${CHAIN_LIST_FILE}.tmp"
    sed -n "1,${MAX_CHAINS}p" "$CHAIN_LIST_FILE" > "$tmp_file"
    mv "$tmp_file" "$CHAIN_LIST_FILE"
    N_CHAINS="$MAX_CHAINS"
fi

# ---------- abort if a session is already running -------------------
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "error: tmux session '$TMUX_SESSION' is already running." >&2
    echo "       run scripts/kill_runs.sh --variant $VARIANT first, or pick a different --tmux-session." >&2
    exit 2
fi
ACTIVE_RUNNERS="$(active_runner_rows)"
if [ -n "$ACTIVE_RUNNERS" ]; then
    echo "error: found existing chain runner(s) for variant '$VARIANT' writing to:" >&2
    echo "       $TRIALS_DIR" >&2
    echo "$ACTIVE_RUNNERS" | sed 's/^/       /' >&2
    echo "       run scripts/kill_runs.sh --variant $VARIANT, or use a fresh --trials-dir." >&2
    exit 2
fi

# ---------- record run metadata so status / kill can find us --------
META_FILE="$LOGS_DIR/run_meta.json"
"$PYTHON_BIN" - <<PY > "$META_FILE"
import json, os, time
print(json.dumps({
    "variant":          "$VARIANT",
    "tmux_session":     "$TMUX_SESSION",
    "project_root":     "$PROJECT_ROOT",
    "dataset":          "$DATASET",
    "trials_dir":       "$TRIALS_DIR",
    "logs_dir":         "$LOGS_DIR",
    "chain_list_file":  "$CHAIN_LIST_FILE",
    "parallel":         int("$PARALLEL"),
    "n_chains":         int("$N_CHAINS"),
    "stop_on_failure":  bool("$STOP_ON_FAILURE"),
    "agent_setup_timeout": "$AGENT_SETUP_TIMEOUT" or None,
    "harbor_bin":       "$HARBOR_BIN",
    "python_bin":       "$PYTHON_BIN",
    "started_at":       time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "started_by":       os.environ.get("USER", "?"),
}, indent=2))
PY

# Also save into the canonical run-meta location at the top of TRIALS_DIR
# so status.py can locate runs even when LOGS_DIR is renamed.
mkdir -p "$TRIALS_DIR/_logs"
cp "$META_FILE" "$TRIALS_DIR/_logs/run_meta.json" 2>/dev/null || true

cat <<MSG
================================================================
 launching harbor-$VARIANT runner
   tmux_session    : $TMUX_SESSION
   variant         : $VARIANT
   parallel        : $PARALLEL chains at a time
   chains          : $N_CHAINS  (from $CHAIN_LIST_FILE)
   trials_dir      : $TRIALS_DIR
   logs_dir        : $LOGS_DIR
   harbor_bin      : $HARBOR_BIN
   python_bin      : $PYTHON_BIN
   stop_on_failure : ${STOP_ON_FAILURE:-no}
   setup_timeout   : ${AGENT_SETUP_TIMEOUT:-default}
================================================================
MSG

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would launch tmux new-session -d -s $TMUX_SESSION ..."
    echo "[dry-run] dispatcher cmd: setsid $PROJECT_ROOT/scripts/_dispatch.sh"
    echo "[dry-run] first 5 chains: $(head -5 "$CHAIN_LIST_FILE" | tr '\n' ' ')"
    exit 0
fi

# ---------- launch tmux session -------------------------------------
# Detached tmux session whose only window runs the dispatcher.
#
# Note: do NOT wrap the dispatcher in ``setsid`` here. tmux's pane is
# itself attached to a pty/session; ``setsid`` would detach the process
# from that pty, tmux's pane would go into "dead" status immediately,
# and the dispatcher would never start. The chain-level ``setsid``
# inside _dispatch.sh is sufficient for clean kill targeting (each
# chain runs in its own session/pgid).
#
# We also turn on ``remain-on-exit`` so if the dispatcher crashes
# during startup, the pane stays around long enough for the user to
# attach and see the error.
TMUX_CMD=$(cat <<EOF
exec env \
    PROJECT_ROOT='$PROJECT_ROOT' \
    DATASET='$DATASET' \
    VARIANT='$VARIANT' \
    TRIALS_DIR='$TRIALS_DIR' \
    LOGS_DIR='$LOGS_DIR' \
    CHAIN_LIST_FILE='$CHAIN_LIST_FILE' \
    PARALLEL='$PARALLEL' \
    HARBOR_BIN='$HARBOR_BIN' \
    PYTHON_BIN='$PYTHON_BIN' \
    STOP_ON_FAILURE='$STOP_ON_FAILURE' \
    AGENT_SETUP_TIMEOUT='$AGENT_SETUP_TIMEOUT' \
    EXTRA_RUN_CHAIN_ARGS='$EXTRA_RUN_CHAIN_ARGS' \
    bash '$PROJECT_ROOT/scripts/_dispatch.sh'
EOF
)

tmux new-session -d -s "$TMUX_SESSION" -n dispatcher "bash -lc \"$TMUX_CMD\""
tmux set-option -t "$TMUX_SESSION" remain-on-exit on 2>/dev/null || true

# Give the dispatcher a moment to write its pid file, so users have a
# meaningful target to kill.
for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "$LOGS_DIR/dispatcher.pid" ] && break
    sleep 0.5
done

if [ -f "$LOGS_DIR/dispatcher.pid" ]; then
    DPID=$(cat "$LOGS_DIR/dispatcher.pid")
    echo "dispatcher pid=$DPID (also pgid; lives at $LOGS_DIR/dispatcher.pid)"
else
    echo "warn: dispatcher pid file did not appear yet at $LOGS_DIR/dispatcher.pid"
    echo "      check 'tmux attach -t $TMUX_SESSION' to see startup output."
fi

cat <<MSG

now running detached. Useful follow-ups:
  attach (read-only OK):  tmux attach -t $TMUX_SESSION
  detach when attached:   Ctrl-b d
  watch dispatcher log:   tail -f $LOGS_DIR/dispatcher.log
  watch one chain:        tail -f $LOGS_DIR/chain.<chain_id>.log
  status across both:     scripts/status.py
  kill cleanly:           scripts/kill_runs.sh --variant $VARIANT

This shell exits now — that is normal. Chains/harbour trials run inside tmux until done.

MSG
