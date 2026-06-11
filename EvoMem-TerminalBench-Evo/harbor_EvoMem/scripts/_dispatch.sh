#!/usr/bin/env bash
# scripts/_dispatch.sh — internal dispatcher invoked by the tmux launcher.
#
# Reads a list of chain ids (one per line) from $CHAIN_LIST_FILE, runs
# them through ``run_chain.py`` with concurrency $PARALLEL.
#
# Each chain runs as its OWN session leader (``setsid``), so the chain
# can be killed cleanly by sending a signal to its process group ID.
#
# Required env (all set by launch_runs.sh):
#   PROJECT_ROOT           absolute path to the harbor_EvoMem repo
#   DATASET                path to Terminal-Bench-Evo dataset
#   VARIANT                "terminus2_evomem" or "terminus2_baseline"
#   TRIALS_DIR             where harbor writes trial outputs
#   LOGS_DIR               $TRIALS_DIR/_logs/runner
#   CHAIN_LIST_FILE        text file: one chain id per line
#   PARALLEL               int; max chains running concurrently
#   HARBOR_BIN             path to the harbor CLI
#   PYTHON_BIN             python interpreter that has harbor_EvoMem installed
#
# Optional env:
#   STOP_ON_FAILURE        "1" → run_chain.py stops a chain after a task
#                          fails. Default "" → keep going (so the rest of
#                          the chain still gets memory).
#   AGENT_SETUP_TIMEOUT    forwarded to Harbor as --agent-setup-timeout.
#   EXTRA_RUN_CHAIN_ARGS   forwarded verbatim to run_chain.py (string)
#
# Side effects:
#   * writes $LOGS_DIR/dispatcher.pid    (this dispatcher's own pid)
#   * writes $LOGS_DIR/dispatcher.log    (mirror of stdout/stderr)
#   * per chain id ``X`` written to $LOGS_DIR/:
#       chain.X.log    full stdout/stderr from ``run_chain.py``
#       chain.X.pgid   pgid of the chain's process group (for kill)
#       chain.X.exit   numeric exit code once the chain has finished

set -uo pipefail

mkdir -p "$LOGS_DIR"

DISPATCHER_LOG="$LOGS_DIR/dispatcher.log"
exec > >(tee -a "$DISPATCHER_LOG") 2>&1

echo "=================================================================="
echo " harbor-${VARIANT} dispatcher starting"
echo "  project_root : $PROJECT_ROOT"
echo "  dataset      : $DATASET"
echo "  variant      : $VARIANT"
echo "  trials_dir   : $TRIALS_DIR"
echo "  parallel     : $PARALLEL"
echo "  chains       : $(wc -l < "$CHAIN_LIST_FILE") (from $CHAIN_LIST_FILE)"
echo "  started_at   : $(date -Iseconds)"
echo "=================================================================="

echo $$ > "$LOGS_DIR/dispatcher.pid"

# Drop stale per-chain exit markers from *previous* launches. status.py
# treats ``chain.<id>.exit`` as authoritative; without this, a new run
# still looks "DONE" (often exit=143) until the chain worker overwrites
# the file at the end — which can be hours later.
while IFS= read -r _c; do
    [[ -z "${_c//[[:space:]]/}" ]] && continue
    rm -f "$LOGS_DIR/chain.${_c}.exit"
done < "$CHAIN_LIST_FILE"

# Propagate signals to every descendant. Chains run with ``setsid`` so
# they live in their own process groups — ``kill 0`` (our own pgid)
# alone wouldn't reach them, so we also iterate over the recorded
# per-chain pgid files and forward SIGTERM to each.
shutdown() {
    local sig="${1:-TERM}"
    echo "[dispatcher] received SIG${sig} at $(date -Iseconds) — propagating"
    for pgid_file in "$LOGS_DIR"/chain.*.pgid; do
        [ -e "$pgid_file" ] || continue
        local pgid
        pgid=$(cat "$pgid_file" 2>/dev/null)
        if [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null; then
            echo "  → SIG${sig} pgid=$pgid ($(basename "$pgid_file" .pgid))"
            kill "-$sig" "-$pgid" 2>/dev/null || true
        fi
    done
    # And finally tear down our own pgid (xargs + bash subshell pool).
    kill "-$sig" 0 2>/dev/null || true
    exit 143
}
trap 'shutdown TERM' INT TERM

# ---- worker --------------------------------------------------------
# Runs one chain inside its own process group so the kill script can
# target it precisely. Closure-captures the run-time env via export.
run_one_chain() {
    local chain="$1"
    local chain_log="$LOGS_DIR/chain.${chain}.log"
    local chain_pgid_file="$LOGS_DIR/chain.${chain}.pgid"
    local chain_exit_file="$LOGS_DIR/chain.${chain}.exit"

    rm -f "$chain_exit_file"

    : > "$chain_log"
    {
        echo "[chain $chain] started_at=$(date -Iseconds) variant=$VARIANT"
    } >> "$chain_log"

    # Build run_chain.py argv as an array (no eval, no quoting horrors).
    local -a run_chain_argv=(
        "$PYTHON_BIN" scripts/run_chain.py
        --chain "$chain"
        --variant "$VARIANT"
        --dataset "$DATASET"
        --trials-dir "$TRIALS_DIR"
        --harbor-bin "$HARBOR_BIN"
    )
    if [ -z "${STOP_ON_FAILURE:-}" ]; then
        run_chain_argv+=(--no-stop-on-failure)
    fi
    if [ -n "${AGENT_SETUP_TIMEOUT:-}" ]; then
        run_chain_argv+=(-- --agent-setup-timeout "$AGENT_SETUP_TIMEOUT")
    fi
    if [ -n "${EXTRA_RUN_CHAIN_ARGS:-}" ]; then
        # Word-splitting on whitespace is intentional here; users pass
        # extra args as a single space-separated string from the
        # launcher. There's no quoted-string nesting required because
        # all our --ae / --kwarg values are simple key=value pairs.
        # shellcheck disable=SC2206
        local extra=( ${EXTRA_RUN_CHAIN_ARGS} )
        run_chain_argv+=( "${extra[@]}" )
    fi

    cd "$PROJECT_ROOT"

    # ``setsid`` makes the child its own session leader, so its pid IS
    # its pgid; we record that pgid as soon as the child starts, then
    # wait on it. ``setsid -f`` would background, but we want to be
    # able to ``wait`` so xargs respects the worker slot — so use the
    # foreground form and capture pid via ``$!`` once we put it in the
    # background ourselves.
    setsid -- "${run_chain_argv[@]}" >> "$chain_log" 2>&1 &
    local pid=$!
    # The child's pid IS its pgid because setsid made it a session
    # leader. Record both names so kill_runs.sh can be liberal.
    echo "$pid" > "$chain_pgid_file"
    echo "[chain $chain] pgid=$pid" >> "$chain_log"

    wait "$pid"
    local rc=$?
    echo "$rc" > "$chain_exit_file"
    echo "[chain $chain] finished_at=$(date -Iseconds) exit=$rc" >> "$chain_log"
    return "$rc"
}

export -f run_one_chain
export PROJECT_ROOT DATASET VARIANT TRIALS_DIR LOGS_DIR HARBOR_BIN PYTHON_BIN \
       STOP_ON_FAILURE AGENT_SETUP_TIMEOUT EXTRA_RUN_CHAIN_ARGS

# ``xargs -P`` keeps a stable pool of $PARALLEL workers. The trick is
# that we run xargs *backgrounded* and then ``wait`` on it explicitly —
# this lets the bash trap interrupt the wait if the user kills us.
xargs -a "$CHAIN_LIST_FILE" -P "$PARALLEL" -I {} \
    bash -c 'run_one_chain "$1"' _ {} &
XARGS_PID=$!

wait "$XARGS_PID"
RC=$?

echo "=================================================================="
echo " harbor-${VARIANT} dispatcher finished_at=$(date -Iseconds) exit=$RC"
echo "=================================================================="
exit "$RC"
