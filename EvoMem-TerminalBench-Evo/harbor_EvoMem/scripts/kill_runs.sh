#!/usr/bin/env bash
# scripts/kill_runs.sh — clean teardown of a launch_runs.sh deployment.
#
# What gets stopped, in order:
#   1. The tmux session                     (so detached users notice)
#   2. The per-chain process groups         (each chain has its own pgid)
#   3. The dispatcher process group         (xargs + bash subshell pool)
#   4. Any docker containers owned by this run
#
# We start with SIGTERM so Harbor gets a chance to run its
# atexit handlers (which is how ``DockerEnvironment.delete=True`` cleans
# the per-task container). After ``--grace`` seconds we follow up with
# SIGKILL for any straggler.
#
# Usage:
#   scripts/kill_runs.sh --variant terminus2_evomem
#   scripts/kill_runs.sh --variant terminus2_baseline --grace 60
#   scripts/kill_runs.sh --all
#   scripts/kill_runs.sh --variant evomem --no-reap-docker

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 (--variant {terminus2_evomem|terminus2_baseline} | --all) [options]

  --variant …               tear down one variant's runner. ``terminus2-evomem`` → ``terminus2_evomem``.
  --all                     tear down all supported variants (incl. Terminus2 EvoMem + baseline).

  IMPORTANT naming:
    ``--variant terminus2_evomem`` → ``harbor-terminus2-evomem`` (launch_terminus2_evomem.sh).
    ``--variant terminus2_baseline`` → ``harbor-terminus2-baseline``.

  --grace SECONDS           seconds to wait between SIGTERM and SIGKILL (default: 30)
  --trials-dir PATH         override trials dir (defaults to <repo>/runs/full-<variant>)
  --tmux-session NAME       override tmux session name (default: harbor-<variant>)
  --reap-docker             docker cleanup is enabled by default; this flag is kept
                            for compatibility with older invocations
  --no-reap-docker          skip docker cleanup
  --dry-run                 print what would be killed, take no action

The harbor agent records pgids for every chain it runs at:
  <trials-dir>/_logs/runner/{dispatcher,chain.<id>}.pgid
We use those files; if they're missing we still tmux-kill and best-effort
``pkill`` stragglers. Docker cleanup removes Compose task containers whose
project/name matches trial directories under <trials-dir>, plus Terminus2
runtime containers created after this run started.
EOF
}

VARIANTS=()
GRACE=30
TRIALS_DIR=""
TMUX_SESSION=""
REAP_DOCKER=1
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --variant)       VARIANTS+=("$2"); shift 2 ;;
        --all)           VARIANTS=(terminus2_evomem terminus2_baseline); shift ;;
        --grace)         GRACE="$2"; shift 2 ;;
        --trials-dir)    TRIALS_DIR="$2"; shift 2 ;;
        --tmux-session)  TMUX_SESSION="$2"; shift 2 ;;
        --reap-docker)   REAP_DOCKER=1; shift ;;
        --no-reap-docker) REAP_DOCKER=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [ ${#VARIANTS[@]} -eq 0 ]; then
    echo "error: pass --variant or --all" >&2
    usage
    exit 2
fi
if [ ${#VARIANTS[@]} -gt 1 ] && { [ -n "$TRIALS_DIR" ] || [ -n "$TMUX_SESSION" ]; }; then
    echo "error: --trials-dir / --tmux-session don't make sense with multiple variants" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------- helpers --------------------------------------------------
maybe() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] $*"
    else
        "$@" || true
    fi
}

is_alive() {
    # Negative arg means "is any process in this pgid alive?".
    kill -0 "-$1" 2>/dev/null
}

term_pgid() {
    local label="$1" pgid="$2"
    if is_alive "$pgid"; then
        echo "  SIGTERM  $label  pgid=$pgid"
        maybe kill -TERM "-$pgid"
    else
        echo "  (gone)   $label  pgid=$pgid"
    fi
}

kill_pgid() {
    local label="$1" pgid="$2"
    if is_alive "$pgid"; then
        echo "  SIGKILL  $label  pgid=$pgid"
        maybe kill -KILL "-$pgid"
    fi
}

active_runner_pgids() {
    local variant="$1" trials_dir="$2"
    ps -eo pgid=,args= \
        | awk -v variant="$variant" -v trials="$trials_dir" '
            index($0, "scripts/run_chain.py") && index($0, "--variant " variant) && index($0, "--trials-dir " trials) {
                print $1
            }' \
        | sort -n -u
}

run_start_epoch() {
    local logs_dir="$1"
    local meta_file="$logs_dir/run_meta.json"
    local since_iso=""
    if [ -f "$meta_file" ]; then
        since_iso=$(grep -oE '"started_at"\s*:\s*"[^"]+"' "$meta_file" \
                    | head -1 | sed 's/.*"started_at"\s*:\s*"\([^"]*\)".*/\1/')
    fi
    if [ -z "$since_iso" ] && [ -f "$logs_dir/dispatcher.pid" ]; then
        # Fallback: ctime of dispatcher.pid (1s precision is fine).
        since_iso=$(stat -c '%y' "$logs_dir/dispatcher.pid" 2>/dev/null \
                    | awk '{print $1"T"$2}' | cut -c1-19)
    fi
    [ -n "$since_iso" ] || return 1

    local since_epoch
    since_epoch=$(date -d "$since_iso" +%s 2>/dev/null) || return 1
    echo $((since_epoch - 30))
}

reap_docker_containers() {
    local trials_dir="$1" logs_dir="$2"

    if ! command -v docker >/dev/null 2>&1; then
        echo "  (docker not found; skipping docker cleanup)"
        return 0
    fi

    echo "  -- reaping docker containers owned by this run --"

    local since_epoch=""
    if since_epoch=$(run_start_epoch "$logs_dir"); then
        echo "  runtime cutoff: containers created after $(date -d "@$since_epoch" -Iseconds)"
    else
        echo "  (could not determine run start time; only reaping containers matched to trial dirs)"
    fi

    local targets=()
    mapfile -t targets < <(python3 - "$trials_dir" "${since_epoch:-}" <<'PY'
import datetime as dt
import json
import os
import re
import subprocess
import sys

trials_dir = sys.argv[1]
since_epoch = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None

trial_projects = set()
if os.path.isdir(trials_dir):
    for name in os.listdir(trials_dir):
        path = os.path.join(trials_dir, name)
        if "__" in name and os.path.isdir(path):
            trial_projects.add(name.lower())

def created_epoch(value):
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        match = re.match(r"(\d+)(.*)", tail)
        if match:
            text = f"{head}.{match.group(1)[:6]}{match.group(2)}"
    try:
        return int(dt.datetime.fromisoformat(text).timestamp())
    except ValueError:
        return None

try:
    ids_output = subprocess.check_output(
        ["docker", "ps", "-a", "--format", "{{.ID}}"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    sys.exit(0)

ids = [line.strip() for line in ids_output.splitlines() if line.strip()]
if not ids:
    sys.exit(0)

try:
    inspected = subprocess.check_output(
        ["docker", "inspect", *ids],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except subprocess.CalledProcessError:
    sys.exit(0)

for container in json.loads(inspected):
    cid = container.get("Id", "")[:12]
    raw_name = container.get("Name", "").lstrip("/")
    names = {raw_name} if raw_name else set()
    labels = (container.get("Config") or {}).get("Labels") or {}
    project = labels.get("com.docker.compose.project", "").lower()
    created = container.get("Created", "")
    epoch = created_epoch(created)

    reason = ""
    if project and project in trial_projects:
        reason = f"compose project {project} matches trials-dir"
    else:
        for name in names:
            lower = name.lower()
            if any(lower == f"{proj}-main-1" or lower.startswith(f"{proj}-") for proj in trial_projects):
                reason = f"container name {name} matches trials-dir"
                break

    if not reason and raw_name.startswith("harbor-"):
        if since_epoch is not None and epoch is not None and epoch >= since_epoch:
            reason = f"runtime container created after cutoff"

    if reason:
        print(f"{cid}\t{raw_name}\t{created}\t{reason}")
PY
    )

    if [ "${#targets[@]}" -eq 0 ]; then
        echo "  (no run-owned containers; nothing reaped)"
        return 0
    fi

    local line cid name created reason n_killed=0
    for line in "${targets[@]}"; do
        IFS=$'\t' read -r cid name created reason <<<"$line"
        [ -n "$cid" ] || continue
        echo "  docker rm -f $name  ($reason; created $created)"
        maybe docker rm -f "$cid" >/dev/null
        n_killed=$((n_killed + 1))
    done
    if [ "$n_killed" -eq 0 ]; then
        echo "  (no run-owned containers; nothing reaped)"
    fi
}

# ---------- per-variant kill ---------------------------------------

# Session names match ``launch_terminus2_*.sh`` (hyphens): not ``harbor-terminus2_evomem``.
_default_tmux_for_variant() {
    case "$1" in
        terminus2_evomem) echo harbor-terminus2-evomem ;;
        terminus2_baseline) echo harbor-terminus2-baseline ;;
        *) echo "harbor-$1" ;;
    esac
}

kill_variant() {
    local variant="$1"
    local default_trials="$PROJECT_ROOT/runs/full-$variant"
    local trials_dir="${TRIALS_DIR:-$default_trials}"
    local tmux_session="${TMUX_SESSION:-$(_default_tmux_for_variant "$variant")}"
    local logs_dir="$trials_dir/_logs/runner"

    echo "================================================================"
    echo " tearing down harbor-$variant"
    echo "   tmux_session : $tmux_session"
    echo "   trials_dir   : $trials_dir"
    echo "   logs_dir     : $logs_dir"
    echo "================================================================"

    # 1. Kill the tmux session — detaches any attached client and
    # SIGHUPs the foreground process. This is mostly for the user's
    # mental model; the actual process kill happens via pgid below.
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        echo "  tmux kill-session -t $tmux_session"
        maybe tmux kill-session -t "$tmux_session"
    elif { [ "$variant" = terminus2_evomem ] && tmux has-session -t "harbor-terminus2_evomem" 2>/dev/null; }; then
        echo "  tmux kill-session -t harbor-terminus2_evomem (legacy session name)"
        maybe tmux kill-session -t "harbor-terminus2_evomem"
    elif { [ "$variant" = terminus2_baseline ] && tmux has-session -t "harbor-terminus2_baseline" 2>/dev/null; }; then
        echo "  tmux kill-session -t harbor-terminus2_baseline (legacy session name)"
        maybe tmux kill-session -t "harbor-terminus2_baseline"
    else
        echo "  (no tmux session named $tmux_session)"
    fi

    # 2. SIGTERM each chain's pgid, then the dispatcher's.
    if [ -d "$logs_dir" ]; then
        echo "  -- SIGTERM phase --"
        for pgid_file in "$logs_dir"/chain.*.pgid; do
            [ -e "$pgid_file" ] || continue
            local chain pgid
            chain=$(basename "$pgid_file" .pgid)
            pgid=$(cat "$pgid_file" 2>/dev/null || true)
            [ -z "$pgid" ] && continue
            term_pgid "$chain" "$pgid"
        done
        if [ -f "$logs_dir/dispatcher.pid" ]; then
            local dpid
            dpid=$(cat "$logs_dir/dispatcher.pid")
            term_pgid "dispatcher" "$dpid"
        fi
        for pgid in $(active_runner_pgids "$variant" "$trials_dir"); do
            term_pgid "orphan-run_chain" "$pgid"
        done

        # 3. Grace period, then SIGKILL stragglers.
        echo "  -- waiting ${GRACE}s for graceful shutdown --"
        if [ "$DRY_RUN" != "1" ]; then sleep "$GRACE"; fi

        echo "  -- SIGKILL phase --"
        for pgid_file in "$logs_dir"/chain.*.pgid; do
            [ -e "$pgid_file" ] || continue
            local chain pgid
            chain=$(basename "$pgid_file" .pgid)
            pgid=$(cat "$pgid_file" 2>/dev/null || true)
            [ -z "$pgid" ] && continue
            kill_pgid "$chain" "$pgid"
        done
        if [ -f "$logs_dir/dispatcher.pid" ]; then
            local dpid
            dpid=$(cat "$logs_dir/dispatcher.pid")
            kill_pgid "dispatcher" "$dpid"
        fi
        for pgid in $(active_runner_pgids "$variant" "$trials_dir"); do
            kill_pgid "orphan-run_chain" "$pgid"
        done
    else
        echo "  (no logs_dir at $logs_dir; nothing to read pgids from)"
        echo "  -- scanning for orphaned run_chain processes --"
        for pgid in $(active_runner_pgids "$variant" "$trials_dir"); do
            term_pgid "orphan-run_chain" "$pgid"
        done
        echo "  -- waiting ${GRACE}s for graceful shutdown --"
        if [ "$DRY_RUN" != "1" ]; then sleep "$GRACE"; fi
        for pgid in $(active_runner_pgids "$variant" "$trials_dir"); do
            kill_pgid "orphan-run_chain" "$pgid"
        done
    fi

    # 4. Docker cleanup. Harbor task environments are docker-compose
    # containers keyed by the trial directory. Terminus2 LocalRuntime
    # containers do not carry that label, so those are matched by the
    # run start timestamp instead.
    if [ "$REAP_DOCKER" = "1" ]; then
        reap_docker_containers "$trials_dir" "$logs_dir"
    fi

    echo "  done."
    echo
}

for variant in "${VARIANTS[@]}"; do
    orig="$variant"
    case "$variant" in
        terminus2-evomem) variant=terminus2_evomem ;;
        terminus2-baseline) variant=terminus2_baseline ;;
    esac
    if [ "$variant" != "terminus2_evomem" ] \
       && [ "$variant" != "terminus2_baseline" ]; then
        echo "skipping unknown variant: $orig" >&2
        continue
    fi
    kill_variant "$variant"
done
