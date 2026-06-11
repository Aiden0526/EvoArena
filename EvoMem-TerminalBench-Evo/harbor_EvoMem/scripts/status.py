#!/usr/bin/env python3
"""Report the live status + accuracy of Terminus2 runs.

Reads, for each variant:

* ``runs/full-<variant>/_logs/runner/run_meta.json``           run config
* ``runs/full-<variant>/_logs/runner/dispatcher.{pid,log}``    dispatcher liveness
* ``runs/full-<variant>/_logs/runner/chain.<id>.{exit,log}``   per-chain progress
* ``runs/full-<variant>/<task>__<rand>/result.json``           per-task verdict
* ``runs/full-<variant>/<task>__<rand>/verifier/reward.txt``   raw reward fallback

and prints a compact, terminal-friendly report.

Usage:
    scripts/status.py                       # report on both variants
    scripts/status.py --variant terminus2_evomem
    scripts/status.py --detail              # also list per-task pass/fail
    scripts/status.py --watch               # refresh every 30s (Ctrl-C to stop)
    scripts/status.py --json                # machine-readable

Liveness rules:
* RUNNING       tmux session exists AND dispatcher pid is alive
* STOPPED       tmux session does not exist (and pid is dead)
* INCONSISTENT  one of the above is true but not the other (rare; usually
                means the dispatcher was killed but tmux didn't notice)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_EVO_RE = re.compile(r"^(?P<base>.+?)-EVO-\d+$", re.IGNORECASE)
_V_RE   = re.compile(r"^(?P<base>.+?)-v-[A-Za-z0-9_-]+$")


def _dispatcher_start_epoch(meta: dict) -> Optional[float]:
    """Unix time for ``run_meta.json`` ``started_at`` (this tmux launch)."""

    raw = meta.get("started_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def derive_chain_id(task_name: str, all_dirs: set[str]) -> str:
    m_evo = _EVO_RE.match(task_name)
    if m_evo and m_evo.group("base") in all_dirs:
        return m_evo.group("base")
    m_v = _V_RE.match(task_name)
    if m_v and m_v.group("base") in all_dirs:
        return m_v.group("base")
    return task_name


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrialResult:
    task_name: str
    chain_id: str
    trial_dir: Path
    finished: bool
    passed: Optional[bool]
    reward: Optional[float]
    elapsed_sec: Optional[float]
    exception: Optional[str]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def discover_trials(trials_dir: Path, all_task_names: set[str]) -> list[TrialResult]:
    if not trials_dir.is_dir():
        return []
    out: list[TrialResult] = []
    for child in sorted(trials_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        # Trial dirs look like ``<task_name>__<random>``. We split on the
        # last ``__`` so task names with their own underscores survive.
        if "__" not in child.name:
            continue
        task_name, _, _suffix = child.name.rpartition("__")
        chain_id = derive_chain_id(task_name, all_task_names)

        result_json = _read_json(child / "result.json")
        finished, passed, reward, elapsed, exc = False, None, None, None, None
        if result_json is not None:
            verifier = result_json.get("verifier_result") or {}
            rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
            if isinstance(rewards, dict) and "reward" in rewards:
                try:
                    reward = float(rewards["reward"])
                    passed = reward > 0
                except (TypeError, ValueError):
                    reward = None
            finished = True
            exc_info = result_json.get("exception_info")
            if exc_info:
                # exception_info is typically a dict with a 'message' key.
                if isinstance(exc_info, dict):
                    exc = str(exc_info.get("message") or exc_info.get("type") or exc_info)
                else:
                    exc = str(exc_info)
            try:
                started = result_json.get("started_at")
                finished_at = result_json.get("finished_at")
                if started and finished_at:
                    from datetime import datetime
                    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                    elapsed = (
                        datetime.strptime(finished_at, fmt)
                        - datetime.strptime(started, fmt)
                    ).total_seconds()
            except Exception:
                elapsed = None
        else:
            # Fallback: maybe verifier wrote a reward file even without
            # a complete result.json (very rare; defensive).
            reward_path = child / "verifier" / "reward.txt"
            if reward_path.exists():
                try:
                    reward = float(reward_path.read_text().strip())
                    passed = reward > 0
                    finished = True
                except Exception:
                    pass

        out.append(TrialResult(
            task_name=task_name,
            chain_id=chain_id,
            trial_dir=child,
            finished=finished,
            passed=passed,
            reward=reward,
            elapsed_sec=elapsed,
            exception=exc,
        ))
    return out


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tmux_session_alive(name: str) -> bool:
    if not shutil.which("tmux"):
        return False
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


@dataclasses.dataclass
class VariantStatus:
    variant: str
    trials_dir: Path
    logs_dir: Path
    meta_present: bool
    tmux_alive: bool
    dispatcher_pid: Optional[int]
    dispatcher_alive: bool
    expected_chains: list[str]
    expected_tasks: int          # number of dataset tasks belonging to expected_chains
    expected_tasks_by_chain: dict[str, list[str]]
    chain_exit_codes: dict[str, int]
    running_chains: list[str]
    pending_chains: list[str]
    trials: list[TrialResult]


def _expected_tasks_for_chain_list(dataset: Path, chains: list[str]) -> int:
    """Return the number of dataset tasks that belong to the given chain list."""
    if not dataset.is_dir() or not chains:
        return 0
    all_dirs = {p.name for p in dataset.iterdir() if p.is_dir() and not p.name.startswith(".")}
    chain_set = set(chains)
    n = 0
    for name in all_dirs:
        cid = derive_chain_id(name, all_dirs)
        if cid in chain_set:
            n += 1
    return n


def _expected_tasks_by_chain(dataset: Path, chains: list[str]) -> dict[str, list[str]]:
    if not dataset.is_dir() or not chains:
        return {}
    all_dirs = sorted(
        p.name for p in dataset.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    all_set = set(all_dirs)
    chain_set = set(chains)
    out: dict[str, list[str]] = {chain: [] for chain in chains}
    for name in all_dirs:
        cid = derive_chain_id(name, all_set)
        if cid in chain_set:
            out.setdefault(cid, []).append(name)
    for chain in out:
        out[chain].sort()
    return out


def discover_variant(variant: str, project_root: Path, dataset: Path) -> VariantStatus:
    trials_dir = project_root / "runs" / f"full-{variant}"
    logs_dir = trials_dir / "_logs" / "runner"

    meta_path = logs_dir / "run_meta.json"
    meta = _read_json(meta_path) or {}
    meta_present = bool(meta)

    tmux_session = meta.get("tmux_session", f"harbor-{variant}")
    tmux_alive = _tmux_session_alive(tmux_session)

    dispatcher_pid: Optional[int] = None
    pid_path = logs_dir / "dispatcher.pid"
    if pid_path.exists():
        try:
            dispatcher_pid = int(pid_path.read_text().strip())
        except ValueError:
            dispatcher_pid = None
    dispatcher_alive = bool(dispatcher_pid) and _pid_alive(dispatcher_pid)

    chain_list_path = logs_dir / "chains.txt"
    expected_chains: list[str] = []
    if chain_list_path.exists():
        expected_chains = [
            line.strip() for line in chain_list_path.read_text().splitlines() if line.strip()
        ]

    chain_exit_codes: dict[str, int] = {}
    running_chains: list[str] = []
    launch_epoch = _dispatcher_start_epoch(meta)

    for chain in expected_chains:
        exit_file = logs_dir / f"chain.{chain}.exit"
        trust_exit = False
        if exit_file.exists():
            # Ignore exits from before this launch (kill_runs often leaves 143).
            if launch_epoch is None:
                trust_exit = True
            else:
                try:
                    trust_exit = exit_file.stat().st_mtime + 0.5 >= launch_epoch
                except OSError:
                    trust_exit = True
        if trust_exit:
            try:
                chain_exit_codes[chain] = int(exit_file.read_text().strip())
            except ValueError:
                chain_exit_codes[chain] = -1
            continue
        # Not finished. Is it currently alive?
        pgid_file = logs_dir / f"chain.{chain}.pgid"
        if pgid_file.exists():
            try:
                pgid = int(pgid_file.read_text().strip())
            except ValueError:
                pgid = 0
            try:
                os.kill(-pgid, 0) if pgid else None
                if pgid:
                    running_chains.append(chain)
            except OSError:
                pass

    pending_chains = [
        c for c in expected_chains
        if c not in chain_exit_codes and c not in running_chains
    ]

    # Build a lookup of all task dir names from the dataset to inform
    # chain_id derivation for trials.
    all_dirs: set[str] = set()
    if dataset.is_dir():
        for d in dataset.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                all_dirs.add(d.name)
    trials = discover_trials(trials_dir, all_dirs)

    expected_tasks = _expected_tasks_for_chain_list(dataset, expected_chains)
    expected_tasks_by_chain = _expected_tasks_by_chain(dataset, expected_chains)

    return VariantStatus(
        variant=variant,
        trials_dir=trials_dir,
        logs_dir=logs_dir,
        meta_present=meta_present,
        tmux_alive=tmux_alive,
        dispatcher_pid=dispatcher_pid,
        dispatcher_alive=dispatcher_alive,
        expected_chains=expected_chains,
        expected_tasks=expected_tasks,
        expected_tasks_by_chain=expected_tasks_by_chain,
        chain_exit_codes=chain_exit_codes,
        running_chains=running_chains,
        pending_chains=pending_chains,
        trials=trials,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


PASS_GLYPH = "PASS"
FAIL_GLYPH = "FAIL"
PEND_GLYPH = "----"
RUN_GLYPH  = "..  "
ERR_GLYPH  = "ERR "


def _fmt_pct(n_pass: int, n_done: int) -> str:
    if n_done == 0:
        return "  -  "
    return f"{(100.0 * n_pass / n_done):5.1f}%"


def _summarize_metrics(status: VariantStatus) -> dict[str, dict[str, int | float | str]]:
    trials = status.trials
    by_chain: dict[str, list[TrialResult]] = {}
    by_task: dict[str, TrialResult] = {}
    for t in trials:
        by_chain.setdefault(t.chain_id, []).append(t)
        by_task[t.task_name] = t

    task_done = sum(1 for t in trials if t.finished)
    task_pass = sum(1 for t in trials if t.finished and t.passed)

    chain_done = 0
    chain_pass = 0
    for chain_id in status.expected_chains:
        expected_tasks = status.expected_tasks_by_chain.get(chain_id, [])
        if not expected_tasks:
            continue
        trial_map = {t.task_name: t for t in by_chain.get(chain_id, [])}
        if not all(task_name in trial_map and trial_map[task_name].finished for task_name in expected_tasks):
            continue
        chain_done += 1
        if all(trial_map[task_name].passed is True for task_name in expected_tasks):
            chain_pass += 1

    root_done = 0
    root_pass = 0
    for chain_id in status.expected_chains:
        root_trial = by_task.get(chain_id)
        if root_trial is None or not root_trial.finished:
            continue
        root_done += 1
        if root_trial.passed is True:
            root_pass += 1

    return {
        "tasks": {
            "scope": "all_tasks",
            "passed": task_pass,
            "done": task_done,
            "expected": status.expected_tasks,
        },
        "chains": {
            "scope": "all_tasks_in_chain_must_pass",
            "passed": chain_pass,
            "done": chain_done,
            "expected": len(status.expected_chains),
        },
        "roots": {
            "scope": "prototype_only",
            "passed": root_pass,
            "done": root_done,
            "expected": len(status.expected_chains),
        },
    }


def _format_variant(status: VariantStatus, *, detail: bool) -> str:
    trials = status.trials
    by_chain: dict[str, list[TrialResult]] = {}
    for t in trials:
        by_chain.setdefault(t.chain_id, []).append(t)

    metrics = _summarize_metrics(status)
    task_done = int(metrics["tasks"]["done"])
    task_pass = int(metrics["tasks"]["passed"])
    task_fail = task_done - task_pass

    # Liveness rules:
    #   tmux alive + dispatcher alive   → RUNNING
    #   tmux alive + dispatcher dead    → FINISHED  (pane in remain-on-exit;
    #                                                run completed, user can
    #                                                inspect via tmux attach,
    #                                                kill cleanly via kill_runs.sh)
    #   tmux dead  + dispatcher alive   → INCONSISTENT (rare; manual SIGKILL etc.)
    #   tmux dead  + dispatcher dead + meta_present → STOPPED
    #   no run_meta.json                → NOT-STARTED
    if status.tmux_alive and status.dispatcher_alive:
        live = "RUNNING"
    elif status.tmux_alive and not status.dispatcher_alive:
        live = "FINISHED"
    elif (not status.tmux_alive) and status.dispatcher_alive:
        live = "INCONSISTENT"
    elif status.meta_present:
        live = "STOPPED"
    else:
        live = "NOT-STARTED"

    expected_chains = set(status.expected_chains)
    finished_chains = set(status.chain_exit_codes)
    running_chains = set(status.running_chains)
    pending_chains = expected_chains - finished_chains - running_chains

    lines: list[str] = []
    lines.append(f"╔══ harbor-{status.variant}  [{live}]")
    lines.append(f"║  trials_dir         : {status.trials_dir}")
    if status.dispatcher_pid is not None:
        lines.append(
            f"║  dispatcher_pid     : {status.dispatcher_pid}"
            f"  {'(alive)' if status.dispatcher_alive else '(dead)'}"
        )
    lines.append(
        f"║  chains             : {len(finished_chains)} done"
        f" / {len(running_chains)} running / {len(pending_chains)} pending"
        f"  (expected {len(expected_chains)})"
    )
    lines.append(
        f"║  task accuracy      : {task_done} finished"
        f" / {task_pass} pass / {task_fail} fail"
        f"   accuracy={_fmt_pct(task_pass, task_done)}"
        f"   (expected={metrics['tasks']['expected']})"
    )
    lines.append(
        f"║  chain accuracy     : {metrics['chains']['passed']} pass"
        f" / {metrics['chains']['done']} finished"
        f"   accuracy={_fmt_pct(int(metrics['chains']['passed']), int(metrics['chains']['done']))}"
        f"   (expected={metrics['chains']['expected']})"
    )
    lines.append(
        f"║  root-task accuracy : {metrics['roots']['passed']} pass"
        f" / {metrics['roots']['done']} finished"
        f"   accuracy={_fmt_pct(int(metrics['roots']['passed']), int(metrics['roots']['done']))}"
        f"   (expected={metrics['roots']['expected']})"
    )
    if running_chains:
        head = ", ".join(sorted(running_chains)[:5])
        more = f" (+{len(running_chains)-5} more)" if len(running_chains) > 5 else ""
        lines.append(f"║  currently running  : {head}{more}")
    if pending_chains and len(pending_chains) <= 10:
        lines.append(f"║  pending chains     : {', '.join(sorted(pending_chains))}")
    elif pending_chains:
        head = ", ".join(sorted(pending_chains)[:8])
        lines.append(f"║  pending chains     : {head} (+{len(pending_chains)-8} more)")

    if detail:
        lines.append("║")
        lines.append("║   per-chain accuracy:")
        for chain_id in sorted(set(by_chain) | expected_chains):
            trials_in = by_chain.get(chain_id, [])
            done = sum(1 for t in trials_in if t.finished)
            ok   = sum(1 for t in trials_in if t.finished and t.passed)
            expected_task_names = status.expected_tasks_by_chain.get(chain_id, [])
            total = len(expected_task_names) if expected_task_names else done
            chain_complete = (
                bool(expected_task_names)
                and all(
                    any(t.task_name == task_name and t.finished for t in trials_in)
                    for task_name in expected_task_names
                )
            )
            chain_ok = (
                chain_complete
                and all(
                    any(t.task_name == task_name and t.passed is True for t in trials_in)
                    for task_name in expected_task_names
                )
            )
            run_state = (
                "RUN " if chain_id in running_chains
                else "DONE" if chain_id in finished_chains
                else "WAIT"
            )
            lines.append(
                f"║    [{run_state}]  {chain_id:38s}"
                f"  task {ok:>2}/{done:>2}"
                f"  chain={'PASS' if chain_ok else 'FAIL' if chain_complete else '----'}"
                f"  root={'PASS' if any(t.task_name == chain_id and t.passed is True for t in trials_in) else 'FAIL' if any(t.task_name == chain_id and t.finished for t in trials_in) else '----'}"
                f"  acc={_fmt_pct(ok, done)}"
                f"  exp={total}"
            )
            for t in sorted(trials_in, key=lambda x: x.task_name):
                if t.finished and t.passed:
                    g = PASS_GLYPH
                elif t.finished and t.passed is False:
                    g = FAIL_GLYPH
                elif t.finished and t.passed is None:
                    g = ERR_GLYPH
                else:
                    g = PEND_GLYPH
                rew = f"r={t.reward:.2f}" if t.reward is not None else "r=  ?"
                el  = f"{t.elapsed_sec:.0f}s" if t.elapsed_sec is not None else "    "
                exc = f"  ! {t.exception[:60]}" if t.exception else ""
                lines.append(
                    f"║         {g}  {t.task_name:50s}  {rew}  {el}{exc}"
                )
    lines.append("╚══")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        choices=("terminus2_evomem", "terminus2_baseline"),
        help="Restrict to one variant. Repeat to include multiple. "
             "Default: report on all variants.",
    )
    parser.add_argument("--detail", action="store_true", help="Per-task pass/fail listing.")
    parser.add_argument("--watch", action="store_true", help="Refresh every --interval seconds.")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between --watch refreshes.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="harbor_EvoMem repo root (default: auto-detected).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Terminal-Bench-Evo dataset root (default: ../Terminal-Bench-Evo).",
    )
    args = parser.parse_args()
    if args.dataset is None:
        from harbor_EvoMem.memory_bridge import default_dataset_root

        args.dataset = default_dataset_root()

    variants = args.variant or ["terminus2_evomem", "terminus2_baseline"]

    def render_once() -> str:
        statuses = [discover_variant(v, args.project_root, args.dataset) for v in variants]
        if args.json:
            return json.dumps(
                [
                    {
                        "variant":          s.variant,
                        "trials_dir":       str(s.trials_dir),
                        "tmux_alive":       s.tmux_alive,
                        "dispatcher_alive": s.dispatcher_alive,
                        "dispatcher_pid":   s.dispatcher_pid,
                        "expected_chains":  s.expected_chains,
                        "expected_tasks":   s.expected_tasks,
                        "expected_tasks_by_chain": s.expected_tasks_by_chain,
                        "running_chains":   s.running_chains,
                        "pending_chains":   s.pending_chains,
                        "chain_exit_codes": s.chain_exit_codes,
                        "metrics": _summarize_metrics(s),
                        "n_tasks_finished": sum(1 for t in s.trials if t.finished),
                        "n_tasks_passed":   sum(1 for t in s.trials if t.finished and t.passed),
                        "tasks": [
                            {
                                "task_name":   t.task_name,
                                "chain_id":    t.chain_id,
                                "finished":    t.finished,
                                "passed":      t.passed,
                                "reward":      t.reward,
                                "elapsed_sec": t.elapsed_sec,
                                "exception":   t.exception,
                            }
                            for t in s.trials
                        ],
                    }
                    for s in statuses
                ],
                indent=2,
            )
        chunks = [
            f"as of {time.strftime('%Y-%m-%d %H:%M:%S %Z')}    "
            f"(repo {args.project_root}; dataset {args.dataset})\n"
        ]
        for s in statuses:
            chunks.append(_format_variant(s, detail=args.detail))
        return "\n".join(chunks)

    if not args.watch:
        print(render_once())
        return 0

    try:
        while True:
            os.system("clear")
            print(render_once())
            print(f"\n  refreshing every {args.interval}s — Ctrl-C to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
