#!/usr/bin/env python3
"""Run one or more Terminal-shift chains through Terminus2 agents.

A "chain" is a directory layout like::

    <dataset_root>/<chain_id>/                    (prototype)
    <dataset_root>/<chain_id>-EVO-1/              (variant)
    <dataset_root>/<chain_id>-EVO-2/
    <dataset_root>/<chain_id>-v-rocker/           (or -v-* style variants)

For EvoMem to work, the prototype must run *before* its variants
and successive runs must observe each other's records. This script
enforces sequential order **inside** a chain and lets multiple chains
run in parallel via a process pool.

Each task is launched by spawning ``harbor trials start`` as a
subprocess, so the chain runner is decoupled from any in-process
Harbor state.

Examples
--------

Single chain, with EvoMem::

    python scripts/run_chain.py \
        --dataset ../Terminal-Bench-Evo \
        --chain bn-fit-modify \
        --variant terminus2_evomem

Same chain, baseline ablation (no EvoMem)::

    python scripts/run_chain.py \
        --dataset ../Terminal-Bench-Evo \
        --chain bn-fit-modify \
        --variant terminus2_baseline

Multiple chains in parallel::

    python scripts/run_chain.py \
        --dataset ../Terminal-Bench-Evo \
        --chain bn-fit-modify \
        --chain adaptive-rejection-sampler \
        --variant terminus2_evomem \
        --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Map agent variant -> (Harbor --agent-import-path, friendly tag)
VARIANTS = {
    "terminus2_evomem": "harbor_EvoMem.agents.terminus2_evomem:Terminus2EvoMem",
    "terminus2_baseline": "harbor_EvoMem.agents.terminus2_baseline:Terminus2Baseline",
}

# Suffixes recognised as "this is a chain variant of <prefix>". Mirrors
# harbor_EvoMem.chain_id but expressed for filesystem discovery.
_EVO_RE = re.compile(r"^(?P<base>.+?)-EVO-\d+$", re.IGNORECASE)
# Variant token after ``-v-`` may include hyphens (``-v-registers-xyz``).
_V_RE = re.compile(r"^(?P<base>.+?)-v-[A-Za-z0-9_-]+$")


def _uses_evomem(variant: str) -> bool:
    return variant == "terminus2_evomem"


def _has_kwarg(entries: list[str], key: str) -> bool:
    prefix = f"{key}="
    return any(entry.startswith(prefix) for entry in entries)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _read_changed_files(path: Path) -> list[str]:
    return [line.strip() for line in _read_text(path).splitlines() if line.strip()]


def _reward_is_success(trial_dir: Path) -> bool:
    reward = _read_text(trial_dir / "verifier" / "reward.txt").strip()
    try:
        return float(reward) > 0
    except ValueError:
        return False


def _latest_trial_dir(trials_dir: Path, task_name: str) -> Optional[Path]:
    candidates = [
        child
        for child in trials_dir.glob(f"{task_name}__*")
        if child.is_dir() and (child / "agent").is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _has_validated_task_memory(paths: Any, task_name: str) -> bool:
    from harbor_EvoMem import family_trace_memory

    return any(
        str(record.get("task_name") or "") == task_name
        and record.get("status") == "validated"
        for record in family_trace_memory.load_task_memories(paths)
    )


def seed_validated_memory_from_trial(
    *,
    chain_id: str,
    task_dir: Path,
    trial_dir: Path,
    trials_dir: Path,
) -> dict[str, Any]:
    """Backfill compact family memory after the verifier has produced reward=1."""

    if not _reward_is_success(trial_dir):
        return {"seeded": False, "reason": "reward_not_success", "trial_dir": str(trial_dir)}

    from harbor_EvoMem import family_trace_memory, memory_bridge

    host_root = trials_dir / "_evomem_store"
    paths = family_trace_memory.FamilyPaths(chain_id, host_root)
    if _has_validated_task_memory(paths, task_dir.name):
        return {"seeded": False, "reason": "validated_record_exists", "trial_dir": str(trial_dir)}

    instruction = _read_text(task_dir / "instruction.md")
    history = memory_bridge.load_terminus2_history(trial_dir / "agent" / "trajectory.json")
    diff_text = _read_text(trial_dir / "agent" / "harbor_evomem" / "diff.patch")
    changed_files = _read_changed_files(trial_dir / "agent" / "harbor_evomem" / "changed_files.txt")
    terminal_trace = family_trace_memory.build_terminal_trace(history)
    probe_path = trial_dir / "agent" / "harbor_evomem" / "terminal_family_probe.json"
    try:
        probe = json.loads(_read_text(probe_path)) if probe_path.is_file() else {}
    except json.JSONDecodeError:
        probe = {}

    prev_task_name, prev_instruction = family_trace_memory.previous_release_snapshot_task(
        paths,
        task_dir.name,
    )
    family_trace_memory.persist_instruction_snapshot(paths, task_dir.name, instruction)
    family_trace_memory.update_family_state(
        paths,
        task_dir.name,
        family_trace_memory.probe_summary_md(probe),
    )
    task_record, patch_record = family_trace_memory.store_family_memory(
        paths,
        task_name=task_dir.name,
        probe=probe,
        prev_task_name=prev_task_name,
        prev_instruction_text=prev_instruction,
        instruction_text=instruction,
        history=history,
        diff_text=diff_text,
        changed_files=changed_files,
        validation_passed=True,
        file_snapshots={},
        terminal_trace=terminal_trace,
    )
    return {
        "seeded": True,
        "trial_dir": str(trial_dir),
        "task_record": task_record.get("record_id") if isinstance(task_record, dict) else None,
        "patch_record": patch_record.get("record_id") if isinstance(patch_record, dict) else None,
    }


@dataclass
class TaskRun:
    chain_id: str
    task_name: str
    task_dir: Path
    trials_dir: Path
    log_path: Path
    cmd: list[str]
    env: dict[str, str]


def discover_chain_tasks(dataset: Path, chain_id: str) -> list[Path]:
    """Find prototype + variants for ``chain_id`` and return them in order."""
    prototype = dataset / chain_id
    if not prototype.is_dir():
        raise FileNotFoundError(f"Prototype task not found: {prototype}")

    variants: list[Path] = []
    for child in sorted(dataset.iterdir()):
        if not child.is_dir() or child == prototype:
            continue
        m_evo = _EVO_RE.match(child.name)
        m_v = _V_RE.match(child.name)
        if m_evo and m_evo.group("base") == chain_id:
            variants.append(child)
        elif m_v and m_v.group("base") == chain_id:
            variants.append(child)

    # Sort variants lexicographically; works for both -EVO-N (1..4) and
    # -v-<token>. Prototype always comes first.
    variants.sort(key=lambda p: p.name)
    return [prototype, *variants]


def build_run(
    chain_id: str,
    task_dir: Path,
    *,
    variant: str,
    trials_dir: Path,
    model: Optional[str],
    extra_env: list[str],
    extra_kwargs: list[str],
    extra_args: list[str],
    harbor_bin: str,
) -> TaskRun:
    task_name = task_dir.name
    log_path = trials_dir / "_logs" / f"{task_name}.harbor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    extra_kwargs = list(extra_kwargs)

    if _uses_evomem(variant) and not _has_kwarg(extra_kwargs, "host_root"):
        evomem_host_root = trials_dir / "_evomem_store"
        evomem_host_root.mkdir(parents=True, exist_ok=True)
        extra_kwargs.append(f"host_root={evomem_host_root}")

    cmd: list[str] = [
        harbor_bin,
        "trials",
        "start",
        "--path", str(task_dir),
        "--agent-import-path", VARIANTS[variant],
        "--trials-dir", str(trials_dir),
        "--ae", f"HARBOR_EVOMEM_CHAIN_ID={chain_id}",
    ]
    if not model or not str(model).strip():
        raise RuntimeError("internal error: resolved LLM_MODEL missing; run_chain.main must set args.model")
    cmd += ["--model", model]
    for entry in extra_env:
        cmd += ["--ae", entry]
    for entry in extra_kwargs:
        cmd += ["--agent-kwarg", entry]
    cmd += list(extra_args)

    env = os.environ.copy()
    env.setdefault("HARBOR_EVOMEM_CHAIN_ID", chain_id)

    return TaskRun(
        chain_id=chain_id,
        task_name=task_name,
        task_dir=task_dir,
        trials_dir=trials_dir,
        log_path=log_path,
        cmd=cmd,
        env=env,
    )


def run_one_task(run: TaskRun, *, dry_run: bool = False) -> dict:
    """Execute a single task (subprocess). Returns a status dict."""
    started = time.time()
    if dry_run:
        return {
            "chain_id": run.chain_id,
            "task_name": run.task_name,
            "skipped": True,
            "cmd": run.cmd,
            "elapsed_sec": 0.0,
        }
    with run.log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# cwd={os.getcwd()}\n# cmd={run.cmd}\n\n")
        logf.flush()
        proc = subprocess.run(
            run.cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=run.env,
            check=False,
        )
    elapsed = time.time() - started
    return {
        "chain_id": run.chain_id,
        "task_name": run.task_name,
        "exit_code": proc.returncode,
        "log_path": str(run.log_path),
        "elapsed_sec": round(elapsed, 1),
    }


def run_chain(
    chain_id: str,
    dataset: Path,
    *,
    variant: str,
    trials_dir: Path,
    model: Optional[str],
    extra_env: list[str],
    extra_kwargs: list[str],
    extra_args: list[str],
    harbor_bin: str,
    stop_on_failure: bool,
    dry_run: bool,
    start_index: int = 1,
    max_tasks: int = 0,
) -> list[dict]:
    tasks = discover_chain_tasks(dataset, chain_id)
    if start_index < 1:
        raise ValueError("start_index must be >= 1")
    selected_tasks = tasks[start_index - 1 :]
    if max_tasks > 0:
        selected_tasks = selected_tasks[:max_tasks]
    if not selected_tasks:
        raise ValueError(f"no tasks selected for chain={chain_id} start_index={start_index}")
    print(
        f"[chain {chain_id}] {len(selected_tasks)}/{len(tasks)} task(s): "
        f"{', '.join(t.name for t in selected_tasks)}",
        flush=True,
    )
    results: list[dict] = []
    for idx, task_dir in enumerate(selected_tasks, start=start_index):
        run = build_run(
            chain_id=chain_id,
            task_dir=task_dir,
            variant=variant,
            trials_dir=trials_dir,
            model=model,
            extra_env=extra_env,
            extra_kwargs=extra_kwargs,
            extra_args=extra_args,
            harbor_bin=harbor_bin,
        )
        print(
            f"[chain {chain_id}] ({idx}/{len(tasks)}) starting {run.task_name}",
            flush=True,
        )
        result = run_one_task(run, dry_run=dry_run)
        results.append(result)
        print(
            f"[chain {chain_id}] ({idx}/{len(tasks)}) {run.task_name} -> "
            f"exit={result.get('exit_code')} elapsed={result.get('elapsed_sec')}s",
            flush=True,
        )
        if _uses_evomem(variant) and not dry_run:
            trial_dir = _latest_trial_dir(trials_dir, run.task_name)
            if trial_dir is not None:
                try:
                    backfill = seed_validated_memory_from_trial(
                        chain_id=chain_id,
                        task_dir=task_dir,
                        trial_dir=trial_dir,
                        trials_dir=trials_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    backfill = {"seeded": False, "reason": f"error: {exc}"}
                result["evomem_validated_memory"] = backfill
                if backfill.get("seeded"):
                    print(
                        f"[chain {chain_id}] ({idx}/{len(tasks)}) seeded validated EvoMem memory from {trial_dir.name}",
                        flush=True,
                    )
        if stop_on_failure and result.get("exit_code", 0) != 0 and not dry_run:
            print(
                f"[chain {chain_id}] aborting due to failure on {run.task_name}",
                flush=True,
            )
            break
    return results


def main() -> int:
    from harbor_EvoMem.memory_bridge import default_dataset_root

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_root(),
        help="Root directory containing per-task subdirectories (default: ../Terminal-Bench-Evo).",
    )
    parser.add_argument(
        "--chain",
        action="append",
        required=True,
        help="Chain id (i.e. prototype task dir name). May be passed multiple times.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="terminus2_evomem",
        help="Which agent to use (default: terminus2_evomem).",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=_REPO_ROOT / "runs",
        help="Where Harbor trial outputs are stored.",
    )
    parser.add_argument(
        "--ae",
        action="append",
        default=[],
        help="Extra --ae forwarded to harbor (LLM_* are always overwritten from the local Terminus2 env file).",
    )
    parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        help="Extra --agent-kwarg key=value entries (forwarded verbatim).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Number of chains to run in parallel (each chain is itself sequential).",
    )
    parser.add_argument(
        "--harbor-bin",
        default=os.environ.get("HARBOR_BIN", "harbor"),
        help="Path to the harbor CLI (default: $HARBOR_BIN or 'harbor').",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue subsequent tasks in a chain even after one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the harbor commands but don't execute them.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based task index within each chain to start from (default: 1).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit number of tasks per chain after --start-index (default: all).",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Anything after `--` is forwarded verbatim to `harbor trials start`.",
    )
    args = parser.parse_args()

    # argparse with REMAINDER also captures the trailing `--`; drop it.
    extra_args = [a for a in args.extra_args if a != "--"]

    from harbor_EvoMem.memory_bridge import MissingLLMConfigError, authoritative_llm_config

    try:
        lm_cfg = authoritative_llm_config()
    except MissingLLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    filtered_ae = [
        e
        for e in args.ae
        if isinstance(e, str)
        and not e.startswith(("LLM_API_KEY=", "LLM_MODEL=", "LLM_BASE_URL="))
    ]
    args.ae = filtered_ae + [
        f"LLM_API_KEY={lm_cfg.api_key}",
        f"LLM_BASE_URL={lm_cfg.base_url}",
        f"LLM_MODEL={lm_cfg.model}",
    ]
    args.model = lm_cfg.model

    args.trials_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.trials_dir / "_logs" / "chain_runner_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    chains = args.chain
    if args.max_parallel <= 1 or len(chains) == 1:
        all_results: dict[str, list[dict]] = {}
        for chain_id in chains:
            results = run_chain(
                chain_id=chain_id,
                dataset=args.dataset,
                variant=args.variant,
                trials_dir=args.trials_dir,
                model=args.model,
                extra_env=args.ae,
                extra_kwargs=args.kwarg,
                extra_args=extra_args,
                harbor_bin=args.harbor_bin,
                stop_on_failure=not args.no_stop_on_failure,
                dry_run=args.dry_run,
                start_index=args.start_index,
                max_tasks=args.max_tasks,
            )
            all_results[chain_id] = results
    else:
        all_results = {}
        with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {
                pool.submit(
                    run_chain,
                    chain_id,
                    args.dataset,
                    variant=args.variant,
                    trials_dir=args.trials_dir,
                    model=args.model,
                    extra_env=args.ae,
                    extra_kwargs=args.kwarg,
                    extra_args=extra_args,
                    harbor_bin=args.harbor_bin,
                    stop_on_failure=not args.no_stop_on_failure,
                    dry_run=args.dry_run,
                    start_index=args.start_index,
                    max_tasks=args.max_tasks,
                ): chain_id
                for chain_id in chains
            }
            for fut in as_completed(futures):
                chain_id = futures[fut]
                try:
                    all_results[chain_id] = fut.result()
                except Exception as exc:
                    print(f"[chain {chain_id}] crashed: {exc}", file=sys.stderr)
                    all_results[chain_id] = [{"error": str(exc)}]

    summary_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "dataset": str(args.dataset),
                "trials_dir": str(args.trials_dir),
                "chains": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}", flush=True)

    failures = [
        (chain, task)
        for chain, tasks in all_results.items()
        for task in tasks
        if isinstance(task, dict) and task.get("exit_code") not in (0, None)
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
