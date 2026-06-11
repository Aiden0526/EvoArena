"""Family-scoped execution memory for Terminal-shift chains.

This module treats each chain as a short sequence of closely related tasks.
It persists auditable structured records, then compiles them into a compact
agent-facing family memory:

* a stable base recipe
* a short list of relevant conditional patches

The intent is to preserve EvoMem while avoiding long transcript-like
context that distracts strict terminal agents.
"""

from __future__ import annotations

import difflib
import json
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from harbor.environments.base import BaseEnvironment
except Exception:  # pragma: no cover - test environment may not have Harbor installed
    BaseEnvironment = Any  # type: ignore[misc,assignment]

from .memory.summarizer import PatchMemorySummarizer
from .memory_bridge import authoritative_llm_config, chain_root

DEFAULT_WORKDIR = "/app"
DEFAULT_FAMILY_WORKSPACE_DIR = f"{DEFAULT_WORKDIR}/.harbor_evomem"
FAMILY_MEMORY_FILE = f"{DEFAULT_FAMILY_WORKSPACE_DIR}/family_memory.md"
CLASSIC_PATCH_MEMORY_FILE = f"{DEFAULT_FAMILY_WORKSPACE_DIR}/retrieved_evomem.md"
CLASSIC_PATCH_INDEX_FILE = f"{DEFAULT_FAMILY_WORKSPACE_DIR}/retrieved_evomem_index.md"
MAX_FILE_SNAPSHOT_BYTES = 120_000
MAX_TERMINAL_TRACE_CHARS = 200_000
MAX_SOLUTION_SCRIPT_CHARS = 80_000
MAX_AGENT_FAMILY_MEMORY_CHARS = 6000
MAX_RENDERED_PATCHES = 5


async def _exec_environment(
    environment: BaseEnvironment,
    command: str,
    *,
    timeout_sec: int,
    logger: Any = None,
):
    from .patch_capture import _exec

    return await _exec(environment, command, timeout_sec=timeout_sec, logger=logger)

_FORBID_PY = re.compile(
    r"(?i)(\b(?:no|do\s+not|without|never)\s+(?:use\s+)?python\b|python\s+forbidden|"
    r"\b(?:only|must\s+use)\s+bash\b|perl|awk\b.*\binstead\b)",
)
_NEED_PY = re.compile(r"(?i)(\bmust\b.*\bpython\b|\busing\s+python\b|\bin\s+python\b)")
_FORBID_NET = re.compile(r"(?i)(\bno\s+network\b|without\s+network\b|offline\b)")
_FORBID_INST = re.compile(r"(?i)(\b(?:do\s+not|must\s+not|never)\s+install\b)")
_NEED_BASH = re.compile(r"(?i)(\bbash\b.*\bonl|shell\s+only|posix\s+sh\b)")
_PATH_RE = re.compile(r"/(?:app|opt|usr/local/bin|usr/local/sbin|tmp)/[A-Za-z0-9_./-]+")
_BACKTICK_RE = re.compile(r"`([^`]{1,160})`")
_VARIANT_HINT_RE = re.compile(
    r"(?i)(variant|environment note|output contract|success criteria|must not|do not modify|"
    r"workdir|workspace|staging|publish|promote|toolchain|absolute|exact|artifact|output)"
)
_MEMORY_PATH_MARKERS = (
    "/app/.harbor_evomem",
    ".harbor_evomem/",
    "family_memory.md",
    "solution_memory.md",
    "evomem.md",
    "execution_trace.md",
)
_SHELL_SETUP_LINE_RE = re.compile(
    r"^\s*(?:set\s+[-+][A-Za-z]+(?:\s+[-+][A-Za-z]+)*|cd\s+\S+|pwd)\s*$"
)
_SCAFFOLD_BASENAME_RE = re.compile(
    r"(?i)(^test_|_test\.|^tests?$|^test_outputs\.py$|^pytest\.ini$|^conftest\.py$)"
)
_OUTPUT_NAME_RE = re.compile(
    r"(?i)(^run\.py$|answer|artifact|output|result|solution|out\.|out_|^dist\.|model|snapshot)"
)
_MUTATING_COMMAND_RE = re.compile(
    r"(?i)(>|>>|<<|"
    r"\b(?:apt-get|apt\s+(?:install|source|update|upgrade|download)|pip|uv|uvx|npm|pnpm|yarn|cargo|go|make|cmake|configure|gcc|g\+\+|cc|"
    r"rustc|javac|mvn|gradle|git\s+(?:clone|apply|checkout|am|submodule)|curl|wget|tar|unzip|"
    r"dpkg-source|dpkg-buildpackage|debuild|install|cp|mv|mkdir|chmod|chown|ln|sed\s+-i|"
    r"patch|python3?|perl|ruby|Rscript|qemu-system)\b)"
)
_PURE_INSPECTION_RE = re.compile(
    r"^\s*(?:pwd|ls(?:\s|$)|cat(?:\s|$)|sed\s+-n(?:\s|$)|grep(?:\s|$)|rg(?:\s|$)|"
    r"find(?:\s|$)|head(?:\s|$)|tail(?:\s|$)|wc(?:\s|$)|file(?:\s|$)|ldd(?:\s|$)|apt-cache(?:\s|$)|"
    r"which(?:\s|$)|command\s+-v(?:\s|$)|printf(?:\s|$)|echo(?:\s|$))"
)
_SETUP_ONLY_COMMAND_RE = re.compile(r"(?is)^\s*(?:sudo\s+)?apt-get\s+update\s*$")


def family_terminal_dir(chain_id: str, host_root: str | Path | None) -> Path:
    root = chain_root(chain_id, host_root)
    d = root / "terminal_family_v2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "instructions").mkdir(parents=True, exist_ok=True)
    return d


def _safe_task_filename(task_name: str) -> str:
    return (
        "".join(ch if ch.isalnum() or ch in ".-_," else "_" for ch in task_name).strip("_")
        or "task"
    )


@dataclass
class FamilyState:
    release_order: list[str] = field(default_factory=list)
    last_probe_summary: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> FamilyState:
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        release_order = data.get("release_order")
        if not isinstance(release_order, list):
            release_order = []
        return cls(
            release_order=[str(item) for item in release_order if item],
            last_probe_summary=(
                str(data.get("last_probe_summary")) if data.get("last_probe_summary") else None
            ),
        )


class FamilyPaths:
    def __init__(self, chain_id: str, host_root: str | Path | None) -> None:
        self.chain_id = chain_id
        self.root = family_terminal_dir(chain_id, host_root)
        self.state_path = self.root / "family_state.json"
        self.task_memories_jsonl = self.root / "task_memories.jsonl"
        self.memory_patches_jsonl = self.root / "memory_patches.jsonl"
        self.family_memory_md = self.root / "family_memory.md"

    def instruction_path(self, task_name: str) -> Path:
        return self.root / "instructions" / f"{_safe_task_filename(task_name)}.md"


def extract_instruction_signals(text: str, *, task_name: Optional[str] = None) -> set[str]:
    signals: set[str] = set()
    if text:
        if _FORBID_PY.search(text):
            signals.add("CMP_FORBID_PYTHON")
        if _NEED_PY.search(text) and not _FORBID_PY.search(text):
            signals.add("CMP_NEED_PYTHON")
        if _FORBID_NET.search(text):
            signals.add("CMP_FORBID_NETWORK")
        if _FORBID_INST.search(text):
            signals.add("CMP_FORBID_INSTALL")
        if _NEED_BASH.search(text):
            signals.add("CMP_NEED_BASH_SHELL")
    if task_name and "-v-" in task_name:
        variant = task_name.split("-v-", 1)[1].strip().upper().replace("-", "_")
        if variant:
            signals.add(f"VARIANT_{variant[:48]}")
    return signals


def signals_from_probe(probe: dict[str, Any]) -> set[str]:
    if not isinstance(probe, dict):
        return set()
    out: set[str] = set()
    if probe.get("python3_path"):
        out.add("CAP_PYTHON3")
    if probe.get("python_path"):
        out.add("CAP_PYTHON")
    if probe.get("gcc_path"):
        out.add("CAP_GCC")
    if probe.get("node_path"):
        out.add("CAP_NODE")
    if probe.get("make_path"):
        out.add("CAP_MAKE")
    if probe.get("awk_path"):
        out.add("CAP_AWK")
    if probe.get("sed_path"):
        out.add("CAP_SED")
    if probe.get("curl_path"):
        out.add("CAP_CURL")
    return out


def build_current_signals(
    probe: dict[str, Any],
    instruction: str,
    *,
    task_name: Optional[str] = None,
) -> set[str]:
    return signals_from_probe(probe) | extract_instruction_signals(instruction, task_name=task_name)


async def probe_environment_for_family(
    environment: BaseEnvironment,
    *,
    workdir: str = DEFAULT_WORKDIR,
    timeout_sec: int = 60,
    logger: Any = None,
) -> dict[str, Any]:
    wd_q = shlex.quote(workdir)
    probe_script = rf"""
echo '__PMF__:uname'; uname -a 2>/dev/null || echo n/a
echo '__PMF__:id'; id 2>/dev/null || echo n/a
echo '__PMF__:pwd'; pwd 2>/dev/null || echo n/a
echo '__PMF__:which_python3'; command -v python3 2>/dev/null || echo NO
echo '__PMF__:which_python'; command -v python 2>/dev/null || echo NO
echo '__PMF__:which_node'; command -v node 2>/dev/null || echo NO
echo '__PMF__:which_gcc'; command -v gcc 2>/dev/null || echo NO
echo '__PMF__:which_make'; command -v make 2>/dev/null || echo NO
echo '__PMF__:which_awk'; command -v awk 2>/dev/null || echo NO
echo '__PMF__:which_sed'; command -v sed 2>/dev/null || echo NO
echo '__PMF__:which_curl'; command -v curl 2>/dev/null || echo NO
echo '__PMF__:app_ls'
find {wd_q} -maxdepth 3 -type f 2>/dev/null | LC_ALL=C sort | head -n 120 || true
"""
    res = await _exec_environment(
        environment,
        probe_script.strip(),
        timeout_sec=timeout_sec,
        logger=logger,
    )
    out = (getattr(res, "stdout", None) or "") or ""
    return _parse_probe_output(out)


def _parse_probe_output(text: str) -> dict[str, Any]:
    parts = text.split("__PMF__:")
    buckets: dict[str, str] = {}
    for blob in parts[1:]:
        lines = blob.strip().splitlines()
        if not lines:
            continue
        tag = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        buckets[tag] = body

    def _pick(tag: str) -> Optional[str]:
        raw = buckets.get(tag, "")
        if not raw:
            return None
        fn = raw.splitlines()[0].strip()
        if fn.endswith("NO") or fn == "" or fn == "n/a":
            return None
        return fn

    return {
        "raw": text[:8000],
        "uname": buckets.get("uname", "")[:1200],
        "id_line": buckets.get("id", "")[:800],
        "pwd": buckets.get("pwd"),
        "python3_path": _pick("which_python3"),
        "python_path": _pick("which_python"),
        "node_path": _pick("which_node"),
        "gcc_path": _pick("which_gcc"),
        "make_path": _pick("which_make"),
        "awk_path": _pick("which_awk"),
        "sed_path": _pick("which_sed"),
        "curl_path": _pick("which_curl"),
        "app_file_list_head": buckets.get("app_ls", "")[:4000],
    }


def probe_summary_md(probe: dict[str, Any]) -> str:
    if not isinstance(probe, dict):
        probe = {}
    tools = [
        key.replace("_path", "")
        for key in (
            "python3_path",
            "python_path",
            "gcc_path",
            "node_path",
            "make_path",
            "awk_path",
            "sed_path",
            "curl_path",
        )
        if probe.get(key)
    ]
    lines = [
        f"- kernel: {(probe.get('uname') or 'n/a')[:320]}".rstrip(),
        f"- ids: {(probe.get('id_line') or 'n/a')[:200]}",
        f"- tools: {', '.join(tools) if tools else 'none detected explicitly'}",
        f"- `{DEFAULT_WORKDIR}` files (truncated):\n```\n{(probe.get('app_file_list_head') or '')[:1000]}```",
    ]
    return "\n".join(lines)


def task_diff(prev: Optional[str], cur: str) -> str:
    if not prev or not prev.strip():
        return "(no prior release instruction snapshot in this chain — prototype or cold start)."
    pd = prev.splitlines()
    cd = cur.splitlines()
    out = "\n".join(
        difflib.unified_diff(pd, cd, fromfile="prior_release.md", tofile="current.md", lineterm="")
    )
    if not out:
        return "(instructions identical to prior snapshot)."
    return out[:4000] + ("\n... [truncated]" if len(out) > 4000 else "")


def _read_jsonl(path: Path, *, max_lines: int = 5000) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            if idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except Exception:
                continue
            if isinstance(blob, dict):
                rows.append(blob)
    return rows


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def persist_instruction_snapshot(paths: FamilyPaths, task_name: str, instruction: str) -> None:
    paths.instruction_path(task_name).write_text(instruction, encoding="utf-8")


def update_family_state(paths: FamilyPaths, task_name: str, probe_md: Optional[str]) -> None:
    st = FamilyState.load(paths.state_path)
    if task_name not in st.release_order:
        st.release_order.append(task_name)
    elif not st.release_order or st.release_order[-1] != task_name:
        st.release_order = [name for name in st.release_order if name != task_name] + [task_name]
    if probe_md:
        st.last_probe_summary = probe_md[:2000]
    paths.state_path.write_text(
        json.dumps(
            {"release_order": st.release_order, "last_probe_summary": st.last_probe_summary},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def previous_release_snapshot_task(
    paths: FamilyPaths, current_task_name: str
) -> tuple[Optional[str], Optional[str]]:
    st = FamilyState.load(paths.state_path)
    candidates = list(st.release_order)
    while candidates and candidates[-1] == current_task_name:
        candidates.pop()
    prev_name = candidates[-1] if candidates else None
    if not prev_name:
        return None, None
    ip = paths.instruction_path(prev_name)
    try:
        return prev_name, ip.read_text(encoding="utf-8", errors="replace") if ip.is_file() else None
    except Exception:
        return prev_name, None


def harvest_commands(history: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    worked: list[str] = []
    failed: list[str] = []

    def _observe(cmd: Optional[str], out: str) -> None:
        if not cmd:
            return
        cmd = _strip_memory_read_lines(cmd)
        if not cmd:
            return
        clean = " ".join(cmd.strip().split())[:420]
        if not clean:
            return
        substantive_lines = _substantive_shell_lines(cmd)
        if not substantive_lines or all(_PURE_INSPECTION_RE.search(line) for line in substantive_lines):
            return
        lowered = out.lower()
        bad_markers = (
            "error",
            "fatal:",
            "permission denied",
            "command not found",
            "exit status 1",
            "syntax error",
            "no such file",
            "traceback",
        )
        if any(marker in lowered for marker in bad_markers):
            failed.append(clean)
        else:
            worked.append(clean)

    for ev in history:
        if str(ev.get("type")) != "tool_result":
            continue
        obs = ev.get("observation") if isinstance(ev.get("observation"), dict) else {}
        cmd = str(obs.get("command") or "") if obs else ""
        parts = obs.get("content") or []
        content = ""
        if isinstance(parts, list):
            chunks: list[str] = []
            for piece in parts:
                if isinstance(piece, dict) and piece.get("text"):
                    chunks.append(str(piece["text"]))
                elif isinstance(piece, str):
                    chunks.append(piece)
            content = "\n".join(chunks)
        elif obs:
            content = str(obs.get("content") or "")
        else:
            content = str(ev.get("content") or "")
        _observe(cmd or None, content)

    def _dedupe(seq: list[str], cap: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in seq:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
            if len(out) >= cap:
                break
        return out

    return _dedupe(worked, 12), _dedupe(failed, 12)


def _is_memory_read_command(command: str) -> bool:
    lowered = str(command or "").lower()
    return any(marker.lower() in lowered for marker in _MEMORY_PATH_MARKERS)


def _strip_memory_read_lines(command: str) -> str:
    """Remove memory-inspection lines from a shell batch before extraction.

    Terminus2 often reads ``/app/.harbor_evomem/*`` together with ordinary
    inspection or solution commands in one response. Treating the whole batch
    as "memory read" discards useful commands; storing it verbatim creates
    self-referential execution traces. This keeps non-memory lines while
    dropping the memory plumbing.
    """
    lines: list[str] = []
    for line in str(command or "").splitlines():
        if _is_memory_read_command(line):
            continue
        stripped = line.strip()
        if stripped in {";", "&&", "||"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _substantive_shell_lines(command: str) -> list[str]:
    out: list[str] = []
    for raw in str(command or "").splitlines():
        line = raw.strip()
        if not line or line in {";", "&&", "||"}:
            continue
        if _SHELL_SETUP_LINE_RE.match(line):
            continue
        out.append(line)
    return out


def _looks_like_solution_command(command: str) -> bool:
    command = _strip_memory_read_lines(command)
    if not command:
        return False
    normalized = " ".join(command.split())
    substantive_lines = _substantive_shell_lines(command)
    if not substantive_lines:
        return False
    if _SETUP_ONLY_COMMAND_RE.match(normalized):
        return False
    if all(_PURE_INSPECTION_RE.search(line) for line in substantive_lines):
        return False
    return True


def _looks_like_effective_solution_command(command: str) -> bool:
    command = _strip_memory_read_lines(command)
    return bool(command and _looks_like_solution_command(command) and _MUTATING_COMMAND_RE.search(command))


def _extract_script_from_agent_message(text: str) -> str:
    raw = str(text or "").strip()
    if not raw or "script" not in raw:
        return ""
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        script = data.get("script")
        if isinstance(script, str) and _looks_like_solution_command(script):
            return _strip_memory_read_lines(script)
    return ""


def extract_solution_commands(history: list[dict[str, Any]], *, limit: int = 40) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for ev in history:
        command = ""
        if str(ev.get("type")) == "tool_call":
            action = ev.get("action") if isinstance(ev.get("action"), dict) else {}
            command = str(action.get("command") or ev.get("content") or "").strip()
        elif str(ev.get("type")) == "message" and str(ev.get("source") or "") == "agent":
            command = _extract_script_from_agent_message(str(ev.get("content") or ""))
        command = _strip_memory_read_lines(command)
        if not command or command in seen:
            continue
        if not _looks_like_solution_command(command):
            continue
        seen.add(command)
        commands.append(command)
        if len(commands) >= limit:
            break
    return commands


def render_solution_script(commands: list[str]) -> str:
    if not commands:
        return ""
    if len(commands) == 1 and commands[0].lstrip().startswith("#!"):
        script = commands[0].rstrip() + "\n"
        if len(script) > MAX_SOLUTION_SCRIPT_CHARS:
            return script[:MAX_SOLUTION_SCRIPT_CHARS].rstrip() + "\n# ... [solution script truncated]\n"
        return script
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for command in commands:
        lines.append(command.rstrip())
        lines.append("")
    script = "\n".join(lines).rstrip() + "\n"
    if len(script) > MAX_SOLUTION_SCRIPT_CHARS:
        return script[:MAX_SOLUTION_SCRIPT_CHARS].rstrip() + "\n# ... [solution script truncated]\n"
    return script


async def try_read_verifier_reward(
    environment: BaseEnvironment, *, timeout_sec: int = 15
) -> Optional[bool]:
    res = await _exec_environment(
        environment,
        "(test -r /logs/verifier/reward.txt && cat /logs/verifier/reward.txt) || echo NOFILE",
        timeout_sec=timeout_sec,
    )
    txt = ((getattr(res, "stdout", None) or "") or "").strip()
    if txt in {"", "NOFILE"}:
        return None
    if txt.startswith("1"):
        return True
    if txt.startswith("0"):
        return False
    return None


def _status_from_validation(validation_passed: Optional[bool]) -> str:
    if validation_passed is True:
        return "validated"
    if validation_passed is False:
        return "failed"
    return "tentative"


def _first_nonempty_line(text: str, limit: int = 220) -> str:
    for line in text.splitlines():
        stripped = " ".join(line.strip().split())
        if stripped:
            return stripped[:limit]
    return ""


def _extract_paths_and_clues(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _PATH_RE.findall(text or ""):
        if match not in seen:
            seen.add(match)
            out.append(match)
    for match in _BACKTICK_RE.findall(text or ""):
        clean = " ".join(match.strip().split())
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean[:180])
    for line in text.splitlines():
        stripped = " ".join(line.strip().split())
        if stripped and _VARIANT_HINT_RE.search(stripped) and stripped not in seen:
            seen.add(stripped)
            out.append(stripped[:220])
    return out[:12]


def _is_memory_path(path: str) -> bool:
    lowered = str(path or "").lower()
    return any(marker.lower() in lowered for marker in _MEMORY_PATH_MARKERS)


def _normalize_container_path(path: str, *, workdir: str = DEFAULT_WORKDIR) -> str:
    clean = str(path or "").strip().strip("'\"")
    clean = clean.rstrip(".,:;)")
    if not clean:
        return ""
    if clean.startswith("/"):
        return clean
    return f"{workdir.rstrip('/')}/{clean.lstrip('./')}"


def _is_scaffold_or_verifier_path(path: str) -> bool:
    p = str(path or "").strip()
    if not p:
        return True
    if _is_memory_path(p):
        return True
    rel = p
    for prefix in ("/app/", "app/", "./"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
    parts = [part for part in rel.split("/") if part]
    if not parts:
        return True
    if any(part in {"tests", "test", "__pycache__", ".git"} for part in parts):
        return True
    basename = parts[-1]
    if _SCAFFOLD_BASENAME_RE.search(basename):
        return True
    return False


def _looks_like_solution_output_path(path: str) -> bool:
    p = str(path or "").strip()
    if not p or _is_scaffold_or_verifier_path(p):
        return False
    basename = Path(p).name
    if _OUTPUT_NAME_RE.search(basename):
        return True
    if any(segment in p for segment in ("/out/", "/output/", "/outputs/", "/result/", "/results/", "/artifacts/")):
        return True
    return False


def _filter_solution_paths(paths: list[str], *, workdir: str = DEFAULT_WORKDIR) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        path = _normalize_container_path(str(raw or ""), workdir=workdir)
        if not path or _is_memory_path(path) or _is_scaffold_or_verifier_path(path):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _extract_written_paths_from_history(
    history: list[dict[str, Any]],
    *,
    workdir: str = DEFAULT_WORKDIR,
) -> list[str]:
    patterns = [
        re.compile(r"(?:^|[\s;])(?:cat|printf|echo)\b[^|;&\n]*?(?:>|>>)\s*([^\s;&|]+)"),
        re.compile(r"(?:^|[\s;])tee(?:\s+-a)?\s+([^\s;&|]+)"),
        re.compile(r"(?:^|[\s;])touch\s+([^\s;&|]+)"),
        re.compile(r"(?:^|[\s;])cp\b\s+[^\n;&|]+\s+([^\s;&|]+)"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for ev in history:
        command = ""
        if isinstance(ev.get("action"), dict):
            command = str(ev["action"].get("command") or "")
        if not command and isinstance(ev.get("observation"), dict):
            command = str(ev["observation"].get("command") or "")
        if not command:
            command = str(ev.get("content") or "")
        command = _strip_memory_read_lines(command)
        if not command:
            continue
        for pattern in patterns:
            for raw in pattern.findall(command):
                path = _normalize_container_path(raw, workdir=workdir)
                if (
                    not path.startswith(f"{workdir.rstrip('/')}/")
                    or _is_scaffold_or_verifier_path(path)
                    or path in seen
                ):
                    continue
                seen.add(path)
                out.append(path)
    return out


def _instruction_solution_candidate_paths(
    instruction_text: str,
    *,
    workdir: str = DEFAULT_WORKDIR,
    limit: int = 8,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for clue in _extract_paths_and_clues(instruction_text):
        if not clue.startswith(("/app/", "/opt/", "/usr/local/bin/", "/usr/local/sbin/", "/tmp/")):
            continue
        path = _normalize_container_path(clue, workdir=workdir)
        if _looks_like_solution_output_path(path) and path not in seen:
            seen.add(path)
            out.append(path)
        if len(out) >= limit:
            break
    return out


def _normalize_list(value: Any, *, limit: int = 8, item_limit: int = 240) -> list[str]:
    raw_items = value if isinstance(value, list) else [value] if value else []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        text = text[:item_limit]
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_command_list(value: Any, *, limit: int = 8, item_limit: int = 300) -> list[str]:
    return [
        item
        for raw in _normalize_list(value, limit=limit * 2, item_limit=item_limit)
        for item in [_strip_memory_read_lines(raw)]
        if item
    ][:limit]


def _extract_variant_lines(text: str, *, limit: int = 6) -> list[str]:
    return _normalize_list(
        [
            " ".join(line.strip().split())
            for line in text.splitlines()
            if line.strip() and _VARIANT_HINT_RE.search(line)
        ],
        limit=limit,
    )


def _guess_code_fence(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".diff": "diff",
        ".patch": "diff",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
    }.get(suffix, "text")


def _filter_diff_text(diff_text: str) -> str:
    text = str(diff_text or "").strip()
    if not text:
        return ""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git ") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    kept: list[str] = []
    for block in blocks:
        first = block.splitlines()[0] if block.splitlines() else ""
        paths = re.findall(r"\s[ab]/([^\s]+)", first)
        if paths and all(_is_memory_path(path) or _is_scaffold_or_verifier_path(path) for path in paths):
            continue
        if "Subproject commit " in block and "-dirty" in block and len(block.splitlines()) <= 8:
            continue
        kept.append(block)
    return "\n".join(kept).strip()


def _render_snapshot_change_details(file_snapshots: dict[str, str]) -> str:
    if not file_snapshots:
        return ""
    parts = ["Captured final solution file contents:"]
    for path, content in file_snapshots.items():
        parts.extend(
            [
                "",
                f"### `{path}`",
                "",
                f"```{_guess_code_fence(path)}",
                str(content),
                "```",
            ]
        )
    return "\n".join(parts).strip()


def _unified_text_diff(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "\n".join(
        difflib.unified_diff(
            str(before or "").splitlines(),
            str(after or "").splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    ).strip()


def _render_solution_delta(
    *,
    previous_memory: Optional[dict[str, Any]],
    current_solution_script: str,
    current_file_snapshots: dict[str, str],
    fallback_diff: str,
    from_task: Optional[str],
    to_task: str,
) -> tuple[str, str]:
    if not isinstance(previous_memory, dict):
        return "", ""

    parts: list[str] = []
    prev_task_label = from_task or str(previous_memory.get("task_name") or "previous")
    prev_script = str(previous_memory.get("solution_script") or "").strip()
    cur_script = str(current_solution_script or "").strip()
    if cur_script and prev_script:
        script_diff = _unified_text_diff(
            prev_script,
            cur_script,
            fromfile=f"{prev_task_label}/solution.sh",
            tofile=f"{to_task}/solution.sh",
        )
        if script_diff:
            parts.extend(["### Solution Script Diff", "", "```diff", script_diff, "```"])
    elif cur_script:
        script_diff = _unified_text_diff(
            "",
            cur_script,
            fromfile="/dev/null",
            tofile=f"{to_task}/solution.sh",
        )
        if script_diff:
            parts.extend(["### Solution Script Diff", "", "```diff", script_diff, "```"])

    prev_snapshots = previous_memory.get("file_snapshots")
    if not isinstance(prev_snapshots, dict):
        prev_snapshots = {}
    prev_file_snapshots = {
        str(path): str(content)
        for path, content in prev_snapshots.items()
        if path and content
    }
    cur_file_snapshots = {
        str(path): str(content)
        for path, content in (current_file_snapshots or {}).items()
        if path and content
    }
    file_diffs: list[str] = []
    for path in sorted(cur_file_snapshots):
        before = prev_file_snapshots.get(path, "")
        after = cur_file_snapshots.get(path, "")
        if before == after:
            continue
        file_diff = _unified_text_diff(
            before,
            after,
            fromfile=path if before else "/dev/null",
            tofile=path if after else "/dev/null",
        )
        if file_diff:
            file_diffs.append(file_diff)
    if file_diffs:
        parts.extend(
            [
                "### Solution File Snapshot Diff",
                "",
                "```diff",
                "\n\n".join(file_diffs),
                "```",
            ]
        )

    cleaned_fallback = str(fallback_diff or "").strip()
    if cleaned_fallback:
        parts.extend(["### Git Diff", "", "```diff", cleaned_fallback, "```"])

    if not parts:
        return "", ""
    return "\n\n".join(parts).strip(), "solution_delta"


def _task_memory_candidate_paths(
    instruction_text: str,
    changed_files: list[str],
    *,
    history: Optional[list[dict[str, Any]]] = None,
    workdir: str = DEFAULT_WORKDIR,
    limit: int = 10,
) -> list[str]:
    base = workdir.rstrip("/")
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        clean = str(path or "").strip()
        if not clean:
            return
        if clean.startswith("/"):
            full = clean
        else:
            full = f"{base}/{clean.lstrip('./')}"
        if full in seen:
            return
        seen.add(full)
        candidates.append(full)

    for path in _filter_solution_paths(changed_files):
        _add(path)
    for path in _extract_written_paths_from_history(history or [], workdir=workdir):
        _add(path)
    for path in _instruction_solution_candidate_paths(instruction_text, workdir=workdir):
        _add(path)
    return candidates[:limit]


async def capture_file_snapshots(
    environment: BaseEnvironment,
    candidate_paths: list[str],
    *,
    timeout_sec: int = 60,
    logger: Any = None,
    byte_limit: int = MAX_FILE_SNAPSHOT_BYTES,
) -> dict[str, str]:
    candidates = [path for path in candidate_paths if path]
    if not candidates:
        return {}

    commands: list[str] = []
    markers: list[tuple[str, str, str]] = []
    for idx, path in enumerate(candidates):
        token = uuid.uuid4().hex
        begin = f"__HARBOR_EVOMEM_FILE_BEGIN__{idx}__{token}__"
        end = f"__HARBOR_EVOMEM_FILE_END__{idx}__{token}__"
        path_q = shlex.quote(path)
        markers.append((path, begin, end))
        commands.append(
            "\n".join(
                [
                    f"if [ -f {path_q} ]; then",
                    f"  MIME=$(file -b --mime-type {path_q} 2>/dev/null || echo application/octet-stream)",
                    "  case \"$MIME\" in",
                    "    text/*|application/json|application/xml|application/x-sh|application/x-shellscript)",
                    f"      printf '%s%s\\n' '{begin}' {path_q}",
                    f"      head -c {int(byte_limit)} {path_q} 2>/dev/null || cat {path_q} 2>/dev/null || true",
                    "      printf '\\n'",
                    f"      printf '%s\\n' '{end}'",
                    "      ;;",
                    "  esac",
                    "fi",
                ]
            )
        )

    res = await _exec_environment(
        environment,
        "\n".join(commands),
        timeout_sec=timeout_sec,
        logger=logger,
    )
    out = str((getattr(res, "stdout", None) or "") or "")
    snapshots: dict[str, str] = {}
    for path, begin, end in markers:
        start_marker = begin + path
        start = out.find(start_marker)
        if start < 0:
            continue
        start += len(start_marker)
        if start < len(out) and out[start] == "\n":
            start += 1
        finish = out.find(end, start)
        if finish < 0:
            continue
        content = out[start:finish].rstrip("\n")
        if content:
            snapshots[path] = content
    return snapshots


def build_terminal_trace(history: list[dict[str, Any]], *, char_limit: int = MAX_TERMINAL_TRACE_CHARS) -> str:
    parts = ["# Previous Execution Trace", ""]
    step_no = 0
    for ev in history:
        if str(ev.get("type")) != "tool_result":
            continue
        obs = ev.get("observation") if isinstance(ev.get("observation"), dict) else {}
        raw_command = str(obs.get("command") or "")
        command = _strip_memory_read_lines(raw_command).strip()
        if not command:
            continue
        content = obs.get("content") or []
        output_parts: list[str] = []
        if isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and piece.get("text"):
                    output_parts.append(str(piece.get("text")))
                elif isinstance(piece, str):
                    output_parts.append(piece)
        elif content:
            output_parts.append(str(content))
        output = "\n".join(part for part in output_parts if part).strip()
        if _is_memory_read_command(raw_command) and raw_command.strip() != command:
            output = "(terminal output omitted because this command batch included memory reads)"
        step_no += 1
        parts.append(f"## Step {step_no}")
        parts.append("")
        parts.append("```bash")
        parts.append(command)
        parts.append("```")
        parts.append("")
        if output:
            parts.append("```text")
            parts.append(output)
            parts.append("```")
            parts.append("")
        if sum(len(part) + 1 for part in parts) >= char_limit:
            parts.append("...[trace truncated]")
            break
    if step_no == 0:
        parts.append("No terminal command trace was captured.")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _instruction_change_lines(
    prev_text: Optional[str],
    cur_text: str,
    *,
    limit: int = 6,
) -> list[str]:
    cur_lines = [" ".join(line.strip().split()) for line in cur_text.splitlines() if line.strip()]
    if not prev_text:
        return _normalize_list(_extract_variant_lines(cur_text) or cur_lines, limit=limit)
    prev_set = {" ".join(line.strip().split()) for line in prev_text.splitlines() if line.strip()}
    added = [line for line in cur_lines if line and line not in prev_set]
    prioritized = [line for line in added if _VARIANT_HINT_RE.search(line)]
    return _normalize_list(prioritized + added, limit=limit)


def _fallback_task_memory(
    *,
    task_name: str,
    instruction_text: str,
    changed_files: list[str],
    worked: list[str],
    failed: list[str],
) -> dict[str, Any]:
    clues = _instruction_solution_candidate_paths(instruction_text)
    goal = _first_nonempty_line(instruction_text, limit=280) or task_name
    file_hint = ", ".join(_normalize_list(changed_files + clues, limit=3))
    worked_hint = "; ".join(_normalize_list(worked, limit=2))
    return {
        "task_goal": goal,
        "execution_summary": (
            f"Prior execution for `{task_name}` focused on {file_hint or 'the main task files'}"
            + (f" and relied on commands such as {worked_hint}." if worked_hint else ".")
        ),
        "key_files": _normalize_list(_filter_solution_paths(changed_files) + clues, limit=8),
        "artifacts": _normalize_list([item for item in clues if item.startswith("/app/")], limit=6),
        "commands_that_worked": _normalize_command_list(worked, limit=6),
        "commands_that_failed": _normalize_command_list(failed, limit=6),
        "verification_summary": _trim_list(
            _normalize_list(_extract_variant_lines(instruction_text), limit=4),
            limit=3,
        ),
    }


def _fallback_memory_patch(
    *,
    from_task: Optional[str],
    to_task: str,
    prev_instruction_text: Optional[str],
    instruction_text: str,
    changed_files: list[str],
    worked: list[str],
    failed: list[str],
) -> dict[str, Any]:
    delta_lines = _instruction_change_lines(prev_instruction_text, instruction_text, limit=6)
    changed = _normalize_list(delta_lines, limit=6)
    clue_files = _instruction_solution_candidate_paths(instruction_text)
    files = _normalize_list(_filter_solution_paths(changed_files) or clue_files, limit=6)
    command_deltas = _normalize_command_list(worked[:3] + failed[:2], limit=6)
    change_types = _infer_change_types(
        "\n".join(delta_lines),
        "\n".join(files),
        to_task,
    )
    return {
        "change_type": change_types or ["other"],
        "prior_requirement": _redact_prior_output_hints(prev_instruction_text or ""),
        "new_requirement": _redact_prior_output_hints("; ".join(delta_lines) or instruction_text),
        "observed_adaptation": _normalize_list(delta_lines + files, limit=6),
        "general_pattern": (
            "When a future task has the same kind of change, compare the current instruction against "
            "the prior contract and update only the corresponding current path/format/input/environment details."
        ),
        "do_not_copy": _normalize_list(files, limit=4),
        "what_changed": changed,
        "why_changed": (
            "The current task changes execution constraints relative to the previous family task: "
            + "; ".join(changed[:2])
            if from_task
            else "This task establishes the initial family execution context."
        ),
        "change_context": (
            f"Adapt execution from `{from_task}` to `{to_task}` by updating the changed task contract."
            if from_task
            else f"`{to_task}` is the first recorded execution in this family."
        ),
        "required_updates": _normalize_list(delta_lines + _filter_solution_paths(changed_files), limit=6),
        "file_deltas": files,
        "command_deltas": command_deltas,
        "verification_deltas": _normalize_list(delta_lines, limit=4),
        "adaptation_summary": (
            f"Carry over the previous execution from `{from_task}` into `{to_task}` and only "
            "adjust the changed execution constraints before re-verifying."
            if from_task
            else f"No prior family task was available; treat `{to_task}` as the prototype memory."
        ),
    }


def _derive_recipe_steps(
    *,
    task_memory: dict[str, Any],
    solution_commands: Optional[list[str]],
    worked: list[str],
    execution_summary: str = "",
) -> list[str]:
    explicit = _normalize_list(task_memory.get("recipe_steps"), limit=8, item_limit=220)
    if explicit:
        return explicit

    candidates: list[str] = []
    for command in _normalize_command_list(solution_commands or worked, limit=6, item_limit=260):
        first = _first_nonempty_line(command, limit=220)
        if first:
            candidates.append(first)
    if candidates:
        return _normalize_list(candidates, limit=6, item_limit=220)

    summary = str(task_memory.get("execution_summary") or execution_summary or "").strip()
    if summary:
        return _normalize_list([summary], limit=1, item_limit=240)
    return []


def _family_memory_llm(
    *,
    family_id: str,
    task_name: str,
    from_task: Optional[str],
    prev_instruction_text: Optional[str],
    instruction_text: str,
    current_probe_md: str,
    previous_task_memory: Optional[dict[str, Any]],
    diff_excerpt: str,
    changed_files: list[str],
    worked: list[str],
    failed: list[str],
) -> dict[str, Any]:
    cfg = authoritative_llm_config()
    summarizer = PatchMemorySummarizer(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
    prev_mem_excerpt = json.dumps(previous_task_memory or {}, ensure_ascii=False, indent=2)[:3000]
    prompt = f"""
You are building memory for a chained terminal benchmark family.

Return JSON only with keys:
{{
  "task_memory": {{
    "task_goal": "short string",
    "execution_summary": "short paragraph describing what the execution did",
    "recipe_steps": ["short imperative step", "..."],
    "key_files": ["..."],
    "artifacts": ["..."],
    "commands_that_worked": ["..."],
    "commands_that_failed": ["..."],
    "verification_summary": "short string"
  }},
  "memory_patch": {{
    "change_type": ["output_path | input_path | output_format | input_contract | cwd | data_directory | environment | toolchain | hint_removed | other"],
    "prior_requirement": "literal previous requirement/contract, if identifiable",
    "new_requirement": "literal new requirement/contract, if identifiable",
    "observed_adaptation": ["what the prior run changed in response", "..."],
    "general_pattern": "general lesson for future tasks with the same change type",
    "do_not_copy": ["old concrete path/value/output fragment that should not be copied unless current task says so"],
    "trigger": "legacy concise description of the transition",
    "action_delta": "legacy concise observed adaptation, not an instruction for future tasks",
    "avoid": ["mistake to avoid", "..."],
    "what_changed": ["..."],
    "why_changed": "short paragraph",
    "change_context": "short paragraph",
    "required_updates": ["..."],
    "file_deltas": ["..."],
    "command_deltas": ["..."],
    "verification_deltas": ["..."],
    "adaptation_summary": "short paragraph"
  }}
}}

Principles:
- "task_memory" is a compact snapshot of the actual prior execution for `{task_name}`.
- Preserve concrete actions from the run: exact commands, scripts, file paths, dataset/config names, columns, output paths, and verification checks.
- "commands_that_worked" should contain the most reusable exact commands/scripts, not paraphrases.
- "recipe_steps" may be short, but it must be grounded in what was actually done.
- "memory_patch" is a transition example, not an instruction for the next task.
- The most important patch fields are "change_type", "prior_requirement", "new_requirement", "observed_adaptation", "general_pattern", and "do_not_copy".
- Patch fields should describe what changed from the previous task and how the prior run adapted.
- Do not tell a future agent to copy old concrete paths, values, prefixes, or final answers unless those appear in its current instruction.
- Focus on execution constraints, commands, artifacts, verifier behavior, workdir/path/toolchain changes.
- Prior code may be wrong. Do not include execution outcome or pass/fail reporting.
- Do not present previous execution as guaranteed correct.
- Keep every field concise, literal, and operational.

family_id={family_id}
from_task={from_task or "NONE"}
to_task={task_name}

Previous task instruction:
<<<PREV
{(prev_instruction_text or "")[:2600]}
PREV>>>

Current task instruction:
<<<CUR
{instruction_text[:5000]}
CUR>>>

Previous task memory:
<<<PREV_MEMORY
{prev_mem_excerpt}
PREV_MEMORY>>>

Current environment probe:
<<<PROBE
{current_probe_md[:2200]}
PROBE>>>

Changed files:
{json.dumps(changed_files[:30], ensure_ascii=False)}

Git diff excerpt:
<<<DIFF
{diff_excerpt[:4500]}
DIFF>>>

Commands that looked successful:
{json.dumps(worked[:12], ensure_ascii=False, indent=2)}

Commands or outputs that looked bad:
{json.dumps(failed[:12], ensure_ascii=False, indent=2)}
""".strip()
    return summarizer._call_llm_json(prompt, "terminal_family_task_memory")


def _build_task_memory_record(
    *,
    family_id: str,
    task_name: str,
    from_task: Optional[str],
    instruction_text: str,
    probe: dict[str, Any],
    changed_files: list[str],
    diff_text: str,
    worked: list[str],
    failed: list[str],
    validation_passed: Optional[bool],
    task_memory: dict[str, Any],
    file_snapshots: Optional[dict[str, str]] = None,
    terminal_trace: str = "",
    solution_commands: Optional[list[str]] = None,
) -> dict[str, Any]:
    status = _status_from_validation(validation_passed)
    signals = sorted(build_current_signals(probe, instruction_text, task_name=task_name))
    meaningful_execution = bool(
        solution_commands
        or changed_files
        or (diff_text or "").strip()
        or file_snapshots
    )

    record = {
        "record_id": str(uuid.uuid4()),
        "record_type": "task_memory",
        "family_id": family_id,
        "task_name": task_name,
        "from_task": from_task,
        "created_ts": time.time(),
        "status": status,
        "meaningful_execution": meaningful_execution,
        "variant_signals": signals,
        "instruction_excerpt": instruction_text[:2400],
        "probe_summary": probe_summary_md(probe)[:2000],
        "task_goal": str(task_memory.get("task_goal") or "")[:400],
        "execution_summary": str(task_memory.get("execution_summary") or "")[:1200],
        "recipe_steps": _derive_recipe_steps(
            task_memory=task_memory,
            solution_commands=solution_commands,
            worked=worked,
            execution_summary=str(task_memory.get("execution_summary") or ""),
        ),
        "key_files": _normalize_list(
            _filter_solution_paths(task_memory.get("key_files") or [])
            or changed_files,
            limit=8,
        ),
        "artifacts": _normalize_list(
            _filter_solution_paths(task_memory.get("artifacts") or []), limit=8
        ),
        "commands_that_worked": _normalize_command_list(
            task_memory.get("commands_that_worked") or worked, limit=8
        ),
        "commands_that_failed": _normalize_command_list(
            task_memory.get("commands_that_failed") or failed, limit=8
        ),
        "verification_summary": str(task_memory.get("verification_summary") or "")[:600],
        "changed_files": _normalize_list(changed_files, limit=20),
        "diff_excerpt": (diff_text or "")[:2400],
        "diff_text": diff_text or "",
        "file_snapshots": {
            str(path): str(content)
            for path, content in (file_snapshots or {}).items()
            if path and content
        },
        "solution_commands": _normalize_command_list(solution_commands or [], limit=40, item_limit=2000),
        "solution_script": render_solution_script(solution_commands or []),
        "terminal_trace": terminal_trace or "",
        "validation": {"passed": validation_passed},
    }
    record["summary_line"] = (
        f"{record['task_name']} [{status}] | execution={record['execution_summary'][:140]} | "
        f"verify={record['verification_summary'][:120]}"
    )[:420]
    return record


def _patch_trigger(memory_patch: dict[str, Any], task_delta_lines: list[str], signals: list[str]) -> str:
    trigger = " ".join(str(memory_patch.get("trigger") or "").split()).strip()
    if trigger:
        return trigger[:240]
    for line in _normalize_list(memory_patch.get("what_changed"), limit=3, item_limit=200):
        if line:
            return line[:240]
    for line in task_delta_lines:
        if line:
            return line[:240]
    if signals:
        return ", ".join(signals[:3])[:240]
    return "this variant changes the task contract"


def _patch_action_delta(memory_patch: dict[str, Any]) -> str:
    action = " ".join(str(memory_patch.get("action_delta") or "").split()).strip()
    if action:
        return action[:320]
    for key in ("required_updates", "command_deltas", "file_deltas", "adaptation_summary"):
        values = _normalize_list(memory_patch.get(key), limit=2, item_limit=220)
        if values:
            return "; ".join(values)[:320]
    return ""


def _patch_sentence(*, trigger: str, action_delta: str) -> str:
    trigger = trigger.rstrip(". ")
    action_delta = action_delta.rstrip(". ")
    if not action_delta:
        return ""
    trigger = re.sub(r"^(if|when)\s+", "", trigger, flags=re.IGNORECASE).strip()
    return f"If {trigger}, then {action_delta}."


def _build_memory_patch_record(
    *,
    family_id: str,
    from_task: Optional[str],
    to_task: str,
    prev_instruction_text: Optional[str],
    instruction_text: str,
    probe: dict[str, Any],
    changed_files: list[str],
    validation_passed: Optional[bool],
    memory_patch: dict[str, Any],
    diff_text: str,
    previous_memory: Optional[dict[str, Any]] = None,
    file_snapshots: Optional[dict[str, str]] = None,
    solution_commands: Optional[list[str]] = None,
) -> dict[str, Any]:
    status = _status_from_validation(validation_passed)
    signals = sorted(build_current_signals(probe, instruction_text, task_name=to_task))
    task_delta_lines = _instruction_change_lines(prev_instruction_text, instruction_text, limit=10)
    raw_transition_text = "\n".join(
        [
            "\n".join(task_delta_lines),
            str(memory_patch.get("trigger") or ""),
            str(memory_patch.get("action_delta") or ""),
            "\n".join(_normalize_list(memory_patch.get("what_changed"), limit=8)),
            "\n".join(_normalize_list(memory_patch.get("required_updates"), limit=8)),
            "\n".join(_normalize_list(memory_patch.get("file_deltas"), limit=8)),
            to_task,
        ]
    )
    change_types = _normalize_change_types(
        memory_patch.get("change_type") or memory_patch.get("change_types"),
        fallback_texts=(raw_transition_text,),
        signals=signals,
    )
    trigger = _redact_prior_output_hints(_patch_trigger(memory_patch, task_delta_lines, signals))
    action_delta = _redact_prior_output_hints(_patch_action_delta(memory_patch))
    avoid = _normalize_list(
        [_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("avoid"), limit=3, item_limit=180)],
        limit=3,
        item_limit=180,
    )
    cleaned_diff = _filter_diff_text(diff_text)
    snapshot_details = _render_snapshot_change_details(file_snapshots or {})
    solution_script = render_solution_script(solution_commands or [])
    solution_delta, solution_delta_kind = _render_solution_delta(
        previous_memory=previous_memory,
        current_solution_script=solution_script,
        current_file_snapshots=file_snapshots or {},
        fallback_diff=cleaned_diff,
        from_task=from_task,
        to_task=to_task,
    )
    full_changes = _redact_final_answer_literals(
        solution_delta or cleaned_diff or snapshot_details or solution_script
    )
    cleaned_diff = _redact_final_answer_literals(cleaned_diff)
    meaningful_change = bool(
        solution_commands
        or cleaned_diff
        or snapshot_details
        or changed_files
    )
    if solution_delta_kind:
        full_changes_kind = solution_delta_kind
    elif cleaned_diff:
        full_changes_kind = "diff"
    elif snapshot_details:
        full_changes_kind = "snapshot"
    elif solution_script:
        full_changes_kind = "script"
    else:
        full_changes_kind = ""

    file_delta_paths = _filter_solution_paths(changed_files)
    if not file_delta_paths:
        extracted: list[str] = []
        for item in _normalize_list(memory_patch.get("file_deltas"), limit=8, item_limit=220):
            extracted.extend(sorted(_paths_in_text(item)))
        file_delta_paths = _filter_solution_paths(extracted)

    record = {
        "record_id": str(uuid.uuid4()),
        "record_type": "memory_patch",
        "family_id": family_id,
        "from_task": from_task,
        "to_task": to_task,
        "created_ts": time.time(),
        "status": status,
        "meaningful_change": meaningful_change,
        "variant_signals": signals,
        "previous_task_description": (prev_instruction_text or "")[:8000],
        "current_task_description": instruction_text[:8000],
        "task_delta": task_delta_lines,
        "change_type": change_types,
        "change_types": change_types,
        "prior_requirement": _redact_prior_output_hints(str(memory_patch.get("prior_requirement") or prev_instruction_text or "")[:1200]),
        "new_requirement": _redact_prior_output_hints(str(memory_patch.get("new_requirement") or "; ".join(task_delta_lines) or instruction_text)[:1200]),
        "observed_adaptation": _normalize_list(
            [_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("observed_adaptation") or memory_patch.get("required_updates") or task_delta_lines, limit=8, item_limit=220)],
            limit=8,
            item_limit=220,
        ),
        "general_pattern": _redact_prior_output_hints(
            str(
                memory_patch.get("general_pattern")
                or memory_patch.get("adaptation_summary")
                or "For future tasks with this change type, identify the current requirement and update only the corresponding current path/format/input/environment detail."
            )[:900]
        ),
        "do_not_copy": _normalize_list(
            [_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("do_not_copy") or memory_patch.get("file_deltas") or [], limit=8, item_limit=180)],
            limit=8,
            item_limit=180,
        ),
        "trigger": trigger,
        "action_delta": action_delta,
        "avoid": avoid,
        "patch_sentence": _patch_sentence(trigger=trigger, action_delta=action_delta),
        "what_changed": _normalize_list([_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("what_changed") or task_delta_lines, limit=8)], limit=8),
        "why_changed": _redact_prior_output_hints(str(memory_patch.get("why_changed") or "")[:900]),
        "change_context": _redact_prior_output_hints(str(memory_patch.get("change_context") or "")[:900]),
        "required_updates": _normalize_list([_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("required_updates"), limit=8)], limit=8),
        "file_deltas": _normalize_list(file_delta_paths, limit=8),
        "command_deltas": _normalize_command_list([_redact_prior_output_hints(item) for item in _normalize_command_list(memory_patch.get("command_deltas"), limit=8)], limit=8),
        "verification_deltas": _normalize_list([_redact_prior_output_hints(item) for item in _normalize_list(memory_patch.get("verification_deltas"), limit=8)], limit=8),
        "adaptation_summary": _redact_prior_output_hints(str(memory_patch.get("adaptation_summary") or "")[:1200]),
        "full_changes": full_changes,
        "full_changes_kind": full_changes_kind,
        "diff_text": cleaned_diff,
        "validation": {"passed": validation_passed},
    }
    record["summary_line"] = (
        f"{from_task or 'prototype'} -> {to_task} [{status}] | "
        f"why={record['why_changed'][:120]} | update={'; '.join(record['required_updates'][:2])}"
    )[:420]
    return record


def store_family_memory(
    paths: FamilyPaths,
    *,
    task_name: str,
    probe: dict[str, Any],
    prev_task_name: Optional[str],
    prev_instruction_text: Optional[str],
    instruction_text: str,
    history: list[dict[str, Any]],
    diff_text: str,
    changed_files: list[str],
    validation_passed: Optional[bool],
    file_snapshots: Optional[dict[str, str]] = None,
    terminal_trace: str = "",
    logger: Any = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    changed_files = _filter_solution_paths(changed_files)
    diff_text = _filter_diff_text(diff_text)
    file_snapshots = {
        path: content
        for path, content in (file_snapshots or {}).items()
        if path and content and not _is_scaffold_or_verifier_path(path)
    }
    worked, failed = harvest_commands(history)
    solution_commands = extract_solution_commands(history)
    previous_memory = (
        _task_memory_for_task(paths, prev_task_name)
        if prev_task_name
        else None
    )
    try:
        llm_blob = _family_memory_llm(
            family_id=paths.chain_id,
            task_name=task_name,
            from_task=prev_task_name,
            prev_instruction_text=prev_instruction_text,
            instruction_text=instruction_text,
            current_probe_md=probe_summary_md(probe),
            previous_task_memory=previous_memory,
            diff_excerpt=(diff_text or "")[:6000],
            changed_files=changed_files,
            worked=worked,
            failed=failed,
        )
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("family_trace_memory llm synthesis failed: %s", exc)
        llm_blob = {}

    task_memory_blob = llm_blob.get("task_memory") if isinstance(llm_blob, dict) else None
    if not isinstance(task_memory_blob, dict):
        task_memory_blob = _fallback_task_memory(
            task_name=task_name,
            instruction_text=instruction_text,
            changed_files=changed_files,
            worked=worked,
            failed=failed,
        )

    patch_blob = llm_blob.get("memory_patch") if isinstance(llm_blob, dict) else None
    if not isinstance(patch_blob, dict):
        patch_blob = _fallback_memory_patch(
            from_task=prev_task_name,
            to_task=task_name,
            prev_instruction_text=prev_instruction_text,
            instruction_text=instruction_text,
            changed_files=changed_files,
            worked=worked,
            failed=failed,
        )

    task_record = _build_task_memory_record(
        family_id=paths.chain_id,
        task_name=task_name,
        from_task=prev_task_name,
        instruction_text=instruction_text,
        probe=probe,
        changed_files=changed_files,
        diff_text=diff_text,
        worked=worked,
        failed=failed,
        validation_passed=validation_passed,
        task_memory=task_memory_blob,
        file_snapshots=file_snapshots,
        terminal_trace=terminal_trace,
        solution_commands=solution_commands,
    )
    _append_jsonl(paths.task_memories_jsonl, task_record)
    if logger:
        logger.info(
            "family_trace_memory: stored task_memory task=%s status=%s",
            task_name,
            task_record["status"],
        )

    patch_record: Optional[dict[str, Any]] = None
    if prev_task_name:
        patch_record = _build_memory_patch_record(
            family_id=paths.chain_id,
            from_task=prev_task_name,
            to_task=task_name,
            prev_instruction_text=prev_instruction_text,
            instruction_text=instruction_text,
            probe=probe,
            changed_files=changed_files,
            validation_passed=validation_passed,
            memory_patch=patch_blob,
            diff_text=diff_text,
            previous_memory=previous_memory,
            file_snapshots=file_snapshots,
            solution_commands=solution_commands,
        )
        _append_jsonl(paths.memory_patches_jsonl, patch_record)
        if logger:
            logger.info(
                "family_trace_memory: stored memory_patch %s -> %s status=%s",
                prev_task_name,
                task_name,
                patch_record["status"],
            )

    try:
        write_family_memory_document(paths)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("family_trace_memory: failed to write family_memory.md: %s", exc)

    return task_record, patch_record


def _release_order_lookup(paths: FamilyPaths) -> dict[str, int]:
    state = FamilyState.load(paths.state_path)
    return {name: idx for idx, name in enumerate(state.release_order)}


def _sort_by_release_order(paths: FamilyPaths, records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    order = _release_order_lookup(paths)
    return sorted(records, key=lambda record: order.get(str(record.get(key) or ""), 10**9))


def load_task_memories(paths: FamilyPaths) -> list[dict[str, Any]]:
    return _read_jsonl(paths.task_memories_jsonl)


def load_memory_patches(paths: FamilyPaths) -> list[dict[str, Any]]:
    return _read_jsonl(paths.memory_patches_jsonl)


def prototype_task_memory(paths: FamilyPaths, *, prefer_status: str = "validated") -> Optional[dict[str, Any]]:
    records = _sort_by_release_order(paths, load_task_memories(paths), "task_name")
    preferred = [record for record in records if record.get("status") == prefer_status]
    for record in preferred:
        if record.get("task_name") == paths.chain_id:
            return record
    return preferred[0] if preferred else (records[0] if records else None)


def latest_prior_task_memory(
    paths: FamilyPaths,
    current_task_name: str,
    *,
    prefer_status: str = "validated",
) -> Optional[dict[str, Any]]:
    order = _release_order_lookup(paths)
    cur_idx = order.get(current_task_name, 10**9)
    candidates = [
        record
        for record in load_task_memories(paths)
        if order.get(str(record.get("task_name") or ""), 10**9) < cur_idx
    ]
    preferred = [record for record in candidates if record.get("status") == prefer_status]
    if preferred:
        return sorted(preferred, key=lambda record: order.get(str(record.get("task_name") or ""), -1))[-1]
    if prefer_status:
        return None
    if candidates:
        return sorted(candidates, key=lambda record: order.get(str(record.get("task_name") or ""), -1))[-1]
    return None


def _task_memory_for_task(
    paths: FamilyPaths,
    task_name: str,
    *,
    prefer_status: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    records = [
        record for record in load_task_memories(paths)
        if str(record.get("task_name") or "") == task_name
    ]
    if not records:
        return None
    order = _release_order_lookup(paths)
    records = sorted(records, key=lambda record: record.get("created_ts", order.get(task_name, 0)))
    if prefer_status is not None:
        preferred = [record for record in records if record.get("status") == prefer_status]
        if preferred:
            return preferred[-1]
        return None
    return records[-1]


_COMMON_MEMORY_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "task",
    "variant",
    "current",
    "previous",
    "must",
    "should",
    "then",
    "into",
    "under",
    "file",
    "files",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]{3,}")


def _text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _COMMON_MEMORY_TOKENS:
            continue
        tokens.add(token)
    return tokens


def _paths_in_text(text: str) -> set[str]:
    return {match.rstrip(".,:;)\"'") for match in _PATH_RE.findall(text or "")}


def _memory_record_is_usable(record: Optional[dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    status = record.get("status")
    if status in {"failed", "tentative"}:
        return False
    if record.get("meaningful_execution") is False:
        return False
    validation = record.get("validation")
    if isinstance(validation, dict) and validation.get("passed") is not True:
        return False
    solution_script = str(record.get("solution_script") or "").strip()
    worked = _normalize_command_list(record.get("commands_that_worked"), limit=4)
    has_solution_command = bool(
        (solution_script and _looks_like_effective_solution_command(solution_script))
        or any(_looks_like_effective_solution_command(command) for command in worked)
    )
    has_snapshot = isinstance(record.get("file_snapshots"), dict) and bool(record.get("file_snapshots"))
    has_diff = bool(str(record.get("diff_text") or "").strip())
    has_changed_files = bool(_normalize_list(record.get("changed_files"), limit=1))

    if record.get("meaningful_execution") is True:
        return bool(
            _normalize_list(record.get("recipe_steps"), limit=1)
            or has_solution_command
            or has_snapshot
            or has_diff
            or has_changed_files
            or str(record.get("execution_summary") or "").strip()
        )

    return bool(
        _normalize_list(record.get("recipe_steps"), limit=1)
        or has_solution_command
        or has_snapshot
        or has_diff
        or has_changed_files
    )


def _patch_record_is_actionable(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    status = record.get("status")
    if status in {"failed", "tentative"}:
        return False
    if record.get("meaningful_change") is False:
        return False
    validation = record.get("validation")
    if isinstance(validation, dict) and validation.get("passed") is not True:
        return False
    if "meaningful_change" not in record:
        has_evidence = bool(
            str(record.get("full_changes") or "").strip()
            or str(record.get("diff_text") or "").strip()
            or _normalize_list(record.get("command_deltas"), limit=1)
            or _normalize_list(record.get("file_deltas"), limit=1)
        )
        if not has_evidence:
            return False
    return bool(
        str(record.get("patch_sentence") or "").strip()
        or str(record.get("action_delta") or "").strip()
        or _normalize_list(record.get("required_updates"), limit=1)
        or _normalize_list(record.get("command_deltas"), limit=1)
    )


def _base_task_memory(paths: FamilyPaths) -> Optional[dict[str, Any]]:
    records = [
        record
        for record in _sort_by_release_order(paths, load_task_memories(paths), "task_name")
        if _memory_record_is_usable(record)
    ]
    if not records:
        return None
    validated = [record for record in records if record.get("status") == "validated"]
    candidates = validated or records
    for record in candidates:
        if record.get("task_name") == paths.chain_id:
            return record
    return candidates[0]


def _patch_match_score(
    record: dict[str, Any],
    *,
    query_text: str,
    applicability_text: str,
    current_change_types: set[str],
    current_signals: set[str],
    previous_task_name: Optional[str],
) -> int:
    if not _patch_record_is_actionable(record):
        return 0
    patch_change_types = set(_normalize_change_types(record.get("change_types") or record.get("change_type")))
    if patch_change_types:
        if not current_change_types:
            return 0
        type_overlap = patch_change_types & current_change_types
        if not type_overlap:
            return 0
    else:
        type_overlap = set()
    score = 0
    score += 8 * len(type_overlap)
    patch_signals = {str(item) for item in _normalize_list(record.get("variant_signals"), limit=20)}
    score += 3 * len(current_signals & patch_signals)

    patch_text = "\n".join(
        [
            str(record.get("trigger") or ""),
            str(record.get("action_delta") or ""),
            str(record.get("patch_sentence") or ""),
            "\n".join(_normalize_list(record.get("what_changed"), limit=8)),
            "\n".join(_normalize_list(record.get("required_updates"), limit=8)),
            "\n".join(_normalize_list(record.get("verification_deltas"), limit=4)),
            str(record.get("to_task") or ""),
        ]
    )
    shared_paths = _paths_in_text(query_text) & _paths_in_text(patch_text)
    score += min(len(shared_paths), 2)

    shared_tokens = _text_tokens(query_text) & _text_tokens(patch_text)
    score += min(len(shared_tokens), 4)

    if previous_task_name and str(record.get("to_task") or "") == previous_task_name:
        score += 1
    return score


def _patch_required_paths(record: dict[str, Any]) -> set[str]:
    """Explicit output/file target paths a patch tells the next run to use.

    Only ``file_deltas`` are treated as hard applicability gates. Command paths
    can be reusable even when they are not named in the current instruction.
    """

    file_deltas = _normalize_list(record.get("file_deltas"), limit=12, item_limit=220)
    if not file_deltas:
        return set()
    paths = _paths_in_text("\n".join(file_deltas))
    return {
        path
        for path in paths
        if path.startswith("/app/") or path.startswith("app/") or path.startswith("./")
    }


def _select_relevant_patches(
    paths: FamilyPaths,
    *,
    task_name: str,
    query_text: str,
    applicability_text: str,
    current_change_types: set[str],
    current_signals: set[str],
    previous_task_name: Optional[str],
    limit: int = MAX_RENDERED_PATCHES,
) -> list[dict[str, Any]]:
    order = _release_order_lookup(paths)
    cur_idx = order.get(task_name, 10**9)
    scored: list[tuple[int, int, float, dict[str, Any]]] = []
    for record in load_memory_patches(paths):
        to_task = str(record.get("to_task") or "")
        to_idx = order.get(to_task, 10**9)
        if to_idx >= cur_idx:
            continue
        score = _patch_match_score(
            record,
            query_text=query_text,
            applicability_text=applicability_text,
            current_change_types=current_change_types,
            current_signals=current_signals,
            previous_task_name=previous_task_name,
        )
        if score <= 0:
            continue
        scored.append((score, to_idx, float(record.get("created_ts") or 0.0), record))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = [record for _, _, _, record in scored[:limit]]
    selected.sort(
        key=lambda record: (
            order.get(str(record.get("to_task") or ""), 10**9),
            float(record.get("created_ts") or 0.0),
        )
    )
    return selected


def retrieve_family_context(
    paths: FamilyPaths,
    *,
    task_name: str,
    instruction_text: str,
    probe: dict[str, Any],
) -> dict[str, Any]:
    prev_task_name, prev_instruction = previous_release_snapshot_task(paths, task_name)
    current_delta = task_diff(prev_instruction, instruction_text)
    delta_summary = _instruction_change_lines(prev_instruction, instruction_text, limit=5)
    applicability_text = "\n".join([task_name, instruction_text, "\n".join(delta_summary)])
    current_signals = sorted(build_current_signals(probe, instruction_text, task_name=task_name))
    current_change_types = _infer_change_types(
        current_delta,
        "\n".join(delta_summary),
        task_name,
        signals=current_signals,
    )
    retrieval_change_types = _primary_current_change_types(
        current_change_types,
        delta_summary,
        current_delta,
    )
    query = "\n".join(
        [
            task_name,
            instruction_text[:4000],
            current_delta[:1800],
            probe_summary_md(probe)[:1200],
            " ".join(current_signals),
        ]
    )

    previous_memory = None
    if prev_task_name:
        previous_memory = _task_memory_for_task(paths, prev_task_name, prefer_status="validated")
        if previous_memory is None:
            previous_memory = latest_prior_task_memory(paths, task_name)

    base_memory = _base_task_memory(paths)
    relevant_patches = _select_relevant_patches(
        paths,
        task_name=task_name,
        query_text=query,
        applicability_text=applicability_text,
        current_change_types=set(retrieval_change_types),
        current_signals=set(current_signals),
        previous_task_name=prev_task_name,
    )

    return {
        "task_name": task_name,
        "previous_task_name": prev_task_name,
        "previous_instruction_text": prev_instruction or "",
        "current_instruction_text": instruction_text,
        "current_delta": current_delta,
        "delta_summary": delta_summary,
        "current_signals": current_signals,
        "current_change_types": current_change_types,
        "retrieval_change_types": retrieval_change_types,
        "base_memory": base_memory,
        "previous_memory": previous_memory,
        "relevant_patches": relevant_patches,
        "prior_patches": relevant_patches,
    }


def _trim_list(values: list[str], *, limit: int = 3) -> str:
    items = [value for value in values if value][:limit]
    return "; ".join(items)


def _render_task_memory_block(title: str, record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {title}",
            "",
            f"- task: `{record.get('task_name')}`",
            f"- execution: {str(record.get('execution_summary') or '')[:420]}",
            f"- key files/artifacts: {_trim_list(_normalize_list(record.get('key_files'), limit=4) + _normalize_list(record.get('artifacts'), limit=4), limit=4)}",
            f"- worked commands: {_trim_list(_normalize_list(record.get('commands_that_worked'), limit=3), limit=3)}",
            f"- failed commands: {_trim_list(_normalize_list(record.get('commands_that_failed'), limit=2), limit=2)}",
            f"- verification: {str(record.get('verification_summary') or '')[:240]}",
            "",
        ]
    )


def _render_patch_block(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- `{record.get('from_task') or 'prototype'} -> {record.get('to_task')}`",
            f"  - what changed: {_trim_list(_normalize_list(record.get('what_changed'), limit=3), limit=3)}",
            f"  - why: {str(record.get('why_changed') or '')[:220]}",
            f"  - context: {str(record.get('change_context') or '')[:220]}",
            f"  - revise: {_trim_list(_normalize_list(record.get('required_updates'), limit=3), limit=3)}",
            f"  - verification deltas: {_trim_list(_normalize_list(record.get('verification_deltas'), limit=2), limit=2)}",
            "",
        ]
    )


def summarize_classic_evomem(rendered_classic: str, *, limit: int = 2) -> list[str]:
    lines = [line.strip() for line in (rendered_classic or "").splitlines()]
    summaries: list[str] = []
    for line in lines:
        if line.startswith("- `"):
            summaries.append(line)
            if len(summaries) >= limit:
                break
    return summaries


def render_classic_patch_pointer(rendered_classic: str) -> str:
    summaries = summarize_classic_evomem(rendered_classic)
    parts = [
        "### Retrieved EvoMem",
        "",
        f"Full retrieved EvoMem has been hydrated into `{CLASSIC_PATCH_MEMORY_FILE}`.",
        f"A short index is available at `{CLASSIC_PATCH_INDEX_FILE}`.",
        "Read these files if you need the detailed prior patches or diffs for overlapping files.",
    ]
    if summaries:
        parts.append("Top retrieved patches:")
        for line in summaries:
            parts.append(line)
    return "\n".join(parts).strip() + "\n"


def build_classic_workspace_files(rendered_classic: str) -> dict[str, str]:
    index_lines = [
        "# Retrieved EvoMem Index",
        "",
        f"Full EvoMem content is in `{CLASSIC_PATCH_MEMORY_FILE}`.",
        "",
    ]
    summaries = summarize_classic_evomem(rendered_classic, limit=6)
    if summaries:
        index_lines.append("Top retrieved patches:")
        index_lines.extend(summaries)
    else:
        index_lines.append("No retrieved patch summaries available.")
    index_lines.append("")
    return {
        CLASSIC_PATCH_MEMORY_FILE: rendered_classic.strip() + "\n",
        CLASSIC_PATCH_INDEX_FILE: "\n".join(index_lines),
    }


def _render_patch_sentence_from_record(record: dict[str, Any]) -> str:
    existing = " ".join(str(record.get("patch_sentence") or "").split()).strip()
    if existing:
        return existing[:420]
    trigger = " ".join(str(record.get("trigger") or "").split()).strip()
    action = " ".join(str(record.get("action_delta") or "").split()).strip()
    if not action:
        action = _patch_action_delta(record)
    if not trigger:
        trigger = _patch_trigger(
            record,
            _normalize_list(record.get("task_delta"), limit=3),
            _normalize_list(record.get("variant_signals"), limit=3),
        )
    return _patch_sentence(trigger=trigger, action_delta=action)[:420]


def _render_transition_example_lines(record: dict[str, Any], idx: int) -> list[str]:
    change_types = _normalize_change_types(
        record.get("change_types") or record.get("change_type"),
        fallback_texts=(
            str(record.get("trigger") or ""),
            str(record.get("action_delta") or ""),
            "\n".join(_normalize_list(record.get("what_changed"), limit=8)),
            "\n".join(_normalize_list(record.get("required_updates"), limit=8)),
            "\n".join(_normalize_list(record.get("file_deltas"), limit=8)),
        ),
        signals=_normalize_list(record.get("variant_signals"), limit=8),
    )
    label = ", ".join(change_types) if change_types else "task_contract"
    from_task = str(record.get("from_task") or "previous task")
    to_task = str(record.get("to_task") or "later task")
    lines = [f"- Example {idx} ({label}): `{from_task}` -> `{to_task}`"]

    prior = _redact_prior_output_hints(str(record.get("prior_requirement") or "").strip())
    new = _redact_prior_output_hints(str(record.get("new_requirement") or "").strip())
    if prior:
        lines.append(f"  Prior requirement: {prior[:240]}")
    if new:
        lines.append(f"  New requirement: {new[:240]}")

    observed = _normalize_list(
        [_redact_prior_output_hints(item) for item in _normalize_list(record.get("observed_adaptation") or record.get("required_updates"), limit=3, item_limit=180)],
        limit=3,
        item_limit=180,
    )
    if observed:
        lines.append(f"  Observed adaptation: {_trim_list(observed, limit=3)}")

    pattern = _redact_prior_output_hints(str(record.get("general_pattern") or record.get("adaptation_summary") or "").strip())
    if pattern:
        lines.append(f"  Pattern to consider: {pattern[:240]}")

    do_not_copy = _normalize_list(
        [_redact_prior_output_hints(item) for item in _normalize_list(record.get("do_not_copy"), limit=3, item_limit=160)],
        limit=3,
        item_limit=160,
    )
    if do_not_copy:
        lines.append(f"  Do not copy unless current instruction says so: {_trim_list(do_not_copy, limit=3)}")
    else:
        lines.append("  Do not copy old concrete paths, values, or output fragments unless the current instruction says so.")
    return lines


def _render_command_block(lines: list[str], command: str, *, char_limit: int = 900) -> None:
    command = command.strip()
    if not command:
        return
    command = _redact_prior_output_hints(command)
    if len(command) > char_limit:
        command = command[:char_limit].rsplit("\n", 1)[0].rstrip() + "\n# ... truncated"
    lines.extend(["```bash", command, "```"])


def _redact_final_answer_literals(text: str) -> str:
    """Avoid leaking prior final-answer values through memory transcripts."""

    text = re.sub(
        r"(\banswer\.txt\b[^\n;,.]{0,80}?\b)([0-9]{2,})\b",
        r"\1<integer>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[0-9]{2,}\b(?=[^\n]{0,100}\banswer\.txt\b)",
        "<integer>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^([+-])([0-9]{2,})$",
        r"\1<integer>",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\bprintf\s+(['\"]?)[0-9]{2,}\1\s*>\s*((?:/app/)?(?:results/)?answer\.txt)",
        r"printf '<integer>' > \2",
        text,
    )
    text = re.sub(
        r"\becho\s+(['\"]?)[0-9]{2,}\1\s*>\s*((?:/app/)?(?:results/)?answer\.txt)",
        r"echo '<integer>' > \2",
        text,
    )
    text = re.sub(
        r"(answer\.txt['\"]?\s*[:=]\s*['\"]?)[0-9]{2,}",
        r"\1<integer>",
        text,
    )
    text = re.sub(
        r"\b[A-Fa-f0-9]{24,}\b",
        "<prior-output-fragment>",
        text,
    )
    text = re.sub(
        r"\b[A-Fa-f0-9]{6,}\b",
        "<prior-output-fragment>",
        text,
    )
    return text


def _redact_prior_output_hints(text: str) -> str:
    text = _redact_final_answer_literals(str(text or ""))
    if re.fullmatch(r"\s*[0-9]{4,}\s*", text):
        return "<integer>"
    if re.fullmatch(r"\s*[A-Fa-f0-9]{4,}\s*", text):
        return "<prior-output-fragment>"
    text = re.sub(
        r"\b(begins|starts)\s+with\s+[`'\"]?[A-Za-z0-9_./:-]{4,}[`'\"]?",
        r"\1 with <prior-output-prefix>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(prefix|substring)\s+[`'\"]?[A-Za-z0-9_./:-]{4,}[`'\"]?",
        r"\1 <prior-output-fragment>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(hint)\s+(?:as\s+)?(?:a\s+)?[`'\"]?[A-Fa-f0-9]{4,}[`'\"]?",
        r"\1 <prior-output-fragment>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[A-Fa-f0-9]{24,}\b",
        "<prior-output-fragment>",
        text,
    )
    return text


def _infer_change_types(*texts: str, signals: Optional[list[str] | set[str]] = None) -> list[str]:
    blob = "\n".join(str(text or "") for text in texts).lower()
    sig_blob = " ".join(str(s).lower() for s in (signals or []))
    combined = f"{blob}\n{sig_blob}"
    types: set[str] = set()

    def has(*needles: str) -> bool:
        return any(needle in combined for needle in needles)

    if has("output path", "answer path", "artifact path", "path variant — output", "write target", "final write target", "/app/results", "/app/artifacts", "/app/out/"):
        types.add("output_path")
    if re.search(r"\bwrite\b.{0,120}\b(/app/[^\s`'\"),]+)", combined) and has("answer", "output", "artifact"):
        types.add("output_path")
    if has("image is now", "input path", "input image", "assets/code.png", "code.png", "netlist from", "circuit_gates_path", "calendar files", "/app/cal/", "/app/dat/", "/app/data/"):
        types.add("input_path")
    if has("data directory", "datapath", "/app/dat/", "/app/data/", "/app/cal/"):
        types.add("data_directory")
    if has("current working directory", "cwd=", "cwd /app", "relative to /app", "working directory"):
        types.add("cwd")
    if has("output contract", "exact string", "exact output", "no extra", "trailing newline", "prefix", "formatting", "case-sensitive", "result:"):
        types.add("output_format")
    if has("standard input", "stdin", "command-line arguments", "argv", "no command-line"):
        types.add("input_contract")
    if has(
        "toolchain",
        "python 3.12",
        "pytest",
        "clang",
        "gcc",
        "ca-certificates",
        "dockerfile",
        "debian_frontend",
        "environment variant",
        "verifier runs",
        "apt-get",
        "apt install",
        "do not use apt",
        "package manager",
        "cmp_forbid_install",
    ):
        types.add("environment")
    if has("no hint", "without hint", "substring hint"):
        types.add("hint_removed")
    return sorted(types)


def _normalize_change_types(value: Any, *, fallback_texts: tuple[str, ...] = (), signals: Optional[list[str] | set[str]] = None) -> list[str]:
    explicit = _normalize_list(value, limit=8, item_limit=80)
    if explicit:
        return sorted({item.strip().lower().replace(" ", "_") for item in explicit if item.strip()})
    return _infer_change_types(*fallback_texts, signals=signals)


def _primary_current_change_types(change_types: list[str], delta_summary: list[str], current_delta: str) -> list[str]:
    summary_text = "\n".join(delta_summary).lower()
    delta_text = summary_text or current_delta.lower()
    env_markers = (
        "environment variant",
        "environment note",
        "toolchain",
        "python 3.12",
        "pytest",
        "clang",
        "gcc",
        "ca-certificates",
        "dockerfile",
        "debian_frontend",
        "verifier runs",
        "apt-get",
        "apt install",
        "do not use apt",
        "package manager",
    )
    non_environment = [item for item in change_types if item != "environment"]
    current_has_environment_change = any(marker in delta_text for marker in env_markers)
    if "environment" in change_types:
        if current_has_environment_change:
            return ["environment"]
        if summary_text and non_environment:
            return non_environment
    return change_types


def _select_reusable_commands(record: dict[str, Any], *, limit: int = 3) -> list[str]:
    candidates = _normalize_command_list(
        record.get("solution_commands") or record.get("commands_that_worked"),
        limit=12,
        item_limit=1800,
    )
    if not candidates:
        return []

    def score(command: str) -> tuple[int, int]:
        text = command.lower()
        value = 0
        for needle in (
            "answer.txt",
            "/app/results",
            "printf",
            "autotokenizer",
            "qwen",
            "deepseek",
            "metadata",
            "read_parquet",
            "load_dataset",
            "hf_hub_download",
            "domain",
            "token",
            "value_counts",
        ):
            if needle in text:
                value += 2
        for setup_only in ("pwd && ls", "importlib.util", "command -v", "list_repo_files"):
            if setup_only in text:
                value -= 2
        if text.strip() in {"c-c", "^c"}:
            value -= 10
        return (value, len(command))

    ranked = sorted(enumerate(candidates), key=lambda item: (score(item[1]), -item[0]), reverse=True)
    selected = [command for _, command in ranked if score(command)[0] > 0][:limit]
    if not selected:
        selected = candidates[:limit]
    return selected


def _render_family_memory_file(
    context: dict[str, Any],
    *,
    char_budget: int = MAX_AGENT_FAMILY_MEMORY_CHARS,
    include_current: bool = True,
) -> str:
    base = context.get("base_memory") or context.get("previous_memory")
    patches = list(context.get("relevant_patches") or context.get("prior_patches") or [])[:MAX_RENDERED_PATCHES]

    lines = [
        "# Family Memory",
        "",
        "Use this compact memory as guidance only. The current task instruction is authoritative.",
        "",
    ]

    if include_current:
        deltas = _normalize_list(context.get("delta_summary"), limit=4, item_limit=220)
        lines.extend(["## Current Differences", ""])
        if deltas:
            lines.extend(f"- {delta}" for delta in deltas)
        else:
            lines.append("- No explicit current-task difference was detected.")
        lines.append("")

    lines.extend(["## Base Execution Ledger", ""])
    if isinstance(base, dict) and _memory_record_is_usable(base):
        goal = str(base.get("task_goal") or base.get("execution_summary") or "").strip()
        if goal:
            lines.append(f"- Prior goal: {_redact_prior_output_hints(goal[:260])}")
        key_files = _normalize_list(base.get("key_files"), limit=5, item_limit=160)
        artifacts = _normalize_list(base.get("artifacts"), limit=5, item_limit=160)
        if key_files:
            lines.append(f"- Files/data used: {_trim_list(key_files, limit=5)}")
        if artifacts:
            lines.append(
                f"- Artifacts produced: {_redact_prior_output_hints(_trim_list(artifacts, limit=5))}"
            )
        commands = _select_reusable_commands(base, limit=3)
        if commands:
            lines.append("- Prior commands/scripts observed:")
            for command in commands:
                _render_command_block(lines, command)
    else:
        lines.append("- No concrete prior execution is available yet.")
    lines.append("")

    lines.extend(["## Base Recipe", ""])
    if isinstance(base, dict) and _memory_record_is_usable(base):
        recipe_steps = _normalize_list(base.get("recipe_steps"), limit=8, item_limit=220)
        if not recipe_steps:
            recipe_steps = _derive_recipe_steps(
                task_memory=base,
                solution_commands=_normalize_command_list(base.get("solution_commands"), limit=6),
                worked=_normalize_command_list(base.get("commands_that_worked"), limit=6),
                execution_summary=str(base.get("execution_summary") or ""),
            )
        if recipe_steps:
            for idx, step in enumerate(recipe_steps, start=1):
                lines.append(f"{idx}. {_redact_prior_output_hints(step)}")
        else:
            execution = str(base.get("execution_summary") or "").strip()
            fallback = execution[:240] if execution else "Infer the solution from the current task."
            lines.append(f"1. {_redact_prior_output_hints(fallback)}")

        artifacts = _normalize_list(base.get("artifacts"), limit=4, item_limit=120)
        verification = str(base.get("verification_summary") or "").strip()
        if artifacts:
            lines.append(
                f"- Final artifacts from base task: {_redact_prior_output_hints(_trim_list(artifacts, limit=4))}"
            )
    else:
        lines.append("- No usable prior base recipe is available yet.")
    lines.append("")

    lines.extend(["## Relevant Patches", ""])
    actionable = [record for record in patches if _patch_record_is_actionable(record)]
    if actionable:
        for idx, record in enumerate(actionable, start=1):
            lines.extend(_render_transition_example_lines(record, idx))
    else:
        lines.append("- No relevant prior patches matched this task.")
    lines.append("")

    lines.extend(["## Current Recommendation", ""])
    if actionable:
        lines.append(
            "Treat the prior patches as transition examples only. Infer the current task's own path, format, input, and environment requirements from the current instruction."
        )
    else:
        lines.append("Use the base recipe only when it fits the current task; otherwise solve from the instruction.")
    lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if len(text) <= char_budget:
        return text
    clipped = text[:char_budget].rsplit("\n", 1)[0].rstrip()
    return clipped + "\n\n...[family memory truncated]\n"


def _latest_actionable_patch_library(paths: FamilyPaths, *, limit: int = MAX_RENDERED_PATCHES) -> list[dict[str, Any]]:
    order = _release_order_lookup(paths)
    patches = [record for record in load_memory_patches(paths) if _patch_record_is_actionable(record)]
    patches.sort(
        key=lambda record: (
            order.get(str(record.get("to_task") or ""), 10**9),
            float(record.get("created_ts") or 0.0),
        ),
        reverse=True,
    )
    return list(reversed(patches[:limit]))


def write_family_memory_document(paths: FamilyPaths, *, char_budget: int = MAX_AGENT_FAMILY_MEMORY_CHARS) -> str:
    context = {
        "base_memory": _base_task_memory(paths),
        "relevant_patches": _latest_actionable_patch_library(paths),
        "delta_summary": [],
    }
    rendered = _render_family_memory_file(context, char_budget=char_budget, include_current=False)
    paths.family_memory_md.write_text(rendered, encoding="utf-8")
    return rendered


def family_context_has_reusable_memory(context: dict[str, Any]) -> bool:
    base = context.get("base_memory") or context.get("previous_memory")
    patches = list(context.get("relevant_patches") or context.get("prior_patches") or [])
    return bool(
        (isinstance(base, dict) and _memory_record_is_usable(base))
        or any(_patch_record_is_actionable(record) for record in patches if isinstance(record, dict))
    )


def build_family_workspace_files(
    context: dict[str, Any],
    *,
    workspace_dir: str = DEFAULT_FAMILY_WORKSPACE_DIR,
) -> dict[str, str]:
    del workspace_dir
    return {
        FAMILY_MEMORY_FILE: _render_family_memory_file(context),
    }


def render_family_pointer_markdown(
    context: dict[str, Any],
    *,
    workspace_dir: str = DEFAULT_FAMILY_WORKSPACE_DIR,
) -> str:
    return "\n".join(
        [
            "### Memory References",
            "",
            "Compact family memory is available at:",
            f"`{workspace_dir}/family_memory.md`",
            "",
            "Read that file once before solving. It contains a prior execution ledger, base recipe plus relevant conditional patches.",
            "Use it as guidance only; the current task instruction is authoritative.",
        ]
    ).strip() + "\n"


async def hydrate_family_workspace_files(
    environment: BaseEnvironment,
    context: dict[str, Any],
    *,
    workspace_dir: str = DEFAULT_FAMILY_WORKSPACE_DIR,
    timeout_sec: int = 60,
    logger: Any = None,
) -> dict[str, str]:
    files = build_family_workspace_files(context, workspace_dir=workspace_dir)
    await hydrate_workspace_text_files(
        environment,
        files,
        timeout_sec=timeout_sec,
        logger=logger,
    )
    return files


async def hydrate_workspace_text_files(
    environment: BaseEnvironment,
    files: dict[str, str],
    *,
    timeout_sec: int = 60,
    logger: Any = None,
) -> None:
    dirs = sorted({str(Path(path).parent) for path in files})
    commands = [f"mkdir -p {shlex.quote(directory)}" for directory in dirs]
    for idx, (path, content) in enumerate(files.items()):
        delim = f"__HARBOR_EVOMEM_FAMILY_{idx}_{uuid.uuid4().hex}__"
        path_q = shlex.quote(path)
        commands.append(f"cat > {path_q} <<'{delim}'\n{content}\n{delim}")
    await _exec_environment(
        environment,
        "\n".join(commands),
        timeout_sec=timeout_sec,
        logger=logger,
    )


def render_family_context_markdown(
    context: dict[str, Any],
    *,
    probe_md: str,
    char_budget: int,
) -> str:
    del probe_md
    text = "\n".join(
        [
            "### Memory References",
            "",
            "Workspace hydration failed, so the same memory is shown inline below.",
            "Treat this compact family memory as guidance only.",
            "",
            _render_family_memory_file(context, char_budget=char_budget).strip(),
        ]
    ).strip() + "\n"
    if len(text) <= char_budget:
        return text
    clipped = text[:char_budget].rsplit("\n", 1)[0]
    return clipped + "\n\n...[chain memory truncated]"


__all__ = [
    "CLASSIC_PATCH_INDEX_FILE",
    "CLASSIC_PATCH_MEMORY_FILE",
    "DEFAULT_FAMILY_WORKSPACE_DIR",
    "FAMILY_MEMORY_FILE",
    "FamilyPaths",
    "build_current_signals",
    "build_classic_workspace_files",
    "build_family_workspace_files",
    "family_context_has_reusable_memory",
    "family_terminal_dir",
    "harvest_commands",
    "hydrate_family_workspace_files",
    "hydrate_workspace_text_files",
    "latest_prior_task_memory",
    "load_memory_patches",
    "load_task_memories",
    "persist_instruction_snapshot",
    "previous_release_snapshot_task",
    "probe_environment_for_family",
    "probe_summary_md",
    "prototype_task_memory",
    "render_classic_patch_pointer",
    "render_family_context_markdown",
    "render_family_pointer_markdown",
    "retrieve_family_context",
    "summarize_classic_evomem",
    "signals_from_probe",
    "store_family_memory",
    "task_diff",
    "try_read_verifier_reward",
    "update_family_state",
    "write_family_memory_document",
]
