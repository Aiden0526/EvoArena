"""Capture code changes inside a Harbor environment via a scratch git repo.

Terminal-shift tasks ship plain Docker images with no pre-existing git
state. To feed something patch-shaped to the EvoMem summariser we
initialise a one-shot git repo at the task work dir before the agent
starts, commit a baseline, then read ``git diff HEAD`` plus
``git diff --name-only HEAD`` after the agent finishes.

The scratch repo lives **only** under the task work dir, never touches
the agent's home directory, and is invisible to the validator (the
validator scripts in Terminal-shift only check generated artefacts, not
git history). We still scope the baseline commit to the work dir so the
image's other system files are not snapshotted into ``.git``.

Everything here runs from the host process via
:meth:`harbor.environments.base.BaseEnvironment.exec`; nothing assumes a
local filesystem.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Any, Optional

from harbor.environments.base import BaseEnvironment


# Default work dir for Terminal-shift; matches every task's Dockerfile WORKDIR
DEFAULT_WORKDIR = "/app"

# Marker file used to store our baseline commit hash so post-run we can
# diff against it explicitly even if the agent fiddles with HEAD.
BASELINE_HASH_FILE = "/tmp/harbor_evomem_baseline_sha"

# Files / patterns ignored when computing the baseline / final diff.
# Keeps the noise out of records (bytecode, virtualenvs, big artefacts).
GITIGNORE_LINES = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".ipynb_checkpoints/",
    "*.egg-info/",
    "node_modules/",
    "venv/",
    ".venv/",
    ".tox/",
    "*.log",
    ".DS_Store",
)


@dataclass
class CapturedDiff:
    """Result of post-run patch capture."""

    diff_text: str
    changed_files: list[str]
    baseline_sha: Optional[str]
    head_sha: Optional[str]


def _q(s: str) -> str:
    return shlex.quote(s)


async def _exec(
    environment: BaseEnvironment,
    command: str,
    *,
    user: str | int | None = "root",
    cwd: str | None = None,
    timeout_sec: int | None = 300,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Wrapper around environment.exec that logs failures but doesn't raise."""
    if logger:
        logger.debug("patch_capture exec: %s", command)
    return await environment.exec(
        command=f"set +e; {command}",
        user=user,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )


async def init_baseline(
    environment: BaseEnvironment,
    *,
    workdir: str = DEFAULT_WORKDIR,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Initialise a git repo at *workdir* and commit the current state.

    Returns the resulting commit sha, or ``None`` if anything went wrong.
    Errors are logged but do not propagate — patch capture is a
    best-effort sidecar and must not break a successful agent run.
    """
    # Make sure git is available. Terminal-shift containers run as root by
    # default, so this is a best-effort setup step.
    install_git = (
        "if ! command -v git >/dev/null 2>&1; then "
        "  apt-get update >/dev/null 2>&1 && "
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git >/dev/null 2>&1 || true; "
        "fi"
    )
    await _exec(environment, install_git, logger=logger)

    quoted_workdir = _q(workdir)
    gitignore = "\n".join(GITIGNORE_LINES) + "\n"
    quoted_gitignore = _q(gitignore)

    bootstrap = f"""
mkdir -p {quoted_workdir}
cd {quoted_workdir}
if [ ! -d .git ]; then
  git init -q .
  git config user.email 'harbor-evomem@local'
  git config user.name  'harbor-evomem'
fi
# Refresh ignore list every time so it stays in sync with this module
printf '%s' {quoted_gitignore} > .gitignore
git add -A
git commit -q -m 'harbor-evomem baseline' --allow-empty || true
git rev-parse HEAD > {_q(BASELINE_HASH_FILE)} || true
cat {_q(BASELINE_HASH_FILE)} 2>/dev/null || true
"""
    result = await _exec(environment, bootstrap, logger=logger)
    sha = (getattr(result, "stdout", "") or "").strip().splitlines()
    return sha[-1] if sha else None


async def capture_diff(
    environment: BaseEnvironment,
    *,
    workdir: str = DEFAULT_WORKDIR,
    logger: Optional[logging.Logger] = None,
) -> CapturedDiff:
    """Capture diff vs. the baseline commit recorded by :func:`init_baseline`.

    Returns an empty :class:`CapturedDiff` if no baseline was ever
    recorded (e.g. ``init_baseline`` was skipped or failed). Any
    failure of the underlying git commands is swallowed — we always
    return a value so the caller can keep marching.
    """
    quoted_workdir = _q(workdir)
    quoted_baseline_file = _q(BASELINE_HASH_FILE)

    # Stage everything the agent left behind so even un-committed
    # working-tree changes show up in the diff.
    capture = f"""
cd {quoted_workdir} 2>/dev/null || exit 0
[ -d .git ] || exit 0
BASELINE=$(cat {quoted_baseline_file} 2>/dev/null || echo "")
git add -A 2>/dev/null
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
echo '__HARBOR_EVOMEM_DELIM__:baseline'
echo "$BASELINE"
echo '__HARBOR_EVOMEM_DELIM__:head'
echo "$HEAD_SHA"
echo '__HARBOR_EVOMEM_DELIM__:files'
if [ -n "$BASELINE" ]; then
  git diff --name-only "$BASELINE" -- 2>/dev/null
  git diff --cached --name-only "$BASELINE" -- 2>/dev/null
else
  git diff --name-only HEAD -- 2>/dev/null
  git diff --cached --name-only HEAD -- 2>/dev/null
fi
echo '__HARBOR_EVOMEM_DELIM__:diff'
if [ -n "$BASELINE" ]; then
  git --no-pager diff "$BASELINE" -- 2>/dev/null
  git --no-pager diff --cached "$BASELINE" -- 2>/dev/null
else
  git --no-pager diff HEAD -- 2>/dev/null
  git --no-pager diff --cached HEAD -- 2>/dev/null
fi
"""
    result = await _exec(environment, capture, logger=logger)
    raw = getattr(result, "stdout", "") or ""

    sections = _split_sections(raw)
    baseline = sections.get("baseline", "").strip() or None
    head = sections.get("head", "").strip() or None
    files_block = sections.get("files", "")
    diff_text = sections.get("diff", "")

    seen: set[str] = set()
    changed_files: list[str] = []
    for line in files_block.splitlines():
        path = line.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        changed_files.append(path)

    return CapturedDiff(
        diff_text=diff_text.strip("\n"),
        changed_files=changed_files,
        baseline_sha=baseline,
        head_sha=head,
    )


def _split_sections(raw: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []
    for line in raw.splitlines():
        if line.startswith("__HARBOR_EVOMEM_DELIM__:"):
            if current is not None:
                sections[current] = "\n".join(buf)
            current = line.split(":", 1)[1]
            buf = []
            continue
        if current is None:
            continue
        buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    return sections


__all__ = [
    "CapturedDiff",
    "DEFAULT_WORKDIR",
    "BASELINE_HASH_FILE",
    "init_baseline",
    "capture_diff",
]
