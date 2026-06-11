"""Chain-ID detection for Terminal-shift task names.

EvoMem needs a *stable* identifier shared by a prototype task and
all of its variants so that records written while solving the prototype
are visible to subsequent variants. Terminal-shift encodes that
relationship in the task directory name itself, e.g.::

    bn-fit-modify                           (prototype)
    bn-fit-modify-EVO-1                     (variant)
    bn-fit-modify-EVO-2
    ...
    adaptive-rejection-sampler              (prototype)
    adaptive-rejection-sampler-v-rocker     (variant)
    adaptive-rejection-sampler-v-sampledir
    ...

Heuristic precedence:

1. Explicit override via ``HARBOR_EVOMEM_CHAIN_ID`` env var (passed through
   ``harbor trials start --ae HARBOR_EVOMEM_CHAIN_ID=...``).
2. Strip a trailing ``-EVO-<N>`` (case-insensitive, ``N`` numeric).
3. Strip a ``-v-<token>`` suffix (Terminal-shift uses one ``-v-<token>``
   per variant).
4. Otherwise return the task name unchanged (singleton chain).

Anything calling ``derive_chain_id`` should be tolerant of None.
"""

from __future__ import annotations

import os
import re
from typing import Optional


_EVO_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-EVO-\d+$", re.IGNORECASE)
# The variant token after ``-v-`` is allowed to contain hyphens, so e.g.
# ``large-scale-text-editing-v-registers-xyz`` reduces to chain
# ``large-scale-text-editing``. ``.+?`` keeps the base non-greedy so the
# *first* ``-v-`` in the name is taken as the chain boundary.
_V_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-v-[A-Za-z0-9_-]+$")
_OVERRIDE_ENV = "HARBOR_EVOMEM_CHAIN_ID"


def derive_chain_id(
    task_name: Optional[str],
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Best-effort chain id from a task name.

    Parameters
    ----------
    task_name:
        ``environment.environment_name`` from Harbor (==task directory
        name in Terminal-shift). May be ``None`` for non-task contexts.
    explicit:
        Caller-supplied override; usually the value of
        ``$HARBOR_EVOMEM_CHAIN_ID``. Beats automatic detection.
    """
    if explicit:
        return explicit.strip() or None
    env_override = os.environ.get(_OVERRIDE_ENV)
    if env_override:
        return env_override.strip() or None
    if not task_name:
        return None

    name = task_name.strip()
    if not name:
        return None

    m = _EVO_SUFFIX_RE.match(name)
    if m:
        return m.group("base")

    m = _V_SUFFIX_RE.match(name)
    if m:
        return m.group("base")

    return name


__all__ = ["derive_chain_id"]
