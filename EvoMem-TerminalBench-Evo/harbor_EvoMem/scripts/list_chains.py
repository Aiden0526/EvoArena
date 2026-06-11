#!/usr/bin/env python3
"""Enumerate every chain in a Terminal-shift dataset.

A chain is one prototype task plus all of its variants:

* ``foo``                 (prototype)
* ``foo-EVO-1`` ... ``foo-EVO-N``
* ``foo-v-<token>``       (variant token may itself contain hyphens,
                           e.g. ``foo-v-registers-xyz``)

We discover chains by walking the dataset directory and matching each
sibling against the prototype name. A directory whose computed base
does **not** exist as a sibling is treated as its own (single-task)
chain — that's how the unusual ``install-windows-3.11`` /
``install-windows-v-imgpath`` layout (no shared prototype) shows up:
five singleton chains rather than one bogus group.

Output formats:

* ``--format names``  (default): one chain id per line (the chain id is
  the prototype task directory name — pass it straight to
  ``run_chain.py --chain X``).
* ``--format json``: dumps ``{chain_id: [task_name, ...]}`` to stdout.

The launcher scripts use ``--format names`` to feed chains to ``xargs``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_EVO_RE = re.compile(r"^(?P<base>.+?)-EVO-\d+$", re.IGNORECASE)
_V_RE = re.compile(r"^(?P<base>.+?)-v-[A-Za-z0-9_-]+$")


def discover_chains(dataset: Path) -> dict[str, list[str]]:
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    all_dirs = sorted(
        p.name for p in dataset.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    all_set = set(all_dirs)

    chains: dict[str, list[str]] = {}
    for name in all_dirs:
        m_evo = _EVO_RE.match(name)
        m_v = _V_RE.match(name)
        if m_evo and m_evo.group("base") in all_set:
            base = m_evo.group("base")
        elif m_v and m_v.group("base") in all_set:
            base = m_v.group("base")
        else:
            base = name
        chains.setdefault(base, []).append(name)
    return chains


def main() -> int:
    from harbor_EvoMem.memory_bridge import default_dataset_root

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_root(),
        help="Terminal-Bench-Evo dataset root (default: ../Terminal-Bench-Evo).",
    )
    parser.add_argument(
        "--format",
        choices=("names", "json", "count"),
        default="names",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        default=True,
        help="(default on) Include chains with only the prototype, no variants.",
    )
    parser.add_argument(
        "--no-singletons",
        action="store_false",
        dest="include_singletons",
        help="Drop singleton chains (EvoMem has nothing to show off there).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict output to specific chain id(s). Can repeat.",
    )
    args = parser.parse_args()

    chains = discover_chains(args.dataset)
    if not args.include_singletons:
        chains = {k: v for k, v in chains.items() if len(v) > 1}
    if args.only:
        wanted = set(args.only)
        chains = {k: v for k, v in chains.items() if k in wanted}
        missing = wanted - set(chains)
        if missing:
            print(f"# warning: requested chain(s) not found: {sorted(missing)}", file=sys.stderr)

    if args.format == "names":
        for chain_id in sorted(chains):
            print(chain_id)
    elif args.format == "json":
        json.dump(chains, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.format == "count":
        total_tasks = sum(len(v) for v in chains.values())
        print(f"chains={len(chains)} tasks={total_tasks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
