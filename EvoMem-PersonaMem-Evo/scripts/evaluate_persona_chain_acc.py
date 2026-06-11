#!/usr/bin/env python3
"""Compute chain exact-match accuracy for Persona result CSVs.

Use this after running robust/patch evaluations on a benchmark that contains chain_id.
The Persona runners copy benchmark columns into their result CSVs, so chain_id will be
available automatically when the annotated benchmark is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def detect_correct_col(fieldnames: list[str], size: str | None) -> str:
    if size:
        col = f"is_correct_mcq_{size}"
        if col in fieldnames:
            return col
        raise ValueError(f"Column not found: {col}")
    candidates = [field for field in fieldnames if field.startswith("is_correct_mcq_")]
    if not candidates:
        raise ValueError("No is_correct_mcq_* column found")
    return candidates[0]


def evaluate_file(path: Path, chain_col: str, size: str | None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if chain_col not in fieldnames:
        raise ValueError(f"{path} has no {chain_col!r} column")
    correct_col = detect_correct_col(fieldnames, size)

    processed = [row for row in rows if row.get(correct_col) in ("True", "False")]
    qa_correct = sum(1 for row in processed if row.get(correct_col) == "True")

    chains: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_chain_id = 0
    for row in processed:
        chain_id = normalize(row.get(chain_col, ""))
        if not chain_id:
            missing_chain_id += 1
            continue
        chains[chain_id].append(row)

    chain_pass = sum(1 for chain_rows in chains.values() if all(row[correct_col] == "True" for row in chain_rows))
    size_counter = Counter(len(chain_rows) for chain_rows in chains.values())

    return {
        "path": str(path),
        "rows": len(rows),
        "processed_qa": len(processed),
        "qa_correct": qa_correct,
        "qa_acc": qa_correct / len(processed) if processed else 0.0,
        "chains": len(chains),
        "chain_pass": chain_pass,
        "chain_acc": chain_pass / len(chains) if chains else 0.0,
        "missing_chain_id": missing_chain_id,
        "chain_size_distribution": ", ".join(f"{size}q:{count}" for size, count in sorted(size_counter.items())),
        "correct_col": correct_col,
        "chain_col": chain_col,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute chain exact-match accuracy from Persona result CSVs.")
    parser.add_argument("results", nargs="+", type=Path, help="Result CSV path(s)")
    parser.add_argument("--chain_col", default="chain_id")
    parser.add_argument("--size", default=None, help="Optional size suffix, e.g. 32k")
    parser.add_argument("--output", type=Path, default=None, help="Optional summary CSV output")
    args = parser.parse_args()

    summaries = []
    for path in args.results:
        summary = evaluate_file(path, args.chain_col, args.size)
        summaries.append(summary)
        print(
            f"{path}: QA {summary['qa_correct']}/{summary['processed_qa']}={summary['qa_acc']:.3f}; "
            f"Chain {summary['chain_pass']}/{summary['chains']}={summary['chain_acc']:.3f}; "
            f"sizes: {summary['chain_size_distribution']}"
        )
        if summary["missing_chain_id"]:
            print(f"  WARNING: {summary['missing_chain_id']} processed rows missing {args.chain_col}")

    if args.output:
        fieldnames = [
            "path",
            "rows",
            "processed_qa",
            "qa_correct",
            "qa_acc",
            "chains",
            "chain_pass",
            "chain_acc",
            "missing_chain_id",
            "chain_size_distribution",
            "correct_col",
            "chain_col",
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                row = dict(summary)
                row["qa_acc"] = f"{row['qa_acc']:.6f}"
                row["chain_acc"] = f"{row['chain_acc']:.6f}"
                writer.writerow(row)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
