#!/usr/bin/env python3
"""Build JSONL manifests for unfinished tasks under a results directory."""

import argparse
import hashlib
import json
from pathlib import Path
from local_paths import RESULTS_DIR 


def stable_task_key(row: dict) -> str:
    data = dict(row)
    data.pop("task_id", None)
    data.pop("task_key", None)
    data.pop("slurm", None)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def pending_rows(tasks_path: Path):
    root = tasks_path.parent
    with tasks_path.open() as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("task_key") or stable_task_key(row)
            if (root / key / "DONE").exists():
                continue
            row = dict(row)
            row["task_key"] = key
            yield row


def main():
    base = RESULTS_DIR / 'basic_sudoku/sweep'
    if not base.exists():
        raise SystemExit(f"{base} does not exist")

    manifests = sorted(base.rglob("tasks.jsonl"))
    if not manifests:
        raise SystemExit(f"No tasks.jsonl files found under {base}")

    total_pending = 0
    for tasks_path in manifests:
        pending = list(pending_rows(tasks_path))
        out_path = tasks_path.parent / "pending_manifest.jsonl"
        with out_path.open("w") as dst:
            for row in pending:
                dst.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        total_pending += len(pending)
        print(f"{tasks_path}: wrote {len(pending)} pending tasks to {out_path}")

    print(f"Total pending tasks: {total_pending}")


if __name__ == "__main__":
    main()
