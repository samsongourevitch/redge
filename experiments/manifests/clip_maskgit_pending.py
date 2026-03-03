#!/usr/bin/env python3
"""Build pending task manifest for MaskGIT + CLIP sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from local_paths import RESULTS_DIR


def jsonl_dump(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def task_key(row: dict) -> str:
    data = dict(row)
    data.pop("task_key", None)
    payload = jsonl_dump(data)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def read_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open() as src:
        for line in src:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pending_rows(tasks_path: Path):
    root = tasks_path.parent
    for row in read_rows(tasks_path):
        key = row.get("task_key") or task_key(row)
        task_dir = root / row["task_dir"]
        manifest_path = root / row["run_manifest"]
        runs = read_rows(manifest_path)
        for idx, run in enumerate(runs):
            run_dir = task_dir / run["run_dir"]
            if (run_dir / "DONE").exists():
                continue
            out = dict(row)
            out["task_key"] = key
            out["run_name"] = run.get("run_name") or run["run_dir"]
            out["run_dir"] = run["run_dir"]
            out["run_index"] = int(run.get("run_index", idx))
            if "prompt_indices" in run:
                out["prompt_indices"] = run["prompt_indices"]
            if "prompts" in run:
                out["prompts"] = run["prompts"]
            yield out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="maskgit/sweep", help="Relative path under RESULTS_DIR")
    parser.add_argument("--run-name", default=None, help="Specific run directory under --base")
    args = parser.parse_args()

    base = Path(RESULTS_DIR) / args.base
    if args.run_name:
        base = base / args.run_name
    if not base.exists():
        raise SystemExit(f"{base} not found")

    manifests = sorted(base.rglob("tasks.jsonl"))
    if not manifests:
        raise SystemExit("No tasks.jsonl found")

    total = 0
    for tasks_path in manifests:
        pending = list(pending_rows(tasks_path))
        out_path = tasks_path.parent / "pending_manifest.jsonl"
        with out_path.open("w") as dst:
            for row in pending:
                dst.write(jsonl_dump(row) + "\n")
        total += len(pending)
        print(f"{tasks_path}: {len(pending)} pending tasks")

    print(f"Total pending tasks: {total}")


if __name__ == "__main__":
    main()
