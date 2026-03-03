#!/usr/bin/env python3
"""Generate MaskGIT + CLIP task manifests with sampler-based folders."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from local_paths import RESULTS_DIR


def jsonl_dump(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_key(row: dict) -> str:
    data = dict(row)
    data.pop("task_key", None)
    payload = jsonl_dump(data)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_prompts(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [{"prompt_index": idx, **item} for idx, item in enumerate(data)]


def chunk(prompts: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        raise ValueError("n_samples_per_job must be > 0")
    return [prompts[i : i + size] for i in range(0, len(prompts), size)]


def param_grid(options: dict) -> list[dict]:
    if not options:
        return [{}]
    keys = sorted(options.keys())
    grid = []
    for values in product(*[options[k] for k in keys]):
        grid.append({k: v for k, v in zip(keys, values)})
    return grid


def clean(value) -> str:
    txt = str(value)
    return txt.replace("/", "-").replace(" ", "_")


@hydra.main(config_path="../../configs", config_name="clip_sweep", version_base=None)
def main(cfg: DictConfig):
    prompts_path = Path(cfg.prompt_file)
    if not prompts_path.is_absolute(): 
        prompts_path = Path.cwd() / prompts_path
    prompts = load_prompts(prompts_path)
    if not prompts:
        raise SystemExit("No prompts found.")

    per_job = int(cfg.n_samples_per_job)
    runs = chunk(prompts, per_job)

    run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(RESULTS_DIR) / cfg.results_dir / run_name
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "tasks.jsonl"

    existing = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add(row.get("task_key") or make_key(row))

    seeds = OmegaConf.to_container(cfg.seeds, resolve=True)
    lrs = OmegaConf.to_container(cfg.learning_rates, resolve=True)
    steps = OmegaConf.to_container(cfg.grad_steps_list, resolve=True)
    sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)

    new_rows = []
    steps_list = list(steps)
    for sampler_name, sampler_opts in sampler_cfg.items():
        sampler_dir = root / sampler_name
        for overrides in param_grid(sampler_opts or {}):
            override_tokens = [f"{k.replace('.', '_')}={clean(v)}" for k, v in overrides.items()]
            override_tag = "__".join(override_tokens) if override_tokens else "base"
            for lr in lrs:
                for seed in seeds:
                    steps_token = "steps_seq=" + "-".join(clean(step) for step in steps_list)
                    dir_parts = [override_tag, f"lr={clean(lr)}", steps_token, f"seed={clean(seed)}"]
                    task_dir = sampler_dir / "__".join(dir_parts)
                    run_manifest = task_dir / "runs.jsonl"
                    row = {
                        "sampler": sampler_name,
                        "lr": float(lr),
                        "grad_steps": list(steps_list),
                        "seed": int(seed),
                        "prompt_file": str(prompts_path),
                        "n_prompts": len(prompts),
                        "n_samples_per_job": per_job,
                        "n_runs": len(runs),
                        "task_dir": str(task_dir.relative_to(root)),
                        "run_manifest": str(run_manifest.relative_to(root)),
                    }
                    for key, value in overrides.items():
                        row[f"sampler.{key}"] = value
                    key = make_key(row)
                    if key in existing:
                        continue

                    task_dir.mkdir(parents=True, exist_ok=True)
                    if not run_manifest.exists():
                        with run_manifest.open("w") as out:
                            for idx, batch in enumerate(runs):
                                entry = {
                                    "run_name": f"run_{idx:02d}",
                                    "run_dir": f"run_{idx:02d}",
                                    "run_index": idx,
                                    "prompt_indices": [item["prompt_index"] for item in batch],
                                    "prompts": batch,
                                }
                                out.write(jsonl_dump(entry) + "\n")

                    row["task_key"] = key
                    new_rows.append(row)
                    existing.add(key)

    if not new_rows:
        print("No new tasks.")
        return

    with manifest_path.open("a") as dst:
        for row in new_rows:
            dst.write(jsonl_dump(row) + "\n")

    print(f"Appended {len(new_rows)} tasks to {manifest_path}")


if __name__ == "__main__":
    main()
