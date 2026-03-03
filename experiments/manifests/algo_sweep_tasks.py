#!/usr/bin/env python
"""Generate JSONL task manifest for guided MDM sweeps."""

import argparse
import json, hashlib
from datetime import datetime
from itertools import cycle
from pathlib import Path
import hydra
from itertools import product

from omegaconf import OmegaConf
from local_paths import RESULTS_DIR

def jsonl_dump(d: dict):
    return json.dumps(d, sort_keys=True, separators=(",", ":"))

def task_key(row: dict) -> str:
    # hash everything that defines the task.
    return hashlib.sha1(jsonl_dump(row).encode("utf-8")).hexdigest()[:12]

def mk_product(d): 
    keys = list(d.keys())
    new_keys = [f"sampler.{key}" for key in d]
    vals = [d[k] for k in keys]
    return [dict(zip(new_keys, combo)) for combo in product(*vals)]


def load_existing_keys(path: Path) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # support older files that didn't have task_key stored
            if "task_key" in obj:
                keys.add(obj["task_key"])
            else:
                obj.pop("task_id", None)
                keys.add(task_key(obj))
    return keys

@hydra.main(config_path="../../configs", config_name="basic_sudoku_sweep", version_base=None)
def main(cfg):

    acc_list = cfg.model_acc
    seeds = OmegaConf.to_container(cfg.seeds, resolve=True)

    for sampler in cfg.sampler: 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{sampler}_sweep"

        results_dir = RESULTS_DIR
        results_root = results_dir / cfg.results_dir / run_name 
        results_root.mkdir(parents=True, exist_ok=True)
        output_path = results_root / "tasks.jsonl"
        
        existing = load_existing_keys(output_path)
        new_rows = []

        params_combo = mk_product(cfg.sampler[sampler])
        for params_dict in params_combo:
            for lr in cfg.learning_rates:
                for acc in acc_list:
                    row = params_dict.copy()
                    row.update({
                        "sampler": sampler,
                        "lr": lr,
                        "acc": acc,
                        "seeds": seeds,
                        "grad_steps": OmegaConf.to_container(cfg.grad_steps_list, resolve=True),
                    })
                    key = task_key(row)
                    if key in existing: 
                        continue 
                    row['task_key'] = key
                    new_rows.append(row)
                    existing.add(key)

        # append to manifest
        with output_path.open("a") as f:
            for r in new_rows:
                f.write(jsonl_dump(r) + "\n")

        print(f"Appended {len(new_rows)} new tasks to {output_path}")


if __name__ == "__main__":
    main()
