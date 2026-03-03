#!/usr/bin/env python3
import hashlib
import hydra
import json
import linecache
import os
import time
from dataclasses import dataclass
from datasets import load_from_disk
from pathlib import Path
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from local_paths import REPO_DIR, DATA_DIR
from samplers import SAMPLERS
from experiments.sudoku.guidance import sudoku_sampler
from experiments.sudoku.sudoku_utils import build_unit_indices, count_violations_batch, SimpleSudokuDataset
from experiments.utils import fix_seed


def jsonl_dump(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_task_key(row: dict) -> str:
    # Exclude purely metadata fields if present
    r = dict(row)
    r.pop("task_id", None)
    r.pop("slurm", None)
    s = jsonl_dump(r)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def read_jsonl_line(path: Path, index0: int) -> dict:
    # linecache is 1-based; returns "" on missing line
    line = linecache.getline(str(path), index0 + 1)
    if not line:
        raise IndexError(f"Index {index0} out of range for {path}")
    return json.loads(line)


def fmt_scalar(x) -> str:
    # stable, filesystem-friendly formatting
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)


def soft_penalty_wrapper(samples, row_idx, col_idx): 
    # samples: [B, 81, 9]
    B, V = samples.shape[0], samples.shape[-1]
    sudoku_grid = samples.view(B, 9, 9, V)
    units_sum = sudoku_grid[:, row_idx, col_idx, :].sum(dim=-2)
    return (units_sum - 1.0).square().sum() 


@dataclass
class Task:
    sampler: str
    sampler_overrides: dict
    lr: float
    acc: float
    seeds: list
    grad_steps: list
    extra_overrides: dict
    task_key: str
    manifest_row: dict
    gpu_id: Optional[int] = None


def task_from_row(row: dict) -> Task:
    # Your manifest uses "hyperparam"+"value" to represent one override
    sampler = row["sampler"]
    sampler_overrides = {}

    hps = [key for key in list(row.keys()) if key.startswith('sampler.')]
    for hp in hps: 
        sampler_overrides[hp] = row[hp]

    # merge any extra overrides (optional)
    extra = row.get("extra_overrides") or {}
    # if you want extra overrides to override the simple hp/value, do update after
    sampler_overrides.update(extra.get("sampler_overrides", {}))

    grad_steps = row["grad_steps"]
    # OmegaConf ListConfig -> plain list if needed
    if not isinstance(grad_steps, list):
        grad_steps = OmegaConf.to_container(grad_steps, resolve=True)

    seeds = row.get("seeds")
    if seeds is None:
        seeds = [int(row["seed"])]
    if not isinstance(seeds, list):
        seeds = OmegaConf.to_container(seeds, resolve=True)
    seeds = [int(s) for s in seeds]

    key = row.get("task_key") or stable_task_key(row)

    gpu_id = None
    try:
        # Slurm often sets CUDA_VISIBLE_DEVICES to a single id like "0"
        # so this is mostly for logging.
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
    except Exception:
        pass

    return Task(
        sampler=sampler,
        sampler_overrides=sampler_overrides,
        lr=float(row["lr"]),
        acc=float(row["acc"]),
        seeds=seeds,
        grad_steps=list(grad_steps),
        extra_overrides=row.get("extra_overrides") or {},
        task_key=key,
        manifest_row=row,
        gpu_id=gpu_id,
    )


def resolve_index(cfg) -> int:
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_id is not None:
        return int(env_id)
    idx = OmegaConf.select(cfg, "index")
    if idx is None:
        raise ValueError("Set index or provide SLURM_ARRAY_TASK_ID.")
    idx = int(idx)
    if idx < 0:
        raise ValueError("index must be >= 0 when no SLURM array id is provided.")
    return idx


def task_output_dir(results_root: Path, task: Task) -> Path:
    # Adjust to your taste; the important thing is determinism + readability
    parts = [
        f"sampler={task.sampler}",
        f"acc={fmt_scalar(task.acc)}",
        f"lr={fmt_scalar(task.lr)}",
    ]
    # include one simple override in path if you like
    if len(task.sampler_overrides) == 1:
        k, v = next(iter(task.sampler_overrides.items()))
        parts.insert(1, f"{k}={fmt_scalar(v)}")

    return results_root.joinpath(*parts, f"task={task.task_key}")


def apply_overrides(base_cfg, task: Task):
    """
    Returns a NEW OmegaConf config derived from base_cfg, with task overrides applied.
    Handles dotted keys like "schedule.t_1".
    """
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    # sampler overrides (dotted keys are fine with OmegaConf.update)
    for k, v in task.sampler_overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)

    # any other overrides you keep in the manifest
    # e.g., {"guidance": {"foo": 1}} or {"some.dotted": 2}
    extra = task.extra_overrides or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            OmegaConf.update(cfg, k, v, merge=True)

    return cfg

@hydra.main(config_path="../configs", config_name="basic_sudoku_sweep", version_base=None)
def main(cfg):
    manifest_path = Path(cfg.pending_manifest)
    index0 = resolve_index(cfg)

    row = read_jsonl_line(manifest_path, index0)
    task = task_from_row(row)

    results_root = manifest_path.parent
    results_root.mkdir(parents=True, exist_ok=True)

    run_dir = results_root / task.task_key
    run_dir.mkdir(parents=True, exist_ok=True)

    done_path = run_dir / "DONE"
    if done_path.exists():
        print(f"[index={index0}] DONE exists -> skipping: {run_dir}")
        return 0

    base_cfg = OmegaConf.load(REPO_DIR / "configs/guided_mdm.yaml")
    sampler_cfg_path = REPO_DIR / "configs/sampler" / f"{task.sampler}.yaml"
    if not sampler_cfg_path.exists():
        raise FileNotFoundError(f"Sampler config not found: {sampler_cfg_path}")
    sampler_cfg = OmegaConf.load(sampler_cfg_path)
    OmegaConf.update(base_cfg, "sampler", sampler_cfg, merge=False)

    OmegaConf.update(base_cfg, "guidance.lr", task.lr, merge=False)

    hcfg = apply_overrides(base_cfg, task)

    configured_device = OmegaConf.select(hcfg, "device")
    backend_available = torch.cuda.is_available()
    if configured_device:
        requested_device = torch.device(str(configured_device))
        if requested_device.type in {"cuda", "hip"} and not backend_available:
            print(
                f"Requested device '{requested_device}' but no CUDA/HIP backend is available. Falling back to CPU.",
                flush=True,
            )
            device = torch.device("cpu")
        else:
            device = requested_device
    else:
        device = torch.device("cuda" if backend_available else "cpu")

    torch.set_default_device(device)
    gpu_like = device.type in {"cuda", "hip"}
    gpu_available = backend_available
    device_index = device.index if device.index is not None else 0

    if gpu_like and gpu_available:
        torch.cuda.set_device(device_index)

    track_mem_cfg = OmegaConf.select(cfg, "track_mem")
    if track_mem_cfg is None:
        track_mem = gpu_like and gpu_available
    else:
        track_mem = bool(track_mem_cfg)
    track_mem = track_mem and gpu_like and gpu_available

    max_samples, batch_size = hcfg.max_samples, hcfg.batch_size 
    dataset = load_from_disk(DATA_DIR / "validation")
    ds_small = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    dataset = SimpleSudokuDataset(ds_small)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    row_idxs, col_idxs = build_unit_indices()
    row_idxs = row_idxs.to(device)
    col_idxs = col_idxs.to(device)
    penalty_fn = partial(soft_penalty_wrapper, row_idx=row_idxs, col_idx=col_idxs)

    sampler_fn = SAMPLERS[hcfg.sampler.sampler_name]

    (run_dir / "task_manifest.json").write_text(json.dumps(task.manifest_row, indent=2))

    single_value = None
    if len(task.sampler_overrides) == 1:
        single_value = next(iter(task.sampler_overrides.values()))

    seed_summaries = []
    run_start = time.time()

    for seed in task.seeds:
        fix_seed(seed)

        per_step = []
        global_peak_bytes = 0
        global_peak_mb = 0.0

        for grad_steps in task.grad_steps:
            step_cfg = OmegaConf.create(OmegaConf.to_container(hcfg, resolve=True))
            step_cfg.guidance.n_opt_steps = int(grad_steps)
            step_cfg.guidance.lr = float(task.lr)

            if track_mem:
                torch.cuda.reset_peak_memory_stats(device_index)

            n_violations = 0.0
            n_correct = 0.0
            total = 0
            t0 = time.time()

            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                samples = sudoku_sampler(
                    sampler=sampler_fn, 
                    clues=batch['prompt'], 
                    config=step_cfg,
                    penalty_fn=penalty_fn,
                    )
                violations, correct = count_violations_batch(samples, row_idxs, col_idxs)
                n_violations += violations.item()
                n_correct += correct
                total += samples.size(0)

                print(f"{hcfg.sampler.sampler_name} n_violations: {n_violations / 1000} | correct: {n_correct / 1000}")

            mean_violations = n_violations / total if total > 0 else float("nan")
            correct_rate = n_correct / total if total > 0 else float("nan")
            peak_mem_bytes = None
            peak_mem_mb = None
            if track_mem:
                torch.cuda.synchronize(device_index)
                peak_mem_bytes = torch.cuda.max_memory_allocated(device_index)
                peak_mem_mb = peak_mem_bytes / (1024 ** 2)
                global_peak_bytes = max(global_peak_bytes, peak_mem_bytes)
                global_peak_mb = max(global_peak_mb, peak_mem_mb)

            per_step.append(
                {
                    "seed": seed,
                    "grad_steps": int(grad_steps),
                    "mean_violations": mean_violations,
                    "correct_rate": correct_rate,
                    "elapsed_s": time.time() - t0,
                    "peak_memory_bytes": peak_mem_bytes,
                    "peak_memory_mib": peak_mem_mb,
                }
            )

        seed_summaries.extend(per_step)

    metrics_path = run_dir / "metrics_steps.jsonl"
    metrics_path.write_text("\n".join(json.dumps(m) for m in seed_summaries) + "\n")
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(OmegaConf.to_container(hcfg, resolve=True), indent=2))

    summary_path = run_dir / "summary.json"
    summary = {
        "run_dir": str(run_dir),
        "pending_manifest": str(manifest_path),
        "index": index0,
        "sampler": task.sampler,
        "sampler_overrides": task.sampler_overrides,
        "value": single_value,
        "lr": task.lr,
        "acc": task.acc,
        "seeds": task.seeds,
        "grad_steps": task.grad_steps,
        "gpu_id": task.gpu_id,
        "device": str(device),
        "task_key": task.task_key,
        "per_step": seed_summaries,
        "total_elapsed_s": time.time() - run_start,
        "peak_memory_bytes": max((entry["peak_memory_bytes"] or 0) for entry in seed_summaries) if track_mem else None,
        "peak_memory_mib": max((entry["peak_memory_mib"] or 0) for entry in seed_summaries) if track_mem else None,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
    }

    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(summary_path)

    (run_dir / "DONE").write_text("ok\n")
    print(f"[index={index0}] Wrote {summary_path} and DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())