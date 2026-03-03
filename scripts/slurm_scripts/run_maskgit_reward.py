#!/usr/bin/env python3
"""Run MaskGIT reward guidance experiments from a pending manifest."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl
import hashlib
import hydra
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from imscore.preference.model import CLIPScore
from local_paths import REPO_DIR, MODELS_DIR
from experiments.maskgit.guided_sampler import guided_remdm_sampler
from experiments.maskgit.model import MaskGit
from samplers import SAMPLERS
from experiments.utils import fix_seed, save_im

CLIPSCORE_MODEL_ID = "RE-N-Y/clipscore-vit-large-patch14"
CLIPSCORE_ENV_VAR = "CLIPSCORE_MODEL_PATH"


def jsonl_dump(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_task_key(row: dict) -> str:
    payload = jsonl_dump({k: v for k, v in row.items() if k != "task_key"})
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def clipscore_candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get(CLIPSCORE_ENV_VAR)
    if env_path:
        paths.append(Path(env_path).expanduser())

    base = MODELS_DIR / "reward_models"
    hf_root = base / f"models--{CLIPSCORE_MODEL_ID.replace('/', '--')}"
    paths.append(hf_root)
    snapshots = hf_root / "snapshots"
    if snapshots.exists():
        for child in sorted(snapshots.iterdir()):
            if child.is_dir():
                paths.append(child)
    return paths


def load_clipscore_model() -> CLIPScore:
    for path in clipscore_candidate_paths():
        if (path / "config.json").exists():
            return CLIPScore.from_pretrained(str(path), local_files_only=True)
    raise FileNotFoundError(
        "Local CLIPScore checkpoint not found. "
        f"Set {CLIPSCORE_ENV_VAR} or place files under {MODELS_DIR / 'reward_models'}"
    )


@dataclass
class RunSpec:
    name: str
    rel_dir: Path
    abs_dir: Path
    prompt_entries: list[dict]
    prompt_indices: list[int]
    index: int


@dataclass
class Task:
    sampler: str
    sampler_overrides: dict[str, Any]
    lr: float
    grad_steps: list[int]
    seed: int
    task_dir_abs: Path
    manifest_row: dict
    task_key: str
    n_samples_per_job: int
    total_runs: int
    run_manifest_abs: Path
    run: RunSpec


def read_jsonl_line(path: Path, index0: int) -> dict:
    with path.open() as src:
        for idx, line in enumerate(src):
            if idx == index0:
                return json.loads(line)
    raise IndexError(f"Index {index0} out of range for {path}")


def load_prompt_file(path: Path) -> list[dict]:
    with path.open() as handle:
        data = json.load(handle)
    entries = []
    for idx, item in enumerate(data):
        entry = dict(item)
        entry["prompt_index"] = idx
        entries.append(entry)
    return entries


def select_prompts(prompt_data: list[dict], indices: list[int]) -> list[dict]:
    return [prompt_data[idx] for idx in indices]


def parse_grad_steps(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def build_task(row: dict, root: Path) -> Task:
    overrides = {k: row[k] for k in row if k.startswith("sampler.")}
    task_rel = Path(row["task_dir"])
    task_dir = root / task_rel
    key = row.get("task_key")
    if not key:
        key = stable_task_key(row)
    manifest_path = root / row["run_manifest"]
    grad_steps_value = row.get("grad_steps") or row.get("grad_steps_list")
    grad_steps = parse_grad_steps(grad_steps_value)
    if not grad_steps:
        raise ValueError("grad_steps must contain at least one value.")
    run_name = row.get("run_name") or row.get("run_dir")
    pending_runs = row.get("pending_runs")
    if not run_name and pending_runs:
        if len(pending_runs) != 1:
            raise ValueError("pending_runs must contain exactly one run when no run_name is provided.")
        run_name = pending_runs[0]
    if not run_name:
        raise ValueError("Pending manifest row must include run_name/run_dir.")
    run_rel = Path(run_name)
    run_dir = task_dir / run_rel

    prompt_entries = row.get("prompts")
    prompt_indices = row.get("prompt_indices")
    run_index = row.get("run_index")

    manifest_runs = read_run_manifest(manifest_path)
    selected_entry = None
    for entry in manifest_runs:
        entry_name = entry.get("run_dir") or entry.get("run_name")
        if entry_name == run_name:
            selected_entry = entry
            if run_index is None:
                run_index = entry.get("run_index")
            if prompt_entries is None:
                prompt_entries = entry.get("prompts")
            if prompt_indices is None:
                prompt_indices = entry.get("prompt_indices")
            break

    if run_index is None:
        raise ValueError(f"Run index not found for {run_name}")
    if prompt_entries is None:
        if prompt_indices is None:
            raise ValueError(f"Prompt indices missing for run {run_name}")
        prompt_data = load_prompt_file(Path(row["prompt_file"]))
        prompt_entries = select_prompts(prompt_data, prompt_indices)
    if prompt_indices is None:
        prompt_indices = [int(entry["prompt_index"]) for entry in prompt_entries]

    manifest_row = {k: v for k, v in row.items() if k not in {"run_name", "run_dir", "run_index", "prompt_indices", "prompts"}}
    manifest_row["task_key"] = key
    manifest_row["grad_steps"] = list(grad_steps)

    samples_per_job = int(row.get("n_samples_per_job", len(prompt_entries)))
    total_runs = int(row.get("n_runs") or len(manifest_runs) or 1)
    run_spec = RunSpec(
        name=run_name,
        rel_dir=task_rel / run_rel,
        abs_dir=run_dir,
        prompt_entries=prompt_entries,
        prompt_indices=[int(idx) for idx in prompt_indices],
        index=int(run_index),
    )
    return Task(
        sampler=row["sampler"],
        sampler_overrides=overrides,
        lr=float(row["lr"]),
        grad_steps=grad_steps,
        seed=int(row["seed"]),
        task_dir_abs=task_dir,
        manifest_row=manifest_row,
        task_key=key,
        n_samples_per_job=samples_per_job,
        total_runs=total_runs,
        run_manifest_abs=manifest_path,
        run=run_spec,
    )


def read_run_manifest(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry.setdefault("run_index", idx)
            rows.append(entry)
    return rows


def write_json_once(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, default=str) + "\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)


def append_jsonl_locked(path: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for entry in entries:
            handle.write(jsonl_dump(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def base_summary(task: Task, metrics_file: Path, pending_manifest: Path, manifest_index: int) -> dict:
    summary = dict(task.manifest_row)
    summary.update(
        {
            "task_key": task.task_key,
            "task_dir": str(task.task_dir_abs),
            "sampler": task.sampler,
            "sampler_overrides": task.sampler_overrides,
            "lr": task.lr,
            "grad_steps": task.grad_steps,
            "seed": task.seed,
            "n_samples_per_job": task.n_samples_per_job,
            "n_runs": task.total_runs,
            "metrics_file": str(metrics_file),
            "pending_manifest": str(pending_manifest),
            "manifest_index": manifest_index,
        }
    )
    return summary


def update_task_summary(
    task: Task,
    metrics_file: Path,
    run: RunSpec,
    run_seed: int,
    pending_manifest: Path,
    manifest_index: int,
) -> None:
    summary_path = task.task_dir_abs / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        if raw:
            summary = json.loads(raw)
        else:
            summary = base_summary(task, metrics_file, pending_manifest, manifest_index)
        runs = summary.get("runs", [])
        runs = [entry for entry in runs if entry.get("run_dir") != str(run.rel_dir)]
        runs.append(
            {
                "run_dir": str(run.rel_dir),
                "run_name": run.name,
                "run_index": run.index,
                "prompt_indices": run.prompt_indices,
                "n_prompts": len(run.prompt_entries),
                "seed": run_seed,
                "grad_steps": list(task.grad_steps),
                "n_grad_steps": len(task.grad_steps),
            }
        )
        runs.sort(key=lambda entry: entry.get("run_index", 0))
        summary["runs"] = runs
        summary["n_runs_completed"] = len(runs)
        summary["total_prompts"] = sum(entry["n_prompts"] for entry in runs)
        summary["metrics_file"] = str(metrics_file)
        summary["completed"] = summary["n_runs_completed"] >= task.total_runs
        summary["last_update_s"] = time.time()
        handle.seek(0)
        json.dump(summary, handle, indent=2)
        handle.write("\n")
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def mark_task_done_if_complete(task: Task) -> bool:
    manifest_runs = read_run_manifest(task.run_manifest_abs)
    for entry in manifest_runs:
        run_name = entry.get("run_dir") or entry.get("run_name")
        if not (task.task_dir_abs / run_name / "DONE").exists():
            return False
    done_path = task.task_dir_abs / "DONE"
    done_path.write_text("ok\n")
    return True


def apply_overrides(base_cfg: DictConfig, task: Task) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    sampler_cfg_path = REPO_DIR / "configs" / "sampler" / f"{task.sampler}.yaml"
    if not sampler_cfg_path.exists():
        raise FileNotFoundError(f"Sampler config not found: {sampler_cfg_path}")
    sampler_cfg = OmegaConf.load(sampler_cfg_path)
    OmegaConf.update(cfg, "sampler", sampler_cfg, merge=False)
    for key, value in task.sampler_overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.update(cfg, "guidance.lr", task.lr, merge=False)
    first_step = task.grad_steps[0] if task.grad_steps else 1
    OmegaConf.update(cfg, "guidance.n_opt_steps", int(first_step), merge=False)
    OmegaConf.update(cfg, "seed", task.seed, merge=False)
    OmegaConf.update(cfg, "n_samples", task.n_samples_per_job, merge=False)
    return cfg


def resolve_device(device_value: Any) -> torch.device:
    backend_available = torch.cuda.is_available()
    if isinstance(device_value, str):
        if device_value.startswith(("cuda", "cpu", "hip")):
            device = torch.device(device_value)
        else:
            try:
                idx = int(device_value)
                device = torch.device(f"cuda:{idx}" if backend_available else "cpu")
            except ValueError:
                device = torch.device("cuda" if backend_available else "cpu")
    elif isinstance(device_value, (int, float)):
        idx = int(device_value)
        device = torch.device(f"cuda:{idx}" if backend_available else "cpu")
    else:
        device = torch.device("cuda" if backend_available else "cpu")
    if device.type == "cuda" and not backend_available:
        device = torch.device("cpu")
    return device


def prompt_output_dir(parent_dir: Path, prompt_index: int) -> Path:
    out = parent_dir / f"prompt_{prompt_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_prompt_batch(
    cfg: DictConfig,
    model: MaskGit,
    reward_model: CLIPScore,
    sampler_fn,
    device: torch.device,
    prompt_entries: list[dict],
    run_dir: Path,
    run_seed: int,
    grad_steps: int,
) -> list[dict]:
    prompt_indices = [int(entry["prompt_index"]) for entry in prompt_entries]
    prompts = [entry["prompt"] for entry in prompt_entries]
    class_ids = [int(entry["class_id"]) for entry in prompt_entries]

    fix_seed(run_seed)

    cond = torch.tensor(class_ids, dtype=torch.long, device=device)
    step_dir = run_dir / f"grad_{int(grad_steps):03d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    def reward_fn(x):
        texts = prompts[: x.size(0)]
        scores = reward_model.score((x + 1) / 2, texts)
        return cfg.reward_scaling * scores.sum()

    start = time.perf_counter()
    samples = guided_remdm_sampler(
        config=cfg,
        model=model,
        cond=cond,
        n_steps=cfg.mdm_sampler.step,
        n_samples=len(prompt_entries),
        reward_fn=reward_fn,
        sampler=sampler_fn,
        hard=cfg.hard,
        cfg_weight=cfg.mdm_sampler.cfg_w,
        sigma=cfg.mdm_sampler.sigma,
        schedule_type="linear",
        callbacks=None,
        drop_label=cfg.mdm_sampler.drop_label,
    )
    elapsed = time.perf_counter() - start
    samples = samples.clamp(-1.0, 1.0)

    with torch.no_grad():
        clip_scores = reward_model.score((samples + 1) / 2, prompts)

    metrics = []
    for idx, prompt_idx in enumerate(prompt_indices):
        prompt_dir = prompt_output_dir(step_dir, prompt_idx)
        sample = samples[idx : idx + 1].detach().cpu()
        save_im(sample, save_path=prompt_dir / "final.png")
        base_score = float(clip_scores[idx].item())
        entry = {
            "prompt_index": prompt_idx,
            "class_id": class_ids[idx],
            "prompt": prompts[idx],
            "clip_score": base_score,
            "scaled_reward": base_score * float(cfg.reward_scaling),
            "elapsed_s": elapsed,
            "seed": run_seed,
            "grad_steps": int(grad_steps),
        }
        (prompt_dir / "result.json").write_text(json.dumps(entry, indent=2))
        metrics.append(entry)
    return metrics


def resolve_index(cfg: DictConfig) -> int:
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_id is not None:
        return int(env_id)
    idx = cfg.get("index")
    if idx is None:
        raise ValueError("Set index or provide SLURM_ARRAY_TASK_ID.")
    idx = int(idx)
    if idx < 0:
        raise ValueError("index must be >= 0 when no SLURM array id is provided.")
    return idx


@hydra.main(config_path="../configs", config_name="clip_sweep", version_base=None)
def main(cfg: DictConfig):
    manifest_value = cfg.get("pending_manifest")
    if not manifest_value:
        raise SystemExit("Set cfg.pending_manifest to the pending_manifest.jsonl you want to consume.")
    manifest_path = Path(manifest_value)
    if not manifest_path.exists():
        raise SystemExit(f"Pending manifest not found: {manifest_path}")

    index0 = resolve_index(cfg)
    row = read_jsonl_line(manifest_path, index0)
    root = manifest_path.parent
    try:
        task = build_task(row, root)
    except ValueError as exc:
        print(f"[index={index0}] {exc}")
        return 0

    task.task_dir_abs.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(REPO_DIR / "configs" / "config_maskgit.yaml")
    hcfg = apply_overrides(base_cfg, task)

    task_manifest_path = task.task_dir_abs / "task_manifest.json"
    write_json_once(task_manifest_path, task.manifest_row)

    device = resolve_device(hcfg.device)
    torch.set_default_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    fix_seed(hcfg.seed)

    model = MaskGit(config=hcfg)
    reward_model = load_clipscore_model()
    reward_model = reward_model.to(device)
    reward_model.eval()
    reward_model.requires_grad_(False)
    sampler_fn = SAMPLERS[hcfg.sampler.sampler_name]

    config_path = task.task_dir_abs / "config.json"
    write_json_once(config_path, OmegaConf.to_container(hcfg, resolve=True))

    metrics_path = task.task_dir_abs / "prompt_metrics.jsonl"
    run = task.run
    run.abs_dir.mkdir(parents=True, exist_ok=True)
    done_file = run.abs_dir / "DONE"
    if done_file.exists():
        print(f"[index={index0}] Run {run.name} already completed, skipping.")
        return 0

    run_seed = task.seed + run.index
    appended_any = False
    for grad_step in task.grad_steps:
        step_cfg = OmegaConf.create(OmegaConf.to_container(hcfg, resolve=True))
        OmegaConf.update(step_cfg, "n_samples", len(run.prompt_entries), merge=False)
        OmegaConf.update(step_cfg, "seed", run_seed, merge=False)
        OmegaConf.update(step_cfg, "guidance.n_opt_steps", int(grad_step), merge=False)
        batch_metrics = run_prompt_batch(
            step_cfg,
            model,
            reward_model,
            sampler_fn,
            device,
            run.prompt_entries,
            run.abs_dir,
            run_seed,
            int(grad_step),
        )
        step_dir = run.abs_dir / f"grad_{int(grad_step):03d}"
        batch_path = step_dir / "prompt_metrics.jsonl"
        with batch_path.open("w") as dst:
            for entry in batch_metrics:
                dst.write(jsonl_dump(entry) + "\n")

        append_jsonl_locked(metrics_path, batch_metrics)
        appended_any = True

    if appended_any:
        update_task_summary(task, metrics_path, run, run_seed, manifest_path, index0)
    done_file.write_text("ok\n")
    mark_task_done_if_complete(task)

    print(f"[index={index0}] Completed run {run.name} -> {run.abs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())