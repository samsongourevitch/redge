#!/usr/bin/env python3
"""Run guided MDM Sudoku generation for one sampler."""

from __future__ import annotations

import json
import time
from datetime import datetime
from functools import partial

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from local_paths import DATA_DIR, MODELS_DIR, REPO_DIR
from samplers import SAMPLERS
from experiments.sudoku.guidance import guided_sampler
from experiments.sudoku.model import load_model
from experiments.sudoku.sudoku_utils import SudokuDataset, build_unit_indices, collate, count_violations_batch
from experiments.utils import fix_seed


def resolve_device(device_value: str | int) -> torch.device:
    requested = torch.device(str(device_value))
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device '{requested}' but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return requested


def soft_penalty(samples, src_mask, lookup, row_idx, col_idx):
    batch_size, n_tokens = samples.shape[0], samples.shape[-1]
    sudoku_grid = samples[~src_mask.bool()].view(batch_size, -1, n_tokens)
    sudoku_grid = sudoku_grid[:, :-1]
    sudoku_grid = sudoku_grid * (lookup >= 1.0)
    sudoku_grid = sudoku_grid.view(batch_size, 9, 9, n_tokens)
    units_sum = sudoku_grid[:, row_idx, col_idx, :].sum(dim=-2)
    digit_mask = (lookup >= 1.0).float().view(1, 1, -1)
    return (units_sum - digit_mask).square().sum(dim=(1, 2))


@hydra.main(config_path="../configs", config_name="guided_mdm", version_base=None)
def main(config: DictConfig) -> None:
    fix_seed(int(config.seed))
    device = resolve_device(config.device)
    torch.set_default_device(device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_rel = OmegaConf.select(config, "model.checkpoint_rel")
    checkpoint_path = MODELS_DIR / checkpoint_rel if checkpoint_rel else None
    model, tokenizer = load_model(
        config_path=REPO_DIR / config.model.tokenizer_dir,
        checkpoint_path=checkpoint_path,
        mdm_config_path=MODELS_DIR / config.model.mdm_config_rel,
        device=str(device),
    )

    eval_ds = SudokuDataset(
        DATA_DIR / config.data.eval_csv,
        tokenizer=tokenizer,
        cutoff_len=int(config.data.cutoff_len),
        max_samples=int(config.max_samples),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        collate_fn=collate,
    )

    lookup = eval_ds.build_digit_lookup().to(device)
    row_idxs, col_idxs = build_unit_indices()
    row_idxs = row_idxs.to(device)
    col_idxs = col_idxs.to(device)
    penalty_fn = partial(soft_penalty, lookup=lookup, row_idx=row_idxs, col_idx=col_idxs)

    sampler_name = config.sampler.sampler_name
    sampler_fn = SAMPLERS[sampler_name]

    total_violations = 0.0
    total_solved = 0
    total_exact = 0
    total_samples = 0
    start_time = time.perf_counter()

    pbar = tqdm(eval_loader, desc="eval", total=len(eval_loader))
    for batch in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        samples = guided_sampler(
            model=model,
            sampler=sampler_fn,
            batch=batch,
            tokenizer=tokenizer,
            config=config,
            penalty_fn=penalty_fn,
            callbacks=None,
        )
        grid = eval_ds.ids_to_grid(samples, lookup, src_mask=batch["src_mask"])
        violations, solved = count_violations_batch(grid, row_idxs, col_idxs)
        total_violations += float(violations.item())
        total_solved += int(solved)
        total_exact += int((samples == batch["input_ids"]).all(dim=1).sum().item())
        total_samples += int(samples.shape[0])
        current_mean_violations = total_violations / max(total_samples, 1)
        current_solved_rate = total_solved / max(total_samples, 1)
        pbar.set_postfix(
            avg_violations=f"{current_mean_violations:.4f}",
            solve_rate=f"{current_solved_rate:.4f}",
        )

    elapsed_s = time.perf_counter() - start_time
    mean_violations = total_violations / max(total_samples, 1)
    solved_rate = total_solved / max(total_samples, 1)
    exact_match_rate = total_exact / max(total_samples, 1)

    peak_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        REPO_DIR
        / "outputs/guided_mdm/examples"
        / f"{ts}_{sampler_name}_steps{config.guidance.n_opt_steps}_lr{config.guidance.lr}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "sampler": sampler_name,
        "n_samples": total_samples,
        "mean_violations": mean_violations,
        "solved_rate": solved_rate,
        "exact_match_rate": exact_match_rate,
        "runtime_s": elapsed_s,
        "gpu_peak_gb": peak_gb,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)
    )

    print("===================")
    print(f"sampler: {sampler_name}")
    print(f"n_samples: {total_samples}")
    print(f"mean_violations: {mean_violations:.6f}")
    print(f"solved_rate: {solved_rate:.6f}")
    print(f"exact_match_rate: {exact_match_rate:.6f}")
    print(f"runtime_s: {elapsed_s:.2f}")
    if peak_gb is not None:
        print(f"gpu_peak_gb: {peak_gb:.2f}")
    print(f"saved_to: {run_dir}")
    print("===================")


if __name__ == "__main__":
    main()
