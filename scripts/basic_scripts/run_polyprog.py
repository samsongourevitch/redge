from functools import partial
import csv
import time

import numpy as np
import torch
from tqdm import tqdm
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from local_paths import REPO_DIR
from samplers import SAMPLERS
from experiments.poly_prog import Polyprog
from experiments.utils import fix_seed

PROBLEM = "polynomial_prog"
BASE_CONFIG = [
    "device=cpu",
    "batch_size=256",
    "n_opt_steps=1000",
]

P_SWEEP = [1.5, 2.0, 3.0]

SAMPLERS_TO_RUN = [
    "redge",
    "reindge",
    "redge_cov",
    "gumbel",
    "st",
    "reinmax",
]

T1_SWEEP = [0.1]
T1_SWEEP_cov = [0.2]
T1_SWEEP_cutoff = [0.8]
TAU_SWEEP = [0.02]
SWEEP_OVERRIDES = {
    "redge": ("sampler.t_1", T1_SWEEP),
    "reindge": ("sampler.t_1", T1_SWEEP_cutoff),
    "redge_cov": ("sampler.t_1", T1_SWEEP_cov),
    "gumbel": ("sampler.tau", TAU_SWEEP),
}

INITIAL_SEED = 42
N_SEEDS = 1
SHOW_PROGRESS = True

LOG_ROOT = REPO_DIR / "outputs/poly_prog"


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _sanitize(value):
    return str(value).replace(".", "p").replace("=", "-")


def _sweep_values(sampler_name):
    if sampler_name in SWEEP_OVERRIDES:
        key, values = SWEEP_OVERRIDES[sampler_name]
        return [{key: value} for value in values]
    return [{}]


def _loss_summary(losses):
    losses = np.asarray(losses)
    window = min(100, len(losses))
    return float(losses[-1]), float(losses[-window:].mean())


def _stack_runs(runs):
    min_len = min(len(run) for run in runs)
    trimmed = [run[:min_len] for run in runs]
    return np.asarray(trimmed)


def _init_csv(csv_path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "sampler",
                "p",
                "sweep_param",
                "sweep_value",
                "seed",
                "final_loss",
                "mean_last_100",
                "losses_path",
            ]
        )


def _init_summary_csv(csv_path):
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sampler",
                "p",
                "sweep_param",
                "sweep_value",
                "n_seeds",
                "final_loss_mean",
                "final_loss_std",
                "mean_last_100_mean",
                "mean_last_100_std",
                "loss_mean_path",
                "loss_std_path",
            ]
        )


def _log_row(csv_path, row):
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _save_config(run_dir, tag, config):
    path = run_dir / f"config_{tag}.yaml"
    OmegaConf.save(config, path)


def run_single(config, sampler_name, seed, show_progress=True):
    torch.set_default_device(config.device)
    poly_prog_instance = Polyprog(**config.problem)
    optimizer = poly_prog_instance.get_optimizer()
    loss_fn = partial(poly_prog_instance.loss, **config.problem)

    losses = []
    pbar = tqdm(
        range(config.n_opt_steps),
        disable=not show_progress,
        leave=False,
        desc=f"{sampler_name} seed={seed}",
    )
    for _ in pbar:
        optimizer.zero_grad()
        logits = poly_prog_instance.get_logits()
        logits_expanded = logits.unsqueeze(0).expand(config.batch_size, *logits.shape)
        x = SAMPLERS[sampler_name](
            logits=logits_expanded,
            **config.sampler,
        )
        loss = loss_fn(x).mean()
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if show_progress:
            pbar.set_postfix({"loss": loss_val})

    final_loss, mean_last = _loss_summary(losses)
    return losses, final_loss, mean_last


def run_experiment():
    run_dir = LOG_ROOT / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "runs.csv"
    summary_path = run_dir / "summary.csv"
    _init_csv(csv_path)
    _init_summary_csv(summary_path)

    summary_cache = {}
    run_id = 0
    with initialize(config_path="../configs"):
        for p_value in P_SWEEP:
            for sampler_name in SAMPLERS_TO_RUN:
                for sweep in _sweep_values(sampler_name):
                    sweep_param, sweep_value = "", ""
                    if sweep:
                        sweep_param, sweep_value = next(iter(sweep.items()))
                    for seed_idx in range(N_SEEDS):
                        seed_value = INITIAL_SEED + seed_idx
                        overrides = [
                            f"sampler={sampler_name}",
                            f"problem={PROBLEM}",
                            *BASE_CONFIG,
                            f"problem.pow={p_value}",
                        ]
                        for key, value in sweep.items():
                            overrides.append(f"{key}={value}")

                        config = compose(config_name="config", overrides=overrides)
                        sweep_label = "default"
                        if sweep:
                            sweep_label = f"{sweep_param}={sweep_value}"
                        tag = (
                            f"{sampler_name}_p{_sanitize(p_value)}_"
                            f"{_sanitize(sweep_label)}_seed{seed_value}"
                        )
                        _save_config(run_dir, tag, config)
                        fix_seed(seed_value)
                        losses, final_loss, mean_last = run_single(
                            config,
                            sampler_name,
                            seed_value,
                            show_progress=SHOW_PROGRESS,
                        )
                        losses_path = run_dir / f"losses_{tag}.npy"
                        np.save(losses_path, np.asarray(losses))

                        _log_row(
                            csv_path,
                            [
                                run_id,
                                sampler_name,
                                p_value,
                                sweep_param,
                                sweep_value,
                                seed_value,
                                final_loss,
                                mean_last,
                                str(losses_path),
                            ],
                        )

                        key = (sampler_name, p_value, sweep_param, sweep_value)
                        summary_cache.setdefault(key, []).append(
                            {
                                "losses": losses,
                                "final_loss": final_loss,
                                "mean_last": mean_last,
                            }
                        )

                        print(
                            f"[{run_id}] {sampler_name} p={p_value} {sweep_label} "
                            f"seed={seed_value} loss={final_loss:.4f}"
                        )
                        run_id += 1

    for key, runs in summary_cache.items():
        sampler_name, p_value, sweep_param, sweep_value = key
        losses_arr = _stack_runs([run["losses"] for run in runs])
        loss_mean = losses_arr.mean(axis=0)
        loss_std = losses_arr.std(axis=0)
        loss_mean_path = run_dir / (
            f"loss_mean_{sampler_name}_p{_sanitize(p_value)}_"
            f"{_sanitize(sweep_param or 'default')}"
            f"{'' if sweep_value == '' else '-' + _sanitize(sweep_value)}.npy"
        )
        loss_std_path = run_dir / (
            f"loss_std_{sampler_name}_p{_sanitize(p_value)}_"
            f"{_sanitize(sweep_param or 'default')}"
            f"{'' if sweep_value == '' else '-' + _sanitize(sweep_value)}.npy"
        )
        np.save(loss_mean_path, loss_mean)
        np.save(loss_std_path, loss_std)

        final_losses = [run["final_loss"] for run in runs]
        mean_last_vals = [run["mean_last"] for run in runs]

        _log_row(
            summary_path,
            [
                sampler_name,
                p_value,
                sweep_param,
                sweep_value,
                len(runs),
                float(np.mean(final_losses)),
                float(np.std(final_losses)),
                float(np.mean(mean_last_vals)),
                float(np.std(mean_last_vals)),
                str(loss_mean_path),
                str(loss_std_path),
            ],
        )

    return run_dir


if __name__ == "__main__":
    run_experiment()
