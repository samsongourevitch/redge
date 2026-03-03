import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.sudoku.guidance import guided_sampler
from experiments.sudoku.model import load_model
from experiments.sudoku.sudoku_utils import (
    SudokuDataset,
    build_unit_indices,
    collate,
    count_violations_batch,
)
from experiments.utils import fix_seed
from local_paths import DATA_DIR, MODELS_DIR, REPO_DIR
from samplers import SAMPLERS

REQUIRED_TRANSFORMERS_VERSION = "4.37.2"


def ensure_transformers_version() -> None:
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Please install transformers==4.37.2"
        ) from exc

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise SystemExit(
            "This demo requires transformers==4.37.2, "
            f"but found transformers=={transformers.__version__}. "
            "Please run: pip install transformers==4.37.2"
        )


def build_soft_penalty(
    lookup: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor
):
    lookup_mask = (lookup >= 1.0).float()

    def soft_penalty(samples: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = samples.shape[0], samples.shape[-1]
        sudoku_grid = samples[~src_mask.bool()].view(batch_size, -1, vocab_size)
        sudoku_grid = sudoku_grid[:, :-1]  # remove EOS token
        sudoku_grid = sudoku_grid * lookup_mask
        sudoku_grid = sudoku_grid.view(batch_size, 9, 9, vocab_size)
        units_sum = sudoku_grid[:, row_idx, col_idx, :].sum(dim=-2)
        return (units_sum - lookup_mask).square().sum(dim=(1, 2))

    return soft_penalty


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def count_exact_matches(
    samples: torch.Tensor, batch: dict[str, torch.Tensor], tokenizer
) -> int:
    correct = 0
    for idx in range(samples.shape[0]):
        predicted = tokenizer.decode(samples[idx].tolist())
        target = tokenizer.decode(batch["input_ids"][idx].tolist())
        correct += int(predicted == target)
    return correct


def run(cfg: DictConfig) -> None:
    ensure_transformers_version()
    torch.set_default_device(cfg.device)
    fix_seed(cfg.seed)

    config_dir = REPO_DIR / cfg.demo.config_dir
    mdm_config = REPO_DIR / cfg.demo.mdm_config
    checkpoint = MODELS_DIR / cfg.demo.checkpoint
    eval_csv = DATA_DIR / cfg.demo.eval_csv

    model, tokenizer = load_model(
        config_path=config_dir,
        checkpoint_path=checkpoint,
        mdm_config_path=mdm_config,
        device=str(cfg.device),
    )

    eval_ds = SudokuDataset(
        str(eval_csv),
        tokenizer,
        cutoff_len=cfg.demo.cutoff_len,
        max_samples=cfg.demo.max_samples,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.demo.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    lookup_cpu = eval_ds.build_digit_lookup()
    row_idx_cpu, col_idx_cpu = build_unit_indices()

    lookup_device = lookup_cpu
    row_idx = row_idx_cpu
    col_idx = col_idx_cpu
    penalty_fn = build_soft_penalty(lookup_device, row_idx, col_idx)

    sampler_name = cfg.sampler.sampler_name
    sampler_fn = SAMPLERS[sampler_name]
    runtime_cfg = OmegaConf.create(
        {
            "hard": cfg.demo.hard,
            "sampler": OmegaConf.to_container(cfg.sampler, resolve=True),
            "guidance": OmegaConf.to_container(cfg.demo.guidance, resolve=True),
        }
    )

    total_violations = 0.0
    total_correct = 0
    total_samples = 0

    for batch in eval_loader:
        samples = guided_sampler(
            model=model,
            sampler=sampler_fn,
            batch=batch,
            tokenizer=tokenizer,
            config=runtime_cfg,
            penalty_fn=penalty_fn,
        )

        samples_cpu = samples.detach().cpu()
        total_correct += count_exact_matches(samples_cpu, batch, tokenizer)
        grid = eval_ds.ids_to_grid(samples_cpu, lookup_cpu, src_mask=batch["src_mask"])
        n_violations, _ = count_violations_batch(grid, row_idx_cpu, col_idx_cpu)

        total_violations += n_violations.item()
        total_samples += samples_cpu.shape[0]

    if total_samples == 0:
        print("No samples were evaluated.")
        return

    violation_rate = total_violations / total_samples
    correct_rate = total_correct / total_samples
    print(f"sampler={sampler_name} | violations/sample: {violation_rate:.3f}")
    print(f"sampler={sampler_name} | exact-match: {correct_rate:.3f}")


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REPO_DIR / "configs")):
        cfg = compose(
            config_name="config",
            overrides=["demo=sudoku_mdm", *sys.argv[1:]],
        )
    run(cfg)


if __name__ == "__main__":
    main()
