# %%
import sys
import time
import os
import json
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import torch
from PIL import Image
from hydra import compose, initialize_config_dir

from imscore.preference.model import CLIPScore
from datetime import datetime

from local_paths import REPO_DIR
from experiments.utils import fix_seed, save_im, display, plot_callbacks
from experiments.maskgit.model import MaskGit
from experiments.maskgit.guided_sampler import guided_remdm_sampler, build_callbacks
from samplers import SAMPLERS
from local_paths import MODELS_DIR, REPO_DIR


def run(cfg: DictConfig):

    fix_seed(seed=cfg.demo.seed)
    device = torch.device(cfg.device)
    torch.set_default_device(device)

    # load model
    model = MaskGit(config=cfg.demo)

    reward_model = CLIPScore.from_pretrained(
        "RE-N-Y/clipscore-vit-large-patch14", cache_dir=MODELS_DIR / "reward_models"
    )

    reward_model = reward_model.to(device)
    reward_model.eval()
    reward_model.requires_grad_(False)
    prompt = cfg.demo.prompt

    def reward_fn(x):
        scores = reward_model.score((x + 1) / 2, [prompt] * x.size(0))
        return cfg.demo.reward_scaling * scores.sum()

    callbacks = build_callbacks()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"reconstructions/{ts}_{cfg.sampler.sampler_name}_{cfg.demo.prompt}"
    save_folder = REPO_DIR / save_folder
    save_folder.mkdir(parents=True, exist_ok=True)

    def save_reconstruction(sample, t, opt_step, **kwargs):
        if opt_step == cfg.demo.guidance.n_opt_steps - 1:
            save_path = Path(save_folder) / f"im_t={t}"
            save_im(sample, save_path=save_path)

    save_reconstruction.log = None
    save_reconstruction.descr = None
    callbacks.append(save_reconstruction)

    start_time = time.perf_counter()

    print(f"guidance cfg: {cfg.demo.guidance}")
    print(f"sampler confing: {cfg.demo.mdm_sampler}")
    labels = torch.tensor([cfg.demo.label] * cfg.demo.n_samples, dtype=torch.long)

    samples = guided_remdm_sampler(
        config=cfg,
        model=model,
        cond=labels,
        n_steps=cfg.demo.mdm_sampler.step,
        n_samples=cfg.demo.n_samples,
        reward_fn=reward_fn,
        sampler=SAMPLERS[cfg.sampler.sampler_name],
        hard=cfg.demo.hard,
        cfg_weight=cfg.demo.mdm_sampler.cfg_w,
        sigma=cfg.demo.mdm_sampler.sigma,
        schedule_type="linear",
        callbacks=None,
        drop_label=cfg.demo.mdm_sampler.drop_label,
    )

    plot_callbacks(callbacks, save_path=save_folder / "callbacks")
    end_time = time.perf_counter()

    samples = samples.clamp(-1.0, 1.0)
    peak_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024**3)
    print("===================")
    print(
        f"{cfg.sampler.sampler_name} reward: {reward_fn(samples).sum().item() / cfg.demo.reward_scaling}"
    )
    print(f"{'runtime':10}: {end_time - start_time}")
    if peak_gb is not None:
        print(f"{'gpu_peak_gb':10}: {peak_gb:.2f}")
    print("===================")

    cfg = OmegaConf.to_container(cfg, resolve=True)
    (save_folder / "cfg.json").write_text(json.dumps(cfg, indent=2, default=str))
    save_im(samples, save_path=save_folder / "final_im")


def main():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_DIR / "configs")):
        cfg = compose(
            config_name="config",
            overrides=["demo=maskgit", *sys.argv[1:]],
        )
    run(cfg)


if __name__ == "__main__":
    main()
