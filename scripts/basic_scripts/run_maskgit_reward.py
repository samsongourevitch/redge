# %%
import time
import os
import json
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from imscore.preference.model import CLIPScore
from datetime import datetime

from local_paths import REPO_DIR
from experiments.utils import fix_seed, save_im, plot_callbacks
from experiments.maskgit.model import MaskGit
from experiments.maskgit.guided_sampler import guided_remdm_sampler, build_callbacks
from samplers import SAMPLERS
from local_paths import MODELS_DIR

# os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
# os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


torch.cuda.empty_cache()


@hydra.main(config_path="../configs", config_name="config_maskgit")
def run_sampler(config: DictConfig):

    fix_seed(seed=config.seed)
    device = torch.device(
        f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
    )
    torch.set_default_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    # load model
    model = MaskGit(config=config)

    reward_model = CLIPScore.from_pretrained(
        "RE-N-Y/clipscore-vit-large-patch14", cache_dir=MODELS_DIR / "reward_models"
    )

    reward_model = reward_model.to(device)
    reward_model.eval()
    reward_model.requires_grad_(False)
    prompt = config.prompt

    def reward_fn(x):
        scores = reward_model.score((x + 1) / 2, [prompt] * x.size(0))
        return config.reward_scaling * scores.sum()

    callbacks = build_callbacks()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"reconstructions/{ts}_{config.sampler.sampler_name}_{config.prompt}"
    save_folder = REPO_DIR / save_folder
    save_folder.mkdir(parents=True, exist_ok=True)

    def save_reconstruction(sample, t, opt_step, **kwargs):
        if opt_step == config.guidance.n_opt_steps - 1:
            save_path = Path(save_folder) / f"im_t={t}"
            save_im(sample, save_path=save_path)

    save_reconstruction.log = None
    save_reconstruction.descr = None
    callbacks.append(save_reconstruction)

    start_time = time.perf_counter()

    print(f"guidance config: {config.guidance}")
    print(f"sampler confing: {config.mdm_sampler}")
    labels = torch.tensor([config.label] * config.n_samples, dtype=torch.long)

    samples = guided_remdm_sampler(
        config=config,
        model=model,
        cond=labels,
        n_steps=config.mdm_sampler.step,
        n_samples=config.n_samples,
        reward_fn=reward_fn,
        sampler=SAMPLERS[config.sampler.sampler_name],
        hard=config.hard,
        cfg_weight=config.mdm_sampler.cfg_w,
        sigma=config.mdm_sampler.sigma,
        schedule_type="linear",
        callbacks=None,
        drop_label=config.mdm_sampler.drop_label,
    )

    plot_callbacks(callbacks, save_path=save_folder / "callbacks")
    end_time = time.perf_counter()

    samples = samples.clamp(-1.0, 1.0)
    peak_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024**3)
    cfg = OmegaConf.to_container(config, resolve=True)
    (save_folder / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    save_im(samples, save_path=save_folder / "final_im")

    print("===================")
    print(
        f"{config.sampler.sampler_name} reward: {reward_fn(samples).sum().item() / config.reward_scaling}"
    )
    print(f"{'runtime':10}: {end_time - start_time}")
    if peak_gb is not None:
        print(f"{'gpu_peak_gb':10}: {peak_gb:.2f}")
    print("===================")


if __name__ == "__main__":
    run_sampler()
