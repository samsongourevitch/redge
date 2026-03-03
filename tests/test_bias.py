import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from hydra.experimental import compose, initialize
import numpy as np

from samplers import SAMPLERS
from local_paths import REPO_DIR
from utils import fix_seed

K = 100
n_samples = 10000
n_seeds = 10
metric = "wasserstein"  # "wasserstein" or "kl"
# metric = "kl"
n_steps_sweep = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
t_1_sweep = [0.1, 0.3, 0.5, 0.7, 0.9]
fix_seed(42)
logits = torch.randn(K, requires_grad=True)
distr = Categorical(logits=logits)
gt_samples = distr.sample((n_samples,))
gt_probs = logits.softmax(-1).detach().cpu().numpy()
samplers = [
    "redge",
    "redge_cov",
]
baseline_sampler = "st"

def kl(p, q, eps=1e-10):
    p = p + eps
    q = q + eps
    return (p * np.log(p / q)).sum()

def eval_metric(sample_indices):
    if metric == "wasserstein":
        return wasserstein_distance(gt_samples.numpy(), sample_indices)
    if metric == "kl":
        counts = np.bincount(sample_indices, minlength=K).astype(np.float64)
        p_emp = counts / counts.sum()
        return kl(p_emp, gt_probs)
    raise ValueError(f"Unknown metric: {metric}")

def sample_with_config(sampler, n_steps=None, t_1=None):
    overrides = [f"sampler={sampler}"]
    if n_steps is not None:
        overrides.append(f"sampler.n_steps={n_steps}")
    if t_1 is not None:
        overrides.append(f"sampler.t_1={t_1}")
    config = compose(
        config_name="config",
        overrides=overrides,
    )
    logits_expanded = logits.unsqueeze(0).expand(n_samples, -1)
    return SAMPLERS[sampler](
        logits=logits_expanded,
        **config.sampler,
    )

results = {sampler: {t_1: {} for t_1 in t_1_sweep} for sampler in samplers}
baseline_vals = []

with initialize(config_path="../configs"):
    for seed in range(n_seeds):
        fix_seed(seed)
        baseline_probs = sample_with_config(baseline_sampler)
        baseline_vals.append(eval_metric(baseline_probs.argmax(-1).numpy()))

    for sampler in samplers:
        for t_1 in t_1_sweep:
            for n_steps in n_steps_sweep:
                vals = []
                for seed in range(n_seeds):
                    fix_seed(seed)
                    probs = sample_with_config(sampler, n_steps=n_steps, t_1=t_1)
                    vals.append(eval_metric(probs.argmax(-1).numpy()))
                results[sampler][t_1][n_steps] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                }

baseline_mean = float(np.mean(baseline_vals))
baseline_std = float(np.std(baseline_vals, ddof=0))

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(t_1_sweep),
    figsize=(4 * len(t_1_sweep), 4),
    sharey=True,
)
if len(t_1_sweep) == 1:
    axes = [axes]

for ax, t_1 in zip(axes, t_1_sweep):
    for sampler in samplers:
        means = [results[sampler][t_1][n]["mean"] for n in n_steps_sweep]
        stds = [results[sampler][t_1][n]["std"] for n in n_steps_sweep]
        ax.errorbar(
            n_steps_sweep,
            means,
            yerr=stds,
            capsize=3,
            marker="o",
            linewidth=1.5,
        )
    ax.axhline(
        baseline_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Empirical Law",
    )
    ax.fill_between(
        n_steps_sweep,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="red",
        alpha=0.1,
    )
    ax.set_title(rf"$t_1={t_1}$")
    ax.set_xlabel("$n$")
    ax.set_xticks(n_steps_sweep)

axes[0].set_ylabel("Wasserstein" if metric == "wasserstein" else metric)
axes[0].legend(frameon=False)
plt.tight_layout()
output_dir = REPO_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "bias_diff.png", dpi=300)
plt.show()
plt.close()
