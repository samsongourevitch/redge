import argparse
import copy
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import torch
import time
from torchvision.utils import make_grid

from local_paths import REPO_DIR
from experiments.vae.vae_data import get_vae_train_loader
from experiments.vae.vae_losses import vae_loss
from experiments.vae.vae_models import BernoulliVAESimple

from experiments.utils import fix_seed

from samplers import SAMPLERS


def train_vae(method, train_loader, cfg_sampler, cfg_problem, epochs, device, seed):
    fix_seed(seed)
    latent_dim = cfg_problem['length']
    cat_dim = cfg_problem['vocab_size']
    model = BernoulliVAESimple(input_dim=cfg_problem['input_dim'], cat_dim=cat_dim, latent_dim=latent_dim).to(device)
    lr = float(cfg_problem['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    true_losses = []
    times = []
    steps = 0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        true_loss = 0.0
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device).view(batch_x.size(0), -1)

            optimizer.zero_grad()

            logits = model.encode(batch_x).view(-1, latent_dim, cat_dim)

            p = logits.softmax(-1)

            z = SAMPLERS[method](
                logits=logits,
                **cfg_sampler,
            )

            recon_batch = model.decode(z.view(-1, latent_dim * cat_dim))
            
            loss = vae_loss(x_binary=batch_x, x_recon=recon_batch, p=p).mean()

            with torch.no_grad():
                z_true = torch.distributions.one_hot_categorical.OneHotCategorical(
                    probs=p
                ).sample()
                recon_true = model.decode(z_true.view(-1, latent_dim * cat_dim))
                true_loss_batch = vae_loss(x_binary=batch_x, x_recon=recon_true, p=p).mean()

            loss.backward()
            optimizer.step()
            steps += 1
            train_loss += loss.item()
            true_loss += true_loss_batch.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        times.append(epoch_time)
        avg_loss = train_loss / len(train_loader)
        losses.append(avg_loss)
        avg_true_loss = true_loss / len(train_loader)
        true_losses.append(avg_true_loss)
        print(
            f"[{method}] Epoch {epoch+1}/{epochs} - "
            f"Avg loss: {avg_loss:.4f} - True loss: {avg_true_loss:.4f} - Steps: {steps}"
        )

    return model, losses, true_losses, times


def _parse_sweep_list(raw_value):
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return [float(x.strip()) for x in raw_value.split(",") if x.strip()]

def _parse_int_sweep_list(raw_value):
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return [int(x.strip()) for x in raw_value.split(",") if x.strip()]

def _parse_sampler_list(raw_value):
    if raw_value == "all":
        return sorted(SAMPLERS.keys())
    return [x.strip() for x in raw_value.split(",") if x.strip()]


def _parse_seed_list(raw_value):
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return [int(x.strip()) for x in raw_value.split(",") if x.strip()]


def _load_best_params(summary_path, metric_name):
    best_by_sampler = {}
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sampler = row.get("sampler", "")
            if not sampler:
                continue
            metric_raw = row.get(metric_name, "")
            if metric_raw == "":
                continue
            try:
                metric_value = float(metric_raw)
            except ValueError:
                continue
            param_name = row.get("param_name", "")
            param_value_raw = row.get("param_value", "")
            if param_value_raw == "":
                param_value = None
            else:
                try:
                    param_value = float(param_value_raw)
                except ValueError:
                    param_value = param_value_raw

            current_best = best_by_sampler.get(sampler)
            if current_best is None or metric_value < current_best["metric_value"]:
                best_by_sampler[sampler] = {
                    "param_name": param_name,
                    "param_value": param_value,
                    "metric_value": metric_value,
                }
    return best_by_sampler


def _save_samples(model, cat_dim, latent_dim, n_samples, device, out_path):
    model.eval().to(device)
    z = torch.distributions.one_hot_categorical.OneHotCategorical(
        logits=torch.zeros(n_samples, latent_dim, cat_dim)
    ).sample().to(device).reshape(-1, latent_dim * cat_dim)

    with torch.no_grad():
        x_recon = torch.sigmoid(model.decode(z)).cpu()

    x_recon = x_recon.view(-1, 1, 28, 28)
    grid = make_grid(x_recon, nrow=4, padding=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/problem/bernoulli_vae.yaml')
    parser.add_argument('--samplers', type=str, default='all')
    parser.add_argument('--t1-sweep', type=str, default=None)
    parser.add_argument('--n-steps-sweep', type=str, default=None)
    parser.add_argument('--tau-sweep', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed-start', type=int, default=None)
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--best-from', type=str, default=None)
    parser.add_argument('--best-metric', type=str, default='final_loss')
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--log-dir', type=str, default=str(REPO_DIR / "outputs/bernoulli_vae/sweep"))
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_problem = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    seed = args.seed

    cfg_problem['seed'] = seed

    batch_size = cfg_problem['batch_size']
    train_loader = get_vae_train_loader(batch_size)

    best_params = None
    if args.best_from:
        best_params = _load_best_params(args.best_from, args.best_metric)
        if args.samplers == "all":
            samplers = sorted(best_params.keys())
        else:
            samplers = _parse_sampler_list(args.samplers)
    else:
        samplers = _parse_sampler_list(args.samplers)
    t1_sweep = _parse_sweep_list(args.t1_sweep)
    tau_sweep = _parse_sweep_list(args.tau_sweep)
    n_steps_sweep = _parse_int_sweep_list(args.n_steps_sweep)
    seed_list = _parse_seed_list(args.seeds)
    if seed_list is None:
        base_seed = seed if args.seed_start is None else args.seed_start
        seed_list = [base_seed + i for i in range(args.n_seeds)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "problem": cfg_problem,
        "samplers": samplers,
        "t1_sweep": t1_sweep,
        "tau_sweep": tau_sweep,
        "n_steps_sweep": n_steps_sweep,
        "epochs": epochs,
        "seed_list": seed_list,
        "device": device,
        "best_from": args.best_from,
        "best_metric": args.best_metric,
        "n_samples": args.n_samples,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_payload, f)

    losses_path = run_dir / "losses.csv"
    summary_path = run_dir / "summary.csv"

    with open(losses_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "sampler",
                "param_name",
                "param_value",
                "seed",
                "epoch",
                "loss",
                "true_loss",
                "epoch_time",
            ],
        )
        writer.writeheader()

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "sampler",
                "param_name",
                "param_value",
                "seed",
                "epochs",
                "final_loss",
                "final_true_loss",
                "mean_epoch_time",
            ],
        )
        writer.writeheader()

    run_configs = []
    run_idx = 0

    for sampler in samplers:
        print(f"Starting training for method: {sampler}")
        sampler_path = Path("configs/sampler") / f"{sampler}.yaml"
        if not sampler_path.exists():
            print(f"Sampler config not found for {sampler}, skipping.")
            continue
        with open(sampler_path, 'r') as f:
            cfg_sampler_base = yaml.safe_load(f)

        if best_params is not None:
            best_entry = best_params.get(sampler)
            if best_entry is None:
                print(f"No best parameters found for {sampler}, skipping.")
                continue
            param_name = best_entry["param_name"]
            sweep_values = [best_entry["param_value"]]
        else:
            if sampler in {"redge", "reindge", "redge_cov"}:
                t1_values = t1_sweep if t1_sweep is not None else [cfg_sampler_base["t_1"]]
                n_steps_values = n_steps_sweep if n_steps_sweep is not None else [cfg_sampler_base["n_steps"]]
                sweep_values = [(t1, n_steps) for t1 in t1_values for n_steps in n_steps_values]
                param_name = "t_1,n_steps"
            elif sampler == "gumbel":
                sweep_values = tau_sweep if tau_sweep is not None else [cfg_sampler_base["tau"]]
                param_name = "tau"
            else:
                sweep_values = [None]
                param_name = ""

        for value in sweep_values:
            for run_seed in seed_list:
                run_idx += 1
                run_id = f"run_{run_idx:03d}"
                cfg_sampler = copy.deepcopy(cfg_sampler_base)
                if param_name == "t_1,n_steps":
                    t1_value, n_steps_value = value
                    cfg_sampler["t_1"] = t1_value
                    cfg_sampler["n_steps"] = n_steps_value
                elif param_name == "t_1":
                    cfg_sampler["t_1"] = value
                elif param_name == "tau":
                    cfg_sampler["tau"] = value

                run_configs.append(
                    {
                        "run_id": run_id,
                        "sampler": sampler,
                        "param_name": param_name,
                        "param_value": value,
                        "seed": run_seed,
                        "sampler_config": cfg_sampler,
                    }
                )

                try:
                    model, losses, true_losses, times = train_vae(
                        method=sampler,
                        train_loader=train_loader,
                        cfg_sampler=cfg_sampler,
                        cfg_problem=cfg_problem,
                        epochs=epochs,
                            device=device,
                            seed=run_seed,
                        )
                except Exception as e:
                    print(f"An error occurred while training with method {sampler}: {e}")
                    continue

                sample_name = f"{run_id}_{sampler}_seed{run_seed}"
                if param_name:
                    sample_name += f"_{param_name}{value}"
                sample_path = samples_dir / f"{sample_name}.png"
                _save_samples(
                    model=model,
                    cat_dim=cfg_problem['vocab_size'],
                    latent_dim=cfg_problem['length'],
                    n_samples=args.n_samples,
                    device=device,
                    out_path=sample_path,
                )

                with open(losses_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "run_id",
                            "sampler",
                            "param_name",
                            "param_value",
                            "seed",
                            "epoch",
                            "loss",
                            "true_loss",
                            "epoch_time",
                        ],
                    )
                    for epoch_idx, (loss, true_loss, epoch_time) in enumerate(
                        zip(losses, true_losses, times), start=1
                    ):
                        writer.writerow(
                            {
                                "run_id": run_id,
                                "sampler": sampler,
                                "param_name": param_name,
                                "param_value": value if value is not None else "",
                                "seed": run_seed,
                                "epoch": epoch_idx,
                                "loss": loss,
                                "true_loss": true_loss,
                                "epoch_time": epoch_time,
                            }
                        )

                with open(summary_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "run_id",
                            "sampler",
                            "param_name",
                            "param_value",
                            "seed",
                            "epochs",
                            "final_loss",
                            "final_true_loss",
                            "mean_epoch_time",
                        ],
                    )
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "sampler": sampler,
                            "param_name": param_name,
                            "param_value": value if value is not None else "",
                            "seed": run_seed,
                            "epochs": epochs,
                            "final_loss": losses[-1] if losses else "",
                            "final_true_loss": true_losses[-1] if true_losses else "",
                            "mean_epoch_time": sum(times) / len(times) if times else "",
                        }
                    )

    with open(run_dir / "runs.yaml", "w") as f:
        yaml.safe_dump(run_configs, f)

if __name__ == '__main__':
    main()
