import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import PIL
import math


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def visualize_samples(model, cat_dim=10, latent_dim=20, n_samples=8, device="cpu"):
    model.eval().to(device)
    # Sample from prior
    z = (
        torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=torch.zeros(n_samples, latent_dim, cat_dim)
        )
        .sample()
        .to(device)
        .reshape(-1, latent_dim * cat_dim)
    )

    # Decode to get images
    with torch.no_grad():
        x_recon = torch.sigmoid(model.decode(z)).cpu()

    # Plot images in a grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(x_recon[i].view(28, 28), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_top_bottom_samples(
    model, test_loader, cat_dim=2, latent_dim=200, n_samples=8, device="cpu"
):
    model.eval().to(device)
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(device)
    top, bottom = imgs[:, : 14 * 28], imgs[:, 14 * 28 :]

    with torch.no_grad():
        q_logits = model.encode(top)

    z = (
        torch.distributions.one_hot_categorical.OneHotCategorical(logits=q_logits)
        .sample()
        .to(device)
        .reshape(-1, latent_dim * cat_dim)
    )

    with torch.no_grad():
        bottom_recon = torch.sigmoid(model.decode(z)).cpu()

    n_samples = min(n_samples, imgs.size(0))
    plt.figure(figsize=(8, 4 * n_samples))

    for i in range(n_samples):
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.imshow(imgs[i].view(28, 28).cpu(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        recon_full = torch.cat([top[i], bottom_recon[i]], dim=0)
        recon_full = recon_full.view(28, 28)

        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.imshow(recon_full.cpu(), cmap="gray")
        plt.title("Reconstructed Bottom")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_im(x, save_path=None):
    sample = x.squeeze(0).float().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.squeeze()
    sample = sample.cpu().numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    fig.savefig(save_path)
    plt.close()


def display(x, save_path=None, title=None):
    sample = x.squeeze(0).float().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.squeeze()
    sample = sample.cpu().numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def plot_and_save(all_losses, out_dir, fname_prefix="losses"):
    os.makedirs(out_dir, exist_ok=True)
    # Plot each method
    plt.figure()
    for method, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Avg loss (per sample)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(out_dir, f"{fname_prefix}.png")
    plt.savefig(out_path)
    plt.close()


def show_images_grid(batch, nrow=4, padding=2):
    """
    Displays a batch of images concatenated into a single grid using PyTorch's make_grid.

    Args:
        batch (torch.Tensor): Batch of images, shape (B, C, H, W), with values in range [-1, 1].
        nrow (int): Number of images in each row of the grid.
        padding (int): Padding between images in the grid.
    """
    # Unnormalize the tensor from [-1, 1] to [0, 1]
    batch = (batch + 1) / 2.0

    # Clamp to ensure all values are in [0, 1]
    batch = batch.clamp(0, 1)

    # Create the grid
    grid = make_grid(batch, nrow=nrow, padding=padding)

    # Move the grid to CPU and convert to numpy
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # Display the grid
    plt.figure(figsize=(nrow * 2, (len(batch) // nrow + 1) * 2))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def plot_callbacks(callbacks, title="Callback logs", save_path=None):
    """Plot callback logs in a grid of 3 columns for readability."""
    n = len(callbacks)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, cb in enumerate(callbacks):
        if cb.log is not None:
            axes[idx].plot(cb.log)
            axes[idx].set_title(getattr(cb, "descr", f"Callback {idx}"))
            axes[idx].grid(True)

    # hide any empty subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)
