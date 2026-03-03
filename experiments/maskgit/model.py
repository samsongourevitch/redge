import os
import torch
from experiments.maskgit.vq_model import VQ_MODELS
from experiments.maskgit.transformer import Transformer
from local_paths import MODELS_DIR
from torch import nn

# Borrowed from Halton-MaskGIT: https://github.com/valeoai/Halton-MaskGIT

class MaskGit(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.vit, self.ae = load_models(config)
        self.input_size = config.img_size // config.f_factor
        self.mask = config.mask_value
        self.codebook_size = config.codebook_size

    def forward(self, x, labels, cfg_weight=0.0, drop_label=False):
        drop = torch.ones_like(labels, dtype=torch.bool, device=x.device)
        assert (
            drop_label == False or cfg_weight == 0
        ), "MaskGit forward only supports cfg with drop=False"
        if cfg_weight != 0:
            logits = self.vit(
                torch.cat([x, x], dim=0),
                torch.cat([labels, labels], dim=0),
                torch.cat([~drop, drop], dim=0),
            )
            logits_c, logits_u = torch.chunk(logits, 2, dim=0)
            logits = (1 + cfg_weight) * logits_c - cfg_weight * logits_u
        else:
            if drop_label:
                logits = self.vit(x, labels, drop)
            else:
                logits = self.vit(x, labels, ~drop)

        return self._carry_over_unmasking(x, logits)

    def decode(self, code):
        return self.ae.decode_code(code)

    def _carry_over_unmasking(self, x, logits):
        x_flat = x.reshape(x.shape[0], -1)
        mask = x_flat == self.mask
        if (~mask).any():
            logits[~mask] = -1e9
            logits[~mask, x_flat[~mask]] = 0.0
        return logits


def transformer_size(size):
    """Returns the transformer configuration based on the specified size.
    :param:
        size -> str: Transformer size (tiny, small, base, large, xlarge).
    :returns
        (hidden_dim, depth, heads) -> tuple: Transformer settings.
    """

    if size == "tiny":
        hidden_dim, depth, heads = 384, 6, 6
    elif size == "small":
        hidden_dim, depth, heads = 512, 12, 6
    elif size == "base":
        hidden_dim, depth, heads = 768, 12, 12
    elif size == "large":
        hidden_dim, depth, heads = 1024, 24, 16
    elif size == "xlarge":
        hidden_dim, depth, heads = 1152, 28, 16
    else:
        hidden_dim, depth, heads = 768, 12, 12

    return hidden_dim, depth, heads


def load_models(config):
    """return the network, load checkpoint if config.resume == True
    :param
        arch -> str: vit|autoencoder|ema, the architecture to load
    :return
        model -> nn.Module: the network
    """
    input_size = config.img_size // config.f_factor

    config.vit_path = MODELS_DIR / "maskgit" / config.vit_path
    config.vqgan_path = MODELS_DIR / "maskgit" / config.vqgan_path
    if config.is_master:
        if not os.path.exists(config.vit_path):
            os.makedirs(config.vit_path)
            print(f"Folder created: {config.vit_path}")

    # Define transformer architecture parameters
    hidden_dim, depth, heads = transformer_size(config.vit_size)
    vit_model = Transformer(
        input_size=input_size,
        nclass=config.nb_class,
        c=hidden_dim,
        hidden_dim=hidden_dim,
        codebook_size=config.codebook_size,
        depth=depth,
        heads=heads,
        mlp_dim=hidden_dim * 4,
        dropout=config.dropout,
        register=config.register,
        proj=config.proj,
    )

    # Load model checkpoint
    if os.path.isfile(config.vit_path):
        ckpt = config.vit_path
    else:
        raise FileNotFoundError(f"Checkpoint not found: {config.vit_path}")

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    vit_model.load_state_dict(new_state_dict, strict=True)

    if config.is_master:
        print("Load ckpt from:", ckpt)

    # Initialize and load VQGAN model
    ae_model = VQ_MODELS[f"VQ-{config.f_factor}"](
        codebook_size=16384, codebook_embed_dim=8
    )
    checkpoint = torch.load(config.vqgan_path, map_location="cpu", weights_only=False)
    ae_model.load_state_dict(checkpoint["model"])
    vit_model.requires_grad_(False)
    ae_model.requires_grad_(False)
    vit_model.eval()
    ae_model.eval()

    # elif arch == "ema":
    #     # Load Exponential Moving Average (EMA) model
    #     model = self.get_ema(self.vit, device=config.device)
    #     if config.resume and os.path.isfile(config.vit_path + "ema.pth"):
    #         ckpt = config.vit_path + "ema.pth"
    #         checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    #         state_dict = checkpoint['model_state_dict']
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = "_orig_mod." + k if config.compile else k
    #             k = "module." + k if config.is_multi_gpus else k
    #             new_state_dict[k] = v
    #         model.module.load_state_dict(new_state_dict, strict=True)
    #     return model

    # model = model.to(config.device)

    return vit_model, ae_model
