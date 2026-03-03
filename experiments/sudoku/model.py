# %%
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import hydra
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM
from experiments.utils import fix_seed
import yaml


class CustomTokenizer:
    def __init__(self, vocab: List[str], model_max_length: int):
        self.vocab = vocab
        self.model_max_length = model_max_length
        self.pad_token = "[PAD]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        self._vocab_str_to_int = {
            self.pad_token: 0,
            self.sep_token: 1,
            self.mask_token: 2,
            self.eos_token: 3,
            self.unk_token: 4,
            **{ch: i + 5 for i, ch in enumerate(vocab)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def pad_token_id(self) -> int:
        return self._vocab_str_to_int[self.pad_token]

    @property
    def sep_token_id(self) -> int:
        return self._vocab_str_to_int[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self._vocab_str_to_int[self.mask_token]

    @property
    def eos_token_id(self) -> int:
        return self._vocab_str_to_int[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self._vocab_str_to_int[self.unk_token]

    def encode(self, text: str) -> List[int]:
        splits = text.split(" ")
        tokens: List[str] = []
        for split in splits:
            if split != "":
                if split in self._vocab_str_to_int:
                    tokens.extend([split, " "])
                else:
                    tokens.extend(list(split) + [" "])
        if tokens:
            tokens = tokens[:-1]
        return [self._vocab_str_to_int.get(tok, self.unk_token_id) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        return "".join([self._vocab_int_to_str.get(i, self.unk_token) for i in ids])

    @classmethod
    def from_pretrained(cls, config_dir: str) -> "CustomTokenizer":
        cfg = json.load(open(Path(config_dir) / "tokenizer_config.json"))
        return cls(cfg["vocab"], cfg["model_max_length"])


class DiffusionModel(nn.Module):
    def __init__(self, base_model, config, diffusion_steps: int, checkpoint_path=None):
        super().__init__()
        self.model = base_model
        self.config = self.model.config

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.model.device)
            cleaned = {
                k.replace("model.", "", 1): v
                for k, v in state.items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(cleaned)
        self.vocab_size = config.vocab_size
        self.embed_tokens = self.model.transformer.wte
        self.denoise_model = self.model.transformer
        for block in self.model.transformer.h:
            block.attn.bias.fill_(True)  # remove causal mask
        self.lm_head = self.model.lm_head
        self.diffusion_steps = diffusion_steps

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x_embed = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.denoise_model(
            inputs_embeds=x_embed, attention_mask=attention_mask, return_dict=False
        )[0]
        logits = self.lm_head(x)
        return logits


def load_model(config_path, checkpoint_path, mdm_config_path, device="cuda:0"):
    tokenizer = CustomTokenizer.from_pretrained(config_path)
    config = AutoConfig.from_pretrained(config_path)
    config = AutoConfig.from_pretrained(config_path)
    base_model = AutoModelForCausalLM.from_config(config)
    diffusion_steps = OmegaConf.load(mdm_config_path).diffusion_steps
    model = DiffusionModel(
        base_model,
        config,
        diffusion_steps=diffusion_steps,
        checkpoint_path=checkpoint_path,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


def sampler_step(x_0, x_t, t, mask, src_mask):
    unmasked = x_t != mask
    x_0[unmasked] = x_t[unmasked]  # keep unmasked x_t

    if t > 0:
        unmasking_prob = 1 / (t + 1)
        to_mask = (
            torch.rand_like(x_0.float()) > unmasking_prob
        ) & ~src_mask  # to be masked
        x_0.masked_fill_((to_mask & ~unmasked).bool(), mask)
        x_t = x_0
    else:
        x_t = x_0

    return x_t


def mdm_sampler(model, batch, tokenizer):
    x = batch["input_ids"]
    src_mask = batch["src_mask"]
    attention_mask = batch["attention_mask"]
    x_t = x.masked_fill(~src_mask.bool(), tokenizer.mask_token_id)
    for t in range(model.diffusion_steps - 1, -1, -1):
        logits = model(input_ids=x_t, attention_mask=attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs[..., tokenizer.vocab_size :] = -1e3  # set pads to 0 just in case
        _, x_0 = log_probs.max(-1)
        x_t = sampler_step(
            x_0=x_0, x_t=x_t, t=t, mask=tokenizer.mask_token_id, src_mask=src_mask
        )

    return x_t


# def generate(model, batch, tokenizer, diffusion_steps):
#     model.eval()
#     x = batch["input_ids"].to(device)
#     src_mask = batch["src_mask"].bool().to(device)
#     maskable_mask = ~src_mask
#     attention_mask = torch.ones_like(x)

#     xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
#     for t in range(diffusion_steps - 1, -1, -1):
#         t_tensor = torch.full((x.size(0),), t, device=device)
#         logits = model(xt, t_tensor, attention_mask=attention_mask)
#         # logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
#         scores = torch.log_softmax(logits, dim=-1)
#         scores[:, :, tokenizer.vocab_size :] = -1000
#         _, x0 = scores.max(-1)
#         x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
#         if t > 0:
#             unmask_prob = 1 / (t + 1)
#             mask_to_x0 = (torch.rand_like(xt, dtype=torch.float) < unmask_prob) & maskable_mask
#             xt[mask_to_x0] = x0[mask_to_x0]
#             maskable_mask.masked_fill_(mask_to_x0, False)
#         else:
#             xt = x0
#     return xt
