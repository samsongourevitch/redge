import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Seq2SeqTransformerConfig:
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    max_position_embeddings: int
    pad_token_id: int
    eos_token_id: int
    dropout: float = 0.1

    @classmethod
    def from_base_config(cls, tokenizer, base_config, encoder_max_len: int, decoder_max_len: int):
        hidden = getattr(base_config, "n_embd", getattr(base_config, "hidden_size", 256))
        heads = getattr(base_config, "n_head", getattr(base_config, "num_attention_heads", 8))
        layers = getattr(base_config, "n_layer", getattr(base_config, "num_hidden_layers", 6))
        pos_budget = encoder_max_len + decoder_max_len + 2
        base_pos = getattr(base_config, "n_positions", getattr(base_config, "max_position_embeddings", 0))
        max_pos = max(pos_budget, base_pos)
        dropout = getattr(base_config, "attn_pdrop", getattr(base_config, "dropout", 0.1))
        return cls(
            vocab_size=tokenizer.vocab_size,
            hidden_size=hidden,
            num_heads=heads,
            num_layers=layers,
            max_position_embeddings=max_pos,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            dropout=dropout,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def save_pretrained(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = x if key_value is None else key_value
        bsz, tgt_len, _ = x.size()
        src_len = kv.size(1)

        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (bsz,1,1,src_len)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: Seq2SeqTransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ff = FeedForward(config.hidden_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        attn_output = self.self_attn(self.ln1(x), attn_mask=None, key_padding_mask=attention_mask)
        x = x + self.dropout(attn_output)
        ff_output = self.ff(self.ln2(x))
        x = x + self.dropout(ff_output)
        return x


class Seq2SeqTransformerModel(nn.Module):
    def __init__(self, config: Seq2SeqTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def _embed(self, input_ids: torch.Tensor):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        return self.dropout(self.token_embed(input_ids) + self.pos_embed(positions))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        **kwargs, 
    ):
        enc_mask = attention_mask.bool()
        dec_mask = decoder_attention_mask.bool()

        combined_ids = torch.cat([input_ids, decoder_input_ids], dim=1)
        combined_mask = torch.cat([enc_mask, dec_mask], dim=1)

        hidden = self._embed(combined_ids)
        for block in self.blocks:
            hidden = block(hidden, combined_mask)

        dec_hidden = hidden[:, input_ids.size(1) :, :]
        logits = self.lm_head(self.ln_final(dec_hidden))
        return SimpleNamespace(logits=logits)