import torch 
from torch.nn.functional import one_hot 
from typing import List 
from torch.utils.data import Dataset
import json 
from pathlib import Path
import pandas as pd

def collate(batch):
    return {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in batch[0]}

# ---------------- Tokenizer ----------------
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


# ---------------- Dataset ----------------
class SudokuDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: CustomTokenizer, cutoff_len: int = 164, max_samples: int = None):
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        if max_samples:
            df = df.head(max_samples)
        self.samples = []
        for _, row in df.iterrows():
            prompt = str(row["quizzes"])
            tgt = str(row["solutions"])
            self.samples.append(self._encode_example(prompt, tgt, cutoff_len))

    @staticmethod
    def _pad(seq, pad_id, length):
        return seq[:length] + [pad_id] * max(0, length - len(seq))

    def _encode_example(self, prompt: str, tgt: str, cutoff_len: int, **kwargs):
        src_ids = self.tokenizer.encode(prompt)
        tgt_ids = self.tokenizer.encode(tgt)
        assert len(src_ids) == len(tgt_ids), "uneven length"
        tgt_ids = tgt_ids[: cutoff_len - 2]
        src_ids = src_ids[-(cutoff_len - 2 - len(tgt_ids)) :]
        input_ids = src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        src_mask = [1] * (len(src_ids) + 1) + [0] * len(tgt_ids) + [0]

        input_ids = self._pad(input_ids, self.tokenizer.pad_token_id, cutoff_len)
        attention_mask = self._pad(attention_mask, 1, cutoff_len)
        src_mask = self._pad(src_mask, 0, cutoff_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "src_mask": src_mask,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    
    def build_digit_lookup(self): 
        lookup = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float)
        for tok, idx in self.tokenizer._vocab_str_to_int.items(): 
            if tok.isdigit(): 
                lookup[idx] = int(tok)
        return lookup
            

    def ids_to_grid(self, ids, lookup, src_mask): 
        #ids: [B, len_grid]
        sudoku_grid = lookup[ids][~src_mask.bool()]
        sudoku_grid = sudoku_grid.reshape(ids.shape[0], -1)[..., :-1] # [B, 81]
        #if non digit token present, add duplicate so it counts as violations
        min_digit = sudoku_grid.min(dim=-1)[0].reshape(-1, 1)
        sudoku_grid = sudoku_grid.clamp(min=min_digit) 
        to_onehot = one_hot(sudoku_grid.long(), num_classes=10) # [B, 81, 10]
        return to_onehot[..., 1:].reshape(-1, 9, 9, 9) 

class SimpleSudokuDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        p = torch.tensor(list(map(int, d["puzzle"].strip())), dtype=torch.long).view(9, 9)
        s = torch.tensor(list(map(int, d["solution"].strip())), dtype=torch.long).view(9, 9)
        return {"prompt": p, "tgt": s}


def build_unit_indices():
    """
    row_idxs, col_idx: [27, 9]
    """
    rows = []
    cols = []
    nonets = []
    for i in range(9):
        rows.append([(i, j) for j in range(9)])
        cols.append([(j, i) for j in range(9)])
    for i in [0, 3, 6]:
        for k in [0, 3, 6]:
            nonets.append([(i + j // 3, k + j % 3) for j in range(9)])
    units = rows + cols + nonets
    row_idx = torch.tensor([[rc[0] for rc in unit] for unit in units], dtype=torch.long)
    col_idx = torch.tensor([[rc[1] for rc in unit] for unit in units], dtype=torch.long)
    return row_idx, col_idx

def count_violations_batch(samples, row_idx, col_idx):
    # samples: [B,81,9]
    samples = samples.view(samples.shape[0], 9, 9, 9)
    units_sum = samples[:, row_idx, col_idx, :].sum(2)   # [B,27,9,9]
    units_nviolations = (units_sum - 1).clamp(min=0) 
    total_violations = units_nviolations.sum(dim=(1, 2))
    total_correct = total_violations[total_violations == 0].numel()
    return total_violations.sum(), total_correct

def clamp_clues(logits, clue_vals):
    # logits: [B,L,V]; clue_vals: [B, L]
    logits.masked_fill_((clue_vals > 0.).unsqueeze(-1), -1e3)
    b, l = clue_vals.nonzero(as_tuple=True)
    v = clue_vals[b, l]
    logits[b, l, v-1] = 0  #-1 to shift digits into [0:8]
