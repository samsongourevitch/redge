import torch
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: sequence of dicts with keys 'puzzle' and 'solution' strings of length 81.
        puzzle uses '0' for empty.
        We will encode digits '1'..'9' -> 0..8. Empties: fill with 0 (any value), but mask marks them.
        """
        self.entries = data_list

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        d = self.entries[idx]
        p = d['puzzle'].strip()
        s = d['solution'].strip()
        assert len(p) == 81 and len(s) == 81

        sol = torch.tensor([int(c) for c in s], dtype=torch.long).view(9,9)  # 0..8
        puzzle_vals = []
        mask_vals = []
        for c in p:
            if c == '0':
                puzzle_vals.append(0)   # filler
                mask_vals.append(0.0)
            else:
                puzzle_vals.append(int(c))
                mask_vals.append(1.0)
        puzzle = torch.tensor(puzzle_vals, dtype=torch.long).view(9,9)
        mask = torch.tensor(mask_vals, dtype=torch.float32).view(9,9)
        return {'x0': sol, 'puzzle': puzzle, 'mask': mask}
