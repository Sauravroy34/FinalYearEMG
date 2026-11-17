"""Sequence dataset: each item = `seq_len` consecutive windows + MFI of the last one."""
import numpy as np
import torch
from torch.utils.data import Dataset


class FatigueSequenceDataset(Dataset):
    def __init__(self, subjects: list[dict], seq_len: int, augment: bool = False):
        self.subjects = subjects
        self.seq_len = seq_len
        self.augment = augment
        self.index = []  # (subject_idx, last_window_idx)
        for si, s in enumerate(subjects):
            for w in range(seq_len - 1, len(s["windows"])):
                self.index.append((si, w))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, w = self.index[i]
        s = self.subjects[si]
        x = s["windows"][w - self.seq_len + 1: w + 1].copy()   # (S, C, L)
        if self.augment:
            # mild, label-preserving augmentations (strong amplitude scaling would
            # destroy the amplitude-increase fatigue cue, so keep it small)
            x *= np.float32(np.random.uniform(0.9, 1.1))
            x += np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
        y = s["labels"][w]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), si, w
