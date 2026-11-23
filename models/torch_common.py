"""
Shared PyTorch helpers for the CNN  and SupCon detectors.
Images are loaded from the prepared folder (256x256 grayscale PNG) as float32 in [0, 1], replicated to
3 channels and ImageNet-normalised. In the robustness stage, `degrade` is applied to the float array
before conversion to a tensor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from utils.common import load_gray_float01

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


def load_arrays(df: pd.DataFrame, degrade=None) -> list[np.ndarray]:
    """Load (and optionally degrade) every prepared image of `df`.

    Images are kept in memory as uint8 (4x smaller than float32) and converted to float32
    [0,1] on the fly by `to_float01`. The prepared images are 8-bit PNGs, so the 1/255 rounding
    loses nothing for clean images; degraded images are re-quantised to 8 bits."""
    out = []
    for p in df["prepared_path"]:
        img = load_gray_float01(p)
        if degrade is not None:
            img = degrade(img)
        out.append(np.ascontiguousarray(np.clip(np.rint(img * 255.0), 0, 255), dtype=np.uint8))
    return out


def to_float01(a: np.ndarray) -> torch.Tensor:
    """uint8 (H,W) array -> float32 (H,W) tensor in [0,1]."""
    return torch.from_numpy(a).float().div_(255.0)


def normalise(x1: torch.Tensor) -> torch.Tensor:
    """(1,H,W) in [0,1] -> (3,H,W) ImageNet-normalised."""
    return (x1.expand(3, -1, -1) - _MEAN) / _STD


class ArrayDataset(Dataset):
    """In-memory single-view dataset (optional horizontal flip)."""

    def __init__(self, arrays, labels=None, hflip=False):
        self.arrays, self.labels, self.hflip = arrays, labels, hflip

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        x = to_float01(self.arrays[i]).unsqueeze(0)
        if self.hflip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        y = -1.0 if self.labels is None else float(self.labels[i])
        return normalise(x), torch.tensor(y, dtype=torch.float32)


def make_densenet121(pretrained: bool):
    """torchvision DenseNet121, ImageNet weights if requested (needs internet once)."""
    from torchvision.models import DenseNet121_Weights, densenet121
    return densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
