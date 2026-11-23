"""
Small helpers shared by every stage: seeding, device selection, image
loading, DICOM-id parsing and figure saving.  No stage-specific logic here.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

import config

# MIMIC-CXR DICOM ids look like  ea8a6c49-71ce56ef-c091d6ee-16f084b7-3788e7dd.
# Every family keeps the source DICOM id in the file name, so this regex links
# a synthetic image back to its real source whatever the family naming:
#   IP        p13282748_s58553239_<dicom>.jpg
#   CycleGAN  normal_to_pathology_001_p..._s..._<dicom>.jpg
#   Diffusion <dicom>.png
DICOM_RE = re.compile(r"([0-9a-f]{8}(?:-[0-9a-f]{8}){4})", re.IGNORECASE)
SUBJECT_RE = re.compile(r"p(\d+)_s(\d+)_")


def parse_dicom_id(name: str) -> str | None:
    m = DICOM_RE.search(name)
    return m.group(1).lower() if m else None


def parse_subject_study(name: str) -> tuple[str | None, str | None]:
    m = SUBJECT_RE.search(name)
    return (m.group(1), m.group(2)) if m else (None, None)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Resolve config.DET_DEVICE ('auto' | 'cuda' | 'cpu') to a torch.device."""
    import torch
    want = config.DET_DEVICE
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{config.DET_GPU_INDEX}")
    if want == "cuda":
        raise RuntimeError("config.DET_DEVICE='cuda' but CUDA is unavailable.")
    return torch.device("cpu")


def load_gray_uint8(path) -> np.ndarray:
    """Load any image as a single-channel uint8 array (H, W)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def to_float01(arr_uint8: np.ndarray) -> np.ndarray:
    return arr_uint8.astype(np.float32) / 255.0


def load_gray_float01(path) -> np.ndarray:
    return to_float01(load_gray_uint8(path))


def float01_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def save_figure(fig, name: str, subdir: str = "") -> Path | None:
    """Save a matplotlib figure under config.DET_FIGURE_DIR (if enabled)."""
    if not config.FIG_SAVE:
        return None
    out_dir = Path(config.DET_FIGURE_DIR) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{config.FIG_FORMAT}"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    return path


def ensure_parent(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
