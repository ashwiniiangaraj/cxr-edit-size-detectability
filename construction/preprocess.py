"""
PREPROCESSING
Every image of the master manifest becomes a *prepared* image:
grayscale -> 256x256 (only if not already 256) -> JPEG q95 round trip -> lossless PNG
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config
from utils.common import ensure_parent

_INTERP = {"area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4, "bicubic": cv2.INTER_CUBIC}
_BACKENDS = ("pil_lanczos", "cv2")


def prepare_one(src, dst: Path) -> bool:
    """
    src : path of the source image (any size / format)
    dst : path of the prepared PNG to write
    returns True if the image had to be RESIZED (False if it was already size x size)
    """
    size = config.PREP_IMAGE_SIZE
    backend = config.PREP_RESIZE_BACKEND
    if backend not in _BACKENDS:
        raise ValueError(f"PREP_RESIZE_BACKEND={backend!r} must be one of {_BACKENDS}")

    with Image.open(src) as im:
        gray = im.convert("L")
        resized = gray.size != (size, size)
        if resized:
            if backend == "pil_lanczos":
                gray = gray.resize((size, size), Image.Resampling.LANCZOS)
            else:
                small = cv2.resize(np.asarray(gray, dtype=np.uint8), (size, size),
                                   interpolation=_INTERP[config.PREP_RESIZE_INTERPOLATION])
                gray = Image.fromarray(small)
        arr = np.asarray(gray, dtype=np.uint8)

    if config.PREP_HARMONISE_JPEG:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=config.PREP_JPEG_QUALITY)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst, format="PNG")
    return resized


def prepare_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prepare every image; returns (and saves) the manifest with `prepared_path`."""
    if df is None:
        df = pd.read_csv(config.DET_MASTER_MANIFEST, dtype={"subject_id": str, "study_id": str})
    root = Path(config.DET_PREPARED_DIR)
    if config.PREP_RESIZE_BACKEND != "pil_lanczos":
        print("WARNING: PREP_RESIZE_BACKEND is not the frozen 'pil_lanczos' - "
              "the Real group will NOT share the generators' resize filter.")
    out_paths, n_new, errors = [], 0, []
    resized_per_group = Counter()
    for i, row in enumerate(df.itertuples(index=False), 1):
        dst = root / row.split / row.frequency_group / f"{row.image_id}.png"
        try:
            stale = dst.exists() and dst.stat().st_mtime < Path(row.path).stat().st_mtime
            if config.PREP_OVERWRITE or not dst.exists() or stale:
                if prepare_one(row.path, dst):
                    resized_per_group[row.frequency_group] += 1
                n_new += 1
        except Exception as exc:                                  # noqa: BLE001
            errors.append((row.image_id, repr(exc)))
        out_paths.append(str(dst))
        if i % 500 == 0:
            print(f"  prepared {i}/{len(df)}")
    if errors:
        raise RuntimeError(f"{len(errors)} images failed preprocessing, e.g. {errors[:3]}")
    # Freeze guard: every prepared file must be exactly size x size, 8-bit grayscale.
    size = config.PREP_IMAGE_SIZE
    bad = []
    for pth in out_paths:
        with Image.open(pth) as im:
            if im.size != (size, size) or im.mode != "L":
                bad.append((pth, im.size, im.mode))
    if bad:
        raise RuntimeError(f"{len(bad)} prepared files are not {size}x{size} mode L, e.g. {bad[:3]}")
    df = df.copy()
    df["prepared_path"] = out_paths
    df.to_csv(ensure_parent(config.DET_PREPARED_MANIFEST), index=False)
    print(f"Prepared {len(df)} images ({n_new} new) -> {root}")
    print(f"  size={config.PREP_IMAGE_SIZE} harmonise_jpeg={config.PREP_HARMONISE_JPEG} "
          f"q={config.PREP_JPEG_QUALITY} resize_backend={config.PREP_RESIZE_BACKEND}")
    print("  images RESIZED in this run, per group (expected: only Real):")
    for g, n in sorted(resized_per_group.items()):
        print(f"    {g:<24s} {n}")
    if not resized_per_group and n_new:
        print("    (none)")
    return df


def load_prepared_manifest() -> pd.DataFrame:
    return pd.read_csv(config.DET_PREPARED_MANIFEST, dtype={"subject_id": str, "study_id": str})