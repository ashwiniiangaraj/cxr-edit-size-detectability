# ============================================================
# shared helpers for all generators
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- make "import config" work when a script is launched with %run ---
# %run executes the script with the notebook's working directory, so we
# add the project root (parent of generation/) to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402  (must come after the sys.path fix)


# ------------------------------------------------------------
# 1. FROZEN PROTOCOL SETTINGS
# ------------------------------------------------------------

# Name of the resize filter, read from config.py (default "lanczos").
RESIZE_FILTER_NAME = str(getattr(config, "GEN_RESIZE_FILTER", "lanczos")).lower()

_FILTERS = {
    "lanczos":  Image.Resampling.LANCZOS,
    "bicubic":  Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box":      Image.Resampling.BOX,      
    "nearest":  Image.Resampling.NEAREST,
}
if RESIZE_FILTER_NAME not in _FILTERS:
    raise ValueError(
        f"GEN_RESIZE_FILTER={RESIZE_FILTER_NAME!r} is not one of {sorted(_FILTERS)}"
    )
RESIZE_FILTER = _FILTERS[RESIZE_FILTER_NAME]

IMAGE_FORMAT = str(getattr(config, "GEN_IMAGE_FORMAT", "PNG")).upper()
if IMAGE_FORMAT != "PNG":
    raise ValueError(
        "Frozen protocol: generators must write lossless PNG "
        f"(got GEN_IMAGE_FORMAT={IMAGE_FORMAT!r})."
    )
IMAGE_EXT = ".png"


GENERATOR_VERSION = "stage1_fair_v2"

_STALE_EXTS = (".jpg", ".jpeg")


# ------------------------------------------------------------
# 2. LOADING  (one place, one filter)
# ------------------------------------------------------------

def load_gray_image(path: Path | str, size: int) -> Image.Image:
    """
    Open `path`, convert to 8-bit grayscale ("L") and resize to
    size x size with the frozen filter.

    path : file to read (the real MIMIC image, any size)
    size : target side length in pixels (config.IP_CONFIG['working_size'] = 256 for IP / CycleGAN)

    If the image already has the requested size Pillow performs no
    resampling, so an input that already has the target size is returned unchanged.
    """
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), RESIZE_FILTER)
    return image


def load_gray_array(path: Path | str, size: int) -> np.ndarray:
    """Same as load_gray_image but returns a float32 array of shape (size, size)."""
    return np.asarray(load_gray_image(path, size), dtype=np.float32)


# ------------------------------------------------------------
# 3. SAVING  (lossless PNG + read-back proof)
# ------------------------------------------------------------

def save_gray_png(array_uint8: np.ndarray, path: Path | str, verify: bool = True) -> None:
    """
    Save a 2-D uint8 array as 8-bit grayscale PNG.

    array_uint8 : image to write, dtype must be uint8, shape (H, W)
    path        : destination; must end in .png
    verify      : re-open the file and check every pixel is identical.
                  This is the proof that nothing lossy happened.
    """
    path = Path(path)
    if path.suffix.lower() != IMAGE_EXT:
        raise ValueError(f"Expected a .png path, got {path}")
    if array_uint8.dtype != np.uint8 or array_uint8.ndim != 2:
        raise ValueError(
            f"save_gray_png needs 2-D uint8, got {array_uint8.dtype} {array_uint8.shape}"
        )
    # No mode= argument: Pillow infers "L" from a 2-D uint8 array
    # (the mode= keyword is deprecated in recent Pillow).
    Image.fromarray(array_uint8).save(path, format="PNG")

    if verify:
        with Image.open(path) as check:
            back = np.asarray(check)
        if back.shape != array_uint8.shape or not np.array_equal(back, array_uint8):
            raise RuntimeError(f"PNG round trip is not lossless for {path}")


# ------------------------------------------------------------
# 4. STALE-FILE GUARD
# ------------------------------------------------------------

def ensure_clean_image_dir(directory: Path | str, label: str) -> None:
    """
    Stop if `directory` still contains JPEGs from an OLD run.

    Old q95 JPEGs mixed with new PNGs would silently put two
    different compression histories into one class.

    directory : folder that is about to receive new PNG images
    label     : human-readable name used in the error message
    """
    directory = Path(directory)
    if not directory.exists():
        return
    stale = [p for p in directory.iterdir() if p.suffix.lower() in _STALE_EXTS]
    if stale:
        raise RuntimeError(
            f"[{label}] {len(stale)} old JPEG file(s) found in {directory}\n"
            f"  e.g. {stale[0].name}\n"
            "Delete or move the old generation output (see the notebook "
            "'clean old Stage 1 output' cell) and run again."
        )


def assert_output_matches(directory: Path | str, written_paths, label: str) -> None:
    """
    Stop if `directory` holds PNG files that THIS run did not write.

    A re-run into a folder left over from different settings would otherwise leave old
    images next to new ones, and they would be picked up as part of the class.

    directory     : folder that received this run's images
    written_paths : paths (str / Path) of the images this run wrote
    label         : human-readable name used in the error message
    """
    directory = Path(directory)
    if not directory.exists():
        return
    written = {Path(p).resolve() for p in written_paths}
    extra = [p for p in directory.iterdir()
             if p.suffix.lower() == IMAGE_EXT and p.resolve() not in written]
    if extra:
        raise RuntimeError(
            f"[{label}] {len(extra)} PNG file(s) in {directory} were not written by this run\n"
            f"  e.g. {extra[0].name}\n"
            "They are left over from an earlier run with other settings. Delete the output "
            "folder (or use a new run folder) and generate again."
        )


# ------------------------------------------------------------
# 5. EDIT-STRENGTH DIAGNOSTICS
# ------------------------------------------------------------

def edit_strength(real: np.ndarray, synthetic: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    How much did the generator change the image?  Stored in metadata so
    that a detector's success can later be compared with edit strength.

    real      : source image, 2-D array (any numeric dtype)
    synthetic : generated image, same shape
    mask      : optional 2-D array, non-zero where the edit was made

    Returns
      mean_abs_diff_image    mean |synthetic - real| over all pixels (0-255 scale)
      mean_abs_diff_in_mask  same, over mask pixels only (NaN if no mask)
      edit_area_fraction     share of pixels inside the mask (NaN if no mask)
    """
    diff = np.abs(synthetic.astype(np.float32) - real.astype(np.float32))
    out = {"mean_abs_diff_image": float(diff.mean())}
    if mask is not None and np.any(mask > 0):
        inside = mask > 0
        out["mean_abs_diff_in_mask"] = float(diff[inside].mean())
        out["edit_area_fraction"] = float(inside.mean())
    else:
        out["mean_abs_diff_in_mask"] = float("nan")
        out["edit_area_fraction"] = float("nan")
    return out
