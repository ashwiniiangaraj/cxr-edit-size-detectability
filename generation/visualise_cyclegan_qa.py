"""
CycleGAN visualisation: Real source, generated image, difference heatmap,
and strongest-change overlay.

"""

from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METADATA_PATH = Path(config.CYCLEGAN_GENERATION_METADATA)

N_PER_DIRECTION = getattr(config, "CYCLEGAN_VISUAL_QA_N_PER_DIRECTION", 5)
PERCENTILE = getattr(config, "CYCLEGAN_VISUAL_QA_PERCENTILE", 90)
RANDOM_SEED = getattr(config, "CYCLEGAN_VISUAL_QA_RANDOM_SEED", config.RANDOM_SEED)
SPLIT_FILTER = getattr(config, "CYCLEGAN_VISUAL_QA_SPLIT", "test")

IMAGE_SIZE = getattr(config, "CYCLEGAN_IMAGE_SIZE", 256)


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def load_gray(path):
    """Load an image as grayscale uint8."""
    return np.array(Image.open(path).convert("L"))


def resize_to_target(image, size=IMAGE_SIZE):
    """
    Resize a uint8 grayscale array to size x size with Lanczos
    (the frozen project filter). No-op if already the right size.
    """
    if image.shape != (size, size):
        image = np.asarray(
            Image.fromarray(image).resize(
                (size, size),
                Image.Resampling.LANCZOS,
            )
        )
    return image


# ---------------------------------------------------------------------
# Difference-map logic copied from the HTML workflow
# ---------------------------------------------------------------------

def create_difference_map(source, generated):
    """
    Compute absolute normalized pixel difference.

    This follows the HTML notebook:
        difference = abs(generated - source)
        difference = difference / 255.0
    """
    source = source.astype(np.float32)
    generated = generated.astype(np.float32)

    difference = np.abs(generated - source)
    difference = difference / 255.0

    return difference


def create_change_mask(difference, percentile=PERCENTILE):
    """
    Create the strongest-change mask using the HTML notebook logic.

    The threshold is the requested percentile of the difference map,
    followed by 3x3 morphological opening and closing.
    """
    threshold = np.percentile(difference, percentile)

    mask = (difference >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask, threshold


def create_overlay(source, change_mask):
    """
    Create the red change overlay used for visual inspection.

    The mask is a difference-derived visualisation, NOT a ground-truth
    lesion mask and NOT an attention/causal mask.
    """
    overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)

    overlay[change_mask > 0] = [255, 0, 0]

    return overlay


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------

def show_difference(
    source_path,
    generated_path,
    percentile=PERCENTILE,
    title_prefix="",
):
    """
    Display one 1x4 visualisation:

        1. Real/source
        2. CycleGAN generated
        3. Absolute difference heatmap
        4. Strongest-change overlay
    """
    source = resize_to_target(load_gray(source_path))
    generated = resize_to_target(load_gray(generated_path))

    difference = create_difference_map(source, generated)

    change_mask, threshold = create_change_mask(
        difference,
        percentile=percentile,
    )

    overlay = create_overlay(source, change_mask)

    mean_difference = float(difference.mean())
    changed_fraction = float((change_mask > 0).mean())

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(source, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Real / Source")
    axes[0].axis("off")

    axes[1].imshow(generated, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("CycleGAN Generated")
    axes[1].axis("off")

    axes[2].imshow(difference, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Strongest Changes")
    axes[3].axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)

    plt.tight_layout()
    plt.show()

    print(f"Mean absolute difference: {mean_difference:.6f}")
    print(f"{percentile}th percentile threshold: {threshold:.6f}")
    print(f"Highlighted area: {changed_fraction:.4%}")


# ---------------------------------------------------------------------
# Metadata / example selection
# ---------------------------------------------------------------------

def _generated_path_from_row(row):
    """
    Support both metadata naming conventions:
        - synthetic_path
        - generated_path
    """
    if "synthetic_path" in row and pd.notna(row["synthetic_path"]):
        return Path(row["synthetic_path"])

    if "generated_path" in row and pd.notna(row["generated_path"]):
        return Path(row["generated_path"])

    raise KeyError(
        "Metadata must contain either 'synthetic_path' or 'generated_path'."
    )


def select_examples(
    metadata,
    direction,
    n=N_PER_DIRECTION,
    split_filter=SPLIT_FILTER,
    seed=RANDOM_SEED,
):
    """Select reproducible examples for one CycleGAN direction."""
    df = metadata.copy()

    if "direction" not in df.columns:
        raise KeyError("Metadata must contain a 'direction' column.")

    df = df[df["direction"] == direction].copy()

    if split_filter is not None:
        if "source_split" not in df.columns:
            raise KeyError(
                "Metadata must contain 'source_split' when "
                "CYCLEGAN_VISUAL_QA_SPLIT is not None."
            )

        df = df[df["source_split"] == split_filter].copy()

    if len(df) == 0:
        raise ValueError(
            f"No examples found for direction={direction!r}, "
            f"split={split_filter!r}."
        )

    n = min(n, len(df))

    return df.sample(
        n=n,
        random_state=seed,
    ).reset_index(drop=True)


def show_direction(
    metadata,
    direction,
    n=N_PER_DIRECTION,
    percentile=PERCENTILE,
    split_filter=SPLIT_FILTER,
    seed=RANDOM_SEED,
):
    """Visualise selected examples for one translation direction."""
    examples = select_examples(
        metadata=metadata,
        direction=direction,
        n=n,
        split_filter=split_filter,
        seed=seed,
    )

    for i, (_, row) in enumerate(examples.iterrows(), start=1):
        source_path = Path(row["source_path"])
        generated_path = _generated_path_from_row(row)

        source_id = row.get("source_dicom_id", source_path.stem)
        target_domain = row.get("target_domain", "")

        title = (
            f"{direction} | Example {i} | "
            f"Source: {source_id} | Target: {target_domain}"
        )

        show_difference(
            source_path=source_path,
            generated_path=generated_path,
            percentile=percentile,
            title_prefix=title,
        )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def visualise_all(
    n_per_direction=N_PER_DIRECTION,
    percentile=PERCENTILE,
    split_filter=SPLIT_FILTER,
    seed=RANDOM_SEED,
):
    """
    Visualise both CycleGAN directions.

    Default configuration is controlled from config.py:
        CYCLEGAN_VISUAL_QA_N_PER_DIRECTION
        CYCLEGAN_VISUAL_QA_PERCENTILE
        CYCLEGAN_VISUAL_QA_RANDOM_SEED
        CYCLEGAN_VISUAL_QA_SPLIT
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"CycleGAN metadata not found:\n{METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {
        "direction",
        "source_path",
        "source_split",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise KeyError(
            f"Missing required metadata columns: {sorted(missing)}"
        )

    print(f"Metadata: {METADATA_PATH}")
    print(f"Examples per direction: {n_per_direction}")
    print(f"Percentile: {percentile}")
    print(f"Split filter: {split_filter}")
    print(f"Random seed: {seed}")
    print()

    show_direction(
        metadata=metadata,
        direction="normal_to_pathology",
        n=n_per_direction,
        percentile=percentile,
        split_filter=split_filter,
        seed=seed,
    )

    show_direction(
        metadata=metadata,
        direction="pathology_to_normal",
        n=n_per_direction,
        percentile=percentile,
        split_filter=split_filter,
        seed=seed + 1,
    )


if __name__ == "__main__":
    visualise_all()
