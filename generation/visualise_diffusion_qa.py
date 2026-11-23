"""
RadEdit visual QA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config


META_PATH = Path(config.DIFFUSION_PRODUCTION_METADATA)
OUT_DIR = Path(config.DIFFUSION_VISUAL_QA_OUTPUT_DIR)
N_PER_DIRECTION = config.DIFFUSION_VISUAL_QA_N_PER_DIRECTION
SPLIT_FILTER = config.DIFFUSION_VISUAL_QA_SPLIT
RANDOM_SEED = config.DIFFUSION_VISUAL_QA_RANDOM_SEED
DISPLAY_SIZE = config.DIFFUSION_FINAL_SIZE


def source_path(row):
    split = str(row["split"]).lower()
    disease = str(row["source_disease_status"]).lower()
    subject = str(row["subject_id"])
    study = str(row["study_id"])
    dicom = str(row["dicom_id"])

    return (
        Path(config.DIFFUSION_SOURCE_ROOT)
        / split
        / disease
        / f"p{subject}_s{study}_{dicom}.jpg"
    )


def load_gray(path):
    return np.asarray(
        Image.open(path).convert("L").resize(
            (DISPLAY_SIZE, DISPLAY_SIZE),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    )


def select_cases(meta, direction):
    subset = meta[meta["direction"] == direction].copy()

    if SPLIT_FILTER is not None:
        subset = subset[subset["split"] == SPLIT_FILTER].copy()

    if len(subset) == 0:
        return subset

    selected = []
    selected_ids = set()

    # Preserve the HTML's coverage logic when possible.
    group_cols = ["split", "source_disease_status", "prompt"]

    for _, group in subset.groupby(group_cols, sort=True):
        row = group.sort_values("dicom_id").iloc[0]
        selected.append(row.to_dict())
        selected_ids.add(str(row["dicom_id"]))

    remaining = subset[
        ~subset["dicom_id"].astype(str).isin(selected_ids)
    ].sort_values("dicom_id")

    needed = max(0, N_PER_DIRECTION - len(selected))

    if needed > 0:
        # Use a fixed random sample for the remaining cases.
        remaining = remaining.sample(
            n=min(needed, len(remaining)),
            random_state=RANDOM_SEED,
        )
        selected.extend(remaining.to_dict("records"))

    return pd.DataFrame(selected).head(N_PER_DIRECTION)


def make_contact_sheet(direction_df, direction_name):
    if len(direction_df) == 0:
        print(f"No cases available for {direction_name}.")
        return

    n = len(direction_df)

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(12, 3.2 * n),
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (_, row) in enumerate(direction_df.iterrows()):
        real_path = source_path(row)
        synthetic_path = Path(row["output_path"])
        mask_path = Path(row["edit_mask"])

        real = load_gray(real_path)
        synthetic = load_gray(synthetic_path)

        mask = np.asarray(
            Image.open(mask_path)
            .convert("L")
            .resize(
                (DISPLAY_SIZE, DISPLAY_SIZE),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )

        difference = np.abs(synthetic - real)
        vmax = np.percentile(difference, 99)

        axes[i, 0].imshow(
            real,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[i, 0].set_title("Real source")

        axes[i, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[i, 1].set_title("Edit mask")

        axes[i, 2].imshow(
            synthetic,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[i, 2].set_title("RadEdit output")

        axes[i, 3].imshow(
            difference,
            cmap="inferno",
            vmin=0,
            vmax=max(vmax, 1),
        )
        axes[i, 3].set_title("Absolute difference")

        label = (
            f"{str(row['dicom_id'])[:8]} | "
            f"{row['split']} | "
            f"{row['source_disease_status']} | "
            f"{row['prompt']}"
        )

        axes[i, 0].set_ylabel(label, fontsize=7)

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(
        f"RadEdit Visual QA — {direction_name}",
        fontsize=16,
    )

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUT_DIR / f"RadEdit_visual_QA_{direction_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def visualise_all():
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"RadEdit metadata not found: {META_PATH}"
        )

    meta = pd.read_csv(META_PATH)

    required = {
        "dicom_id",
        "subject_id",
        "study_id",
        "split",
        "source_disease_status",
        "direction",
        "prompt",
        "edit_mask",
        "output_path",
    }

    missing = required - set(meta.columns)
    if missing:
        raise KeyError(
            f"Metadata missing columns: {sorted(missing)}"
        )

    print(f"Metadata: {META_PATH}")
    print(f"Cases per direction: {N_PER_DIRECTION}")
    print(f"Split filter: {SPLIT_FILTER}")

    for direction in [
        "Normal_to_Pathology",
        "Pathology_to_Normal",
    ]:
        selected = select_cases(meta, direction)
        print(
            f"{direction}: {len(selected)} selected"
        )
        make_contact_sheet(selected, direction)

    print("\nRadEdit visual QA complete.")


if __name__ == "__main__":
    visualise_all()
