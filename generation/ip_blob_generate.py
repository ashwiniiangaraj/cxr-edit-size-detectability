# ============================================================
# IP BLOB 
# ============================================================
#
# Generates exactly ONE blob-edited image for every real source image
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_common   

from config import (
    REAL_DIR,
    MANIFEST_DIR,
    IP_SYNTHETIC_ROOT,
    IP_CONFIG,
)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = REAL_DIR.parent

SOURCE_POOL = IP_CONFIG["source_pool"]
SPLIT_CSV = MANIFEST_DIR / f"{SOURCE_POOL}_all.csv"

IMAGE_DIR = (
    REAL_DIR
)

OUTPUT_DIR = (
    IP_SYNTHETIC_ROOT / "blob"
)

TRAIN_IMAGE_DIR = OUTPUT_DIR / "train"
TRAIN_MASK_DIR = OUTPUT_DIR / "masks" / "train"

VAL_IMAGE_DIR = OUTPUT_DIR / "val"
VAL_MASK_DIR = OUTPUT_DIR / "masks" / "val"

TEST_IMAGE_DIR = OUTPUT_DIR / "test"
TEST_MASK_DIR = OUTPUT_DIR / "masks" / "test"

METADATA_DIR = OUTPUT_DIR / "metadata"

for directory in [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    METADATA_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# Refuse to mix new PNGs with old JPEGs from a previous run.
for _d in (TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TEST_IMAGE_DIR):
    gen_common.ensure_clean_image_dir(_d, "blob")

# ------------------------------------------------------------
# 2. FINAL PRODUCTION SETTINGS
# ------------------------------------------------------------

WORKING_SIZE = tuple(IP_CONFIG["working_size"])

# Fixed safe ROI
ROI_WIDTH_FRACTION = IP_CONFIG["blob"]["roi_width_fraction"]
ROI_HEIGHT_FRACTION = IP_CONFIG["blob"]["roi_height_fraction"]

OPACITY_MIN = IP_CONFIG["blob"]["opacity_min"]
OPACITY_MAX = IP_CONFIG["blob"]["opacity_max"]

RADIUS_MIN = IP_CONFIG["blob"]["radius_min"]
RADIUS_MAX = IP_CONFIG["blob"]["radius_max"]

# Fixed circular shape
SHAPE = "circular"

# Derived smoothing
SIGMA_DIVISOR = IP_CONFIG["blob"]["sigma_divisor"]

# Adaptive intensity rule
BACKGROUND_THRESHOLD = IP_CONFIG["blob"]["background_threshold"]
BRIGHT_TARGET = IP_CONFIG["blob"]["bright_target"]
DARK_TARGET = IP_CONFIG["blob"]["dark_target"]

# Binary mask threshold
MASK_THRESHOLD = IP_CONFIG["blob"]["mask_threshold"]

# Reproducibility
MASTER_SEED = IP_CONFIG["master_seed"]

# ------------------------------------------------------------
# 3. LOAD LOCKED SPLIT
# ------------------------------------------------------------

split_df = pd.read_csv(SPLIT_CSV)

print("=" * 70)
print("FINAL BLOB SYNTHETIC DATASET GENERATION")
print("=" * 70)

print(f"\nTotal source images: {len(split_df)}")

# ------------------------------------------------------------
# 4. BASIC SPLIT CHECKS
# ------------------------------------------------------------

import config as _cfg   # sizes come from config.POOL_SIZE_PER_POOL
expected_splits = dict(_cfg.SPLIT_IMAGE_COUNTS)

actual_counts = (
    split_df["split"]
    .value_counts()
    .to_dict()
)

print("\nSplit counts:")

for split_name, expected_count in expected_splits.items():

    actual_count = actual_counts.get(
        split_name,
        0
    )

    print(
        f"  {split_name:12s}: "
        f"{actual_count} "
        f"(expected {expected_count})"
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Unexpected number of {split_name} images."
        )

if len(split_df) != _cfg.POOL_SIZE_PER_POOL:
    raise ValueError(
        f"Expected exactly {_cfg.POOL_SIZE_PER_POOL} source images."
    )

# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "split",
    "subject_id",
    "study_id",
    "dicom_id",
    "disease_status",
    "ViewPosition",
]

missing_columns = [
    col
    for col in required_columns
    if col not in split_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 6. HELPER — SOURCE IMAGE PATH
# ------------------------------------------------------------

def get_source_image_path(row):

    subject_id = str(row["subject_id"])
    study_id = str(row["study_id"])
    dicom_id = str(row["dicom_id"])

    disease = str(row["disease_status"]).strip().lower()
    
    filename = (
        f"p{subject_id}_"
        f"s{study_id}_"
        f"{dicom_id}.jpg"
    )

    return (
        IMAGE_DIR
        / str(row["split"])
        / disease
        / filename
    )

# ------------------------------------------------------------
# 7. HELPER — FIXED SAFE ROI
# ------------------------------------------------------------

def get_fixed_roi():

    width = WORKING_SIZE[0]
    height = WORKING_SIZE[1]

    roi_width = int(
        round(
            width
            * ROI_WIDTH_FRACTION
        )
    )

    roi_height = int(
        round(
            height
            * ROI_HEIGHT_FRACTION
        )
    )

    roi_x = int(
        round(
            (width - roi_width) / 2
        )
    )

    roi_y = int(
        round(
            (height - roi_height) / 2
        )
    )

    return (
        roi_x,
        roi_y,
        roi_width,
        roi_height
    )

# ------------------------------------------------------------
# 8. HELPER — RANDOM VALID LOCATION
# ------------------------------------------------------------

def sample_valid_location(
    rng,
    roi_x,
    roi_y,
    roi_width,
    roi_height,
    radius
):
    """
    Randomly sample the Blob centre while ensuring that the
    entire circular Blob remains inside the fixed safe ROI.

    The centre therefore cannot be closer to the ROI boundary
    than the Blob radius.
    """

    min_x = roi_x + radius
    max_x = (
        roi_x
        + roi_width
        - radius
    )

    min_y = roi_y + radius
    max_y = (
        roi_y
        + roi_height
        - radius
    )

    if min_x >= max_x:
        raise ValueError(
            "Invalid X range for Blob location."
        )

    if min_y >= max_y:
        raise ValueError(
            "Invalid Y range for Blob location."
        )

    center_x = rng.uniform(
        min_x,
        max_x
    )

    center_y = rng.uniform(
        min_y,
        max_y
    )

    return center_x, center_y

# ------------------------------------------------------------
# 9. HELPER — CREATE CIRCULAR SOFT MASK
# ------------------------------------------------------------

def create_circular_mask(
    center_x,
    center_y,
    radius,
):
    """
    Create a circular binary mask on the working canvas.
    """

    height = WORKING_SIZE[1]
    width = WORKING_SIZE[0]

    yy, xx = np.ogrid[
        :height,
        :width
    ]

    distance_squared = (
        (xx - center_x) ** 2
        +
        (yy - center_y) ** 2
    )

    mask = (
        distance_squared
        <= radius ** 2
    )

    return mask.astype(
        np.float32
    )

# ------------------------------------------------------------
# 10. GET FIXED ROI
# ------------------------------------------------------------

(
    ROI_X,
    ROI_Y,
    ROI_WIDTH,
    ROI_HEIGHT
) = get_fixed_roi()

print("\nFixed safe ROI:")
print(
    f"  x      = {ROI_X}"
)
print(
    f"  y      = {ROI_Y}"
)
print(
    f"  width  = {ROI_WIDTH}"
)
print(
    f"  height = {ROI_HEIGHT}"
)

# ------------------------------------------------------------
# 11. GENERATE DATASET
# ------------------------------------------------------------

all_records = []

missing_images = []
generation_errors = []

split_order = [
    "train",
    "val",
    "test",
]

for split_name in split_order:

    split_subset = (
        split_df[
            split_df["split"]
            == split_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("\n" + "-" * 70)
    print(
        f"Generating {split_name}: "
        f"{len(split_subset)} images"
    )
    print("-" * 70)

    if split_name == "train":

        image_output_dir = TRAIN_IMAGE_DIR
        mask_output_dir = TRAIN_MASK_DIR

    elif split_name == "val":

        image_output_dir = VAL_IMAGE_DIR
        mask_output_dir = VAL_MASK_DIR

    elif split_name == "test":

        image_output_dir = TEST_IMAGE_DIR
        mask_output_dir = TEST_MASK_DIR

    else:

        raise ValueError(
            f"Unknown split: {split_name}"
        )

    for index, row in split_subset.iterrows():

        source_path = get_source_image_path(
            row
        )

        calibration_seed = (
            MASTER_SEED
            + index
            + {
                "train": 100000,
                "val": 200000,
                "test": 300000,
            }[split_name]
        )

        rng = np.random.default_rng(
            calibration_seed
        )

        # ----------------------------------------------------
        # Check source image
        # ----------------------------------------------------

        if not source_path.exists():

            missing_images.append(
                str(source_path)
            )

            continue

        try:

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            # gen_common applies the frozen filter (Lanczos) and
            # converts to 8-bit grayscale. 'image' is a PIL image.
            image = gen_common.load_gray_image(
                source_path,
                WORKING_SIZE[0]
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Sample opacity
            # ------------------------------------------------

            opacity = rng.uniform(
                OPACITY_MIN,
                OPACITY_MAX
            )

            # ------------------------------------------------
            # Sample radius fraction
            # ------------------------------------------------

            radius_fraction = rng.uniform(
                RADIUS_MIN,
                RADIUS_MAX
            )

            # Radius relative to fixed ROI width
            radius_pixels = (
                radius_fraction
                * ROI_WIDTH
            )

            # ------------------------------------------------
            # Sample valid location
            # ------------------------------------------------

            center_x, center_y = (
                sample_valid_location(
                    rng=rng,
                    roi_x=ROI_X,
                    roi_y=ROI_Y,
                    roi_width=ROI_WIDTH,
                    roi_height=ROI_HEIGHT,
                    radius=radius_pixels,
                )
            )

            # ------------------------------------------------
            # Create circular mask
            # ------------------------------------------------

            binary_geometry = (
                create_circular_mask(
                    center_x=center_x,
                    center_y=center_y,
                    radius=radius_pixels,
                )
            )

            # ------------------------------------------------
            # Gaussian smoothing
            # ------------------------------------------------

            sigma = (
                radius_pixels
                / SIGMA_DIVISOR
            )

            mask_image = Image.fromarray(
                np.uint8(
                    binary_geometry
                    * 255
                )
            )

            mask_image = mask_image.filter(
                ImageFilter.GaussianBlur(
                    radius=float(sigma)
                )
            )

            soft_mask = (
                np.asarray(
                    mask_image,
                    dtype=np.float32
                )
                / 255.0
            )

            # ------------------------------------------------
            # Local background mean
            # ------------------------------------------------
            #
            # Use the original image pixels inside the
            # geometric Blob region.
            # ------------------------------------------------

            local_region = (
                binary_geometry > 0
            )

            if np.any(local_region):

                local_background_mean = float(
                    np.mean(
                        image_array[
                            local_region
                        ]
                    )
                )

            else:

                local_background_mean = float(
                    np.mean(image_array)
                )

            # ------------------------------------------------
            # Adaptive target intensity
            # ------------------------------------------------

            if (
                local_background_mean
                < BACKGROUND_THRESHOLD
            ):

                target_intensity = (
                    BRIGHT_TARGET
                )

            else:

                target_intensity = (
                    DARK_TARGET
                )

            # ------------------------------------------------
            # Blend Blob
            # ------------------------------------------------

            synthetic = (
                image_array
                +
                opacity
                * soft_mask
                * (
                    target_intensity
                    - image_array
                )
            )

            # FREEZE FIX: round to nearest grey level (astype alone
            # truncates, darkening edited pixels by ~0.5 on average).
            synthetic = np.rint(np.clip(
                synthetic,
                0,
                255
            )).astype(
                np.uint8
            )

            # ------------------------------------------------
            # Binary ground-truth mask
            # ------------------------------------------------

            binary_mask = (
                soft_mask
                >= MASK_THRESHOLD
            )

            binary_mask = np.uint8(
                binary_mask * 255
            )

            # ------------------------------------------------
            # File name
            # ------------------------------------------------

            subject_id = str(
                row["subject_id"]
            )

            study_id = str(
                row["study_id"]
            )

            dicom_id = str(
                row["dicom_id"]
            )

            # Preserve the original DICOM identifier as the filename.
            # The technique-specific directory provides the synthetic identity.
            # Preserve the exact source-image basename for the synthetic image.
            base_name = (
                f"p{subject_id}_"
                f"s{study_id}_"
                f"{dicom_id}"
            )

            synthetic_path = (
                image_output_dir
                / f"{base_name}{gen_common.IMAGE_EXT}"
            )

            mask_path = (
                mask_output_dir
                / f"{base_name}_mask.png"
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            # Lossless PNG; gen_common re-reads it and checks that
            # every pixel is identical to 'synthetic'.
            gen_common.save_gray_png(
                synthetic,
                synthetic_path
            )

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            Image.fromarray(
                binary_mask
            ).save(
                mask_path,
                format="PNG"
            )

            # ------------------------------------------------
            # Verify output files
            # ------------------------------------------------

            if not synthetic_path.exists():
                raise RuntimeError(
                    "Synthetic image was not created."
                )

            if not mask_path.exists():
                raise RuntimeError(
                    "Mask was not created."
                )

            # ------------------------------------------------
            # Verify mask is non-empty
            # ------------------------------------------------

            mask_pixels = int(
                np.sum(
                    binary_mask > 0
                )
            )

            if mask_pixels == 0:
                raise RuntimeError(
                    "Generated Blob mask is empty."
                )

            # ------------------------------------------------
            # Store complete provenance
            # ------------------------------------------------

            # How strongly the blob changed the image (0-255 scale).
            strength = gen_common.edit_strength(
                image_array,
                synthetic,
                binary_mask
            )

            record = {
                "generator": "blob",

                "generator_version": gen_common.GENERATOR_VERSION,
                "resize_filter": gen_common.RESIZE_FILTER_NAME,
                "output_format": gen_common.IMAGE_FORMAT,
                **strength,

                "split": split_name,

                "subject_id": row[
                    "subject_id"
                ],

                "study_id": row[
                    "study_id"
                ],

                "dicom_id": row[
                    "dicom_id"
                ],

                "disease_status": row[
                    "disease_status"
                ],

                "ViewPosition": row[
                    "ViewPosition"
                ],

                "shape": SHAPE,

                "working_width": WORKING_SIZE[0],
                "working_height": WORKING_SIZE[1],

                "roi_x": ROI_X,
                "roi_y": ROI_Y,
                "roi_width": ROI_WIDTH,
                "roi_height": ROI_HEIGHT,

                "opacity": float(
                    opacity
                ),

                "radius_fraction": float(
                    radius_fraction
                ),

                "radius_pixels": float(
                    radius_pixels
                ),

                "center_x": float(
                    center_x
                ),

                "center_y": float(
                    center_y
                ),

                "x_relative": float(
                    (center_x - ROI_X)
                    / ROI_WIDTH
                ),

                "y_relative": float(
                    (center_y - ROI_Y)
                    / ROI_HEIGHT
                ),

                "sigma": float(
                    sigma
                ),

                "local_background_mean": float(
                    local_background_mean
                ),

                "background_threshold": (
                    BACKGROUND_THRESHOLD
                ),

                "target_intensity": float(
                    target_intensity
                ),

                "mask_threshold": (
                    MASK_THRESHOLD
                ),

                "mask_pixels": mask_pixels,

                "source_image_path": str(
                    source_path
                ),

                "synthetic_path": str(
                    synthetic_path
                ),

                "mask_path": str(
                    mask_path
                ),

                "seed": int(
                    calibration_seed
                ),
            }

            all_records.append(
                record
            )

        except Exception as exc:

            generation_errors.append({
                "split": split_name,
                "calibration_index": index,
                "source_path": str(
                    source_path
                ),
                "error": str(exc),
            })

    print(
        f"Completed {split_name}"
    )

# ------------------------------------------------------------
# 12. SAVE PROVENANCE
# ------------------------------------------------------------

metadata_df = pd.DataFrame(
    all_records
)

metadata_path = (
    METADATA_DIR
    / "blob_synthetic_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# Output folders must hold exactly the images written by this run.
_written = list(metadata_df["synthetic_path"]) if len(metadata_df) else []
for _d in (TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TEST_IMAGE_DIR):
    gen_common.assert_output_matches(_d, _written, "blob")

# ------------------------------------------------------------
# 13. GENERATION SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATION SUMMARY")
print("=" * 70)

print(
    f"\nSuccessfully generated: "
    f"{len(metadata_df)}"
)

print(
    f"Missing source images: "
    f"{len(missing_images)}"
)

print(
    f"Generation errors: "
    f"{len(generation_errors)}"
)

print("\nSynthetic images by split:")

print(
    metadata_df["split"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSynthetic images by disease:")

print(
    metadata_df["disease_status"]
    .value_counts()
    .to_string()
)

print("\nSynthetic images by split and disease:")

print(
    pd.crosstab(
        metadata_df["split"],
        metadata_df["disease_status"]
    ).to_string()
)

# ------------------------------------------------------------
# 14. PARAMETER SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLED PARAMETER SUMMARY")
print("=" * 70)

if len(metadata_df) > 0:

    print(
        metadata_df[
            [
                "opacity",
                "radius_fraction",
                "radius_pixels",
                "x_relative",
                "y_relative",
                "sigma",
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )

# ------------------------------------------------------------
# 15. CHECK PARAMETER RANGES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PARAMETER RANGE CHECK")
print("=" * 70)

if len(metadata_df) > 0:

    opacity_ok = (
        metadata_df["opacity"]
        .between(
            OPACITY_MIN,
            OPACITY_MAX,
            inclusive="both"
        )
        .all()
    )

    radius_ok = (
        metadata_df["radius_fraction"]
        .between(
            RADIUS_MIN,
            RADIUS_MAX,
            inclusive="both"
        )
        .all()
    )

    x_ok = (
        metadata_df["x_relative"]
        .between(0, 1)
        .all()
    )

    y_ok = (
        metadata_df["y_relative"]
        .between(0, 1)
        .all()
    )

    print(
        f"Opacity range valid       : {opacity_ok}"
    )

    print(
        f"Radius range valid        : {radius_ok}"
    )

    print(
        f"Relative X location valid : {x_ok}"
    )

    print(
        f"Relative Y location valid : {y_ok}"
    )

# ------------------------------------------------------------
# 16. SAVE ERROR LOG IF REQUIRED
# ------------------------------------------------------------

if generation_errors:

    errors_df = pd.DataFrame(
        generation_errors
    )

    errors_path = (
        METADATA_DIR
        / "blob_generation_errors.csv"
    )

    errors_df.to_csv(
        errors_path,
        index=False
    )

    print(
        f"\nError log:"
        f"\n{errors_path}"
    )

# ------------------------------------------------------------
# 17. FINAL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BLOB GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nExpected synthetic images: {_cfg.POOL_SIZE_PER_POOL}"
)

print(
    f"Actual synthetic images:   "
    f"{len(metadata_df)}"
)

if len(metadata_df) == _cfg.POOL_SIZE_PER_POOL:

    print(
        f"\nSTATUS: SUCCESS — exactly {_cfg.POOL_SIZE_PER_POOL} "
        "Blob synthetic images generated."
    )

else:

    print(
        "\nSTATUS: CHECK REQUIRED — "
        f"expected {_cfg.POOL_SIZE_PER_POOL} synthetic images."
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMetadata:")
print(metadata_path)

print(
    "\nEach synthetic image has:"
    "\n  - one source real image"
    "\n  - one Blob mask"
    "\n  - sampled opacity"
    "\n  - sampled radius"
    "\n  - sampled location"
    "\n  - derived smoothing"
    "\n  - adaptive target intensity"
    "\n  - complete provenance"
)