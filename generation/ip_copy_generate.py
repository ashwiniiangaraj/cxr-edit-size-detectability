# ============================================================
# IP COPY-MOVE 
# ============================================================

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json
import hashlib
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_common  

from config import (
    REAL_DIR,
    MANIFEST_DIR,
    IP_SYNTHETIC_ROOT,
    IP_CONFIG,
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = REAL_DIR.parent

SPLIT_CSV = (
    MANIFEST_DIR / f"{IP_CONFIG['source_pool']}_all.csv"
)

# ------------------------------------------------------------
# Final Copy-Move output
# ------------------------------------------------------------

OUTPUT_ROOT = (
    IP_SYNTHETIC_ROOT / "copy"
)

# Synthetic images live directly under technique/train|val|test.
IMAGE_ROOT = OUTPUT_ROOT

MASK_ROOT = OUTPUT_ROOT / "masks"

REPORT_ROOT = (
    OUTPUT_ROOT
    / "reports"
)

CONFIG_ROOT = (
    OUTPUT_ROOT
    / "config"
)

# Create directories.
for split in [
    "train",
    "val",
    "test"
]:

    (
        IMAGE_ROOT
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        MASK_ROOT
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

for _split in ["train", "val", "test"]:
    gen_common.ensure_clean_image_dir(IMAGE_ROOT / _split, "copy-move")

CONFIG_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

MASTER_SEED = IP_CONFIG["master_seed"]

np.random.seed(
    MASTER_SEED
)

random.seed(
    MASTER_SEED
)


# ============================================================
# 3. WORKING IMAGE SETTINGS
# ============================================================

WORKING_SIZE = IP_CONFIG["working_size"][0]


# ============================================================
# 4. LOCKED COPY-MOVE PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Patch size
# ------------------------------------------------------------

PATCH_MIN = IP_CONFIG["copy"]["patch_min"]
PATCH_MAX = IP_CONFIG["copy"]["patch_max"]


# ------------------------------------------------------------
# Rotation
# ------------------------------------------------------------

ROTATION_MIN = IP_CONFIG["copy"]["rotation_min"]
ROTATION_MAX = IP_CONFIG["copy"]["rotation_max"]


# ------------------------------------------------------------
# Scale
# ------------------------------------------------------------

SCALE_MIN = IP_CONFIG["copy"]["scale_min"]
SCALE_MAX = IP_CONFIG["copy"]["scale_max"]


# ------------------------------------------------------------
# Feathering
# ------------------------------------------------------------

FEATHER_FRACTION = IP_CONFIG["copy"]["feather_fraction"]


# ------------------------------------------------------------
# Source/destination separation
# ------------------------------------------------------------

# Read from config.py 
MIN_DISTANCE_MULTIPLIER = IP_CONFIG["copy"]["min_distance_multiplier"]
     


# ------------------------------------------------------------
# Destination selection
# ------------------------------------------------------------

N_DESTINATION_CANDIDATES = IP_CONFIG["copy"]["n_destination_candidates"]

TOP_DESTINATION_CANDIDATES = IP_CONFIG["copy"]["top_destination_candidates"]


# ============================================================
# 5. FIXED CONSERVATIVE ROI
# ============================================================

ROI_X = IP_CONFIG["copy"]["roi_x"]
ROI_Y = IP_CONFIG["copy"]["roi_y"]

ROI_WIDTH = IP_CONFIG["copy"]["roi_width"]
ROI_HEIGHT = IP_CONFIG["copy"]["roi_height"]


# ============================================================
# 6. HELPER — DETERMINISTIC SEED
# ============================================================

def make_seed(*values):
    """
    Create a reproducible integer seed from several values.
    """

    text = "|".join(
        str(v)
        for v in values
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16
    )


# ============================================================
# 7. HELPER — CLEAN IDENTIFIERS
# ============================================================

def clean_id(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove pandas-style .0 from integer identifiers.
    if text.endswith(".0"):
        text = text[:-2]

    return text


# ============================================================
# 8. HELPER — FIND MIMIC IMAGE
# ============================================================

def find_mimic_image(subject_id, study_id, dicom_id, disease_status, split_name):
    """Return the source image from the locked real dataset structure."""
    disease = clean_id(disease_status).lower()
    # The locked real dataset uses:
    #     p{subject_id}_s{study_id}_{dicom_id}.jpg
    filename = (
        f"p{clean_id(subject_id)}_"
        f"s{clean_id(study_id)}_"
        f"{clean_id(dicom_id)}.jpg"
    )

    path = (
        REAL_DIR
        / clean_id(split_name)
        / disease
        / filename
    )
    if not path.exists():
        raise FileNotFoundError(f"Source image not found: {path}")
    return path


# ============================================================
# 9. HELPER — LOAD WORKING IMAGE
# ============================================================

def load_working_image(path):
    """
    Load grayscale image and resize to the working canvas (256 x 256) with Lanczos.

    path : real source image (any size)
    returns float32 array (256, 256), values 0-255
    """

    # gen_common: PIL open -> grayscale -> Lanczos resize -> float32
    return gen_common.load_gray_array(
        path,
        WORKING_SIZE
    )


# ============================================================
# 10. HELPER — RANDOM PATCH LOCATION
# ============================================================

def random_point_inside_roi(
    rng,
    patch_size
):
    """
    Select a square patch completely inside the fixed ROI.
    """

    max_x = (
        ROI_X
        + ROI_WIDTH
        - patch_size
    )

    max_y = (
        ROI_Y
        + ROI_HEIGHT
        - patch_size
    )

    if (
        max_x < ROI_X
        or
        max_y < ROI_Y
    ):

        raise ValueError(
            "Patch is too large for the fixed ROI."
        )

    x = int(
        rng.integers(
            ROI_X,
            max_x + 1
        )
    )

    y = int(
        rng.integers(
            ROI_Y,
            max_y + 1
        )
    )

    return x, y


# ============================================================
# 11. HELPER — PATCH CENTRE
# ============================================================

def patch_center(
    x,
    y,
    patch_size
):

    return (
        x + patch_size / 2.0,
        y + patch_size / 2.0
    )


# ============================================================
# 12. HELPER — PHOTOMETRIC COMPATIBILITY
# ============================================================

def patch_compatibility_score(
    source_patch,
    destination_patch
):
    """
    Lower score means the source and destination regions
    are more photometrically compatible.
    """

    source_mean = float(
        np.mean(source_patch)
    )

    destination_mean = float(
        np.mean(destination_patch)
    )

    source_std = float(
        np.std(source_patch)
    )

    destination_std = float(
        np.std(destination_patch)
    )

    mean_difference = abs(
        source_mean
        -
        destination_mean
    )

    std_difference = abs(
        source_std
        -
        destination_std
    )

    return (
        mean_difference
        +
        std_difference
    )


# ============================================================
# 13. HELPER — LOCAL STATISTICAL MATCHING
# ============================================================

def match_patch_statistics(
    source_patch,
    destination_patch
):
    """
    Match source mean/std to destination mean/std.

    This reduces obvious brightness and contrast mismatch.
    """

    source = (
        source_patch
        .astype(np.float32)
    )

    destination = (
        destination_patch
        .astype(np.float32)
    )

    source_mean = np.mean(
        source
    )

    source_std = np.std(
        source
    )

    destination_mean = np.mean(
        destination
    )

    destination_std = np.std(
        destination
    )

    if source_std < 1e-6:

        adjusted = (
            source
            -
            source_mean
            +
            destination_mean
        )

    else:

        adjusted = (
            (source - source_mean)
            /
            source_std
        )

        adjusted = (
            adjusted
            *
            destination_std
            +
            destination_mean
        )

    adjusted = np.clip(
        adjusted,
        0,
        255
    )

    return adjusted.astype(
        np.float32
    )


# ============================================================
# 14. HELPER — FEATHER MASK
# ============================================================

def create_feather_mask(
    size,
    feather_fraction
):
    """
    Create a smooth square blending mask.
    """

    feather_pixels = max(
        1,
        int(
            round(
                size
                *
                feather_fraction
            )
        )
    )

    yy, xx = np.indices(
        (size, size)
    )

    distance_to_edge = (
        np.minimum.reduce([
            xx,
            yy,
            size - 1 - xx,
            size - 1 - yy
        ])
        .astype(np.float32)
    )

    mask = np.clip(
        distance_to_edge
        /
        feather_pixels,
        0,
        1
    )

    sigma = max(
        0.8,
        feather_pixels / 3.0
    )

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    return np.clip(
        mask,
        0,
        1
    )


# ============================================================
# 15. HELPER — TRANSFORM SOURCE PATCH
# ============================================================

def transform_patch(
    patch,
    rotation_degrees,
    scale
):
    """
    Apply rotation and isotropic scaling.

    The output remains the same patch footprint so that
    the destination GT mask size remains tied to the
    sampled patch size.
    """

    size = patch.shape[0]

    center = (
        size / 2.0,
        size / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        rotation_degrees,
        scale
    )

    transformed = cv2.warpAffine(
        patch,
        matrix,
        (
            size,
            size
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# 16. HELPER — CHOOSE SOURCE AND DESTINATION
# ============================================================

def choose_source_destination(
    image,
    patch_size,
    rng
):
    """
    Select source and destination regions.

    Destination selection considers local photometric
    compatibility while preserving spatial randomness.
    """

    # --------------------------------------------------------
    # Source location
    # --------------------------------------------------------

    source_x, source_y = (
        random_point_inside_roi(
            rng,
            patch_size
        )
    )

    (
        source_cx,
        source_cy
    ) = patch_center(
        source_x,
        source_y,
        patch_size
    )

    source_patch = image[
        source_y:
            source_y + patch_size,
        source_x:
            source_x + patch_size
    ]

    if source_patch.shape != (
        patch_size,
        patch_size
    ):

        raise RuntimeError(
            "Invalid source patch shape."
        )

    # --------------------------------------------------------
    # Minimum separation
    # --------------------------------------------------------

    minimum_distance = (
        MIN_DISTANCE_MULTIPLIER
        *
        patch_size
    )

    candidates = []

    # --------------------------------------------------------
    # Test random destination candidates.
    # --------------------------------------------------------

    for _ in range(
        N_DESTINATION_CANDIDATES
    ):

        (
            destination_x,
            destination_y
        ) = random_point_inside_roi(
            rng,
            patch_size
        )

        (
            destination_cx,
            destination_cy
        ) = patch_center(
            destination_x,
            destination_y,
            patch_size
        )

        distance = np.sqrt(
            (
                destination_cx
                -
                source_cx
            ) ** 2
            +
            (
                destination_cy
                -
                source_cy
            ) ** 2
        )

        if distance < minimum_distance:
            continue

        destination_patch = image[
            destination_y:
                destination_y + patch_size,
            destination_x:
                destination_x + patch_size
        ]

        if destination_patch.shape != (
            patch_size,
            patch_size
        ):
            continue

        compatibility = (
            patch_compatibility_score(
                source_patch,
                destination_patch
            )
        )

        candidates.append(
            (
                compatibility,
                destination_x,
                destination_y,
                destination_cx,
                destination_cy,
                distance
            )
        )

    if len(candidates) == 0:

        raise RuntimeError(
            "Could not find a valid destination."
        )

    # Best photometric matches first.
    candidates.sort(
        key=lambda item: item[0]
    )

    top_n = min(
        TOP_DESTINATION_CANDIDATES,
        len(candidates)
    )

    selected = candidates[
        int(
            rng.integers(
                0,
                top_n
            )
        )
    ]

    (
        compatibility,
        destination_x,
        destination_y,
        destination_cx,
        destination_cy,
        distance
    ) = selected

    return {
        "source_x":
            source_x,

        "source_y":
            source_y,

        "source_center_x":
            source_cx,

        "source_center_y":
            source_cy,

        "destination_x":
            destination_x,

        "destination_y":
            destination_y,

        "destination_center_x":
            destination_cx,

        "destination_center_y":
            destination_cy,

        "source_destination_distance":
            distance,

        "minimum_required_distance":
            minimum_distance,

        "destination_compatibility_score":
            compatibility
    }


# ============================================================
# 17. CREATE ONE FINAL COPY-MOVE IMAGE
# ============================================================

def create_copy_move(
    image,
    patch_fraction,
    rotation_degrees,
    scale,
    rng
):
    """
    Create one final Copy-Move synthetic image.
    """

    # --------------------------------------------------------
    # Patch size.
    # --------------------------------------------------------

    patch_size = max(
        4,
        int(
            round(
                patch_fraction
                *
                ROI_WIDTH
            )
        )
    )

    # --------------------------------------------------------
    # Source/destination.
    # --------------------------------------------------------

    locations = (
        choose_source_destination(
            image,
            patch_size,
            rng
        )
    )

    source_x = locations[
        "source_x"
    ]

    source_y = locations[
        "source_y"
    ]

    destination_x = locations[
        "destination_x"
    ]

    destination_y = locations[
        "destination_y"
    ]

    # --------------------------------------------------------
    # Extract patches.
    # --------------------------------------------------------

    source_patch = image[
        source_y:
            source_y + patch_size,
        source_x:
            source_x + patch_size
    ].copy()

    destination_patch = image[
        destination_y:
            destination_y + patch_size,
        destination_x:
            destination_x + patch_size
    ].copy()

    # --------------------------------------------------------
    # Transform source.
    # --------------------------------------------------------

    transformed_patch = (
        transform_patch(
            source_patch,
            rotation_degrees,
            scale
        )
    )

    # --------------------------------------------------------
    # Match local photometric statistics.
    # --------------------------------------------------------

    transformed_patch = (
        match_patch_statistics(
            transformed_patch,
            destination_patch
        )
    )

    # --------------------------------------------------------
    # Create feather mask.
    # --------------------------------------------------------

    soft_mask = (
        create_feather_mask(
            patch_size,
            FEATHER_FRACTION
        )
    )

    # --------------------------------------------------------
    # Blend.
    # --------------------------------------------------------

    blended_patch = (
        destination_patch
        *
        (1.0 - soft_mask)
        +
        transformed_patch
        *
        soft_mask
    )

    # Small final smoothing.
    blended_patch = (
        cv2.GaussianBlur(
            blended_patch,
            (0, 0),
            sigmaX=0.6,
            sigmaY=0.6
        )
    )

    # --------------------------------------------------------
    # Insert manipulated destination.
    # --------------------------------------------------------

    synthetic = image.copy()

    synthetic[
        destination_y:
            destination_y + patch_size,
        destination_x:
            destination_x + patch_size
    ] = blended_patch

    # FREEZE FIX: round to nearest grey level (astype alone truncates,
    # darkening edited pixels by ~0.5 on average).
    synthetic = np.rint(np.clip(
        synthetic,
        0,
        255
    )).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Destination-only GT mask.
    # --------------------------------------------------------

    gt_mask = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    gt_mask[
        destination_y:
            destination_y + patch_size,
        destination_x:
            destination_x + patch_size
    ] = 255

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    metadata = {
        "patch_fraction":
            patch_fraction,

        "patch_size_pixels":
            patch_size,

        "rotation_degrees":
            rotation_degrees,

        "scale":
            scale,

        "feather_fraction":
            FEATHER_FRACTION,

        **locations,

        "mask_pixels":
            int(
                np.sum(
                    gt_mask > 0
                )
            )
    }

    return (
        synthetic,
        gt_mask,
        metadata
    )


# ============================================================
# 18. LOAD LOCKED SPLIT
# ============================================================

if not SPLIT_CSV.exists():

    raise FileNotFoundError(
        f"Locked split file not found:\n"
        f"{SPLIT_CSV}"
    )

split_df = pd.read_csv(
    SPLIT_CSV
)

print("=" * 75)
print("FINAL COPY-MOVE PRODUCTION")
print("=" * 75)

print(
    f"Total split rows: {len(split_df)}"
)

print()

print(
    split_df[
        "split"
    ]
    .value_counts()
    .sort_index()
)

print()


# ============================================================
# 19. CHECK SPLIT COUNTS
# ============================================================

import config as _cfg   # sizes come from config.POOL_SIZE_PER_POOL
expected_split_counts = dict(_cfg.SPLIT_IMAGE_COUNTS)

actual_counts = (
    split_df[
        "split"
    ]
    .value_counts()
    .to_dict()
)

for split_name, expected_count in (
    expected_split_counts.items()
):

    actual_count = actual_counts.get(
        split_name,
        0
    )

    if actual_count != expected_count:

        raise RuntimeError(
            f"Unexpected {split_name} count: "
            f"{actual_count}; "
            f"expected {expected_count}."
        )


# ============================================================
# 20. CHECK FOR DUPLICATE IMAGES
# ============================================================

image_keys = (
    split_df[
        [
            "subject_id",
            "study_id",
            "dicom_id"
        ]
    ]
    .astype(str)
    .agg(
        "_".join,
        axis=1
    )
)

if image_keys.duplicated().any():

    duplicates = (
        split_df[
            image_keys.duplicated(
                keep=False
            )
        ]
    )

    raise RuntimeError(
        "Duplicate source images detected "
        "in locked split."
    )


# ============================================================
# 21. SAVE PRODUCTION CONFIG
# ============================================================

production_config = {

    "dataset":
        "Copy-Move",

    "version":
        "final_production_v1",

    "master_seed":
        MASTER_SEED,

    "working_size":
        WORKING_SIZE,

    "resize_filter":
        gen_common.RESIZE_FILTER_NAME,

    "output_format":
        gen_common.IMAGE_FORMAT,

    "patch_fraction":
        {
            "distribution":
                "Uniform",
            "min":
                PATCH_MIN,
            "max":
                PATCH_MAX
        },

    "rotation":
        {
            "distribution":
                "Uniform",
            "min_degrees":
                ROTATION_MIN,
            "max_degrees":
                ROTATION_MAX
        },

    "scale":
        {
            "distribution":
                "Uniform",
            "min":
                SCALE_MIN,
            "max":
                SCALE_MAX
        },

    "feather_fraction":
        FEATHER_FRACTION,

    "roi":
        {
            "x":
                ROI_X,
            "y":
                ROI_Y,
            "width":
                ROI_WIDTH,
            "height":
                ROI_HEIGHT,
            "description":
                "Fixed conservative spatial placement "
                "region; not lung segmentation."
        },

    "minimum_distance_multiplier":
        MIN_DISTANCE_MULTIPLIER,

    "destination_candidates":
        N_DESTINATION_CANDIDATES,

    "top_destination_candidates":
        TOP_DESTINATION_CANDIDATES,

    "compositing":
        {
            "photometric_compatible_destination":
                True,
            "local_statistical_matching":
                True,
            "feathered_blending":
                True,
            "final_gaussian_smoothing_sigma":
                0.6
        },

    "ground_truth":
        {
            "mask_region":
                "destination_only",
            "source_region_included":
                False
        }
}

config_path = (
    CONFIG_ROOT
    / "copy_move_final_production_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        production_config,
        f,
        indent=2
    )


# ============================================================
# 22. GENERATE FINAL DATASET
# ============================================================

metadata_rows = []
error_rows = []

total_expected = len(
    split_df
)

successful = 0

for index, row in split_df.iterrows():

    # --------------------------------------------------------
    # Source identifiers.
    # --------------------------------------------------------

    subject_id = clean_id(
        row["subject_id"]
    )

    study_id = clean_id(
        row["study_id"]
    )

    dicom_id = clean_id(
        row["dicom_id"]
    )

    split_name = clean_id(
        row["split"]
    )

    disease_status = clean_id(
        row["disease_status"]
    )

    view_position = clean_id(
        row["ViewPosition"]
    )

    # --------------------------------------------------------
    # Find source image.
    # --------------------------------------------------------

    try:

        source_path = (
            find_mimic_image(
                subject_id,
                study_id,
                dicom_id,
                disease_status,
                split_name,
            )
        )

        image = (
            load_working_image(
                source_path
            )
        )

    except Exception as exc:

        error_rows.append({

            "subject_id":
                subject_id,

            "study_id":
                study_id,

            "dicom_id":
                dicom_id,

            "split":
                split_name,

            "error":
                f"Source image error: {exc}"
        })

        continue

    # --------------------------------------------------------
    # Deterministic per-image seed.
    # --------------------------------------------------------

    seed = make_seed(
        MASTER_SEED,
        "copy_move_final",
        subject_id,
        study_id,
        dicom_id
    )

    rng = np.random.default_rng(
        seed
    )

    try:

        # ----------------------------------------------------
        # Sample calibrated parameters continuously.
        # ----------------------------------------------------

        patch_fraction = float(
            rng.uniform(
                PATCH_MIN,
                PATCH_MAX
            )
        )

        rotation_degrees = float(
            rng.uniform(
                ROTATION_MIN,
                ROTATION_MAX
            )
        )

        scale = float(
            rng.uniform(
                SCALE_MIN,
                SCALE_MAX
            )
        )

        # ----------------------------------------------------
        # Create synthetic image.
        # ----------------------------------------------------

        (
            synthetic,
            gt_mask,
            meta
        ) = create_copy_move(
            image=image,
            patch_fraction=patch_fraction,
            rotation_degrees=rotation_degrees,
            scale=scale,
            rng=rng
        )

        # ----------------------------------------------------
        # Output names.
        # ----------------------------------------------------
        #
        # Keep the original DICOM identifier in the filename
        # so the synthetic image can always be traced back to
        # its real source.
        # --------------------------------------------------------

        # Preserve the exact source-image basename.
        image_filename = (
            f"p{subject_id}_"
            f"s{study_id}_"
            f"{dicom_id}{gen_common.IMAGE_EXT}"
        )

        mask_filename = (
            f"p{subject_id}_"
            f"s{study_id}_"
            f"{dicom_id}_mask.png"
        )

        image_path = (
            IMAGE_ROOT
            / split_name
            / image_filename
        )

        mask_path = (
            MASK_ROOT
            / split_name
            / mask_filename
        )

        # ----------------------------------------------------
        # Save image.
        # ----------------------------------------------------

        # Lossless PNG, with a read-back check that every pixel matches.
        gen_common.save_gray_png(
            synthetic,
            image_path
        )

        # ----------------------------------------------------
        # Save mask.
        # ----------------------------------------------------

        mask_ok = cv2.imwrite(
            str(mask_path),
            gt_mask
        )

        if not mask_ok:

            raise RuntimeError(
                "Failed to save GT mask."
            )

        # ----------------------------------------------------
        # Edit strength (how much the image changed, 0-255 scale).
        # ----------------------------------------------------

        strength = gen_common.edit_strength(
            image,
            synthetic,
            gt_mask
        )

        # ----------------------------------------------------
        # Verify source/destination separation.
        # --------------------------------------------------------

        separation_ok = (
            meta[
                "source_destination_distance"
            ]
            >=
            meta[
                "minimum_required_distance"
            ]
        )

        if not separation_ok:

            raise RuntimeError(
                "Source/destination separation failed."
            )

        # ----------------------------------------------------
        # Verify dimensions.
        # --------------------------------------------------------

        if synthetic.shape != (
            WORKING_SIZE,
            WORKING_SIZE
        ):

            raise RuntimeError(
                "Synthetic image has incorrect dimensions."
            )

        if gt_mask.shape != (
            WORKING_SIZE,
            WORKING_SIZE
        ):

            raise RuntimeError(
                "GT mask has incorrect dimensions."
            )

        # ----------------------------------------------------
        # Verify files.
        # --------------------------------------------------------

        if not image_path.exists():

            raise RuntimeError(
                "Synthetic image missing after save."
            )

        if not mask_path.exists():

            raise RuntimeError(
                "GT mask missing after save."
            )

        # ----------------------------------------------------
        # Record metadata.
        # --------------------------------------------------------

        metadata_rows.append({

            "synthetic_id":
                f"{dicom_id}_copy_move",

            "subject_id":
                subject_id,

            "study_id":
                study_id,

            "dicom_id":
                dicom_id,

            "split":
                split_name,

            "disease_status":
                disease_status,

            "ViewPosition":
                view_position,

            "working_width":
                WORKING_SIZE,

            "working_height":
                WORKING_SIZE,

            "roi_x":
                ROI_X,

            "roi_y":
                ROI_Y,

            "roi_width":
                ROI_WIDTH,

            "roi_height":
                ROI_HEIGHT,

            "patch_fraction":
                meta[
                    "patch_fraction"
                ],

            "patch_size_pixels":
                meta[
                    "patch_size_pixels"
                ],

            "rotation_degrees":
                meta[
                    "rotation_degrees"
                ],

            "scale":
                meta[
                    "scale"
                ],

            "feather_fraction":
                meta[
                    "feather_fraction"
                ],

            "source_x":
                meta[
                    "source_x"
                ],

            "source_y":
                meta[
                    "source_y"
                ],

            "source_center_x":
                meta[
                    "source_center_x"
                ],

            "source_center_y":
                meta[
                    "source_center_y"
                ],

            "destination_x":
                meta[
                    "destination_x"
                ],

            "destination_y":
                meta[
                    "destination_y"
                ],

            "destination_center_x":
                meta[
                    "destination_center_x"
                ],

            "destination_center_y":
                meta[
                    "destination_center_y"
                ],

            "source_destination_distance":
                meta[
                    "source_destination_distance"
                ],

            "minimum_required_distance":
                meta[
                    "minimum_required_distance"
                ],

            "destination_compatibility_score":
                meta[
                    "destination_compatibility_score"
                ],

            "mask_pixels":
                meta[
                    "mask_pixels"
                ],

            "source_path":
                str(source_path),

            "synthetic_path":
                str(image_path),

            "mask_path":
                str(mask_path),

            "seed":
                seed,

            "separation_ok":
                separation_ok,

            "generator_version":
                gen_common.GENERATOR_VERSION,

            "resize_filter":
                gen_common.RESIZE_FILTER_NAME,

            "output_format":
                gen_common.IMAGE_FORMAT,

            **strength
        })

        successful += 1

    except Exception as exc:

        error_rows.append({

            "subject_id":
                subject_id,

            "study_id":
                study_id,

            "dicom_id":
                dicom_id,

            "split":
                split_name,

            "error":
                str(exc)
        })

    # --------------------------------------------------------
    # Progress.
    # --------------------------------------------------------

    completed = (
        successful
        +
        len(error_rows)
    )

    if (
        completed % 25 == 0
        or
        completed == total_expected
    ):

        print(
            f"Progress: "
            f"{completed}/{total_expected} "
            f"| successful={successful} "
            f"| errors={len(error_rows)}"
        )


# ============================================================
# 23. SAVE FINAL METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_path = (
    REPORT_ROOT
    / "copy_move_final_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# Output folders must hold exactly the images written by this run.
_written = list(metadata_df["synthetic_path"]) if len(metadata_df) else []
for _split in ("train", "val", "test"):
    gen_common.assert_output_matches(IMAGE_ROOT / _split, _written, "copy-move")


# ============================================================
# 24. SAVE ERRORS
# ============================================================

errors_df = pd.DataFrame(
    error_rows
)

errors_path = (
    REPORT_ROOT
    / "copy_move_final_errors.csv"
)

errors_df.to_csv(
    errors_path,
    index=False
)


# ============================================================
# 25. VERIFY FINAL COUNTS
# ============================================================

print()
print("=" * 75)
print("FINAL COPY-MOVE PRODUCTION — VERIFICATION")
print("=" * 75)

print(
    f"Expected images: "
    f"{total_expected}"
)

print(
    f"Successful: "
    f"{successful}"
)

print(
    f"Errors: "
    f"{len(error_rows)}"
)

print()

if len(metadata_df) > 0:

    print(
        "Generated images by split:"
    )

    print(
        metadata_df[
            "split"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Generated images by disease status:"
    )

    print(
        metadata_df[
            "disease_status"
        ]
        .value_counts()
    )

    print()

    print(
        "Sampled patch fraction range:"
    )

    print(
        f"min = "
        f"{metadata_df['patch_fraction'].min():.4f}"
    )

    print(
        f"max = "
        f"{metadata_df['patch_fraction'].max():.4f}"
    )

    print()

    print(
        "Sampled rotation range:"
    )

    print(
        f"min = "
        f"{metadata_df['rotation_degrees'].min():.4f}"
    )

    print(
        f"max = "
        f"{metadata_df['rotation_degrees'].max():.4f}"
    )

    print()

    print(
        "Sampled scale range:"
    )

    print(
        f"min = "
        f"{metadata_df['scale'].min():.4f}"
    )

    print(
        f"max = "
        f"{metadata_df['scale'].max():.4f}"
    )

    print()

    print(
        "Source/destination separation:"
    )

    print(
        f"minimum observed = "
        f"{metadata_df['source_destination_distance'].min():.2f}"
    )

    print(
        f"minimum required = "
        f"{metadata_df['minimum_required_distance'].min():.2f}"
    )

    print()

    print(
        "All separation constraints passed:"
    )

    print(
        bool(
            metadata_df[
                "separation_ok"
            ].all()
        )
    )


# ============================================================
# 26. FINAL FILE COUNTS ON DISK
# ============================================================

print()
print(
    "Files on disk:"
)

for split in [
    "train",
    "val",
    "test"
]:

    image_count = len(
        list(
            (
                IMAGE_ROOT
                / split
            ).glob("*" + gen_common.IMAGE_EXT)
        )
    )

    mask_count = len(
        list(
            (
                MASK_ROOT
                / split
            ).glob("*.png")
        )
    )

    print(
        f"{split:12s} "
        f"images={image_count:3d} "
        f"masks={mask_count:3d}"
    )


# ============================================================
# 27. FINAL STATUS
# ============================================================

print()
print("=" * 75)

if (
    successful == total_expected
    and
    len(error_rows) == 0
):

    print(
        "FINAL COPY-MOVE DATASET GENERATED SUCCESSFULLY"
    )

    print(
        f"{_cfg.POOL_SIZE_PER_POOL}/{_cfg.POOL_SIZE_PER_POOL} synthetic images generated."
    )

else:

    print(
        "WARNING: PRODUCTION GENERATION INCOMPLETE"
    )

    print(
        "Check the error report before proceeding."
    )

print("=" * 75)

print()
print(
    f"Images:\n{IMAGE_ROOT}"
)

print(
    f"Masks:\n{MASK_ROOT}"
)

print(
    f"Metadata:\n{metadata_path}"
)

print(
    f"Errors:\n{errors_path}"
)

print(
    f"Config:\n{config_path}"
)

print()