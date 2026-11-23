# ============================================================
# IP TEXTURE TRANSPLANT
# ============================================================

from pathlib import Path
import json
import random
import sys

import cv2
import numpy as np
import pandas as pd

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

# Locked split file
SPLIT_FILE = MANIFEST_DIR / f"{IP_CONFIG['source_pool']}_all.csv"

# Real MIMIC images
IMAGE_ROOT = REAL_DIR

# Final Texture Transplant output
OUTPUT_ROOT = IP_SYNTHETIC_ROOT / "texture"

IMAGE_DIR = OUTPUT_ROOT
MASK_DIR = OUTPUT_ROOT / "masks"
REPORT_DIR = OUTPUT_ROOT / "reports"
CONFIG_DIR = OUTPUT_ROOT / "config"


for directory in [
    IMAGE_DIR / "train",
    IMAGE_DIR / "val",
    IMAGE_DIR / "test",
    MASK_DIR / "train",
    MASK_DIR / "val",
    MASK_DIR / "test",
    REPORT_DIR,
    CONFIG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Refuse to mix new PNGs with old JPEGs from a previous run.
for _split in ["train", "val", "test"]:
    gen_common.ensure_clean_image_dir(IMAGE_DIR / _split, "texture")


# ============================================================
# 2. LOCKED PARAMETERS
# ============================================================

MASTER_SEED = IP_CONFIG["master_seed"]

WORK_SIZE = IP_CONFIG["working_size"][0]

# Conservative central spatial placement region.
# This is NOT a lung segmentation mask.
ROI_X = IP_CONFIG["texture"]["roi_x"]
ROI_Y = IP_CONFIG["texture"]["roi_y"]
ROI_W = IP_CONFIG["texture"]["roi_width"]
ROI_H = IP_CONFIG["texture"]["roi_height"]

PATCH_FRAC_MIN = IP_CONFIG["texture"]["patch_min"]
PATCH_FRAC_MAX = IP_CONFIG["texture"]["patch_max"]

ROTATION_MIN = IP_CONFIG["texture"]["rotation_min"]
ROTATION_MAX = IP_CONFIG["texture"]["rotation_max"]

SCALE_MIN = IP_CONFIG["texture"]["scale_min"]
SCALE_MAX = IP_CONFIG["texture"]["scale_max"]

FEATHER_FRACTION = IP_CONFIG["texture"]["feather_fraction"]

# Number of donor candidates considered for compatibility.
N_DONOR_CANDIDATES = IP_CONFIG["texture"]["n_donor_candidates"]

# Randomly choose among the best few candidates.
TOP_DONORS = IP_CONFIG["texture"]["top_donors"]

# Minimum source/destination separation.
MIN_SEPARATION_MULTIPLIER = IP_CONFIG["texture"]["min_separation_multiplier"]


# ============================================================
# 3. REPRODUCIBLE RANDOM NUMBER GENERATOR
# ============================================================

master_rng = np.random.default_rng(MASTER_SEED)


# ============================================================
# 4. LOAD LOCKED SPLIT
# ============================================================

split_df = pd.read_csv(SPLIT_FILE)

print("Loaded split:")
print(split_df["split"].value_counts())
print()


# ------------------------------------------------------------
# Basic column check
# ------------------------------------------------------------

required_columns = [
    "subject_id",
    "study_id",
    "dicom_id",
    "disease_status",
    "split",
]

missing = [c for c in required_columns if c not in split_df.columns]

if missing:
    raise ValueError(
        f"Missing required columns in split file: {missing}\n"
        f"Available columns: {list(split_df.columns)}"
    )


# ============================================================
# 5. IMAGE PATH FUNCTION
# ============================================================

def get_image_path(row):
    """Construct the source image path from the locked manifest."""
    disease = str(row["disease_status"]).strip().lower()
    # The locked real dataset uses:
    #     p{subject_id}_s{study_id}_{dicom_id}.jpg
    subject_id = str(row["subject_id"]).strip()
    study_id = str(row["study_id"]).strip()
    dicom_id = str(row["dicom_id"]).strip()

    filename = (
        f"p{subject_id}_"
        f"s{study_id}_"
        f"{dicom_id}.jpg"
    )

    return (
        REAL_DIR
        / str(row["split"])
        / disease
        / filename
    )


# ============================================================
# 6. IMAGE LOADING
# ============================================================

def load_grayscale(path):
    """
    Load an image as a 2-D grayscale float32 array.

    Everything is converted to the common working canvas (256 x 256)
    with the frozen Lanczos filter (gen_common).
    """

    return gen_common.load_gray_array(path, WORK_SIZE)


# ============================================================
# 7. FIXED SAFE ROI
# ============================================================

def inside_safe_roi(x, y, w, h, roi_x, roi_y, roi_w, roi_h):
    """
    Check whether a rectangle lies completely inside the
    conservative placement ROI.
    """

    return (
        x >= roi_x
        and y >= roi_y
        and x + w <= roi_x + roi_w
        and y + h <= roi_y + roi_h
    )


# ============================================================
# 8. SAMPLE A LOCATION INSIDE THE SAFE ROI
# ============================================================

def sample_patch_location(rng, patch_w, patch_h):
    """
    Randomly sample a patch location inside the fixed ROI.

    The location is radius/patch-size aware so that the entire
    patch stays inside the allowed spatial region.
    """

    min_x = ROI_X
    max_x = ROI_X + ROI_W - patch_w

    min_y = ROI_Y
    max_y = ROI_Y + ROI_H - patch_h

    if max_x < min_x or max_y < min_y:
        raise ValueError("Patch is larger than the safe ROI.")

    x = int(rng.integers(min_x, max_x + 1))
    y = int(rng.integers(min_y, max_y + 1))

    return x, y


# ============================================================
# 9. EXTRACT PATCH
# ============================================================

def extract_patch(image, x, y, w, h):
    """
    Extract a rectangular donor patch.
    """

    patch = image[y:y+h, x:x+w].copy()

    if patch.shape != (h, w):
        raise ValueError(
            f"Patch extraction failed. "
            f"Expected {(h, w)}, got {patch.shape}"
        )

    return patch


# ============================================================
# 10. ROTATE + SCALE DONOR PATCH
# ============================================================

def transform_patch(patch, angle, scale):
    """
    Apply isotropic scaling and rotation.

    The transformed patch is then center-cropped back to the
    original patch dimensions.
    """

    h, w = patch.shape

    # --------------------------------------------------------
    # Scale first
    # --------------------------------------------------------

    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    scaled = cv2.resize(
        patch,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Rotate around centre
    # --------------------------------------------------------

    center = (new_w / 2.0, new_h / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated = cv2.warpAffine(
        scaled,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # --------------------------------------------------------
    # Centre crop/pad to original dimensions
    # --------------------------------------------------------

    output = np.zeros((h, w), dtype=np.float32)

    crop_h = min(h, rotated.shape[0])
    crop_w = min(w, rotated.shape[1])

    src_y = max(0, (rotated.shape[0] - crop_h) // 2)
    src_x = max(0, (rotated.shape[1] - crop_w) // 2)

    dst_y = max(0, (h - crop_h) // 2)
    dst_x = max(0, (w - crop_w) // 2)

    output[
        dst_y:dst_y+crop_h,
        dst_x:dst_x+crop_w
    ] = rotated[
        src_y:src_y+crop_h,
        src_x:src_x+crop_w
    ]

    return output


# ============================================================
# 11. CONTRAST MATCHING
# ============================================================

def match_contrast(source_patch, target_region):
    """
    Match donor patch mean and standard deviation to the
    target region.

    This reduces obvious donor/target exposure differences.
    """

    source_mean = float(np.mean(source_patch))
    source_std = float(np.std(source_patch))

    target_mean = float(np.mean(target_region))
    target_std = float(np.std(target_region))

    # Avoid division by zero in very uniform regions.
    if source_std < 1e-6:
        matched = source_patch - source_mean + target_mean
    else:
        matched = (
            (source_patch - source_mean)
            / source_std
            * max(target_std, 1.0)
            + target_mean
        )

    return np.clip(matched, 0, 255).astype(np.float32)


# ============================================================
# 12. LOCAL NOISE MATCHING
# ============================================================

def local_noise_match(source_patch, target_region):
    """
    Match the high-frequency residual magnitude of the donor
    patch to the local target region.

    A Gaussian-smoothed image is used to estimate the local
    low-frequency component.
    """

    sigma = 1.5

    source_smooth = cv2.GaussianBlur(
        source_patch,
        (0, 0),
        sigma,
    )

    target_smooth = cv2.GaussianBlur(
        target_region,
        (0, 0),
        sigma,
    )

    source_noise = source_patch - source_smooth
    target_noise = target_region - target_smooth

    source_std = float(np.std(source_noise))
    target_std = float(np.std(target_noise))

    if source_std < 1e-6:
        return source_patch

    scale = target_std / source_std

    # Keep the adjustment conservative.
    scale = np.clip(scale, 0.5, 2.0)

    adjusted = (
        source_smooth
        + source_noise * scale
    )

    return np.clip(adjusted, 0, 255).astype(np.float32)


# ============================================================
# 13. DONOR/TARGET COMPATIBILITY SCORE
# ============================================================

def compatibility_score(donor_patch, target_region):
    """
    Measure how similar the donor patch is to the target
    region before blending.

    Lower score = more photometrically compatible.
    """

    donor_mean = float(np.mean(donor_patch))
    target_mean = float(np.mean(target_region))

    donor_std = float(np.std(donor_patch))
    target_std = float(np.std(target_region))

    mean_difference = abs(donor_mean - target_mean)
    std_difference = abs(donor_std - target_std)

    return mean_difference + std_difference


# ============================================================
# 14. GAUSSIAN FEATHER MASK
# ============================================================

def make_feather_mask(h, w, feather_fraction):
    """
    Create a smooth rectangular blending mask.

    The centre has weight ~1 and the boundary smoothly
    approaches 0.
    """

    yy, xx = np.mgrid[0:h, 0:w]

    # Distance to nearest edge.
    distance_to_edge = np.minimum.reduce([
        xx,
        yy,
        w - 1 - xx,
        h - 1 - yy,
    ]).astype(np.float32)

    feather_width = max(
        1.0,
        feather_fraction * min(h, w)
    )

    # Smooth transition.
    mask = np.clip(
        distance_to_edge / feather_width,
        0,
        1,
    )

    # Smoothstep.
    mask = mask * mask * (3.0 - 2.0 * mask)

    return mask.astype(np.float32)


# ============================================================
# 15. BUILD DONOR MASK
# ============================================================

def make_binary_mask(h, w):
    """
    Destination-only binary ground-truth mask.

    1 = transplanted region
    0 = unchanged target
    """

    return np.ones(
        (h, w),
        dtype=np.uint8,
    )


# ============================================================
# 16. MINIMUM DISTANCE CHECK
# ============================================================

def patch_centres_far_enough(
    source_x,
    source_y,
    destination_x,
    destination_y,
    patch_w,
    patch_h,
):
    """
    Check whether source and destination centres are separated
    sufficiently.

    This is mostly relevant conceptually for keeping the two
    sampled regions spatially distinct within the target/donor
    images.
    """

    source_cx = source_x + patch_w / 2.0
    source_cy = source_y + patch_h / 2.0

    destination_cx = destination_x + patch_w / 2.0
    destination_cy = destination_y + patch_h / 2.0

    distance = np.sqrt(
        (source_cx - destination_cx) ** 2
        + (source_cy - destination_cy) ** 2
    )

    minimum_distance = (
        MIN_SEPARATION_MULTIPLIER
        * max(patch_w, patch_h)
    )

    return distance >= minimum_distance


# ============================================================
# 17. FIND COMPATIBLE DONOR LOCATION
# ============================================================

def choose_donor_patch(
    donor_image,
    target_image,
    patch_w,
    patch_h,
    rng,
):
    """
    Randomly sample donor locations from the donor image,
    score their compatibility with a random target location,
    and choose randomly among the best few.

    Importantly, donor and target coordinates are NOT forced
    to correspond anatomically.
    """

    target_x, target_y = sample_patch_location(
        rng,
        patch_w,
        patch_h,
    )

    target_region = extract_patch(
        target_image,
        target_x,
        target_y,
        patch_w,
        patch_h,
    )

    candidates = []

    for _ in range(N_DONOR_CANDIDATES):

        donor_x, donor_y = sample_patch_location(
            rng,
            patch_w,
            patch_h,
        )

        donor_patch = extract_patch(
            donor_image,
            donor_x,
            donor_y,
            patch_w,
            patch_h,
        )

        score = compatibility_score(
            donor_patch,
            target_region,
        )

        candidates.append(
            (
                score,
                donor_x,
                donor_y,
                target_x,
                target_y,
            )
        )

    candidates.sort(key=lambda item: item[0])

    # Randomly select from the best few rather than always
    # choosing the single mathematically closest patch.
    n_top = min(TOP_DONORS, len(candidates))

    selected = candidates[
        int(rng.integers(0, n_top))
    ]

    (
        score,
        donor_x,
        donor_y,
        target_x,
        target_y,
    ) = selected

    return (
        donor_x,
        donor_y,
        target_x,
        target_y,
        score,
    )


# ============================================================
# 18. TEXTURE TRANSPLANT FUNCTION
# ============================================================

def texture_transplant(
    target_image,
    donor_image,
    rng,
):
    """
    Perform one complete Texture Transplant.

    Returns:
        synthetic_image
        binary_GT_mask
        metadata dictionary
    """

    # --------------------------------------------------------
    # Sample locked parameters
    # --------------------------------------------------------

    patch_fraction = float(
        rng.uniform(
            PATCH_FRAC_MIN,
            PATCH_FRAC_MAX,
        )
    )

    rotation = float(
        rng.uniform(
            ROTATION_MIN,
            ROTATION_MAX,
        )
    )

    scale = float(
        rng.uniform(
            SCALE_MIN,
            SCALE_MAX,
        )
    )

    # Patch size is relative to the fixed ROI width.
    patch_size = max(
        8,
        int(round(patch_fraction * ROI_W))
    )

    patch_w = patch_size
    patch_h = patch_size

    # --------------------------------------------------------
    # Select donor and target locations independently
    # --------------------------------------------------------

    (
        donor_x,
        donor_y,
        target_x,
        target_y,
        compatibility,
    ) = choose_donor_patch(
        donor_image,
        target_image,
        patch_w,
        patch_h,
        rng,
    )

    # --------------------------------------------------------
    # Extract donor patch
    # --------------------------------------------------------

    donor_patch = extract_patch(
        donor_image,
        donor_x,
        donor_y,
        patch_w,
        patch_h,
    )

    # --------------------------------------------------------
    # Geometric transformation
    # --------------------------------------------------------

    transformed_patch = transform_patch(
        donor_patch,
        rotation,
        scale,
    )

    # --------------------------------------------------------
    # Target region
    # --------------------------------------------------------

    target_region = extract_patch(
        target_image,
        target_x,
        target_y,
        patch_w,
        patch_h,
    )

    # --------------------------------------------------------
    # Contrast matching
    # --------------------------------------------------------

    matched_patch = match_contrast(
        transformed_patch,
        target_region,
    )

    # --------------------------------------------------------
    # Local noise matching
    # --------------------------------------------------------

    matched_patch = local_noise_match(
        matched_patch,
        target_region,
    )

    # --------------------------------------------------------
    # Feathered blending
    # --------------------------------------------------------

    feather_mask = make_feather_mask(
        patch_h,
        patch_w,
        FEATHER_FRACTION,
    )

    blended_patch = (
        feather_mask * matched_patch
        + (1.0 - feather_mask) * target_region
    )

    # --------------------------------------------------------
    # Put blended patch into target image
    # --------------------------------------------------------

    synthetic = target_image.copy()

    synthetic[
        target_y:target_y+patch_h,
        target_x:target_x+patch_w
    ] = blended_patch

    # FREEZE FIX: round to nearest grey level (astype alone truncates,
    # darkening edited pixels by ~0.5 on average).
    synthetic = np.rint(np.clip(
        synthetic,
        0,
        255,
    )).astype(np.uint8)

    # --------------------------------------------------------
    # Destination-only binary GT mask
    # --------------------------------------------------------

    gt_mask = np.zeros(
        (WORK_SIZE, WORK_SIZE),
        dtype=np.uint8,
    )

    gt_mask[
        target_y:target_y+patch_h,
        target_x:target_x+patch_w
    ] = 1

    metadata = {
        "patch_fraction": patch_fraction,
        "patch_pixels": patch_size,
        "rotation_deg": rotation,
        "scale": scale,
        "feather_fraction": FEATHER_FRACTION,
        "donor_x": donor_x,
        "donor_y": donor_y,
        "target_x": target_x,
        "target_y": target_y,
        "donor_target_compatibility": compatibility,
    }

    return synthetic, gt_mask, metadata


# ============================================================
# 19. PREPARE SPLIT DATA
# ============================================================

split_data = {
    split_name: split_df[
        split_df["split"] == split_name
    ].copy().reset_index(drop=True)
    for split_name in ["train", "val", "test"]
}


for split_name, df in split_data.items():

    print(
        f"{split_name}: "
        f"{len(df)} images, "
        f"{df['subject_id'].nunique()} patients"
    )

print()


# ============================================================
# 20. GENERATE ONE SYNTHETIC IMAGE PER REAL TARGET
# ============================================================

all_metadata = []
all_errors = []

global_counter = 0


for split_name in ["train", "val", "test"]:

    split_rows = split_data[split_name]

    # --------------------------------------------------------
    # Build donor pool for this split.
    # --------------------------------------------------------

    donor_pool = split_rows.to_dict("records")

    print(
        f"Generating Texture Transplant images for "
        f"{split_name}: {len(split_rows)}"
    )

    for row_index, target_row in split_rows.iterrows():

        global_counter += 1

        # Deterministic seed for this target image.
        target_seed = (
            MASTER_SEED
            + global_counter * 1009
        )

        rng = np.random.default_rng(target_seed)

        try:

            # ------------------------------------------------
            # Target information
            # ------------------------------------------------

            target_subject = str(
                target_row["subject_id"]
            )

            target_study = str(
                target_row["study_id"]
            )

            target_dicom = str(
                target_row["dicom_id"]
            )

            target_disease = str(
                target_row["disease_status"]
            )

            target_path = get_image_path(
                target_row
            )

            # ------------------------------------------------
            # Choose donor from same split but different
            # patient.
            # ------------------------------------------------

            valid_donors = [
                donor
                for donor in donor_pool
                if str(donor["subject_id"]) != target_subject
            ]

            if len(valid_donors) == 0:
                raise ValueError(
                    "No valid donor from a different patient "
                    f"for target {target_subject}"
                )

            donor_index = int(
                rng.integers(
                    0,
                    len(valid_donors),
                )
            )

            donor_row = valid_donors[donor_index]

            donor_subject = str(
                donor_row["subject_id"]
            )

            donor_study = str(
                donor_row["study_id"]
            )

            donor_dicom = str(
                donor_row["dicom_id"]
            )

            donor_disease = str(
                donor_row["disease_status"]
            )

            donor_path = get_image_path(
                donor_row
            )

            # ------------------------------------------------
            # Load images
            # ------------------------------------------------

            target_image = load_grayscale(
                target_path
            )

            donor_image = load_grayscale(
                donor_path
            )

            # ------------------------------------------------
            # Generate transplant
            # ------------------------------------------------

            synthetic_image, gt_mask, gen_meta = (
                texture_transplant(
                    target_image,
                    donor_image,
                    rng,
                )
            )

            # ------------------------------------------------
            # Synthetic ID
            # ------------------------------------------------

            # Preserve the exact target/source basename:
            #     p{subject_id}_s{study_id}_{dicom_id}.jpg
            synthetic_id = (
                f"p{target_subject}_"
                f"s{target_study}_"
                f"{target_dicom}"
            )

            image_path = (
                IMAGE_DIR
                / split_name
                / f"{synthetic_id}{gen_common.IMAGE_EXT}"
            )

            mask_path = (
                MASK_DIR
                / split_name
                / f"{synthetic_id}_mask.png"
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            # Lossless PNG with a read-back pixel check.
            gen_common.save_gray_png(
                synthetic_image,
                image_path,
            )

            # ------------------------------------------------
            # Save binary mask as 0/255 PNG.
            #
            # Internally the mask is 0/1.
            # On disk we save 0/255 for easy visualisation.
            # ------------------------------------------------

            success_mask = cv2.imwrite(
                str(mask_path),
                gt_mask * 255,
            )

            if not success_mask:
                raise IOError(
                    f"Could not save mask: {mask_path}"
                )

            # ------------------------------------------------
            # Edit strength (0-255 scale) for the metadata.
            # ------------------------------------------------

            strength = gen_common.edit_strength(
                target_image,
                synthetic_image,
                gt_mask,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata_row = {
                "synthetic_id": synthetic_id,

                "split": split_name,

                # Target
                "target_subject_id": target_subject,
                "target_study_id": target_study,
                "target_dicom_id": target_dicom,
                "target_disease": target_disease,

                # Donor
                "donor_subject_id": donor_subject,
                "donor_study_id": donor_study,
                "donor_dicom_id": donor_dicom,
                "donor_disease": donor_disease,

                # Paths
                "target_image_path": str(target_path),
                "donor_image_path": str(donor_path),
                "synthetic_image_path": str(image_path),
                "mask_path": str(mask_path),

                # Reproducibility
                "seed": target_seed,

                # Protocol and edit strength
                "generator_version": gen_common.GENERATOR_VERSION,
                "resize_filter": gen_common.RESIZE_FILTER_NAME,
                "output_format": gen_common.IMAGE_FORMAT,
                **strength,

                # Generator parameters
                **gen_meta,
            }

            all_metadata.append(
                metadata_row
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (row_index + 1) % 25 == 0
                or row_index == 0
                or row_index == len(split_rows) - 1
            ):
                print(
                    f"  {row_index + 1:3d}/"
                    f"{len(split_rows)} completed"
                )

        except Exception as exc:

            error_row = {
                "global_index": global_counter,
                "split": split_name,
                "target_subject_id": str(
                    target_row["subject_id"]
                ),
                "target_study_id": str(
                    target_row["study_id"]
                ),
                "target_dicom_id": str(
                    target_row["dicom_id"]
                ),
                "error": repr(exc),
            }

            all_errors.append(error_row)

            print(
                f"  ERROR: "
                f"{target_row['dicom_id']} -> {repr(exc)}"
            )


# ============================================================
# 21. SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    all_metadata
)

metadata_file = (
    REPORT_DIR
    / "texture_transplant_final_metadata.csv"
)

metadata_df.to_csv(
    metadata_file,
    index=False,
)

# Output folders must hold exactly the images written by this run.
_written = list(metadata_df["synthetic_image_path"]) if len(metadata_df) else []
for _split in ("train", "val", "test"):
    gen_common.assert_output_matches(IMAGE_DIR / _split, _written, "texture")


# ============================================================
# 22. SAVE ERROR LOG
# ============================================================

error_df = pd.DataFrame(
    all_errors
)

error_file = (
    REPORT_DIR
    / "texture_transplant_final_errors.csv"
)

error_df.to_csv(
    error_file,
    index=False,
)


# ============================================================
# 23. SAVE FINAL CONFIGURATION
# ============================================================

production_config = {   # renamed from 'config' (it clobbered the module)
    "generator": "Texture Transplant",
    "version": "final_production",

    "master_seed": MASTER_SEED,

    "resize_filter": gen_common.RESIZE_FILTER_NAME,
    "output_format": gen_common.IMAGE_FORMAT,

    "working_size": [
        WORK_SIZE,
        WORK_SIZE,
    ],

    "safe_roi": {
        "x": ROI_X,
        "y": ROI_Y,
        "width": ROI_W,
        "height": ROI_H,
        "description": (
            "Conservative spatial placement region; "
            "not a lung segmentation mask."
        ),
    },

    "patch_fraction": {
        "distribution": "uniform",
        "min": PATCH_FRAC_MIN,
        "max": PATCH_FRAC_MAX,
    },

    "rotation_degrees": {
        "distribution": "uniform",
        "min": ROTATION_MIN,
        "max": ROTATION_MAX,
    },

    "scale": {
        "distribution": "uniform",
        "min": SCALE_MIN,
        "max": SCALE_MAX,
    },

    "feather_fraction": FEATHER_FRACTION,

    "donor_sampling": {
        "same_split_as_target": True,
        "different_patient_required": True,
        "independent_location": True,
        "candidate_count": N_DONOR_CANDIDATES,
        "random_choice_from_top": TOP_DONORS,
    },

    "contrast_matching": True,
    "local_noise_matching": True,

    "compositing": (
        "Feathered blending; cv2.seamlessClone not used."
    ),

    "ground_truth_mask": (
        "Destination-only binary mask."
    ),

    "label_policy": (
        "Synthetic image inherits disease label of target."
    ),

    "expected_counts": {
        **__import__("config").SPLIT_IMAGE_COUNTS,
        "total": __import__("config").POOL_SIZE_PER_POOL,
    },
}

config_file = (
    CONFIG_DIR
    / "texture_transplant_final_production_config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        production_config,
        f,
        indent=2,
    )


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("TEXTURE TRANSPLANT PRODUCTION COMPLETE")
print("=" * 60)

print(
    f"Generated successfully: {len(metadata_df)}"
)

print(
    f"Errors: {len(error_df)}"
)

if len(metadata_df) > 0:

    print()
    print("Images by split:")
    print(
        metadata_df["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Target disease:")
    print(
        metadata_df["target_disease"]
        .value_counts()
    )

    print()
    print("Donor disease:")
    print(
        metadata_df["donor_disease"]
        .value_counts()
    )

    print()
    print("Direction:")
    metadata_df["direction"] = (
        metadata_df["donor_disease"]
        + "_to_"
        + metadata_df["target_disease"]
    )

    print(
        metadata_df["direction"]
        .value_counts()
    )

    print()
    print("Parameter ranges actually generated:")

    print(
        "Patch fraction:",
        round(
            metadata_df["patch_fraction"].min(),
            4,
        ),
        "to",
        round(
            metadata_df["patch_fraction"].max(),
            4,
        ),
    )

    print(
        "Rotation:",
        round(
            metadata_df["rotation_deg"].min(),
            4,
        ),
        "to",
        round(
            metadata_df["rotation_deg"].max(),
            4,
        ),
    )

    print(
        "Scale:",
        round(
            metadata_df["scale"].min(),
            4,
        ),
        "to",
        round(
            metadata_df["scale"].max(),
            4,
        ),
    )

    print(
        "Feather:",
        metadata_df["feather_fraction"].unique()
    )

print()
print("Metadata:")
print(metadata_file)

print()
print("Errors:")
print(error_file)

print()
print("Config:")
print(config_file)

print()
print("Output root:")
print(OUTPUT_ROOT)

print()
print("=" * 60)