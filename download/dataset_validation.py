"""
Validation of the completed real-image dataset.
Validates (counts follow POOL_SIZE_PER_POOL and the configured fractions):
    - total number of images
    - normal / pathology balance
    - train / val / test image counts
    - no patient leakage between train/val/test
    - every manifest image exists
    - no unexpected JPEG files
    - JPEG integrity using Pillow
    - manifest consistency
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from config import (
    MANIFEST_DIR,
    REAL_DIR,
    POOL_SIZE_PER_POOL,
    NORMAL_FRACTION,
    PATHOLOGY_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
    VALIDATION_REPORT,
    VALIDATE_IMAGE_CONTENT,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

POOLS = {
    "real": {
        "root": REAL_DIR,
        "manifest": MANIFEST_DIR / "real_all.csv",
    },
}

SPLITS = ("train", "val", "test")
CLASSES = ("normal", "pathology")


# =============================================================================
# EXPECTED COUNTS
# =============================================================================

def expected_split_counts(pool_size: int) -> dict[str, int]:
    """Calculate expected train/val/test image counts."""

    counts = {
        "train": int(round(pool_size * TRAIN_FRACTION)),
        "val": int(round(pool_size * VAL_FRACTION)),
        "test": int(round(pool_size * TEST_FRACTION)),
    }

    if sum(counts.values()) != pool_size:
        raise ValueError(
            f"Split counts {counts} do not sum to pool size "
            f"{pool_size}."
        )

    return counts


def expected_class_counts(pool_size: int) -> dict[str, int]:
    """Calculate expected complete-pool class counts."""

    counts = {
        "normal": int(round(pool_size * NORMAL_FRACTION)),
        "pathology": int(round(pool_size * PATHOLOGY_FRACTION)),
    }

    if sum(counts.values()) != pool_size:
        raise ValueError(
            f"Class counts {counts} do not sum to pool size "
            f"{pool_size}."
        )

    return counts


# =============================================================================
# FAILURE HELPER
# =============================================================================

def fail(message: str) -> None:
    raise AssertionError(message)


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def validate_manifest_structure(
    pool_name: str,
    manifest_path: Path,
) -> pd.DataFrame:
    """
    Load and validate the actual manifest schema produced by the downloader.
    """

    print(f"\nChecking manifest: {manifest_path}")

    if not manifest_path.exists():
        fail(
            f"{pool_name}: manifest does not exist:\n"
            f"{manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    required_columns = {
        "dicom_id",
        "subject_id",
        "study_id",
        "ViewPosition",
        "disease_status",
        "patient_category",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            f"{pool_name}: manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        fail(f"{pool_name}: manifest is empty.")

    # -------------------------------------------------------------------------
    # Check class values.
    # -------------------------------------------------------------------------

    valid_classes = set(CLASSES)

    observed_classes = set(
        df["disease_status"]
        .dropna()
        .astype(str)
    )

    invalid_classes = observed_classes - valid_classes

    if invalid_classes:
        fail(
            f"{pool_name}: invalid disease_status values: "
            f"{sorted(invalid_classes)}"
        )

    # Missing disease_status values are not allowed in selected images.
    if df["disease_status"].isna().any():
        count = int(df["disease_status"].isna().sum())

        fail(
            f"{pool_name}: {count} selected images have missing "
            f"disease_status."
        )

    # -------------------------------------------------------------------------
    # Check split values.
    # -------------------------------------------------------------------------

    observed_splits = set(
        df["split"]
        .dropna()
        .astype(str)
    )

    invalid_splits = observed_splits - set(SPLITS)

    if invalid_splits:
        fail(
            f"{pool_name}: invalid split values: "
            f"{sorted(invalid_splits)}"
        )

    if df["split"].isna().any():
        count = int(df["split"].isna().sum())

        fail(
            f"{pool_name}: {count} selected images have missing split."
        )

    # -------------------------------------------------------------------------
    # Check duplicate images.
    # -------------------------------------------------------------------------

    image_key = [
        "subject_id",
        "study_id",
        "dicom_id",
    ]

    duplicate_mask = df.duplicated(
        subset=image_key,
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            image_key,
        ]

        fail(
            f"{pool_name}: duplicate image entries found.\n"
            f"{duplicates.head(10).to_string(index=False)}"
        )

    return df


# =============================================================================
# FILE PATH
# =============================================================================

def expected_image_path(
    root: Path,
    row: pd.Series,
) -> Path:
    """
    Reconstruct the physical JPEG path used by the downloader.
    """

    filename = (
        f"p{row['subject_id']}_"
        f"s{row['study_id']}_"
        f"{row['dicom_id']}.jpg"
    )

    return (
        root
        / str(row["split"])
        / str(row["disease_status"])
        / filename
    )


# =============================================================================
# SINGLE POOL VALIDATION
# =============================================================================

def validate_pool(
    pool_name: str,
    root: Path,
    manifest_path: Path,
    expected_splits: dict[str, int],
    expected_classes: dict[str, int],
) -> set[str]:

    print("\n" + "-" * 78)
    print(f"VALIDATING POOL: {pool_name}")
    print("-" * 78)

    df = validate_manifest_structure(
        pool_name,
        manifest_path,
    )

    # =========================================================================
    # TOTAL IMAGE COUNT
    # =========================================================================

    total_images = len(df)

    print(f"Total manifest images: {total_images}")

    if total_images != POOL_SIZE_PER_POOL:
        fail(
            f"{pool_name}: expected "
            f"{POOL_SIZE_PER_POOL} images, found "
            f"{total_images}."
        )

    # =========================================================================
    # COMPLETE-POOL CLASS BALANCE
    # =========================================================================

    class_counts = (
        df["disease_status"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_dict()
    )

    print(f"Class counts: {class_counts}")

    if class_counts != expected_classes:
        fail(
            f"{pool_name}: class balance mismatch.\n"
            f"Expected: {expected_classes}\n"
            f"Found:    {class_counts}"
        )

    # =========================================================================
    # SPLIT IMAGE COUNTS
    # =========================================================================

    split_counts = (
        df["split"]
        .value_counts()
        .reindex(SPLITS, fill_value=0)
        .to_dict()
    )

    print(f"Split counts: {split_counts}")

    if split_counts != expected_splits:
        fail(
            f"{pool_name}: split count mismatch.\n"
            f"Expected: {expected_splits}\n"
            f"Found:    {split_counts}"
        )

    # =========================================================================
    # PATIENT-LEVEL SPLIT ISOLATION
    # =========================================================================

    patient_sets = {
        split: set(
            df.loc[
                df["split"] == split,
                "subject_id",
            ].astype(str)
        )
        for split in SPLITS
    }

    for i, first_split in enumerate(SPLITS):

        for second_split in SPLITS[i + 1:]:

            overlap = (
                patient_sets[first_split]
                & patient_sets[second_split]
            )

            if overlap:
                fail(
                    f"{pool_name}: patient leakage between "
                    f"{first_split} and {second_split}.\n"
                    f"Patients: {sorted(overlap)[:20]}"
                )

    print("Patient leakage within pool: 0")

    # =========================================================================
    # PATIENT CATEGORY CONSISTENCY
    # =========================================================================

    # A pathology patient may have both normal and pathology-labelled images.
    # Therefore we do NOT require disease_status == patient_category.
    #
    # We only check that patient_category itself contains valid values.

    valid_patient_categories = {"normal", "pathology"}

    observed_patient_categories = set(
        df["patient_category"]
        .dropna()
        .astype(str)
    )

    invalid_categories = (
        observed_patient_categories
        - valid_patient_categories
    )

    if invalid_categories:
        fail(
            f"{pool_name}: invalid patient_category values: "
            f"{sorted(invalid_categories)}"
        )

    print("Patient-category values: PASSED")

    # =========================================================================
    # MANIFEST -> FILE CHECK
    # =========================================================================

    expected_files = set()
    missing_files = []
    invalid_files = []

    for _, row in df.iterrows():

        path = expected_image_path(
            root,
            row,
        )

        expected_files.add(path.resolve())

        if not path.exists():

            missing_files.append(path)
            continue

        if path.stat().st_size == 0:

            invalid_files.append(path)
            continue

        # ---------------------------------------------------------------------
        # JPEG integrity.
        # ---------------------------------------------------------------------

        if VALIDATE_IMAGE_CONTENT:

            try:

                with Image.open(path) as image:
                    image.verify()

                with Image.open(path) as image:

                    if image.width <= 0 or image.height <= 0:
                        invalid_files.append(path)

            except Exception:
                invalid_files.append(path)

    if missing_files:

        fail(
            f"{pool_name}: {len(missing_files)} manifest images "
            f"are missing.\n"
            f"Examples:\n"
            f"{missing_files[:5]}"
        )

    if invalid_files:

        fail(
            f"{pool_name}: {len(invalid_files)} invalid/corrupt "
            f"image files found.\n"
            f"Examples:\n"
            f"{invalid_files[:5]}"
        )

    print("Manifest-to-file check: PASSED")

    if VALIDATE_IMAGE_CONTENT:
        print("JPEG/Pillow integrity check: PASSED")

    # =========================================================================
    # CHECK FOR UNEXPECTED JPEG FILES
    # =========================================================================

    actual_files = {
        path.resolve()
        for path in root.glob("**/*.jpg")
        if path.is_file()
    }

    unexpected_files = actual_files - expected_files

    if unexpected_files:

        fail(
            f"{pool_name}: {len(unexpected_files)} unexpected "
            f"JPEG files found.\n"
            f"Examples:\n"
            f"{list(unexpected_files)[:5]}"
        )

    print("No unexpected JPEG files: PASSED")

    # =========================================================================
    # RETURN PATIENT IDS (for the report)
    # =========================================================================

    return set(
        df["subject_id"].astype(str)
    )


# =============================================================================
# MAIN
# =============================================================================

def validate_dataset() -> dict:

    expected_splits = expected_split_counts(
        POOL_SIZE_PER_POOL
    )

    expected_classes = expected_class_counts(
        POOL_SIZE_PER_POOL
    )

    print("=" * 78)
    print("DATASET VALIDATION")
    print("=" * 78)

    print(f"Pool size: {POOL_SIZE_PER_POOL}")
    print(
        f"Expected class counts: "
        f"{expected_classes}"
    )

    print(
        f"Expected split counts: "
        f"{expected_splits}"
    )

    patient_sets = {}

    report_lines = [
        "=" * 78,
        "MIMIC-CXR DATASET VALIDATION REPORT",
        "=" * 78,
        "",
        f"Pool size: {POOL_SIZE_PER_POOL}",
        f"Expected class counts: {expected_classes}",
        f"Expected split counts: {expected_splits}",
        "",
    ]

    try:

        # =====================================================================
        # VALIDATE THE REAL POOL
        # =====================================================================

        for pool_name, settings in POOLS.items():

            patients = validate_pool(
                pool_name=pool_name,
                root=settings["root"],
                manifest_path=settings["manifest"],
                expected_splits=expected_splits,
                expected_classes=expected_classes,
            )

            patient_sets[pool_name] = patients

            report_lines.extend(
                [
                    f"POOL: {pool_name}",
                    f"Root: {settings['root']}",
                    f"Manifest: {settings['manifest']}",
                    f"Patients: {len(patients)}",
                    f"Images: {POOL_SIZE_PER_POOL}",
                    f"Class counts: {expected_classes}",
                    f"Split counts: {expected_splits}",
                    "Status: PASSED",
                    "",
                ]
            )

        report_lines.extend(
            [
                "=" * 78,
                "OVERALL RESULT: PASSED",
                "=" * 78,
            ]
        )

        # =====================================================================
        # SAVE REPORT
        # =====================================================================

        VALIDATION_REPORT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        VALIDATION_REPORT.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("SUCCESS — DATASET VALIDATION PASSED")
        print("=" * 78)
        print(
            f"Validation report: "
            f"{VALIDATION_REPORT}"
        )
        print("=" * 78)

        return {
            "status": "PASSED",
            "report": VALIDATION_REPORT,
            "pool_size": POOL_SIZE_PER_POOL,
            "class_counts": expected_classes,
            "split_counts": expected_splits,
        }

    except Exception as exc:

        report_lines.extend(
            [
                "",
                "=" * 78,
                "OVERALL RESULT: FAILED",
                "=" * 78,
                str(exc),
            ]
        )

        VALIDATION_REPORT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        VALIDATION_REPORT.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("DATASET VALIDATION FAILED")
        print("=" * 78)
        print(str(exc))
        print(
            f"Validation report: "
            f"{VALIDATION_REPORT}"
        )
        print("=" * 78)

        raise


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":
    validate_dataset()