from __future__ import annotations

"""
===============================================================================
MIMIC-CXR-JPG REAL-IMAGE DATASET BUILDER (single real pool)
===============================================================================
Download rules:
    * PA/AP images only.
    * Normal/pathology classes are derived from CheXpert labels.
    * MULTIPLE images per patient are allowed.
    * A patient may contribute several images.
    * A patient is assigned to exactly ONE train/validation/test split.
    * Therefore, patient leakage across splits is prohibited.
    * Each pool contains an exact 50/50 image-level class balance.
    * Train/validation/test are dynamically calculated as:
          60% train
          20% validation
          20% test
"""

# =============================================================================
# IMPORTS
# =============================================================================

import getpass
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    MANIFEST_DIR,
    REAL_DIR,
    BASE_URL,
    PHYSIONET_USERNAME,
    PHYSIONET_PASSWORD,
    METADATA_FILE,
    CHEXPERT_FILE,
    RANDOM_SEED,
    POOL_SIZE_PER_POOL,
    NORMAL_FRACTION,
    PATHOLOGY_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
    CLASS_BALANCE,
    ALLOWED_VIEWS,
    PATHOLOGY_COLUMNS,
    NO_FINDING_COLUMN,
    ONE_IMAGE_PER_PATIENT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    CHUNK_SIZE,
    REFRESH_CATALOGUE,
    VERIFY_SSL,
)


# =============================================================================
# GENERAL CONSTANTS
# =============================================================================

# These are the only split names used by the project.
SPLITS = ("train", "val", "test")

# These are the only two class labels used by the project.
LABELS = ("normal", "pathology")


# Suppress the warning generated because the original PhysioNet workflow
# disables SSL certificate verification.
warnings.filterwarnings(
    "ignore",
    message="Unverified HTTPS request",
)


# =============================================================================
# SIMPLE LOGGING HELPER
# =============================================================================

def log(message: str = "") -> None:
    """
    Print a simple progress message.

    Keeping this in one function makes the console output easy to follow
    while the dataset is being constructed.
    """
    print(message)


# =============================================================================
# PHYSIONET CREDENTIALS
# =============================================================================

def get_credentials() -> tuple[str, str]:
    """
    Obtain PhysioNet credentials.

    If username/password are left blank in config.py, the user is prompted.

    The password is requested with getpass so that it is not displayed
    directly in the notebook output.
    """

    username = PHYSIONET_USERNAME.strip()
    password = PHYSIONET_PASSWORD

    if not username:
        username = input("PhysioNet username: ").strip()

    if not password:
        password = getpass.getpass(
            "PhysioNet password: "
        )

    if not username or not password:
        raise RuntimeError(
            "PhysioNet username/password cannot be empty."
        )

    return username, password


# =============================================================================
# PHYSIONET AUTHENTICATION
# =============================================================================

def create_physionet_session(
    username: str,
    password: str,
) -> requests.Session:
    """
    Log into PhysioNet and return an authenticated requests.Session.

    This deliberately follows the browser-style login approach from the
    original working notebook.

    The same session is reused for:

        1. PhysioNet login
        2. metadata download
        3. CheXpert download
        4. individual JPEG downloads

    This is important because the authentication cookies obtained during
    login need to persist across subsequent requests.
    """

    log(
        "\nAuthenticating with PhysioNet using requests.Session()..."
    )

    # -------------------------------------------------------------------------
    # Create persistent HTTP session.
    # -------------------------------------------------------------------------

    session = requests.Session()

    # Match the original working workflow.
    session.verify = VERIFY_SSL

    # Use a browser-like User-Agent.
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
        }
    )

    login_url = "https://physionet.org/login/"

    # -------------------------------------------------------------------------
    # First GET the login page.
    #
    # This allows us to extract the CSRF token used by the PhysioNet login
    # form.
    # -------------------------------------------------------------------------

    login_page = session.get(
        login_url,
        timeout=REQUEST_TIMEOUT,
    )

    login_page.raise_for_status()

    # -------------------------------------------------------------------------
    # Extract CSRF token.
    #
    # Django-style forms can place the name/value attributes in slightly
    # different orders, so two patterns are attempted.
    # -------------------------------------------------------------------------

    csrf_match = re.search(
        r'name=["\']csrfmiddlewaretoken["\'][^>]*'
        r'value=["\']([^"\']+)',
        login_page.text,
        flags=re.IGNORECASE,
    )

    if csrf_match is None:
        csrf_match = re.search(
            r'value=["\']([^"\']+)["\'][^>]*'
            r'name=["\']csrfmiddlewaretoken["\']',
            login_page.text,
            flags=re.IGNORECASE,
        )

    csrf_token = (
        csrf_match.group(1)
        if csrf_match
        else ""
    )

    # -------------------------------------------------------------------------
    # Submit the login form.
    # -------------------------------------------------------------------------

    login_data = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
        "next": "/",
    }

    response = session.post(
        login_url,
        data=login_data,
        headers={"Referer": login_url},
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    # -------------------------------------------------------------------------
    # Verify that the authenticated session can actually access MIMIC-CXR-JPG.
    #
    # Merely checking for the word "logout" in the returned HTML is weaker
    # than making an authenticated request for a credentialed dataset file.
    # -------------------------------------------------------------------------

    probe_url = (
        f"{BASE_URL}/{METADATA_FILE}"
    )

    probe = session.get(
        probe_url,
        stream=True,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )

    try:
        content_type = (
            probe.headers.get("Content-Type") or ""
        ).lower()

        if (
            probe.status_code == 200
            and "text/html" not in content_type
        ):
            log(
                "PhysioNet authentication: PASSED"
            )

            return session

        if probe.status_code in (401, 403):
            raise RuntimeError(
                "PhysioNet login completed, but the credentialed "
                "MIMIC-CXR-JPG request was denied with "
                f"HTTP {probe.status_code}. "
                "Check that the PhysioNet account has accepted the "
                "required data-use agreement and has access to "
                "MIMIC-CXR-JPG."
            )

        raise RuntimeError(
            "PhysioNet session verification did not return the "
            "expected data file. "
            f"HTTP={probe.status_code}, "
            f"Content-Type={content_type!r}."
        )

    finally:
        probe.close()


# =============================================================================
# GENERIC FILE DOWNLOAD
# =============================================================================

def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    force: bool = False,
) -> None:
    """
    Download one file using the authenticated requests.Session.

    A temporary .part file is used so a failed download never leaves a
    partially-written file that could later be mistaken for a complete file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Reuse an existing complete file unless force=True.
    # -------------------------------------------------------------------------

    if (
        not force
        and output_path.exists()
        and output_path.stat().st_size > 0
    ):
        size_mb = (
            output_path.stat().st_size
            / (1024 ** 2)
        )

        log(
            f"Already exists: {output_path} "
            f"({size_mb:.1f} MB)"
        )

        return

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    # -------------------------------------------------------------------------
    # Retry transient failures.
    # -------------------------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:
            log(
                f"\nDownloading: {url}"
            )

            log(
                f"Saving to:  {output_path}"
            )

            with session.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            ) as response:

                # Explicitly handle credential failures.
                if response.status_code in (
                    401,
                    403,
                ):
                    raise RuntimeError(
                        "PhysioNet returned HTTP "
                        f"{response.status_code}."
                    )

                response.raise_for_status()

                content_type = (
                    response.headers.get(
                        "Content-Type"
                    )
                    or ""
                ).lower()

                # If PhysioNet redirects us to a login/access page,
                # that page must never be saved as the catalogue file.
                if "text/html" in content_type:
                    raise RuntimeError(
                        "PhysioNet returned HTML instead of "
                        "the requested data file. "
                        "The session may have been redirected "
                        "to a login/access page."
                    )

                # -------------------------------------------------------------
                # Stream the file to disk.
                #
                # CHUNK_SIZE only controls network transfer. It does NOT
                # resize or recompress images.
                # -------------------------------------------------------------

                with open(
                    tmp_path,
                    "wb",
                ) as output:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if chunk:
                            output.write(chunk)

            # -----------------------------------------------------------------
            # Confirm the temporary file exists and is not empty.
            # -----------------------------------------------------------------

            if (
                not tmp_path.exists()
                or tmp_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            # -----------------------------------------------------------------
            # Atomic replacement.
            #
            # Only a complete file becomes visible under the final filename.
            # -----------------------------------------------------------------

            tmp_path.replace(
                output_path
            )

            size_mb = (
                output_path.stat().st_size
                / (1024 ** 2)
            )

            log(
                f"Completed: {output_path.name} "
                f"({size_mb:.1f} MB)"
            )

            return

        except Exception as exc:

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download {url} after "
                    f"{MAX_RETRIES} attempts: {exc}"
                ) from exc

            log(
                f"Attempt {attempt}/{MAX_RETRIES} "
                f"failed: {exc}"
            )

            time.sleep(
                2 * attempt
            )


# =============================================================================
# DOWNLOAD METADATA / CHEXPERT CATALOGUES
# =============================================================================

def download_catalogues(
    session: requests.Session,
) -> tuple[Path, Path]:
    """
    Download the MIMIC-CXR-JPG metadata and CheXpert catalogue files.

    These are catalogue files only. No image resizing or image manipulation
    occurs here.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        RAW_DIR / METADATA_FILE
    )

    chexpert_path = (
        RAW_DIR / CHEXPERT_FILE
    )

    # -------------------------------------------------------------------------
    # Metadata catalogue.
    # -------------------------------------------------------------------------

    download_file(
        session,
        f"{BASE_URL}/{METADATA_FILE}",
        metadata_path,
        force=REFRESH_CATALOGUE,
    )

    # -------------------------------------------------------------------------
    # CheXpert label catalogue.
    # -------------------------------------------------------------------------

    download_file(
        session,
        f"{BASE_URL}/{CHEXPERT_FILE}",
        chexpert_path,
        force=REFRESH_CATALOGUE,
    )

    return (
        metadata_path,
        chexpert_path,
    )


# =============================================================================
# BUILD IMAGE-LEVEL CANDIDATE TABLE
# =============================================================================

def build_candidate_table(
    metadata_path: Path,
    chexpert_path: Path,
) -> pd.DataFrame:
    """
    Build the eligible IMAGE-LEVEL catalogue.

    IMPORTANT:
        Multiple images per patient are intentionally retained.

    We do NOT call drop_duplicates(subject_id).

    Eligibility:
        * PA/AP view only
        * explicit normal OR pathology label
        * metadata and CheXpert records match on subject_id + study_id
    """

    log(
        "\nLoading metadata and CheXpert labels..."
    )

    # -------------------------------------------------------------------------
    # Load MIMIC metadata.
    # -------------------------------------------------------------------------

    meta = pd.read_csv(
        metadata_path,
        compression="gzip",
    )

    # -------------------------------------------------------------------------
    # Load CheXpert labels.
    # -------------------------------------------------------------------------

    labels = pd.read_csv(
        chexpert_path,
        compression="gzip",
    )

    # -------------------------------------------------------------------------
    # Confirm required metadata columns exist.
    # -------------------------------------------------------------------------

    required_meta = {
        "dicom_id",
        "subject_id",
        "study_id",
        "ViewPosition",
    }

    missing_meta = (
        required_meta
        - set(meta.columns)
    )

    if missing_meta:
        raise RuntimeError(
            "Metadata file is missing required columns: "
            f"{sorted(missing_meta)}"
        )

    # -------------------------------------------------------------------------
    # Confirm required label columns exist.
    # -------------------------------------------------------------------------

    required_labels = {
        "subject_id",
        "study_id",
        NO_FINDING_COLUMN,
    } | set(PATHOLOGY_COLUMNS)

    missing_labels = (
        required_labels
        - set(labels.columns)
    )

    if missing_labels:
        raise RuntimeError(
            "CheXpert file is missing required columns: "
            f"{sorted(missing_labels)}"
        )

    # -------------------------------------------------------------------------
    # Restrict images to PA/AP.
    # -------------------------------------------------------------------------

    meta = meta[
        meta["ViewPosition"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(ALLOWED_VIEWS)
    ].copy()

    # -------------------------------------------------------------------------
    # Convert CheXpert columns to numeric.
    #
    # This makes explicit 1.0 findings detectable even if the CSV contains
    # strings or missing values.
    # -------------------------------------------------------------------------

    for column in (
        PATHOLOGY_COLUMNS
        + [NO_FINDING_COLUMN]
    ):
        labels[column] = pd.to_numeric(
            labels[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Determine image/study-level pathology status.
    #
    # A study is pathology-positive if ANY of the configured pathology
    # columns is explicitly 1.0.
    # -------------------------------------------------------------------------

    labels["pathology_positive"] = (
        labels[PATHOLOGY_COLUMNS]
        .eq(1.0)
        .any(axis=1)
    )

    # -------------------------------------------------------------------------
    # Determine explicit normal status.
    #
    # A study is normal only when:
    #
    #     No Finding == 1
    #
    # AND there is no explicit pathology-positive finding.
    #
    # Pathology therefore takes priority if both are present.
    # -------------------------------------------------------------------------

    labels["normal_positive"] = (
        labels[NO_FINDING_COLUMN].eq(1.0)
        & ~labels["pathology_positive"]
    )

    # -------------------------------------------------------------------------
    # IMPORTANT NUMPY COMPATIBILITY FIX
    #
    # Do NOT use:
    #
    # np.where(..., "normal", np.nan)
    #
    # with modern NumPy because mixing strings and floating-point NaN can
    # trigger:
    #
    #     DTypePromotionError
    #
    # Instead, explicitly create a pandas nullable string column.
    # -------------------------------------------------------------------------

    labels["disease_status"] = pd.Series(
        pd.NA,
        index=labels.index,
        dtype="string",
    )

    # Assign normal rows.
    labels.loc[
        labels["normal_positive"],
        "disease_status",
    ] = "normal"

    # Assign pathology rows.
    labels.loc[
        labels["pathology_positive"],
        "disease_status",
    ] = "pathology"

    # -------------------------------------------------------------------------
    # Remove rows that cannot be assigned to either thesis class.
    # -------------------------------------------------------------------------

    labels = labels[
        labels["disease_status"].isin(LABELS)
    ].copy()

    # -------------------------------------------------------------------------
    # Merge image metadata with study-level labels.
    #
    # There can be multiple images for one study, which is why the merge is
    # many-to-one from image metadata to study labels.
    # -------------------------------------------------------------------------

    df = meta.merge(
        labels[
            [
                "subject_id",
                "study_id",
                "disease_status",
            ]
        ],
        on=[
            "subject_id",
            "study_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    if df.empty:
        raise RuntimeError(
            "No eligible PA/AP labelled images were found."
        )

    # -------------------------------------------------------------------------
    # Standardise ID columns.
    # -------------------------------------------------------------------------

    for column in (
        "subject_id",
        "study_id",
        "dicom_id",
    ):
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # -------------------------------------------------------------------------
    # Patient-level category.
    #
    # This is informational and is NOT used to collapse the patient to one
    # image.
    #
    # If a patient has at least one pathology-labelled eligible study, the
    # patient category is pathology.
    # -------------------------------------------------------------------------

    df["patient_category"] = (
        df.groupby("subject_id")[
            "disease_status"
        ].transform(
            lambda values: (
                "pathology"
                if (values == "pathology").any()
                else "normal"
            )
        )
    )

    # -------------------------------------------------------------------------
    # Sort deterministically.
    # -------------------------------------------------------------------------

    df = (
        df.sort_values(
            [
                "subject_id",
                "study_id",
                "dicom_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# INTEGER COUNT HELPERS
# =============================================================================

def _exact_count(
    value: float,
    name: str,
) -> int:
    """
    Convert a fractional count to an integer only if it is exactly integral.

    This prevents the code from silently changing requested sample sizes.
    """

    rounded = int(
        round(value)
    )

    if not np.isclose(
        value,
        rounded,
    ):
        raise ValueError(
            f"{name}={value} is not an integer. "
            "Choose a compatible pool size."
        )

    return rounded


def compute_pool_class_counts(
    pool_size: int,
) -> dict[str, int]:
    """
    Calculate exact normal/pathology IMAGE counts for one pool.
    """

    if (
        not isinstance(pool_size, int)
        or pool_size <= 0
    ):
        raise ValueError(
            "pool_size must be a positive integer."
        )

    if not np.isclose(
        NORMAL_FRACTION
        + PATHOLOGY_FRACTION,
        1.0,
    ):
        raise ValueError(
            "NORMAL_FRACTION + PATHOLOGY_FRACTION "
            "must equal 1.0."
        )

    normal = _exact_count(
        pool_size * NORMAL_FRACTION,
        "pool_size * NORMAL_FRACTION",
    )

    pathology = _exact_count(
        pool_size * PATHOLOGY_FRACTION,
        "pool_size * PATHOLOGY_FRACTION",
    )

    if (
        normal + pathology
        != pool_size
    ):
        raise ValueError(
            "Normal/pathology counts do not sum "
            "to pool_size."
        )

    return {
        "normal": normal,
        "pathology": pathology,
    }


def compute_split_counts(
    pool_size: int,
) -> dict[str, int]:
    """
    Calculate dynamic train/validation/test IMAGE counts.
    """

    if not np.isclose(
        TRAIN_FRACTION
        + VAL_FRACTION
        + TEST_FRACTION,
        1.0,
    ):
        raise ValueError(
            "TRAIN_FRACTION + VAL_FRACTION + "
            "TEST_FRACTION must equal 1.0."
        )

    raw = {
        "train": (
            pool_size
            * TRAIN_FRACTION
        ),
        "val": (
            pool_size
            * VAL_FRACTION
        ),
        "test": (
            pool_size
            * TEST_FRACTION
        ),
    }

    counts = {
        name: _exact_count(
            value,
            f"{name} split count",
        )
        for name, value in raw.items()
    }

    if sum(counts.values()) != pool_size:
        raise ValueError(
            f"Split counts {counts} do not sum "
            f"to pool_size={pool_size}."
        )

    return counts


# =============================================================================
# PATIENT-LEVEL SUBSET-SUM SELECTION
# =============================================================================

def choose_patient_subset_exact(
    patient_sizes: pd.Series,
    target_images: int,
    rng: np.random.Generator,
) -> list[str] | None:
    """
    Select a set of WHOLE PATIENTS whose combined number of images equals
    target_images exactly.

    This is the key mechanism that allows:

        * multiple images per patient

    while simultaneously guaranteeing:

        * exact requested image count
        * no patient splitting across train/val/test

    Example:

        Patient A -> 4 images
        Patient B -> 2 images
        Patient C -> 7 images

    If the target is 6 images, A+B can be selected.

    The function uses dynamic programming over reachable image counts.

    Returns:
        list of patient IDs if an exact solution exists
        None otherwise
    """

    if target_images < 0:
        return None

    # Convert Series into a list of:
    #
    #     (patient_id, image_count)
    #
    items = list(
        patient_sizes.items()
    )

    if not items:
        return (
            []
            if target_images == 0
            else None
        )

    # Randomise patient processing order reproducibly.
    order = rng.permutation(
        len(items)
    )

    items = [
        items[index]
        for index in order
    ]

    # -------------------------------------------------------------------------
    # reachable[image_count] = predecessor information
    #
    # Example:
    #
    # reachable[7] = (4, "patient123")
    #
    # means:
    #
    #     7 was reached by adding patient123 to a previous total of 4.
    # -------------------------------------------------------------------------

    reachable: dict[
        int,
        tuple[int | None, str | None],
    ] = {
        0: (None, None)
    }

    for patient_id, size in items:

        size = int(size)

        # A patient contributing zero images is irrelevant.
        if size <= 0:
            continue

        # A single patient larger than the target cannot be included.
        if size > target_images:
            continue

        # Snapshot current reachable totals.
        current_sums = list(
            reachable.keys()
        )

        for current_sum in current_sums:

            new_sum = (
                current_sum
                + size
            )

            if new_sum > target_images:
                continue

            # Already reachable; keep the first solution encountered.
            if new_sum in reachable:
                continue

            reachable[new_sum] = (
                current_sum,
                patient_id,
            )

            # Exact target found.
            if new_sum == target_images:

                selected = []

                cursor = new_sum

                while cursor != 0:

                    previous_sum, selected_patient = (
                        reachable[cursor]
                    )

                    assert (
                        previous_sum is not None
                    )

                    assert (
                        selected_patient is not None
                    )

                    selected.append(
                        selected_patient
                    )

                    cursor = previous_sum

                return selected

    return None


# =============================================================================
# SELECT ONE COMPLETE IMAGE POOL
# =============================================================================

def select_image_balanced_pool(
    candidates: pd.DataFrame,
    pool_size: int,
    seed: int,
    excluded_patients: set[str] | None = None,
) -> pd.DataFrame:
    """
    Select exactly pool_size IMAGES.

    Multiple images from the same patient are allowed.

    However, if a patient is selected, all selected images from that patient
    remain together and can later be assigned to exactly one split.

    Class balance is exact at the IMAGE level.

    excluded_patients:
        Patients that are forbidden from entering this pool.

        This is how pool B is made independent of pool A.
    """

    if excluded_patients is None:
        excluded_patients = set()

    class_counts = (
        compute_pool_class_counts(
            pool_size
        )
    )

    # Remove patients already used by the other pool.
    available = candidates[
        ~candidates["subject_id"].isin(
            excluded_patients
        )
    ].copy()

    selected_parts = []

    # -------------------------------------------------------------------------
    # Select normal images and pathology images independently.
    # -------------------------------------------------------------------------

    for class_index, label in enumerate(
        LABELS
    ):

        target = class_counts[label]

        class_df = available[
            available["disease_status"]
            == label
        ].copy()

        # Number of eligible images contributed by each patient.
        patient_sizes = (
            class_df
            .groupby("subject_id")
            .size()
            .sort_index()
        )

        if patient_sizes.sum() < target:
            raise RuntimeError(
                f"Not enough {label} images after "
                f"patient exclusions. "
                f"Need {target}, have "
                f"{patient_sizes.sum()}."
            )

        # Use a reproducible random ordering so that multiple valid patient
        # subsets do not always choose the same patients.
        rng = np.random.default_rng(
            seed + class_index
        )

        patient_ids = (
            choose_patient_subset_exact(
                patient_sizes=patient_sizes,
                target_images=target,
                rng=rng,
            )
        )

        if patient_ids is None:
            raise RuntimeError(
                f"Could not select exactly "
                f"{target} {label} images using "
                "whole patients."
            )

        # IMPORTANT:
        #
        # We retain ALL eligible images belonging to the selected patients
        # within this class.
        #
        # This is what allows multiple images per patient.
        chosen = class_df[
            class_df["subject_id"].isin(
                patient_ids
            )
        ].copy()

        if len(chosen) != target:
            raise AssertionError(
                f"Internal selection error for "
                f"{label}: expected {target} "
                f"images, got {len(chosen)}."
            )

        selected_parts.append(
            chosen
        )

    # Combine normal + pathology.
    pool = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Hard checks.
    # -------------------------------------------------------------------------

    if len(pool) != pool_size:
        raise AssertionError(
            "Selected pool has the wrong "
            "number of images."
        )

    class_actual = (
        pool["disease_status"]
        .value_counts()
        .to_dict()
    )

    assert (
        class_actual.get("normal", 0)
        == class_counts["normal"]
    )

    assert (
        class_actual.get("pathology", 0)
        == class_counts["pathology"]
    )

    # -------------------------------------------------------------------------
    # Informative diagnostic:
    #
    # duplicated subject IDs are EXPECTED because multiple images per patient
    # are explicitly allowed.
    # -------------------------------------------------------------------------

    duplicate_mask = (
        pool["subject_id"]
        .duplicated(keep=False)
    )

    if duplicate_mask.any():

        duplicated_patients = (
            pool.loc[
                duplicate_mask,
                "subject_id",
            ]
            .nunique()
        )

        log(
            f"  {duplicated_patients} patients "
            "contribute multiple images "
            "to this pool."
        )

    return (
        pool.sort_values(
            [
                "subject_id",
                "study_id",
                "dicom_id",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# SPLIT ONE POOL AT PATIENT LEVEL
# =============================================================================

def split_pool(
    pool: pd.DataFrame,
    pool_name: str,
    pool_size: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split a pool into train/validation/test.

    IMPORTANT:
        The split is performed using WHOLE PATIENT GROUPS.

    Therefore:

        patient 123's images -> train

    means ALL selected images for patient 123 go to train.

    They cannot simultaneously appear in validation or test.

    Because patients contribute different numbers of images, the number of
    patients in each split will not necessarily be 60/20/20.

    The IMAGE counts are exact.
    """

    counts = compute_split_counts(
        pool_size
    )

    # -------------------------------------------------------------------------
    # Count how many images each patient contributes.
    # -------------------------------------------------------------------------

    patient_sizes = (
        pool.groupby("subject_id")
        .size()
        .sort_index()
    )

    if patient_sizes.sum() != pool_size:
        raise AssertionError(
            "Patient image counts do not sum "
            "to pool size."
        )

    # -------------------------------------------------------------------------
    # Select exact train image count using whole patients.
    # -------------------------------------------------------------------------

    train_ids = (
        choose_patient_subset_exact(
            patient_sizes=patient_sizes,
            target_images=counts["train"],
            rng=np.random.default_rng(
                seed
            ),
        )
    )

    if train_ids is None:
        raise RuntimeError(
            f"{pool_name}: could not form an "
            f"exact {counts['train']}-image "
            "train split using whole patients."
        )

    train_ids_set = set(
        train_ids
    )

    # -------------------------------------------------------------------------
    # Remove train patients.
    # -------------------------------------------------------------------------

    remaining = patient_sizes.drop(
        train_ids_set,
        errors="ignore",
    )

    # -------------------------------------------------------------------------
    # Select exact validation image count from remaining patients.
    # -------------------------------------------------------------------------

    val_ids = (
        choose_patient_subset_exact(
            patient_sizes=remaining,
            target_images=counts["val"],
            rng=np.random.default_rng(
                seed + 1
            ),
        )
    )

    if val_ids is None:
        raise RuntimeError(
            f"{pool_name}: could not form an "
            f"exact {counts['val']}-image "
            "validation split."
        )

    val_ids_set = set(
        val_ids
    )

    # -------------------------------------------------------------------------
    # Everything remaining becomes test.
    # -------------------------------------------------------------------------

    test_ids_set = (
        set(remaining.index)
        - val_ids_set
    )

    # -------------------------------------------------------------------------
    # Build image-level dataframes.
    #
    # Because selection is by subject_id, all images belonging to a patient
    # stay together.
    # -------------------------------------------------------------------------

    train = pool[
        pool["subject_id"].isin(
            train_ids_set
        )
    ].copy()

    val = pool[
        pool["subject_id"].isin(
            val_ids_set
        )
    ].copy()

    test = pool[
        pool["subject_id"].isin(
            test_ids_set
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Shuffle image row order within each split.
    #
    # This changes only row order, NOT patient membership.
    # -------------------------------------------------------------------------

    train = train.sample(
        frac=1,
        random_state=seed + 10,
    ).reset_index(drop=True)

    val = val.sample(
        frac=1,
        random_state=seed + 11,
    ).reset_index(drop=True)

    test = test.sample(
        frac=1,
        random_state=seed + 12,
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Validate all split constraints.
    # -------------------------------------------------------------------------

    verify_split(
        train,
        val,
        test,
        pool_name,
        pool_size,
    )

    return (
        train,
        val,
        test,
    )


# =============================================================================
# VERIFY SPLITS
# =============================================================================

def verify_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    pool_name: str,
    pool_size: int,
) -> None:
    """
    Verify:

        * exact image counts
        * no patient leakage
        * complete patient coverage
    """

    counts = compute_split_counts(
        pool_size
    )

    frames = {
        "train": train,
        "val": val,
        "test": test,
    }

    # -------------------------------------------------------------------------
    # Verify exact IMAGE counts.
    # -------------------------------------------------------------------------

    for split_name, frame in frames.items():

        expected = counts[
            split_name
        ]

        assert len(frame) == expected, (
            f"{pool_name}: {split_name} "
            f"has {len(frame)} images; "
            f"expected {expected}."
        )

    # -------------------------------------------------------------------------
    # Get patient sets.
    # -------------------------------------------------------------------------

    train_ids = set(
        train["subject_id"]
    )

    val_ids = set(
        val["subject_id"]
    )

    test_ids = set(
        test["subject_id"]
    )

    # -------------------------------------------------------------------------
    # No patient may occur in more than one split.
    # -------------------------------------------------------------------------

    assert not (
        train_ids & val_ids
    ), (
        f"{pool_name}: train/val "
        "patient overlap."
    )

    assert not (
        train_ids & test_ids
    ), (
        f"{pool_name}: train/test "
        "patient overlap."
    )

    assert not (
        val_ids & test_ids
    ), (
        f"{pool_name}: val/test "
        "patient overlap."
    )

    # -------------------------------------------------------------------------
    # Every selected patient must occur in exactly one split.
    # -------------------------------------------------------------------------

    all_ids = (
        train_ids
        | val_ids
        | test_ids
    )

    assert len(all_ids) == (
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
    ), (
        f"{pool_name}: a patient appears "
        "in multiple splits."
    )

    # -------------------------------------------------------------------------
    # All images must be accounted for.
    # -------------------------------------------------------------------------

    total_images = sum(
        len(frame)
        for frame in frames.values()
    )

    assert total_images == pool_size, (
        f"{pool_name}: split images sum to "
        f"{total_images}, expected {pool_size}."
    )


# =============================================================================
# PHYSIONET IMAGE URL
# =============================================================================

def source_url(
    row: pd.Series,
) -> str:
    """
    Construct the original MIMIC-CXR-JPG JPEG URL.

    Example structure:

        .../files/p10/p10000032/s50414267/xxxx.jpg
    """

    subject_id = str(
        row["subject_id"]
    )

    # MIMIC stores each subject under the first two digits of subject_id.
    prefix = subject_id[:2]

    return (
        f"{BASE_URL}/files/"
        f"p{prefix}/"
        f"p{subject_id}/"
        f"s{row['study_id']}/"
        f"{row['dicom_id']}.jpg"
    )


# =============================================================================
# LOCAL IMAGE PATH
# =============================================================================

def local_path(
    root: Path,
    split: str,
    row: pd.Series,
) -> Path:
    """
    Build the local output path.

    Example:

        data/real/train/normal/
            p10000032_s50414267_xxxxx.jpg

    """

    filename = (
        f"p{row['subject_id']}_"
        f"s{row['study_id']}_"
        f"{row['dicom_id']}.jpg"
    )

    return (
        root
        / split
        / str(row["disease_status"])
        / filename
    )


# =============================================================================
# DOWNLOAD ONE JPEG
# =============================================================================

def download_one(
    session: requests.Session,
    url: str,
    destination: Path,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, int]:
    """
    Download one JPEG.

    The original JPEG is saved directly.

    There is NO resizing, recompression, normalization, or image processing
    here.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Reuse an existing file.
    # -------------------------------------------------------------------------

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        return (
            "already_exists",
            destination.stat().st_size,
        )

    tmp = destination.with_suffix(
        destination.suffix + ".part"
    )

    # -------------------------------------------------------------------------
    # Retry failed downloads.
    # -------------------------------------------------------------------------

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            with session.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            ) as response:

                # Credential/access failure.
                if response.status_code in (
                    401,
                    403,
                ):
                    return (
                        f"http_{response.status_code}",
                        0,
                    )

                # Retry transient server/rate-limit failures.
                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise RuntimeError(
                        f"HTTP {response.status_code}"
                    )

                response.raise_for_status()

                content_type = (
                    response.headers.get(
                        "Content-Type"
                    )
                    or ""
                ).lower()

                # Do not save an HTML login page as a JPEG.
                if "text/html" in content_type:
                    raise RuntimeError(
                        "HTML/login page returned "
                        "instead of JPEG."
                    )

                # -------------------------------------------------------------
                # Stream the original bytes directly to disk.
                # -------------------------------------------------------------

                with open(
                    tmp,
                    "wb",
                ) as output:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if chunk:
                            output.write(chunk)

            # -----------------------------------------------------------------
            # Confirm the download is non-empty.
            # -----------------------------------------------------------------

            if (
                not tmp.exists()
                or tmp.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Empty image download."
                )

            # -----------------------------------------------------------------
            # Basic JPEG magic-byte check.
            #
            # JPEG files begin with FF D8.
            # -----------------------------------------------------------------

            with open(
                tmp,
                "rb",
            ) as image_file:

                if (
                    image_file.read(2)
                    != b"\xff\xd8"
                ):
                    raise RuntimeError(
                        "Downloaded file is not a JPEG."
                    )

            # Only expose the final filename after validation.
            tmp.replace(
                destination
            )

            return (
                "downloaded",
                destination.stat().st_size,
            )

        except Exception as exc:

            if tmp.exists():
                tmp.unlink()

            if attempt < max_retries:

                time.sleep(
                    2 * attempt
                )

            else:

                return (
                    f"error: {exc}",
                    0,
                )

    return (
        "failed",
        0,
    )


# =============================================================================
# IMAGE VALIDATION
# =============================================================================

def validate_image(
    path: Path,
) -> bool:
    """
    Verify that a downloaded JPEG can actually be opened.

    Pillow verification is performed after the file has been downloaded.
    """

    try:

        # First verify the file structure.
        with Image.open(path) as image:
            image.verify()

        # Re-open after verify() and confirm valid dimensions.
        with Image.open(path) as image:

            return (
                image.width > 0
                and image.height > 0
            )

    except Exception:

        return False


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

def prepare_output_dirs() -> None:
    """
    Create the output directory structure (paths come from config.py).
    """

    for root in (REAL_DIR, MANIFEST_DIR):
        root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for label in LABELS:
            (REAL_DIR / split / label).mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAVE MANIFESTS
# =============================================================================

def save_master_catalogue(
    candidates: pd.DataFrame,
) -> Path:
    """
    Save the complete eligible IMAGE-level catalogue.

    This is metadata only.

    It is useful for reproducibility and future dataset-size experiments.
    """

    path = (
        MANIFEST_DIR
        / "master_eligible_images.csv"
    )

    candidates.to_csv(
        path,
        index=False,
    )

    return path


def save_manifests(
    pool_name: str,
    splits: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Save:

        pool_train.csv
        pool_val.csv
        pool_test.csv
        pool_all.csv
    """

    names = (
        "train",
        "val",
        "test",
    )

    for split_name, frame in zip(
        names,
        splits,
    ):

        frame.to_csv(
            MANIFEST_DIR
            / f"{pool_name}_{split_name}.csv",
            index=False,
        )

    # Combined manifest includes an explicit split column.
    combined = pd.concat(
        [
            frame.assign(
                split=name
            )
            for name, frame in zip(
                names,
                splits,
            )
        ],
        ignore_index=True,
    )

    combined.to_csv(
        MANIFEST_DIR
        / f"{pool_name}_all.csv",
        index=False,
    )


def save_pool_summary(
    pool_name: str,
    splits: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Save a compact summary of image and patient counts.

    This is particularly useful because patient counts will not necessarily
    follow the 60/20/20 image fractions.
    """

    names = (
        "train",
        "val",
        "test",
    )

    rows = []

    for split_name, frame in zip(
        names,
        splits,
    ):

        rows.append(
            {
                "pool": pool_name,
                "split": split_name,
                "images": len(frame),
                "unique_patients": (
                    frame["subject_id"]
                    .nunique()
                ),
                "normal_images": int(
                    (
                        frame["disease_status"]
                        == "normal"
                    ).sum()
                ),
                "pathology_images": int(
                    (
                        frame["disease_status"]
                        == "pathology"
                    ).sum()
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        MANIFEST_DIR
        / f"{pool_name}_summary.csv",
        index=False,
    )


# =============================================================================
# DOWNLOAD ONE COMPLETE POOL
# =============================================================================

def download_pool(
    session: requests.Session,
    pool_name: str,
    root: Path,
    splits: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Download and validate every selected image for one pool.
    """

    for split_name, frame in zip(
        SPLITS,
        splits,
    ):

        log(
            f"\nDownloading "
            f"{pool_name}/{split_name}: "
            f"{len(frame)} images"
        )

        for index, (_, row) in enumerate(
            frame.iterrows(),
            start=1,
        ):

            url = source_url(
                row
            )

            destination = local_path(
                root,
                split_name,
                row,
            )

            status, _ = download_one(
                session,
                url,
                destination,
            )

            # -----------------------------------------------------------------
            # A failed selected image should stop the dataset build.
            #
            # We do NOT silently produce a partial thesis dataset.
            # -----------------------------------------------------------------

            if (
                status.startswith("http_")
                or status.startswith("error")
            ):

                raise RuntimeError(
                    f"Download failed for "
                    f"{pool_name}/{split_name} "
                    f"{index}/{len(frame)}\n"
                    f"Patient: {row['subject_id']}\n"
                    f"URL: {url}\n"
                    f"Status: {status}"
                )

            # -----------------------------------------------------------------
            # Validate the actual JPEG.
            # -----------------------------------------------------------------

            if not validate_image(
                destination
            ):

                raise RuntimeError(
                    "Invalid/corrupt JPEG after "
                    f"download: {destination}"
                )

            log(
                f"[{index:03d}/"
                f"{len(frame):03d}] "
                f"{row['disease_status']:9s} | "
                f"patient={row['subject_id']} | "
                f"{status}"
            )


# =============================================================================
# FINAL FILESYSTEM VERIFICATION
# =============================================================================

def verify_files(
    root: Path,
    pool_size: int,
) -> None:
    """
    Confirm that the filesystem contains exactly the expected number of
    images in every split.

    We do NOT require 50/50 class balance within each split because patients
    are kept intact and patients contribute unequal numbers of images.

    The exact 50/50 requirement applies to the COMPLETE POOL.
    """

    counts = compute_split_counts(
        pool_size
    )

    for split in SPLITS:

        expected_total = counts[
            split
        ]

        found_total = 0

        for label in LABELS:

            found = len(
                list(
                    (
                        root
                        / split
                        / label
                    ).glob("*.jpg")
                )
            )

            found_total += found

        assert (
            found_total
            == expected_total
        ), (
            f"{root}: {split} contains "
            f"{found_total} images; "
            f"expected {expected_total}."
        )


# =============================================================================
# TOP-LEVEL DATASET BUILD
# =============================================================================

def build_and_download(
    pool_size: int = POOL_SIZE_PER_POOL,
):
    """
    Run the complete end-to-end construction of the single real pool.

    Workflow:

        1. Validate configuration.
        2. Authenticate with PhysioNet.
        3. Download metadata + CheXpert.
        4. Build the eligible IMAGE-level catalogue.
        5. Select the pool (exact 50/50 classes, whole patients).
        6. Split it by patient (exact 60/20/20 image counts).
        7. Verify no leakage.
        8. Save manifests.
        9. Download the selected JPEGs.
       10. Validate filesystem counts.
    """

    if ONE_IMAGE_PER_PATIENT:
        raise ValueError(
            "ONE_IMAGE_PER_PATIENT must be False "
            "for the thesis design. Multiple images "
            "per patient are intentionally allowed."
        )

    pool_class_counts = compute_pool_class_counts(pool_size)
    split_counts = compute_split_counts(pool_size)

    log("=" * 78)
    log("MIMIC-CXR-JPG - REAL IMAGE POOL")
    log("=" * 78)
    log(f"Project root       : {PROJECT_ROOT}")
    log(f"Pool size          : {pool_size} IMAGES")
    log(
        f"Pool balance       : {pool_class_counts['normal']} normal / "
        f"{pool_class_counts['pathology']} pathology"
    )
    log(
        f"Split              : {split_counts['train']} train / "
        f"{split_counts['val']} validation / {split_counts['test']} test"
    )
    log("Patient sampling   : MULTIPLE IMAGES PER PATIENT ALLOWED")
    log("Patient leakage    : FORBIDDEN across splits")
    log(f"Random seed        : {RANDOM_SEED}")

    for folder in (DATA_DIR, RAW_DIR, MANIFEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Authenticate.
    username, password = get_credentials()
    session = create_physionet_session(username, password)

    # 2. Catalogue files.
    log("\nDownloading/refreshing PhysioNet catalogue files...")
    metadata_path, chexpert_path = download_catalogues(session)

    # 3. Image-level candidate catalogue (no one-image-per-patient reduction).
    candidates = build_candidate_table(metadata_path, chexpert_path)
    log(f"Eligible PA/AP image rows: {len(candidates):,}")
    log(f"Eligible unique patients: {candidates['subject_id'].nunique():,}")

    master_path = save_master_catalogue(candidates)
    log(f"Master image catalogue saved: {master_path}")

    # 4. Select the pool and check that an exact patient-level split exists,
    #    retrying with another deterministic sample if it does not.
    max_attempts = 30
    real_pool = None

    for attempt in range(max_attempts):
        try:
            log(f"\nPool construction attempt {attempt + 1}/{max_attempts}")

            real_pool = select_image_balanced_pool(
                candidates=candidates,
                pool_size=pool_size,
                seed=RANDOM_SEED + attempt * 100,
                excluded_patients=set(),
            )

            split_pool(
                real_pool,
                "real",
                pool_size,
                seed=RANDOM_SEED + 100_000 + attempt,
            )

            break

        except RuntimeError as exc:
            real_pool = None

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Could not construct a pool with exact image counts and "
                    f"an exact patient-level split after {max_attempts} "
                    "attempts. Try another POOL_SIZE_PER_POOL."
                ) from exc

            log(
                f"Sampling attempt {attempt + 1} was not feasible; "
                "trying another deterministic sample..."
            )

    assert real_pool is not None

    # 5. Final split (same seed as the feasibility test).
    real_splits = split_pool(
        real_pool,
        "real",
        pool_size,
        seed=RANDOM_SEED + 100_000 + attempt,
    )

    # 6. Directories and manifests.
    prepare_output_dirs()
    save_manifests("real", real_splits)
    save_pool_summary("real", real_splits)

    log("\nAllocation checks PASSED.")
    log(
        f"  real: {len(real_pool)} images, "
        f"{real_pool['subject_id'].nunique()} unique patients"
    )

    for split_name, frame in zip(SPLITS, real_splits):
        normal_count = int((frame["disease_status"] == "normal").sum())
        pathology_count = int((frame["disease_status"] == "pathology").sum())
        log(
            f"    {split_name:10s}: {len(frame)} images, "
            f"{frame['subject_id'].nunique()} patients, "
            f"normal={normal_count}, pathology={pathology_count}"
        )

    # 7. Download, then verify the filesystem.
    download_pool(session, "real", REAL_DIR, real_splits)
    verify_files(REAL_DIR, pool_size)

    log("\n" + "=" * 78)
    log("SUCCESS - DATASET READY")
    log("=" * 78)
    log(f"real/           : {REAL_DIR}")
    log(f"Total images    : {pool_size}")
    log(f"Manifests       : {MANIFEST_DIR}")
    log("Pool balance    : exact 50% normal / 50% pathology")
    log("Split           : dynamic 60% / 20% / 20%")
    log("Patient rule    : multiple images allowed; no patient leakage")

    return {
        "real_manifest": MANIFEST_DIR / "real_all.csv",
        "master_manifest": master_path,
        "real_dir": REAL_DIR,
        "pool_size": pool_size,
        "split_counts": split_counts,
    }


# =============================================================================
# BACKWARD-COMPATIBLE FUNCTION NAME
# =============================================================================

def download_dataset(
    pool_size: int = POOL_SIZE_PER_POOL,
):
    """
    Backward-compatible alias.

    This allows an older notebook to use:

        download_dataset(pool_size=500)

    while the preferred function is:

        build_and_download(pool_size=500)
    """

    return build_and_download(
        pool_size=pool_size
    )


# =============================================================================
# DIRECT SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":

    build_and_download()