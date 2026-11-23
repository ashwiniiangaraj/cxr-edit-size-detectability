# ============================================================
# VISUAL QA FOR SYNTHETIC CHEST X-RAYS
# ============================================================

from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from IPython.display import display

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Main project directory.

PROJECT_ROOT = Path(config.PROJECT_ROOT)


# ------------------------------------------------------------
# Real image directory
# ------------------------------------------------------------
REAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "real"
)


# ------------------------------------------------------------
# Synthetic image directory
# ------------------------------------------------------------
#
# All three IP techniques are stored below this directory.
#
SYNTHETIC_ROOT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "IP"
)


# ------------------------------------------------------------
# Output directory for visual QA figures
# ------------------------------------------------------------
#
# The script saves the final 3-example figures here.
#
VISUAL_QA_DIR = (
    SYNTHETIC_ROOT
    / "visual_qa"
)


# ============================================================
# 2. VISUALISATION SETTINGS
# ============================================================

# Number of examples to display for EACH technique.
#
# Therefore the final output will contain:
#
#   3 Blob examples
#   3 Copy-Move examples
#   3 Texture Transplant examples
#
N_EXAMPLES = 3
SELECTION_MODE = "random"
RANDOM_SEED = 42
SPLIT_FILTER = None


# ============================================================
# 3. TECHNIQUE LOCATIONS
# ============================================================
TECHNIQUES = {

    "Blob": {

        "root": (
            SYNTHETIC_ROOT
            / "blob"
        ),

        "metadata_candidates": [

            (
                SYNTHETIC_ROOT
                / "blob"
                / "metadata"
                / "blob_synthetic_metadata.csv"
            )

        ],
    },


    "Copy-Move": {

        "root": (
            SYNTHETIC_ROOT
            / "copy"
        ),

        "metadata_candidates": [

            (
                SYNTHETIC_ROOT
                / "copy"
                / "reports"
                / "copy_move_final_metadata.csv"
            )

        ],
    },


    "Texture Transplant": {

        "root": (
            SYNTHETIC_ROOT
            / "texture"
        ),

        "metadata_candidates": [

            (
                SYNTHETIC_ROOT
                / "texture"
                / "reports"
                / "texture_transplant_final_metadata.csv"
            )

        ],
    },
}


# ============================================================
# 4. ID CLEANING
# ============================================================

def clean_id(value):
    """
    Convert an ID from the metadata CSV into a clean string.

    This is useful because pandas can sometimes interpret an ID
    as a number or float.

    For example:

        13282748

    should become:

        "13282748"

    rather than:

        "13282748.0"

    Parameters
    ----------
    value:
        Value from a pandas DataFrame cell.

    Returns
    -------
    str
        Clean string representation of the ID.
    """

    # Missing values become an empty string.
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Handle IDs accidentally represented as floats.
    if value.endswith(".0"):

        try:
            value = str(
                int(float(value))
            )

        except Exception:
            pass

    return value


# ============================================================
# 5. FIND METADATA FILE
# ============================================================

def find_metadata(candidates):
    """
    Find the first metadata CSV that actually exists.

    Parameters
    ----------
    candidates:
        List of possible metadata paths.

    Returns
    -------
    Path
        Existing metadata path.

    Raises
    ------
    FileNotFoundError
        If none of the candidate files exist.
    """

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the metadata file.\n\n"
        "Checked:\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )


# ============================================================
# 6. READ AN IMAGE
# ============================================================

def read_image(path):
    """
    Read an image as an 8-bit grayscale NumPy array.

    EXIF orientation is respected.

    Parameters
    ----------
    path:
        Path to JPEG/PNG image.

    Returns
    -------
    np.ndarray
        2D uint8 grayscale image.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Image does not exist:\n{path}"
        )

    with Image.open(path) as img:

        # Respect any orientation information stored
        # in the image metadata.
        img = ImageOps.exif_transpose(img)

        # Convert to grayscale.
        img = img.convert("L")

        return np.asarray(
            img,
            dtype=np.uint8
        )


# ============================================================
# 7. RESOLVE A PATH STORED IN METADATA
# ============================================================

def resolve_existing_path(
    value,
    base_dir=None
):
    """
    Try to resolve a path stored in a metadata CSV.

    Different generators may store paths as:

        /absolute/path/file.jpg

    or:

        relative/path/file.jpg

    This function tries several sensible locations.

    Parameters
    ----------
    value:
        Path value from metadata.

    base_dir:
        Technique-specific directory.

    Returns
    -------
    Path or None
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    path = Path(value)

    candidates = []

    # Absolute path.
    if path.is_absolute():
        candidates.append(path)

    # Relative to the supplied base directory.
    if base_dir is not None:
        candidates.append(
            base_dir / path
        )

    # Relative to project root.
    candidates.append(
        PROJECT_ROOT / path
    )

    # Relative to current working directory.
    candidates.append(path)

    for candidate in candidates:

        if candidate.exists():

            return candidate.resolve()

    return None


# ============================================================
# 8. PREPARE METADATA
# ============================================================

def prepare_metadata(metadata_path):
    """
    Read and standardise metadata.

    The major purpose of this function is to make the different
    metadata formats compatible.

    """

    df = pd.read_csv(
        metadata_path
    )

    print(
        f"\nMetadata: {metadata_path}"
    )

    print(
        f"Rows: {len(df)}"
    )


    # --------------------------------------------------------
    # CASE 1: Blob / Copy-Move
    # --------------------------------------------------------
    #
    # These already have the generic identifiers.
    #

    generic_columns = {
        "subject_id",
        "study_id",
        "dicom_id",
    }

    if generic_columns.issubset(
        df.columns
    ):

        df["subject_id"] = (
            df["subject_id"]
            .map(clean_id)
        )

        df["study_id"] = (
            df["study_id"]
            .map(clean_id)
        )

        df["dicom_id"] = (
            df["dicom_id"]
            .map(clean_id)
        )


    # --------------------------------------------------------
    # CASE 2: Texture Transplant
    # --------------------------------------------------------
    #
    # Texture Transplant uses TARGET identifiers.
    #
    #

    elif {
        "target_subject_id",
        "target_study_id",
        "target_dicom_id",
    }.issubset(df.columns):

        df["subject_id"] = (
            df["target_subject_id"]
            .map(clean_id)
        )

        df["study_id"] = (
            df["target_study_id"]
            .map(clean_id)
        )

        df["dicom_id"] = (
            df["target_dicom_id"]
            .map(clean_id)
        )


    else:

        raise ValueError(
            f"{metadata_path} does not contain "
            "recognised target identifiers.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    #
    # Every synthetic image should inherit the split of its
    # real source image.
    #

    if "split" not in df.columns:

        raise ValueError(
            f"{metadata_path} is missing "
            "'split'."
        )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Disease label
    # --------------------------------------------------------
    #
    # Different generators may use slightly different names.
    #

    if "disease_status" in df.columns:

        df["disease_status"] = (
            df["disease_status"]
            .astype(str)
            .str.strip()
        )

    elif "target_disease" in df.columns:

        df["disease_status"] = (
            df["target_disease"]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # Clean path columns
    # --------------------------------------------------------

    path_columns = [

        "target_image_path",

        "donor_image_path",

        "synthetic_image_path",

        "mask_path",

    ]

    for column in path_columns:

        if column in df.columns:

            df[column] = df[column].apply(

                lambda value:
                    str(value).strip()
                    if not pd.isna(value)
                    else ""

            )


    return df


# ============================================================
# 9. BUILD REAL IMAGE FILENAME
# ============================================================

def build_real_filename(
    subject_id,
    study_id,
    dicom_id
):
    """
    Build the real image filename used by this project.

    The project's real images follow:

        p{subject_id}_s{study_id}_{dicom_id}.jpg

    Example:

        p13282748_s58553239_ABC123.jpg
    """

    return (
        f"p{clean_id(subject_id)}_"
        f"s{clean_id(study_id)}_"
        f"{clean_id(dicom_id)}.jpg"
    )


# ============================================================
# 10. FIND EXACT REAL SOURCE IMAGE
# ============================================================

def find_real_source(row):
    """
    Find the exact real image corresponding to the synthetic image.

    For Texture Transplant:
        target_image_path is preferred.

    Otherwise the filename is constructed from:

        subject_id
        study_id
        dicom_id

    which, after metadata normalisation, represent the TARGET.
    """

    # --------------------------------------------------------
    # First preference:
    #
    # Texture Transplant explicitly provides:
    #
    #     target_image_path
    #
    # This is the most direct way to identify the target image.
    # --------------------------------------------------------

    if "target_image_path" in row.index:

        explicit_path = (
            resolve_existing_path(
                row["target_image_path"],
                base_dir=REAL_DIR,
            )
        )

        if explicit_path is not None:

            return explicit_path


    # --------------------------------------------------------
    # Construct the filename.
    # --------------------------------------------------------

    filename = build_real_filename(

        row["subject_id"],

        row["study_id"],

        row["dicom_id"],

    )

    split_name = clean_id(
        row["split"]
    )


    # --------------------------------------------------------
    # Try the disease-specific directory first.
    # --------------------------------------------------------

    disease = ""

    if "disease_status" in row.index:

        disease = clean_id(
            row["disease_status"]
        )

    candidate_directories = []


    if disease:

        candidate_directories.append(

            REAL_DIR
            / split_name
            / disease

        )


    # --------------------------------------------------------
    # Fallback: search both class directories.
    # --------------------------------------------------------

    candidate_directories.extend([

        REAL_DIR
        / split_name
        / "normal",

        REAL_DIR
        / split_name
        / "pathology",

    ])


    for directory in candidate_directories:

        candidate = (
            directory
            / filename
        )

        if candidate.exists():

            return candidate.resolve()


    # --------------------------------------------------------
    # Final fallback:
    #
    # Search recursively under data/real.
    # --------------------------------------------------------

    matches = list(
        REAL_DIR.rglob(filename)
    )

    if matches:

        return matches[0].resolve()


    return None


# ============================================================
# 11. FIND SYNTHETIC IMAGE
# ============================================================

def find_synthetic_image(
    row,
    technique_root
):
    """
    Find the generated synthetic image.

    Texture Transplant provides:
        synthetic_image_path

    so this is normally the preferred method.

    If that is unavailable, the function tries filename-based
    fallbacks.
    """

    # --------------------------------------------------------
    # Explicit synthetic path.
    # --------------------------------------------------------

    if "synthetic_image_path" in row.index:

        path = resolve_existing_path(

            row["synthetic_image_path"],

            base_dir=technique_root,

        )

        if path is not None:

            return path


    # --------------------------------------------------------
    # Try other possible metadata path names.
    # --------------------------------------------------------

    for column in [

        "image_path",

        "output_path",

        "synthetic_path",

    ]:

        if column in row.index:

            path = resolve_existing_path(

                row[column],

                base_dir=technique_root,

            )

            if path is not None:

                return path


    # --------------------------------------------------------
    # Construct standard filename.
    # --------------------------------------------------------

    filename = build_real_filename(

        row["subject_id"],

        row["study_id"],

        row["dicom_id"],

    )


    # CHANGED: synthetic images are now PNG; the real-image name ends in
    # .jpg, so look for the .png twin first, then the old .jpg name.
    for name in (
        filename[:-4] + ".png",
        filename,
    ):

        matches = list(
            technique_root.rglob(name)
        )

        if matches:

            return matches[0].resolve()


    # --------------------------------------------------------
    # Try synthetic_id if it exists.
    # --------------------------------------------------------

    if "synthetic_id" in row.index:

        synthetic_id = clean_id(
            row["synthetic_id"]
        )

        possible_names = [

            f"{synthetic_id}.jpg",

            f"{synthetic_id}.jpeg",

            f"{synthetic_id}.png",

        ]

        for name in possible_names:

            matches = list(
                technique_root.rglob(name)
            )

            if matches:

                return matches[0].resolve()


    return None


# ============================================================
# 12. FIND CHANGED-REGION MASK
# ============================================================

def find_mask(
    row,
    technique_root,
    synthetic_path=None
):
    """
    Find the mask identifying the region changed by the
    synthetic generation technique.

    Priority:

        1. Explicit mask_path from metadata
        2. Mask next to synthetic image
        3. Common masks directory
        4. Recursive search
    """

    # --------------------------------------------------------
    # Explicit metadata mask.
    # --------------------------------------------------------

    if "mask_path" in row.index:

        path = resolve_existing_path(

            row["mask_path"],

            base_dir=technique_root,

        )

        if path is not None:

            return path


    # --------------------------------------------------------
    # Infer mask from synthetic image filename.
    #
    # Example:
    #
    #     image.jpg
    #
    # becomes:
    #
    #     image_mask.png
    # --------------------------------------------------------

    if synthetic_path is not None:

        stem = synthetic_path.stem

        candidates = [

            synthetic_path.with_name(
                f"{stem}_mask.png"
            ),

            (
                synthetic_path.parent
                / "masks"
                / f"{stem}_mask.png"
            ),

            (
                technique_root
                / "masks"
                / f"{stem}_mask.png"
            ),

        ]

        for candidate in candidates:

            if candidate.exists():

                return candidate.resolve()


    # --------------------------------------------------------
    # Construct mask name from target IDs.
    # --------------------------------------------------------

    base_name = (
        f"p{clean_id(row['subject_id'])}_"
        f"s{clean_id(row['study_id'])}_"
        f"{clean_id(row['dicom_id'])}"
    )

    possible_names = [

        f"{base_name}_mask.png",

        f"{base_name}_mask.jpg",

        f"{base_name}_mask.jpeg",

    ]


    for name in possible_names:

        matches = list(
            technique_root.rglob(name)
        )

        if matches:

            return matches[0].resolve()


    # --------------------------------------------------------
    # Final fallback using synthetic_id.
    # --------------------------------------------------------

    if "synthetic_id" in row.index:

        synthetic_id = clean_id(
            row["synthetic_id"]
        )

        for name in [

            f"{synthetic_id}_mask.png",

            f"{synthetic_id}_mask.jpg",

            f"{synthetic_id}_mask.jpeg",

        ]:

            matches = list(
                technique_root.rglob(name)
            )

            if matches:

                return matches[0].resolve()


    return None


# ============================================================
# 13. NORMALISE MASK
# ============================================================

def normalise_mask(
    mask_array,
    image_shape
):
    """
    Convert a mask into a binary mask matching the synthetic
    image dimensions.

    A binary mask contains:

        False = unchanged region
        True  = changed region
    """

    if mask_array is None:

        return None


    mask = np.asarray(
        mask_array
    )


    # --------------------------------------------------------
    # If the mask somehow has multiple channels, use the first.
    # --------------------------------------------------------

    if mask.ndim == 3:

        mask = mask[..., 0]


    target_height = image_shape[0]
    target_width = image_shape[1]


    # --------------------------------------------------------
    # Resize mask if necessary.
    #
    # NEAREST is important for masks because interpolation
    # should not create intermediate values.
    # --------------------------------------------------------

    if mask.shape != (
        target_height,
        target_width
    ):

        mask_image = Image.fromarray(
            mask.astype(np.uint8)
        )

        mask_image = mask_image.resize(

            (
                target_width,
                target_height,
            ),

            Image.Resampling.NEAREST,

        )

        mask = np.asarray(
            mask_image,
            dtype=np.uint8
        )


    # --------------------------------------------------------
    # Convert to binary.
    # --------------------------------------------------------

    if mask.max() > 1:

        mask = mask > 0

    else:

        mask = mask.astype(bool)


    return mask


# ============================================================
# 14. LOAD MASK
# ============================================================

def load_mask(
    path,
    image_shape
):
    """
    Load a mask image and convert it to a binary mask.
    """

    if path is None:

        return None


    try:

        with Image.open(path) as img:

            mask = np.asarray(

                img.convert("L"),

                dtype=np.uint8

            )


        return normalise_mask(

            mask,

            image_shape

        )


    except Exception as exc:

        warnings.warn(

            f"Could not load mask "
            f"{path}: {exc}"

        )

        return None


# ============================================================
# 15. CREATE HIGHLIGHT IMAGE
# ============================================================

def create_highlight_image(
    image,
    mask,
    alpha=0.40
):
    """
    Create a visualisation of the synthetic image with the
    changed region highlighted.

    The original image is NOT changed.

    The changed region receives:

        red translucent overlay

    and its boundary receives:

        yellow contour

    """

    image = np.asarray(
        image,
        dtype=np.uint8
    )


    if mask is None:

        return image


    if not mask.any():

        return image


    # --------------------------------------------------------
    # Convert grayscale image into RGB.
    # --------------------------------------------------------

    rgb = np.stack(

        [
            image,
            image,
            image,
        ],

        axis=-1

    ).astype(np.float32)


    # --------------------------------------------------------
    # Create red overlay.
    # --------------------------------------------------------

    red = np.zeros_like(
        rgb
    )

    red[..., 0] = 255


    # Blend the red colour with the original image.
    #
    # alpha = 0.40 means approximately:
    #
    #     60% original image
    #     40% red overlay
    #

    rgb[mask] = (

        (1 - alpha)
        * rgb[mask]

        +

        alpha
        * red[mask]

    )


    rgb = np.clip(

        rgb,

        0,

        255

    ).astype(np.uint8)


    # --------------------------------------------------------
    # Create yellow boundary.
    # --------------------------------------------------------
    #
    # We compare each mask pixel with its four neighbours.
    # Pixels at the edge of the mask become the contour.
    #

    padded = np.pad(

        mask,

        1,

        mode="constant",

        constant_values=False

    )


    interior = (

        padded[:-2, 1:-1]

        &

        padded[2:, 1:-1]

        &

        padded[1:-1, :-2]

        &

        padded[1:-1, 2:]

    )


    contour = (
        mask
        &
        ~interior
    )


    # Yellow contour.
    rgb[contour] = [
        255,
        255,
        0,
    ]


    return rgb


# ============================================================
# 16. CHOOSE THREE EXAMPLES
# ============================================================

def choose_examples(df):
    """
    Select the examples that will be visualised.

    The split filter is applied first.
    """

    working = df.copy()


    # --------------------------------------------------------
    # Optional split restriction.
    # --------------------------------------------------------

    if SPLIT_FILTER is not None:

        working = working[

            working["split"].str.lower()

            ==

            SPLIT_FILTER.lower()

        ].copy()


    if len(working) == 0:

        raise ValueError(

            f"No examples available "
            f"for split '{SPLIT_FILTER}'."

        )


    # We cannot select more examples than exist.
    n = min(

        N_EXAMPLES,

        len(working)

    )


    # --------------------------------------------------------
    # Select first examples.
    # --------------------------------------------------------

    if SELECTION_MODE == "first":

        selected = working.head(n)


    # --------------------------------------------------------
    # Select reproducible random examples.
    # --------------------------------------------------------

    elif SELECTION_MODE == "random":

        selected = working.sample(

            n=n,

            random_state=RANDOM_SEED

        )


    else:

        raise ValueError(

            "SELECTION_MODE must be "
            "'random' or 'first'."

        )


    return selected.reset_index(
        drop=True
    )


# ============================================================
# 17. VISUALISE ONE TECHNIQUE
# ============================================================

def visualise_technique(
    technique_name,
    info,
    save=True
):
    """
    Generate one visual QA figure for a technique.

    Layout:

        Real       Synthetic       Highlighted

        Real       Synthetic       Highlighted

        Real       Synthetic       Highlighted

    Each row represents one synthetic image.
    """

    root = info["root"]


    # --------------------------------------------------------
    # Find and read metadata.
    # --------------------------------------------------------

    metadata_path = find_metadata(

        info["metadata_candidates"]

    )

    df = prepare_metadata(
        metadata_path
    )


    # --------------------------------------------------------
    # Select examples.
    # --------------------------------------------------------

    examples = choose_examples(
        df
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        technique_name
    )

    print(
        "=" * 78
    )

    print(
        f"Metadata rows: "
        f"{len(df)}"
    )

    print(
        f"Selected examples: "
        f"{len(examples)}"
    )


    # --------------------------------------------------------
    # Create figure.
    #
    # 3 columns:
    #
    #   1. Real
    #   2. Synthetic
    #   3. Highlighted
    #
    # Number of rows = N_EXAMPLES.
    # --------------------------------------------------------

    fig, axes = plt.subplots(

        nrows=len(examples),

        ncols=3,

        figsize=(
            12,
            4 * len(examples)
        )

    )


    # When there is only one example, matplotlib returns a
    # one-dimensional axes array. Convert it into a 2D shape
    # so the rest of the code remains consistent.
    if len(examples) == 1:

        axes = np.expand_dims(

            axes,

            axis=0

        )


    successful = 0


    # ========================================================
    # PROCESS EACH EXAMPLE
    # ========================================================

    for row_index, (_, row) in enumerate(
        examples.iterrows()
    ):

        try:

            # ------------------------------------------------
            # Find the REAL source/target.
            # ------------------------------------------------
            #
            # For Texture Transplant this is the TARGET.
            #
            # For Blob/Copy-Move this is the original source.
            #

            real_path = find_real_source(
                row
            )


            if real_path is None:

                raise FileNotFoundError(

                    "Could not resolve "
                    "the real source image."

                )


            # ------------------------------------------------
            # Find synthetic image.
            # ------------------------------------------------

            synthetic_path = find_synthetic_image(

                row,

                root

            )


            if synthetic_path is None:

                raise FileNotFoundError(

                    "Could not resolve "
                    "the synthetic image."

                )


            # ------------------------------------------------
            # Find changed-region mask.
            # ------------------------------------------------

            mask_path = find_mask(

                row,

                root,

                synthetic_path

            )


            # ------------------------------------------------
            # Read real and synthetic images.
            # ------------------------------------------------

            real_image = read_image(

                real_path

            )

            synthetic_image = read_image(

                synthetic_path

            )
            if real_image.shape != synthetic_image.shape:

                real_image = np.asarray(
                    Image.fromarray(real_image).resize(
                        (
                            synthetic_image.shape[1],
                            synthetic_image.shape[0],
                        ),
                        Image.Resampling.LANCZOS,
                    )
                )


            # ------------------------------------------------
            # Read changed-region mask.
            # ------------------------------------------------

            mask = None

            if mask_path is not None:

                mask = load_mask(

                    mask_path,

                    synthetic_image.shape

                )


            # ------------------------------------------------
            # Create highlighted synthetic image.
            # ------------------------------------------------

            highlighted = (
                create_highlight_image(

                    synthetic_image,

                    mask

                )
            )


            # ------------------------------------------------
            # COLUMN 1:
            # REAL SOURCE / TARGET
            # ------------------------------------------------

            axes[
                row_index,
                0
            ].imshow(

                real_image,

                cmap="gray",

                vmin=0,

                vmax=255

            )


            axes[
                row_index,
                0
            ].set_title(
                "Real source / target"
            )


            # ------------------------------------------------
            # COLUMN 2:
            # SYNTHETIC
            # ------------------------------------------------

            axes[
                row_index,
                1
            ].imshow(

                synthetic_image,

                cmap="gray",

                vmin=0,

                vmax=255

            )


            axes[
                row_index,
                1
            ].set_title(
                "Synthetic"
            )


            # ------------------------------------------------
            # COLUMN 3:
            # SYNTHETIC + CHANGED REGION
            # ------------------------------------------------

            axes[
                row_index,
                2
            ].imshow(

                highlighted

            )


            axes[
                row_index,
                2
            ].set_title(

                "Synthetic + changed region"

            )


            # ------------------------------------------------
            # Remove axes/ticks.
            # ------------------------------------------------

            for column_index in range(3):

                axes[
                    row_index,
                    column_index
                ].axis("off")


            # ------------------------------------------------
            # Disease and split information.
            # ------------------------------------------------

            disease = ""

            if (
                "disease_status"
                in row.index
            ):

                disease = clean_id(

                    row["disease_status"]

                )


            elif (
                "target_disease"
                in row.index
            ):

                disease = clean_id(

                    row["target_disease"]

                )


            split_name = clean_id(

                row["split"]

            )


            # ------------------------------------------------
            # Add a small identifier beside each row.
            # ------------------------------------------------

            row_label = (

                f"{split_name} | "

                f"{disease} | "

                f"p{row['subject_id']}"

            )


            axes[
                row_index,
                0
            ].set_ylabel(

                row_label,

                rotation=90,

                fontsize=9

            )


            # ------------------------------------------------
            # Print detailed information to notebook.
            # ------------------------------------------------

            print(
                f"\nExample "
                f"{row_index + 1}:"
            )

            print(
                f"  Split:     "
                f"{split_name}"
            )

            print(
                f"  Disease:   "
                f"{disease}"
            )

            print(
                f"  Real:      "
                f"{real_path}"
            )

            print(
                f"  Synthetic: "
                f"{synthetic_path}"
            )

            print(
                f"  Mask:      "
                f"{mask_path if mask_path else 'NOT FOUND'}"
            )

            if mask is not None:

                print(

                    f"  Changed "
                    f"pixels: "
                    f"{int(mask.sum())}"

                )

            else:

                print(

                    "  Changed "
                    "pixels: N/A"

                )


            successful += 1


        # ----------------------------------------------------
        # If one example fails, continue with the other
        # examples rather than stopping the entire technique.
        # ----------------------------------------------------

        except Exception as exc:

            print(
                f"\nExample "
                f"{row_index + 1} FAILED:"
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )


            for column_index in range(3):

                axes[
                    row_index,
                    column_index
                ].axis("off")


            axes[
                row_index,
                1
            ].text(

                0.5,

                0.5,

                f"ERROR\n{exc}",

                ha="center",

                va="center",

                wrap=True

            )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(

        f"{technique_name} — "
        f"Real vs Synthetic",

        fontsize=16,

        y=0.995

    )


    plt.tight_layout(

        rect=[
            0,
            0,
            1,
            0.98
        ]

    )


    # ========================================================
    # SAVE FIGURE
    # ========================================================

    if save:

        VISUAL_QA_DIR.mkdir(

            parents=True,

            exist_ok=True

        )


        # Convert technique name into a filename.
        #
        # Examples:
        #
        #     Blob
        #       ->
        #     blob_visual_qa.png
        #
        #     Copy-Move
        #       ->
        #     copy_move_visual_qa.png
        #

        safe_name = (

            technique_name

            .lower()

            .replace(
                " ",
                "_"
            )

            .replace(
                "-",
                "_"
            )

        )


        output_path = (

            VISUAL_QA_DIR

            / f"{safe_name}_visual_qa.png"

        )


        fig.savefig(

            output_path,

            dpi=200,

            bbox_inches="tight"

        )


        print(
            f"\nSaved figure:"
            f"\n  {output_path}"
        )


    # Display figure in notebook.
    plt.show()


    print(

        f"\nSuccessfully "
        f"visualised "
        f"{successful}/"
        f"{len(examples)} examples."

    )


    return fig


# ============================================================
# 18. RUN ALL THREE TECHNIQUES
# ============================================================

def visualise_all(save=True):
    """
    Run visual QA for:

        Blob
        Copy-Move
        Texture Transplant

    Each technique gets its own figure containing 3 examples.
    """

    results = {}


    for technique_name, info in (
        TECHNIQUES.items()
    ):

        try:

            fig = visualise_technique(

                technique_name,

                info,

                save=save

            )


            results[
                technique_name
            ] = fig


        except Exception as exc:

            # ------------------------------------------------
            # Important:
            # One technique failing should not prevent the
            # other techniques from being visualised.
            # ------------------------------------------------

            print(
                "\n"
                + "=" * 78
            )

            print(
                f"{technique_name} FAILED"
            )

            print(
                "=" * 78
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )


    return results


# ============================================================
# 19. DISPLAY ALREADY-SAVED FIGURES
# ============================================================

def show_saved_figures():
    """
    Display any visual QA figures that have already been saved.

    This is optional and is not required to generate new figures.
    """

    if not VISUAL_QA_DIR.exists():

        print(
            "Visual QA directory does not exist:"
        )

        print(
            VISUAL_QA_DIR
        )

        return


    files = sorted(

        VISUAL_QA_DIR.glob(
            "*_visual_qa.png"
        )

    )


    if not files:

        print(
            "No saved visual QA figures found."
        )

        return


    for path in files:

        print(
            f"\n{path.name}"
        )

        display(
            Image.open(path)
        )