"""
RadEdit production generation for the dataset.
"""

from pathlib import Path
import json
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

import config


# ============================================================
# Source image path
# ============================================================

GENERATOR_VERSION = "stage1_fair_v2"      


def _file_md5(path):
    """MD5 of a file's bytes (None if the file does not exist)."""
    import hashlib
    path = Path(path)
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def _settings_fingerprint(settings):
    """Short hash of every setting that changes the generated pixels (device is excluded)."""
    import hashlib
    relevant = {k: v for k, v in settings.items() if k not in ("device", "settings_fingerprint")}
    return hashlib.md5(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def source_path(row):
    """
    Construct the source-image path from the manifest row.
    """

    split = str(row["split"]).lower()
    disease = str(row["disease_status"]).lower()
    subject = str(row["subject_id"])
    study = str(row["study_id"])
    dicom = str(row["dicom_id"])

    return (
        Path(config.DIFFUSION_SOURCE_ROOT)
        / split
        / disease
        / f"p{subject}_s{study}_{dicom}.jpg"
    )


# ============================================================
# Deterministic seed
# ============================================================

def make_seed(image_number, weight):
    """
    Match the deterministic seed construction used in the
    working Diffusion HTML.

    HTML:
        SEED
        + image_number * 100
        + int(weight * 10)
    """

    return (
        int(config.DIFFUSION_MASTER_SEED)
        + int(image_number) * 100
        + int(float(weight) * 10)
    )


def set_all_seeds(seed):
    """
    Match the HTML deterministic seeding behaviour.
    """

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()


# ============================================================
# Load RadEdit
# ============================================================

def load_radedit():

    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        DiffusionPipeline,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )

    from transformers import (
        AutoModel,
        AutoTokenizer,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        f"cuda:{config.DIFFUSION_GPU_INDEX}"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Dtype
    # --------------------------------------------------------

    dtype = (
        torch.float16
        if device.startswith("cuda")
        else torch.float32
    )

    print("=" * 70)
    print("LOADING MICROSOFT RADEDIT")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Dtype : {dtype}")

    # --------------------------------------------------------
    # 1. RadEdit U-Net
    # --------------------------------------------------------

    print()
    print("[1/5] Loading RadEdit U-Net...")

    unet = UNet2DConditionModel.from_pretrained(
        config.DIFFUSION_MODEL_ID,
        subfolder="unet",
        torch_dtype=dtype,
    )

    print("      PASS")

    # --------------------------------------------------------
    # 2. SDXL VAE
    # --------------------------------------------------------

    print()
    print("[2/5] Loading SDXL VAE...")

    vae = AutoencoderKL.from_pretrained(
        config.DIFFUSION_VAE_MODEL_ID,
        torch_dtype=dtype,
    )

    print("      PASS")

    # --------------------------------------------------------
    # 3. BioViL-T tokenizer
    #
    # IMPORTANT:
    # The working HTML explicitly sets model_max_length=128.
    # Without this, RadEdit's tokenizer call can produce:
    #
    # OverflowError: int too big to convert
    #
    # --------------------------------------------------------

    print()
    print("[3/5] Loading BioViL-T tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.DIFFUSION_TEXT_MODEL_ID,
        model_max_length=128,
        trust_remote_code=True,
    )

    print(
        f"      PASS "
        f"(model_max_length={tokenizer.model_max_length})"
    )

    # --------------------------------------------------------
    # 4. BioViL-T text encoder
    # --------------------------------------------------------

    print()
    print("[4/5] Loading BioViL-T text encoder...")

    text_encoder = AutoModel.from_pretrained(
        config.DIFFUSION_TEXT_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    print("      PASS")

    # --------------------------------------------------------
    # 5. Official RadEdit DDIM scheduler
    # --------------------------------------------------------

    print()
    print("[5/5] Creating DDIM scheduler...")

    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )

    print("      PASS")

    # --------------------------------------------------------
    # Base Stable Diffusion pipeline
    # --------------------------------------------------------

    print()
    print("Building base diffusion pipeline...")

    base_pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )

    print("      Base pipeline ready.")

    # --------------------------------------------------------
    # Convert to RadEdit
    # --------------------------------------------------------

    print()
    print("Converting to RadEdit editing pipeline...")

    radedit_pipeline = DiffusionPipeline.from_pipe(
        base_pipeline,
        custom_pipeline=config.DIFFUSION_MODEL_ID,
        trust_remote_code=True,
    )

    radedit_pipeline = radedit_pipeline.to(device)

    print("      RadEdit pipeline ready.")

    # --------------------------------------------------------
    # Final tokenizer verification
    # --------------------------------------------------------

    print()
    print("Tokenizer verification:")
    print(
        f"  tokenizer.model_max_length = "
        f"{tokenizer.model_max_length}"
    )

    if tokenizer.model_max_length != 128:
        raise RuntimeError(
            "Unexpected tokenizer model_max_length. "
            f"Expected 128, found "
            f"{tokenizer.model_max_length}."
        )

    print("      PASS")

    print()
    print("=" * 70)
    print("RADEDIT PIPELINE LOADED SUCCESSFULLY")
    print("=" * 70)

    return radedit_pipeline, device


# ============================================================
# Manifest validation
# ============================================================

def validate_manifest(df):

    required = {
        "dicom_id",
        "subject_id",
        "study_id",
        "split",
        "disease_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    expected_total = int(
        config.DIFFUSION_REQUIRE_TOTAL_IMAGES
    )

    if len(df) != expected_total:
        raise ValueError(
            f"Expected {expected_total} manifest rows, "
            f"found {len(df)}."
        )

    disease_counts = (
        df["disease_status"]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    if config.DIFFUSION_REQUIRE_BALANCED_SOURCE:

        expected_normal = int(
            round(
                expected_total
                * float(config.NORMAL_FRACTION)
            )
        )

        expected_pathology = int(
            round(
                expected_total
                * float(config.PATHOLOGY_FRACTION)
            )
        )

        actual_normal = int(
            disease_counts.get("normal", 0)
        )

        actual_pathology = int(
            disease_counts.get("pathology", 0)
        )

        if actual_normal != expected_normal:
            raise ValueError(
                "Unexpected Normal source count: "
                f"expected {expected_normal}, "
                f"found {actual_normal}."
            )

        if actual_pathology != expected_pathology:
            raise ValueError(
                "Unexpected Pathology source count: "
                f"expected {expected_pathology}, "
                f"found {actual_pathology}."
            )

    print()
    print("=" * 70)
    print("MANIFEST VALIDATION")
    print("=" * 70)

    print(
        f"Total images: {len(df)}"
    )

    print(
        "Disease counts:"
    )

    print(
        disease_counts
        .sort_index()
        .to_string()
    )

    print()
    print("Split counts:")

    print(
        df["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Manifest validation: PASS")


# ============================================================
# Production generation
# ============================================================

def generate():

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest_path = Path(
        config.DIFFUSION_MANIFEST
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(
        manifest_path
    )

    validate_manifest(df)

    # --------------------------------------------------------
    # Edit masks
    # --------------------------------------------------------

    mask_root = Path(
        config.DIFFUSION_EDIT_MASK_ROOT
    )

    if not mask_root.exists():
        raise FileNotFoundError(
            f"Edit-mask directory not found: "
            f"{mask_root}\n"
            "Run diffusion_edit_masks.py first."
        )

    # --------------------------------------------------------
    # Output/report paths
    # --------------------------------------------------------

    metadata_path = Path(
        config.DIFFUSION_PRODUCTION_METADATA
    )

    error_path = Path(
        config.DIFFUSION_PRODUCTION_ERRORS
    )

    config_path = Path(
        config.DIFFUSION_PRODUCTION_CONFIG
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if metadata_path.exists():

        metadata = pd.read_csv(
            metadata_path
        )

        if "dicom_id" in metadata.columns:

            completed_ids = set(
                metadata["dicom_id"]
                .astype(str)
            )

        else:

            completed_ids = set()

    else:

        metadata = pd.DataFrame()
        completed_ids = set()

    # --------------------------------------------------------
    # Existing errors
    # --------------------------------------------------------

    if error_path.exists():

        errors = pd.read_csv(
            error_path
        )

    else:

        errors = pd.DataFrame(
            columns=[
                "dicom_id",
                "subject_id",
                "study_id",
                "split",
                "error",
            ]
        )

    # --------------------------------------------------------
    # Save exact production configuration
    # --------------------------------------------------------

    production_config = {
        "model":
            config.DIFFUSION_MODEL_ID,

        "vae_model":
            config.DIFFUSION_VAE_MODEL_ID,

        "text_model":
            config.DIFFUSION_TEXT_MODEL_ID,

        "tokenizer_model_max_length":
            128,

        "generation_resolution":
            config.DIFFUSION_GENERATION_SIZE,

        "final_resolution":
            config.DIFFUSION_FINAL_SIZE,

        "weight":
            config.DIFFUSION_WEIGHT,

        "num_inference_steps":
            config.DIFFUSION_NUM_INFERENCE_STEPS,

        "skip_ratio":
            config.DIFFUSION_SKIP_RATIO,

        "eta":
            config.DIFFUSION_ETA,

        "invert_prompt":
            config.DIFFUSION_INVERT_PROMPT,

        "normal_to_pathology_prompts":
            config.DIFFUSION_NORMAL_TO_PATHOLOGY_PROMPTS,

        "pathology_to_normal_prompts":
            config.DIFFUSION_PATHOLOGY_TO_NORMAL_PROMPTS,

        "master_seed":
            config.DIFFUSION_MASTER_SEED,

        "synthetic_label":
            1,

        "source_manifest":
            str(manifest_path),

        "edit_mask_directory":
            str(mask_root),

        "device":
            config.DIFFUSION_DEVICE,

        "dtype":
            config.DIFFUSION_DTYPE,

        # Edit-mask design: the images depend on the masks, so the masks are part of the settings.
        "edit_mask_target_area_fraction":
            config.DIFFUSION_TARGET_AREA_FRACTION,

        "edit_mask_metadata_md5":
            _file_md5(config.DIFFUSION_EDIT_MASK_METADATA),

        "generator_version":
            GENERATOR_VERSION,
    }

    # Resume safety: refuse to continue a run that was started with other settings
    # (different masks, prompts, seed, steps ...). Otherwise old and new images would be mixed.
    production_config["settings_fingerprint"] = _settings_fingerprint(production_config)
    if config_path.exists() and completed_ids:
        with open(config_path) as f:
            previous = json.load(f)
        if previous.get("settings_fingerprint") != production_config["settings_fingerprint"]:
            raise RuntimeError(
                "The diffusion settings changed since this run was started "
                f"({len(completed_ids)} images already generated).\n"
                "Resuming would mix images made with different settings. "
                "Delete the diffusion image folder, metadata and error files "
                "(or use a new run folder) and start again."
            )

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            production_config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Load RadEdit
    # --------------------------------------------------------

    radedit_pipeline, device = load_radedit()

    # --------------------------------------------------------
    # Generation timer
    # --------------------------------------------------------

    start_time = time.time()

    total_images = len(df)

    # --------------------------------------------------------
    # Generation loop
    # --------------------------------------------------------

    for idx, row in df.iterrows():

        dicom_id = str(
            row["dicom_id"]
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if dicom_id in completed_ids:
            continue

        disease = str(
            row["disease_status"]
        ).lower()

        split = str(
            row["split"]
        ).lower()

        # ----------------------------------------------------
        # Determine direction
        # ----------------------------------------------------

        if disease == "normal":

            direction = "Normal_to_Pathology"

            prompt_bank = (
                config.DIFFUSION_NORMAL_TO_PATHOLOGY_PROMPTS
            )

        elif disease == "pathology":

            direction = "Pathology_to_Normal"

            prompt_bank = (
                config.DIFFUSION_PATHOLOGY_TO_NORMAL_PROMPTS
            )

        else:

            error_row = {
                "dicom_id": dicom_id,
                "subject_id": str(
                    row["subject_id"]
                ),
                "study_id": str(
                    row["study_id"]
                ),
                "split": split,
                "error": (
                    f"Unsupported disease_status: "
                    f"{disease}"
                ),
            }

            errors = pd.concat(
                [
                    errors,
                    pd.DataFrame([error_row]),
                ],
                ignore_index=True,
            )

            errors.to_csv(
                error_path,
                index=False,
            )

            print(
                f"ERROR [{dicom_id}]: "
                f"unsupported disease status "
                f"{disease}"
            )

            continue

        # ----------------------------------------------------
        # HTML-style deterministic seed
        # ----------------------------------------------------

        seed = make_seed(
            idx,
            config.DIFFUSION_WEIGHT,
        )

        # ----------------------------------------------------
        # Deterministic prompt selection
        # ----------------------------------------------------

        prompt_index = (
            seed % len(prompt_bank)
        )

        prompt = prompt_bank[
            prompt_index
        ]

        # ----------------------------------------------------
        # Source image
        # ----------------------------------------------------

        src = source_path(row)

        # ----------------------------------------------------
        # Edit mask
        # ----------------------------------------------------

        mask_path = (
            mask_root
            / f"{dicom_id}_edit_mask.png"
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        output_dir = (
            Path(config.DIFFUSION_IMAGE_ROOT)
            / split
            / direction
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"{dicom_id}.png"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            # ------------------------------------------------
            # Check source
            # ------------------------------------------------

            if not src.exists():

                raise FileNotFoundError(
                    f"Missing source image: {src}"
                )

            # ------------------------------------------------
            # Check mask
            # ------------------------------------------------

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"Missing edit mask: {mask_path}"
                )

            # ------------------------------------------------
            # Load source
            # ------------------------------------------------

            source_image = (
                Image.open(src)
                .convert("RGB")
                .resize(
                    (
                        config.DIFFUSION_GENERATION_SIZE,
                        config.DIFFUSION_GENERATION_SIZE,
                    ),
                    Image.Resampling.LANCZOS,
                )
            )

            # ------------------------------------------------
            # Load edit mask
            # ------------------------------------------------

            edit_mask = (
                Image.open(mask_path)
                .convert("L")
                .resize(
                    (
                        config.DIFFUSION_GENERATION_SIZE,
                        config.DIFFUSION_GENERATION_SIZE,
                    ),
                    Image.Resampling.NEAREST,
                )
            )

            # ------------------------------------------------
            # Keep mask = inverse edit mask
            # ------------------------------------------------

            keep_mask = ImageOps.invert(
                edit_mask
            )

            # ------------------------------------------------
            # Set seeds
            # ------------------------------------------------

            set_all_seeds(
                seed
            )

            # ------------------------------------------------
            # Print generation information
            # ------------------------------------------------

            print()
            print(
                f"[{idx + 1}/{total_images}] "
                f"{direction}"
            )

            print(
                f"  DICOM       : {dicom_id}"
            )

            print(
                f"  Split       : {split}"
            )

            print(
                f"  Seed        : {seed}"
            )

            print(
                f"  Prompt      : {prompt}"
            )

            print(
                f"  Weight      : "
                f"{config.DIFFUSION_WEIGHT}"
            )

            print(
                f"  Steps       : "
                f"{config.DIFFUSION_NUM_INFERENCE_STEPS}"
            )

            print(
                f"  Skip ratio  : "
                f"{config.DIFFUSION_SKIP_RATIO}"
            )

            # ------------------------------------------------
            # Run RadEdit
            # ------------------------------------------------

            with torch.inference_mode():

                result = radedit_pipeline(
                    prompt=prompt,

                    image=source_image,

                    edit_mask=edit_mask,

                    keep_mask=keep_mask,

                    weights=[
                        config.DIFFUSION_WEIGHT
                    ],

                    invert_prompt=(
                        config.DIFFUSION_INVERT_PROMPT
                    ),

                    num_inference_steps=(
                        config.DIFFUSION_NUM_INFERENCE_STEPS
                    ),

                    skip_ratio=(
                        config.DIFFUSION_SKIP_RATIO
                    ),

                    eta=(
                        config.DIFFUSION_ETA
                    ),

                    prog_bar=False,

                    output_type="pil",
                )

            # ------------------------------------------------
            # Extract result
            # ------------------------------------------------

            if isinstance(result, list):

                generated = result[0]

            elif hasattr(
                result,
                "images",
            ):

                generated = result.images[0]

            else:

                raise RuntimeError(
                    "Unexpected RadEdit output type: "
                    f"{type(result)}"
                )

            # ------------------------------------------------
            # Final resolution
            # ------------------------------------------------

            generated = (
                generated
                .convert("RGB")
                .resize(
                    (
                        config.DIFFUSION_FINAL_SIZE,
                        config.DIFFUSION_FINAL_SIZE,
                    ),
                    Image.Resampling.LANCZOS,
                )
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            generated.save(
                output_path,
                format="PNG",
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            record = {
                "dicom_id":
                    dicom_id,

                "subject_id":
                    str(row["subject_id"]),

                "study_id":
                    str(row["study_id"]),

                "split":
                    split,

                "source_disease_status":
                    disease,

                "direction":
                    direction,

                "prompt":
                    prompt,

                "prompt_index":
                    prompt_index + 1,

                "seed":
                    seed,

                "weight":
                    config.DIFFUSION_WEIGHT,

                "num_inference_steps":
                    config.DIFFUSION_NUM_INFERENCE_STEPS,

                "skip_ratio":
                    config.DIFFUSION_SKIP_RATIO,

                "eta":
                    config.DIFFUSION_ETA,

                "invert_prompt":
                    config.DIFFUSION_INVERT_PROMPT,

                "generation_resolution":
                    config.DIFFUSION_GENERATION_SIZE,

                "final_resolution":
                    config.DIFFUSION_FINAL_SIZE,

                "edit_area_fraction":
                    float((np.asarray(edit_mask) > 127).mean()),

                "generator_version":
                    GENERATOR_VERSION,

                "edit_mask":
                    str(mask_path),

                "output_path":
                    str(output_path),

                "synthetic_label":
                    1,
            }

            metadata = pd.concat(
                [
                    metadata,
                    pd.DataFrame([record]),
                ],
                ignore_index=True,
            )

            metadata.to_csv(
                metadata_path,
                index=False,
            )

            completed_ids.add(
                dicom_id
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            elapsed = (
                time.time() - start_time
            ) / 60

            print(
                f"  SUCCESS "
                f"({len(completed_ids)}/{total_images}) "
                f"elapsed={elapsed:.1f} min"
            )

        # ----------------------------------------------------
        # Error handling
        # ----------------------------------------------------

        except Exception as exc:

            error_row = {
                "dicom_id":
                    dicom_id,

                "subject_id":
                    str(row["subject_id"]),

                "study_id":
                    str(row["study_id"]),

                "split":
                    split,

                "error":
                    repr(exc),
            }

            errors = pd.concat(
                [
                    errors,
                    pd.DataFrame([error_row]),
                ],
                ignore_index=True,
            )

            errors.to_csv(
                error_path,
                index=False,
            )

            print()
            print(
                f"ERROR [{dicom_id}]: "
                f"{repr(exc)}"
            )

            print(
                "Full traceback:"
            )

            traceback.print_exc()

            print(
                "-" * 70
            )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("RadEdit production generation finished.")
    print("=" * 70)

    if metadata_path.exists():

        final_meta = pd.read_csv(
            metadata_path
        )

    else:

        final_meta = pd.DataFrame()

    print(
        f"Successful images: {len(final_meta)}"
    )

    print(
        f"Errors:            {len(errors)}"
    )

    print(
        f"Metadata:          {metadata_path}"
    )

    print(
        f"Errors:            {error_path}"
    )


if __name__ == "__main__":
    generate()