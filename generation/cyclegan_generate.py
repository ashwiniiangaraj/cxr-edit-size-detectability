
"""
Modular CycleGAN production generation.

Loads the checkpoint produced by cyclegan_train.py and generates:
    Normal    -> Pathology
    Pathology -> Normal
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from PIL import Image

import torch

# Make 'import gen_common' / 'import cyclegan_train' work under %run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_common   # shared loader / PNG saver / stale-file guard

import config
from cyclegan_train import Generator


def get_device() -> torch.device:
    if (
        config.CYCLEGAN_DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device(
            f"cuda:{config.CYCLEGAN_GPU_INDEX}"
        )

    if config.CYCLEGAN_DEVICE == "cuda":
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_index() -> dict:
    """
    FREEZE FIX (speed): build ONE lookup {dicom_id: [paths]} of the source
    folder the first time it is needed. The old code called rglob over the whole
    folder for every image, which is O(N^2) and very slow at 10-20k images.
    File names are p<subject>_s<study>_<dicom>.jpg and a DICOM id contains no
    underscore, so the id is the text after the last underscore.
    """
    global _SRC_INDEX
    if _SRC_INDEX is None:
        _SRC_INDEX = {}
        for p in Path(config.REAL_DIR).rglob("*.jpg"):
            _SRC_INDEX.setdefault(p.stem.rsplit("_", 1)[-1], []).append(p)
    return _SRC_INDEX


_SRC_INDEX = None


def resolve_source_image(row: pd.Series) -> Path:
    dicom_id = str(row["dicom_id"])

    matches = list(_source_index().get(dicom_id, []))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple real images for {dicom_id}: "
            f"{matches}"
        )

    split = str(row["split"])
    disease = str(row["disease_status"])

    candidate = (
        config.REAL_DIR
        / split
        / disease
        / f"p{row['subject_id']}_"
          f"s{row['study_id']}_"
          f"{dicom_id}.jpg"
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Real source image not found for {dicom_id}"
    )


def load_tensor(path: Path):
    """
    path -> (tensor, source_array).

    tensor       : (1, 3, 256, 256) in [-1, 1], network input
    source_array : the same source image as uint8 (256, 256), kept so the
                   edit strength can be measured after generation
    """
    image = gen_common.load_gray_image(
        path,
        config.CYCLEGAN_IMAGE_SIZE,
    )

    source_uint8 = np.asarray(image, dtype=np.uint8).copy()

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(array)[None, :, :]
    tensor = tensor.repeat(3, 1, 1)
    tensor = tensor * 2.0 - 1.0

    return tensor.unsqueeze(0), source_uint8


def tensor_to_grayscale_image(
    tensor: torch.Tensor,
) -> Image.Image:

    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0

    # Average the three output channels. They are generated from
    # grayscale input but the model operates in 3-channel space.
    tensor = tensor.mean(dim=1)[0]

    array = (
        tensor.numpy() * 255.0
    ).round().astype(np.uint8)

    # 2-D uint8 array -> 8-bit grayscale image (no mode= argument).
    return Image.fromarray(array)


def load_generators(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = {
        "G_AB",
        "G_BA",
        "epoch",
    }

    missing = required - set(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint missing keys: {sorted(missing)}"
        )

    G_AB = Generator(
        input_channels=config.CYCLEGAN_INPUT_NC,
        output_channels=config.CYCLEGAN_OUTPUT_NC,
        n_residual_blocks=(
            config.CYCLEGAN_GENERATOR_RES_BLOCKS
        ),
        base_channels=(
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS
        ),
    ).to(device)

    G_BA = Generator(
        input_channels=config.CYCLEGAN_INPUT_NC,
        output_channels=config.CYCLEGAN_OUTPUT_NC,
        n_residual_blocks=(
            config.CYCLEGAN_GENERATOR_RES_BLOCKS
        ),
        base_channels=(
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS
        ),
    ).to(device)

    G_AB.load_state_dict(
        checkpoint["G_AB"],
        strict=True,
    )

    G_BA.load_state_dict(
        checkpoint["G_BA"],
        strict=True,
    )

    G_AB.eval()
    G_BA.eval()

    print(
        f"Loaded CycleGAN checkpoint epoch "
        f"{checkpoint['epoch']}"
    )
    print("G_AB: Normal -> Pathology")
    print("G_BA: Pathology -> Normal")

    return G_AB, G_BA


def validate_manifest(df):
    required_columns = {
        "dicom_id",
        "subject_id",
        "study_id",
        "disease_status",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    total = len(df)

    split_counts = (
        df["split"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_fractions = {
        "train": config.TRAIN_FRACTION,
        "val": config.VAL_FRACTION,
        "test": config.TEST_FRACTION,
    }

    for split, fraction in expected_fractions.items():

        actual_count = split_counts.get(split, 0)
        expected_count = total * fraction

        if actual_count != expected_count:
            raise ValueError(
                f"Unexpected {split} count: "
                f"{actual_count}; expected {expected_count:.0f} "
                f"from configured fraction {fraction:.2f}"
            )

    disease_counts = (
        df["disease_status"]
        .value_counts()
        .to_dict()
    )

    print("Manifest validation passed.")
    print(f"  Total images: {total}")
    print(f"  Splits: {split_counts}")
    print(f"  Disease: {disease_counts}")


def generate_all() -> None:

    seed_everything(
        config.CYCLEGAN_MASTER_SEED
    )

    device = get_device()

    print("=" * 72)
    print("CYCLEGAN PRODUCTION GENERATION")
    print("=" * 72)

    df = pd.read_csv(
        config.CYCLEGAN_MANIFEST
    )

    validate_manifest(df)

    G_AB, G_BA = load_generators(
        config.CYCLEGAN_FINAL_CHECKPOINT,
        device,
    )

    out_AtoB = (
        config.CYCLEGAN_OUTPUT_NORMAL_TO_PATHOLOGY
    )
    out_BtoA = (
        config.CYCLEGAN_OUTPUT_PATHOLOGY_TO_NORMAL
    )

    out_AtoB.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_BtoA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Refuse to mix new PNGs with old JPEGs from a previous run.
    gen_common.ensure_clean_image_dir(out_AtoB, "cyclegan normal->pathology")
    gen_common.ensure_clean_image_dir(out_BtoA, "cyclegan pathology->normal")

    report_dir = (
        config.CYCLEGAN_OUTPUT_ROOT / "reports"
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []
    errors = []

    normal_df = df[
        df["disease_status"].astype(str).str.lower()
        == "normal"
    ].copy()

    pathology_df = df[
        df["disease_status"].astype(str).str.lower()
        == "pathology"
    ].copy()

    _half = config.POOL_SIZE_PER_POOL // 2
    if len(normal_df) != _half or len(pathology_df) != _half:
        raise RuntimeError(
            f"Production source pools are not {_half}/{_half}."
        )

    # --------------------------------------------------------
    # Normal -> Pathology
    # --------------------------------------------------------
    for index, (_, row) in enumerate(
        normal_df.iterrows(),
        start=1,
    ):

        source_path = None

        try:
            source_path = resolve_source_image(row)

            tensor, source_uint8 = load_tensor(
                source_path
            )
            tensor = tensor.to(device)

            with torch.inference_mode():
                generated = G_AB(tensor)

            image = tensor_to_grayscale_image(
                generated
            )

            output_name = (
                f"normal_to_pathology_"
                f"{index:03d}_"
                f"p{row['subject_id']}_"
                f"s{row['study_id']}_"
                f"{row['dicom_id']}{gen_common.IMAGE_EXT}"
            )

            output_path = out_AtoB / output_name

            # Lossless PNG with a read-back pixel check.
            generated_uint8 = np.asarray(image, dtype=np.uint8)
            gen_common.save_gray_png(
                generated_uint8,
                output_path,
            )

            strength = gen_common.edit_strength(
                source_uint8,
                generated_uint8,
            )

            metadata.append({
                "synthetic_id": output_path.stem,
                "source_dicom_id": row["dicom_id"],
                "source_subject_id": row["subject_id"],
                "source_study_id": row["study_id"],
                "source_disease_status": "normal",
                "target_domain": "pathology",
                "direction": "normal_to_pathology",
                "source_split": row["split"],
                "source_path": str(source_path),
                "synthetic_path": str(output_path),
                "checkpoint_epoch":
                    config.CYCLEGAN_EPOCHS,
                "resize_filter": gen_common.RESIZE_FILTER_NAME,
                "output_format": gen_common.IMAGE_FORMAT,
                "mean_abs_diff_image":
                    strength["mean_abs_diff_image"],
            })

        except Exception as exc:
            errors.append({
                "source_dicom_id": row["dicom_id"],
                "direction": "normal_to_pathology",
                "error": str(exc),
                "source_path": (
                    str(source_path)
                    if source_path else ""
                ),
            })

    # --------------------------------------------------------
    # Pathology -> Normal
    # --------------------------------------------------------
    for index, (_, row) in enumerate(
        pathology_df.iterrows(),
        start=1,
    ):

        source_path = None

        try:
            source_path = resolve_source_image(row)

            tensor, source_uint8 = load_tensor(
                source_path
            )
            tensor = tensor.to(device)

            with torch.inference_mode():
                generated = G_BA(tensor)

            image = tensor_to_grayscale_image(
                generated
            )

            output_name = (
                f"pathology_to_normal_"
                f"{index:03d}_"
                f"p{row['subject_id']}_"
                f"s{row['study_id']}_"
                f"{row['dicom_id']}{gen_common.IMAGE_EXT}"
            )

            output_path = out_BtoA / output_name

            # Lossless PNG with a read-back pixel check.
            generated_uint8 = np.asarray(image, dtype=np.uint8)
            gen_common.save_gray_png(
                generated_uint8,
                output_path,
            )

            strength = gen_common.edit_strength(
                source_uint8,
                generated_uint8,
            )

            metadata.append({
                "synthetic_id": output_path.stem,
                "source_dicom_id": row["dicom_id"],
                "source_subject_id": row["subject_id"],
                "source_study_id": row["study_id"],
                "source_disease_status": "pathology",
                "target_domain": "normal",
                "direction": "pathology_to_normal",
                "source_split": row["split"],
                "source_path": str(source_path),
                "synthetic_path": str(output_path),
                "checkpoint_epoch":
                    config.CYCLEGAN_EPOCHS,
                "resize_filter": gen_common.RESIZE_FILTER_NAME,
                "output_format": gen_common.IMAGE_FORMAT,
                "mean_abs_diff_image":
                    strength["mean_abs_diff_image"],
            })

        except Exception as exc:
            errors.append({
                "source_dicom_id": row["dicom_id"],
                "direction": "pathology_to_normal",
                "error": str(exc),
                "source_path": (
                    str(source_path)
                    if source_path else ""
                ),
            })

    metadata_df = pd.DataFrame(metadata)
    errors_df = pd.DataFrame(errors)

    metadata_df.to_csv(
        config.CYCLEGAN_GENERATION_METADATA,
        index=False,
    )

    errors_df.to_csv(
        config.CYCLEGAN_GENERATION_ERRORS,
        index=False,
    )

    generation_config = {
        "checkpoint": str(
            config.CYCLEGAN_FINAL_CHECKPOINT
        ),
        "checkpoint_epoch":
            config.CYCLEGAN_EPOCHS,
        "manifest": str(
            config.CYCLEGAN_MANIFEST
        ),
        "output_root": str(
            config.CYCLEGAN_OUTPUT_ROOT
        ),
        "normal_to_pathology_count": int(
            (metadata_df["direction"]
             == "normal_to_pathology").sum()
        ) if not metadata_df.empty else 0,
        "pathology_to_normal_count": int(
            (metadata_df["direction"]
             == "pathology_to_normal").sum()
        ) if not metadata_df.empty else 0,
        "total_generated": len(metadata_df),
        "total_errors": len(errors_df),
        "image_size": config.CYCLEGAN_IMAGE_SIZE,
        "resize_filter": gen_common.RESIZE_FILTER_NAME,
        "output_format": gen_common.IMAGE_FORMAT,
        "seed": config.CYCLEGAN_MASTER_SEED,
    }

    config.CYCLEGAN_GENERATION_CONFIG.write_text(
        json.dumps(
            generation_config,
            indent=2,
        )
    )

    print("\n" + "=" * 72)
    print("CYCLEGAN GENERATION COMPLETE")
    print("=" * 72)
    print(
        f"Generated images: {len(metadata_df)}"
    )
    print(
        f"Errors: {len(errors_df)}"
    )
    print(
        f"Metadata: "
        f"{config.CYCLEGAN_GENERATION_METADATA}"
    )
    print(
        f"Errors: "
        f"{config.CYCLEGAN_GENERATION_ERRORS}"
    )


if __name__ == "__main__":
    generate_all()
