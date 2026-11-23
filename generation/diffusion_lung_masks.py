"""
Prepare NIAID CXR lung masks for RadEdit production.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
from PIL import Image

import config


def _load_niaid():
    repo = Path(config.DIFFUSION_NIAID_REPO)
    if not repo.exists():
        raise FileNotFoundError(
            f"NIAID repository not found: {repo}\n"
            f"Clone it from: {config.DIFFUSION_NIAID_REPO_URL}"
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from segment_lung_cxr.inference.inference_lung_segment import (
        _load_model,
        _predict_prob_model_input_size,
        _read_image,
        _resample_prob_to_original_size,
        postprocess_segmentation,
    )
    return (
        _load_model,
        _predict_prob_model_input_size,
        _read_image,
        _resample_prob_to_original_size,
        postprocess_segmentation,
    )


def source_path(row):
    disease = str(row["disease_status"]).lower()
    split = str(row["split"]).lower()
    subject = str(row["subject_id"])
    study = str(row["study_id"])
    dicom = str(row["dicom_id"])

    return (
        Path(config.DIFFUSION_SOURCE_ROOT)
        / split
        / disease
        / f"p{subject}_s{study}_{dicom}.jpg"
    )


def generate_lung_masks():
    required = {
        "dicom_id", "subject_id", "study_id",
        "split", "disease_status",
    }

    manifest_path = Path(config.DIFFUSION_MANIFEST)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")

    if len(df) != config.DIFFUSION_REQUIRE_TOTAL_IMAGES:
        raise ValueError(
            f"Expected {config.DIFFUSION_REQUIRE_TOTAL_IMAGES} images, "
            f"found {len(df)}."
        )

    out_root = Path(config.DIFFUSION_LUNG_MASK_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = Path(config.DIFFUSION_LUNG_MASK_METADATA)
    error_path = Path(config.DIFFUSION_LUNG_MASK_ERRORS)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    (
        load_model,
        predict_prob_model_input_size,
        read_image,
        resample_prob_to_original_size,
        postprocess_segmentation,
    ) = _load_niaid()

    checkpoint = Path(config.DIFFUSION_NIAID_CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"NIAID checkpoint not found: {checkpoint}"
        )

    model, device = load_model(checkpoint)

    records = []
    errors = []

    for i, row in df.reset_index(drop=True).iterrows():
        dicom_id = str(row["dicom_id"])
        src = source_path(row)
        output = out_root / f"{dicom_id}_lung_mask.png"

        try:
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")

            original_img = read_image(str(src))

            probability_map = predict_prob_model_input_size(
                original_img,
                model,
                device,
                model_input_size=[config.DIFFUSION_MASK_SIZE] * 2,
                patch_size=[config.DIFFUSION_MASK_SIZE] * 2,
            )

            probability_original = resample_prob_to_original_size(
                probability_map,
                original_img,
            )

            binary_mask = probability_original > 0.5
            binary_mask = postprocess_segmentation(binary_mask)

            # Match the HTML: convert to array and resize to 512x512.
            mask_array = sitk.GetArrayFromImage(binary_mask).astype(np.uint8)

            mask_512 = np.asarray(
                Image.fromarray(mask_array * 255).resize(
                    (config.DIFFUSION_MASK_SIZE, config.DIFFUSION_MASK_SIZE),
                    Image.Resampling.NEAREST,
                )
            )
            mask_binary = mask_512 > 0

            Image.fromarray(
                (mask_binary * 255).astype(np.uint8)
            ).save(output)

            records.append({
                "dicom_id": dicom_id,
                "subject_id": str(row["subject_id"]),
                "study_id": str(row["study_id"]),
                "split": row["split"],
                "disease_status": row["disease_status"],
                "view_position": row.get("ViewPosition", ""),
                "lung_mask_path": str(output),
                "lung_area_512": int(mask_binary.sum()),
            })

        except Exception as exc:
            errors.append({
                "dicom_id": dicom_id,
                "subject_id": str(row["subject_id"]),
                "study_id": str(row["study_id"]),
                "error": repr(exc),
            })

        if (i + 1) % 25 == 0:
            print(
                f"{i + 1}/{len(df)} processed | "
                f"success={len(records)} | errors={len(errors)}"
            )

    pd.DataFrame(records).to_csv(meta_path, index=False)
    pd.DataFrame(errors).to_csv(error_path, index=False)

    print("\nLung-mask generation complete.")
    print(f"Successful: {len(records)}")
    print(f"Errors:     {len(errors)}")
    print(f"Metadata:   {meta_path}")
    print(f"Errors:     {error_path}")


if __name__ == "__main__":
    generate_lung_masks()
