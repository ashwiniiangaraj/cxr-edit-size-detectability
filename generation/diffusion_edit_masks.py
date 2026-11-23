"""
Create the focal RadEdit edit masks (production).

Mode is set by config.DIFFUSION_TARGET_AREA_FRACTION:
    a number (default 0.05): each mask is a lung-clipped ellipse covering that share of the 512 x 512 canvas
                             (same recipe as the size ladder, diffusion_size_ladder.py)
    None:                    legacy small ellipse sized from the lung width (~0.1-0.2 % of the image)

Images whose lungs cannot reach the target area are listed in the errors file.

"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config

S = int(config.DIFFUSION_MASK_SIZE)                          
_YY, _XX = np.mgrid[0:S, 0:S]


def source_row_seed(row_index):
    return config.DIFFUSION_MASTER_SEED + row_index


# ----------------------------------------------------------------------------
# shared geometry (area-targeted mode and size ladder)
# ----------------------------------------------------------------------------
def safe_lungs(lung512, margin=None):
    """Two largest lung components, eroded by `margin` px. Returns a 0/1 uint8 image."""
    margin = int(config.DIFFUSION_BOUNDARY_MARGIN if margin is None else margin)
    n, lab, st, _ = cv2.connectedComponentsWithStats((lung512 > 0).astype(np.uint8), connectivity=8)
    comps = sorted([(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)[:2]
    if len(comps) < 2:
        raise ValueError("fewer than two lung components")
    union = np.isin(lab, [c[1] for c in comps]).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.erode(union, kernel)


def clipped_ellipse(safe, cy, cx, width, aspect=None):
    """Ellipse (height = aspect x width) centred on (cy, cx), clipped to the safe lung. 0/1 uint8."""
    aspect = float(config.DIFFUSION_REGION_HEIGHT_FRACTION if aspect is None else aspect)
    a, b = width / 2.0, max(width * aspect / 2.0, 0.5)
    return ((((_XX - cx) / a) ** 2 + ((_YY - cy) / b) ** 2) <= 1.0).astype(np.uint8) & safe


def mask_for_target(safe, frac, rng, tol=None, tries=None, margin=None):
    """Ellipse centred on an interior lung pixel, scaled by bisection until its CLIPPED area = frac x canvas.

    Returns the 0/1 mask. Raises ValueError if no tried centre reaches the target within `tol` (relative).
    """
    tol = float(config.DIFFUSION_AREA_TOLERANCE if tol is None else tol)
    tries = int(config.DIFFUSION_AREA_TRIES if tries is None else tries)
    margin = float(config.DIFFUSION_BOUNDARY_MARGIN if margin is None else margin)
    target = frac * safe.size
    if safe.sum() < target * (1 + tol):
        raise ValueError(f"safe lung area {int(safe.sum())} < target {target:.0f}")
    dist = cv2.distanceTransform(safe, cv2.DIST_L2, 5)
    cand = np.argwhere(dist >= max(margin, 0.45 * dist.max()))
    if not len(cand):
        raise ValueError("no interior centre")
    for k in rng.permutation(len(cand))[:tries]:
        cy, cx = cand[k]
        if clipped_ellipse(safe, cy, cx, 4 * S).sum() < target * (1 + tol):
            continue                                          # this centre cannot reach the target
        lo, hi = 2.0, 4.0 * S
        for _ in range(30):
            mid = (lo + hi) / 2
            if clipped_ellipse(safe, cy, cx, mid).sum() < target:
                lo = mid
            else:
                hi = mid
        m = clipped_ellipse(safe, cy, cx, hi)
        if abs(m.sum() - target) / target <= tol:
            return m
    raise ValueError("no centre reached the target area")


# ----------------------------------------------------------------------------
# production masks
# ----------------------------------------------------------------------------
def create_targeted_edit_mask(lung_mask, rng, frac):
    """AREA-TARGETED mode. Returns (edit_mask, safe_lung, region_width, region_height)."""
    safe = safe_lungs(lung_mask)
    edit_mask = mask_for_target(safe, frac, rng).astype(np.uint8)
    xs = np.where(edit_mask.any(axis=0))[0]
    width = int(xs.max() - xs.min() + 1)
    height = int(round(width * float(config.DIFFUSION_REGION_HEIGHT_FRACTION)))
    return edit_mask, safe, width, height


def create_focal_edit_mask(lung_mask, rng):
    """LEGACY mode (config.DIFFUSION_TARGET_AREA_FRACTION = None): small ellipse in ONE lung.

    Returns (edit_mask, selected_lung, safe_lung, region_width, region_height).
    """
    lung_binary = (lung_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lung_binary, connectivity=8)

    components = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0]
    if len(components) < 2:
        raise ValueError("Could not identify two lung components.")
    components = sorted(components, reverse=True)[:2]
    lung_labels = [components[0][1], components[1][1]]

    selected_label = int(rng.choice(lung_labels))
    selected_lung = (labels == selected_label).astype(np.uint8)

    margin = int(config.DIFFUSION_BOUNDARY_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    safe_lung = cv2.erode(selected_lung, kernel)
    if safe_lung.sum() == 0:
        raise ValueError("Safe lung became empty after erosion.")

    distance = cv2.distanceTransform(safe_lung, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        raise ValueError("Could not find an interior lung region.")

    candidate_pixels = np.argwhere(distance >= max(float(margin), 0.45 * max_distance))
    if len(candidate_pixels) == 0:
        raise ValueError("No valid interior candidate pixels.")
    cy, cx = candidate_pixels[rng.integers(0, len(candidate_pixels))]

    ys, xs = np.where(selected_lung > 0)
    lung_width = int(xs.max() - xs.min() + 1)
    region_width = max(int(config.DIFFUSION_MIN_REGION_WIDTH),
                       int(round(config.DIFFUSION_REGION_WIDTH_FRACTION * lung_width)))
    region_height = max(int(config.DIFFUSION_MIN_REGION_HEIGHT),
                        int(round(config.DIFFUSION_REGION_HEIGHT_FRACTION * region_width)))

    edit_mask = np.zeros_like(selected_lung, dtype=np.uint8)
    cv2.ellipse(edit_mask, (int(cx), int(cy)), (max(1, region_width // 2), max(1, region_height // 2)), 0, 0, 360, 1, -1)
    edit_mask = (edit_mask & safe_lung).astype(np.uint8)

    edit_area = int(edit_mask.sum())
    if edit_area < config.DIFFUSION_MIN_EDIT_AREA:
        raise ValueError(f"Edit region too small: {edit_area} pixels.")
    return edit_mask, selected_lung, safe_lung, region_width, region_height


def generate_edit_masks():
    manifest_path = Path(config.DIFFUSION_MANIFEST)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if len(df) != config.DIFFUSION_REQUIRE_TOTAL_IMAGES:
        raise ValueError(f"Expected {config.DIFFUSION_REQUIRE_TOTAL_IMAGES} images, found {len(df)}.")

    target = config.DIFFUSION_TARGET_AREA_FRACTION
    mode = "area-targeted" if target is not None else "legacy single-lung"
    print(f"Edit-mask mode: {mode}" + (f" (target {100 * target:.2f}% of the {S}x{S} canvas)" if target is not None else ""))

    lung_root = Path(config.DIFFUSION_LUNG_MASK_ROOT)
    edit_root = Path(config.DIFFUSION_EDIT_MASK_ROOT)
    edit_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(config.DIFFUSION_EDIT_MASK_METADATA)
    errors_path = Path(config.DIFFUSION_EDIT_MASK_ERRORS)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    records, errors = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        dicom_id = str(row["dicom_id"])
        lung_path = lung_root / f"{dicom_id}_lung_mask.png"
        edit_path = edit_root / f"{dicom_id}_edit_mask.png"
        rng = np.random.default_rng(source_row_seed(i))

        try:
            if not lung_path.exists():
                raise FileNotFoundError(f"Missing lung mask: {lung_path}")
            lung_mask = (np.asarray(Image.open(lung_path).convert("L")) > 0).astype(np.uint8)
            lung_mask_512 = cv2.resize(lung_mask, (S, S), interpolation=cv2.INTER_NEAREST)

            if target is not None:
                edit_mask, safe_lung, region_width, region_height = create_targeted_edit_mask(lung_mask_512, rng, float(target))
                selected_area = int(lung_mask_512.sum())
            else:
                edit_mask, selected_lung, safe_lung, region_width, region_height = create_focal_edit_mask(lung_mask_512, rng)
                selected_area = int(selected_lung.sum())

            Image.fromarray(edit_mask.astype(np.uint8) * 255).save(edit_path)
            records.append({
                "dicom_id": dicom_id,
                "subject_id": str(row["subject_id"]),
                "study_id": str(row["study_id"]),
                "split": row["split"],
                "disease_status": row["disease_status"],
                "seed": source_row_seed(i),
                "lung_mask_path": str(lung_path),
                "edit_mask_path": str(edit_path),
                "selected_lung_area": selected_area,
                "safe_lung_area": int(safe_lung.sum()),
                "edit_area": int(edit_mask.sum()),
                "edit_area_fraction": float(edit_mask.sum() / edit_mask.size),
                "target_area_fraction": float(target) if target is not None else np.nan,
                "edit_fraction_of_safe_lung": float(edit_mask.sum() / max(1, safe_lung.sum())),
                "edit_width_target": int(region_width),
                "edit_height_target": int(region_height),
                "boundary_margin_px": int(config.DIFFUSION_BOUNDARY_MARGIN),
            })
        except Exception as exc:                                    # noqa: BLE001 - recorded, never silent
            errors.append({"dicom_id": dicom_id, "subject_id": str(row["subject_id"]),
                           "study_id": str(row["study_id"]), "error": repr(exc)})

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(df)} processed | success={len(records)} | errors={len(errors)}")

    rec = pd.DataFrame(records)
    rec.to_csv(metadata_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    print("\nFocal edit-mask generation complete.")
    print(f"Successful: {len(records)}   Errors: {len(errors)}")
    if len(rec):
        f = rec["edit_area_fraction"]
        print(f"Edit area: median {100 * f.median():.2f}%  range {100 * f.min():.2f}-{100 * f.max():.2f}% of the image")
    print(f"Metadata:   {metadata_path}\nErrors:     {errors_path}")


if __name__ == "__main__":
    generate_edit_masks()
