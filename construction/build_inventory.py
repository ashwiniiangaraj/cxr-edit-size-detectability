"""
ORGANISE DATASET
===========================
Scans every image family and builds ONE master table (dataset_master.csv):

    image_id, dataset_key, group, frequency_group, generator, label (0 real / 1 synthetic),
    split, dicom_id, subject_id, study_id, source_disease_status, direction,
    folder_split, native_width/height, file_ext, file_size_bytes, filename, path
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

import config
from utils.common import ensure_parent, parse_dicom_id, parse_subject_study


def _is_excluded(path: Path, root: Path) -> bool:
    """True for mask / report / config files that must not be treated as images."""
    rel_parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    if any(p in config.DATASET_EXCLUDE_DIR_NAMES for p in rel_parts):
        return True
    return path.stem.lower().endswith(tuple(config.DATASET_EXCLUDE_FILE_SUFFIXES))


def scan_family(key: str, spec: dict) -> list[dict]:
    """List every valid image file of one family."""
    root = Path(spec["root"])
    if not root.exists():
        raise FileNotFoundError(f"[{key}] folder not found: {root}")

    exts = {e.lower() for e in spec["extensions"]}
    pattern = root.rglob("*") if spec.get("recursive", True) else root.glob("*")

    rows = []
    for p in sorted(pattern):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if _is_excluded(p, root):
            continue

        rel = [x.lower() for x in p.relative_to(root).parts[:-1]]
        folder_split = next((x for x in rel if x in config.SPLIT_NAMES), None)
        direction = next((x for x in rel if "_to_" in x), None)
        subject, study = parse_subject_study(p.name)

        with Image.open(p) as im:
            width, height = im.size

        rows.append({
            "dataset_key": key, "group": spec["group"],
            "frequency_group": spec["frequency_group"], "generator": spec["generator"],
            "label": spec.get("label", 0 if spec["group"] == config.REAL_GROUP else 1),
            "path": str(p), "filename": p.name, "dicom_id": parse_dicom_id(p.name),
            "subject_id": subject, "study_id": study, "folder_split": folder_split,
            "direction": direction, "native_width": width, "native_height": height,
            "file_ext": p.suffix.lower(), "file_size_bytes": p.stat().st_size,
        })
    return rows


def build_master_manifest(save: bool = True) -> pd.DataFrame:
    """Scan all families, attach splits and write dataset_master.csv."""
    print("Scanning image families ...")
    rows = []
    for key, spec in config.FORENSIC_CHARACTERISATION_DATASETS.items():
        found = scan_family(key, spec)
        print(f"  {key:<24s} {len(found):5d} images")
        rows.extend(found)
    df = pd.DataFrame(rows)

    # Split, subject, study and disease status of every image come from the real-pool
    # manifest, looked up by DICOM id (for a synthetic image: the id of its source image).
    pool = pd.read_csv(config.SOURCE_POOL_MANIFEST, dtype=str)
    pool["dicom_id"] = pool["dicom_id"].str.lower()
    pool = pool.drop_duplicates("dicom_id").set_index("dicom_id")

    key = df["dicom_id"].str.lower()
    df["split"] = key.map(pool["split"].str.lower())
    df["source_disease_status"] = key.map(pool["disease_status"].str.lower())
    for col in ("subject_id", "study_id"):                 # authoritative in the manifest
        df[col] = key.map(pool[col]).fillna(df[col])

    n_unmatched = int(df["split"].isna().sum())
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} images have a DICOM id that is not in {config.SOURCE_POOL_MANIFEST.name}")

    df["image_id"] = df["frequency_group"] + "__" + df["filename"].str.rsplit(".", n=1).str[0]
    df = df[[
        "image_id", "dataset_key", "group", "frequency_group", "generator", "label",
        "split", "dicom_id", "subject_id", "study_id", "source_disease_status",
        "direction", "folder_split", "native_width", "native_height", "file_ext",
        "file_size_bytes", "filename", "path",
    ]].reset_index(drop=True)

    if save:
        df.to_csv(ensure_parent(config.DET_MASTER_MANIFEST), index=False)
        print(f"Saved master manifest ({len(df)} images) -> {config.DET_MASTER_MANIFEST}")
    return df


def summarise_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Group x split count table."""
    return (df.pivot_table(index="frequency_group", columns="split", values="image_id",
                           aggfunc="count", fill_value=0)
              .reindex(columns=list(config.SPLIT_NAMES))
              .assign(total=lambda t: t.sum(axis=1)))