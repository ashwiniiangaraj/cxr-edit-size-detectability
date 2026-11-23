from __future__ import annotations

"""
subset_from_downloaded.py - build a SMALLER real pool (e.g. 2500 images) from a real pool that is already downlaoded with 10K images.

"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import MANIFEST_DIR, REAL_DIR, RANDOM_SEED, POOL_SIZE_PER_POOL
try:                                    
    from download.download_dataset import (
        SPLITS, LABELS, log,
        compute_pool_class_counts, compute_split_counts,
        select_image_balanced_pool, split_pool,
        local_path, save_manifests, save_pool_summary,
        validate_image, verify_files, verify_split,
    )
except ModuleNotFoundError:             
    from download_dataset import (
        SPLITS, LABELS, log,
        compute_pool_class_counts, compute_split_counts,
        select_image_balanced_pool, split_pool,
        local_path, save_manifests, save_pool_summary,
        validate_image, verify_files, verify_split,
    )

POOL = "real"         
REQUIRED = {"dicom_id", "subject_id", "study_id", "ViewPosition", "disease_status", "patient_category", "split"}


# ----------------------------------------------------------------------------
# 1. find and read the downloaded pool
# ----------------------------------------------------------------------------
def find_source(src_root, manifest=None, image_root=None):
    """Locate real_all.csv and the real/ image folder under src_root (or use the given paths)."""
    src_root = Path(src_root)
    if manifest is None:
        found = sorted(src_root.rglob(f"{POOL}_all.csv"))
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {POOL}_all.csv under {src_root}, found {len(found)}: {found}. "
                "Pass manifest=... explicitly.")
        manifest = found[0]
    if image_root is None:
        found = [d for d in sorted(src_root.rglob(POOL)) if d.is_dir() and (d / "train").is_dir()]
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '{POOL}/' image folder (with train/) under {src_root}, found {found}. "
                "Pass image_root=... explicitly.")
        image_root = found[0]
    return Path(manifest), Path(image_root)


def load_source(manifest: Path, image_root: Path, drop_incomplete: bool = False):
    """Read the source manifest and check which images exist. Returns (candidates, {dicom_id: source path}).

    If some listed images are missing (e.g. the download did not finish) this stops with a table of what is
    missing, unless drop_incomplete=True: then every patient with ANY missing or empty image is removed, so the
    remaining patients are complete (whole patients only, the same rule as the downloader)."""
    df = pd.read_csv(manifest, dtype={"subject_id": str, "study_id": str, "dicom_id": str})
    missing_cols = REQUIRED - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{manifest} is missing columns {sorted(missing_cols)}")
    if df.duplicated(["subject_id", "study_id", "dicom_id"]).any():
        raise RuntimeError("duplicate images in the source manifest")
    src = {r["dicom_id"]: local_path(image_root, str(r["split"]), r) for _, r in df.iterrows()}
    df["present"] = [src[d].exists() and src[d].stat().st_size > 0 for d in df["dicom_id"]]
    if not df["present"].all():
        table = (df.assign(missing=~df["present"]).groupby(["split", "disease_status"])
                   .agg(listed=("dicom_id", "size"), missing=("missing", "sum")))
        log("Images listed in the source manifest but missing/empty on disk:\n" + table.to_string())
        if not drop_incomplete:
            raise RuntimeError(f"{int((~df['present']).sum())} source images are missing/empty (table above). "
                               "Finish the download, or call build_subset(..., drop_incomplete=True) to use only "
                               "patients whose images are all present.")
        bad = set(df.loc[~df["present"], "subject_id"])
        df = df[~df["subject_id"].isin(bad)]
        log(f"drop_incomplete: removed {len(bad)} patients with a missing image; "
            f"{len(df)} images / {df.subject_id.nunique()} patients remain")
    df = df.drop(columns=["present"])
    candidates = df.drop(columns=["split"]).reset_index(drop=True)   # the new split replaces the old one
    return candidates, {d: src[d] for d in candidates["dicom_id"]}


# ----------------------------------------------------------------------------
# 2. choose the subset (same rule as the downloader)
# ----------------------------------------------------------------------------
def _pick_patients(counts: pd.DataFrame, n_norm: int, n_path: int, rng):
    """Choose WHOLE patients whose images add up to exactly (n_norm normal, n_path pathology), or None.
    counts: index = subject_id, columns = ['normal', 'pathology'] (images per patient).
    Dynamic programme over the (normal, pathology) totals; first[i, j] = step at which that total was first reached."""
    ids = list(counts.index)
    order = rng.permutation(len(ids))
    reach = np.zeros((n_norm + 1, n_path + 1), dtype=bool)
    reach[0, 0] = True
    first = np.full(reach.shape, -1, dtype=np.int32)
    for step, k in enumerate(order):
        a, b = int(counts.iloc[k]["normal"]), int(counts.iloc[k]["pathology"])
        if (a == 0 and b == 0) or a > n_norm or b > n_path:
            continue
        new = np.zeros_like(reach)
        new[a:, b:] = reach[: n_norm + 1 - a, : n_path + 1 - b]
        fresh = new & ~reach
        first[fresh] = step
        reach |= fresh
        if reach[n_norm, n_path]:
            break
    if not reach[n_norm, n_path]:
        return None
    chosen, i, j = [], n_norm, n_path
    while (i, j) != (0, 0):
        k = order[first[i, j]]
        chosen.append(ids[k])
        i -= int(counts.iloc[k]["normal"]); j -= int(counts.iloc[k]["pathology"])
    return chosen


def split_pool_balanced(pool: pd.DataFrame, pool_size: int, seed: int):
    """Like download_dataset.split_pool (whole patients, exact 60/20/20 image counts, no patient in two splits)
    but ALSO exactly 50/50 normal/pathology INSIDE every split. Needs an exactly balanced pool."""
    sizes = compute_split_counts(pool_size)
    cls = compute_pool_class_counts(pool_size)
    per_class = {s: {l: int(round(sizes[s] * cls[l] / pool_size)) for l in LABELS} for s in SPLITS}
    counts = pool.groupby(["subject_id", "disease_status"]).size().unstack(fill_value=0).reindex(columns=list(LABELS), fill_value=0)
    train_ids = _pick_patients(counts, per_class["train"]["normal"], per_class["train"]["pathology"], np.random.default_rng(seed))
    if train_ids is None:
        raise RuntimeError("no class-balanced train split exists with whole patients")
    rest = counts.drop(train_ids)
    val_ids = _pick_patients(rest, per_class["val"]["normal"], per_class["val"]["pathology"], np.random.default_rng(seed + 1))
    if val_ids is None:
        raise RuntimeError("no class-balanced validation split exists with whole patients")
    test_ids = set(rest.index) - set(val_ids)
    frames = []
    for name, ids, off in (("train", set(train_ids), 10), ("val", set(val_ids), 11), ("test", test_ids, 12)):
        fr = pool[pool["subject_id"].isin(ids)].sample(frac=1, random_state=seed + off).reset_index(drop=True)
        got = fr["disease_status"].value_counts().to_dict()
        if any(got.get(l, 0) != per_class[name][l] for l in LABELS):
            raise RuntimeError(f"{name}: class counts {got} differ from {per_class[name]}")
        frames.append(fr)
    verify_split(frames[0], frames[1], frames[2], "real", pool_size)
    return tuple(frames)


def choose_subset(candidates: pd.DataFrame, pool_size: int, max_attempts: int = 30, balance_splits: bool = False):
    compute_pool_class_counts(pool_size)       # raises if the size is not compatible with the fractions
    compute_split_counts(pool_size)
    splitter = split_pool_balanced if balance_splits else (lambda p, n, seed: split_pool(p, POOL, n, seed=seed))
    for attempt in range(max_attempts):
        try:
            pool = select_image_balanced_pool(candidates, pool_size, seed=RANDOM_SEED + attempt * 100)
            splits = splitter(pool, pool_size, RANDOM_SEED + 100_000 + attempt)     # raises if not feasible
            return pool, splits, attempt
        except RuntimeError as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"no feasible {pool_size}-image subset after {max_attempts} attempts: {exc}")
            log(f"attempt {attempt + 1} not feasible ({exc}), trying the next deterministic sample ...")


# ----------------------------------------------------------------------------
# 3. place the files
# ----------------------------------------------------------------------------
def place(src: Path, dst: Path, link: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return "already_exists"
    if link == "hardlink":
        try:
            dst.hardlink_to(src)
            return "hardlinked"
        except OSError:
            pass                                  
    elif link == "symlink":
        dst.symlink_to(src.resolve())
        return "symlinked"
    shutil.copy2(src, dst)
    return "copied"


def build_subset(src_root, pool_size: int = POOL_SIZE_PER_POOL, link: str = "copy",
                 manifest=None, image_root=None, overwrite: bool = False, check_images: bool = True,
                 drop_incomplete: bool = False, dry_run: bool = False, balance_splits: bool = False):
    """link: 'copy' (safest), 'hardlink' (no extra disk, same filesystem) or 'symlink'.
    drop_incomplete: if the source download is incomplete, use only patients whose images are ALL present.
    dry_run: check the source, choose the subset and print the plan, but create and copy NOTHING.
    balance_splits: also make train, val and test EACH exactly 50/50 normal/pathology (patients stay whole)."""
    assert link in ("copy", "hardlink", "symlink")
    manifest, image_root = find_source(src_root, manifest, image_root)
    new_manifest = Path(MANIFEST_DIR) / f"{POOL}_all.csv"
    if image_root.resolve() == Path(REAL_DIR).resolve() or manifest.resolve() == new_manifest.resolve():
        raise RuntimeError("The new project's REAL_DIR / MANIFEST_DIR point at the SOURCE folders. "
                           "Change config.py first so the new run writes to its own folders.")
    if new_manifest.exists() and not overwrite and not dry_run:
        raise RuntimeError(f"{new_manifest} already exists (use overwrite=True to rebuild).")

    log(f"source manifest : {manifest}\nsource images   : {image_root}")
    candidates, src = load_source(manifest, image_root, drop_incomplete)
    log(f"source pool     : {len(candidates)} images, {candidates.subject_id.nunique()} patients, "
        f"{candidates.disease_status.value_counts().to_dict()}")

    pool, splits, attempt = choose_subset(candidates, pool_size, balance_splits=balance_splits)
    assert set(pool.subject_id) <= set(candidates.subject_id) and set(pool.dicom_id) <= set(src)

    if dry_run:
        log(f"\nDRY RUN OK - nothing was created or copied. Plan: {pool_size} images, "
            f"{pool.subject_id.nunique()} patients (attempt {attempt + 1})")
        for name, frame in zip(SPLITS, splits):
            log(f"  {name:5s}: {len(frame)} images, {frame.subject_id.nunique()} patients, "
                f"{frame.disease_status.value_counts().to_dict()}")
        log(f"would write images to {REAL_DIR} and manifests to {MANIFEST_DIR}")
        return dict(pool=pool, splits=splits, attempt=attempt)

    for root in [Path(REAL_DIR) / s / l for s in SPLITS for l in LABELS] + [Path(MANIFEST_DIR)]:
        root.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, frame in zip(SPLITS, splits):
        for _, row in frame.iterrows():
            dst = local_path(Path(REAL_DIR), name, row)
            how = place(src[row["dicom_id"]], dst, link)
            counts[how] = counts.get(how, 0) + 1
            if check_images and not validate_image(dst):
                raise RuntimeError(f"corrupt image after placing: {dst}")

    save_manifests(POOL, splits)
    save_pool_summary(POOL, splits)
    master = manifest.parent / "master_eligible_images.csv"
    if master.exists():
        shutil.copy2(master, Path(MANIFEST_DIR) / "master_eligible_images.csv")
    pd.DataFrame([dict(source_manifest=str(manifest), source_images=str(image_root), source_images_n=len(candidates),
                       source_patients=candidates.subject_id.nunique(), subset_size=pool_size, attempt=attempt,
                       random_seed=RANDOM_SEED, link=link, balance_splits=balance_splits, note="nested subset of the downloaded pool; split recomputed at patient level "
                       "with the downloader's functions" + ("; each split exactly 50/50" if balance_splits else ""))]
                 ).to_csv(Path(MANIFEST_DIR) / "subset_provenance.csv", index=False)

    verify_files(Path(REAL_DIR), pool_size)
    log(f"\nSUBSET READY: {pool_size} images, {pool.subject_id.nunique()} patients  ({counts})")
    for name, frame in zip(SPLITS, splits):
        log(f"  {name:5s}: {len(frame)} images, {frame.subject_id.nunique()} patients, "
            f"{frame.disease_status.value_counts().to_dict()}")
    log(f"manifests: {MANIFEST_DIR}\nimages   : {REAL_DIR}\nNext: run dataset_validation.validate_dataset()")
    return dict(pool=pool, splits=splits, attempt=attempt)

if __name__ == "__main__":
    import sys
    build_subset(sys.argv[1])