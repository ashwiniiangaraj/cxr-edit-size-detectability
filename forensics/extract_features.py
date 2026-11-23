"""
batch feature extraction over every prepared 256x256 image (multiprocessing).
Writes config.FORENSIC_FEATURE_CSV (42 features + manifest columns), config.DET_RADIAL_PROFILES,
config.FORENSIC_FEATURE_ERRORS_CSV and config.RESIDUAL_FEATURE_CSV (6 specres_* features).
Runs the extractors only; formulas are in stage3_forensics/feature_extractors.py. Output paths are
optional, so the ladder and sweep image sets can be processed without touching the main table.
Safety: unique image_ids and existing files are required; a failed image stops the run (strict=True);
a table containing NaN or inf is never written.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from forensics.feature_extractors import extract_feature_dict, spectral_features_residual
from utils.common import ensure_parent, load_gray_float01

_ID_DTYPES = {"subject_id": str, "study_id": str}


# ------------------------------- workers --------------------------------------
def _init_worker() -> None:
    """One OpenCV thread per worker process, so n_jobs processes do not oversubscribe the CPU."""
    import cv2
    cv2.setNumThreads(1)


def _worker(args):
    image_id, path = args
    try:
        feats, profile = extract_feature_dict(load_gray_float01(path))
        return image_id, feats, profile, None
    except Exception as exc:                                       # noqa: BLE001
        return image_id, None, None, repr(exc)


def _worker_residual(args):
    image_id, path = args
    try:
        feats, _ = spectral_features_residual(load_gray_float01(path))
        return image_id, feats, None
    except Exception as exc:                                       # noqa: BLE001
        return image_id, None, repr(exc)


def _run(worker, jobs, n_jobs):
    """Map `worker` over `jobs`, in order, with n_jobs processes (or in-process when n_jobs == 1)."""
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
            return list(ex.map(worker, jobs, chunksize=16))
    return [worker(j) for j in jobs]


def _check_manifest(manifest: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the (image_id, prepared_path) jobs after checking ids and files."""
    for col in ("image_id", "prepared_path"):
        if col not in manifest.columns:
            raise KeyError(f"manifest has no '{col}' column")
    if manifest["image_id"].duplicated().any():
        dup = manifest.loc[manifest["image_id"].duplicated(), "image_id"].head(3).tolist()
        raise ValueError(f"duplicated image_id in the manifest, e.g. {dup}")
    missing = [p for p in manifest["prepared_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prepared images are missing, e.g. {missing[:3]}")
    return list(zip(manifest["image_id"], manifest["prepared_path"]))


# --------------------------- the 42 main features ------------------------------
def extract_all_features(manifest: pd.DataFrame, n_jobs: int | None = None,
                         feature_csv=None, profiles_npz=None, errors_csv=None,
                         strict: bool = True) -> pd.DataFrame:
    """42 features + radial profile for every row of `manifest` (needs image_id, prepared_path)."""
    n_jobs       = n_jobs or config.DET_N_JOBS
    feature_csv  = feature_csv  or config.FORENSIC_FEATURE_CSV
    profiles_npz = profiles_npz or config.DET_RADIAL_PROFILES
    errors_csv   = errors_csv   or config.FORENSIC_FEATURE_ERRORS_CSV
    jobs = _check_manifest(manifest)
    print(f"Extracting {len(config.FEATURE_COLUMNS)} features from {len(jobs)} images (n_jobs={n_jobs}) ...")
    results = _run(_worker, jobs, n_jobs)

    ok = [r for r in results if r[3] is None]
    errors = pd.DataFrame([(r[0], r[3]) for r in results if r[3] is not None],
                          columns=["image_id", "error"])
    errors.to_csv(ensure_parent(errors_csv), index=False)
    if len(errors):
        msg = f"{len(errors)} of {len(jobs)} images failed - see {errors_csv}; first: {errors.iloc[0].tolist()}"
        if strict:
            raise RuntimeError(msg)
        print("WARNING:", msg)
    if not ok:
        raise RuntimeError("no image could be processed")

    feat_df = pd.DataFrame([r[1] for r in ok])[config.FEATURE_COLUMNS]
    if not np.isfinite(feat_df.to_numpy(dtype=np.float64)).all():
        bad = [r[0] for r, f in zip(ok, np.isfinite(feat_df.to_numpy()).all(axis=1)) if not f][:5]
        raise ValueError(f"NaN / inf features produced, e.g. for {bad}")
    feat_df.insert(0, "image_id", [r[0] for r in ok])

    out = manifest.merge(feat_df, on="image_id", how="inner", validate="one_to_one")
    assert len(out) == len(ok), "merge changed the number of rows"
    out.to_csv(ensure_parent(feature_csv), index=False)
    np.savez_compressed(ensure_parent(profiles_npz),
                        image_id=np.array([r[0] for r in ok]), profile=np.stack([r[2] for r in ok]))
    print(f"Saved {len(out)} rows -> {feature_csv}")
    return out


def load_feature_table(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.FORENSIC_FEATURE_CSV, dtype=_ID_DTYPES)


def load_radial_profiles(path=None) -> pd.DataFrame:
    """DataFrame indexed by image_id, one column per radial bin."""
    z = np.load(path or config.DET_RADIAL_PROFILES, allow_pickle=False)
    return pd.DataFrame(z["profile"], index=z["image_id"])


# ---------------- secondary set: spectral features on the high-pass residual ----------------
def extract_residual_features(manifest: pd.DataFrame, n_jobs: int | None = None,
                              out_csv=None) -> pd.DataFrame:
    """6 'specres_*' features per prepared image -> config.RESIDUAL_FEATURE_CSV (the 42 are untouched)."""
    n_jobs = n_jobs or config.DET_N_JOBS
    out_csv = out_csv or config.RESIDUAL_FEATURE_CSV
    jobs = _check_manifest(manifest)
    print(f"Extracting {len(config.RESIDUAL_SPECTRAL_FEATURES)} residual spectral features "
          f"from {len(jobs)} images (n_jobs={n_jobs}) ...")
    results = _run(_worker_residual, jobs, n_jobs)
    bad = [r for r in results if r[2] is not None]
    if bad:
        raise RuntimeError(f"{len(bad)} images failed, e.g. {[(r[0], r[2]) for r in bad[:3]]}")
    df = pd.DataFrame([r[1] for r in results])[config.RESIDUAL_SPECTRAL_FEATURES]
    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        raise ValueError("NaN / inf residual features produced")
    df.insert(0, "image_id", [r[0] for r in results])
    out = manifest.merge(df, on="image_id", how="inner", validate="one_to_one")
    assert len(out) == len(manifest), "merge changed the number of rows"
    out.to_csv(ensure_parent(out_csv), index=False)
    print(f"Saved {len(out)} rows -> {out_csv}")
    return out


def load_residual_feature_table(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.RESIDUAL_FEATURE_CSV, dtype=_ID_DTYPES)
