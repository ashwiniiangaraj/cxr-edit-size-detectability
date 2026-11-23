"""
DATA INTEGRITY CHECKS
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from utils.common import ensure_parent


def _average_hash(path: str, size: int) -> tuple[np.ndarray, int]:
    """size x size average hash (bool vector) + number of distinct grey levels."""
    with Image.open(path) as im:
        g = im.convert("L")
        full = np.asarray(g)
        small = np.asarray(g.resize((size, size), Image.Resampling.BOX), dtype=np.float32)
    return (small > small.mean()).ravel(), int(len(np.unique(full)))


def _near_duplicate_pairs(hashes: np.ndarray, max_hamming: int, block: int = 2048):
    """All pairs (i<j) with Hamming distance <= max_hamming, computed in blocks so
    memory stays at block x N (an N x N float32 matrix would be ~14 GB at N=60,000)."""
    h = hashes.astype(np.float32)
    n, n_bits = h.shape
    hn = 1.0 - h
    pairs = []
    for s0 in range(0, n, block):
        e0 = min(s0 + block, n)
        ham = n_bits - (h[s0:e0] @ h.T + hn[s0:e0] @ hn.T)
        ii, jj = np.nonzero(ham <= max_hamming)
        for i, j in zip(ii + s0, jj):
            if i < j:
                pairs.append((int(i), int(j), int(round(ham[i - s0, j]))))
    return pairs


def check_counts(df):
    """Counts must match POOL_SIZE_PER_POOL (config) within the allowed shortfall."""
    actual = df.groupby("frequency_group").size().to_dict()
    problems = {}
    for g, expected in config.DATASET_EXPECTED_COUNTS.items():
        got = actual.get(g, 0)
        short = int(np.ceil(expected * config.DATASET_ALLOWED_SHORTFALL.get(g, 0.0)))
        lo, hi = expected - short - config.DATASET_COUNT_TOLERANCE, expected + config.DATASET_COUNT_TOLERANCE
        if not lo <= got <= hi:
            problems[g] = {"expected": expected, "allowed_range": [lo, hi], "found": got}
    extra = sorted(set(actual) - set(config.DATASET_EXPECTED_COUNTS))
    return (not problems and not extra), {"actual": actual, "problems": problems,
                                          "unexpected_groups": extra}


def check_splits(df):
    missing = df[df["split"].isna() | ~df["split"].isin(config.SPLIT_NAMES)]
    bad_folder = df[df["folder_split"].notna() & (df["folder_split"] != df["split"])]
    return (missing.empty and bad_folder.empty), {
        "images_without_valid_split": len(missing),
        "folder_split_disagrees_with_source_split": len(bad_folder),
        "examples": bad_folder["image_id"].head(5).tolist(),
        "counts": df.pivot_table(index="frequency_group", columns="split", values="image_id",
                                 aggfunc="count", fill_value=0).to_dict("index"),
    }


def check_patient_separation(df):
    """No patient may appear in more than one split."""
    d = df.dropna(subset=["subject_id", "split"])
    per_subject = d.groupby("subject_id")["split"].nunique()
    leaking = per_subject[per_subject > 1].index.tolist()
    return not leaking, {
        "images_without_subject_id": int(df["subject_id"].isna().sum()),
        "n_patients_in_multiple_splits": len(leaking),
        "patients_in_multiple_splits": leaking[:20],
    }


def check_source_integrity(df):
    """Every synthetic image must come from a Real-group image and share its split."""
    real = df[df["group"] == config.REAL_GROUP].drop_duplicates("dicom_id").set_index("dicom_id")
    synth = df[df["group"] != config.REAL_GROUP]
    no_source = synth[~synth["dicom_id"].isin(real.index)]
    in_pool = synth[synth["dicom_id"].isin(real.index)]
    wrong_split = in_pool[in_pool["split"].values != real.loc[in_pool["dicom_id"], "split"].values]
    return (no_source.empty and wrong_split.empty), {
        "synthetic_images_without_real_source": len(no_source),
        "synthetic_images_with_a_different_split_than_their_source": len(wrong_split),
        "examples": pd.concat([no_source, wrong_split])["image_id"].head(5).tolist(),
    }


def check_content_and_duplicates(df):
    size = config.VALIDATION_NEAR_DUP_HASH_SIZE
    unreadable, mask_like, hashes = [], [], []
    for p in df["path"]:
        try:
            h, n_levels = _average_hash(p, size)
            hashes.append(h)
            if n_levels <= config.VALIDATION_MASK_MAX_UNIQUE_VALUES:
                mask_like.append(p)
        except Exception as exc:                               # noqa: BLE001
            unreadable.append({"path": p, "error": repr(exc)})
            hashes.append(np.zeros(size * size, dtype=bool))
    pairs = _near_duplicate_pairs(np.stack(hashes), config.VALIDATION_NEAR_DUP_MAX_HAMMING)

    ids, splits, sources = df["image_id"].values, df["split"].values, df["dicom_id"].values
    cross, same = [], 0
    for i, j, d in pairs:
        if splits[i] != splits[j]:
            cross.append({"a": ids[i], "b": ids[j], "hamming": d,
                          "split_a": splits[i], "split_b": splits[j]})
        else:
            same += 1
    sibling = sum(1 for i, j, _ in pairs if sources[i] is not None and sources[i] == sources[j])
    passed = not unreadable and not mask_like and not cross
    return passed, {
        "n_unreadable": len(unreadable), "unreadable": unreadable[:10],
        "n_mask_like": len(mask_like), "mask_like_images": mask_like[:10],
        "near_duplicate_pairs_total": len(pairs),
        "near_duplicate_pairs_within_same_split": same,
        "of_which_same_source_siblings": sibling,
        "near_duplicate_pairs_ACROSS_splits": len(cross), "cross_split_examples": cross[:10],
    }


def check_masks_by_path(df):
    """Re-verify (independently of the scanner) that no mask / report file is in the table."""
    bad = []
    for key, p in zip(df["dataset_key"], df["path"]):
        root = Path(config.FORENSIC_CHARACTERISATION_DATASETS[key]["root"])
        pp = Path(p)
        rel_dirs = [x.lower() for x in pp.relative_to(root).parts[:-1]]
        if (any(d in config.DATASET_EXCLUDE_DIR_NAMES for d in rel_dirs)
                or pp.stem.lower().endswith(tuple(config.DATASET_EXCLUDE_FILE_SUFFIXES))):
            bad.append(p)
    return not bad, {"suspicious_paths": bad[:10], "n_suspicious": len(bad)}


def shortcut_audit(df):
    """Per-family trivial statistics + a metadata-only 'shortcut probe'."""
    table = (df.groupby("frequency_group")
               .agg(n=("image_id", "size"), width_median=("native_width", "median"),
                    height_median=("native_height", "median"), width_min=("native_width", "min"),
                    width_max=("native_width", "max"),
                    ext=("file_ext", lambda s: ",".join(sorted(set(s)))),
                    kb_median=("file_size_bytes", lambda s: float(np.median(s)) / 1024))
               .round(1))
    feats = pd.DataFrame({
        "w": df["native_width"], "h": df["native_height"],
        "log_bytes": np.log10(df["file_size_bytes"].clip(lower=1)),
        "is_png": (df["file_ext"] == ".png").astype(int),
        "aspect": df["native_width"] / df["native_height"],
        "bytes_per_px": df["file_size_bytes"] / (df["native_width"] * df["native_height"]),
    })
    tr = (df["split"] == "train") & df["label"].isin([0, 1])
    te = (df["split"] == "test") & df["label"].isin([0, 1])
    y = df["label"].values
    auroc = float("nan")
    if tr.any() and te.any() and len(set(y[tr])) == 2 and len(set(y[te])) == 2:
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feats[tr]), y[tr])
        auroc = float(roc_auc_score(y[te], clf.predict_proba(sc.transform(feats[te]))[:, 1]))
    return table, {"metadata_only_probe_auroc": auroc,
                   "warn_threshold": config.VALIDATION_SHORTCUT_AUROC_WARN,
                   "shortcut_warning": bool(auroc > config.VALIDATION_SHORTCUT_AUROC_WARN)}


def validate_dataset(df: pd.DataFrame | None = None, strict: bool | None = None) -> dict:
    """Run every check, print + save a report, optionally raise on HARD failures."""
    if df is None:
        df = pd.read_csv(config.DET_MASTER_MANIFEST, dtype={"subject_id": str, "study_id": str})
    strict = config.VALIDATION_STRICT if strict is None else strict

    results = {
        "counts": dict(zip(("passed", "details"), check_counts(df))),
        "splits": dict(zip(("passed", "details"), check_splits(df))),
        "patient_separation": dict(zip(("passed", "details"), check_patient_separation(df))),
        "source_integrity": dict(zip(("passed", "details"), check_source_integrity(df))),
    }
    ok, det = check_content_and_duplicates(df)
    results["content_and_duplicates"] = {"passed": ok, "details": det}
    results["masks_excluded_by_path"] = dict(zip(("passed", "details"), check_masks_by_path(df)))
    table, probe = shortcut_audit(df)
    results["shortcut_audit"] = {"passed": not probe["shortcut_warning"], "details": probe,
                                 "per_family_table": table.reset_index().to_dict("records")}

    hard = ["counts", "splits", "patient_separation", "source_integrity", "content_and_duplicates",
            "masks_excluded_by_path"]
    all_hard = all(results[k]["passed"] for k in hard)
    results["ALL_HARD_CHECKS_PASSED"] = all_hard

    lines = ["DATASET VALIDATION REPORT", "=" * 60, f"images: {len(df)}", ""]
    for k in hard + ["shortcut_audit"]:
        tag = "PASS" if results[k]["passed"] else ("WARN" if k == "shortcut_audit" else "FAIL")
        lines.append(f"[{tag}] {k}")
        for kk, vv in results[k]["details"].items():
            if kk in ("counts", "actual") and isinstance(vv, dict):
                continue
            lines.append(f"        {kk}: {vv}")
    lines += ["", "Per-family trivial metadata (shortcut audit):", table.to_string(), "",
              f"OVERALL (hard checks): {'PASS' if all_hard else 'FAIL'}"]
    text = "\n".join(lines)
    ensure_parent(config.DET_VALIDATION_REPORT_TXT).write_text(text)
    ensure_parent(config.DET_VALIDATION_REPORT_JSON).write_text(json.dumps(results, indent=2, default=str))
    print(text)

    if strict and not all_hard:
        raise RuntimeError(f"Dataset validation FAILED – see {config.DET_VALIDATION_REPORT_TXT}. "
                           "Fix the data or set config.VALIDATION_STRICT = False.")
    return results