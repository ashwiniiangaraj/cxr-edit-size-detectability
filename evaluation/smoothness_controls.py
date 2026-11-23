"""
Smoothness controls: is the detectability of CycleGAN and full diffusion largely a non-specific smoothness
cue, and do local edits survive it?
A  Train on Real vs smoothed Real (no generator); test on each synthetic set vs Real.
B  Score fitted detectors on real images blurred to match the spectral slope of Diffusion_Reconstruction and CycleGAN.
C  Match each CycleGAN image to the grey-level histogram of its real source; compare AUROC before and after.
D  Blur a 0.5-10 % patch of real test images; report AUROC vs real and paired win rate vs the original.
"""
from __future__ import annotations

import traceback
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import config
from forensics.feature_extractors import extract_feature_vector
from experiments import ladder_scoring as ls
from evaluation import provenance_control as pc
from experiments.runner import load_modelling_table
from models import get_detector_class
from models.feature_lr import feature_frame

L = Image.Resampling.LANCZOS
OUT = Path(ls.OUT_DIR) / "smoothness_controls"
_ID = pc._ID
TARGETS = ["IP", "CycleGAN", "Diffusion", "Diffusion_EditOnly", "Diffusion_Reconstruction"]
GROUPS_OF = {"IP": ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"], "CycleGAN": ["CycleGAN"], "Diffusion": ["Diffusion"],
             "Diffusion_EditOnly": ["Diffusion_EditOnly"], "Diffusion_Reconstruction": ["Diffusion_Reconstruction"]}
AREAS = [0.005, 0.01, 0.02, 0.05, 0.10]
PATCH_SIGMA = 1.5          
SLOPE_COL = "spectral_slope"


# ------------------------------------------------------------------ helpers
def _paths(df):
    if "path" in df.columns:
        return df.reset_index(drop=True)
    m = pd.read_csv(config.DET_MASTER_MANIFEST, dtype=str)[["image_id", "path"]]
    return df.merge(m, on="image_id", how="left").reset_index(drop=True)


def _load256(path) -> np.ndarray:
    """native file -> greyscale -> Lanczos 256 (uint8), i.e. Stage 2 BEFORE the JPEG step."""
    S = config.PREP_IMAGE_SIZE
    im = Image.open(path).convert("L")
    return np.asarray(im.resize((S, S), L) if im.size != (S, S) else im, dtype=np.uint8)


def smooth(a: np.ndarray, kind: str, param=None) -> np.ndarray:
    S = a.shape[0]
    if kind == "updown":
        im = Image.fromarray(a).resize((2 * S, 2 * S), L).resize((S, S), L)
        return np.asarray(im, dtype=np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(a, (0, 0), float(param))
    raise ValueError(kind)


def _jpeg_roundtrip(a: np.ndarray) -> np.ndarray:
    import io
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=config.PREP_JPEG_QUALITY)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


def _write(a, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(dst, format="PNG")


def make_control_df(real: pd.DataFrame, arrays: dict, kind: str, param, tag: str, label=1, overwrite=False) -> pd.DataFrame:
    """Smoothed copies of `real` rows (new image_ids, label=`label`, 42 features recomputed from the new files)."""
    d = real.copy().reset_index(drop=True)
    d["image_id"] = f"sm_{tag}__" + real["image_id"].astype(str).to_numpy()
    d["prepared_path"] = [str(OUT / "images" / tag / f"{i}.png") for i in d["image_id"]]
    for src_id, dst in zip(real["image_id"], d["prepared_path"]):
        if overwrite or not Path(dst).exists():
            _write(_jpeg_roundtrip(smooth(arrays[src_id], kind, param)), Path(dst))
    d["label"] = label
    d[list(config.FEATURE_COLUMNS)] = feature_frame(d, degrade=_ID).to_numpy()
    return d


def calibrate(real_train: pd.DataFrame, arrays: dict, T: pd.DataFrame, n=60, grid=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)):
    """Blur sigma whose median spectral slope (on n real TRAIN images) is closest to the target group's median slope."""
    idx = list(config.FEATURE_COLUMNS).index(SLOPE_COL)
    ids = real_train["image_id"].iloc[:n].tolist()
    med = {}
    for s in grid:
        v = [extract_feature_vector(_jpeg_roundtrip(cv2.GaussianBlur(arrays[i], (0, 0), s)).astype(np.float32) / 255.0)[idx]
             for i in ids]
        med[s] = float(np.median(v))
    tgt = {g: float(T[(T["frequency_group"] == g) & (T["split"] == "train")][SLOPE_COL].median())
           for g in ["Diffusion_Reconstruction", "CycleGAN", "Diffusion", config.REAL_GROUP]}
    best = {g: min(grid, key=lambda s: abs(med[s] - tgt[g])) for g in ["Diffusion_Reconstruction", "CycleGAN", "Diffusion"]}
    print("\nCalibration of blur strength by spectral slope (median over real train images):")
    print("  blur sigma ->", {s: round(m, 3) for s, m in med.items()})
    print("  target group medians:", {g: round(v, 3) for g, v in tgt.items()})
    print("  chosen sigma:", best, "(if a chosen sigma sits at the edge of the grid, widen `grid`)")
    return best, med, tgt


def _fitted(models=None, seeds=(42,)):
    """[(train_label, model, seed, detector)] of the already fitted detectors (never trained here)."""
    if models is None:
        models = sorted({f.name.rsplit("_seed", 1)[0] for s in ls.train_slugs().values()
                         for f in (Path(config.DET_MODEL_DIR) / s).glob("*_seed*")})
    out = []
    for tl, slug in ls.train_slugs().items():
        for m in models:
            for sd in [s for s in ls.model_seeds(m, slug) if s in seeds]:
                det = ls.load_detector(m, slug, sd)
                if det is not None:
                    out.append((tl, m, sd, det))
    return out


def score_sets(sets: dict, real: pd.DataFrame, models=None, seeds=(42,), tag=""):
    """AUROC of each set vs ALL real test images + paired win rate vs the real version (only if the set has the same rows
    as `real`, in the same order).  Fitted detectors, identity degradation => every model reads the image files."""
    rows = []
    for tl, m, sd, det in _fitted(models, seeds):
        s_real = np.asarray(det.predict_scores(real, degrade=_ID), float)
        for name, d in sets.items():
            s = np.asarray(det.predict_scores(d, degrade=_ID), float)
            paired = len(d) == len(real)
            rows.append(dict(train=tl, model=m, seed=sd, set=name, auroc=ls.auc(s, s_real),
                             win=ls.win(s, s_real) if paired else np.nan))
        print(f"  {tag} {tl:>17s} {m:>16s} seed {sd}: " + "  ".join(f"{r['set']}={r['auroc']:.2f}" for r in rows[-len(sets):]))
    return pd.DataFrame(rows)


def _auc(df, s):
    return float(roc_auc_score(df["label"].to_numpy(), np.asarray(s, float)))


# ------------------------------------------------------------------ A
def exp_A(T, real_all, arrays, sets_train, models=("spectral_lr", "handcrafted_lr", "lbp_lr"), nets=False):
    print("\n=== A. detector trained on Real vs SMOOTHED REAL only (no generator) ===")
    models = list(models) + (["cnn_densenet"] if nets else [])
    ref = None
    try:
        r = pd.read_csv(config.DET_RESULTS_CSV)
        r = r[(r["role"] == "test_cross") & (r["seed"] == 42)]
        col = "auroc" if "auroc" in r.columns else [c for c in r.columns if "auroc" in c.lower()][0]
        ref = r.groupby(["model", "test_family"])[col].mean().rename("grid_mean_over_train_families").reset_index()
    except Exception as e:                                           # noqa: BLE001
        print("  (reference from the main grid not available:", repr(e), ")")
    real = {sp: real_all[real_all["split"] == sp].reset_index(drop=True) for sp in ("train", "val", "test")}
    for sp in real:
        real[sp]["label"] = 0
    rows = []
    for kind_tag, cdf in sets_train.items():                         # cdf: {'train','val','test'} control dfs (label 1)
        tr = pd.concat([real["train"], cdf["train"]], ignore_index=True)
        va = pd.concat([real["val"], cdf["val"]], ignore_index=True)
        for m in models:
            det = get_detector_class(m)(seed=42).fit(tr, va)
            row = {"trained_on": f"Real vs {kind_tag}", "model": m}
            ins = pd.concat([real["test"], cdf["test"]], ignore_index=True)
            row["in_family(smoothed vs real)"] = _auc(ins, det.predict_scores(ins))
            for tg in TARGETS:
                g = T[T["frequency_group"].isin(GROUPS_OF[tg]) & (T["split"] == "test")].copy()
                g["label"] = 1
                te = pd.concat([real["test"], g], ignore_index=True)
                row[tg] = _auc(te, det.predict_scores(te))
            rows.append(row)
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "A_real_vs_smoothed_detector.csv", index=False)
    print("\nA - AUROC of the smoothing-only detector on each generator family (0.5 = nothing):")
    print(R.round(3).to_string(index=False))
    if ref is not None:
        ref = ref[ref["model"].isin(models)]
        print("\nreference: mean cross-family AUROC of detectors trained on real GENERATORS (main grid, seed 42):")
        print(ref.pivot(index="model", columns="test_family", values="grid_mean_over_train_families").round(3))
        ref.to_csv(OUT / "A_reference_grid_means.csv", index=False)
    return R


# ------------------------------------------------------------------ B
def exp_B(real_test, sets_test, models=None):
    print("\n=== B. fitted detectors on real images blurred to match generator smoothness ===")
    R = score_sets(sets_test, real_test, models, tag="B")
    R.to_csv(OUT / "B_matched_blur_fitted_detectors.csv", index=False)
    print(R.groupby(["model", "set"])[["auroc", "win"]].mean().round(3).unstack("set"))
    return R


# ------------------------------------------------------------------ C
def _match_hist(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_vals, s_idx, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_cdf = np.cumsum(s_cnt) / src.size
    r_cdf = np.cumsum(r_cnt) / ref.size
    return np.interp(s_cdf, r_cdf, r_vals).astype(np.uint8)[s_idx].reshape(src.shape)


def exp_C(T, real_test, models=None):
    print("\n=== C. CycleGAN with the grey-level histogram of its own real source ===")
    cg = T[(T["frequency_group"] == "CycleGAN") & (T["split"] == "test")].copy().reset_index(drop=True)
    key = next((k for k in ("dicom_id", "source_image_id", "source_id") if k in cg.columns and k in real_test.columns), None)
    if key is None:
        print("  skipped: no column linking CycleGAN rows to their real source (looked for dicom_id/source_image_id/source_id)")
        return None
    src = real_test.set_index(key)["prepared_path"].to_dict()
    cg = cg[cg[key].isin(src)].reset_index(drop=True)
    new = cg.copy()
    new["image_id"] = "sm_cgtone__" + cg["image_id"].astype(str)
    new["prepared_path"] = [str(OUT / "images" / "cgtone" / f"{i}.png") for i in new["image_id"]]
    for pth_c, pth_r, dst in zip(cg["prepared_path"], cg[key].map(src), new["prepared_path"]):
        a = np.asarray(Image.open(pth_c).convert("L"), dtype=np.uint8)
        r = np.asarray(Image.open(pth_r).convert("L"), dtype=np.uint8)
        _write(_match_hist(a, r), Path(dst))
    sets = {"cyclegan_raw": cg, "cyclegan_tone_matched": new}
    R = score_sets(sets, real_test, models, tag="C")
    R.to_csv(OUT / "C_cyclegan_tone_control.csv", index=False)
    print("  mean grey level  CycleGAN raw %.1f | tone-matched %.1f | real source %.1f" % tuple(
        np.mean([np.asarray(Image.open(p).convert('L')).mean() for p in d["prepared_path"]]) for d in
        (cg, new, real_test[real_test[key].isin(cg[key])])))
    print(R.groupby(["train", "set"])["auroc"].mean().unstack("set").round(3))
    print(R.groupby(["model", "set"])["auroc"].mean().unstack("set").round(3))
    return R


# ------------------------------------------------------------------ D
def _patch_blur(a: np.ndarray, area: float, rng) -> np.ndarray:
    H, W = a.shape
    r = np.sqrt(area * H * W / np.pi)
    # central rectangle used by the IP edits: 70% of width, 75% of height
    x0, x1, y0, y1 = 0.15 * W + r, 0.85 * W - r, 0.125 * H + r, 0.875 * H - r
    cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
    yy, xx = np.mgrid[0:H, 0:W]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 0.10 * (2 * r) / 2.563)           
    af = a.astype(np.float32)
    out = m * cv2.GaussianBlur(af, (0, 0), PATCH_SIGMA) + (1 - m) * af
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def exp_D(real_test, models=None):
    print("\n=== D. blur only a patch of a real image (local smoothing, no generator) ===")
    sets = {}
    for ar in AREAS:
        d = real_test.copy().reset_index(drop=True)
        d["image_id"] = f"sm_patch{ar}__" + real_test["image_id"].astype(str).to_numpy()
        d["prepared_path"] = [str(OUT / "images" / f"patch{ar}" / f"{i}.png") for i in d["image_id"]]
        for src_pth, iid, dst in zip(real_test["prepared_path"], real_test["image_id"], d["prepared_path"]):
            if not Path(dst).exists():
                a = np.asarray(Image.open(src_pth).convert("L"), dtype=np.uint8)
                rng = np.random.default_rng(zlib.crc32(str(iid).encode()) + int(ar * 10000))
                _write(_patch_blur(a, ar, rng), Path(dst))
        sets[f"{100 * ar:g}%"] = d
    R = score_sets(sets, real_test, models, tag="D")
    R.to_csv(OUT / "D_local_smoothing.csv", index=False)
    order = [f"{100 * a:g}%" for a in AREAS]
    print("\nD - mean AUROC over detectors (patch-blurred real vs real), by trained family:")
    print(R.groupby(["train", "set"])["auroc"].mean().unstack("set")[order].round(3))
    print("D - paired win rate vs the unblurred original (0.5 = no preference):")
    print(R.groupby(["train", "set"])["win"].mean().unstack("set")[order].round(3))
    print("D - by detector (AUROC):")
    print(R.groupby(["model", "set"])["auroc"].mean().unstack("set")[order].round(3))
    return R


# ------------------------------------------------------------------ driver
def run_all(nets=False, models=None, which="ABCD"):
    OUT.mkdir(parents=True, exist_ok=True)
    T = load_modelling_table()
    real_all = _paths(T[T["frequency_group"] == config.REAL_GROUP].copy())
    real_test_ls, _ = ls.load_inputs(T)
    real_test_ls = real_test_ls.reset_index(drop=True)
    print(f"Real images: {len(real_all)} (train/val/test = " +
          "/".join(str((real_all['split'] == s).sum()) for s in ('train', 'val', 'test')) + ")")
    arrays = {i: _load256(p) for i, p in zip(real_all["image_id"], real_all["path"])}
    res = {}
    try:
        best, med, tgt = calibrate(real_all[real_all["split"] == "train"], arrays, T)
    except Exception:                                                # noqa: BLE001
        traceback.print_exc()
        best = {}
    tests = real_all[real_all["split"] == "test"]
    # control dataframes (all splits) : updown + calibrated blurs
    kinds = {"updown": ("updown", None)}
    for g in ("Diffusion_Reconstruction", "CycleGAN"):
        if g in best:
            kinds[f"blur{best[g]:g}_matches_{g}"] = ("blur", best[g])
    sets_train = {}
    for tag, (kind, par) in kinds.items():
        sets_train[tag] = {sp: make_control_df(real_all[real_all["split"] == sp].reset_index(drop=True), arrays,
                                               kind, par, f"{tag}_{sp}", label=1) for sp in ("train", "val", "test")}
    if "A" in which:
        try:
            res["A"] = exp_A(T, real_all, arrays, sets_train, nets=nets)
        except Exception:                                            # noqa: BLE001
            traceback.print_exc()
    if "B" in which:
        try:
            # align control test dfs with the order of real_test_ls (needed for the paired win rate)
            order = {i: k for k, i in enumerate(real_test_ls["image_id"])}
            sets_B = {}
            for tag, d in sets_train.items():
                dd = d["test"].copy()
                dd["_o"] = dd["image_id"].str.replace(r"^sm_.*?__", "", regex=True).map(order)
                sets_B[tag] = dd[dd["_o"].notna()].sort_values("_o").drop(columns="_o").reset_index(drop=True)
            res["B"] = exp_B(real_test_ls, sets_B, models)
        except Exception:                                            
            traceback.print_exc()
    if "C" in which:
        try:
            res["C"] = exp_C(T, real_test_ls, models)
        except Exception:                                            
            traceback.print_exc()
    if "D" in which:
        try:
            res["D"] = exp_D(real_test_ls, models)
        except Exception:                                            
            traceback.print_exc()
    print("\nDone. CSVs in", OUT)
    return res
