"""
Smoothness controls 2 (stage4_experiments/smoothness_controls2.py; needs smoothness_controls.py, provenance_control.py).
Follow-up to smoothness_controls.py: tests whether the patch-blur result is specific to smoothing or holds for any
local change, and whether real edits and plain patch perturbations are scored alike image by image.
B2  Fine blur dose-response: real images blurred with sigma 0.35-0.7, scored by the fitted detectors.
D2  Other local perturbations: 0.5-10 % soft-edged patch of a real test image altered by blur (0.7 / 1.5 / 3.0),
    additive noise, sharpening or brightness +/-; same JPEG q95 -> PNG step as all images. Reports AUROC vs real
    and paired win rate per detector, trained family and area.
E   Per-image agreement: per detector, correlation between (score of the real ladder edit - score of its source)
    and (score of the patch-perturbed image - score of the same source), using cached ladder scores.
Usage: from evaluation import smoothness_controls2 as s2; out = s2.run_extra()  (~20-40 min; models=[...] to subset).
Each part runs in try/except. Outputs: <ladder_scoring dir>/smoothness_controls2/*.csv.
"""
from __future__ import annotations

import traceback
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import config
from experiments import ladder_scoring as ls
from evaluation import provenance_control as pc
from evaluation import smoothness_controls as sm
from experiments.runner import load_modelling_table

OUT = Path(ls.OUT_DIR) / "smoothness_controls2"
_ID = pc._ID
AREAS = (0.01, 0.05, 0.10)
SIGMAS_B2 = (0.35, 0.4, 0.45, 0.5, 0.7)
OPS = ["blur0.7", "blur1.5", "blur3.0", "noise8", "sharpen", "bright+12", "bright-12"]


# ------------------------------------------------------------------ patch operations
def _mask(shape, area, rng):
    H, W = shape
    r = np.sqrt(area * H * W / np.pi)
    x0, x1, y0, y1 = 0.15 * W + r, 0.85 * W - r, 0.125 * H + r, 0.875 * H - r      
    cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
    yy, xx = np.mgrid[0:H, 0:W]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 0.10 * (2 * r) / 2.563)                       


def patch_op(a: np.ndarray, area: float, op: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = _mask(a.shape, area, rng)
    af = a.astype(np.float32)
    if op.startswith("blur"):
        alt = cv2.GaussianBlur(af, (0, 0), float(op[4:]))
    elif op == "noise8":
        alt = af + rng.normal(0, 8.0, af.shape).astype(np.float32)
    elif op == "sharpen":
        alt = af + 1.5 * (af - cv2.GaussianBlur(af, (0, 0), 1.5))
    elif op.startswith("bright"):
        alt = af + float(op[6:])
    else:
        raise ValueError(op)
    out = m * alt + (1 - m) * af
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ shared setup
def _setup():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load_modelling_table()
    real_all = sm._paths(T[T["frequency_group"] == config.REAL_GROUP].copy())
    real_test_ls, _ = ls.load_inputs(T)
    real_test_ls = real_test_ls.reset_index(drop=True)
    rt = real_all.set_index("image_id").loc[real_test_ls["image_id"]].reset_index()      
    arrays = {i: sm._load256(p) for i, p in zip(rt["image_id"], rt["path"])}
    return T, real_all, real_test_ls, rt, arrays


def score_images(sets: dict, real: pd.DataFrame, models=None, seeds=(42,)):
    """{(train, model, seed): {'real': scores, set_name: scores}} - per-image scores of every fitted detector."""
    out = {}
    for tl, m, sd, det in sm._fitted(models, seeds):
        d = {"real": np.asarray(det.predict_scores(real, degrade=_ID), float)}
        for name, df in sets.items():
            d[name] = np.asarray(det.predict_scores(df, degrade=_ID), float)
        out[(tl, m, sd)] = d
        print(f"  scored {tl:>17s} {m:>16s} seed {sd}")
    return out


def summarise_scores(S: dict) -> pd.DataFrame:
    rows = []
    for (tl, m, sd), d in S.items():
        for name, s in d.items():
            if name == "real":
                continue
            rows.append(dict(train=tl, model=m, seed=sd, set=name, auroc=ls.auc(s, d["real"]), win=ls.win(s, d["real"])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B2
def exp_B2(models=None):
    print("\n=== B2. fitted detectors on real images blurred with sigma 0.35 ... 0.7 ===")
    T, real_all, real_test_ls, rt, arrays = _setup()
    try:
        sm.calibrate(real_all[real_all["split"] == "train"], arrays_all(real_all), T,
                     grid=(0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1.0))
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
    sets = {f"blur{s:g}": sm.make_control_df(rt, arrays, "blur", s, f"b2_blur{s:g}", label=1) for s in SIGMAS_B2}
    S = score_images(sets, real_test_ls, models)
    R = summarise_scores(S)
    R.to_csv(OUT / "B2_blur_dose_response.csv", index=False)
    print("\nB2 - AUROC vs real (0.5 = no effect), mean over trained families:")
    print(R.groupby(["model", "set"])["auroc"].mean().unstack("set")[[f"blur{s:g}" for s in SIGMAS_B2]].round(3))
    print("B2 - paired win rate vs the unblurred original:")
    print(R.groupby(["model", "set"])["win"].mean().unstack("set")[[f"blur{s:g}" for s in SIGMAS_B2]].round(3))
    ref = None
    try:
        r = pd.read_csv(config.DET_RESULTS_CSV)
        r = r[(r["role"] == "test_cross") & (r["seed"] == 42)]
        col = "auroc" if "auroc" in r.columns else [c for c in r.columns if "auroc" in c.lower()][0]
        ref = r.groupby(["model", "test_family"])[col].mean().unstack("test_family").round(3)
        print("\nreference: main-grid mean cross-family AUROC per detector and test family (feature models):")
        print(ref.loc[[m for m in ("spectral_lr", "handcrafted_lr", "lbp_lr") if m in ref.index]])
    except Exception as e:                                               # noqa: BLE001
        print("(no grid reference:", repr(e), ")")
    return R


def arrays_all(real_all):
    return {i: sm._load256(p) for i, p in zip(real_all["image_id"], real_all["path"])}


# ------------------------------------------------------------------ D2
def exp_D2(models=None, areas=AREAS, ops=OPS):
    print("\n=== D2. patch of a REAL test image altered in different ways (no generator) ===")
    T, real_all, real_test_ls, rt, arrays = _setup()
    sets = {}
    for op in ops:
        for ar in areas:
            d = real_test_ls.copy().reset_index(drop=True)
            tag = f"{op}_{ar:g}"
            d["image_id"] = f"d2_{tag}__" + real_test_ls["image_id"].astype(str).to_numpy()
            d["prepared_path"] = [str(OUT / "images" / tag / f"{i}.png") for i in d["image_id"]]
            for src_id, dst in zip(real_test_ls["image_id"], d["prepared_path"]):
                if not Path(dst).exists():
                    seed = zlib.crc32(str(src_id).encode()) + int(ar * 10000)        # same placement for every op
                    a = patch_op(arrays[src_id], ar, op, seed)
                    sm._write(sm._jpeg_roundtrip(a), Path(dst))
            sets[tag] = d
    S = score_images(sets, real_test_ls, models)
    R = summarise_scores(S)
    R[["op", "area"]] = R["set"].str.rsplit("_", n=1, expand=True)
    R.to_csv(OUT / "D2_patch_perturbations.csv", index=False)
    order = [str(a) for a in areas]
    for metric, label in (("auroc", "AUROC vs real"), ("win", "paired win rate vs unperturbed original")):
        print(f"\nD2 - {label}, mean over detectors, by trained family:")
        print(R.groupby(["train", "op", "area"])[metric].mean().unstack("area")[order].round(3))
    print("\nD2 - AUROC by detector (mean over trained families), area 5%:")
    print(R[R["area"] == "0.05"].groupby(["model", "op"])["auroc"].mean().unstack("op").round(3))
    return R, S, real_test_ls


# ------------------------------------------------------------------ E
def _to_area(x):
    try:
        v = float(str(x).replace("%", ""))
    except ValueError:
        return np.nan
    return v / 100 if (v > 0.5 or "%" in str(x)) else v


def exp_E(S, real_test_ls, areas=AREAS, ops=OPS, seeds=(42,)):
    print("\n=== E. image-by-image agreement: real ladder edit vs plain patch perturbation ===")
    slugs = ls.train_slugs()
    rows = []
    shown = False
    for (tl, m, sd), d in S.items():
        f = Path(ls.OUT_DIR) / "scores" / f"{slugs[tl]}__{m}_seed{sd}.csv"
        if not f.exists():
            continue
        L = pd.read_csv(f, dtype={"dicom_id": str})
        if not shown:
            print("  ladder levels in the cache:", sorted(L[L['role'] == 'edit']['level'].astype(str).unique())[:12], "...")
            shown = True
        base = L[L["role"] == "real"].set_index("dicom_id")["score"]
        e = L[L["role"] == "edit"].copy()
        e["area"] = e["level"].map(_to_area)
        e["delta"] = e["score"].to_numpy() - base.reindex(e["dicom_id"]).to_numpy()
        ids = real_test_ls["dicom_id"].astype(str).to_numpy()
        for ar in areas:
            sel = e[(e["area"] / ar - 1).abs() < 0.2]
            for series, g in sel.groupby("group"):
                g = g.drop_duplicates("dicom_id").set_index("dicom_id")["delta"]
                for op in ops:
                    s = d[f"{op}_{ar:g}"] - d["real"]
                    p = pd.Series(s, index=ids)
                    j = g.index.intersection(p.index)
                    if len(j) < 20 or g.loc[j].std() == 0 or p.loc[j].std() == 0:
                        continue
                    rho = spearmanr(g.loc[j], p.loc[j])[0]
                    rows.append(dict(train=tl, model=m, seed=sd, series=series, area=ar, op=op, spearman=rho, n=len(j)))
    R = pd.DataFrame(rows)
    if R.empty:
        print("  no overlap found (check the ladder score cache / level labels printed above)")
        return R
    R.to_csv(OUT / "E_per_image_agreement.csv", index=False)
    for ar in areas:
        print(f"\nE - mean Spearman correlation of score changes, area {ar:g}  (rows: trained family / ladder series; cols: patch op)")
        print(R[R["area"] == ar].groupby(["train", "series", "op"])["spearman"].mean().unstack("op").round(2))
    return R


# ------------------------------------------------------------------ driver
def run_extra(models=None, which="B2D2E"):
    res = {}
    if "B2" in which:
        try:
            res["B2"] = exp_B2(models)
        except Exception:                                                
            traceback.print_exc()
    if "D2" in which or "E" in which:
        try:
            R, S, real_test_ls = exp_D2(models)
            res["D2"] = R
            if "E" in which:
                try:
                    res["E"] = exp_E(S, real_test_ls)
                except Exception:                                        
                    traceback.print_exc()
        except Exception:                                                
            traceback.print_exc()
    print("\nDone. CSVs in", OUT)
    return res
