"""
Soft-mask check on the saved ladder masks 

"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import config
import size_config as sc

SEED = 0


def _blur(arr01, sigma):
    im = Image.fromarray(np.uint8(np.clip(arr01, 0, 1) * 255)).filter(ImageFilter.GaussianBlur(float(sigma)))
    return np.asarray(im, dtype=np.float32) / 255.0


def _stats(A, Mw, M):
    """A: soft weights; Mw: weight of the binary mask per pixel (0-1, equal to M where already binary); M: bool mask."""
    return dict(
        sumA_over_M=float(A.sum() / Mw.sum()),
        out_share=float(A[~M].sum() / max(A.sum(), 1e-9)),
        in_partial=float((A[M] < 0.99).mean()) if M.any() else float("nan"),
    )


def _agg(rows):
    df = pd.DataFrame(rows)
    out = {"n": len(df)}
    for c in df.columns:
        if c in ("area_pct", "iou") or c in ("sumA_over_M", "out_share", "in_partial", "sigma_px"):
            out[c] = f"{df[c].mean():.3f} ({df[c].min():.3f}-{df[c].max():.3f})"
    return out


# ------------------------------------------------------------------ EditOnly
def editonly_level(level, n, rng):
    files = sorted((sc.DIFF_LADDER_ROOT / level / "masks").glob("*_edit_mask.png"))
    if not files:
        return None
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    final = int(config.DIFFUSION_FINAL_SIZE)
    rows = []
    for i in pick:
        m512 = (np.asarray(Image.open(files[i]).convert("L")) > 127)
        area = float(m512.mean())
        sigma = config.diffusion_paste_sigma(area)
        m = np.asarray(Image.fromarray(m512.astype(np.uint8) * 255).resize((final, final), Image.Resampling.BOX),
                       dtype=np.float32) / 255.0
        A = _blur(m, sigma)
        M = m >= 0.5
        M0 = m > 0                        # "outside M" = pixels with no mask coverage at all
        r = _stats(A, m, M)
        r["out_share"] = float(A[~M0].sum() / A.sum())
        r.update(area_pct=100 * area, sigma_px=sigma)
        rows.append(r)
    return _agg(rows)


# ------------------------------------------------------------------ Blob
def blob_level(level, n, rng):
    root = sc.IP_LADDER_ROOT / level / "blob"
    meta_p = root / "metadata" / "blob_synthetic_metadata.csv"
    if not meta_p.exists():
        return None
    meta = pd.read_csv(meta_p)
    meta["mask_name"] = meta["mask_path"].map(lambda p: Path(str(p)).name)
    thr = float(config.IP_CONFIG["blob"]["mask_threshold"])
    pick = rng.choice(len(meta), size=min(n, len(meta)), replace=False)
    rows = []
    for i in pick:
        r0 = meta.iloc[i]
        mp = Path(str(r0["mask_path"]))
        if not mp.exists():               # path may be from another machine: look it up by name
            hits = list((root / "masks").glob(f"*/{r0['mask_name']}"))
            if not hits:
                continue
            mp = hits[0]
        saved = np.asarray(Image.open(mp).convert("L")) > 127
        h, w = saved.shape
        yy, xx = np.mgrid[:h, :w]
        cy, cx = yy[saved].mean(), xx[saved].mean()           # a disc: centroid = centre
        radius, sigma = float(r0["radius_pixels"]), float(r0["sigma"])
        disc = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2).astype(np.float32)
        A = _blur(disc, sigma)
        M = A >= thr
        iou = float((M & saved).sum() / max((M | saved).sum(), 1))
        r = _stats(A, M.astype(np.float32), M)
        r.update(area_pct=100 * saved.mean(), iou=iou)
        rows.append(r)
    return _agg(rows) if rows else None


def run(n_per_level=40, save_to=None):
    rng = np.random.default_rng(SEED)
    out = []
    for series, fn, levels in (("EditOnly", editonly_level, list(sc.DIFF_LEVELS)),
                               ("Blob", blob_level, list(sc.IP_LEVELS))):
        for lv in levels:
            res = fn(lv, n_per_level, rng)
            if res is None:
                print(f"{series:9s} {lv:5s}: masks not found (check size_config paths)")
                continue
            out.append(dict(series=series, level=lv, **res))
    df = pd.DataFrame(out)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(df.to_string(index=False))
    save_to = Path(save_to) if save_to else Path(sc.IP_LADDER_ROOT) / "soft_mask_check.csv"
    df.to_csv(save_to, index=False)
    print("saved:", save_to)
    return df