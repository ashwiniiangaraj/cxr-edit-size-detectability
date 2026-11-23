"""
PROVENANCE CONTROL  
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
import size_config as sc
from experiments import ladder_scoring as ls

try:
    from evaluation.degradations import Degradation
except ImportError:                                  # fall back to a local identity callable
    class Degradation:
        def __call__(self, img):
            return img
_ID = Degradation()   # identity degradation: forces EVERY model (incl. feature LRs) to recompute from prepared_path
L = Image.Resampling.LANCZOS
OUT = Path(ls.OUT_DIR) / "provenance_control"


def _jpeg_png(arr: np.ndarray, dst: Path):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=config.PREP_JPEG_QUALITY)
    buf.seek(0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)).save(dst, format="PNG")


def build_control(src, dst: Path, kind: str):
    S = config.PREP_IMAGE_SIZE                      # 256
    G = config.DIFFUSION_GENERATION_SIZE            # 512
    im = Image.open(src).convert("L")
    if kind == "direct":
        im = im.resize((S, S), L)
    elif kind == "via512":
        im = im.resize((G, G), L).resize((S, S), L)
    elif kind == "updown":
        im = im.resize((S, S), L).resize((G, G), L).resize((S, S), L)
    else:
        raise ValueError(kind)
    _jpeg_png(np.asarray(im, dtype=np.uint8), dst)


def build_sets(real: pd.DataFrame, kinds=("direct", "via512", "updown"), overwrite=False) -> dict:
    """{kind: dataframe} - copies of the Real test rows pointing at the control images, with NEW image_ids
    (the foundation-encoder embedding cache is keyed by image_id, so ids must not collide with the real ones)."""
    if "path" not in real.columns:                  # native file path comes from the master manifest
        m = pd.read_csv(config.DET_MASTER_MANIFEST, dtype=str)[["image_id", "path"]]
        real = real.merge(m, on="image_id", how="left")
    assert real["path"].notna().all(), "native paths missing for some Real test rows"
    sets = {}
    for k in kinds:
        d = real.copy()
        d["image_id"] = f"prov_{k}__" + real["image_id"].astype(str)
        d["prepared_path"] = [str(OUT / "images" / k / f"{i}.png") for i in d["image_id"]]
        for src, dst in zip(d["path"], d["prepared_path"]):
            if overwrite or not Path(dst).exists():
                build_control(src, Path(dst), k)
        sets[k] = d.reset_index(drop=True)
    return sets


def run(models=None, B=500, kinds=("direct", "via512", "updown"), overwrite=False, seeds=None):
    real, _ = ls.load_inputs()
    real = real.reset_index(drop=True)
    sets = build_sets(real, kinds, overwrite)
    draws = ls.make_draws(real, B, cluster="subject_id", seed=0)
    if models is None:                              # every model that has a fitted folder
        models = sorted({f.name.rsplit("_seed", 1)[0]
                         for s in ls.train_slugs().values()
                         for f in (Path(config.DET_MODEL_DIR) / s).glob("*_seed*")})
    rows = []
    for tl, slug in ls.train_slugs().items():
        for m in models:
            for sd in (seeds or ls.model_seeds(m, slug)):
                det = ls.load_detector(m, slug, sd)
                if det is None:
                    continue
                s_real = np.asarray(det.predict_scores(real, degrade=_ID), float)
                for k, d in sets.items():
                    s_c = np.asarray(det.predict_scores(d, degrade=_ID), float)
                    a = ls.auc(s_c, s_real)
                    w = ls.win(s_c, s_real)
                    ba = np.array([ls.auc(s_c[ix], s_real[ix]) for ix in draws])
                    bw = np.array([ls.win(s_c[ix], s_real[ix]) for ix in draws])
                    rows.append(dict(train=tl, model=m, seed=sd, control=k, auroc=a, win=w,
                                     auroc_lo=np.percentile(ba, 2.5), auroc_hi=np.percentile(ba, 97.5),
                                     win_lo=np.percentile(bw, 2.5), win_hi=np.percentile(bw, 97.5),
                                     max_abs_score_diff=float(np.abs(s_c - s_real).max())))
                print(f"{tl:>18s} {m:>16s} seed {sd}: " +
                      "  ".join(f"{r['control']}={r['auroc']:.2f}" for r in rows[-len(sets):]))
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "provenance_control_results.csv", index=False)
    return R


def summarise(R: pd.DataFrame, flag=0.55):
    """Per detector x control: AUROC / win averaged over train families and seeds, plus a flag."""
    g = (R.groupby(["model", "control"])[["auroc", "win", "auroc_lo", "auroc_hi"]].mean().round(3)
           .reset_index())
    g["flag"] = np.where((g["control"] != "direct") & (g["auroc_lo"] > 0.5) & (g["auroc"] >= flag), "SHORTCUT?", "")
    sanity = R[R["control"] == "direct"]["max_abs_score_diff"].max()
    print(f"\nSanity (direct control must reproduce baseline scores): max |score diff| = {sanity:.2e}")
    print(g.pivot(index="model", columns="control", values="auroc").round(3))
    print("\nwin rate vs own real version (0.5 = no preference):")
    print(g.pivot(index="model", columns="control", values="win").round(3))
    print("\nflagged:", g[g["flag"] != ""][["model", "control", "auroc", "auroc_lo"]].to_string(index=False) or "none")
    return g