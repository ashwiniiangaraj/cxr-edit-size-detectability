# evaluation/compression_ladder.py
from pathlib import Path
import numpy as np, pandas as pd
from experiments import ladder_scoring as ls
from evaluation.degradations import Degradation, JPEG

OUT = Path(ls.OUT_DIR) / "compression"


def _scorer(cond, tag):
    def fn(slug, model, seed, real, lad, overwrite=False):
        cache = OUT / tag / f"{slug}__{model}_seed{seed}.csv"
        if cache.exists() and not overwrite:
            return pd.read_csv(cache, dtype={"dicom_id": str})
        det = ls.load_detector(model, slug, seed)
        if det is None:
            return None
        real, lad = real.reset_index(drop=True), lad.reset_index(drop=True)
        s_real = np.asarray(det.predict_scores(real, degrade=cond), float)       
        s_lad = np.asarray(det.predict_scores(lad, degrade=cond), float)
        out = pd.concat([
            pd.DataFrame(dict(role="real", group="Real", level="real", dicom_id=real["dicom_id"], score=s_real)),
            pd.DataFrame(dict(role="edit", group=lad["group"], level=lad["level"], dicom_id=lad["dicom_id"], score=s_lad))],
            ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False)
        print(f"  scored {tag}: {slug} {model} seed {seed}", flush=True)
        return out
    return fn


def run(quality=75, slugs=None, models=None, B=500):
    """Returns {"clean_files": R, "q<quality>": R} (same columns as ladder_results.csv)."""
    res, orig = {}, ls._scores_for
    try:
        for tag, cond in (("clean_files", Degradation()), (f"q{quality}", JPEG(quality))):
            ls._scores_for = _scorer(cond, tag)
            res[tag] = ls.score_ladder(slugs=slugs, models=models, B=B, save=False)
    finally:
        ls._scores_for = orig                                   
    OUT.mkdir(parents=True, exist_ok=True)
    for k, R in res.items():
        R.to_csv(OUT / f"ladder_{k}.csv", index=False)
    return res


def first_area(R, thr=0.70):
    """Smallest area with AUROC >= thr and lower bound > 0.5, per train x detector x series (full-image series excluded)."""
    R = R[~R["group"].str.lower().str.contains("full")]
    ok = R[(R["auroc"] >= thr) & (R["auroc_lo"] > 0.5)]
    return ok.groupby(["train", "detector", "group"])["area"].min()


def compare(res, quality=75, thr=0.70):
    q = f"q{quality}"
    base = res["clean_files"]
    base = base[~base["group"].str.lower().str.contains("full")]
    U = pd.concat([first_area(res["clean_files"], thr).rename("clean"), first_area(res[q], thr).rename(q)], axis=1)
    U = U.reindex(base.set_index(["train", "detector", "group"]).index.unique())      # NaN = never reached within 10%
    S = U.groupby(["train", "group"]).agg(clean_best=("clean", "min"), clean_n=("clean", "count"),
                                          q_best=(q, "min"), q_n=(q, "count"))
    M = pd.concat([res[k].groupby(["train", "group", "area"])["auroc"].mean().rename(k) for k in ("clean_files", q)], axis=1)
    M["change"] = M[q] - M["clean_files"]
    return U, S, M