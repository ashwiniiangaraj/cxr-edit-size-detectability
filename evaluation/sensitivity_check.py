"""
SENSITIVITY CHECK  
====================================================
do the main results depend on a Stage 2 preprocessing choice (e.g. JPEG quality 95 -> 75, or no JPEG)?
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

_RELOAD = ["config", "construction.preprocess", "forensics.extract_features",
           "experiments.runner", "experiments.splits"]


def run(variant="q75", overrides=None, models=None, seed=None, n_jobs=None) -> dict:
    overrides = overrides or config.SENSITIVITY_VARIANTS[variant]
    models = models or config.SENSITIVITY_MODELS_FAST
    seed = seed if seed is not None else getattr(config, "SENSITIVITY_SEED", 42)

    for name in _RELOAD:
        importlib.reload(sys.modules.get(name) or importlib.import_module(name))
    import config as cfg                                   # re-read after reload
    from construction import preprocess as pre
    import forensics.extract_features as ef
    from experiments import runner

    root = Path(cfg.PROJECT_ROOT) / "sensitivity" / variant
    root.mkdir(parents=True, exist_ok=True)
    redirect = {"DET_PREPARED_DIR": root / "prepared", "DET_PREPARED_MANIFEST": root / "prepared_manifest.csv",
                "FORENSIC_FEATURE_CSV": root / "per_image_forensic_characteristics.csv",
                "DET_RESULTS_CSV": root / "results.csv", "DET_PREDICTIONS_CSV": root / "predictions.csv",
                "DET_MODEL_DIR": root / "models", "DET_EMBEDDING_DIR": root / "embeddings",
                "SAVE_FITTED_MODELS": False, "PREP_OVERWRITE": True, **overrides}
    if n_jobs is not None:
        redirect["DET_N_JOBS"] = n_jobs
    redirect = {k: v for k, v in redirect.items() if hasattr(cfg, k) or k in overrides}   # only existing settings
    originals = {k: getattr(cfg, k) for k in redirect if hasattr(cfg, k)}
    watch = [Path(cfg.DET_PREPARED_MANIFEST), Path(cfg.FORENSIC_FEATURE_CSV), Path(cfg.DET_RESULTS_CSV)]
    mtime0 = {p: p.stat().st_mtime for p in watch if p.exists()}
    orig_res = pd.read_csv(cfg.DET_RESULTS_CSV)            # original grid, read BEFORE redirecting

    t0 = time.time()
    try:
        for k, v in redirect.items():
            setattr(cfg, k, v)
        print("variant:", variant, overrides, "->", root)
        prepared = pre.prepare_dataset()
        print(f"[{time.time()-t0:.0f}s] stage 2 done")
        features = ef.extract_all_features(prepared, n_jobs=n_jobs) if n_jobs else ef.extract_all_features(prepared)
        print(f"[{time.time()-t0:.0f}s] features done", getattr(features, "shape", ""))
        master = runner.load_modelling_table()
        res = runner.run_all_experiments(models=models, seeds=[seed], master=master)
        print(f"[{time.time()-t0:.0f}s] grid done, {len(res)} rows")
    finally:
        for k, v in originals.items():
            setattr(cfg, k, v)                             # always restore
    changed = [p.name for p, t in mtime0.items() if p.stat().st_mtime != t]
    print("original files modified:", changed or "none")

    out = compare(orig_res, res, models, seed)
    out["comparison"].to_csv(root / "comparison_vs_original.csv")
    print("saved ->", root / "comparison_vs_original.csv")
    return out


def compare(orig_res: pd.DataFrame, new_res: pd.DataFrame, models, seed) -> dict:
    keep = ["test_in", "test_cross"]

    def tab(df):
        d = df[df.role.isin(keep) & df.model.isin(models)]
        if "seed" in d:
            d = d[d.seed == seed]
        return d.groupby(["experiment", "model", "role"]).auroc.mean()

    C = pd.concat({"original": tab(orig_res), "variant": tab(new_res)}, axis=1).dropna()
    C["diff"] = C["variant"] - C["original"]
    S = C.groupby(["model", "role"]).agg(original=("original", "mean"), variant=("variant", "mean"),
                                         mean_diff=("diff", "mean"),
                                         max_abs_diff=("diff", lambda x: x.abs().max()))
    S["rank_corr"] = [C.xs((m, r), level=("model", "role"))[["original", "variant"]].corr(method="spearman").iloc[0, 1]
                      for m, r in S.index]
    R = C.reset_index()
    R["train"] = R["experiment"].str.split("->").str[0]
    R["test"] = R["experiment"].str.split("->").str[1]
    cross = R[R.role == "test_cross"]
    by_test = cross.groupby("test")[["original", "variant", "diff"]].mean().round(3)
    by_train_in = (R[R.role == "test_in"].drop_duplicates(["train", "model"])
                   .groupby("train")[["original", "variant", "diff"]].mean().round(3))
    print(f"\n{len(C)} matched rows (experiment x model x role)")
    print(S.round(3).to_string())
    print("\nCross-family change by TEST family:\n", by_test.to_string())
    print("\nIn-family change by TRAIN family:\n", by_train_in.to_string())
    print("\ncells with cross-family AUROC < 0.5: original", int((cross.original < 0.5).sum()),
          "| variant", int((cross.variant < 0.5).sum()), "of", len(cross))
    return {"comparison": C, "summary": S, "by_test_family": by_test, "in_family": by_train_in}