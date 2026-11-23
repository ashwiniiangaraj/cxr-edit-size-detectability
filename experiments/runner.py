"""
EXPERIMENT RUNNER
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config
from evaluation.metrics import choose_threshold, evaluate_scores
from experiments.splits import (Experiment, build_experiments, class_balance_warning,
                                       experiments_overview)
from models import get_detector_class
from utils.common import ensure_parent, seed_everything

ROLES = ("val", "test_in", "test_cross")


# ------------------------------------------------------------------ data
def load_modelling_table(path=None, check_files: bool = True) -> pd.DataFrame:
    """Manifest + prepared paths + the 42 features, one row per image (Stage 3 output)."""
    df = pd.read_csv(path or config.FORENSIC_FEATURE_CSV, dtype={"subject_id": str, "study_id": str})
    need = ["image_id", "frequency_group", "label", "split", "subject_id", "prepared_path",
            *config.FEATURE_COLUMNS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"modelling table lacks columns {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if not np.isfinite(df[list(config.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("the modelling table contains NaN / inf feature values")
    if check_files:
        gone = [p for p in df["prepared_path"] if not Path(p).is_file()]
        if gone:
            raise FileNotFoundError(f"{len(gone)} prepared images are missing, e.g. {gone[:3]}")
    return df


def model_dir(exp: Experiment, model: str, seed: int) -> Path:
    return Path(config.DET_MODEL_DIR) / exp.slug / f"{model}_seed{seed}"


def run_signature(exp: Experiment) -> str:
    """Fingerprint of everything that decides a run's data and decision rule (not the model code)."""
    return f"{exp.design_hash()}-{config.THRESHOLD_RULE}-b{config.BOOTSTRAP_N}"


# ------------------------------------------------------------------ results file
def _done_keys() -> set:
    path = Path(config.DET_RESULTS_CSV)
    if not path.exists():
        return set()
    r = pd.read_csv(path, usecols=lambda c: c in {"experiment", "model", "seed", "role", "design"})
    if "design" not in r.columns:
        raise RuntimeError(f"{path} was written by an older runner (no 'design' column). "
                           "Move it away or call run_all_experiments(resume=False).")
    return set(zip(r["experiment"], r["model"], r["seed"].astype(int), r["role"], r["design"]))


def _append(df: pd.DataFrame, path) -> None:
    path = ensure_parent(path)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ------------------------------------------------------------------ one run
def run_single(exp: Experiment, model_name: str, seed: int, save_model=None):
    """Run one (experiment, model, seed) -> (metric rows, prediction frame, fitted detector)."""
    save_model = config.SAVE_FITTED_MODELS if save_model is None else save_model
    seed_everything(seed)
    det = get_detector_class(model_name)(seed=seed)
    design = run_signature(exp)

    t0 = time.time()
    det.fit(exp.train_df, exp.val_df)
    fit_seconds = time.time() - t0

    s_val = det.predict_scores(exp.val_df)
    thr = choose_threshold(exp.val_df["label"].to_numpy(), s_val)          

    rows, preds = [], []
    for role, df in (("val", exp.val_df), ("test_in", exp.test_in_df), ("test_cross", exp.test_cross_df)):
        s = s_val if role == "val" else det.predict_scores(df)
        s = np.asarray(s, dtype=float)
        if s.shape != (len(df),) or not np.isfinite(s).all():
            raise ValueError(f"{model_name} returned invalid scores for {role} (shape {s.shape})")
        m = evaluate_scores(df["label"].to_numpy(), s, thr, with_ci=(role != "val"))
        rows.append({"experiment": exp.name, "train_family": exp.train_family,
                     "test_family": exp.test_family if role == "test_cross" else exp.train_family,
                     "model": model_name, "seed": seed, "role": role, "design": design,
                     "fit_seconds": round(fit_seconds, 1), **m})
        preds.append(pd.DataFrame({"experiment": exp.name, "model": model_name, "seed": seed, "role": role,
                                   "design": design, "image_id": df["image_id"].values,
                                   "frequency_group": df["frequency_group"].values,
                                   "label": df["label"].values, "score": s, "threshold": thr}))
    if save_model:
        d = model_dir(exp, model_name, seed)
        d.mkdir(parents=True, exist_ok=True)
        det.save(d)
        (d / "meta.json").write_text(json.dumps(
            {"experiment": exp.name, "model": model_name, "seed": seed, "threshold": thr,
             "design": design, "info": det.info()}, indent=2, default=str))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), det


# ------------------------------------------------------------------ the grid
def run_all_experiments(models=None, seeds=None, family_set_name=None, resume=True, master=None,
                        dry_run=False, raise_on_failure=False) -> pd.DataFrame:
    """Run the full grid. Safe to interrupt and re-run (finished runs are skipped).

    dry_run=True       build and check the experiments, print the plan, fit nothing.
    raise_on_failure   True = stop at the first failed run; False = log it and continue.
    """
    models = list(models or config.MODELS_TO_RUN)
    seeds = list(seeds or config.EXPERIMENT_SEEDS)
    classes = {m: get_detector_class(m) for m in models}            
    master = load_modelling_table() if master is None else master
    exps = build_experiments(master, family_set_name)
    print(experiments_overview(exps).to_string(index=False))
    warn = class_balance_warning(exps)
    if warn:
        print("\nNOTE:", warn)

    plan = [(exp, m, s) for exp in exps for m in models
            for s in (seeds if classes[m].stochastic else seeds[:1])]
    print(f"\n{len(exps)} experiments x {len(models)} models -> {len(plan)} runs "
          f"(stochastic models use seeds {seeds}, deterministic ones seed {seeds[0]})\n")
    if dry_run:
        return pd.DataFrame([{"experiment": e.name, "model": m, "seed": s} for e, m, s in plan])

    if not resume:
        for p in (config.DET_RESULTS_CSV, config.DET_PREDICTIONS_CSV):
            Path(p).unlink(missing_ok=True)
    done = _done_keys() if resume else set()

    failures, t_start, n_run = [], time.time(), 0
    for k, (exp, model_name, seed) in enumerate(plan, 1):
        tag = f"[{k}/{len(plan)}] {exp.name:<28s} {model_name:<18s} seed={seed}"
        design = run_signature(exp)
        if all((exp.name, model_name, seed, r, design) in done for r in ROLES):
            print(tag, "-> skipped (done)")
            continue
        try:
            rows, preds, _ = run_single(exp, model_name, seed)
        except Exception:                                             
            print(tag, "-> FAILED")
            traceback.print_exc()
            failures.append((exp.name, model_name, seed))
            if raise_on_failure:
                raise
            continue
        _append(preds, config.DET_PREDICTIONS_CSV)                    
        _append(rows, config.DET_RESULTS_CSV)                         
        n_run += 1
        x = rows[rows.role == "test_cross"].iloc[0]
        i = rows[rows.role == "test_in"].iloc[0]
        elapsed = time.time() - t_start
        print(f"{tag} -> AUROC in={i.auroc:.3f} cross={x.auroc:.3f} ({x.fit_seconds:.0f}s) "
              f"| elapsed {elapsed / 60:.1f} min")

    if failures:
        print(f"\n{len(failures)} run(s) FAILED - they are retried when you run the cell again:")
        for f in failures:
            print("   ", *f)
    print(f"\nFinished: {n_run} run(s) executed, {len(plan) - n_run - len(failures)} already done, "
          f"{len(failures)} failed.")
    if not Path(config.DET_RESULTS_CSV).exists():
        return pd.DataFrame()
    res = pd.read_csv(config.DET_RESULTS_CSV)
    dup = res.duplicated(["experiment", "model", "seed", "role", "design"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicated result rows in {config.DET_RESULTS_CSV}")
    return res
