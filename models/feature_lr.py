"""
Logistic-regression detectors on forensic features.
SpectralLR: 6 spectral features -> StandardScaler -> logistic regression (interpretable baseline).
HandcraftedLR: 42 features, with selection on training data only:
  "kw_epsilon" (Kruskal-Wallis epsilon^2, FDR-significant, top-k), "l1" (L1 logistic regression), "none" (all 42).
k and C are tuned by AUROC on the source validation set (real + training-family); the unseen family is never used.
With `degrade`, features are recomputed from the degraded image; stored CSV values apply to clean images only.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from forensics.feature_extractors import extract_feature_vector
from forensics.statistical_analysis import kruskal_table, transform_features
from models.base import BaseDetector
from utils.common import load_gray_float01


def _features_from_image(path, degrade):
    img = load_gray_float01(path)
    if degrade is not None:
        img = degrade(img)
    return extract_feature_vector(img)


def feature_frame(df: pd.DataFrame, degrade=None) -> pd.DataFrame:
    """Raw 42-feature table for `df`: stored values (clean) or recomputed (degraded)."""
    if degrade is None:
        return df[config.FEATURE_COLUMNS]
    rows = Parallel(n_jobs=config.DET_N_JOBS)(delayed(_features_from_image)(p, degrade)
                                              for p in df["prepared_path"])
    return pd.DataFrame(np.vstack(rows), columns=config.FEATURE_COLUMNS, index=df.index)


def _make_l1_lr(C: float):
    """L1 logistic regression that works on old and new scikit-learn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=config.LR_MAX_ITER,
                                      class_weight=config.LR_CLASS_WEIGHT)
        except Exception:                                          # noqa: BLE001
            return LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                      max_iter=config.LR_MAX_ITER, class_weight=config.LR_CLASS_WEIGHT)


class _FeatureLR(BaseDetector):
    stochastic = False

    def _candidate_sets(self, train_df) -> list[list[str]]:
        raise NotImplementedError

    def fit(self, train_df, val_df):
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        best = {"auc": -1.0}
        for cols in self._candidate_sets(train_df):
            Xtr_raw = transform_features(train_df, cols)
            scaler = StandardScaler().fit(Xtr_raw)
            Xtr, Xv = scaler.transform(Xtr_raw), scaler.transform(transform_features(val_df, cols))
            for C in config.LR_C_GRID:
                clf = LogisticRegression(C=C, class_weight=config.LR_CLASS_WEIGHT, max_iter=config.LR_MAX_ITER)
                clf.fit(Xtr, ytr)
                auc = roc_auc_score(yv, clf.decision_function(Xv))
                if auc > best["auc"] + 1e-9:       # strictly better: earlier (simpler) candidates win ties
                    best = {"auc": auc, "cols": cols, "scaler": scaler, "clf": clf, "C": C}
        self.cols, self.scaler, self.clf = best["cols"], best["scaler"], best["clf"]
        self.best_C, self.val_auroc = best["C"], best["auc"]
        return self

    def predict_scores(self, df, degrade=None):
        raw = feature_frame(df, degrade)
        return self.clf.predict_proba(self.scaler.transform(transform_features(raw, self.cols)))[:, 1]

    def info(self):
        return {"n_features": len(self.cols), "features": self.cols, "C": self.best_C,
                "source_val_auroc": round(self.val_auroc, 4)}

    def coefficients(self) -> pd.Series:
        """Standardised coefficients – the interpretability payoff of this model."""
        return pd.Series(self.clf.coef_[0], index=self.cols).sort_values(key=np.abs, ascending=False)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cols": self.cols, "scaler": self.scaler, "clf": self.clf,
                     "C": self.best_C, "val_auroc": self.val_auroc}, directory / "model.joblib")

    @classmethod
    def load(cls, directory, seed=42):
        d = joblib.load(Path(directory) / "model.joblib")
        obj = cls(seed)
        obj.cols, obj.scaler, obj.clf = d["cols"], d["scaler"], d["clf"]
        obj.best_C, obj.val_auroc = d["C"], d["val_auroc"]
        return obj


class SpectralLR(_FeatureLR):
    """5.1 — the six spectral features only."""
    name = "spectral_lr"

    def _candidate_sets(self, train_df):
        return [list(config.SPECTRAL_MODEL_FEATURES)]


class HandcraftedLR(_FeatureLR):
    """5.2 — all 42 features with training-data-only selection."""
    name = "handcrafted_lr"

    def _candidate_sets(self, train_df):
        feats, method = list(config.FEATURE_COLUMNS), config.HANDCRAFTED_SELECTION
        if method == "none":
            return [feats]
        if method == "kw_epsilon":
            # rank on TRAINING data only: real (0) vs synthetic (1)
            kw = kruskal_table(train_df, feats, group_col="label", alpha=config.HANDCRAFTED_FDR_ALPHA)
            ranked = kw[kw["significant"]]["feature"].tolist() or kw["feature"].tolist()
            sets = []
            for k in sorted(config.HANDCRAFTED_TOP_K_GRID):
                cand = ranked[:k]
                if cand not in sets:
                    sets.append(cand)
            self.selection_table = kw
            return sets
        if method == "l1":
            return [feats]            # handled in fit()
        raise ValueError(f"Unknown HANDCRAFTED_SELECTION={method!r}")

    def fit(self, train_df, val_df):
        if config.HANDCRAFTED_SELECTION != "l1":
            return super().fit(train_df, val_df)
        feats = list(config.FEATURE_COLUMNS)
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        scaler = StandardScaler().fit(transform_features(train_df, feats))
        Xtr, Xv = scaler.transform(transform_features(train_df, feats)), scaler.transform(transform_features(val_df, feats))
        best = {"auc": -1.0}
        for C in config.LR_C_GRID:
            clf = _make_l1_lr(C).fit(Xtr, ytr)
            if not np.any(clf.coef_):
                continue
            auc = roc_auc_score(yv, clf.decision_function(Xv))
            if auc > best["auc"] + 1e-9:
                best = {"auc": auc, "clf": clf, "C": C}
        if best["auc"] < 0:
            raise RuntimeError("L1 model removed every feature for all C values.")
        self.cols = [feats[i] for i in np.flatnonzero(best["clf"].coef_[0])]
        sc = StandardScaler().fit(transform_features(train_df, self.cols))      # compact refit
        clf = LogisticRegression(C=best["C"], class_weight=config.LR_CLASS_WEIGHT, max_iter=config.LR_MAX_ITER)
        clf.fit(sc.transform(transform_features(train_df, self.cols)), ytr)
        self.scaler, self.clf, self.best_C, self.val_auroc = sc, clf, best["C"], best["auc"]
        return self
