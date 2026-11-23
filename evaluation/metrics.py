"""
EVALUATION METRICS
==================
Primary:   AUROC and AUPRC, each with a stratified-bootstrap 95% CI.
Secondary: balanced accuracy, accuracy, precision, recall, F1, specificity and
           the confusion matrix at a FIXED threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

import config


def choose_threshold(y, s, rule=None) -> float:
    """Decision threshold maximising Youden's J (= balanced accuracy up to a constant)."""
    rule = rule or config.THRESHOLD_RULE
    if len(np.unique(y)) < 2:
        return 0.5
    if rule in ("youden", "balanced_accuracy"):
        fpr, tpr, thr = roc_curve(y, s)
        t = float(thr[int(np.argmax(tpr - fpr))])
        return t if np.isfinite(t) else float(np.max(s)) + 1e-9
    raise ValueError(f"Unknown THRESHOLD_RULE={rule!r}")


def _stratified_resample(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(metric_fn, y, s, n_boot=None, seed=None, ci=None):
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    seed = config.BOOTSTRAP_SEED if seed is None else seed
    ci = config.BOOTSTRAP_CI if ci is None else ci
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_resample(y, rng)
        vals[b] = metric_fn(y[idx], s[idx])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, s_a, s_b, n_boot=None, seed=None):
    """Paired stratified bootstrap of AUROC(a) - AUROC(b) on the SAME images -> (diff, lo, hi, p)."""
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    seed = config.BOOTSTRAP_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_resample(y, rng)
        diffs[b] = roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx])
    d = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(d), float(lo), float(hi), float(min(p, 1.0))


def evaluate_scores(y, s, threshold, n_boot=None, seed=None, with_ci=True) -> dict:
    """All metrics for one (labels, scores) pair at a fixed threshold."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_synth": int((y == 1).sum()),
           "prevalence": float(y.mean()), "threshold": float(threshold)}
    if len(np.unique(y)) < 2:
        return {**out, "auroc": np.nan, "auprc": np.nan}

    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    if with_ci:
        out["auroc_lo"], out["auroc_hi"] = bootstrap_ci(roc_auc_score, y, s, n_boot, seed)
        out["auprc_lo"], out["auprc_hi"] = bootstrap_ci(average_precision_score, y, s, n_boot, seed)

    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "recall": float(tpr), "specificity": float(tnr), "precision": float(prec),
                "f1": float(2 * prec * tpr / max(prec + tpr, 1e-12)),
                "accuracy": float((tp + tn) / len(y)), "balanced_acc": float((tpr + tnr) / 2)})
    fpr_c, tpr_c, _ = roc_curve(y, s)
    out["balanced_acc_oracle"] = float(((tpr_c + (1 - fpr_c)) / 2).max())
    return out
