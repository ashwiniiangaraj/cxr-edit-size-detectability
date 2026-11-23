"""
SIZE-LADDER SCORING (cross-model experiment)  
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config
import size_config as sc
from forensics import ladder_frequency as lf

OUT_DIR = Path(getattr(sc, "LADDER_SCORE_DIR", Path(sc.FIG_DIR).parent / "ladder_scoring"))
GROUPS = lf.GROUPS
SERIES_FAMILY = {"IP_Blob": "IP", "IP_CopyMove": "IP", "IP_TextureTransplant": "IP",
                 "Diffusion_full": "Diffusion", "Diffusion_paste": "Diffusion_EditOnly"}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def auc(pos, neg) -> float:
    """Rank-based AUROC: probability that an edited image scores above a real one (ties count half)."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(np.r_[pos, neg])
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def win(e, s) -> float:
    """Share of pairs in which the edited image scores above its OWN unedited source (ties count half)."""
    return float(((e > s) + 0.5 * (e == s)).mean()) if len(e) else float("nan")


# ----------------------------------------------------------------------------
# detectors (loaded, never trained)
# ----------------------------------------------------------------------------
def find_cls(name):
    import models
    for mi in pkgutil.iter_modules(models.__path__):
        mod = importlib.import_module(f"models.{mi.name}")
        for _, c in inspect.getmembers(mod, inspect.isclass):
            if getattr(c, "name", None) == name and hasattr(c, "load") and c.__module__ == mod.__name__:
                return c
    return None


def model_seeds(model: str, slug: str) -> list[int]:
    d = Path(config.DET_MODEL_DIR) / slug
    seeds = []
    for f in sorted(d.glob(f"{model}_seed*")):
        m = re.search(r"_seed(\d+)$", f.name)
        if m and f.is_dir():
            seeds.append(int(m.group(1)))
    return seeds


def load_detector(model: str, slug: str, seed: int):
    cls = find_cls(model)
    if cls is None and model in getattr(config, "CLIP_VARIANTS", {}):
        from models.foundation_lr import make_frozen_detector
        cls = make_frozen_detector(model)
    d = Path(config.DET_MODEL_DIR) / slug / f"{model}_seed{seed}"
    if cls is None or not d.exists():
        print(f"{model} [{slug}] seed {seed}: skipped (class found: {cls is not None}, folder exists: {d.exists()})")
        return None
    return cls.load(d, seed=seed)


def train_slugs() -> dict:
    out = {"IP-trained": sc.IP_TRAIN_SLUG}
    ds = getattr(sc, "DIFF_TRAIN_SLUGS", {})
    out.update(ds if isinstance(ds, dict) else {s: s for s in ds})
    return out


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def load_inputs(T: pd.DataFrame | None = None):
    if T is None:
        from experiments.runner import load_modelling_table
        T = load_modelling_table()
    real = T[(T["split"] == "test") & (T["frequency_group"] == config.REAL_GROUP)].copy()
    real["dicom_id"] = real["dicom_id"].astype(str)
    real = real.drop_duplicates("dicom_id").sort_values("dicom_id").reset_index(drop=True)
    F = pd.read_csv(lf.FEAT_CSV, dtype={"dicom_id": str})
    lad = F[(F["group"] != "Real") & (F["split"] == "test")].reset_index(drop=True)
    missing = sorted(set(lad["dicom_id"]) - set(real["dicom_id"]))
    if missing:
        raise ValueError(f"{len(missing)} ladder source images are not Real test images of the modelling table, "
                         f"e.g. {missing[:3]}")
    return real, lad


def make_draws(real: pd.DataFrame, B: int, cluster="subject_id", seed=0) -> list[np.ndarray]:
    """B index arrays into `real` (rows = source images). cluster="subject_id" resamples patients and keeps all of
    their images; any other column name, or a missing column, resamples single images."""
    rng = np.random.default_rng(seed)
    n = len(real)
    if cluster in real.columns and real[cluster].notna().all():
        members = [np.flatnonzero((real[cluster] == s).to_numpy()) for s in real[cluster].unique()]
        return [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(B)]
    return [rng.integers(0, n, n) for _ in range(B)]


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def _scores_for(slug, model, seed, real, lad, overwrite=False):
    """Per-image scores of one fitted detector (cached; recomputed if the ladder table is newer than the cache)."""
    cache = OUT_DIR / "scores" / f"{slug}__{model}_seed{seed}.csv"
    if cache.exists() and not overwrite and cache.stat().st_mtime >= Path(lf.FEAT_CSV).stat().st_mtime:
        return pd.read_csv(cache, dtype={"dicom_id": str})
    det = load_detector(model, slug, seed)
    if det is None:
        return None
    s_real = np.asarray(det.predict_scores(real.reset_index(drop=True)), dtype=float)
    s_lad = np.asarray(det.predict_scores(lad.reset_index(drop=True)), dtype=float)
    out = pd.concat([
        pd.DataFrame(dict(role="real", group="Real", level="real", dicom_id=real["dicom_id"], score=s_real)),
        pd.DataFrame(dict(role="edit", group=lad["group"], level=lad["level"], dicom_id=lad["dicom_id"], score=s_lad))],
        ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _cell_arrays(sdf: pd.DataFrame, real: pd.DataFrame, draws) -> dict:
    """{(group, level): dict(auc, win, boot_auc, boot_win)} for one detector and seed."""
    pos = {d: i for i, d in enumerate(real["dicom_id"])}
    s_real = sdf[sdf.role == "real"].set_index("dicom_id")["score"].reindex(real["dicom_id"]).to_numpy(float)
    out = {}
    for (g, lv), d in sdf[sdf.role == "edit"].groupby(["group", "level"]):
        e = np.full(len(real), np.nan)
        e[[pos[x] for x in d["dicom_id"]]] = d["score"].to_numpy()
        m = ~np.isnan(e)
        ba, bw = np.empty(len(draws)), np.empty(len(draws))
        for b, idx in enumerate(draws):
            eb, rb = e[idx], s_real[idx]
            mb = ~np.isnan(eb)
            ba[b] = auc(eb[mb], rb)
            bw[b] = win(eb[mb], rb[mb])
        out[(g, lv)] = dict(auc=auc(e[m], s_real), win=win(e[m], s_real[m]), n=int(m.sum()), boot_auc=ba, boot_win=bw)
    return out


def score_ladder(slugs: dict | None = None, models=None, B=None, cluster="subject_id", T=None, seed=0,
                 overwrite=False, save=True) -> pd.DataFrame:
    """Score every ladder series with every fitted detector of every trained family in `slugs`.
    Returns one row per trained family x detector x series x level (see the module docstring)."""
    B = B or getattr(sc, "BOOTSTRAP_B", 1000)
    slugs = slugs or train_slugs()
    models = models or sc.MODELS
    real, lad = load_inputs(T)
    areas = lad.groupby(["group", "level"])["area"].mean()
    draws = make_draws(real, B, cluster, seed)
    print(f"{len(real)} real test images, {len(lad)} ladder images, {B} bootstrap draws (cluster={cluster}), "
          f"{len(draws[0])} images per draw")
    rows = []
    for label, slug in slugs.items():
        for model in models:
            per_seed = []
            for sd in model_seeds(model, slug):
                sdf = _scores_for(slug, model, sd, real, lad, overwrite)
                if sdf is not None:
                    per_seed.append(_cell_arrays(sdf, real, draws))
            if not per_seed:
                print(f"{label} / {model}: no fitted detector found")
                continue
            for cell in per_seed[0]:
                A = np.stack([p[cell]["boot_auc"] for p in per_seed]).mean(0)
                W = np.stack([p[cell]["boot_win"] for p in per_seed]).mean(0)
                pts = np.array([p[cell]["auc"] for p in per_seed])
                q = lambda x: np.nanpercentile(x, [2.5, 97.5])
                bias = float(np.nanmean(A) - pts.mean())     # duplicated source images double-weight their own twin pair
                rows.append(dict(train=label, slug=slug, detector=model, group=cell[0], level=cell[1],
                                 area=float(areas[cell]), n=per_seed[0][cell]["n"], n_seeds=len(per_seed),
                                 auroc=float(pts.mean()), auroc_lo=q(A)[0] - bias, auroc_hi=q(A)[1] - bias, boot_bias=bias,
                                 win=float(np.mean([p[cell]["win"] for p in per_seed])), win_lo=q(W)[0], win_hi=q(W)[1],
                                 seed_sd=float(pts.std(ddof=1)) if len(pts) > 1 else 0.0))
            print(f"{label} / {model}: {len(per_seed)} seed(s) done")
    R = pd.DataFrame(rows).sort_values(["train", "detector", "group", "area"]).reset_index(drop=True)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        R.to_csv(OUT_DIR / "ladder_results.csv", index=False)
        print(f"Saved {len(R)} rows -> {OUT_DIR / 'ladder_results.csv'}")
    return R


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------
def usable_table(R: pd.DataFrame, thr=None) -> pd.DataFrame:
    """Per trained family x detector x series:"""
    thr = thr or getattr(sc, "USABLE_AUROC", 0.70)
    rows = []
    for (tr, det, g), d in R.groupby(["train", "detector", "group"]):
        d = d.sort_values("area")
        a, y = d["area"].to_numpy(float), d["auroc"].to_numpy(float)
        hit = d[(d["auroc"] >= thr) & (d["auroc_lo"] > 0.5)]
        ok = np.flatnonzero(y >= thr)
        cross = np.nan
        if len(ok):
            i = ok[0]
            cross = float(a[0]) if i == 0 else float(np.exp(np.interp(thr, [y[i - 1], y[i]], np.log([a[i - 1], a[i]])))) if y[i] > y[i - 1] else float(a[i])
        rows.append(dict(train=tr, detector=det, group=g, usable=float(hit["area"].iloc[0]) if len(hit) else np.nan,
                         crossing=cross, max_area=float(a.max()), auroc_at_max=float(y[-1])))
    return pd.DataFrame(rows)


def pretty(U: pd.DataFrame, col="usable") -> pd.DataFrame:
    """Pivot detector x series per trained family with text such as '2.0%' or '>10%'."""
    f = lambda r: f"{100 * r[col]:.1f}%" if np.isfinite(r[col]) else f">{100 * r.max_area:.0f}%"
    U = U.assign(text=U.apply(f, axis=1))
    return U.pivot_table(index=["train", "detector"], columns="group", values="text", aggfunc="first")[
        [g for g in GROUPS if g in set(U["group"])]]


def compare_with_base(R: pd.DataFrame, results_csv=None, level_area=0.05) -> pd.DataFrame:
    path = results_csv or getattr(config, "DET_RESULTS_CSV")
    B = pd.read_csv(path)
    need = {"train_family", "test_family", "model", "role", "auroc"}
    if not need <= set(B.columns):
        raise KeyError(f"results file lacks {sorted(need - set(B.columns))}; columns are {list(B.columns)}")
    B = B.assign(Y=np.where(B["role"] == "test_in", B["train_family"], B["test_family"]))
    base = B[B["role"].isin(["test_in", "test_cross"])].groupby(["train_family", "Y", "model"])["auroc"].mean()
    five = R[(R["area"] - level_area).abs() < 0.002]
    rows = []
    for r in five.itertuples():
        X, Y = r.slug.split("__to__")[0], SERIES_FAMILY.get(r.group)
        if (X, Y, r.detector) in base.index:
            rows.append(dict(train=r.train, detector=r.detector, series=r.group, ladder_auroc=r.auroc,
                             base_auroc=float(base[(X, Y, r.detector)]), diff=r.auroc - float(base[(X, Y, r.detector)])))
    return pd.DataFrame(rows).sort_values(["train", "series", "detector"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
_GKEY = {**sc.IP_TECH_OF, **sc.DIFF_VARIANT_OF}      # series -> colour key (size_config)


def plot_auroc_vs_size(R: pd.DataFrame, train: str, thr=None, show=True):
    """One panel per detector: AUROC vs edit area for each series, with the bootstrap band."""
    import matplotlib.pyplot as plt
    thr = thr or getattr(sc, "USABLE_AUROC", 0.70)
    d = R[R["train"] == train]
    dets = list(dict.fromkeys(d["detector"]))
    ncol = min(5, len(dets)); nrow = int(np.ceil(len(dets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, det in zip(axes.ravel(), dets):
        for g in GROUPS:
            s = d[(d["detector"] == det) & (d["group"] == g)].sort_values("area")
            if s.empty:
                continue
            c = sc.COLOURS.get(_GKEY[g], "#444")
            ax.fill_between(100 * s["area"], s["auroc_lo"], s["auroc_hi"], color=c, alpha=0.12, lw=0)
            ax.plot(100 * s["area"], s["auroc"], color=c, marker="o", ms=4, lw=1.5, label=g)
        ax.axhline(0.5, color="#888", lw=0.8); ax.axhline(thr, color="#888", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_title(det, fontsize=9); ax.grid(alpha=.2, lw=.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(dets):]:
        ax.axis("off")
    sizes = sorted(d["area"].round(4).unique())
    axes[-1][0].set_xticks([100 * a for a in sizes]); axes[-1][0].set_xticklabels([f"{100 * a:g}" for a in sizes], fontsize=7)
    axes[0][0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.supxlabel("edit area (% of image)", fontsize=9); fig.supylabel("AUROC", fontsize=9)
    fig.suptitle(f"Detectors trained on {train} (5% base) scored on the size ladder", fontsize=10)
    fig.tight_layout(); lf._save(fig, f"ladder_auroc_vs_size_{re.sub(r'[^A-Za-z0-9]+', '_', train)}")
    if show: plt.show()
    return fig


def plot_heatmap(R: pd.DataFrame, train: str, show=True):
    """Detector x (series, level) AUROC heatmap for one trained family."""
    import matplotlib.pyplot as plt
    d = R[R["train"] == train]
    cols = [(g, a) for g in GROUPS for a in sorted(d[d.group == g]["area"].round(4).unique())]
    dets = list(dict.fromkeys(d["detector"]))
    M = np.full((len(dets), len(cols)), np.nan)
    for j, (g, a) in enumerate(cols):
        s = d[(d.group == g) & ((d.area - a).abs() < 1e-3)].set_index("detector")["auroc"]
        for i, det in enumerate(dets):
            M[i, j] = s.get(det, np.nan)
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 2.5, 0.38 * len(dets) + 1.8))
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(dets))); ax.set_yticklabels(dets, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{100 * a:g}%" for _, a in cols], fontsize=7, rotation=90)
    start = 0
    for g in GROUPS:
        k = sum(1 for c in cols if c[0] == g)
        if k:
            ax.text(start + (k - 1) / 2, -0.9, g.replace("_", " "), ha="center", va="bottom", fontsize=7)
            if start: ax.axvline(start - 0.5, color="white", lw=2)
            start += k
    fig.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    ax.set_title(f"Trained on {train}", fontsize=10, pad=22)
    fig.tight_layout(); lf._save(fig, f"ladder_auroc_heatmap_{re.sub(r'[^A-Za-z0-9]+', '_', train)}")
    if show: plt.show()
    return fig