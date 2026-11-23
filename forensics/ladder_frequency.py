"""
frequency footprint of a local edit vs edit size (stage3_forensics/ladder_frequency.py).
Question: how does the footprint in the 42 features and the radial power spectrum change when only the
edit size changes? Descriptive and paired; no model is trained.
Per group, level and feature: d = f(edited) - f(its own Real source), with features in config.FEATURES_LOG10
log10-transformed. Reports shift_in_real_sd, paired_dz and auroc_dir; Wilcoxon signed-rank test with FDR
over the table; mean paired log10 ratio of the radial power profile per level.
Caveats: features are computed over the whole image, so the shift grows with edit area even at a constant
per-pixel effect (trend_table shows how close the growth is to proportional). Diffusion_full is a control
for the global VAE trace; Diffusion_paste isolates the local edit. Do not select features on test images.
Needs images prepared by ladder_prepare.py and size_config.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
import size_config as sc
from forensics import extract_features as ef, footprint as fpm, statistical_analysis as sa   # Stage 3a-c

OUT_DIR = Path(getattr(sc, "LADDER_FREQ_DIR", Path(config.FORENSIC_ANALYSIS_DIR) / "ladder_frequency"))
FEAT_CSV = OUT_DIR / "ladder_features.csv"
RES_CSV = OUT_DIR / "ladder_residual_features.csv"
PROF_NPZ = OUT_DIR / "ladder_radial_profiles.npz"
ERR_CSV = OUT_DIR / "ladder_feature_errors.csv"

GROUPS = ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant", "Diffusion_paste", "Diffusion_full"]
FAMILIES = ["spectral", "wavelet", "lbp", "srm", "noise"]
_GCOL = {**sc.IP_TECH_OF, **sc.DIFF_VARIANT_OF}      # series -> colour / marker key (size_config)
_FCOL = {"spectral": "#2a78d6", "wavelet": "#1baf7a", "lbp": "#eb6834", "srm": "#8a5cd0", "noise": "#6b6b6b"}
_FMARK = {"spectral": "o", "wavelet": "s", "lbp": "^", "srm": "D", "noise": "v"}


# ----------------------------------------------------------------------------
# 1. manifest and feature extraction
# ----------------------------------------------------------------------------
def read_ladder_manifests() -> pd.DataFrame:
    """The ladder_manifest.csv files written by the preparation step (IP and diffusion)."""
    parts = [pd.read_csv(Path(root) / "ladder_manifest.csv", dtype={"dicom_id": str})
             for root in (sc.PREP_IP_LADDER, sc.PREP_DIFF_LADDER) if (Path(root) / "ladder_manifest.csv").exists()]
    if not parts:
        raise FileNotFoundError("no ladder_manifest.csv found: prepare the ladder images first")
    return pd.concat(parts, ignore_index=True)


def build_manifest() -> pd.DataFrame:
    """Ladder rows + one Real row per source image (group 'Real', level 'real')."""
    man = read_ladder_manifests()
    man = man.assign(dicom_id=man["dicom_id"].astype(str))
    real = man.drop_duplicates("dicom_id")[["dicom_id", "split", "real_prepared_path"]].rename(
        columns={"real_prepared_path": "prepared_path"})
    real = real.assign(image_id="Real__" + real["dicom_id"], family="Real", group="Real", level="real",
                       target_area=0.0, achieved_area=0.0, label=0)
    cols = ["image_id", "family", "group", "level", "target_area", "achieved_area", "dicom_id", "split",
            "label", "prepared_path"]
    out = pd.concat([man[cols], real[cols]], ignore_index=True)
    assert out["image_id"].is_unique
    return add_stage6_columns(out)


_TECH, _VARIANT = sc.IP_TECH_OF, sc.DIFF_VARIANT_OF


def add_stage6_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the scoring stage (Stage 6) reads, so ONE ladder table serves both stages:
    frequency_group (Real rows: config.REAL_GROUP), tech (IP series), variant (diffusion series), area."""
    df = df.copy()
    is_real = df["group"] == "Real"
    df["frequency_group"] = np.where(is_real, config.REAL_GROUP, df["group"] + "@" + df["level"].astype(str))
    df["tech"] = df["group"].map(_TECH)
    df["variant"] = df["group"].map(_VARIANT)
    df["area"] = df["achieved_area"]
    return df


def _up_to_date(csv: Path, man: pd.DataFrame) -> bool:
    """True if `csv` holds exactly the images of `man` and is newer than every image file."""
    if not csv.exists():
        return False
    if set(pd.read_csv(csv, usecols=["image_id"])["image_id"]) != set(man["image_id"]):
        return False
    newest = max(Path(p).stat().st_mtime for p in man["prepared_path"])
    return csv.stat().st_mtime >= newest


def extract(n_jobs=None, residual=True, overwrite=False):
    """42 features + radial profile (+ the 6 residual-spectral features) for ladder and Real images.
    Re-uses the files if they are complete and newer than the images."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = build_manifest()
    if overwrite or not (_up_to_date(FEAT_CSV, man) and PROF_NPZ.exists()):
        feat = ef.extract_all_features(man, n_jobs, FEAT_CSV, PROF_NPZ, ERR_CSV)
    else:
        print(f"re-using {FEAT_CSV}")
        feat = ef.load_feature_table(FEAT_CSV)
        if "frequency_group" not in feat.columns:          # table written before the Stage 6 columns existed
            feat = add_stage6_columns(feat)
            feat.to_csv(FEAT_CSV, index=False)
    res = None
    if residual:
        if overwrite or not _up_to_date(RES_CSV, man):
            res = ef.extract_residual_features(man, n_jobs, RES_CSV)
        else:
            print(f"re-using {RES_CSV}")
            res = ef.load_residual_feature_table(RES_CSV)
    return feat, res


# ----------------------------------------------------------------------------
# 2. paired statistics
# ----------------------------------------------------------------------------
def paired_table(feat: pd.DataFrame, cols=None, family_of=None) -> pd.DataFrame:
    """One row per group x level x feature. See the module docstring for the columns."""
    feat = feat.reset_index(drop=True)
    cols = list(cols or config.FEATURE_COLUMNS)
    family_of = family_of or config.FEATURE_FAMILY_OF
    X = pd.DataFrame(sa.transform_features(feat, cols), columns=cols, index=feat.index)   # Stage 3 log10 rule
    is_real = (feat["group"] == "Real").to_numpy()
    R = X[is_real].set_index(feat.loc[is_real, "dicom_id"].astype(str).to_numpy())
    sd_real = R.std(ddof=1)
    rows = []
    for (g, lv), idx in feat[~is_real].groupby(["group", "level"]).groups.items():
        sub = feat.loc[idx]
        E = X.loc[idx].set_index(sub["dicom_id"].astype(str).to_numpy())
        common = E.index.intersection(R.index)
        E, Rp = E.loc[common], R.loc[common]
        D = E - Rp
        n = len(common)
        # single-feature AUROC from Stage 3 (rank based, so the log transform makes no difference)
        pair = pd.concat([feat.loc[idx].assign(role="edited"),
                          feat[is_real & feat["dicom_id"].astype(str).isin(common)].assign(role="real")])
        auc_tab = sa.single_feature_auroc(pair, cols, "role").query("group == 'edited'").set_index("feature")
        for c in cols:
            d = D[c].to_numpy()
            if np.all(d == 0):
                p = 1.0
            else:
                try:
                    p = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    p = 1.0
            sdd = d.std(ddof=1) if n > 1 else 0.0
            rows.append(dict(group=g, level=lv, target_area=float(sub["target_area"].iloc[0]),
                             area=float(sub["achieved_area"].mean()), n=n, feature=c,
                             family=family_of.get(c, "other"), median_shift=float(np.median(d)),
                             shift_in_real_sd=float(np.median(d) / (sd_real[c] + 1e-12)),
                             paired_dz=float(d.mean() / sdd) if sdd > 0 else 0.0,
                             auroc=float(auc_tab.loc[c, 'auroc']), auroc_dir=float(auc_tab.loc[c, 'auroc_directionless']), p_value=p))
    out = pd.DataFrame(rows)
    out["q_value"] = multipletests(out["p_value"], method=config.FORENSIC_MULTIPLE_TEST_METHOD)[1]   # as Stage 3
    out["significant"] = out["q_value"] < config.FORENSIC_ALPHA
    return out.sort_values(["group", "target_area", "feature"]).reset_index(drop=True)


def level_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """One row per group x level: how many features respond, how large, best single feature."""
    rows = []
    for (g, lv), d in tab.groupby(["group", "level"]):
        best = d.sort_values("auroc_dir", ascending=False).iloc[0]
        rows.append(dict(group=g, level=lv, target_area=d["target_area"].iloc[0], area=d["area"].iloc[0], n=int(d["n"].iloc[0]),
                         n_features=len(d), n_significant=int(d["significant"].sum()),
                         mean_abs_shift_sd=d["shift_in_real_sd"].abs().mean(),
                         best_feature=best["feature"], best_auroc=best["auroc_dir"]))
    return pd.DataFrame(rows).sort_values(["group", "target_area"]).reset_index(drop=True)


def family_curve(tab: pd.DataFrame) -> pd.DataFrame:
    """Mean |shift| (in real-SD units), mean AUROC and number of significant features per family."""
    g = (tab.groupby(["group", "level", "target_area", "family"])
            .agg(mean_abs_shift_sd=("shift_in_real_sd", lambda s: s.abs().mean()),
                 mean_auroc_dir=("auroc_dir", "mean"), n_significant=("significant", "sum"),
                 n_features=("feature", "size")).reset_index())
    return g.sort_values(["group", "family", "target_area"]).reset_index(drop=True)


def trend_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Per group and family (and 'all'): slope of log10(mean |shift|) on log10(target area), and the
    smallest size from which at least one feature stays significant at every larger size.
    slope close to 1  -> footprint grows in proportion to the area (dilution rule)
    slope well below 1 -> grows more slowly than the area
    slope near 0       -> does not depend on size (expected for Diffusion_full)."""
    fam = family_curve(tab)
    rows = []
    for g, dg in tab.groupby("group"):
        sizes = sorted(dg["target_area"].unique())
        sig = dg.groupby("target_area")["significant"].sum().reindex(sizes)
        first = None
        for i, s in enumerate(sizes):
            if (sig.iloc[i:] >= 1).all():
                first = s
                break
        for famname in FAMILIES + ["all"]:
            if famname == "all":
                y = dg.groupby("target_area")["shift_in_real_sd"].apply(lambda s: s.abs().mean()).reindex(sizes)
            else:
                y = fam[(fam.group == g) & (fam.family == famname)].set_index("target_area")["mean_abs_shift_sd"].reindex(sizes)
            ok = (y > 0).to_numpy()
            if ok.sum() >= 3:
                x = np.log10(np.array(sizes)[ok]); yy = np.log10(y.to_numpy()[ok])
                slope, intercept = np.polyfit(x, yy, 1)
                r = float(np.corrcoef(x, yy)[0, 1])
            else:
                slope, r = np.nan, np.nan
            rows.append(dict(group=g, family=famname, loglog_slope=slope, fit_r=r,
                             smallest_size_with_a_significant_feature=first))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. radial spectrum
# ----------------------------------------------------------------------------
def radial_ratio(feat: pd.DataFrame, profiles: pd.DataFrame, eps=1e-12) -> pd.DataFrame:
    """Mean paired log10(P_edited / P_real) per radial bin, group x level. Long table with SEM."""
    feat = feat.reset_index(drop=True)
    is_real = feat["group"] == "Real"
    real_rows = feat[is_real].set_index(feat.loc[is_real, "dicom_id"].astype(str))["image_id"]
    rows = []
    for (g, lv), d in feat[~is_real].groupby(["group", "level"]):
        d = d[d["dicom_id"].astype(str).isin(real_rows.index)]
        pe = np.log10(profiles.loc[d["image_id"]].to_numpy() + eps)
        pr = np.log10(profiles.loc[real_rows.loc[d["dicom_id"].astype(str)].to_numpy()].to_numpy() + eps)
        diff = pe - pr
        m = diff.mean(axis=0)
        sem = diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.zeros_like(m)
        nb = len(m)
        for b in range(nb):
            rows.append(dict(group=g, level=lv, target_area=float(d["target_area"].iloc[0]), n=len(d),
                             freq=(b + 0.5) / nb, log10_ratio=float(m[b]), sem=float(sem[b])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. figures
# ----------------------------------------------------------------------------
def _save(fig, name):
    Path(sc.FIG_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(sc.FIG_DIR) / f"{name}.png", dpi=170, bbox_inches="tight")


def plot_family_curves(tab: pd.DataFrame, show=True):
    """Mean |shift| per feature family vs target area; one panel per group. Log-log axes."""
    import matplotlib.pyplot as plt
    fam = family_curve(tab)
    groups = [g for g in GROUPS if g in set(fam["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.4), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        for f in FAMILIES:
            d = fam[(fam.group == g) & (fam.family == f)].sort_values("target_area")
            ax.plot(100 * d["target_area"], d["mean_abs_shift_sd"], color=_FCOL[f], marker=_FMARK[f],
                    ms=6, lw=2, label=f)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(g, fontsize=10)
        ax.set_xticks(100 * np.array(sorted(fam["target_area"].unique())))
        ax.set_xticklabels([f"{100 * a:g}" for a in sorted(fam["target_area"].unique())], fontsize=8)
        ax.set_xlabel("edit area (% of image)"); ax.grid(alpha=.25, lw=.6)
    axes[0][0].set_ylabel("mean |median shift| / SD(Real)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Footprint of the edit in the forensic features vs edit size (paired with the Real source)", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_frequency_family_curves")
    if show: plt.show()
    return fig


def plot_feature_heatmaps(tab: pd.DataFrame, top_k=12, show=True):
    """Signed shift (units of Real SD) of the top_k features per group; * = BH q < alpha."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f2f2", "#eb6834"])
    groups = [g for g in GROUPS if g in set(tab["group"])]
    vmax = float(np.nanpercentile(tab["shift_in_real_sd"].abs(), 98)) or 1.0
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 0.34 * top_k + 2.2), squeeze=False)
    for ax, g in zip(axes[0], groups):
        d = tab[tab.group == g]
        top = d.groupby("feature")["shift_in_real_sd"].apply(lambda s: s.abs().max()).nlargest(top_k).index.tolist()
        lv = d.sort_values("target_area")[["level", "target_area"]].drop_duplicates()
        M = d.pivot(index="feature", columns="target_area", values="shift_in_real_sd").loc[top]
        S = d.pivot(index="feature", columns="target_area", values="significant").loc[top]
        ax.imshow(M.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if bool(S.iloc[i, j]):
                    ax.text(j, i, "*", ha="center", va="center", fontsize=11)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{100 * a:g}%" for a in M.columns], fontsize=8)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=7); ax.set_title(g, fontsize=10)
    fig.suptitle("Median paired shift / SD(Real) of the most responsive features  (* = q < alpha)", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_frequency_heatmaps")
    if show: plt.show()
    return fig


def plot_radial(rad: pd.DataFrame, show=True):
    """Mean paired log10 power ratio (edited / Real) vs normalised frequency, one line per size."""
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if g in set(rad["group"])]
    sizes = sorted(rad["target_area"].unique())
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(sizes)))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.3), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        for s, col in zip(sizes, shades):
            d = rad[(rad.group == g) & (rad.target_area == s)].sort_values("freq")
            if d.empty:
                continue
            ax.plot(d["freq"], d["log10_ratio"], color=col, lw=2, label=f"{100 * s:g}%")
            ax.fill_between(d["freq"], d["log10_ratio"] - 1.96 * d["sem"], d["log10_ratio"] + 1.96 * d["sem"],
                            color=col, alpha=.12, lw=0)
        ax.axhline(0, color="#888", lw=1); ax.set_xscale("log"); ax.set_title(g, fontsize=10)
        ax.set_xlabel("normalised frequency (1 = Nyquist)"); ax.grid(alpha=.25, lw=.6)
    axes[0][0].set_ylabel("mean log10 (P edited / P real)")
    axes[0][-1].legend(title="edit area", fontsize=8, frameon=False)
    fig.suptitle("Where in the spectrum the edit changes power (paired, band = 95% CI of the mean)", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_frequency_radial")
    if show: plt.show()
    return fig


def plot_auroc(summary: pd.DataFrame, show=True):
    """Best single-feature AUROC (edited vs Real) vs size."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for g in GROUPS:
        d = summary[summary.group == g].sort_values("target_area")
        if d.empty:
            continue
        ax.plot(100 * d["target_area"], d["best_auroc"], color=sc.COLOURS.get(_GCOL[g], "#444"), marker="o", ms=6, lw=2, label=g)
    ax.axhline(0.5, color="#888", lw=1); ax.set_xscale("log")
    ax.set_xticks(100 * np.array(sorted(summary["target_area"].unique())))
    ax.set_xticklabels([f"{100 * a:g}" for a in sorted(summary["target_area"].unique())])
    ax.set_xlabel("edit area (% of image)"); ax.set_ylabel("best single-feature AUROC")
    ax.set_title("Best of the 42 features, edited vs its Real source", fontsize=10)
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); _save(fig, "ladder_frequency_best_auroc")
    if show: plt.show()
    return fig


# ----------------------------------------------------------------------------
# 4b. footprint vs size (definition: stage3_forensics/footprint.py)
# ----------------------------------------------------------------------------
def footprint_report(feat: pd.DataFrame, base: pd.DataFrame | None = None, B=200, seed=0) -> dict:
    """Footprint of every ladder cell (length of the standardised mean shift from the Real TRAIN mean, with a
    bootstrap interval over images), the no-edit null (Real test images, same reference) and the whole-family
    footprints on the test split. `base` = the Stage 3 feature table (default: the main table)."""
    base = ef.load_feature_table() if base is None else base
    ref = fpm.Reference(base)
    edited = feat[feat["group"] != "Real"]
    ladder = fpm.ladder_footprints(edited, ref, ["group"], B=B, seed=seed)
    ladder["area_target"] = ladder["level"].map(edited.drop_duplicates("level").set_index("level")["target_area"])
    null = fpm.real_null(base, ref, split="test", B=B, seed=seed)
    fam = fpm.family_table(base, ref, split="test")
    return {"ladder": ladder.sort_values(["group", "area"]).reset_index(drop=True), "null": null, "families": fam}


def plot_footprint(rep: dict, show=True):
    """Footprint vs edit area; left panel IP techniques, right panel diffusion. Band = 95% bootstrap over images.
    Dashed grey = no-edit null (Real test images) with its band; dotted = whole-family footprints (test split)."""
    import matplotlib.pyplot as plt
    fp, null, fam = rep["ladder"], rep["null"], rep["families"]["footprint"]
    panels = [("IP edits", ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"]),
              ("Diffusion edits", ["Diffusion_paste", "Diffusion_full"])]
    panels = [(t, [g for g in gs if g in set(fp["group"])]) for t, gs in panels]
    panels = [(t, gs) for t, gs in panels if gs]
    sizes = sorted(fp["area"].round(4).unique())
    x0, x1 = 100 * fp["area"].min(), 100 * fp["area"].max()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5), sharey=True, squeeze=False)
    for ax, (title, gs) in zip(axes[0], panels):
        for g in gs:
            d = fp[fp.group == g].sort_values("area")
            c = sc.COLOURS.get(_GCOL[g], "#444")
            ax.fill_between(100 * d["area"], d["lo"], d["hi"], color=c, alpha=0.15, lw=0)
            ax.plot(100 * d["area"], d["footprint"], color=c, marker=getattr(sc, "MARKERS", {}).get(_GCOL[g], "o"),
                    ms=6, lw=1.5, mec="white", label=g)
        ax.axhspan(null["lo"], null["hi"], color="#8a8984", alpha=0.18, lw=0)
        ax.axhline(null["footprint"], color="#52514e", lw=1, ls="--")
        ax.text(x0, null["footprint"], f" no edit (Real test, n={null['n']})", fontsize=8, color="#52514e", va="bottom")
        last = 0.0
        for name, y in fam.sort_values().items():
            if str(name).startswith("IP"):
                continue
            ax.axhline(y, color="#8a8984", lw=0.9, ls=":")
            yy = max(y, last * 1.06); last = yy
            ax.text(x1, yy, f"{name} ", fontsize=8, color="#52514e", va="bottom", ha="right")
        ax.set_xscale("log"); ax.set_xticks([100 * a for a in sizes]); ax.set_xticklabels([f"{100 * a:g}" for a in sizes])
        ax.set_xlabel("edit area (% of image)"); ax.set_title(title, fontsize=10)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0][0].set_ylabel("footprint (distance from Real train, in real SDs)")
    fig.tight_layout(); _save(fig, "ladder_footprint")
    if show: plt.show()
    return fig


# ----------------------------------------------------------------------------
# 4c. further figures: where in frequency, which features, dilution, 2-D spectra
# ----------------------------------------------------------------------------
BANDS = ((0.0, 0.1, "low"), (0.1, 0.3, "mid"), (0.3, 1.01, "high"))      # fractions of Nyquist; band edges are OURS
_BCOL = {"low": "#2a78d6", "mid": "#1baf7a", "high": "#eb6834"}


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f2f2", "#eb6834"])


def plot_freq_heatmap(rad: pd.DataFrame, show=True):
    """Frequency x size heatmap of the mean paired log10 power ratio (edited / Real), one panel per technique.
    Orange = the edit adds power at that frequency, blue = removes power, grey = no change."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    groups = [g for g in GROUPS if g in set(rad["group"])]
    vmax = float(np.nanpercentile(rad["log10_ratio"].abs(), 99)) or 1.0
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups) + 1, 3.8), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        d = rad[rad.group == g]
        sizes = sorted(d["target_area"].unique())
        M = d.pivot_table(index="freq", columns="target_area", values="log10_ratio").sort_index()
        M = M.iloc[1:]                                           # drop the lowest bin (DC neighbourhood)
        nb = len(M) + 1
        edges = np.arange(1, nb + 1) / nb                        # bin edges 1/nb ... 1 (log axis cannot show 0)
        im = ax.pcolormesh(np.arange(len(sizes) + 1), edges, M[sizes].to_numpy(), cmap=_diverging(),
                           norm=TwoSlopeNorm(0, -vmax, vmax), shading="flat")
        ax.set_yscale("log"); ax.set_xticks(np.arange(len(sizes)) + 0.5)
        ax.set_xticklabels([f"{100 * a:g}%" for a in sizes], fontsize=8); ax.set_title(g, fontsize=10)
        ax.set_xlabel("edit area")
    axes[0][0].set_ylabel("normalised frequency (1 = Nyquist)")
    fig.colorbar(im, ax=axes[0].tolist(), label="mean log10 (P edited / P real)", shrink=0.9)
    fig.suptitle("Where in the spectrum the edit changes power, by edit size (paired with the Real source)", fontsize=10)
    _save(fig, "ladder_freq_heatmap")
    if show: plt.show()
    return fig


def plot_family_composition(tab: pd.DataFrame, show=True):
    """Share of the squared paired shift (|median shift| / SD(Real), per feature) carried by each feature family.
    Read the larger sizes: at the smallest sizes the shift is close to the noise and the shares are unstable."""
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if g in set(tab["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.4), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        d = tab[tab.group == g].assign(sq=lambda x: x["shift_in_real_sd"] ** 2)
        P = d.pivot_table(index="target_area", columns="family", values="sq", aggfunc="sum").reindex(columns=FAMILIES).fillna(0)
        P = P.div(P.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        bottom = np.zeros(len(P))
        for f in FAMILIES:
            ax.bar(range(len(P)), P[f], bottom=bottom, color=_FCOL[f], width=0.8, edgecolor="white", linewidth=1.2, label=f)
            bottom += P[f].to_numpy()
        ax.set_xticks(range(len(P))); ax.set_xticklabels([f"{100 * a:g}%" for a in P.index], fontsize=8)
        ax.set_title(g, fontsize=10); ax.set_xlabel("edit area")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("share of squared shift"); axes[0][-1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Which feature families carry the shift, by edit size", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_freq_family_composition")
    if show: plt.show()
    return fig


def plot_footprint_loglog(rep: dict, show=True):
    """Null-corrected footprint vs area on log-log axes with a slope-1 (proportional-to-area) reference line.
    corrected = sqrt(max(0, footprint^2 - null^2)), null = footprint of the Real test images. Hollow marker = the
    lower bound of the interval is not above the null. The reference line passes through the largest-area point."""
    import matplotlib.pyplot as plt
    fp, null = rep["ladder"], rep["null"]["footprint"]
    corr = lambda v: np.sqrt(np.maximum(0.0, np.asarray(v, float) ** 2 - null ** 2))
    panels = [("IP edits", ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"]),
              ("Diffusion edits", ["Diffusion_paste", "Diffusion_full"])]
    panels = [(t, [g for g in gs if g in set(fp["group"])]) for t, gs in panels]
    panels = [(t, gs) for t, gs in panels if gs]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.3), sharey=True, squeeze=False)
    for ax, (title, gs) in zip(axes[0], panels):
        for g in gs:
            d = fp[fp.group == g].sort_values("area")
            c, lo, hi = corr(d["footprint"]), corr(d["lo"]), corr(d["hi"])
            col = sc.COLOURS.get(_GCOL[g], "#444")
            ok = c > 0
            ax.plot(100 * d["area"][ok], c[ok], color=col, lw=1.5, label=g)
            solid = ok & (lo > 0)
            ax.plot(100 * d["area"][solid], c[solid], color=col, lw=0, marker="o", ms=6, mec="white")
            hollow = ok & ~(lo > 0)
            ax.plot(100 * d["area"][hollow], c[hollow], color=col, lw=0, marker="o", ms=6, mfc="white")
            ax.fill_between(100 * d["area"][ok], np.maximum(lo[ok], 1e-3), np.maximum(hi[ok], 1e-3), color=col, alpha=0.12, lw=0)
            if ok.any():
                i = np.where(ok)[0][-1]
                a = 100 * d["area"].to_numpy()
                ax.plot(a, c[i] * a / a[i], color=col, lw=0.9, ls=":", alpha=0.9)
        sizes = sorted(fp["area"].round(4).unique())
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=10)
        ax.set_xticks([100 * z for z in sizes]); ax.set_xticklabels([f"{100 * z:g}" for z in sizes]); ax.minorticks_off()
        ax.set_xlabel("edit area (% of image)"); ax.grid(alpha=.25, lw=.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("null-corrected footprint (real SDs)")
    fig.suptitle("Footprint vs area: dotted = growth proportional to area (slope 1); hollow = not above the no-edit null", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_footprint_loglog")
    if show: plt.show()
    return fig


def band_ratio(feat: pd.DataFrame, profiles: pd.DataFrame, bands=BANDS, eps=1e-12) -> pd.DataFrame:
    """Per image: mean log10 power ratio (edited / Real source) inside each frequency band; then mean and SEM over images."""
    feat = feat.reset_index(drop=True)
    nb = profiles.shape[1]
    freq = (np.arange(nb) + 0.5) / nb
    is_real = feat["group"] == "Real"
    real_id = feat[is_real].set_index(feat.loc[is_real, "dicom_id"].astype(str))["image_id"]
    rows = []
    for (g, lv), d in feat[~is_real].groupby(["group", "level"]):
        d = d[d["dicom_id"].astype(str).isin(real_id.index)]
        pe = np.log10(profiles.loc[d["image_id"]].to_numpy() + eps)
        pr = np.log10(profiles.loc[real_id.loc[d["dicom_id"].astype(str)].to_numpy()].to_numpy() + eps)
        diff = pe - pr
        for lo, hi, name in bands:
            m = (freq >= lo) & (freq < hi)
            v = diff[:, m].mean(axis=1)
            rows.append(dict(group=g, level=lv, area=float(d["target_area"].iloc[0]), band=name, mean=float(v.mean()),
                             sem=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, n=len(v)))
    return pd.DataFrame(rows).sort_values(["group", "band", "area"]).reset_index(drop=True)


def plot_bands(bands_tbl: pd.DataFrame, show=True):
    """Low / mid / high band log power ratio vs edit area (mean +- 1.96 SEM over images)."""
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if g in set(bands_tbl["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.3), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        for name in ("low", "mid", "high"):
            d = bands_tbl[(bands_tbl.group == g) & (bands_tbl.band == name)].sort_values("area")
            ax.plot(100 * d["area"], d["mean"], color=_BCOL[name], marker="o", ms=5, lw=1.8, label=f"{name} band")
            ax.fill_between(100 * d["area"], d["mean"] - 1.96 * d["sem"], d["mean"] + 1.96 * d["sem"], color=_BCOL[name], alpha=0.15, lw=0)
        ax.axhline(0, color="#888", lw=1); ax.set_xscale("log"); ax.set_title(g, fontsize=10)
        ax.set_xlabel("edit area (% of image)"); ax.grid(alpha=.25, lw=.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("mean log10 (P edited / P real)"); axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Power change in three frequency bands (low <0.1, mid 0.1-0.3, high >0.3 of Nyquist)", fontsize=10)
    fig.tight_layout(); _save(fig, "ladder_freq_bands")
    if show: plt.show()
    return fig


def plot_spectrum_difference(target_areas=(0.01, 0.05, 0.10), n_avg=30, seed=0, show=True):
    """2-D map of mean log10 (P edited / P real) over up to n_avg images (same images in every panel), per
    technique (rows) and edit size (columns). Directional structure that the radial average hides shows here.
    Uses Stage 3's power_spectrum (same Hann window and mean removal as the features)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from forensics.feature_extractors import power_spectrum
    from utils.common import load_gray_float01
    man = read_ladder_manifests().assign(dicom_id=lambda m: m["dicom_id"].astype(str))
    groups = [g for g in GROUPS if g in set(man["group"])]
    cells = {}
    for g in groups:
        for t in target_areas:
            d = man[(man.group == g) & ((man.target_area - t).abs() < 1e-4)]
            if len(d):
                cells[(g, t)] = d.set_index("dicom_id")
    common = set.intersection(*[set(c.index) for c in cells.values()])
    ids = sorted(common)
    ids = list(np.random.default_rng(seed).permutation(ids)[:n_avg]) if len(ids) > n_avg else ids
    maps = {}
    for (g, t), c in cells.items():
        acc = 0
        for i in ids:
            pe = power_spectrum(load_gray_float01(c.loc[i, "prepared_path"]))
            pr = power_spectrum(load_gray_float01(c.loc[i, "real_prepared_path"]))
            acc = acc + np.log10(pe + 1e-12) - np.log10(pr + 1e-12)
        maps[(g, t)] = acc / len(ids)
    vmax = float(np.percentile(np.abs(np.stack(list(maps.values()))), 99)) or 1.0
    fig, axes = plt.subplots(len(groups), len(target_areas), figsize=(2.6 * len(target_areas) + 1, 2.6 * len(groups)), squeeze=False)
    for i, g in enumerate(groups):
        for j, t in enumerate(target_areas):
            ax = axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
            if (g, t) in maps:
                im = ax.imshow(maps[(g, t)], cmap=_diverging(), norm=TwoSlopeNorm(0, -vmax, vmax))
            if i == 0: ax.set_title(f"{100 * t:g}%", fontsize=10)
            if j == 0: ax.set_ylabel(g, fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean log10 (P edited / P real)", shrink=0.8)
    fig.suptitle(f"2-D spectrum change (centre = low frequency), mean over {len(ids)} images", fontsize=10)
    _save(fig, "ladder_freq_spectrum_2d")
    if show: plt.show()
    return fig


def extra_figures(out: dict, show=True) -> dict:
    """The five additional views, from the dict returned by run_all (nothing is re-extracted)."""
    prof = ef.load_radial_profiles(PROF_NPZ)
    bands = band_ratio(out["features"], prof)
    bands.to_csv(OUT_DIR / "ladder_freq_bands.csv", index=False)
    figs = {"freq_heatmap": plot_freq_heatmap(out["radial"], show),
            "family_composition": plot_family_composition(out["table"], show),
            "footprint_loglog": plot_footprint_loglog(out["footprint_report"], show),
            "bands": plot_bands(bands, show),
            "spectrum_2d": plot_spectrum_difference(show=show)}
    print(f"Figures -> {sc.FIG_DIR}")
    return figs


# ----------------------------------------------------------------------------
# 5. everything
# ----------------------------------------------------------------------------
def run_all(n_jobs=None, residual=True, overwrite=False, show=True) -> dict:
    feat, res = extract(n_jobs, residual, overwrite)
    tab = paired_table(feat)
    out = {"features": feat, "table": tab, "summary": level_summary(tab),
           "families": family_curve(tab), "trend": trend_table(tab)}
    if res is not None:
        sres = list(config.RESIDUAL_SPECTRAL_FEATURES)
        fam_of = {c: "specres" for c in sres}
        out["table_residual"] = paired_table(res, cols=sres, family_of=fam_of)
        out["summary_residual"] = level_summary(out["table_residual"])
    prof = ef.load_radial_profiles(PROF_NPZ)
    out["radial"] = radial_ratio(feat, prof)
    out["footprint_report"] = footprint_report(feat)
    out["footprint"] = out["footprint_report"]["ladder"]
    for k in ("table", "summary", "families", "trend", "radial", "footprint", "table_residual", "summary_residual"):
        if out.get(k) is not None:
            out[k].to_csv(OUT_DIR / f"ladder_freq_{k}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPER GROUP AND LEVEL (n = paired images):")
    print(out["summary"].round(4).to_string(index=False))
    print("\nSIZE TREND (loglog_slope: ~1 = proportional to area, ~0 = independent of size):")
    print(out["trend"][out["trend"].family == "all"].round(3).to_string(index=False))
    if "summary_residual" in out:
        print("\nRESIDUAL-SPECTRAL SET (6 specres_* features):")
        print(out["summary_residual"].round(4).to_string(index=False))
    out["figures"] = {"family": plot_family_curves(tab, show), "heat": plot_feature_heatmaps(tab, show=show),
                      "radial": plot_radial(out["radial"], show),
                      "footprint": plot_footprint(out["footprint_report"], show), "auroc": plot_auroc(out["summary"], show)}
    print(f"\nTables -> {OUT_DIR}\nFigures -> {sc.FIG_DIR}")
    return out