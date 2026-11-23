"""
statistical screening and characterisation of the 42 features (Q1), on config.SCREENING_SPLIT
("train") only.
Descriptive statistics; Kruskal-Wallis with epsilon-squared; Benjamini-Hochberg FDR over the 42 features;
pairwise Mann-Whitney with Cliff's delta; one-vs-rest single-feature AUROC; PCA; feature-family summary;
real-vs-group table; redundancy pruning (Spearman |rho| > 0.95).
Notes: kruskal_table() is also called by Stage 5 on training data only. With large groups, read the
effect sizes rather than the p-values. Images are not independent (a synthetic image shares its source
with a real one; a patient can contribute several images), so tests describe the groups and do not give
formal inference for new patients. The 42 features are not independent tests. Diffusion_EditOnly and
Diffusion_Reconstruction are diagnostic groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import config
from utils.common import ensure_parent, save_figure

MIN_IMAGES_PER_GROUP = 30        # below this a rank test or AUROC is not meaningful


def transform_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Raw feature columns -> model-ready matrix (log10 for positive heavy-tailed ones)."""
    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    for j, c in enumerate(cols):
        if c in config.FEATURES_LOG10:
            X[:, j] = np.log10(np.maximum(X[:, j], 0) + config.FEATURE_LOG10_EPS)
    return X


def kruskal_table(df, features, group_col, alpha=None) -> pd.DataFrame:
    """Kruskal-Wallis H, p, epsilon^2 and BH-FDR q for each feature (sorted by epsilon^2)."""
    alpha = config.FORENSIC_ALPHA if alpha is None else alpha
    groups = [g for _, g in df.groupby(group_col)]
    n = len(df)
    rows = []
    for f in features:
        samples = [g[f].to_numpy() for g in groups]
        if np.ptp(np.concatenate(samples)) == 0:                      # constant feature
            rows.append((f, np.nan, 1.0, 0.0))
            continue
        H, p = stats.kruskal(*samples)
        rows.append((f, H, p, H / ((n ** 2 - 1) / (n + 1))))
    out = pd.DataFrame(rows, columns=["feature", "H", "p_value", "epsilon_sq"])
    out["q_value"] = multipletests(out["p_value"].fillna(1.0), method=config.FORENSIC_MULTIPLE_TEST_METHOD)[1]
    out["significant"] = out["q_value"] < alpha
    out["family"] = out["feature"].map(config.FEATURE_FAMILY_OF)
    return out.sort_values("epsilon_sq", ascending=False).reset_index(drop=True)


def descriptive_statistics(df, features, group_col) -> pd.DataFrame:
    """Long table: group, feature, mean, std, median, q25, q75."""
    g = df.groupby(group_col)[features]
    parts = {"mean": g.mean(), "std": g.std(), "median": g.median(),
             "q25": g.quantile(0.25), "q75": g.quantile(0.75)}
    rows = []
    for grp in parts["mean"].index:
        for f in features:
            rows.append({group_col: grp, "feature": f, **{k: v.loc[grp, f] for k, v in parts.items()}})
    return pd.DataFrame(rows)


def pairwise_mann_whitney(df, features, group_col) -> pd.DataFrame:
    """All group pairs x all features; FDR across ALL tests; Cliff's delta from U."""
    names = sorted(df[group_col].unique())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = df[df[group_col] == a], df[df[group_col] == b]
            for f in features:
                x, y = A[f].to_numpy(), B[f].to_numpy()
                if np.ptp(np.concatenate([x, y])) == 0:
                    rows.append((a, b, f, np.nan, 1.0, 0.0))
                    continue
                U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                rows.append((a, b, f, U, p, 2.0 * U / (len(x) * len(y)) - 1.0))
    out = pd.DataFrame(rows, columns=["group_a", "group_b", "feature", "U", "p_value", "cliffs_delta"])
    out["q_value"] = multipletests(out["p_value"], method=config.FORENSIC_MULTIPLE_TEST_METHOD)[1]
    return out


def single_feature_auroc(df, features, group_col) -> pd.DataFrame:
    """One-vs-rest AUROC of each raw feature for each group (0.5 = no signal)."""
    rows = []
    for g in sorted(df[group_col].unique()):
        y = (df[group_col] == g).astype(int).to_numpy()
        for f in features:
            a = roc_auc_score(y, df[f].to_numpy())
            rows.append((g, f, a, max(a, 1 - a)))
    return pd.DataFrame(rows, columns=["group", "feature", "auroc", "auroc_directionless"])


def group_vs_reference(pairwise: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    """NEW. Keep only the rows 'reference group vs each other group' from `pairwise_mann_whitney`,
    re-correct p-values with BH-FDR over THIS smaller set, and give the direction in plain words.
    Cliff's delta here is oriented so that  + = feature is HIGHER in the group than in the reference."""
    reference = reference or config.REFERENCE_GROUP
    if reference not in set(pairwise["group_a"]) | set(pairwise["group_b"]):
        raise ValueError(f"reference group '{reference}' is not in the pairwise table")
    a_ref = pairwise["group_a"] == reference
    b_ref = pairwise["group_b"] == reference
    d = pairwise[a_ref | b_ref].copy()
    d["group"] = np.where(d["group_a"] == reference, d["group_b"], d["group_a"])
    # pairwise delta is (a vs b): positive = a higher.  Flip so positive = group higher than reference.
    d["cliffs_delta_group_vs_ref"] = np.where(d["group_a"] == reference, -d["cliffs_delta"], d["cliffs_delta"])
    d["reference"] = reference
    d["q_value_vs_ref"] = multipletests(d["p_value"].fillna(1.0), method=config.FORENSIC_MULTIPLE_TEST_METHOD)[1]
    ad = d["cliffs_delta_group_vs_ref"].abs()
    d["magnitude"] = np.select([ad < .147, ad < .33, ad < .474], ["negligible", "small", "medium"], "large")
    d["direction"] = np.where(d["cliffs_delta_group_vs_ref"] > 0, "higher in group", "lower in group")
    d["family"] = d["feature"].map(config.FEATURE_FAMILY_OF)
    cols = ["group", "feature", "family", "cliffs_delta_group_vs_ref", "magnitude", "direction",
            "p_value", "q_value_vs_ref", "reference"]
    return d[cols].sort_values(["group", "cliffs_delta_group_vs_ref"], key=lambda c: c.abs() if c.name != "group" else c,
                               ascending=[True, False]).reset_index(drop=True)


def redundancy_pruning(df, features, kruskal: pd.DataFrame | None = None, threshold: float | None = None):
    """NEW. Remove near-duplicate features.
    Spearman |rho| between all feature pairs on `df`; features are visited from strongest to weakest
    (epsilon^2 from `kruskal`, else the given order); a feature is DROPPED if it correlates above
    `threshold` with a feature that was already KEPT.   Returns (kept_features, table of dropped features)."""
    threshold = config.REDUNDANCY_SPEARMAN_THRESHOLD if threshold is None else threshold
    rho = df[features].corr(method="spearman").abs()
    order = list(features)
    if kruskal is not None:
        order = [f for f in kruskal.sort_values("epsilon_sq", ascending=False)["feature"] if f in features]
    kept, dropped = [], []
    for f in order:
        dup = next((k for k in kept if rho.loc[f, k] > threshold), None)     # NaN compares False -> kept
        if dup is None:
            kept.append(f)
        else:
            dropped.append({"dropped_feature": f, "duplicate_of": dup, "spearman_abs": float(rho.loc[f, dup]),
                            "family": config.FEATURE_FAMILY_OF.get(f)})
    table = pd.DataFrame(dropped, columns=["dropped_feature", "duplicate_of", "spearman_abs", "family"])
    return kept, table


def feature_family_summary(kw: pd.DataFrame) -> pd.DataFrame:
    return (kw.groupby("family")
              .agg(n_features=("feature", "size"), n_significant=("significant", "sum"),
                   mean_epsilon_sq=("epsilon_sq", "mean"), max_epsilon_sq=("epsilon_sq", "max"),
                   top_feature=("feature", "first"))
              .sort_values("mean_epsilon_sq", ascending=False))


def run_pca(df, features, group_col):
    """PCA on all (log-transformed, standardised) features -> (scores_df, explained_ratio)."""
    X = StandardScaler().fit_transform(transform_features(df, features))
    pca = PCA(n_components=config.FORENSIC_PCA_COMPONENTS, random_state=config.FORENSIC_PCA_RANDOM_STATE)
    Z = pca.fit_transform(X)
    scores = pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(Z.shape[1])], index=df.index)
    scores[group_col] = df[group_col].values
    return scores, pca.explained_variance_ratio_


# ------------------------------- plots --------------------------------------
def plot_pca(scores, ratio, group_col):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=config.FORENSIC_PLOT_FIGSIZE)
    for g, d in scores.groupby(group_col):
        ax.scatter(d["PC1"], d["PC2"], s=14, alpha=.65, label=g, color=config.GROUP_COLOURS.get(g))
    ax.set_xlabel(f"PC1 ({ratio[0]:.2%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.2%})")
    ax.set_title(f"PCA of the {len(config.FEATURE_COLUMNS)} forensic features (train split)"); ax.legend(fontsize=8, markerscale=1.5)
    save_figure(fig, "stage3_pca")
    return fig


def plot_top_features(kw, k=15):
    import matplotlib.pyplot as plt
    top = kw.head(k).iloc[::-1]
    palette = {"spectral": "#1f77b4", "wavelet": "#2ca02c", "lbp": "#ff7f0e", "srm": "#d62728", "noise": "#9467bd"}
    fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1))
    ax.barh(top["feature"], top["epsilon_sq"], color=[palette[f] for f in top["family"]])
    ax.set_xlabel("epsilon-squared (Kruskal-Wallis effect size)"); ax.set_title(f"Top {k} discriminative features")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()], palette.keys(),
              fontsize=8, loc="lower right")
    save_figure(fig, "stage3_top_features")
    return fig


def plot_mean_spectra(df, profiles, group_col):
    """Radial power spectrum per group (log-log): geometric mean over the images of each group."""
    import matplotlib.pyplot as plt
    n_bins = profiles.shape[1]
    freq = (np.arange(n_bins) + 0.5) / n_bins
    fig, ax = plt.subplots(figsize=config.FORENSIC_PLOT_FIGSIZE)
    for g, d in df.groupby(group_col):
        prof = profiles.loc[d["image_id"]].to_numpy()
        ax.plot(freq, 10 ** np.log10(prof + 1e-12).mean(axis=0), label=g, color=config.GROUP_COLOURS.get(g))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("normalised spatial frequency (1 = Nyquist)"); ax.set_ylabel("geometric-mean power")
    ax.set_title("Radial power spectrum by group (geometric mean over images)"); ax.legend(fontsize=8)
    save_figure(fig, "stage3_mean_radial_spectra")
    return fig


def plot_group_mean_heatmap(df, features, group_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    m = df.groupby(group_col)[features].mean()
    z = (m - m.mean()) / (m.std() + 1e-12)
    fig, ax = plt.subplots(figsize=(0.45 * len(features) + 2, 3.5))
    sns.heatmap(z, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "z-score across groups"})
    ax.set_title("Group means of the top features")
    save_figure(fig, "stage3_group_mean_heatmap")
    return fig


def select_screening_data(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for every table: split == config.SCREENING_SPLIT, groups in config.SCREENING_GROUPS.
    Fails loudly when a group is missing or too small instead of silently analysing fewer groups."""
    group_col = config.SCREENING_GROUP_COLUMN
    screen = feature_df if config.SCREENING_SPLIT == "all" else feature_df[feature_df["split"] == config.SCREENING_SPLIT]
    missing = [g for g in config.SCREENING_GROUPS if g not in set(screen[group_col])]
    if missing:
        raise ValueError(f"groups missing from the screening data: {missing}")
    screen = screen[screen[group_col].isin(config.SCREENING_GROUPS)].copy()
    counts = screen[group_col].value_counts()
    small = counts[counts < MIN_IMAGES_PER_GROUP]
    if len(small):
        raise ValueError(f"too few images per group for the tests: {small.to_dict()}")
    if config.REFERENCE_GROUP not in counts.index:
        raise ValueError(f"reference group '{config.REFERENCE_GROUP}' is not in SCREENING_GROUPS")
    bad = ~np.isfinite(screen[list(config.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows contain NaN / inf features")
    return screen


def run_stage3_analysis(feature_df, profiles=None, make_plots=True) -> dict:
    """Run every analysis, save CSVs to the paths in config, return all tables."""
    group_col = config.SCREENING_GROUP_COLUMN
    feats = list(config.FEATURE_COLUMNS)
    screen = select_screening_data(feature_df)
    print(f"Screening on split='{config.SCREENING_SPLIT}': {len(screen)} images, "
          f"{screen[group_col].nunique()} groups")
    print(screen[group_col].value_counts().to_string())

    out = {"descriptive": descriptive_statistics(screen, feats, group_col),
           "kruskal": kruskal_table(screen, feats, group_col)}
    out["family_summary"] = feature_family_summary(out["kruskal"])
    out["single_auroc"] = single_feature_auroc(screen, feats, group_col)
    out["group_means"] = screen.groupby(group_col)[feats].mean()
    if config.PAIRWISE_TESTS_ENABLED:
        out["pairwise"] = pairwise_mann_whitney(screen, feats, group_col)
        out["vs_reference"] = group_vs_reference(out["pairwise"])
    out["kept_features"], out["redundancy"] = redundancy_pruning(screen, feats, out["kruskal"])
    print(f"Redundancy pruning (Spearman |rho| > {config.REDUNDANCY_SPEARMAN_THRESHOLD}): "
          f"{len(out['kept_features'])} of {len(feats)} features kept, {len(out['redundancy'])} dropped")
    out["pca_scores"], out["pca_ratio"] = run_pca(screen, feats, group_col)

    paths = {"descriptive": config.FORENSIC_DESCRIPTIVE_CSV, "kruskal": config.FORENSIC_KRUSKAL_CSV,
             "family_summary": config.FORENSIC_FAMILY_SUMMARY_CSV, "single_auroc": config.FORENSIC_SINGLE_FEATURE_AUROC_CSV,
             "group_means": config.FORENSIC_GROUP_MEANS_CSV, "pairwise": config.FORENSIC_PAIRWISE_CSV,
             "vs_reference": config.FORENSIC_ANALYSIS_DIR / "group_vs_reference.csv",
             "redundancy": config.FORENSIC_ANALYSIS_DIR / "redundancy_dropped_features.csv"}
    for k, p in paths.items():
        if k in out:
            out[k].to_csv(ensure_parent(p), index=k in ("family_summary", "group_means"))
    pd.Series(out["kept_features"], name="feature").to_csv(
        ensure_parent(config.FORENSIC_ANALYSIS_DIR / "kept_features.csv"), index=False)
    pcs = out["pca_scores"].copy()
    pcs["image_id"] = screen["image_id"].values
    pcs.to_csv(ensure_parent(config.FORENSIC_PCA_DIR / "pca_scores.csv"), index=False)

    if make_plots:
        top = out["kruskal"].head(config.SCREENING_TOP_K)["feature"].tolist()
        out["figures"] = {"pca": plot_pca(out["pca_scores"], out["pca_ratio"], group_col),
                          "top": plot_top_features(out["kruskal"]),
                          "heatmap": plot_group_mean_heatmap(screen, top, group_col)}
        if profiles is not None:
            out["figures"]["spectra"] = plot_mean_spectra(screen, profiles, group_col)
    print(f"Saved Stage-3 tables -> {config.FORENSIC_ANALYSIS_DIR}")
    return out
