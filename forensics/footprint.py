"""
footprint and shift direction of a family or edit level in the 42-feature space.
shift vector = (mean feature vector of the set - mean of Real TRAIN) / SD of Real TRAIN.
footprint = length of the shift vector; direction = its orientation (compared by cosine similarity).
null = footprint of Real images of the same split against the Real TRAIN reference (about sqrt(42 / n)
for n images even with no edit). A footprint not clearly above the null (`real_null`, bootstrap band) is
indistinguishable from no edit. Compare cells with similar n. Tables use the TEST split unless `split` says
otherwise. The offset-removed cosine subtracts the null shift vector from both vectors.
"""
import numpy as np
import pandas as pd

import config
from forensics.statistical_analysis import transform_features


class Reference:
    """Mean and std of Real TRAIN features (log-transformed like the models) - the yardstick for every shift."""

    def __init__(self, T, cols=None):
        self.cols = list(cols or config.FEATURE_COLUMNS)
        tr = T[(T.split == "train") & (T.frequency_group == config.REAL_GROUP)]
        X = np.asarray(transform_features(tr, self.cols), dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def shift(self, df):
        X = np.asarray(transform_features(df, self.cols), dtype=float)
        return (X.mean(0) - self.mu) / self.sd

    def footprint(self, df):
        return float(np.linalg.norm(self.shift(df)))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def real_rows(T, split="test"):
    """The Real images of one split (one row per dicom_id): the unedited sources and the no-edit null set."""
    r = T[(T.split == split) & (T.frequency_group == config.REAL_GROUP)]
    return r.drop_duplicates("dicom_id") if "dicom_id" in r else r


def _interval(ref, df, B, rng):
    bs = [ref.footprint(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def real_null(T, ref, split="test", B=200, seed=0):
    """Footprint of the Real images of `split` against the Real train reference, with a bootstrap interval."""
    r = real_rows(T, split)
    lo, hi = _interval(ref, r, B, np.random.default_rng(seed))
    return dict(footprint=ref.footprint(r), lo=lo, hi=hi, n=len(r), split=split)


def family_table(T, ref=None, families=None, split="test"):
    """Footprint of every family on `split`, the null (Real images of that split), and the cosine matrices."""
    ref = ref or Reference(T)
    fam = families or config.FAMILY_SETS["diffusion_split"]
    part = T[T.split == split]
    S = {f: ref.shift(part[part.frequency_group.isin(g)]) for f, g in fam.items()}
    null = ref.shift(real_rows(T, split))
    foot = pd.Series({f: float(np.linalg.norm(v)) for f, v in S.items()}, name="footprint")
    cos = pd.DataFrame({b: {a: cosine(S[a], S[b]) for a in S} for b in S})
    cos0 = pd.DataFrame({b: {a: cosine(S[a] - null, S[b] - null) for a in S} for b in S})
    return dict(footprint=foot, null_footprint=float(np.linalg.norm(null)), cosine=cos, cosine_offset_removed=cos0,
                shifts=S, null_shift=null, split=split)


def ladder_footprints(L, ref, series_cols, B=200, seed=0):
    """Footprint of every ladder cell with a bootstrap interval over images. L needs `level` and `area`."""
    rng, rows = np.random.default_rng(seed), []
    for k, g in L.groupby(series_cols + ["level"]):
        k = k if isinstance(k, tuple) else (k,)
        lo, hi = _interval(ref, g, B, rng)
        rows.append(dict(zip(series_cols + ["level"], k), area=float(g.area.mean()), footprint=ref.footprint(g),
                         lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def ladder_directions(L, T, ref, series_cols, targets=("CycleGAN", "Diffusion", "Diffusion_Reconstruction", "Diffusion_EditOnly"),
                      families=None, remove_offset=True, split="test"):
    """Cosine between each ladder cell's shift vector and each target family's shift vector (same split).

    remove_offset=True subtracts the Real (no-edit) shift of that split from both sides first. At the smallest
    levels the offset-free shift is nearly zero, so those cosines are noisy - read the larger levels."""
    fam = families or config.FAMILY_SETS["diffusion_split"]
    part = T[T.split == split]
    null = ref.shift(real_rows(T, split)) if remove_offset else 0.0
    tg = {t: ref.shift(part[part.frequency_group.isin(fam[t])]) - null for t in targets}
    rows = []
    for k, g in L.groupby(series_cols + ["level"]):
        k = k if isinstance(k, tuple) else (k,)
        v = ref.shift(g) - null
        rows.append(dict(zip(series_cols + ["level"], k), area=float(g.area.mean()), **{t: cosine(v, w) for t, w in tg.items()}))
    return pd.DataFrame(rows).sort_values(series_cols + ["area"])
