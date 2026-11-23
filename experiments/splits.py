from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

import config

MIN_IMAGES_PER_CELL = 10          
_REQUIRED_COLUMNS = ("image_id", "frequency_group", "label", "split", "subject_id", "prepared_path")
_ROLES = ("train_df", "val_df", "test_in_df", "test_cross_df")


@dataclass
class Experiment:
    name: str
    train_family: str
    test_family: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_in_df: pd.DataFrame
    test_cross_df: pd.DataFrame

    @property
    def slug(self) -> str:
        return f"{self.train_family}__to__{self.test_family}"

    def design_hash(self) -> str:
        """Short fingerprint of exactly which images sit in which role (used to make resuming safe)."""
        h = hashlib.sha1()
        for role in _ROLES:
            h.update(role.encode())
            h.update("|".join(sorted(getattr(self, role)["image_id"].astype(str))).encode())
        return h.hexdigest()[:12]

    def summary(self) -> dict:
        def n(d):
            r, s = int((d.label == 0).sum()), int((d.label == 1).sum())
            return f"{r} real + {s} synth"
        return {"experiment": self.name, "train": n(self.train_df), "val": n(self.val_df),
                "test_in": n(self.test_in_df), "test_cross": n(self.test_cross_df),
                "design": self.design_hash()}


# ------------------------------------------------------------------ helpers
def family_set(name=None) -> dict[str, list[str]]:
    return config.FAMILY_SETS[name or config.ACTIVE_FAMILY_SET]


def _subset(df: pd.DataFrame, groups, split: str) -> pd.DataFrame:
    return df[df["frequency_group"].isin(groups) & (df["split"] == split)]


def validate_master(master: pd.DataFrame, fams: dict) -> None:
    """Stop early, with a clear message, when the modelling table cannot support the design."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise KeyError(f"modelling table lacks columns {missing}")
    if master["image_id"].duplicated().any():
        raise ValueError("duplicated image_id in the modelling table")
    if master["subject_id"].isna().any():
        raise ValueError(f"{int(master['subject_id'].isna().sum())} images have no subject_id "
                         "(needed for the patient-level checks)")
    bad_split = set(master["split"].dropna()) - set(config.SPLIT_NAMES)
    if master["split"].isna().any() or bad_split:
        raise ValueError(f"unknown / missing split values: {bad_split or 'NaN'}")
    real = master["frequency_group"] == config.REAL_GROUP
    if (master.loc[real, "label"] != 0).any() or (master.loc[~real, "label"] != 1).any():
        raise ValueError("labels are inconsistent: real must be 0 and every synthetic group 1")
    wanted = [config.REAL_GROUP] + [g for groups in fams.values() for g in groups]
    for g in wanted:
        for sp in config.SPLIT_NAMES:
            n = int(((master["frequency_group"] == g) & (master["split"] == sp)).sum())
            if n < MIN_IMAGES_PER_CELL:
                raise ValueError(f"group '{g}' has only {n} images in the '{sp}' split")


def _balanced_family_rows(master: pd.DataFrame, groups, split: str) -> pd.DataFrame:
    """All rows of `groups` in `split`, or - when config.BALANCE_SYNTH_TO_REAL - a seeded
    subsample with as many synthetic images as there are real ones, equal per group.
    The draw depends only on (seed, split, group), so every experiment sees the same images."""
    d = _subset(master, groups, split)
    if not getattr(config, "BALANCE_SYNTH_TO_REAL", False):
        return d
    target = int(((master["frequency_group"] == config.REAL_GROUP) & (master["split"] == split)).sum())
    if len(d) <= target:
        return d
    parts = {g: d[d["frequency_group"] == g].sort_values("image_id") for g in groups}
    left, quota = target, {}
    for g in sorted(parts, key=lambda k: len(parts[k])):          # water-filling: small groups first
        remaining_groups = len(parts) - len(quota)
        quota[g] = min(len(parts[g]), left // remaining_groups)
        left -= quota[g]
    rng = np.random.default_rng([config.RANDOM_SEED, config.SPLIT_NAMES.index(split)])
    picked = [parts[g].iloc[np.sort(rng.choice(len(parts[g]), quota[g], replace=False))] for g in groups]
    return pd.concat(picked)


# ------------------------------------------------------------------ experiments
def build_experiments(master: pd.DataFrame, family_set_name=None) -> list[Experiment]:
    """All directed cross-generator experiments of the chosen family set."""
    fams = family_set(family_set_name)
    validate_master(master, fams)
    real, exps = [config.REAL_GROUP], []
    for a, b in permutations(fams, 2):
        ga, gb = fams[a], fams[b]
        e = Experiment(
            name=f"{a}->{b}", train_family=a, test_family=b,
            train_df=pd.concat([_subset(master, real, "train"), _balanced_family_rows(master, ga, "train")]),
            val_df=pd.concat([_subset(master, real, "val"), _balanced_family_rows(master, ga, "val")]),
            test_in_df=pd.concat([_subset(master, real, "test"), _balanced_family_rows(master, ga, "test")]),
            test_cross_df=pd.concat([_subset(master, real, "test"), _balanced_family_rows(master, gb, "test")]),
        )
        assert_experiment_integrity(e, fams)
        exps.append(e)
    assert_same_real_images(exps)
    return exps


def assert_experiment_integrity(e: Experiment, fams: dict) -> None:
    ids = {k: set(getattr(e, k).image_id) for k in _ROLES}
    pairs = [(a, b) for i, a in enumerate(_ROLES) for b in _ROLES[i + 1:] if {a, b} != {"test_in_df", "test_cross_df"}]
    for a, b in pairs:                                  
        if ids[a] & ids[b]:
            raise AssertionError(f"[{e.name}] image overlap between {a} and {b}")
    unseen = set(fams[e.test_family])
    for k in ("train_df", "val_df"):
        if set(getattr(e, k).frequency_group) & unseen:
            raise AssertionError(f"[{e.name}] unseen family {e.test_family} leaked into {k}")
    # patient-level separation of every pair of data sets (train / val / test)
    subj = {"train": set(e.train_df.subject_id), "val": set(e.val_df.subject_id),
            "test": set(e.test_in_df.subject_id) | set(e.test_cross_df.subject_id)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = subj[a] & subj[b]
        if shared:
            raise AssertionError(f"[{e.name}] {len(shared)} patients shared by {a} and {b}")
    for k in _ROLES:
        d = getattr(e, k)
        if d.label.nunique() < 2:
            raise AssertionError(f"[{e.name}] {k} contains a single class")
        if d["image_id"].duplicated().any():
            raise AssertionError(f"[{e.name}] {k} contains a duplicated image")
    if set(e.test_in_df[e.test_in_df.label == 0].image_id) != set(e.test_cross_df[e.test_cross_df.label == 0].image_id):
        raise AssertionError(f"[{e.name}] test_in and test_cross use different real images")


def assert_same_real_images(exps: list[Experiment]) -> None:
    """Every experiment must use the identical real train / val / test images."""
    if not exps:
        return
    ref = {k: set(getattr(exps[0], k).query("label == 0").image_id) for k in _ROLES}
    for e in exps[1:]:
        for k in _ROLES:
            if set(getattr(e, k).query("label == 0").image_id) != ref[k]:
                raise AssertionError(f"[{e.name}] real images in {k} differ from the first experiment")


def experiments_overview(exps) -> pd.DataFrame:
    return pd.DataFrame([e.summary() for e in exps])


def class_balance_warning(exps) -> str | None:
    """A message when some role has more than 1.5x as many synthetic as real images (or the reverse)."""
    worst = 1.0
    for e in exps:
        for k in _ROLES:
            d = getattr(e, k)
            r, s = max(int((d.label == 0).sum()), 1), max(int((d.label == 1).sum()), 1)
            worst = max(worst, s / r, r / s)
    if worst > 1.5:
        return (f"Class ratio up to {worst:.1f}:1 between synthetic and real. AUROC is unaffected; "
                "AUPRC, accuracy and the threshold are not. Set config.BALANCE_SYNTH_TO_REAL = True "
                "for equal classes, or read AUPRC against the stored `prevalence`.")
    return None
