"""
PREPARE THE SIZE-LADDER IMAGES 
===================================================================
Every ladder image goes through exactly the preprocessing of the base set, by calling the SAME function
(preprocess.prepare_one):

    grayscale -> 256 x 256 (only if not already 256) -> one JPEG q95 round trip -> lossless PNG

so the ladder differs from the base experiments only in edit size, never in file format or compression.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from PIL import Image

import config
import size_config as sc

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
SPLITS = ("test",)

# technique -> (metadata file relative to <level>/<technique>/, image-path column, dicom-id column, area in pixels)
_IP_META = {
    "blob":    ("metadata/blob_synthetic_metadata.csv",              "synthetic_path",       "dicom_id",        lambda d: d["mask_pixels"]),
    "copy":    ("reports/copy_move_final_metadata.csv",              "synthetic_path",       "dicom_id",        lambda d: d["mask_pixels"]),
    "texture": ("reports/texture_transplant_final_metadata.csv",     "synthetic_image_path", "target_dicom_id", lambda d: d["patch_pixels"] ** 2),
}
_GROUP_OF_TECH = {tech: group for group, tech in sc.IP_TECH_OF.items()}      # blob -> IP_Blob, ...


def _load_preprocess():
    """Import Stage 2c (preprocess.py, next to this file) so that prepare_one is the very same function."""
    spec = importlib.util.spec_from_file_location("preprocess_stage2c", HERE / "preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# 1. file lists (from the generators' metadata, never from a folder scan)
# ----------------------------------------------------------------------------
def collect_ip(splits=SPLITS) -> pd.DataFrame:
    rows = []
    for level in sc.IP_LEVELS:
        for tech in sc.IP_SCRIPTS:
            if (tech, level) in sc.IP_SKIP:
                continue
            rel, path_col, id_col, area = _IP_META[tech]
            meta_p = sc.IP_LADDER_ROOT / level / tech / rel
            if not meta_p.exists():
                print(f"  missing metadata (not generated yet?): {meta_p}")
                continue
            d = pd.read_csv(meta_p, dtype={id_col: str})
            d = d[d["split"].astype(str).str.lower().isin(splits)]
            for r, a in zip(d.itertuples(index=False), (area(d) / 256 ** 2).tolist()):
                rows.append(dict(family="ip", group=_GROUP_OF_TECH[tech], level=level,
                                 target_area=sc.IP_TARGET[level], achieved_area=float(a),
                                 dicom_id=str(getattr(r, id_col)).lower(), split=str(r.split).lower(),
                                 source_path=str(getattr(r, path_col))))
    return pd.DataFrame(rows)


def collect_diffusion(splits=SPLITS) -> pd.DataFrame:
    if not Path(sc.DIFF_LADDER_META).exists():
        print(f"  missing metadata (not generated yet?): {sc.DIFF_LADDER_META}")
        return pd.DataFrame()
    m = pd.read_csv(sc.DIFF_LADDER_META, dtype={"dicom_id": str})
    m = m[m["split"].astype(str).str.lower().isin(splits)]
    rows = []
    for r in m.itertuples(index=False):
        for variant, col in (("full", r.full_path), ("paste", r.paste_path)):
            rows.append(dict(family="diffusion", group=f"Diffusion_{variant}", level=r.level,
                             target_area=float(r.target), achieved_area=float(r.area),
                             dicom_id=str(r.dicom_id).lower(), split=str(r.split).lower(), source_path=str(col)))
    return pd.DataFrame(rows)


def _real_lookup():
    """(dicom_id -> prepared path, dicom_id -> split) of the Real images, from the base set's prepared manifest."""
    base = pd.read_csv(config.DET_PREPARED_MANIFEST, dtype={"subject_id": str, "study_id": str})
    real = base[base["frequency_group"] == config.REAL_GROUP].copy()
    real["dicom_id"] = real["dicom_id"].str.lower()
    real = real.drop_duplicates("dicom_id").set_index("dicom_id")
    return real["prepared_path"], real["split"]


# ----------------------------------------------------------------------------
# 2. preparation
# ----------------------------------------------------------------------------
def _prepare(df: pd.DataFrame, root: Path, label: str) -> pd.DataFrame:
    if df.empty:
        print(f"{label}: nothing to prepare")
        return df
    pre = _load_preprocess()
    real_path, real_split = _real_lookup()

    df = df.copy()
    df["image_id"] = df["group"] + "@" + df["level"] + "__" + df["dicom_id"]
    df["label"] = 1
    df["prepared_path"] = [str(root / lv / g / f"{i}.png") for lv, g, i in zip(df["level"], df["group"], df["image_id"])]

    # Source integrity: every edited image must come from a Real test image of the base set, with the same split.
    unknown = df[~df["dicom_id"].isin(real_path.index)]
    wrong = df[df["dicom_id"].isin(real_path.index) & (df["dicom_id"].map(real_split) != df["split"])]
    if len(unknown) or len(wrong):
        raise RuntimeError(f"{label}: {len(unknown)} images have no Real source in the base manifest, "
                           f"{len(wrong)} have a different split than their source (e.g. "
                           f"{pd.concat([unknown, wrong])['image_id'].head(3).tolist()})")
    df["real_prepared_path"] = df["dicom_id"].map(real_path)

    n_new, errors = 0, []
    for i, r in enumerate(df.itertuples(index=False), 1):
        src, dst = Path(r.source_path), Path(r.prepared_path)
        try:
            if not src.exists():
                raise FileNotFoundError(src)
            stale = dst.exists() and dst.stat().st_mtime < src.stat().st_mtime
            if config.PREP_OVERWRITE or not dst.exists() or stale:
                pre.prepare_one(src, dst)                      # the SAME function as Stage 2c
                n_new += 1
        except Exception as exc:                               # noqa: BLE001
            errors.append((r.image_id, repr(exc)))
        if i % 500 == 0:
            print(f"  {label}: prepared {i}/{len(df)}")
    if errors:
        raise RuntimeError(f"{label}: {len(errors)} images failed, e.g. {errors[:3]}")

    # Freeze guard (same rule as Stage 2c): every prepared file is size x size, mode L.
    size, bad = config.PREP_IMAGE_SIZE, []
    for p in df["prepared_path"]:
        with Image.open(p) as im:
            if im.size != (size, size) or im.mode != "L":
                bad.append((p, im.size, im.mode))
    if bad:
        raise RuntimeError(f"{label}: {len(bad)} prepared files are not {size}x{size} mode L, e.g. {bad[:3]}")

    out = df[["image_id", "family", "group", "level", "target_area", "achieved_area", "dicom_id", "split",
              "label", "source_path", "prepared_path", "real_prepared_path"]]
    Path(root).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(root) / "ladder_manifest.csv", index=False)
    print(f"{label}: {len(out)} images ({n_new} new) -> {root}")
    return out


def prepare_ip_ladder(splits=SPLITS) -> pd.DataFrame:
    return _prepare(collect_ip(splits), Path(sc.PREP_IP_LADDER), "IP ladder")


def prepare_diffusion_ladder(splits=SPLITS) -> pd.DataFrame:
    return _prepare(collect_diffusion(splits), Path(sc.PREP_DIFF_LADDER), "Diffusion ladder")


# ----------------------------------------------------------------------------
# 3. report
# ----------------------------------------------------------------------------
def ladder_report(man: pd.DataFrame) -> pd.DataFrame:
    """Images per group and level, achieved vs target area, and how many images exist at EVERY level."""
    if man.empty:
        return man
    n_levels = man.groupby("group")["level"].nunique()
    per_image = man.groupby(["group", "dicom_id"])["level"].nunique().rename("k").reset_index()
    common = per_image[per_image["k"] == per_image["group"].map(n_levels)].groupby("group").size()
    rep = (man.groupby(["group", "level"])
              .agg(n=("image_id", "size"), target=("target_area", "first"),
                   mean_area=("achieved_area", "mean"), min_area=("achieved_area", "min"),
                   max_area=("achieved_area", "max")).round(4).reset_index())
    rep["n_at_every_level"] = rep["group"].map(common).fillna(0).astype(int)
    return rep.sort_values(["group", "target"]).reset_index(drop=True)


def prepare_all(splits=SPLITS):
    """Prepare both ladders and print the report. Returns (ip_manifest, diffusion_manifest)."""
    ip, dif = prepare_ip_ladder(splits), prepare_diffusion_ladder(splits)
    for name, m in (("IP", ip), ("Diffusion", dif)):
        if len(m):
            print(f"\n{name} ladder report (compare levels on the images present at every level):")
            print(ladder_report(m).to_string(index=False))
    return ip, dif


# ----------------------------------------------------------------------------
# 4. validation of the prepared ladder images
# ----------------------------------------------------------------------------
GROUP_ORDER = ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant", "Diffusion_paste", "Diffusion_full"]


def _read_manifests(ip=None, dif=None) -> pd.DataFrame:
    parts = []
    for m, root in ((ip, sc.PREP_IP_LADDER), (dif, sc.PREP_DIFF_LADDER)):
        if m is None:
            f = Path(root) / "ladder_manifest.csv"
            m = pd.read_csv(f, dtype={"dicom_id": str}) if f.exists() else pd.DataFrame()
        if len(m):
            parts.append(m)
    if not parts:
        raise FileNotFoundError("no ladder manifest found: run prepare_all() first")
    return pd.concat(parts, ignore_index=True)


def _gray(path) -> "np.ndarray":
    import numpy as np
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.int16)


def validate_ladder(ip=None, dif=None, area_tol=0.05, max_per_set=100, strict=True) -> dict:
    """Validate the prepared ladder images. HARD checks raise (strict=True); the leak check is a diagnostic.

    HARD   1. every file opens and is size x size, mode L (what the detectors will read)
           2. image ids are unique
           3. every edited image has an existing Real prepared partner (same dicom_id, test split)
           4. achieved edit area is within `area_tol` (relative) of the target, per group and level
    INFO   5. counts per group and level against the number of Real test images
           6. LEAK DIAGNOSTIC: share of pixels that differ from the Real partner by more than 3 grey levels,
              divided by the achieved edit area. A local edit (IP, diffusion 'paste') should give a ratio near 1
              (JPEG blocks at the border add a little). 'Diffusion_full' is expected to be far above 1: RadEdit
              re-renders the whole image through the VAE.
    """
    import numpy as np
    man = _read_manifests(ip, dif)
    size, hard = config.PREP_IMAGE_SIZE, {}

    bad_files = []
    for p in man["prepared_path"]:
        try:
            with Image.open(p) as im:
                if im.size != (size, size) or im.mode != "L":
                    bad_files.append((p, im.size, im.mode))
        except Exception as exc:                                   # noqa: BLE001
            bad_files.append((p, repr(exc), ""))
    hard["files_readable_256_L"] = (not bad_files, f"{len(bad_files)} bad file(s) {bad_files[:2]}")
    hard["unique_image_ids"] = (man["image_id"].is_unique, f"{int(man['image_id'].duplicated().sum())} duplicate(s)")
    miss_real = [p for p in man["real_prepared_path"].dropna().unique() if not Path(p).exists()]
    hard["real_partner_exists"] = (man["real_prepared_path"].notna().all() and not miss_real,
                                   f"{int(man['real_prepared_path'].isna().sum())} without partner, {len(miss_real)} missing file(s)")
    rel = (man["achieved_area"] / man["target_area"] - 1).abs()
    off = man[rel > area_tol].groupby(["group", "level"]).size()
    hard["area_within_tolerance"] = (off.empty, f"{int(off.sum()) if len(off) else 0} image(s) outside +-{100 * area_tol:.0f}% of target"
                                     + (f": {off.to_dict()}" if len(off) else ""))

    # counts against the Real test set
    base = pd.read_csv(config.DET_PREPARED_MANIFEST, dtype=str)
    n_real = int(((base["frequency_group"] == config.REAL_GROUP) & (base["split"] == "test")).sum())
    counts = (man.groupby(["group", "level"]).agg(n=("image_id", "size"), target=("target_area", "first"))
                 .reset_index().sort_values(["group", "target"]))
    counts["of_real_test"] = n_real

    # leak diagnostic on up to max_per_set images per (group, level)
    rows = []
    for (g, lv), d in man.groupby(["group", "level"]):
        d = d.sort_values("dicom_id").head(max_per_set)
        ch, mad = [], []
        for r in d.itertuples(index=False):
            diff = np.abs(_gray(r.prepared_path) - _gray(r.real_prepared_path))
            ch.append(float((diff > 3).mean())); mad.append(float(diff.mean()))
        rows.append(dict(group=g, level=lv, target=float(d.target_area.iloc[0]), area=float(d.achieved_area.mean()),
                         changed_frac=float(np.mean(ch)), mean_abs_diff=float(np.mean(mad))))
    leak = pd.DataFrame(rows).sort_values(["group", "target"]).reset_index(drop=True)
    leak["changed_over_area"] = (leak.changed_frac / leak.area).round(2)
    leak["flag"] = np.where((leak.group != "Diffusion_full") & (leak.changed_over_area > 2.0), "CHECK", "")
    leak[["area", "changed_frac", "mean_abs_diff"]] = leak[["area", "changed_frac", "mean_abs_diff"]].round(4)

    ok = all(v[0] for v in hard.values())
    print("LADDER VALIDATION")
    for k, (passed, msg) in hard.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {k}: {msg}")
    print(f"\nimages per group and level (Real test images: {n_real}):\n{counts.to_string(index=False)}")
    print("\nleak diagnostic (local edits should have changed_over_area near 1; Diffusion_full is expected to be high):")
    print(leak.to_string(index=False))
    print(f"\nOVERALL (hard checks): {'PASS' if ok else 'FAIL'}")
    if strict and not ok:
        raise RuntimeError("ladder validation failed: see the lines marked FAIL above")
    return {"ok": ok, "hard": hard, "counts": counts, "leak": leak}


# ----------------------------------------------------------------------------
# 5. one source image, every technique and size
# ----------------------------------------------------------------------------
def show_example(dicom_id=None, gain=4, save=True, seed=0):
    """Same source image for every technique (rows) and size (columns).

    Figure 1: the prepared images exactly as the detectors see them (Real in the first column).
    Figure 2: |edited - Real| x `gain`, which shows where each edit changed the image.
    Returns the dicom_id used. The image is picked at random (seed) among those present at every group and level.
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    man = _read_manifests()
    groups = [g for g in GROUP_ORDER if g in set(man["group"])]
    sizes = sorted(man["target_area"].round(4).unique())
    man = man.assign(t=man["target_area"].round(4))
    need = len(groups) * len(sizes)
    per_id = man.groupby("dicom_id").apply(lambda d: d[["group", "t"]].drop_duplicates().shape[0])
    complete = per_id[per_id == need].index.tolist()
    if dicom_id is None:
        pool = complete or per_id.sort_values(ascending=False).index.tolist()[:20]
        dicom_id = str(np.random.default_rng(seed).choice(pool))
    one = man[man["dicom_id"] == dicom_id]
    real = _gray(one["real_prepared_path"].iloc[0])

    def cell(g, t):
        r = one[(one["group"] == g) & (one["t"] == t)]
        return None if r.empty else r.iloc[0]

    out = {}
    for kind in ("images", "difference"):
        ncol = len(sizes) + (1 if kind == "images" else 0)
        fig, axes = plt.subplots(len(groups), ncol, figsize=(2.3 * ncol, 2.4 * len(groups)), squeeze=False)
        for i, g in enumerate(groups):
            for j in range(ncol):
                ax = axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
                if kind == "images" and j == 0:
                    ax.imshow(real, cmap="gray", vmin=0, vmax=255); ax.set_title("Real" if i == 0 else "", fontsize=9)
                    ax.set_ylabel(g, fontsize=9)
                    continue
                t = sizes[j - (1 if kind == "images" else 0)]
                r = cell(g, t)
                if i == 0:
                    ax.set_title(f"{100 * t:g}% target", fontsize=9)
                if j == (0 if kind == "difference" else 1) and kind == "difference":
                    ax.set_ylabel(g, fontsize=9)
                if r is None:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes); continue
                img = _gray(r["prepared_path"])
                if kind == "images":
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(np.clip(np.abs(img - real) * gain, 0, 255), cmap="magma", vmin=0, vmax=255)
                ax.set_xlabel(f"area {100 * r['achieved_area']:.2f}%", fontsize=8)
        title = ("Prepared images (what the detectors read)" if kind == "images"
                 else f"|edited - Real| x {gain}  (bright = changed pixels)")
        fig.suptitle(f"{title}\nsource image {dicom_id[:18]}...", fontsize=10)
        fig.tight_layout()
        if save:
            Path(sc.FIG_DIR).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(sc.FIG_DIR) / f"ladder_example_{kind}.png", dpi=170, bbox_inches="tight")
        out[kind] = fig
    plt.show()
    return dicom_id


if __name__ == "__main__":
    prepare_all()
