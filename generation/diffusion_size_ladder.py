"""
RadEdit size ladder. For each source image and edit-area level
(size_config.DIFF_LEVELS: 0.5, 1, 2, 5, 10 % of the 512x512 canvas), builds a lung-confined elliptical mask
of the target area and runs RadEdit with the production seed and prompt, so only the mask size changes.
Masks use the production per-image seed, so they are nested across levels (the 5 % mask is the production mask).
Saves two 256x256 images per level: `full` (whole RadEdit output) and `paste` (RadEdit pixels inside the
feathered mask only, as in Diffusion_EditOnly).
Functions: build_masks() and check_masks(MK) on CPU; generate(masks) on GPU (resumable).
Writes: <Diffusion_size_ladder>/<level>/{masks, full, paste}, ladder_metadata.csv, ladder_errors.csv.
"""
import importlib.util
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

import config
import size_config as sc


def _load_sibling(name):
    """Import a stage-1 script that sits next to this file (they are not installed as a package)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("generation").resolve()
S = int(config.DIFFUSION_MASK_SIZE)                                   
FINAL = int(config.DIFFUSION_FINAL_SIZE)                               
EM = _load_sibling("diffusion_edit_masks")                            
MASK_RULE = "v2: seed = DIFFUSION_MASTER_SEED + manifest row index (identical for every level and for production)"


# ----------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------
def mask_for_target(safe, frac, rng):
    """Ladder wrapper around the production routine (same geometry, ladder tolerance / tries)."""
    return EM.mask_for_target(safe, frac, rng, tol=sc.DIFF_MASK_TOL, tries=sc.DIFF_MASK_TRIES)


def load_manifest():
    m = pd.read_csv(config.DIFFUSION_MANIFEST)
    m["split"] = m["split"].astype(str).str.lower()
    return m


def build_masks(levels=None, splits=None, verbose=True):
    """Build (or reuse) the edit masks. Returns (MK, ERR).  MK columns: level, target, dicom_id, idx, area."""
    levels = levels or sc.DIFF_LEVELS
    sc.DIFF_LADDER_ROOT.mkdir(parents=True, exist_ok=True)
    marker = sc.DIFF_LADDER_ROOT / "mask_seed_rule.txt"
    if list(sc.DIFF_LADDER_ROOT.glob("*/masks/*_edit_mask.png")) and not (marker.exists() and marker.read_text() == MASK_RULE):
        raise RuntimeError(f"Ladder masks from an older seed rule exist under {sc.DIFF_LADDER_ROOT}. "
                           "Delete the <level>/masks folders (and any images made from them) and run again.")
    marker.write_text(MASK_RULE)
    manifest = load_manifest()
    sel = manifest[manifest.split.isin(splits or sc.DIFF_SPLITS)]
    ex = pd.read_csv(config.DIFFUSION_EDIT_MASK_METADATA)
    ex = ex[ex.split.astype(str).str.lower().isin(splits or sc.DIFF_SPLITS)]
    frac = ex.edit_area / S ** 2
    if verbose:
        print(f"{len(sel)} source images | EXISTING production edits cover median {100 * frac.median():.2f}% "
              f"of the image (range {100 * frac.min():.2f}-{100 * frac.max():.2f}%)")
    recs, errs = [], []
    for lv, target in levels.items():
        (sc.DIFF_LADDER_ROOT / lv / "masks").mkdir(parents=True, exist_ok=True)
        for idx, row in sel.iterrows():
            d = str(row["dicom_id"])
            mp = sc.DIFF_LADDER_ROOT / lv / "masks" / f"{d}_edit_mask.png"
            try:
                if not mp.exists():
                    lung = np.asarray(Image.open(Path(config.DIFFUSION_LUNG_MASK_ROOT) / f"{d}_lung_mask.png").convert("L"))
                    lung = cv2.resize((lung > 0).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
                    rng = np.random.default_rng(EM.source_row_seed(int(idx)))      
                    m = mask_for_target(EM.safe_lungs(lung), target, rng)
                    Image.fromarray(m * 255).save(mp)
                area = float((np.asarray(Image.open(mp).convert("L")) > 127).mean())
                recs.append(dict(level=lv, target=target, dicom_id=d, idx=int(idx), area=area))
            except Exception as e:                                  # noqa: BLE001
                errs.append(dict(level=lv, dicom_id=d, error=repr(e)))
    MK, ERR = pd.DataFrame(recs), pd.DataFrame(errs)
    if verbose and len(MK):
        print("\nMasks per level (of", len(sel), "source images):")
        print(MK.groupby("level").agg(n=("area", "size"), target=("target", "first"), mean_area=("area", "mean"),
                                      min_area=("area", "min"), max_area=("area", "max")).reindex(list(levels)).round(4))
        if len(ERR):
            print("masks that could not reach the target:", ERR.groupby("level").size().to_dict())
    return MK, ERR


def check_masks(MK, verbose=True):
    """Check the built masks. Returns a one-row-per-level-step DataFrame.

    area hit      realised area / target, per level
    nested        share of the smaller mask that lies inside the next larger mask (1.0 = same centre, nested)
    d5 vs prod    share of images whose 5 % ladder mask is pixel-identical to the production mask
    """
    levels = list(sc.DIFF_LEVELS)
    rows, same_prod, n_prod = [], 0, 0
    inside = {i: [] for i in range(len(levels) - 1)}
    for d, g in MK.groupby("dicom_id"):
        m = {lv: np.asarray(Image.open(sc.DIFF_LADDER_ROOT / lv / "masks" / f"{d}_edit_mask.png").convert("L")) > 127
             for lv in g.level}
        for i in range(len(levels) - 1):
            a, b = levels[i], levels[i + 1]
            if a in m and b in m:
                inside[i].append(float((m[a] & m[b]).sum() / m[a].sum()))
        pp = Path(config.DIFFUSION_EDIT_MASK_ROOT) / f"{d}_edit_mask.png"
        if "d5" in m and pp.exists():
            n_prod += 1
            same_prod += int(np.array_equal(m["d5"], np.asarray(Image.open(pp).convert("L")) > 127))
    for i in range(len(levels) - 1):
        v = inside[i]
        rows.append(dict(small=levels[i], large=levels[i + 1], n=len(v),
                         mean_inside=round(float(np.mean(v)), 3) if v else np.nan,
                         min_inside=round(float(np.min(v)), 3) if v else np.nan))
    out = pd.DataFrame(rows)
    area = MK.assign(hit=MK.area / MK.target).groupby("level").hit.agg(["mean", "min", "max"]).reindex(levels).round(3)
    if verbose:
        print("realised area / target:\n", area.to_string())
        print("\nnesting between neighbouring levels:\n", out.to_string(index=False))
        print(f"\n5 % ladder mask identical to production mask: {same_prod}/{n_prod}")
    return out


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------
# One paste-back rule for the production EditOnly images and the ladder's "paste" images:
# the function lives in diffusion_variants.py (area-matched blur, config.diffusion_paste_sigma).
paste_back = _load_sibling("diffusion_variants").paste_back


def _load_production_module():
    spec = importlib.util.spec_from_file_location("diffusion_generate", HERE / "diffusion_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(MK, pipeline=None):
    """Run RadEdit for every row of MK (output of build_masks). Resumable. Needs a GPU."""
    import torch
    G = _load_production_module()
    manifest = load_manifest()
    if pipeline is None:
        pipeline, _ = G.load_radedit()
    meta_p = sc.DIFF_LADDER_META
    meta = pd.read_csv(meta_p, dtype={"dicom_id": str}) if meta_p.exists() else pd.DataFrame()
    have = set(zip(meta.level, meta.dicom_id)) if len(meta) else set()
    errs, t0 = [], time.time()
    for r in MK.itertuples(index=False):
        if (r.level, r.dicom_id) in have:
            continue
        row = manifest.loc[r.idx]
        disease, split = str(row["disease_status"]).lower(), str(row["split"]).lower()
        direction = "Normal_to_Pathology" if disease == "normal" else "Pathology_to_Normal"
        bank = config.DIFFUSION_NORMAL_TO_PATHOLOGY_PROMPTS if disease == "normal" else config.DIFFUSION_PATHOLOGY_TO_NORMAL_PROMPTS
        seed = G.make_seed(r.idx, config.DIFFUSION_WEIGHT)             
        prompt = bank[seed % len(bank)]
        outs = {v: sc.DIFF_LADDER_ROOT / r.level / v / split / f"{r.dicom_id}.png" for v in sc.DIFF_VARIANTS}
        try:
            src_path = G.source_path(row)
            src512 = Image.open(src_path).convert("RGB").resize((config.DIFFUSION_GENERATION_SIZE,) * 2, Image.Resampling.LANCZOS)
            em = Image.open(sc.DIFF_LADDER_ROOT / r.level / "masks" / f"{r.dicom_id}_edit_mask.png").convert("L").resize(src512.size, Image.Resampling.NEAREST)
            G.set_all_seeds(seed)
            with torch.inference_mode():
                res = pipeline(prompt=prompt, image=src512, edit_mask=em, keep_mask=ImageOps.invert(em),
                               weights=[config.DIFFUSION_WEIGHT], invert_prompt=config.DIFFUSION_INVERT_PROMPT,
                               num_inference_steps=config.DIFFUSION_NUM_INFERENCE_STEPS, skip_ratio=config.DIFFUSION_SKIP_RATIO,
                               eta=config.DIFFUSION_ETA, prog_bar=False, output_type="pil")
            gen = (res[0] if isinstance(res, list) else res.images[0]).convert("RGB")
            gen256 = gen.resize((FINAL, FINAL), Image.Resampling.LANCZOS)              
            src256 = np.asarray(Image.open(src_path).convert("RGB").resize((FINAL, FINAL), Image.Resampling.LANCZOS))
            versions = {"full": gen256, "paste": paste_back(src256, np.asarray(gen256), em, r.area)}
            for v, im in versions.items():
                outs[v].parent.mkdir(parents=True, exist_ok=True)
                im.save(outs[v], format="PNG")
            meta = pd.concat([meta, pd.DataFrame([dict(level=r.level, target=r.target, area=r.area, dicom_id=r.dicom_id, split=split,
                              direction=direction, seed=seed, prompt=prompt, full_path=str(outs["full"]), paste_path=str(outs["paste"]))])],
                             ignore_index=True)
            meta.to_csv(meta_p, index=False)
            if len(meta) % 20 == 0:
                print(f"{len(meta)}/{len(MK)} done, {(time.time() - t0) / 60:.1f} min")
        except Exception as e:                                     
            errs.append(dict(level=r.level, dicom_id=r.dicom_id, error=repr(e)))
            traceback.print_exc()
    pd.DataFrame(errs).to_csv(sc.DIFF_LADDER_ROOT / "ladder_errors.csv", index=False)
    print("finished:", len(meta), "image pairs written | errors this run:", len(errs), "|", meta_p)
    if len(meta):
        print(meta.groupby("level").size().reindex(list(sc.DIFF_LEVELS)))
    return meta