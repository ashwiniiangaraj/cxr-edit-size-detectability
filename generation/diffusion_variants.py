"""
diffusion_variants.py - the two controlled variants of the production Diffusion images.

"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("generation").resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                    
import gen_common                

EDIT_ONLY_ROOT = Path(config.SYNTHETIC_DIR) / "diffusion_editonly"
RECONSTRUCTION_ROOT = Path(config.SYNTHETIC_DIR) / "diffusion_reconstruction"
GENERATION_SIZE = int(config.DIFFUSION_GENERATION_SIZE)      
FINAL_SIZE = int(config.DIFFUSION_FINAL_SIZE)                


# =============================================================================
# 1. SHARED HELPERS
# =============================================================================

def source_file(row):
    """Path of the real source image (full-size JPEG) for one metadata row."""
    return (Path(config.DIFFUSION_SOURCE_ROOT) / str(row.split).lower() / str(row.source_disease_status).lower()
            / f"p{row.subject_id}_s{row.study_id}_{row.dicom_id}.jpg")


def load_rgb(path, size=FINAL_SIZE):
    """Open an image as 8-bit RGB and resize it to size x size with Lanczos."""
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def load_metadata():
    """Production Diffusion metadata (all columns as text, so ids keep their exact spelling)."""
    return pd.read_csv(config.DIFFUSION_PRODUCTION_METADATA, dtype=str)


def output_path(root, row):
    """<root>/<split>/<direction>/<dicom_id>.png"""
    return Path(root) / str(row.split).lower() / row.direction / f"{row.dicom_id}.png"


def check_output_folder(root, written_paths, label):
    """Fail if a split / direction folder holds PNGs that this run did not write (see gen_common)."""
    written = [Path(p) for p in written_paths]
    for folder in sorted({p.parent for p in written}):
        gen_common.assert_output_matches(folder, [p for p in written if p.parent == folder], label)


def paste_back(src256, gen256, mask512, area_fraction):
    """
    Soft paste-back of the RadEdit edit region into the source image.

    src256, gen256 : uint8 arrays (256, 256, 3), the real source and the RadEdit output
    mask512        : PIL "L" image, the edit mask on the 512 canvas
    area_fraction  : share of the image that the mask covers (sets the blur)

    The blur sigma is config.diffusion_paste_sigma(area_fraction), so the 10-90 % edge width is
    EDGE_FRAC x the edit's equivalent diameter at every size. Returns a PIL RGB image.
    """
    sigma = config.diffusion_paste_sigma(area_fraction)
    soft = mask512.convert("L").resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.BOX).filter(
        ImageFilter.GaussianBlur(sigma))
    alpha = (np.asarray(soft, dtype=float) / 255.0)[..., None]          # 0 = keep source, 1 = use RadEdit
    comp = src256.astype(float) * (1 - alpha) + gen256.astype(float) * alpha
    return Image.fromarray(np.clip(np.rint(comp), 0, 255).astype(np.uint8))


# =============================================================================
# 2. DIFFUSION_EDITONLY
# =============================================================================

def make_edit_only(meta=None, out_root=EDIT_ONLY_ROOT):
    """Write the EditOnly images and their metadata CSV. Returns the metadata DataFrame."""
    meta = load_metadata() if meta is None else meta
    print(len(meta), "Diffusion records in the production metadata")

    rows, written, missing = [], [], 0
    for r in meta.itertuples(index=False):
        gen_path, src_path, mask_path = Path(r.output_path), source_file(r), Path(r.edit_mask)
        absent = [p.name for p in (gen_path, src_path, mask_path) if not p.exists()]
        if absent:
            missing += 1
            print("missing:", r.dicom_id, absent)
            continue

        gen, src = load_rgb(gen_path), load_rgb(src_path)
        mask = Image.open(mask_path).convert("L")                       
        area = float((np.asarray(mask) > 127).mean())                  
        comp = paste_back(np.asarray(src), np.asarray(gen), mask, area)

        out = output_path(out_root, r)
        out.parent.mkdir(parents=True, exist_ok=True)
        comp.save(out, format="PNG")
        written.append(out)

        gen_f, src_f, comp_f = (np.asarray(x, dtype=float) for x in (gen, src, comp))
        rows.append({
            "dicom_id": r.dicom_id, "split": str(r.split).lower(), "direction": r.direction, "path": str(out),
            "edit_area_frac": area,
            "paste_sigma_px": config.diffusion_paste_sigma(area),
            "mean_abs_change_full_diffusion": float(np.abs(gen_f - src_f).mean()),    
            "mean_abs_change_edit_only": float(np.abs(comp_f - src_f).mean()),       
            "generator_version": gen_common.GENERATOR_VERSION,
        })

    check_output_folder(out_root, written, "Diffusion_EditOnly")
    out_meta = pd.DataFrame(rows)
    out_meta.to_csv(Path(out_root) / "diffusion_editonly_metadata.csv", index=False)

    print(f"\nwritten: {len(out_meta)}   missing: {missing}   folder: {out_root}")
    if len(out_meta):
        print(f"edit area              : {100 * out_meta.edit_area_frac.mean():.2f}% of the image (mean)")
        print(f"paste blur sigma       : {out_meta.paste_sigma_px.mean():.2f} px (mean)")
        print(f"mean |change| full     : {out_meta.mean_abs_change_full_diffusion.mean():.3f} grey levels (RadEdit vs source)")
        print(f"mean |change| edit-only: {out_meta.mean_abs_change_edit_only.mean():.3f} grey levels (edit-only vs source)")
    return out_meta


# =============================================================================
# 3. DIFFUSION_RECONSTRUCTION
# =============================================================================

def make_reconstruction(meta=None, out_root=RECONSTRUCTION_ROOT):
    """Write the VAE-only images and their metadata CSV. Returns the metadata DataFrame."""
    import torch                                    
    from diffusers import AutoencoderKL

    meta = load_metadata() if meta is None else meta
    device = f"cuda:{config.DIFFUSION_GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32                            
    print("device:", device, "| dtype:", dtype)
    vae = AutoencoderKL.from_pretrained(config.DIFFUSION_VAE_MODEL_ID, torch_dtype=dtype).to(device).eval()

    rows, written, missing = [], [], 0
    for k, r in enumerate(meta.itertuples(index=False), 1):
        src_path = source_file(r)
        if not src_path.exists():
            missing += 1
            print("missing source:", src_path.name)
            continue

        # Source squashed to 512, as RadEdit sees it; encode -> decode and nothing else.
        src512 = load_rgb(src_path, GENERATION_SIZE)
        x = (torch.from_numpy(np.asarray(src512, dtype=np.float32) / 127.5 - 1.0)
             .permute(2, 0, 1)[None].to(device, dtype))
        with torch.inference_mode():
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        y = y.float()
        if not torch.isfinite(y).all():
            raise RuntimeError(f"NaN/inf in the VAE output for {r.dicom_id}: check the dtype and rerun.")
        arr = ((y.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rec = Image.fromarray(arr).resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.LANCZOS)

        out = output_path(out_root, r)
        out.parent.mkdir(parents=True, exist_ok=True)
        rec.save(out, format="PNG")
        written.append(out)

        src = np.asarray(load_rgb(src_path), dtype=float)
        rec_f = np.asarray(rec, dtype=float)
        row = {
            "dicom_id": r.dicom_id, "split": str(r.split).lower(), "direction": r.direction, "path": str(out),
            "mean_abs_change_recon": float(np.abs(rec_f - src).mean()),
            "generator_version": gen_common.GENERATOR_VERSION,
        }
        gen_path = Path(r.output_path)
        if gen_path.exists():                        
            gen_f = np.asarray(load_rgb(gen_path), dtype=float)
            row["mean_abs_change_full_diffusion"] = float(np.abs(gen_f - src).mean())
            row["mean_abs_diff_diffusion_vs_recon"] = float(np.abs(gen_f - rec_f).mean())
        rows.append(row)
        if k % 100 == 0:
            print(f"{k}/{len(meta)}")

    check_output_folder(out_root, written, "Diffusion_Reconstruction")
    out_meta = pd.DataFrame(rows)
    out_meta.to_csv(Path(out_root) / "diffusion_reconstruction_metadata.csv", index=False)

    print(f"\nwritten: {len(out_meta)}   missing: {missing}   folder: {out_root}")
    if len(out_meta):
        print(f"mean |change| recon vs source        : {out_meta.mean_abs_change_recon.mean():.3f} grey levels")
        if "mean_abs_change_full_diffusion" in out_meta:
            print(f"mean |change| RadEdit vs source      : {out_meta.mean_abs_change_full_diffusion.mean():.3f}")
            print(f"mean |RadEdit - recon| (what the VAE does not explain): "
                  f"{out_meta.mean_abs_diff_diffusion_vs_recon.mean():.3f}")
    return out_meta


# =============================================================================
# 4. COMMAND LINE / %run
# =============================================================================

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("all", "edit_only", "reconstruction"):
        raise SystemExit(f"unknown variant '{which}': use all, edit_only or reconstruction")
    if which in ("all", "edit_only"):
        make_edit_only()
    if which in ("all", "reconstruction"):
        make_reconstruction()