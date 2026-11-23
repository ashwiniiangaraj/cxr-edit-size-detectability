"""
size_config.py - settings for the edit-size study.
"""
from pathlib import Path

import config


# =============================================================================
# 1. SIZE LEVELS
# =============================================================================
# --- 1.1 Image-processing (IP) ladder: 5 levels on the 256 x 256 canvas ------
IP_LEVELS = ["a0p5", "a1", "a2", "a5", "a10"]
IP_TARGET = {"a0p5": 0.005, "a1": 0.01, "a2": 0.02, "a5": 0.05, "a10": 0.10}

# Copy-move / texture patch side in pixels (area = side^2 / 256^2).
IP_SIDE = {"a0p5": 18, "a1": 26, "a2": 36, "a5": 57, "a10": 81}

# Blob radius as a fraction of the ROI width (179 px). 
IP_BLOB_DIVISOR = 12.8
IP_BLOB_R = {"a0p5": 0.0515, "a1": 0.0728, "a2": 0.1030, "a5": 0.1643, "a10": 0.2324}

# Guard: the radii are only valid for the divisor they were calibrated with.
if config.IP_STRENGTH == "strong" and float(config.IP_CONFIG["blob"]["sigma_divisor"]) != IP_BLOB_DIVISOR:
    raise RuntimeError(
        f"size_config.IP_BLOB_R was calibrated for sigma_divisor {IP_BLOB_DIVISOR}, "
        f"but config has {config.IP_CONFIG['blob']['sigma_divisor']}: recalibrate IP_BLOB_R."
    )

IP_SCRIPTS = {
    "blob": "generation/ip_blob_generate.py",
    "copy": "generation/ip_copy_generate.py",
    "texture": "generation/ip_texture_generate.py",
}

# Levels that cannot be generated. None on the 0.5-10 % ladder 
IP_SKIP = set()

# Levels where some placements fail the separation rule 
IP_ACCEPT_PARTIAL = {("copy", "a10")}

IP_TECH_OF = {"IP_Blob": "blob", "IP_CopyMove": "copy", "IP_TextureTransplant": "texture"}
DIFF_VARIANT_OF = {"Diffusion_full": "full", "Diffusion_paste": "paste"}

# --- 1.2 Diffusion (RadEdit) ladder: area as a share of the 512 x 512 canvas --
DIFF_LEVELS = {"d0p5": 0.005, "d1": 0.01, "d2": 0.02, "d5": 0.05, "d10": 0.10}

# The detectors are already trained, so only test images are generated.
DIFF_SPLITS = ("test",)

DIFF_MASK_TOL = 0.02     # realised mask area within +-2 % (relative) of the target
DIFF_MASK_TRIES = 40     # interior centres tried per image before giving up

DIFF_VARIANTS = {
    "full": "whole RadEdit output",
    "paste": "edit pasted back into the source",
}

# =============================================================================
# 2. FOLDERS AND RESULT FILES
# =============================================================================
IP_LADDER_ROOT = Path(config.IP_SYNTHETIC_ROOT).parent / "IP_size_ladder"
DIFF_LADDER_ROOT = Path(config.DIFFUSION_ROOT).parent / "Diffusion_size_ladder"

PREP_IP_LADDER = Path(config.DET_PREPARED_DIR).parent / "prepared_ladder"
PREP_DIFF_LADDER = Path(config.DET_PREPARED_DIR).parent / "prepared_diff_ladder"

FIG_DIR = Path(config.DET_RESULTS_CSV).parent / "figures"

LADDER_FREQ_DIR = Path(config.FORENSIC_ANALYSIS_DIR) / "ladder_frequency"  
LADDER_SCORE_DIR = Path(config.DET_RESULTS_DIR) / "ladder_scoring"          

DIFF_LADDER_META = DIFF_LADDER_ROOT / "ladder_metadata.csv"


# =============================================================================
# 3. DETECTORS AND EVALUATION
# =============================================================================
MODELS = list(config.MODELS_TO_RUN)     

# Trained detectors that are scored on the ladders.
IP_TRAIN_SLUG = "IP__to__CycleGAN"               
DIFF_TRAIN_SLUGS = {
    "Diffusion-trained": "Diffusion__to__CycleGAN",
    "EditOnly-trained": "Diffusion_EditOnly__to__IP",
}

BOOTSTRAP_B = config.BOOTSTRAP_N   
USABLE_AUROC = 0.70    
   
# =============================================================================
# 4. FIGURE COLOURS
# =============================================================================
COLOURS = {
    "blob": "#2a78d6", "copy": "#eb6834", "texture": "#1baf7a",
    "full": "#2a78d6", "paste": "#eb6834",
}
MARKERS = {"blob": "o", "copy": "s", "texture": "^", "full": "o", "paste": "s"}