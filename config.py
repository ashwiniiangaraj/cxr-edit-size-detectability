"""
Central configuration for the MIMIC-CXR real-vs-synthetic chest X-ray study.

Every path, seed, generator setting and analysis constant lives here.

Sections
    1  Project paths and dataset sampling (one real pool)
    2  Synthetic generation: image-processing (IP) edits, CycleGAN, RadEdit
    3  Forensic feature characterisation 
    4  Detection pipeline

Scaling the run: set CXR_POOL_SIZE (500 smoke test, 5000 or 10000 full run). All counts,
splits and the run folder follow from it.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
# Total number of image files to download
POOL_SIZE_PER_POOL = int(os.environ.get("CXR_POOL_SIZE", 2500))

# Project folder. 
# Change the RUNS_BASE and the DEFAULT_RUN folder based on the actual project file location
_RUNS_BASE = Path("/data/aa8624")
_DEFAULT_RUN = os.environ.get("CXR_RUN_NAME", "thesis_2500")
# The root can also be overridden with an environment variable (smoke tests).
PROJECT_ROOT = Path(os.environ.get("CXR_PROJECT_ROOT", _RUNS_BASE / _DEFAULT_RUN))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
REAL_DIR = DATA_DIR / "real"

# =============================================================================
# PHYSIONET / MIMIC-CXR-JPG
# =============================================================================

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"

METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
CHEXPERT_FILE = "mimic-cxr-2.0.0-chexpert.csv.gz"

# Leave blank to be prompted securely at runtime.
PHYSIONET_USERNAME = ""
PHYSIONET_PASSWORD = ""

# =============================================================================
# DATASET SAMPLING
# =============================================================================
# Normal and pathology fraction
NORMAL_FRACTION = 0.50
PATHOLOGY_FRACTION = 0.50

# Dynamic image-level split fractions.
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

# IMAGE counts per split
SPLIT_IMAGE_COUNTS = {
    "train": int(round(POOL_SIZE_PER_POOL * TRAIN_FRACTION)),
    "val": int(round(POOL_SIZE_PER_POOL * VAL_FRACTION)),
    "test": int(round(POOL_SIZE_PER_POOL * TEST_FRACTION)),
}

# Multiple images from a patient ARE allowed.
# Patient-level separation is enforced only at the split/pool level.
ONE_IMAGE_PER_PATIENT = False

# Keep this True to require exact 50/50 normal/pathology in each pool.
CLASS_BALANCE = True

# Reproducibility.
RANDOM_SEED = 42

# =============================================================================
# IMAGE ELIGIBILITY
# =============================================================================

ALLOWED_VIEWS = {"PA", "AP"}

# These are the 12 pathology findings deliberately choosen.
PATHOLOGY_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

NO_FINDING_COLUMN = "No Finding"

# =============================================================================
# DOWNLOAD SETTINGS
# =============================================================================

REQUEST_TIMEOUT = 180
MAX_RETRIES = 5

# 1 MiB network-transfer chunks. 
CHUNK_SIZE = 1024 * 1024

# Refresh catalogue CSV/GZIP files each run.
# Existing JPEGs are still reused if already present.
REFRESH_CATALOGUE = True

# The original working PhysioNet login flow used certificate verification
# disabled in the requests session.
VERIFY_SSL = False
# =============================================================================
# DATASET VALIDATION
# =============================================================================

# Output report produced by dataset_validation.py
VALIDATION_REPORT = MANIFEST_DIR / "dataset_validation_report.txt"
# Validate the actual JPEG contents with Pillow.
# This is read-only and does not modify any images.
VALIDATE_IMAGE_CONTENT = True

# =============================================================================
# SYNTHETIC IMAGE GENERATION — IP TECHNIQUES
# =============================================================================
#
# These settings define the LOCKED production configuration for the three
# image-manipulation / synthetic-generation techniques used in Stage 2:
#
#   1. Blob
#   2. Copy-Move
#   3. Texture Transplant
# =============================================================================

SYNTHETIC_DIR = DATA_DIR / "synthetic"
IP_SYNTHETIC_ROOT = SYNTHETIC_DIR / "IP"

# Generator-wide protocol variables
GEN_RESIZE_FILTER = "lanczos"
GEN_IMAGE_FORMAT = "PNG"
# -----------------------------------------------------------------------------
# IP Techniques
# -----------------------------------------------------------------------------
#   blob      Gaussian-blurred disc: 10-90 % width = 2.56 * sigma, sigma = r / divisor
#             r = D/2  ->  divisor = 2.56 / (2 * EDGE_FRAC)               = 12.8
#   copy      linear ramp + Gaussian (measured 10-90 % width 4.3-6.5 px for 45-62 px
#             patches, target 5.1-7.0 px): feather fraction 0.10 already close
#   texture   linear ramp + smoothstep: 10-90 % width = 0.608 * feather_width, so
#             feather_fraction = EDGE_FRAC * 1.128 / 0.608                = 0.185
#   diffusion paste-back blur: sigma = EDGE_FRAC * D(256 px) / 2.56  (see
#             diffusion_paste_sigma() below).  The "full" RadEdit output has no
#             controlled border: the model creates it.
EDGE_FRAC = 0.10
BLOB_SIGMA_DIVISOR = round(2.56 / (2.0 * EDGE_FRAC), 1)          # 12.8
TEXTURE_FEATHER_FRACTION = 0.185

IP_STRENGTH = "strong"      # "mild" | "moderate" | "strong"

# Multiple tiers were considered at first to assess the impact of the parameter choices. All experiments here use the "strong" tier.
_IP_TIERS = {
    #            blob opacity     blob radius      copy/texture patch   feather  how many best candidates to pick from
    "mild":     dict(op=(0.05, 0.10), rad=(0.05, 0.10), patch=(0.10, 0.15), feather=0.30, top_copy=5,  top_tex=5),
    "moderate": dict(op=(0.20, 0.35), rad=(0.08, 0.14), patch=(0.18, 0.25), feather=0.20, top_copy=15, top_tex=15),
    "strong":   dict(op=(0.35, 0.55), rad=(0.128, 0.180), patch=(0.25, 0.35), feather=0.10, top_copy=30, top_tex=30),
}
_T = _IP_TIERS[IP_STRENGTH]

IP_CONFIG = {
    "source_pool": "real",
    "master_seed": RANDOM_SEED,
    "working_size": [256, 256],
    "jpeg_quality": 95,          

    "blob": {
        "roi_width_fraction": 0.70,
        "roi_height_fraction": 0.75,
        "opacity_min": _T["op"][0],
        "opacity_max": _T["op"][1],
        "radius_min": _T["rad"][0],
        "radius_max": _T["rad"][1],
        "sigma_divisor": BLOB_SIGMA_DIVISOR if IP_STRENGTH == "strong" else 2.5,   # mild / moderate tiers use a fixed softness
        "background_threshold": 128,
        "bright_target": 215.0,
        "dark_target": 45.0,
        "mask_threshold": 0.10,
    },

    "copy": {
        "patch_min": _T["patch"][0],
        "patch_max": _T["patch"][1],
        "rotation_min": 0.0,
        "rotation_max": 15.0,
        "scale_min": 1.00,
        "scale_max": 1.15,
        "feather_fraction": _T["feather"],
        # Central region of interest (pixels on the 256 x 256 canvas).
        "roi_x": 39,
        "roi_y": 32,
        "roi_width": 178,
        "roi_height": 192,
        "top_destination_candidates": _T["top_copy"],   
        # placement rules 
        "min_distance_multiplier": 1.2,      
        "n_destination_candidates": 100,     
    },

    "texture": {
        # ROI rescaled from 224: (33, 28, 157, 168) x 256/224
        "roi_x": 38,
        "roi_y": 32,
        "roi_width": 179,
        "roi_height": 192,
        "patch_min": _T["patch"][0],
        "patch_max": _T["patch"][1],
        "rotation_min": 0.0,
        "rotation_max": 15.0,
        "scale_min": 1.00,
        "scale_max": 1.15,
        "feather_fraction": TEXTURE_FEATHER_FRACTION if IP_STRENGTH == "strong" else _T["feather"],
        "n_donor_candidates": 30,
        "top_donors": _T["top_tex"],
        "min_separation_multiplier": 1.5,
    },
}

# ============================================================
# CycleGAN — modular direct-PyTorch implementation
# ============================================================

CYCLEGAN_MASTER_SEED = RANDOM_SEED

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
CYCLEGAN_MANIFEST = MANIFEST_DIR / "real_all.csv"
CYCLEGAN_SOURCE_ROOT = REAL_DIR
CYCLEGAN_TRAIN_SPLIT = "train"

CYCLEGAN_DOMAIN_A = "normal"
CYCLEGAN_DOMAIN_B = "pathology"

CYCLEGAN_EXPECTED_TRAIN_IMAGES = int(round(POOL_SIZE_PER_POOL * TRAIN_FRACTION))   

CYCLEGAN_REQUIRE_TRAIN_BALANCE = False

# ------------------------------------------------------------
# Image / loader
# ------------------------------------------------------------
CYCLEGAN_IMAGE_SIZE = 256
CYCLEGAN_INPUT_NC = 3
CYCLEGAN_OUTPUT_NC = 3

CYCLEGAN_BATCH_SIZE = 1
CYCLEGAN_NUM_WORKERS = 0
CYCLEGAN_PIN_MEMORY = True


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
# 50 epochs for the final run (a 20-epoch pilot was used to check the pipeline).
CYCLEGAN_EPOCHS = 50

CYCLEGAN_LEARNING_RATE = 0.0002
CYCLEGAN_BETA1 = 0.5
CYCLEGAN_BETA2 = 0.999

CYCLEGAN_LAMBDA_CYCLE = 10.0

# Identity-loss weight (0.5 x LAMBDA_CYCLE is the usual CycleGAN choice; the
# value used here is kept explicit).
CYCLEGAN_LAMBDA_IDENTITY = 5.0

# ------------------------------------------------------------
# Architecture
# ------------------------------------------------------------
CYCLEGAN_GENERATOR_RES_BLOCKS = 6
CYCLEGAN_GENERATOR_BASE_CHANNELS = 64

CYCLEGAN_DISCRIMINATOR_BASE_CHANNELS = 64
CYCLEGAN_DISCRIMINATOR_LEAKY_RELU = 0.2

# ------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------
CYCLEGAN_INIT_MEAN = 0.0
CYCLEGAN_INIT_STD = 0.02

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
CYCLEGAN_DEVICE = "cuda"
CYCLEGAN_GPU_INDEX = 0

# ------------------------------------------------------------
# Checkpoint / reports
# ------------------------------------------------------------
# The checkpoint folder, loss file, log and config snapshot all use this name.
CYCLEGAN_EXPERIMENT_NAME = "cyclegan_lanczos_256_150ep"

CYCLEGAN_CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "calibration_cyclegan"
    / "checkpoints"
)

CYCLEGAN_CHECKPOINT_DIR = (
    CYCLEGAN_CHECKPOINT_ROOT / CYCLEGAN_EXPERIMENT_NAME
)

CYCLEGAN_REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration_cyclegan"
    / "reports"
)

CYCLEGAN_CONFIG_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration_cyclegan"
    / "configs"
)

# Final checkpoint follows CYCLEGAN_EPOCHS.
CYCLEGAN_FINAL_CHECKPOINT = (
    CYCLEGAN_CHECKPOINT_DIR / f"epoch_{CYCLEGAN_EPOCHS}.pth"
)

CYCLEGAN_LOSS_HISTORY = (
    CYCLEGAN_REPORT_DIR
    / f"{CYCLEGAN_EXPERIMENT_NAME}_losses.csv"
)

CYCLEGAN_TRAINING_LOG = (
    CYCLEGAN_REPORT_DIR
    / f"{CYCLEGAN_EXPERIMENT_NAME}_training.log"
)

# ------------------------------------------------------------
# Production generation
# ------------------------------------------------------------
CYCLEGAN_OUTPUT_ROOT = SYNTHETIC_DIR / "cyclegan"


CYCLEGAN_GENERATION_METADATA = (
    CYCLEGAN_OUTPUT_ROOT
    / "reports"
    / f"cyclegan_epoch{CYCLEGAN_EPOCHS}_production_metadata.csv"
)

CYCLEGAN_GENERATION_ERRORS = (
    CYCLEGAN_OUTPUT_ROOT
    / "reports"
    / f"cyclegan_epoch{CYCLEGAN_EPOCHS}_production_errors.csv"
)

CYCLEGAN_GENERATION_CONFIG = (
    CYCLEGAN_OUTPUT_ROOT
    / "reports"
    / f"cyclegan_epoch{CYCLEGAN_EPOCHS}_production_config.json"
)

CYCLEGAN_OUTPUT_NORMAL_TO_PATHOLOGY = (
    CYCLEGAN_OUTPUT_ROOT / "normal_to_pathology"
)

CYCLEGAN_OUTPUT_PATHOLOGY_TO_NORMAL = (
    CYCLEGAN_OUTPUT_ROOT / "pathology_to_normal"
)

# ------------------------------------------------------------
# Provenance (CycleGAN)
# ------------------------------------------------------------
# The CycleGAN is implemented directly in PyTorch (generation/cyclegan_train.py).
# It does not use the repository's train.py entry point and loads no pretrained weights.
CYCLEGAN_IMPLEMENTATION = "direct_pytorch_from_working_html"
CYCLEGAN_PRETRAINED = False

# Reference repository, recorded in each run's training config JSON for provenance only.
# The repository is not cloned, imported or run, and these fields do not identify any weights.
CYCLEGAN_REPO_URL = (
    "https://github.com/taesungp/"
    "contrastive-unpaired-translation.git"
)
CYCLEGAN_REPO_COMMIT = "b3ac297708dfb6f7589d04662277e53c0d579c27"

# For visualisation
CYCLEGAN_VISUAL_QA_N_PER_DIRECTION = 3
CYCLEGAN_VISUAL_QA_PERCENTILE = 90
CYCLEGAN_VISUAL_QA_RANDOM_SEED = RANDOM_SEED
CYCLEGAN_VISUAL_QA_SPLIT = "test"
# ============================================================
# DIFFUSION / RADEDIT CONFIGURATION
# Based on the working diffusion HTML notebook.
# ============================================================

DIFFUSION_MASTER_SEED = RANDOM_SEED

# Manifest / source data
DIFFUSION_MANIFEST = MANIFEST_DIR / "real_all.csv"
DIFFUSION_SOURCE_ROOT = REAL_DIR
DIFFUSION_REQUIRE_TOTAL_IMAGES = POOL_SIZE_PER_POOL   # derived
DIFFUSION_REQUIRE_BALANCED_SOURCE = True

# RadEdit model components
DIFFUSION_MODEL_ID = "microsoft/radedit"
DIFFUSION_VAE_MODEL_ID = "stabilityai/sdxl-vae"
DIFFUSION_TEXT_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"

# Device / precision
DIFFUSION_DEVICE = "cuda"
DIFFUSION_GPU_INDEX = 0
DIFFUSION_DTYPE = "float16"

# RadEdit generation settings
DIFFUSION_GENERATION_SIZE = 512
DIFFUSION_FINAL_SIZE = 256   
DIFFUSION_WEIGHT = 7.5
DIFFUSION_NUM_INFERENCE_STEPS = 200
DIFFUSION_SKIP_RATIO = 0.30
DIFFUSION_ETA = 1.0
DIFFUSION_INVERT_PROMPT = ""

# Final prompt 
DIFFUSION_NORMAL_TO_PATHOLOGY_PROMPTS = [
    "chest X-ray with focal airspace consolidation",
    "chest X-ray with focal pulmonary opacity",
    "chest X-ray with focal pulmonary infiltrate",
]

DIFFUSION_PATHOLOGY_TO_NORMAL_PROMPTS = [
    "chest X-ray with no focal airspace consolidation",
    "chest X-ray with no focal pulmonary opacity",
    "chest X-ray with no focal pulmonary infiltrate",
]

# NIAID CXR lung segmentation
DIFFUSION_NIAID_REPO = (
    PROJECT_ROOT / "data" / "calibration_diffusion" / "lung_cxr_segmentation"
)
DIFFUSION_NIAID_REPO_URL = "https://github.com/niaid/lung_cxr_segmentation.git"
DIFFUSION_NIAID_CHECKPOINT = (
    DIFFUSION_NIAID_REPO
    / "segment_lung_cxr"
    / "data"
    / "weights"
    / "cxr_lung_segment_including_heart_512.pt"
)

# Lung / focal edit-mask settings
DIFFUSION_MASK_SIZE = 512
DIFFUSION_BOUNDARY_MARGIN = 12
DIFFUSION_REGION_WIDTH_FRACTION = 0.15
DIFFUSION_REGION_HEIGHT_FRACTION = 0.85
DIFFUSION_MIN_REGION_WIDTH = 20
DIFFUSION_MIN_REGION_HEIGHT = 20
DIFFUSION_MIN_EDIT_AREA = 200    # only used by the single-lung fallback mask (target = None)

# Production edit area (share of the 512 x 512 canvas); None = small single-lung fallback mask.
DIFFUSION_TARGET_AREA_FRACTION = 0.05
DIFFUSION_AREA_TOLERANCE = 0.02      # realised area within +-2 % (relative) of the target
DIFFUSION_AREA_TRIES = 40            # interior centres tried per image before giving up

# Soft paste-back used for the Diffusion_EditOnly variant: Gaussian blur on the 256 px
# mask, sigma chosen so the 10-90 % edge width = EDGE_FRAC x equivalent diameter.
DIFFUSION_PASTE_MIN_SIGMA = 0.5


def diffusion_paste_sigma(area_fraction):
    """Paste-back blur sigma (px at 256) for an edit covering `area_fraction` of the image."""
    import math
    d256 = 2.0 * math.sqrt(area_fraction / math.pi) * DIFFUSION_FINAL_SIZE
    return max(DIFFUSION_PASTE_MIN_SIGMA, EDGE_FRAC * d256 / 2.56)

# Production directories
DIFFUSION_ROOT = DATA_DIR / "synthetic" / "diffusion"
DIFFUSION_IMAGE_ROOT = DIFFUSION_ROOT / "images"
DIFFUSION_REPORT_ROOT = DIFFUSION_ROOT / "reports"

DIFFUSION_LUNG_MASK_ROOT = (
    DATA_DIR / "calibration_diffusion" / "production_lung_masks" / "masks"
)
DIFFUSION_EDIT_MASK_ROOT = (
    DATA_DIR / "calibration_diffusion" / "production_radedit_masks" / "edit_masks"
)

DIFFUSION_LUNG_MASK_METADATA = (
    DATA_DIR / "calibration_diffusion"
    / "production_lung_masks"
    / "production_lung_mask_metadata.csv"
)
DIFFUSION_LUNG_MASK_ERRORS = (
    DATA_DIR / "calibration_diffusion"
    / "production_lung_masks"
    / "production_lung_mask_errors.csv"
)

DIFFUSION_EDIT_MASK_METADATA = (
    DATA_DIR / "calibration_diffusion"
    / "production_radedit_masks"
    / "production_radedit_edit_mask_metadata.csv"
)
DIFFUSION_EDIT_MASK_ERRORS = (
    DATA_DIR / "calibration_diffusion"
    / "production_radedit_masks"
    / "production_radedit_edit_mask_errors.csv"
)

DIFFUSION_PRODUCTION_METADATA = (
    DIFFUSION_REPORT_ROOT / "radedit_production_metadata.csv"
)
DIFFUSION_PRODUCTION_ERRORS = (
    DIFFUSION_REPORT_ROOT / "radedit_production_errors.csv"
)
DIFFUSION_PRODUCTION_CONFIG = (
    DIFFUSION_REPORT_ROOT / "radedit_production_config.json"
)

# Visual QA 
DIFFUSION_VISUAL_QA_N_PER_DIRECTION = 24
DIFFUSION_VISUAL_QA_SPLIT = None
DIFFUSION_VISUAL_QA_RANDOM_SEED = RANDOM_SEED
DIFFUSION_VISUAL_QA_OUTPUT_DIR = DIFFUSION_REPORT_ROOT / "radedit_visual_QA"


# ============================================================
# FORENSIC IMAGE CHARACTERISATION
# ============================================================


FORENSIC_CHARACTERISATION_ROOT = (
    DATA_DIR
    / "artifact_characterization"
    / "stage1_characterisation"
)


# ============================================================
# Input image families / dataset definitions
# ============================================================
# ============================================================

FORENSIC_CHARACTERISATION_DATASETS = {
    "REAL": {
        "root": REAL_DIR,
        "group": "Real",
        "frequency_group": "Real",
        "generator": "Real",
        "label": 0,
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "IP_BLOB": {
        "root": SYNTHETIC_DIR / "IP" / "blob",
        "group": "IP",
        "frequency_group": "IP_Blob",
        "generator": "Blob",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "IP_COPY_MOVE": {
        "root": SYNTHETIC_DIR / "IP" / "copy",
        "group": "IP",
        "frequency_group": "IP_CopyMove",
        "generator": "Copy-Move",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "IP_TEXTURE_TRANSPLANT": {
        "root": SYNTHETIC_DIR / "IP" / "texture",
        "group": "IP",
        "frequency_group": "IP_TextureTransplant",
        "generator": "Texture-Transplant",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "CYCLEGAN": {
        "root": SYNTHETIC_DIR / "cyclegan",
        "group": "CycleGAN",
        "frequency_group": "CycleGAN",
        "generator": "CycleGAN",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "DIFFUSION": {
        "root": SYNTHETIC_DIR / "diffusion",
        "group": "Diffusion",
        "frequency_group": "Diffusion",
        "generator": "Diffusion",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },

    "DIFFUSION_EDIT_ONLY": {
        "root": SYNTHETIC_DIR / "diffusion_editonly",
        "group": "Diffusion_EditOnly",
        "frequency_group": "Diffusion_EditOnly",
        "generator": "Diffusion_EditOnly",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },
    "DIFFUSION_RECONSTRUCTION": {
        "root": SYNTHETIC_DIR / "diffusion_reconstruction",
        "group": "Diffusion_Reconstruction",
        "frequency_group": "Diffusion_Reconstruction",
        "generator": "Diffusion_Reconstruction",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png"],
    },
}


# ============================================================
# Within-IP groups
# ============================================================

FORENSIC_CHARACTERISATION_IMAGE_SIZE = 256


# ============================================================
# Spectral / Fourier
# ============================================================

FORENSIC_SPECTRAL_FEATURES = [
    "spectral_hfer",
    "spectral_slope",
    "spectral_slope_r",
    "spectral_anisotropy",
    "spectral_total_energy",
    "spectral_high_energy",
]

FORENSIC_SPECTRAL_RADIAL_BINS = 64          
FORENSIC_SPECTRAL_HIGH_FREQUENCY_FRACTION = 0.50   
FORENSIC_SPECTRAL_SLOPE_MIN_BIN = 2         
FORENSIC_SPECTRAL_SLOPE_MAX_BIN = None      
FORENSIC_EPSILON = 1e-12


# ============================================================
# Wavelet
# ============================================================
FORENSIC_WAVELET_NAME = "db1"
FORENSIC_WAVELET_LEVELS = 3

FORENSIC_WAVELET_FEATURES = [
    "wavelet_LH1_energy",
    "wavelet_HL1_energy",
    "wavelet_HH1_energy",
    "wavelet_LH_HL_corr_L1",
    "wavelet_LH_HH_corr_L1",
    "wavelet_HL_HH_corr_L1",

    "wavelet_LH2_energy",
    "wavelet_HL2_energy",
    "wavelet_HH2_energy",
    "wavelet_LH_HL_corr_L2",
    "wavelet_LH_HH_corr_L2",
    "wavelet_HL_HH_corr_L2",

    "wavelet_LH3_energy",
    "wavelet_HL3_energy",
    "wavelet_HH3_energy",
    "wavelet_LH_HL_corr_L3",
    "wavelet_LH_HH_corr_L3",
    "wavelet_HL_HH_corr_L3",
]


# ============================================================
# LBP
# ============================================================
FORENSIC_LBP_POINTS = 8
FORENSIC_LBP_RADIUS = 1
FORENSIC_LBP_METHOD = "uniform"
FORENSIC_LBP_HISTOGRAM_BINS = 10            


# ============================================================
# SRM / residual
# ============================================================
FORENSIC_SRM_FEATURES = [
    "srm_mean_abs",
    "srm_variance",
    "srm_std",
]

FORENSIC_SRM_FILTERS = [
    # 3x3 high-pass residual filters.
    [
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1],
    ],
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ],
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
]

FORENSIC_SRM_CLIP = None
# ============================================================
# Noise / statistical
# ============================================================

FORENSIC_NOISE_FEATURES = [
    "noise_residual_mean",
    "noise_residual_abs_mean",
    "noise_residual_variance",
    "noise_residual_std",
    "noise_residual_rms",
]

FORENSIC_NOISE_FILTER = "laplacian"
FORENSIC_NOISE_KERNEL_SIZE = 3


# ============================================================
# Statistical analysis
# ============================================================
FORENSIC_ALPHA = 0.05
FORENSIC_MULTIPLE_TEST_METHOD = "fdr_bh"

FORENSIC_PCA_COMPONENTS = 2
FORENSIC_PCA_RANDOM_STATE = RANDOM_SEED


# ============================================================
# Logistic regression
# ============================================================

FORENSIC_LOGISTIC_C_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

FORENSIC_LOGISTIC_MAX_ITER = 5000


# ============================================================
# Outputs
# ============================================================

FORENSIC_FEATURE_CSV = (
    FORENSIC_CHARACTERISATION_ROOT
    / "per_image_forensic_characteristics.csv"
)

FORENSIC_FEATURE_ERRORS_CSV = (
    FORENSIC_CHARACTERISATION_ROOT
    / "feature_extraction_errors.csv"
)


# ------------------------------------------------------------
# Statistical / PCA analysis outputs
# ------------------------------------------------------------

FORENSIC_ANALYSIS_DIR = (
    FORENSIC_CHARACTERISATION_ROOT
    / "analysis"
)

FORENSIC_DESCRIPTIVE_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "descriptive_statistics.csv"
)

FORENSIC_KRUSKAL_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "kruskal_wallis.csv"
)

FORENSIC_PAIRWISE_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "pairwise_mann_whitney_fdr.csv"
)


FORENSIC_SINGLE_FEATURE_AUROC_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "single_feature_one_vs_rest_auroc.csv"
)

FORENSIC_FAMILY_SUMMARY_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "feature_family_summary.csv"
)

FORENSIC_GROUP_MEANS_CSV = (
    FORENSIC_ANALYSIS_DIR
    / "group_means.csv"
)

FORENSIC_PCA_DIR = (
    FORENSIC_ANALYSIS_DIR
    / "pca"
)


# ------------------------------------------------------------
# Logistic-regression outputs
# ------------------------------------------------------------


# ============================================================
# Plotting
# ============================================================

FORENSIC_PLOT_DPI = 200
FORENSIC_PLOT_FIGSIZE = (8, 6)


# #############################################################################
# #############################################################################
#
#   DETECTION PIPELINE  
#
# #############################################################################
# #############################################################################


# =============================================================================
# 0. RUN MODE
# =============================================================================
# QUICK_RUN = True shrinks seeds / epochs / bootstrap draws so the whole
QUICK_RUN = os.environ.get("CXR_QUICK_RUN", "0") == "1"

# "auto" -> CUDA when available, otherwise CPU.  Or force "cuda" / "cpu".
DET_DEVICE = "auto"
DET_GPU_INDEX = 0

# Parallel workers for CPU feature extraction (1 = no multiprocessing).
DET_N_JOBS = max(1, (os.cpu_count() or 2) - 1)


# =============================================================================
# 1. OUTPUT LOCATIONS
# =============================================================================
DETECTION_ROOT = DATA_DIR / "detection"

DET_MANIFEST_DIR = DETECTION_ROOT / "manifests"
DET_PREPARED_DIR = DETECTION_ROOT / "prepared"          
DET_EMBEDDING_DIR = DETECTION_ROOT / "embeddings"       
DET_MODEL_DIR = DETECTION_ROOT / "models"               
DET_RESULTS_DIR = DETECTION_ROOT / "results"
DET_FIGURE_DIR = DETECTION_ROOT / "figures"
DET_REPORT_DIR = DETECTION_ROOT / "reports"

DET_MASTER_MANIFEST = DET_MANIFEST_DIR / "dataset_master.csv"        
DET_VALIDATION_REPORT_TXT = DET_REPORT_DIR / "dataset_validation_report.txt"
DET_VALIDATION_REPORT_JSON = DET_REPORT_DIR / "dataset_validation_report.json"
DET_PREPARED_MANIFEST = DET_MANIFEST_DIR / "dataset_prepared.csv"    
DET_RADIAL_PROFILES = FORENSIC_CHARACTERISATION_ROOT / "radial_profiles.npz"

DET_RESULTS_CSV = DET_RESULTS_DIR / "cross_generator_results.csv"    
DET_PREDICTIONS_CSV = DET_RESULTS_DIR / "cross_generator_predictions.csv"
# =============================================================================
# 2. DATASET INVENTORY / VALIDATION  
# =============================================================================

DATASET_EXCLUDE_DIR_NAMES = {
    "masks", "mask", "reports", "config", "configs",
    "metadata", "logs",
}
DATASET_EXCLUDE_FILE_SUFFIXES = ("_mask", "_edit_mask", "_lung_mask")

DATASET_EXPECTED_COUNTS = {
    "Real": POOL_SIZE_PER_POOL,
    "IP_Blob": POOL_SIZE_PER_POOL,
    "IP_CopyMove": POOL_SIZE_PER_POOL,
    "IP_TextureTransplant": POOL_SIZE_PER_POOL,
    "CycleGAN": POOL_SIZE_PER_POOL,
    "Diffusion": POOL_SIZE_PER_POOL,
    "Diffusion_EditOnly": POOL_SIZE_PER_POOL,
    "Diffusion_Reconstruction": POOL_SIZE_PER_POOL,
}

DATASET_ALLOWED_SHORTFALL = {"Diffusion": 0.01,
                             "Diffusion_EditOnly": 0.01,  # 496/500 = 0.8%
                             "Diffusion_Reconstruction": 0.01}
DATASET_COUNT_TOLERANCE = 0          

SPLIT_NAMES = ("train", "val", "test")

SOURCE_POOL_MANIFEST = MANIFEST_DIR / "real_all.csv"

VALIDATION_STRICT = True             # raise if a HARD check fails
VALIDATION_NEAR_DUP_HASH_SIZE = 16   # average-hash is HASH x HASH bits
VALIDATION_NEAR_DUP_MAX_HAMMING = 6  
VALIDATION_MASK_MAX_UNIQUE_VALUES = 4  
VALIDATION_SHORTCUT_AUROC_WARN = 0.70

# =============================================================================
# 3. PREPROCESSING  (Stage 2c)
# =============================================================================
PREP_IMAGE_SIZE = FORENSIC_CHARACTERISATION_IMAGE_SIZE   # 256
PREP_RESIZE_INTERPOLATION = "area"          # "area" | "lanczos" | "bicubic"
PREP_RESIZE_BACKEND = "pil_lanczos"
PREP_HARMONISE_JPEG = True
PREP_JPEG_QUALITY = 95
PREP_OVERWRITE = False


# =============================================================================
# 4. FORENSIC FEATURES  (Stage 3)
# =============================================================================
FEATURE_COLUMNS = (
    list(FORENSIC_SPECTRAL_FEATURES)
    + list(FORENSIC_WAVELET_FEATURES)
    + [f"lbp_bin_{i:02d}" for i in range(FORENSIC_LBP_HISTOGRAM_BINS)]
    + list(FORENSIC_SRM_FEATURES)
    + list(FORENSIC_NOISE_FEATURES)
)
assert len(FEATURE_COLUMNS) == 42, len(FEATURE_COLUMNS)

FEATURE_FAMILY_OF = {}
for _f in FORENSIC_SPECTRAL_FEATURES: FEATURE_FAMILY_OF[_f] = "spectral"
for _f in FORENSIC_WAVELET_FEATURES:  FEATURE_FAMILY_OF[_f] = "wavelet"
for _f in FEATURE_COLUMNS:
    if _f.startswith("lbp_"):         FEATURE_FAMILY_OF[_f] = "lbp"
for _f in FORENSIC_SRM_FEATURES:      FEATURE_FAMILY_OF[_f] = "srm"
for _f in FORENSIC_NOISE_FEATURES:    FEATURE_FAMILY_OF[_f] = "noise"

SPECTRAL_USE_HANN_WINDOW = True        
SPECTRAL_ANISOTROPY_SECTORS = 12       
SPECTRAL_ANISOTROPY_MIN_RADIUS = 0.10  
SPECTRAL_RESIDUAL_MEDIAN_SIZE = 3      
RESIDUAL_SPECTRAL_FEATURES = [f.replace("spectral_", "specres_", 1) for f in FORENSIC_SPECTRAL_FEATURES]
RESIDUAL_FEATURE_CSV = FORENSIC_CHARACTERISATION_ROOT / "residual_spectral_features.csv"
FEATURES_LOG10 = (
    ["spectral_total_energy", "spectral_high_energy"]
    + [f for f in FORENSIC_WAVELET_FEATURES if f.endswith("_energy")]
    + ["srm_mean_abs", "srm_variance", "srm_std",
       "noise_residual_abs_mean", "noise_residual_variance",
       "noise_residual_std", "noise_residual_rms"]
)
FEATURE_LOG10_EPS = 1e-12

SCREENING_GROUP_COLUMN = "frequency_group"
SCREENING_SPLIT = "train"                    
SCREENING_TOP_K = 10
PAIRWISE_TESTS_ENABLED = True
SCREENING_GROUPS = [
    "Real", "IP_Blob", "IP_CopyMove", "IP_TextureTransplant", "CycleGAN",
    "Diffusion", "Diffusion_EditOnly", "Diffusion_Reconstruction",
]
REFERENCE_GROUP = "Real"                     
REDUNDANCY_SPEARMAN_THRESHOLD = 0.95          

# =============================================================================
# 5. CROSS-GENERATOR EXPERIMENTS  
# =============================================================================

FAMILY_SETS = {
    "3_family": {
        "IP": ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"],
        "CycleGAN": ["CycleGAN"],
        "Diffusion": ["Diffusion"],
    },
    "5_family": {
        "IP_Blob": ["IP_Blob"],
        "IP_CopyMove": ["IP_CopyMove"],
        "IP_TextureTransplant": ["IP_TextureTransplant"],
        "CycleGAN": ["CycleGAN"],
        "Diffusion": ["Diffusion"],
    },
}

FAMILY_SETS["diffusion_split"] = {
    "IP": ["IP_Blob", "IP_CopyMove", "IP_TextureTransplant"],
    "CycleGAN": ["CycleGAN"],
    "Diffusion": ["Diffusion"],
    "Diffusion_EditOnly": ["Diffusion_EditOnly"],
    "Diffusion_Reconstruction": ["Diffusion_Reconstruction"],
}

ACTIVE_FAMILY_SET = "diffusion_split"
BALANCE_SYNTH_TO_REAL = True
REAL_GROUP = "Real"
assert REFERENCE_GROUP == REAL_GROUP, "Stage 3 reference group must be the real class"

EXPERIMENT_SEEDS = [42, 43, 44]

MODELS_TO_RUN = [
    "spectral_lr", "handcrafted_lr", "lbp_lr",
    "cnn_densenet", "supcon", "npr",
    "clip_frozen", "biomedclip_frozen", "rad_dino", "dinov2_frozen",
]

THRESHOLD_RULE = "youden"            # "youden" | "balanced_accuracy"
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = RANDOM_SEED

SAVE_FITTED_MODELS = True            # needed by the robustness stage

# =============================================================================
# 6. MODEL HYPER-PARAMETERS  
# =============================================================================
# --- 5.1 Spectral logistic regression
SPECTRAL_MODEL_FEATURES = list(FORENSIC_SPECTRAL_FEATURES)
LR_C_GRID = list(FORENSIC_LOGISTIC_C_GRID)       
LR_MAX_ITER = FORENSIC_LOGISTIC_MAX_ITER
LR_CLASS_WEIGHT = "balanced"                     

# --- 5.2 Handcrafted-feature model (selection on TRAINING data only)
HANDCRAFTED_SELECTION = "kw_epsilon"      
HANDCRAFTED_TOP_K_GRID = [5, 10]          
HANDCRAFTED_FDR_ALPHA = FORENSIC_ALPHA

# --- 5.3 CNN detector (DenseNet121)
CNN_PRETRAINED = os.environ.get("CXR_PRETRAINED", "1") == "1"   # ImageNet weights
CNN_EPOCHS = 15
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_EARLY_STOP_PATIENCE = 4
CNN_NUM_WORKERS = 0
CNN_USE_AMP = True                   # mixed precision (CUDA only)
CNN_AUG_HFLIP = True                 # forensic-safe augmentation only
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- 5.4 Supervised contrastive learning
SUPCON_EPOCHS = 30
SUPCON_BATCH_SIZE = 64
SUPCON_LEARNING_RATE = 1e-4
SUPCON_WEIGHT_DECAY = 1e-4
SUPCON_TEMPERATURE = 0.07
SUPCON_PROJECTION_DIM = 128
SUPCON_EVAL_EVERY = 5
SUPCON_AUG_TRANSLATE = 0.05
SUPCON_AUG_ROTATE_DEG = 5
SUPCON_AUG_BRIGHTNESS = 0.2
SUPCON_AUG_CONTRAST = 0.2
SUPCON_PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# --- 5.5 Frozen foundation-model baselines (embeddings cached, backbone never trained)
CLIP_VARIANTS = {
    "clip_frozen": {
        "backend": "hf",                                # pip install transformers
        "model_id": "openai/clip-vit-large-patch14",    # as in Ojha et al. 2023
    },
    "biomedclip_frozen": {
        "backend": "open_clip",                         # pip install open_clip_torch
        "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    },
    # Self-supervised image encoders (no text tower).
    "rad_dino": {
        "backend": "hf_dino",
        "model_id": "microsoft/rad-dino",         # chest-X-ray DINOv2 
        "image_size": 252,
    },
    "dinov2_frozen": {
        "backend": "hf_dino",
        "model_id": "facebook/dinov2-base",       # general-purpose DINOv2 
        "image_size": 252,
    },
}
CLIP_BATCH_SIZE = 64
CLIP_C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

# --- 5.6 LBP-only logistic regression
LBP_MODEL_FEATURES = [f"lbp_bin_{i:02d}" for i in range(FORENSIC_LBP_HISTOGRAM_BINS)]

# --- 5.7 NPR forensic CNN
NPR_EPOCHS = 30
NPR_BATCH_SIZE = 32
NPR_LEARNING_RATE = 2e-4
NPR_WEIGHT_DECAY = 0.0
NPR_EARLY_STOP_PATIENCE = 6
NPR_NUM_WORKERS = 0
NPR_USE_AMP = True
NPR_AUG_HFLIP = True
# =============================================================================
# 7. ROBUSTNESS  
# =============================================================================
ROBUSTNESS_JPEG_QUALITIES = [90, 70, 50]
ROBUSTNESS_RESIZE_SCALES = [0.75, 0.5]
ROBUSTNESS_NOISE_SIGMAS = [0.01, 0.02, 0.04]
ROBUSTNESS_BLUR_SIGMAS = [0.5, 1.0]      # [] to skip blur
ROBUSTNESS_NOISE_SEED = RANDOM_SEED

# =============================================================================
# 8. FORENSIC DISTANCE 
# =============================================================================
# profile_l2 | profile_jsd | feature_euclid
ASSOCIATION_BOOTSTRAP_N = 2000

# =============================================================================
# 9. FIGURES
# =============================================================================
FIG_DPI = FORENSIC_PLOT_DPI
FIG_FORMAT = "png"
FIG_SAVE = True
GROUP_COLOURS = {
    "Real": "#1f77b4", "IP_Blob": "#ff7f0e",
    "IP_CopyMove": "#d62728", "IP_TextureTransplant": "#9467bd",
    "CycleGAN": "#2ca02c", "Diffusion": "#8c564b",
    "Diffusion_EditOnly": "#e377c2",
    "Diffusion_Reconstruction": "#17becf",
}

SENSITIVITY_VARIANTS = {                       
    "q75":    {"PREP_JPEG_QUALITY": 75},
    "noharm": {"PREP_HARMONISE_JPEG": False},    
}
SENSITIVITY_MODELS_FAST = ["spectral_lr", "handcrafted_lr", "lbp_lr"]
SENSITIVITY_MODELS_ALL = SENSITIVITY_MODELS_FAST + ["cnn_densenet", "npr", "clip_frozen",
                                                    "biomedclip_frozen", "dinov2_frozen", "rad_dino"]
SENSITIVITY_SEED = 42

# =============================================================================
# QUICK-RUN OVERRIDES 
# =============================================================================
if QUICK_RUN:
    EXPERIMENT_SEEDS = [EXPERIMENT_SEEDS[0]]
    CNN_EPOCHS = 2
    SUPCON_EPOCHS = 2
    SUPCON_EVAL_EVERY = 1
    BOOTSTRAP_N = 100
    ASSOCIATION_BOOTSTRAP_N = 200
    HANDCRAFTED_TOP_K_GRID = [5]
    LR_C_GRID = [0.1, 1.0]
    CLIP_C_GRID = [1.0, 10.0]
    SUPCON_PROBE_C_GRID = [1.0]
    NPR_EPOCHS = 2                   
    ROBUSTNESS_JPEG_QUALITIES = [70]
    ROBUSTNESS_RESIZE_SCALES = [0.5]
    ROBUSTNESS_NOISE_SIGMAS = [0.02]
    ROBUSTNESS_BLUR_SIGMAS = []
