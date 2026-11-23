# Edit Size and Cross-Family Transfer in Synthetic Chest X-Ray Detection

Code for the MSc report *Edit Size and Cross-Family Transfer in Synthetic Chest X-Ray Detection* (Ashwini Angaraj, 2026).

This repository contains the code for a study of how small a synthetic edit to a chest radiograph can be before detectors fail to flag it, measured as the smallest detectable edit area under a fixed detection criterion. Starting from 2,500 MIMIC-CXR-JPG radiographs, it generates seven synthetic sets (three classical image-processing edits, CycleGAN and three RadEdit diffusion variants) grouped into five families (image-processing edits, CycleGAN, RadEdit diffusion, EditOnly and Reconstruction). Ten detectors (feature-based classifiers, convolutional networks and frozen foundation-model encoders) are trained on one family and evaluated on the others across 20 source→target configurations, and the edited area is varied from 0.5% to 10% of the image. The estimates are then stress-tested with preprocessing and image-alteration controls: a provenance (resampling-route) control, JPEG-quality sensitivity (quality 75 and no JPEG), a local-blur control and a JPEG-compression shift.

- **Data.** The real images come from MIMIC-CXR-JPG 2.1.0 and need PhysioNet credentialed access. Neither they nor the synthetic images derived from them are redistributed; both are rebuilt with this code.
- **Licence.** CC BY 4.0 (see `LICENSE`).

> **Note.** Project folders were renamed after the experiments were run, so saved outputs may refer to the old names (for example `stage1_generation`).

## Requirements

- Python 3.12 and CUDA 12.8 (pinned versions in `requirements.txt`)
- A GPU for CycleGAN, RadEdit and the neural detectors; the other steps run on CPU
- A PhysioNet account with access to MIMIC-CXR-JPG 2.1.0

## Setup

```bash
git clone https://github.com/ashwiniiangaraj/cxr-edit-size-detectability.git
cd cxr-edit-size-detectability
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export CXR_PROJECT_ROOT=/path/to/project_folder   # or edit _RUNS_BASE / _DEFAULT_RUN in config.py
jupyter lab                                       
```

All data and results are written under `CXR_PROJECT_ROOT`. PhysioNet credentials are requested at runtime; do not store them in the repository. `CXR_QUICK_RUN=1` runs reduced experiments.

## Configuration

All settings are environment variables read by `config.py`. Set them before launching Jupyter.

| Variable | Purpose | Default |
|---|---|---|
| `CXR_PROJECT_ROOT` | Folder where data, models and results are written | verify before execution (required) |
| `CXR_POOL_SIZE` | Images per pool in the working subset | `2500` |
| `CXR_RUN_NAME` | Name of the run sub-folder under the output base | `thesis_2500` |
| `CXR_SOURCE_POOL` | Folder with the already-downloaded 10,000-image pool (notebook 1, only when rebuilding the subset from it) | `<parent of CXR_PROJECT_ROOT>/newRun_10000` |
| `CXR_QUICK_RUN` | Set to 1 for a small smoke test (not for reported results) | off |

## Dataset

Notebook 1 downloads a pool of 10,000 MIMIC-CXR-JPG images (`CXR_POOL_SIZE=10000`), which takes a long time. It then selects 2,500 of them at patient level with a fixed seed (42): 1,250 normal and 1,250 pathology images, split by patient into 1,500 train, 500 validation and 500 test images, each split balanced between the two classes. All later steps use only this 2,500-image subset.

The subset keeps run time manageable: generation for every synthetic family, the 320 detector runs and the size-ladder experiments all scale with the number of source images. It is a controlled source pool for comparing generators and edit sizes, not a representative clinical sample.

To reproduce the report's images exactly, download the full 10,000-image pool first; downloading only 2,500 images directly gives a different selection. For a quick test, use a small value such as `CXR_POOL_SIZE=500`.


## Execution

Run the notebooks in numerical order, cells from top to bottom. Seeds are fixed in `config.py` (`RANDOM_SEED = 42`). The notebooks were run on a university GPU server, so a few cells are specific to it:

- **GPU selection and one-off installs.** Cell 4 of each notebook pins `CUDA_VISIBLE_DEVICES`, and cell 7 runs a one-off `pip install` (`huggingface-hub`, `open_clip_torch`). Change the GPU index to match your machine, or skip these cells.

| # | Notebook | Code used | Result |
|---|---|---|---|
| 1 | `1_Dataset Download and Validation` | `download/` | 10,000-image pool, then 2,500 real images split at patient level |
| 2 | `2_Synthetic Image Generation` | `generation/` | Classical edits, CycleGAN, RadEdit, EditOnly, Reconstruction |
| 3 | `3_Data Preprocessing` | `construction/` | Master manifest; prepared 256×256 images |
| 4 | `4_Frequency Characterisation` | `forensics/` | 42 forensic features and statistics |
| 5 | `5_Cross Generator Transfer` | `experiments/`, `models/` | 20 source→target experiments, ten detectors |
| 6 | `6_Size Ladder Experiment` | `generation/`, `construction/`, `forensics/`, `experiments/` | Edit-size ladder (0.5–10%) |
| 7 | `7_Sensitivity and Robustness Checks` | `evaluation/` | Provenance, JPEG, blur and compression checks; figures |

Notes:

- Notebook 2 clones the NIAID lung-segmentation repository and checks its weights by size and SHA-256 hash.
- Notebook 5 first prints a plan (`dry_run=True`) and skips finished runs. The full grid is 320 runs; DenseNet121, SupCon and NPR use three seeds.
- Settings for the size ladder are in `size_config.py`.

## Repository structure

```text
config.py, size_config.py   all paths, seeds and settings
download/                   PhysioNet download, subsampling, validation
generation/                 synthetic image generation (classical, CycleGAN, RadEdit, size ladder)
construction/               master manifest, integrity checks, preprocessing
forensics/                  the 42 forensic features, statistics, feature footprint
models/                     ten detectors (logistic regression, DenseNet121, NPR, SupCon, frozen encoders)
experiments/                experiment design and runner; size-ladder scoring
evaluation/                 metrics, robustness and sensitivity checks
utils/                      shared helpers; make_edit_figures_v2.py (report figures)
```

## Reproducibility

Seeds are fixed (42; 42–44 for the three seeded detectors). The real dataset is rebuilt from MIMIC-CXR-JPG 2.1.0 with the same selection seed. Experiments in the report were run on a single NVIDIA RTX 6000 Ada GPU (48 GB).

## Citation

A. Angaraj, *Edit Size and Cross-Family Transfer in Synthetic Chest X-Ray Detection*, MSc report, 2026.