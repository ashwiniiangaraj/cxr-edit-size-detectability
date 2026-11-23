"""
Modular CycleGAN training.

"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Make 'import gen_common' work under %run from the notebook.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_common   # shared loader (Lanczos) used by every generator

import config


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Models — matches the direct-PyTorch HTML implementation
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        n_residual_blocks: int = 6,
        base_channels: int = 64,
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=7,
            ),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        in_channels = base_channels

        for _ in range(2):
            out_channels = in_channels * 2

            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

            in_channels = out_channels

        for _ in range(n_residual_blocks):
            layers.append(
                ResidualBlock(in_channels)
            )

        for _ in range(2):
            out_channels = in_channels // 2

            layers.extend([
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

            in_channels = out_channels

        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                base_channels,
                output_channels,
                kernel_size=7,
            ),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator corresponding to the direct
    CycleGAN implementation in the supplied HTML.
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(
                config.CYCLEGAN_DISCRIMINATOR_LEAKY_RELU,
                inplace=True,
            ),

            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(
                config.CYCLEGAN_DISCRIMINATOR_LEAKY_RELU,
                inplace=True,
            ),

            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(
                config.CYCLEGAN_DISCRIMINATOR_LEAKY_RELU,
                inplace=True,
            ),

            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(
                config.CYCLEGAN_DISCRIMINATOR_LEAKY_RELU,
                inplace=True,
            ),

            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Dataset
# ============================================================

def _source_index() -> dict:
    """
    FREEZE FIX (speed): build ONE lookup {dicom_id: [paths]} of the source
    folder the first time it is needed. The old code called rglob over the whole
    folder for every image, which is O(N^2) and very slow at 10-20k images.
    File names are p<subject>_s<study>_<dicom>.jpg and a DICOM id contains no
    underscore, so the id is the text after the last underscore.
    """
    global _SRC_INDEX
    if _SRC_INDEX is None:
        _SRC_INDEX = {}
        for p in Path(config.CYCLEGAN_SOURCE_ROOT).rglob("*.jpg"):
            _SRC_INDEX.setdefault(p.stem.rsplit("_", 1)[-1], []).append(p)
    return _SRC_INDEX


_SRC_INDEX = None


def resolve_source_image(row: pd.Series) -> Path:
    dicom_id = str(row["dicom_id"])

    matches = list(_source_index().get(dicom_id, []))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple images found for dicom_id={dicom_id}: "
            f"{matches}"
        )

    # Fallback: exact filename convention.
    subject_id = str(row["subject_id"])
    study_id = str(row["study_id"])

    candidate = (
        config.CYCLEGAN_SOURCE_ROOT
        / config.CYCLEGAN_TRAIN_SPLIT
        / str(row["disease_status"])
        / f"p{subject_id}_s{study_id}_{dicom_id}.jpg"
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve image for dicom_id={dicom_id}"
    )


def image_to_tensor(path: Path) -> torch.Tensor:
    """
    path -> tensor of shape (3, H, W), values in [-1, 1].

    Steps: open -> grayscale -> resize with the FROZEN filter
    (Lanczos, via gen_common) -> scale to [-1, 1] -> copy the single
    grey channel into 3 channels (the network is 3-channel).
    """
    image = gen_common.load_gray_image(
        path,
        config.CYCLEGAN_IMAGE_SIZE,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(array)[None, :, :]
    tensor = tensor.repeat(3, 1, 1)

    tensor = tensor * 2.0 - 1.0
    return tensor


class UnpairedChestDataset(Dataset):
    """
    Unpaired CycleGAN dataset.

    A = Normal
    B = Pathology

    The dataset length is max(len(A), len(B)), matching the
    unpaired training concept used in the original notebook.
    """

    def __init__(
        self,
        normal_paths: List[Path],
        pathology_paths: List[Path],
        seed: int,
    ):
        self.normal_paths = normal_paths
        self.pathology_paths = pathology_paths
        self.rng = random.Random(seed)

        self.length = max(
            len(normal_paths),
            len(pathology_paths),
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        a_path = self.normal_paths[
            index % len(self.normal_paths)
        ]

        b_index = self.rng.randrange(
            len(self.pathology_paths)
        )
        b_path = self.pathology_paths[b_index]

        return {
            "A": image_to_tensor(a_path),
            "B": image_to_tensor(b_path),
        }


# ============================================================
# Data preparation
# ============================================================

def prepare_training_data() -> tuple[List[Path], List[Path]]:
    manifest_path = Path(config.CYCLEGAN_MANIFEST)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    required = {
        "dicom_id",
        "subject_id",
        "study_id",
        "disease_status",
        "split",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    train_df = df[
        df["split"].astype(str).str.lower()
        == config.CYCLEGAN_TRAIN_SPLIT.lower()
    ].copy()

    normal_df = train_df[
        train_df["disease_status"].astype(str).str.lower()
        == config.CYCLEGAN_DOMAIN_A
    ]

    pathology_df = train_df[
        train_df["disease_status"].astype(str).str.lower()
        == config.CYCLEGAN_DOMAIN_B
    ]

    total = len(train_df)

    print("\nCycleGAN dataset validation")
    print(f"  Manifest: {manifest_path}")
    print(f"  Training images: {total}")
    print(
        f"  Domain A: {config.CYCLEGAN_DOMAIN_A} "
        f"({len(normal_df)})"
    )
    print(
        f"  Domain B: {config.CYCLEGAN_DOMAIN_B} "
        f"({len(pathology_df)})"
    )

    if total != config.CYCLEGAN_EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            "Unexpected training-image count: "
            f"{total}; expected "
            f"{config.CYCLEGAN_EXPECTED_TRAIN_IMAGES}"
        )

    if len(normal_df) == 0 or len(pathology_df) == 0:
        raise ValueError(
            "Both CycleGAN domains must contain images."
        )

    if (
        config.CYCLEGAN_REQUIRE_TRAIN_BALANCE
        and len(normal_df) != len(pathology_df)
    ):
        raise ValueError(
            "Training domains are not balanced."
        )

    normal_paths = [
        resolve_source_image(row)
        for _, row in normal_df.iterrows()
    ]

    pathology_paths = [
        resolve_source_image(row)
        for _, row in pathology_df.iterrows()
    ]

    return normal_paths, pathology_paths


# ============================================================
# Initialisation
# ============================================================

def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
            ),
        ):
            nn.init.normal_(
                module.weight,
                mean=config.CYCLEGAN_INIT_MEAN,
                std=config.CYCLEGAN_INIT_STD,
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0,
                )

        elif isinstance(
            module,
            nn.InstanceNorm2d,
        ):
            if module.weight is not None:
                nn.init.normal_(
                    module.weight,
                    mean=config.CYCLEGAN_INIT_MEAN,
                    std=config.CYCLEGAN_INIT_STD,
                )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0,
                )


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if (
        config.CYCLEGAN_DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device(
            f"cuda:{config.CYCLEGAN_GPU_INDEX}"
        )

    if config.CYCLEGAN_DEVICE == "cuda":
        raise RuntimeError(
            "CYCLEGAN_DEVICE='cuda' but CUDA is unavailable."
        )

    return torch.device("cpu")


# ============================================================
# Configuration snapshot
# ============================================================

def save_config_snapshot(
    normal_count: int,
    pathology_count: int,
    device: torch.device,
) -> None:

    config.CYCLEGAN_CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = {
        "implementation": config.CYCLEGAN_IMPLEMENTATION,
        "pretrained": config.CYCLEGAN_PRETRAINED,
        "seed": config.CYCLEGAN_MASTER_SEED,
        "device": str(device),
        "manifest": str(config.CYCLEGAN_MANIFEST),
        "source_root": str(config.CYCLEGAN_SOURCE_ROOT),
        "train_split": config.CYCLEGAN_TRAIN_SPLIT,
        "domain_A": config.CYCLEGAN_DOMAIN_A,
        "domain_B": config.CYCLEGAN_DOMAIN_B,
        "normal_count": normal_count,
        "pathology_count": pathology_count,
        "image_size": config.CYCLEGAN_IMAGE_SIZE,
        "resize_filter": gen_common.RESIZE_FILTER_NAME,
        "batch_size": config.CYCLEGAN_BATCH_SIZE,
        "epochs": config.CYCLEGAN_EPOCHS,
        "learning_rate": config.CYCLEGAN_LEARNING_RATE,
        "beta1": config.CYCLEGAN_BETA1,
        "beta2": config.CYCLEGAN_BETA2,
        "lambda_cycle": config.CYCLEGAN_LAMBDA_CYCLE,
        "lambda_identity": config.CYCLEGAN_LAMBDA_IDENTITY,
        "generator_residual_blocks":
            config.CYCLEGAN_GENERATOR_RES_BLOCKS,
        "generator_base_channels":
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS,
        "discriminator_base_channels":
            config.CYCLEGAN_DISCRIMINATOR_BASE_CHANNELS,
        "repo_url": config.CYCLEGAN_REPO_URL,
        "repo_commit": config.CYCLEGAN_REPO_COMMIT,
    }

    path = (
        config.CYCLEGAN_CONFIG_DIR
        / f"{config.CYCLEGAN_EXPERIMENT_NAME}_config.json"
    )

    path.write_text(
        json.dumps(
            snapshot,
            indent=2,
        )
    )

    print(f"\nConfiguration snapshot: {path}")


# ============================================================
# Timing helper
# ============================================================

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    epoch: int,
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    optimizer_G,
    optimizer_D_A,
    optimizer_D_B,
    history: List[Dict],
) -> Path:

    config.CYCLEGAN_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D_A": optimizer_D_A.state_dict(),
        "optimizer_D_B": optimizer_D_B.state_dict(),
        "seed": config.CYCLEGAN_MASTER_SEED,
        "image_size": config.CYCLEGAN_IMAGE_SIZE,
        "batch_size": config.CYCLEGAN_BATCH_SIZE,
        "learning_rate": config.CYCLEGAN_LEARNING_RATE,
        "beta1": config.CYCLEGAN_BETA1,
        "beta2": config.CYCLEGAN_BETA2,
        "lambda_cycle": config.CYCLEGAN_LAMBDA_CYCLE,
        "lambda_identity":
            config.CYCLEGAN_LAMBDA_IDENTITY,
        "generator_residual_blocks":
            config.CYCLEGAN_GENERATOR_RES_BLOCKS,
        "generator_base_channels":
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS,
        "discriminator_base_channels":
            config.CYCLEGAN_DISCRIMINATOR_BASE_CHANNELS,
    }

    path = (
        config.CYCLEGAN_CHECKPOINT_DIR
        / f"epoch_{epoch}.pth"
    )

    torch.save(checkpoint, path)

    if epoch == config.CYCLEGAN_EPOCHS:
        torch.save(
            checkpoint,
            config.CYCLEGAN_FINAL_CHECKPOINT,
        )

    return path


# ============================================================
# Training
# ============================================================

def train() -> None:

    seed_everything(
        config.CYCLEGAN_MASTER_SEED
    )

    device = get_device()

    print("=" * 72)
    print("CYCLEGAN TRAINING FROM SCRATCH")
    print("=" * 72)
    print(f"Implementation: {config.CYCLEGAN_IMPLEMENTATION}")
    print("Pretrained weights: NO")
    print(f"Device: {device}")
    print(f"Epochs: {config.CYCLEGAN_EPOCHS}")

    normal_paths, pathology_paths = (
        prepare_training_data()
    )

    print(
        f"\nNormal training images: "
        f"{len(normal_paths)}"
    )
    print(
        f"Pathology training images: "
        f"{len(pathology_paths)}"
    )

    save_config_snapshot(
        len(normal_paths),
        len(pathology_paths),
        device,
    )

    dataset = UnpairedChestDataset(
        normal_paths,
        pathology_paths,
        seed=config.CYCLEGAN_MASTER_SEED,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.CYCLEGAN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.CYCLEGAN_NUM_WORKERS,
        pin_memory=(
            config.CYCLEGAN_PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    print(
        f"Training iterations per epoch: "
        f"{len(loader)}"
    )

    G_AB = Generator(
        input_channels=config.CYCLEGAN_INPUT_NC,
        output_channels=config.CYCLEGAN_OUTPUT_NC,
        n_residual_blocks=(
            config.CYCLEGAN_GENERATOR_RES_BLOCKS
        ),
        base_channels=(
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS
        ),
    ).to(device)

    G_BA = Generator(
        input_channels=config.CYCLEGAN_INPUT_NC,
        output_channels=config.CYCLEGAN_OUTPUT_NC,
        n_residual_blocks=(
            config.CYCLEGAN_GENERATOR_RES_BLOCKS
        ),
        base_channels=(
            config.CYCLEGAN_GENERATOR_BASE_CHANNELS
        ),
    ).to(device)

    D_A = Discriminator(
        input_channels=config.CYCLEGAN_INPUT_NC,
        base_channels=(
            config.CYCLEGAN_DISCRIMINATOR_BASE_CHANNELS
        ),
    ).to(device)

    D_B = Discriminator(
        input_channels=config.CYCLEGAN_OUTPUT_NC,
        base_channels=(
            config.CYCLEGAN_DISCRIMINATOR_BASE_CHANNELS
        ),
    ).to(device)

    for model in [G_AB, G_BA, D_A, D_B]:
        initialize_weights(model)

    adversarial_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters())
        + list(G_BA.parameters()),
        lr=config.CYCLEGAN_LEARNING_RATE,
        betas=(
            config.CYCLEGAN_BETA1,
            config.CYCLEGAN_BETA2,
        ),
    )

    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(),
        lr=config.CYCLEGAN_LEARNING_RATE,
        betas=(
            config.CYCLEGAN_BETA1,
            config.CYCLEGAN_BETA2,
        ),
    )

    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(),
        lr=config.CYCLEGAN_LEARNING_RATE,
        betas=(
            config.CYCLEGAN_BETA1,
            config.CYCLEGAN_BETA2,
        ),
    )

    config.CYCLEGAN_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: List[Dict] = []

    with open(
        config.CYCLEGAN_TRAINING_LOG,
        "w",
        encoding="utf-8",
    ) as log:

        def write_log(message: str = ""):
            print(message)
            log.write(message + "\n")
            log.flush()

        write_log("\n" + "=" * 72)
        write_log("STARTING MODULAR CYCLEGAN TRAINING")
        write_log("=" * 72)

        # Wall-clock timing (reported only; it does not affect training).
        # Each iteration copies a loss value to the CPU, which waits for the GPU,
        # so the epoch time below is the real elapsed time.
        training_start = time.perf_counter()
        epoch_seconds_history: List[float] = []

        for epoch in range(
            1,
            config.CYCLEGAN_EPOCHS + 1,
        ):

            epoch_start = time.perf_counter()

            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()

            epoch_losses = {
                "G": [],
                "D_A": [],
                "D_B": [],
                "cycle": [],
                "identity": [],
            }

            for batch in loader:

                real_A = batch["A"].to(
                    device,
                    non_blocking=True,
                )

                real_B = batch["B"].to(
                    device,
                    non_blocking=True,
                )

                # --------------------------------------------
                # Generator update
                # --------------------------------------------
                optimizer_G.zero_grad(
                    set_to_none=True
                )

                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)

                pred_fake_B = D_B(fake_B)
                pred_fake_A = D_A(fake_A)

                valid_B = torch.ones_like(
                    pred_fake_B
                )
                valid_A = torch.ones_like(
                    pred_fake_A
                )

                loss_GAN_AB = adversarial_loss(
                    pred_fake_B,
                    valid_B,
                )

                loss_GAN_BA = adversarial_loss(
                    pred_fake_A,
                    valid_A,
                )

                recovered_A = G_BA(fake_B)
                recovered_B = G_AB(fake_A)

                loss_cycle_A = cycle_loss(
                    recovered_A,
                    real_A,
                )

                loss_cycle_B = cycle_loss(
                    recovered_B,
                    real_B,
                )

                identity_A = G_BA(real_A)
                identity_B = G_AB(real_B)

                loss_identity_A = identity_loss(
                    identity_A,
                    real_A,
                )

                loss_identity_B = identity_loss(
                    identity_B,
                    real_B,
                )

                total_cycle = (
                    loss_cycle_A
                    + loss_cycle_B
                )

                total_identity = (
                    loss_identity_A
                    + loss_identity_B
                )

                loss_G = (
                    loss_GAN_AB
                    + loss_GAN_BA
                    + config.CYCLEGAN_LAMBDA_CYCLE
                    * total_cycle
                    + config.CYCLEGAN_LAMBDA_IDENTITY
                    * total_identity
                )

                loss_G.backward()
                optimizer_G.step()

                # --------------------------------------------
                # Discriminator A
                # --------------------------------------------
                optimizer_D_A.zero_grad(
                    set_to_none=True
                )

                pred_real_A = D_A(real_A)
                pred_fake_A = D_A(
                    fake_A.detach()
                )

                loss_D_A_real = adversarial_loss(
                    pred_real_A,
                    torch.ones_like(
                        pred_real_A
                    ),
                )

                loss_D_A_fake = adversarial_loss(
                    pred_fake_A,
                    torch.zeros_like(
                        pred_fake_A
                    ),
                )

                loss_D_A = 0.5 * (
                    loss_D_A_real
                    + loss_D_A_fake
                )

                loss_D_A.backward()
                optimizer_D_A.step()

                # --------------------------------------------
                # Discriminator B
                # --------------------------------------------
                optimizer_D_B.zero_grad(
                    set_to_none=True
                )

                pred_real_B = D_B(real_B)
                pred_fake_B = D_B(
                    fake_B.detach()
                )

                loss_D_B_real = adversarial_loss(
                    pred_real_B,
                    torch.ones_like(
                        pred_real_B
                    ),
                )

                loss_D_B_fake = adversarial_loss(
                    pred_fake_B,
                    torch.zeros_like(
                        pred_fake_B
                    ),
                )

                loss_D_B = 0.5 * (
                    loss_D_B_real
                    + loss_D_B_fake
                )

                loss_D_B.backward()
                optimizer_D_B.step()

                epoch_losses["G"].append(
                    float(loss_G.detach().cpu())
                )

                epoch_losses["D_A"].append(
                    float(loss_D_A.detach().cpu())
                )

                epoch_losses["D_B"].append(
                    float(loss_D_B.detach().cpu())
                )

                epoch_losses["cycle"].append(
                    float(total_cycle.detach().cpu())
                )

                epoch_losses["identity"].append(
                    float(total_identity.detach().cpu())
                )

            epoch_seconds = time.perf_counter() - epoch_start
            epoch_seconds_history.append(epoch_seconds)
            elapsed_seconds = time.perf_counter() - training_start
            mean_epoch_seconds = float(np.mean(epoch_seconds_history))
            remaining_seconds = mean_epoch_seconds * (
                config.CYCLEGAN_EPOCHS - epoch
            )

            row = {
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "G": float(np.mean(epoch_losses["G"])),
                "D_A": float(np.mean(epoch_losses["D_A"])),
                "D_B": float(np.mean(epoch_losses["D_B"])),
                "cycle": float(np.mean(epoch_losses["cycle"])),
                "identity": float(
                    np.mean(epoch_losses["identity"])
                ),
            }

            history.append(row)

            write_log(
                f"Epoch {epoch:03d}/"
                f"{config.CYCLEGAN_EPOCHS} | "
                f"G: {row['G']:.4f} | "
                f"D_A: {row['D_A']:.4f} | "
                f"D_B: {row['D_B']:.4f} | "
                f"Cycle: {row['cycle']:.4f} | "
                f"Identity: {row['identity']:.4f} | "
                f"Epoch time: {epoch_seconds:.1f} s | "
                f"Elapsed: {_format_duration(elapsed_seconds)} | "
                f"Est. remaining: {_format_duration(remaining_seconds)}"
            )

            checkpoint_path = save_checkpoint(
                epoch,
                G_AB,
                G_BA,
                D_A,
                D_B,
                optimizer_G,
                optimizer_D_A,
                optimizer_D_B,
                history,
            )

            if (
                epoch % 10 == 0
                or epoch == config.CYCLEGAN_EPOCHS
            ):
                write_log(
                    f"Checkpoint: {checkpoint_path}"
                )

    pd.DataFrame(history).to_csv(
        config.CYCLEGAN_LOSS_HISTORY,
        index=False,
    )

    print("\n" + "=" * 72)
    print("CYCLEGAN TRAINING COMPLETE")
    print("=" * 72)
    print(
        f"Final checkpoint: "
        f"{config.CYCLEGAN_FINAL_CHECKPOINT}"
    )
    print(
        f"Loss history: "
        f"{config.CYCLEGAN_LOSS_HISTORY}"
    )
    print(
        f"Training log: "
        f"{config.CYCLEGAN_TRAINING_LOG}"
    )


if __name__ == "__main__":
    train()