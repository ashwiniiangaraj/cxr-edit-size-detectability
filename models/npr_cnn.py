"""
NPR forensic CNN (Tan et al., CVPR 2024).
Trained on the residual x - up(down(x)) (2x nearest-neighbour down, then up) with a small ResNet
(ResNet-50 stem, layer1, layer2; one logit). Protocol as in cnn_densenet.py: 256x256 grayscale input,
BCE with pos_weight = n_real / n_synthetic, early stopping on source-validation AUROC (unseen family
never used), repeated over config.EXPERIMENT_SEEDS, `degrade` applied to the float image.
Settings: config.NPR_* (provisional). Caveat: the Lanczos resize and JPEG q95 round trip may weaken the
pixel-level traces NPR relies on, so absolute AUROC may be lower than in the original paper.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models.resnet import Bottleneck

import config
from models.base import BaseDetector
from models.torch_common import ArrayDataset, load_arrays
from utils.common import get_device, seed_everything


def npr_residual(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, W) with even H, W  ->  x minus its nearest-neighbour down/up-sampled copy."""
    down = F.interpolate(x, scale_factor=0.5, mode="nearest")
    return x - F.interpolate(down, scale_factor=2.0, mode="nearest")


def _layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    down = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        down = nn.Sequential(nn.Conv2d(inplanes, planes * Bottleneck.expansion, 1, stride, bias=False),
                             nn.BatchNorm2d(planes * Bottleneck.expansion))
    layers = [Bottleneck(inplanes, planes, stride=stride, downsample=down)]
    layers += [Bottleneck(planes * Bottleneck.expansion, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class NPRNet(nn.Module):
    """ResNet-50 stem + layer1 (3 blocks) + layer2 (4 blocks) on the NPR residual -> 1 logit."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _layer(64, 64, 3, 1)          # -> 256 channels
        self.layer2 = _layer(256, 128, 4, 2)        # -> 512 channels
        self.fc = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):                           # x: (B, 3, H, W), grayscale replicated + normalised
        r = npr_residual(x[:, :1]) * (2.0 / 3.0)    # one channel is enough; 2/3 as in the reference code
        r = self.maxpool(self.relu(self.bn1(self.conv1(r))))
        r = self.layer2(self.layer1(r))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(r, 1), 1))


@torch.no_grad()
def _predict(model, arrays, device, batch_size=64) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(ArrayDataset(arrays), batch_size=batch_size, shuffle=False):
        out.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu().numpy())
    return np.concatenate(out)


class NPRDetector(BaseDetector):
    name = "npr"
    stochastic = True

    def fit(self, train_df, val_df):
        seed_everything(self.seed)
        device = get_device()
        self.model = NPRNet().to(device)

        tr_arr, va_arr = load_arrays(train_df), load_arrays(val_df)
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        gen = torch.Generator().manual_seed(self.seed)
        # horizontal flip keeps the 2x2 pixel grid aligned (width 256 is even), so it is safe for NPR
        loader = DataLoader(ArrayDataset(tr_arr, ytr, hflip=config.NPR_AUG_HFLIP),
                            batch_size=config.NPR_BATCH_SIZE, shuffle=True,
                            drop_last=len(tr_arr) > config.NPR_BATCH_SIZE,
                            num_workers=config.NPR_NUM_WORKERS, generator=gen)

        pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=config.NPR_LEARNING_RATE,
                                weight_decay=config.NPR_WEIGHT_DECAY)
        use_amp = config.NPR_USE_AMP and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
        self.history, t0 = [], time.time()
        for epoch in range(1, config.NPR_EPOCHS + 1):
            self.model.train()
            running = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = loss_fn(self.model(x).squeeze(1), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item() * len(x)

            val_auc = roc_auc_score(yv, _predict(self.model, va_arr, device))
            self.history.append({"epoch": epoch, "train_loss": running / len(tr_arr), "val_auroc": val_auc})
            if val_auc > best_auc + 1e-4:
                best_auc, best_epoch, bad = val_auc, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= config.NPR_EARLY_STOP_PATIENCE:
                    break

        if best_state is None:
            best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        self.best_epoch, self.val_auroc, self.train_seconds = best_epoch, best_auc, time.time() - t0
        self.device = device
        return self

    def predict_scores(self, df, degrade=None):
        device = getattr(self, "device", None) or get_device()
        self.model.to(device)
        return _predict(self.model, load_arrays(df, degrade), device)

    def info(self):
        return {"best_epoch": self.best_epoch, "source_val_auroc": round(self.val_auroc, 4),
                "train_seconds": round(self.train_seconds, 1)}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        self.write_json(directory / "info.json", self.info())

    @classmethod
    def load(cls, directory, seed=42):
        obj = cls(seed)
        obj.model = NPRNet()
        obj.model.load_state_dict(torch.load(Path(directory) / "model.pt", map_location="cpu"))
        obj.model.eval()
        obj.device = get_device()
        obj.model.to(obj.device)
        return obj