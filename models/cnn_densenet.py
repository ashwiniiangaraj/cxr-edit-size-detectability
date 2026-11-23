"""
CNN detector: ImageNet-pretrained DenseNet121, fine-tuned, single-logit output.
Input: 256x256 grayscale radiograph replicated to 3 channels.
Loss: BCE with pos_weight = n_real / n_synthetic. Optimiser: AdamW.
Early stopping on source-validation AUROC; the unseen family is never used.
Augmentation: horizontal flip only (resize, crop and blur would remove high-frequency evidence).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
from models.base import BaseDetector
from models.torch_common import ArrayDataset, load_arrays, make_densenet121
from utils.common import get_device, seed_everything


def _build(pretrained: bool) -> nn.Module:
    m = make_densenet121(pretrained)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    return m


@torch.no_grad()
def _predict(model, arrays, device, batch_size=64) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(ArrayDataset(arrays), batch_size=batch_size, shuffle=False):
        out.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu().numpy())
    return np.concatenate(out)


class CNNDensenet(BaseDetector):
    name = "cnn_densenet"
    stochastic = True

    def fit(self, train_df, val_df):
        seed_everything(self.seed)
        device = get_device()
        self.model = _build(config.CNN_PRETRAINED).to(device)

        tr_arr, va_arr = load_arrays(train_df), load_arrays(val_df)
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(ArrayDataset(tr_arr, ytr, hflip=config.CNN_AUG_HFLIP),
                            batch_size=config.CNN_BATCH_SIZE, shuffle=True,
                            drop_last=len(tr_arr) > config.CNN_BATCH_SIZE,
                            num_workers=config.CNN_NUM_WORKERS, generator=gen)

        pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=config.CNN_LEARNING_RATE,
                                weight_decay=config.CNN_WEIGHT_DECAY)
        use_amp = config.CNN_USE_AMP and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
        self.history, t0 = [], time.time()
        for epoch in range(1, config.CNN_EPOCHS + 1):
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
                if bad >= config.CNN_EARLY_STOP_PATIENCE:
                    break

        if best_state is None:                    # never improved (e.g. constant AUROC)
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
                "pretrained": config.CNN_PRETRAINED, "train_seconds": round(self.train_seconds, 1)}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        self.write_json(directory / "info.json", self.info())

    @classmethod
    def load(cls, directory, seed=42):
        obj = cls(seed)
        obj.model = _build(pretrained=False)
        obj.model.load_state_dict(torch.load(Path(directory) / "model.pt", map_location="cpu"))
        obj.model.eval()
        obj.device = get_device()
        obj.model.to(obj.device)
        return obj
