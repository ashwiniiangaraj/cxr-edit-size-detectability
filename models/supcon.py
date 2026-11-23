"""
supervised contrastive learning (SupCon; Khosla et al., 2020).
A. Representation: DenseNet121 encoder + 2-layer projection head trained with the SupCon loss on two
   augmented views per image (flip, small shift/rotation, brightness/contrast jitter). Binary labels only;
   generator identity is not used. Every SUPCON_EVAL_EVERY epochs a linear probe (fitted on source
   training embeddings, scored on source validation) selects the best encoder.
B. Classifier: logistic regression on frozen, L2-normalised 1024-d embeddings; C tuned on source validation.
The unseen family is never used.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config
from models.base import BaseDetector
from models.torch_common import ArrayDataset, load_arrays, make_densenet121, normalise, to_float01
from utils.common import get_device, seed_everything


class SupConNet(nn.Module):
    """DenseNet121 feature extractor (1024-d) + projection head used by the loss only."""

    def __init__(self, pretrained: bool, proj_dim: int):
        super().__init__()
        base = make_densenet121(pretrained)
        self.features = base.features
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def encode(self, x):                                    # (B, 1024)
        return F.adaptive_avg_pool2d(F.relu(self.features(x)), 1).flatten(1)

    def forward(self, x):                                   # (B, proj_dim), unit norm
        return F.normalize(self.head(self.encode(x)), dim=1)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    SupCon loss.  features: (B, V, D) L2-normalised; labels: (B,).
    For each anchor, positives = all other views of all same-label samples.
    """
    B, V, _ = features.shape
    z = features.reshape(B * V, -1)                         # order: b0v0, b0v1, b1v0, ...
    y = labels.repeat_interleave(V)
    logits = z @ z.T / temperature
    self_mask = torch.eye(B * V, device=z.device, dtype=torch.bool)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    mean_log_prob_pos = (log_prob * pos).sum(1) / n_pos.clamp(min=1)
    return -mean_log_prob_pos[n_pos > 0].mean()


class TwoViewDataset(Dataset):
    """Returns two independently augmented views of each image."""

    def __init__(self, arrays, labels):
        self.arrays, self.labels = arrays, labels
        self.aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=config.SUPCON_AUG_ROTATE_DEG, translate=(config.SUPCON_AUG_TRANSLATE,) * 2),
            T.ColorJitter(brightness=config.SUPCON_AUG_BRIGHTNESS, contrast=config.SUPCON_AUG_CONTRAST),
        ])

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        x = to_float01(self.arrays[i]).unsqueeze(0)
        return normalise(self.aug(x).clamp(0, 1)), normalise(self.aug(x).clamp(0, 1)), int(self.labels[i])


@torch.no_grad()
def embed(model: SupConNet, arrays, device, batch_size=64) -> np.ndarray:
    """L2-normalised encoder embeddings (no augmentation)."""
    model.eval()
    out = []
    for x, _ in DataLoader(ArrayDataset(arrays), batch_size=batch_size, shuffle=False):
        out.append(F.normalize(model.encode(x.to(device)), dim=1).float().cpu().numpy())
    return np.concatenate(out)


def _fit_probe(Etr, ytr, Eva, yva, c_grid):
    """Logistic regression on frozen embeddings; C tuned by validation AUROC."""
    best = (-1.0, None, None)
    for C in c_grid:
        clf = LogisticRegression(C=C, class_weight=config.LR_CLASS_WEIGHT, max_iter=config.LR_MAX_ITER).fit(Etr, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Eva))
        if auc > best[0] + 1e-9:
            best = (auc, clf, C)
    return best


class SupConDetector(BaseDetector):
    name = "supcon"
    stochastic = True

    def fit(self, train_df, val_df):
        seed_everything(self.seed)
        device = get_device()
        self.net = SupConNet(config.CNN_PRETRAINED, config.SUPCON_PROJECTION_DIM).to(device)

        tr_arr, va_arr = load_arrays(train_df), load_arrays(val_df)
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TwoViewDataset(tr_arr, ytr), batch_size=config.SUPCON_BATCH_SIZE, shuffle=True,
                            drop_last=len(tr_arr) > config.SUPCON_BATCH_SIZE,
                            num_workers=config.CNN_NUM_WORKERS, generator=gen)
        opt = torch.optim.AdamW(self.net.parameters(), lr=config.SUPCON_LEARNING_RATE,
                                weight_decay=config.SUPCON_WEIGHT_DECAY)

        best = {"auc": -1.0, "state": None, "epoch": 0}
        self.history, t0 = [], time.time()
        for epoch in range(1, config.SUPCON_EPOCHS + 1):
            self.net.train()
            running, n = 0.0, 0
            for v1, v2, y in loader:
                v1, v2, y = v1.to(device), v2.to(device), y.to(device)
                z = self.net(torch.cat([v1, v2]))
                loss = supcon_loss(torch.stack([z[:len(y)], z[len(y):]], dim=1), y, config.SUPCON_TEMPERATURE)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.item() * len(y); n += len(y)

            row = {"epoch": epoch, "supcon_loss": running / max(n, 1)}
            if epoch % config.SUPCON_EVAL_EVERY == 0 or epoch == config.SUPCON_EPOCHS:
                auc, _, _ = _fit_probe(embed(self.net, tr_arr, device), ytr, embed(self.net, va_arr, device), yv, [1.0])
                row["probe_val_auroc"] = auc
                if auc > best["auc"] + 1e-4:
                    best = {"auc": auc, "epoch": epoch, "state": copy.deepcopy(self.net.state_dict())}
            self.history.append(row)

        if best["state"] is None:
            best["state"] = copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best["state"])
        Etr, Eva = embed(self.net, tr_arr, device), embed(self.net, va_arr, device)
        self.val_auroc, self.clf, self.best_C = _fit_probe(Etr, ytr, Eva, yv, config.SUPCON_PROBE_C_GRID)
        self.best_epoch, self.train_seconds, self.device = best["epoch"], time.time() - t0, device
        return self

    def predict_scores(self, df, degrade=None):
        device = getattr(self, "device", None) or get_device()
        self.net.to(device)
        return self.clf.predict_proba(embed(self.net, load_arrays(df, degrade), device))[:, 1]

    def info(self):
        return {"best_epoch": self.best_epoch, "C": self.best_C, "source_val_auroc": round(self.val_auroc, 4),
                "train_seconds": round(self.train_seconds, 1)}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "encoder.pt")
        joblib.dump(self.clf, directory / "classifier.joblib")
        self.write_json(directory / "info.json", self.info())

    @classmethod
    def load(cls, directory, seed=42):
        directory = Path(directory)
        obj = cls(seed)
        obj.net = SupConNet(pretrained=False, proj_dim=config.SUPCON_PROJECTION_DIM)
        obj.net.load_state_dict(torch.load(directory / "encoder.pt", map_location="cpu"))
        obj.net.eval()
        obj.clf = joblib.load(directory / "classifier.joblib")
        obj.device = get_device()
        obj.net.to(obj.device)
        return obj
