"""
frozen foundation-model baseline (linear probe).
Frozen encoder (CLIP ViT-L/14 as in Ojha et al. 2023, BiomedCLIP, DINOv2 or RAD-DINO); L2-normalised
embeddings are cached in config.DET_EMBEDDING_DIR. Only a logistic regression is trained (C tuned on
source validation). Degraded images are embedded on the fly, uncached. Variants: config.CLIP_VARIANTS.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from models.base import BaseDetector
from utils.common import ensure_parent, float01_to_uint8, get_device, load_gray_float01

_EXTRACTORS: dict = {}
_CACHE_VERSION = "crop-v2"       # bump when the preprocessing below changes, so old caches are ignored


def _centre_crop(im: Image.Image, size: int) -> Image.Image:
    """Centre crop to size x size WITHOUT resampling (a resize would blur the fine detail a
    forensic detector needs). The prepared images are 256 x 256, so a 224 crop drops 16 px per side."""
    w, h = im.size
    if min(w, h) < size:
        raise ValueError(f"image {w}x{h} is smaller than the {size}-px model input")
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size))


class _HFClipExtractor:
    SIZE = 224

    def __init__(self, model_id):
        import torch
        from transformers import CLIPImageProcessor, CLIPModel
        self.torch, self.device = torch, get_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(model_id)

    def embed(self, pil_images):
        t = self.torch
        with t.no_grad():
            crops = [_centre_crop(im, self.SIZE) for im in pil_images]
            px = self.proc(images=crops, return_tensors="pt", do_resize=False,
                           do_center_crop=False)["pixel_values"].to(self.device)
            vis = self.model.vision_model(pixel_values=px)
            f = self.model.visual_projection(vis.pooler_output)
            return t.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


class _OpenClipExtractor:
    """open_clip models (BiomedCLIP). The model's own preprocessing would resize 256 -> 224 with
    bicubic interpolation; we centre-crop to the model's input size first, so its resize step
    becomes a no-op and the pixels are not resampled (same rule as the HF CLIP path)."""

    def __init__(self, model_id):
        import open_clip
        import torch
        self.torch, self.device = torch, get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.model = self.model.to(self.device).eval()
        size = getattr(self.model.visual, "image_size", 224)
        self.size = int(size[0] if isinstance(size, (tuple, list)) else size)

    def embed(self, pil_images):
        t = self.torch
        with t.no_grad():
            batch = t.stack([self.preprocess(_centre_crop(im, self.size)) for im in pil_images])
            f = self.model.encode_image(batch.to(self.device))
            return t.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


class _HFDinoExtractor:
    """DINOv2-family encoders (RAD-DINO, DINOv2): L2-normalised CLS embedding.

    Their image processors resize to a large side (518 for RAD-DINO); up-sampling a 256-px
    image would blur away exactly the high-frequency evidence a forensic detector needs. We
    centre-crop to `image_size` instead and switch the processor's resize and crop off. The
    patch size is 14, so the input should be a multiple of 14 (252 = 18 x 14, default);
    a 256-px input would silently drop its last 4 right / bottom pixels."""

    def __init__(self, model_id, image_size=252):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch, self.device, self.size = torch, get_device(), int(image_size)
        if self.size % 14:
            raise ValueError(f"image_size={self.size} is not a multiple of the patch size 14")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = AutoImageProcessor.from_pretrained(model_id)

    def embed(self, pil_images):
        t = self.torch
        with t.no_grad():
            crops = [_centre_crop(im, self.size) for im in pil_images]
            px = self.proc(images=crops, return_tensors="pt", do_resize=False,
                           do_center_crop=False)["pixel_values"].to(self.device)
            out = self.model(pixel_values=px)
            f = getattr(out, "pooler_output", None)              # CLS token after the final layer norm
            if f is None:
                f = out.last_hidden_state[:, 0]
            return t.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


def _get_extractor(variant: str):
    if variant not in _EXTRACTORS:
        spec = config.CLIP_VARIANTS[variant]
        backend = spec["backend"]
        if backend == "hf_dino":
            _EXTRACTORS[variant] = _HFDinoExtractor(spec["model_id"], spec.get("image_size", 252))
        else:
            _EXTRACTORS[variant] = {"hf": _HFClipExtractor, "open_clip": _OpenClipExtractor}[backend](spec["model_id"])
    return _EXTRACTORS[variant]


def _embed_arrays(variant: str, arrays) -> np.ndarray:
    ex, out, bs = _get_extractor(variant), [], config.CLIP_BATCH_SIZE
    for i in range(0, len(arrays), bs):
        pil = [Image.fromarray(float01_to_uint8(a), mode="L").convert("RGB") for a in arrays[i:i + bs]]
        out.append(ex.embed(pil))
    return np.concatenate(out)


# ------------------------------------------------------------------ embedding cache
def _cache_signature(variant: str) -> str:
    spec = config.CLIP_VARIANTS[variant]
    return f"{_CACHE_VERSION}|{spec['backend']}|{spec['model_id']}|{spec.get('image_size', '')}"


def _read_cache(variant: str) -> dict:
    """image_id -> (embedding, size of the prepared file when it was embedded). A cache made with
    other settings, or an unreadable one, is ignored."""
    path = Path(config.DET_EMBEDDING_DIR) / f"{variant}.npz"
    if not path.exists():
        return {}
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["signature"]) != _cache_signature(variant):
            print(f"    [{variant}] embedding cache was made with other settings - ignored")
            return {}
        return {i: (e, int(n)) for i, e, n in zip(z["image_id"], z["emb"], z["nbytes"])}
    except Exception as exc:                                       # noqa: BLE001
        print(f"    [{variant}] embedding cache unreadable ({exc!r}) - ignored")
        return {}


def _write_cache(variant: str, cache: dict) -> None:
    path = ensure_parent(Path(config.DET_EMBEDDING_DIR) / f"{variant}.npz")
    ids = np.array(list(cache.keys()))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, image_id=ids, emb=np.stack([cache[i][0] for i in ids]),
                        nbytes=np.array([cache[i][1] for i in ids]), signature=_cache_signature(variant))
    os.replace(tmp, path)                                          # a crash never leaves a half-written cache


def get_embeddings(df: pd.DataFrame, variant: str, degrade=None) -> np.ndarray:
    """Embeddings of df's prepared images. Clean images are cached; an entry is reused only if the
    prepared file still has the size it had when embedded (so re-preparing the images cannot
    silently serve old embeddings). Degraded images (robustness stage) are never cached."""
    if degrade is not None:
        return _embed_arrays(variant, [degrade(load_gray_float01(p)) for p in df["prepared_path"]])
    cache = _read_cache(variant)
    sizes = {i: os.path.getsize(p) for i, p in zip(df["image_id"], df["prepared_path"])}
    missing = df[[(i not in cache) or (cache[i][1] != sizes[i]) for i in df["image_id"]]]
    if len(missing):
        print(f"    [{variant}] embedding {len(missing)} new / changed images ...")
        emb = _embed_arrays(variant, [load_gray_float01(p) for p in missing["prepared_path"]])
        for i, e in zip(missing["image_id"], emb):
            cache[i] = (e, sizes[i])
        _write_cache(variant, cache)
    return np.stack([cache[i][0] for i in df["image_id"]])


class FrozenEmbeddingLR(BaseDetector):
    variant: str = "clip_frozen"
    name: str = "clip_frozen"
    stochastic = False

    def fit(self, train_df, val_df):
        Etr, Eva = get_embeddings(train_df, self.variant), get_embeddings(val_df, self.variant)
        ytr, yv = train_df["label"].to_numpy(), val_df["label"].to_numpy()
        best = (-1.0, None, None)
        for C in config.CLIP_C_GRID:
            clf = LogisticRegression(C=C, class_weight=config.LR_CLASS_WEIGHT, max_iter=config.LR_MAX_ITER).fit(Etr, ytr)
            auc = roc_auc_score(yv, clf.decision_function(Eva))
            if auc > best[0] + 1e-9:
                best = (auc, clf, C)
        self.val_auroc, self.clf, self.best_C = best
        return self

    def predict_scores(self, df, degrade=None):
        return self.clf.predict_proba(get_embeddings(df, self.variant, degrade))[:, 1]

    def info(self):
        return {"variant": self.variant, "C": self.best_C, "source_val_auroc": round(self.val_auroc, 4),
                "backbone": config.CLIP_VARIANTS[self.variant]["model_id"]}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "C": self.best_C, "val_auroc": self.val_auroc}, directory / "model.joblib")

    @classmethod
    def load(cls, directory, seed=42):
        d = joblib.load(Path(directory) / "model.joblib")
        obj = cls(seed)
        obj.clf, obj.best_C, obj.val_auroc = d["clf"], d["C"], d["val_auroc"]
        return obj


def make_frozen_detector(variant: str):
    """Subclass of FrozenEmbeddingLR bound to one entry of config.CLIP_VARIANTS."""
    return type(f"Frozen_{variant}", (FrozenEmbeddingLR,), {"variant": variant, "name": variant})