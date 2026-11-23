"""
IMAGE DEGRADATIONS for the robustness test
==========================================

    JPEG(q)       re-compression at quality q          (90, 70, 50)
    Resize(s)     down-scale by s then up-scale back   (0.75, 0.5)
    Noise(sigma)  additive Gaussian noise              (intensity units in [0,1])
    Blur(sigma)   Gaussian blur                        (optional)

Noise is seeded from the image content (CRC32) so the same image always gets the same
noise regardless of evaluation order or parallelism.
"""
from __future__ import annotations

import io
import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import config
from utils.common import float01_to_uint8, to_float01


class Degradation:
    family: str = "clean"
    level: float = 0.0

    @property
    def name(self) -> str:
        return "clean" if self.family == "clean" else f"{self.family}_{self.level:g}"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img


@dataclass
class JPEG(Degradation):
    level: float = 70
    family: str = "jpeg"

    def __call__(self, img):
        buf = io.BytesIO()
        Image.fromarray(float01_to_uint8(img), mode="L").save(buf, format="JPEG", quality=int(self.level))
        buf.seek(0)
        return to_float01(np.asarray(Image.open(buf).convert("L")))


@dataclass
class Resize(Degradation):
    level: float = 0.5
    family: str = "resize"

    def __call__(self, img):
        h, w = img.shape
        small = cv2.resize(img, (max(2, int(round(w * self.level))), max(2, int(round(h * self.level)))),
                           interpolation=cv2.INTER_AREA)
        return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)


@dataclass
class Noise(Degradation):
    level: float = 0.02
    family: str = "noise"

    def __call__(self, img):
        rng = np.random.default_rng(config.ROBUSTNESS_NOISE_SEED + zlib.crc32(img.tobytes()))
        return np.clip(img + rng.normal(0, self.level, img.shape), 0, 1).astype(np.float32)


@dataclass
class Blur(Degradation):
    level: float = 1.0
    family: str = "blur"

    def __call__(self, img):
        return cv2.GaussianBlur(img, (0, 0), self.level)


def build_degradations(include_clean: bool = True) -> list[Degradation]:
    """The conditions defined in config.ROBUSTNESS_*."""
    out = [Degradation()] if include_clean else []
    out += [JPEG(q) for q in config.ROBUSTNESS_JPEG_QUALITIES]
    out += [Resize(s) for s in config.ROBUSTNESS_RESIZE_SCALES]
    out += [Noise(s) for s in config.ROBUSTNESS_NOISE_SIGMAS]
    out += [Blur(s) for s in config.ROBUSTNESS_BLUR_SIGMAS]
    return out
