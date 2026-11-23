"""
Common detector interface 
fit(train_df, val_df): train on source-family data; validation uses source data only.
predict_scores(df, degrade): P(synthetic) per row; `degrade` optionally perturbs images.
save(dir) / load(dir), info(): persistence and a summary of what was learned.
Rows need image_id, label, prepared_path and the 42 forensic feature columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class BaseDetector:
    name: str = "base"
    stochastic: bool = False        # True -> the runner repeats it over several seeds

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "BaseDetector":
        raise NotImplementedError

    def predict_scores(self, df: pd.DataFrame, degrade=None) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path, seed: int = 42) -> "BaseDetector":
        raise NotImplementedError

    def info(self) -> dict:
        return {}

    @staticmethod
    def write_json(path: Path, obj: dict) -> None:
        path.write_text(json.dumps(obj, indent=2, default=str))
