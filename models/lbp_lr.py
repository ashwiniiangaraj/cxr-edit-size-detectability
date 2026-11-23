"""
Stage 5.6: LBP-only logistic regression.
Same model as spectral_lr (StandardScaler -> logistic regression, C tuned on source validation) on the
10 LBP histogram bins (config.LBP_MODEL_FEATURES). Fit, scoring, save/load and coefficients are inherited
from _FeatureLR.
"""
from __future__ import annotations

import config
from models.feature_lr import _FeatureLR


class LBPLR(_FeatureLR):
    """5.6 - the ten LBP bins only."""
    name = "lbp_lr"

    def _candidate_sets(self, train_df):
        return [list(config.LBP_MODEL_FEATURES)]