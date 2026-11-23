"""
the 42 forensic features (+ 6 secondary residual-spectral specres_* features, not part of the 42).
Input: 2-D float32 array in [0, 1] (prepared 256x256 grey image). Uses no labels or reference images.
Families: spectral (6), wavelet (18), LBP (10), SRM-style residuals (3), noise (5).
Parameters are marked STANDARD, ADAPTED or OURS (own choice; needs a sensitivity analysis); fixed values are in config.py.
Spectral: Hann-windowed 2-D power spectrum. Wavelet: Haar, 3 levels. LBP: riu2, P=8, R=1.
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern

import config

EPS = config.FORENSIC_EPSILON


def _as_float32_2d(img: np.ndarray) -> np.ndarray:
    """Validate the input and return a contiguous 2-D float32 copy.
    OpenCV's median / Laplacian / filter2D calls below fail on float64 input, so every public
    entry point converts here instead of trusting the loader."""
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grey image, got shape {a.shape}")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("image contains NaN / inf")
    return a


# ============================ 1. SPECTRAL ====================================
def _radial_geometry(shape):
    """Radius (px), angle in [0, pi) and Nyquist radius for an fftshift-ed spectrum."""
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - cy, xx - cx)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), np.pi)   # spectrum is point-symmetric
    return r, theta, float(min(cy, cx))


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """Centred 2-D power spectrum P = |FFT2((I - mean) * Hann)|^2 (Hann: Harris 1978)."""
    x = img - img.mean()
    if config.SPECTRAL_USE_HANN_WINDOW:
        x = x * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1])).astype(np.float32)
    return np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2


def radial_profile(power: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean power in n_bins concentric annuli between 0 and Nyquist (64 bins, OURS)."""
    r, _, r_max = _radial_geometry(power.shape)
    valid = r <= r_max
    idx = np.minimum((r[valid] / r_max * n_bins).astype(int), n_bins - 1)
    sums = np.bincount(idx, weights=power[valid], minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    return sums / np.maximum(cnts, 1)


def spectral_features(img: np.ndarray) -> tuple[dict, np.ndarray]:
    """The 6 spectral features and the radial profile (re-used in Stage 7). See the table above."""
    n_bins = config.FORENSIC_SPECTRAL_RADIAL_BINS
    power = power_spectrum(img)
    r, theta, r_max = _radial_geometry(power.shape)
    valid = r <= r_max
    profile = radial_profile(power, n_bins)

    total = float(power[valid].sum())
    hf = valid & (r >= config.FORENSIC_SPECTRAL_HIGH_FREQUENCY_FRACTION * r_max)   # cutoff 0.5: OURS
    high = float(power[hf].sum())

    # 1/f^alpha slope of the radial profile (log-log linear fit; Field 1987, van der Schaaf 1996)
    lo = config.FORENSIC_SPECTRAL_SLOPE_MIN_BIN
    hi = config.FORENSIC_SPECTRAL_SLOPE_MAX_BIN or (n_bins - 1)
    b = np.arange(lo, hi + 1)
    x = np.log10((b + 0.5) / n_bins)
    y = np.log10(profile[b] + EPS)
    slope = float(np.polyfit(x, y, 1)[0])
    slope_r = float(np.corrcoef(x, y)[0, 1])

    # anisotropy: coefficient of variation of energy over angular sectors (12 sectors, OURS)
    ring = valid & (r >= config.SPECTRAL_ANISOTROPY_MIN_RADIUS * r_max)
    n_sec = config.SPECTRAL_ANISOTROPY_SECTORS
    sec = np.minimum((theta[ring] / np.pi * n_sec).astype(int), n_sec - 1)
    e_sec = np.bincount(sec, weights=power[ring], minlength=n_sec) / np.maximum(
        np.bincount(sec, minlength=n_sec), 1)
    aniso = float(e_sec.std() / (e_sec.mean() + EPS))

    return {
        "spectral_hfer": high / (total + EPS),
        "spectral_slope": slope,
        "spectral_slope_r": slope_r,
        "spectral_anisotropy": aniso,
        "spectral_total_energy": total / img.size,
        "spectral_high_energy": high / img.size,
    }, profile


# ---- 1b. SPECTRAL FEATURES ON A HIGH-PASS RESIDUAL (secondary set, ADAPTED) -------
def residual_image(img: np.ndarray) -> np.ndarray:
    """
    High-pass residual R = I - median_filter(I).
    Removes the large smooth anatomy and overall tone so that the spectrum reflects fine
    texture. Idea: Ricker et al. 2022 (following Marra et al. 2019, Wang et al. 2020).
    Kernel size: config.SPECTRAL_RESIDUAL_MEDIAN_SIZE (3, OURS).
    """
    x = _as_float32_2d(img)
    return x - cv2.medianBlur(x, config.SPECTRAL_RESIDUAL_MEDIAN_SIZE)


def spectral_features_residual(img: np.ndarray) -> tuple[dict, np.ndarray]:
    """The same 6 spectral features computed on the residual; names start with 'specres_'."""
    feats, profile = spectral_features(residual_image(img))
    return {k.replace("spectral_", "specres_", 1): v for k, v in feats.items()}, profile


# ============================ 2. WAVELET =====================================
def _safe_corr(a, b) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def wavelet_features(img: np.ndarray) -> dict:
    """
    pywt.wavedec2 -> [cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]; level 1
    is the FINEST.  Naming: LH = horizontal detail, HL = vertical, HH = diagonal.
    Energy = mean(c^2) (Unser 1995; Lyu & Farid 2005 use other moments); correlation =
    Pearson r between two orientations at the same level (adapted from Portilla &
    Simoncelli 2000). Haar, 3 levels (Mallat 1989; levels OURS).
    """
    levels = config.FORENSIC_WAVELET_LEVELS
    coeffs = pywt.wavedec2(img, config.FORENSIC_WAVELET_NAME, level=levels)
    out = {}
    for lvl in range(1, levels + 1):
        cH, cV, cD = coeffs[-lvl]
        out[f"wavelet_LH{lvl}_energy"] = float(np.mean(cH ** 2))
        out[f"wavelet_HL{lvl}_energy"] = float(np.mean(cV ** 2))
        out[f"wavelet_HH{lvl}_energy"] = float(np.mean(cD ** 2))
        out[f"wavelet_LH_HL_corr_L{lvl}"] = _safe_corr(cH, cV)
        out[f"wavelet_LH_HH_corr_L{lvl}"] = _safe_corr(cH, cD)
        out[f"wavelet_HL_HH_corr_L{lvl}"] = _safe_corr(cV, cD)
    return out


# ============================ 3. LBP =========================================
def lbp_features(img: np.ndarray) -> dict:
    """
    Normalised histogram of rotation-invariant UNIFORM LBP codes, P=8, R=1 (Ojala et al.
    2002, LBP^riu2_{8,1}). skimage method 'uniform' returns labels 0..P+1 = 10 bins:
    labels 0..8 = uniform patterns (label = number of neighbours >= the centre),
    label 9 = every non-uniform pattern. The bin count follows from P, it is not chosen.
    """
    img8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    # skimage pads with zeros outside the image, which would give the border pixels artificial
    # codes (~3 % of pixels). Reflect-pad by R first and drop the padding afterwards, so every
    # pixel is coded against real neighbours (same border rule as the cv2 filters below).
    r = int(np.ceil(config.FORENSIC_LBP_RADIUS))
    padded = np.pad(img8, r, mode="reflect")
    codes = local_binary_pattern(padded, config.FORENSIC_LBP_POINTS, config.FORENSIC_LBP_RADIUS,
                                 method=config.FORENSIC_LBP_METHOD)[r:-r, r:-r]
    n = config.FORENSIC_LBP_HISTOGRAM_BINS
    hist = np.bincount(codes.astype(int).ravel(), minlength=n)[:n].astype(np.float64)
    hist /= hist.sum() + EPS
    return {f"lbp_bin_{i:02d}": float(hist[i]) for i in range(n)}


# ============================ 4. SRM-STYLE ===================================
_SRM_KERNELS = [np.asarray(k, dtype=np.float32) for k in config.FORENSIC_SRM_FILTERS]


def srm_features(img: np.ndarray) -> dict:
    """
    Mean |r|, variance, std of high-pass residuals r = K * I, averaged over the 3 kernels in
    config.FORENSIC_SRM_FILTERS. Only kernel 1 is an SRM filter (Fridrich & Kodovsky 2012,
    square 3x3, up to a scale factor); kernels 2 (Laplacian) and 3 (Sobel) are OURS.
    """
    stats = []
    for k in _SRM_KERNELS:
        res = cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        if config.FORENSIC_SRM_CLIP is not None:
            res = np.clip(res, -config.FORENSIC_SRM_CLIP, config.FORENSIC_SRM_CLIP)
        stats.append((np.abs(res).mean(), res.var(), res.std()))
    m = np.mean(stats, axis=0)
    return {"srm_mean_abs": float(m[0]), "srm_variance": float(m[1]), "srm_std": float(m[2])}


# ============================ 5. NOISE =======================================
def noise_features(img: np.ndarray) -> dict:
    """
    Moments of a noise residual r (Laplacian 3x3 by default, or image - Gaussian blur).
    Idea: residual-based noise analysis (Lukas et al. 2006; Marra et al. 2019), which use a
    denoising filter; the Laplacian residual is OURS. rms ~ std and variance = std^2.
    """
    k = config.FORENSIC_NOISE_KERNEL_SIZE
    if config.FORENSIC_NOISE_FILTER == "laplacian":
        res = cv2.Laplacian(img, cv2.CV_32F, ksize=k, borderType=cv2.BORDER_REFLECT)
    elif config.FORENSIC_NOISE_FILTER == "gaussian":
        res = img - cv2.GaussianBlur(img, (k, k), 0)
    else:
        raise ValueError(f"Unknown FORENSIC_NOISE_FILTER={config.FORENSIC_NOISE_FILTER!r}")
    return {
        "noise_residual_mean": float(res.mean()),
        "noise_residual_abs_mean": float(np.abs(res).mean()),
        "noise_residual_variance": float(res.var()),
        "noise_residual_std": float(res.std()),
        "noise_residual_rms": float(np.sqrt(np.mean(res ** 2))),
    }


# ============================ 6. ALL AT ONCE =================================
def extract_feature_dict(img: np.ndarray) -> tuple[dict, np.ndarray]:
    """All 42 features (ordered as config.FEATURE_COLUMNS) + radial power profile."""
    img = _as_float32_2d(img)
    spec, profile = spectral_features(img)
    feats = {**spec, **wavelet_features(img), **lbp_features(img),
             **srm_features(img), **noise_features(img)}
    return {c: feats[c] for c in config.FEATURE_COLUMNS}, profile


def extract_feature_vector(img: np.ndarray) -> np.ndarray:
    """42-vector only (used for on-the-fly degraded images in the robustness stage)."""
    d, _ = extract_feature_dict(img)
    return np.fromiter(d.values(), dtype=np.float64, count=len(d))