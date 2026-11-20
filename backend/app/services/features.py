"""Turn a photo into a fixed-length list of numbers (color, texture, edges).

This file is used by BOTH training (ml/src/train.py) and the backend
(backend/app/services/features.py must be an exact copy). If the two ever differ,
the app's predictions will be wrong. Change this file only together with retraining.

    from features import extract_features_from_path
    vec = extract_features_from_path("leaf.jpg")   # numpy array, length FEATURE_LENGTH
"""
import sys

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

IMG_SIZE = 256        # every photo is resized to IMG_SIZE x IMG_SIZE first
HOG_SIZE = 128        # grayscale size used for HOG
GLCM_LEVELS = 32      # gray levels used for GLCM (fewer levels = faster)

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
LBP_SETTINGS = [(8, 1), (16, 2)]  # (points, radius); "uniform" LBP has points + 2 bins

_N_COLOR_STATS = 12                 # mean+std of R,G,B,H,S,V
_N_HSV_HIST = 16 + 8 + 8            # H, S, V histogram bins
_N_GLCM = len(GLCM_PROPS) * len(GLCM_DISTANCES)
_N_LBP = sum(p + 2 for p, _ in LBP_SETTINGS)
_N_HOG = 324                        # 9 orientations, 32px cells on 128px image, 2x2 blocks
FEATURE_LENGTH = _N_COLOR_STATS + _N_HSV_HIST + _N_GLCM + _N_LBP + _N_HOG


def _color_features(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H 0-179, S 0-255, V 0-255
    stats = []
    for img in (rgb, hsv):
        flat = img.reshape(-1, 3).astype(np.float32)
        stats.extend(flat.mean(axis=0) / 255.0)
        stats.extend(flat.std(axis=0) / 255.0)

    hist = []
    for channel, bins, upper in ((0, 16, 180), (1, 8, 256), (2, 8, 256)):
        h, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0, upper))
        hist.extend(h / h.sum())
    return np.array(stats + hist, dtype=np.float32)


def _glcm_features(gray: np.ndarray) -> np.ndarray:
    quantized = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)
    glcm = graycomatrix(quantized, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS,
                        symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        feats.extend(values.mean(axis=1))  # average over angles
    return np.array(feats, dtype=np.float32)


def _lbp_features(gray: np.ndarray) -> np.ndarray:
    feats = []
    for points, radius in LBP_SETTINGS:
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        h, _ = np.histogram(lbp, bins=points + 2, range=(0, points + 2))
        feats.extend(h / h.sum())
    return np.array(feats, dtype=np.float32)


def _hog_features(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
               block_norm="L2-Hys").astype(np.float32)


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Features for one image given as an RGB uint8 array of shape (H, W, 3)."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {rgb.shape}")
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)

    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    vec = np.concatenate([
        _color_features(rgb),
        _glcm_features(gray),
        _lbp_features(gray),
        _hog_features(gray),
    ])
    assert vec.shape[0] == FEATURE_LENGTH, (vec.shape, FEATURE_LENGTH)
    return vec


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes (for example an uploaded file) to an RGB array."""
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    return extract_features(decode_image(data))


def extract_features_from_path(path) -> np.ndarray:
    # np.fromfile instead of cv2.imread so paths with non-ASCII characters work on Windows
    return extract_features_from_bytes(np.fromfile(str(path), dtype=np.uint8).tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python features.py <image>")
    v = extract_features_from_path(sys.argv[1])
    print(f"{len(v)} features, min {v.min():.3f}, max {v.max():.3f}")
