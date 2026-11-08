"""Estimate damage severity from one corn photo: K-Means colour clusters + a colour rule.

Like features.py, this file is shared with the backend (backend/app/services/severity.py
must be an exact copy).

    from severity import assess
    result = assess(rgb)
    result["damage_percent"], result["level"]   # e.g. 23.4, "Moderate"

Steps (see docs/severity_thresholds.md for the reasoning and limits):
  1. Remove the background: white/grey tables, black backgrounds and sky, but only where
     they connect to the photo's edge - dark lesions and holes inside the plant stay.
  2. K-Means groups the remaining plant pixels into K colour clusters (Lab colour space).
  3. Each cluster is called healthy or damaged by a colour rule: green is healthy; yellow,
     brown, black and grey/white (dead tissue) are damage.
  4. Holes inside the leaf outline (chewed by caterpillars) also count as damage.
  5. damage % = damaged area / whole plant area, graded with LOW_UP_TO / MODERATE_UP_TO.

Made for corn leaves, whorls and stalks. Corn ears are not supported: yellow kernels are
healthy but would read as damage under the green rule.
"""
import math

import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

WORK_SIZE = 512        # longest side used for the analysis
K = 4                  # colour clusters
FIT_PIXELS = 20000     # K-Means is fitted on a random sample of plant pixels, then applied to all
HEALTHY_BELOW = 5.0    # less than this % damage is reported as "Healthy"
LOW_UP_TO = 20.0       # Low: 5-20% damage
MODERATE_UP_TO = 40.0  # Moderate: 20-40%; High: above 40%. Placeholders, see docs/severity_thresholds.md
MIN_PLANT_SHARE = 0.05 # fewer plant pixels than this share of the photo -> can't assess
EDGE_BAND = 2          # pixels at the plant's outline left out of the count (colour blending)
HUE_MIN = 105          # Lab hue: ~100 yellow, ~130-160 green; below = yellow/brown damage.
                       # 100 marked fewer healthy olive leaves as damaged but missed more
                       # early yellowing (tested 2026-10-14), so the stricter value is kept.
CHROMA_MIN = 6         # below this a colour is grey/white/black: dead tissue


def _lab(rgb):
    """Lab with L in 0-100 and a, b centred on 0 (OpenCV stores them as 0-255)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _background_mask(rgb):
    """Returns (background, inner): background-coloured pixels touching the photo edge, and
    background-coloured pixels enclosed by the plant (holes, or very dark/pale dead spots)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    white = (s < 0.15) & (v > 0.80)
    black = v < 0.12
    grey = (s < 0.10) & (v >= 0.12)
    sky = (h >= 90) & (h <= 130) & (s > 0.20) & (v > 0.35)
    candidate = white | black | grey | sky

    # Keep only candidate regions that touch the photo edge: a dark lesion or a white
    # spot inside the leaf is damage, not background.
    labels, n = ndimage.label(candidate)
    if n == 0:
        return candidate, candidate
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge = edge[edge > 0]
    background = ndimage.binary_opening(np.isin(labels, edge), iterations=2)
    return background, candidate & ~background


def _is_healthy(center_lab):
    """Colour rule for one cluster centre (L 0-100, a, b): green is healthy."""
    L, a, b = (float(x) for x in center_lab)
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360  # ~100 yellow, ~130-160 green, 40-80 brown/orange
    return chroma >= CHROMA_MIN and HUE_MIN <= hue <= 220 and L >= 20


def grade(damage_percent):
    if damage_percent < HEALTHY_BELOW:
        return "Healthy"
    if damage_percent <= LOW_UP_TO:
        return "Low"
    if damage_percent <= MODERATE_UP_TO:
        return "Moderate"
    return "High"


def assess(rgb, seed=0):
    """Severity of an RGB uint8 image (H, W, 3). Returns a dict, see keys below."""
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    scale = WORK_SIZE / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.GaussianBlur(rgb, (3, 3), 0)

    # Enclosed background-coloured areas are holes (or dead spots): damage, counted separately.
    background, holes = _background_mask(rgb)
    plant = ~background & ~holes
    total = plant.size
    if plant.sum() < MIN_PLANT_SHARE * total:
        return {"damage_percent": None, "level": "Unclear",
                "reason": "could not find the plant in the photo", "plant_share": float(plant.mean())}

    lab = _lab(rgb)
    pixels = lab[plant]
    rng = np.random.default_rng(seed)
    sample = pixels[rng.choice(len(pixels), min(FIT_PIXELS, len(pixels)), replace=False)]
    weights = np.array([0.5, 1.0, 1.0], dtype=np.float32)  # colour matters more than brightness
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(sample * weights)
    cluster = km.predict(pixels * weights)
    centers = km.cluster_centers_ / weights
    healthy_cluster = np.array([_is_healthy(c) for c in centers])

    damaged = np.zeros(plant.shape, dtype=bool)
    damaged[plant] = ~healthy_cluster[cluster]

    # Pixels on the plant's outline blend leaf and background colours and would read as
    # damage; on thin leaves that alone adds ~10%. Leave a thin edge band out of the count.
    edge_band = (plant | holes) & ~ndimage.binary_erosion(plant | holes, iterations=EDGE_BAND)
    plant &= ~edge_band
    damaged &= ~edge_band

    plant_area = plant.sum() + holes.sum()
    damage_area = damaged.sum() + holes.sum()
    pct = 100.0 * damage_area / plant_area
    return {
        "damage_percent": round(float(pct), 1),
        "level": grade(pct),
        "plant_share": round(float(plant_area / total), 3),
        "hole_percent": round(float(100.0 * holes.sum() / plant_area), 1),
        "clusters": [{"lab": [round(float(x), 1) for x in c], "healthy": bool(hc),
                      "share": round(float((cluster == i).mean()), 3)}
                     for i, (c, hc) in enumerate(zip(centers, healthy_cluster))],
        "_masks": {"plant": plant, "damaged": damaged, "holes": holes, "rgb": rgb},
    }


def overlay(result):
    """Picture of what was measured: healthy tissue green, damage red, holes orange,
    background dimmed. Handy for checking results and for figures in the paper."""
    m = result["_masks"]
    out = m["rgb"].astype(np.float32)
    out[~m["plant"] & ~m["holes"]] *= 0.25
    tint = np.zeros_like(out)
    tint[m["plant"] & ~m["damaged"]] = (0, 200, 0)
    tint[m["damaged"]] = (230, 0, 0)
    tint[m["holes"]] = (255, 150, 0)
    painted = m["plant"] | m["holes"]
    out[painted] = 0.55 * out[painted] + 0.45 * tint[painted]
    return out.clip(0, 255).astype(np.uint8)


def assess_path(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {path}")
    return assess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python severity.py <image>")
    r = assess_path(sys.argv[1])
    print(f"{r['damage_percent']}% damage -> {r['level']}")
