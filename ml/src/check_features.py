"""Sanity checks for features.py. Needs no photos.

    ml/.venv/Scripts/python.exe ml/src/check_features.py
"""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from features import (FEATURE_LENGTH, extract_features, extract_features_from_bytes,
                      extract_features_from_path)

rng = np.random.default_rng(0)


def fake_image(h, w):
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img[:, : w // 2, 1] = 200  # greenish left half so it is not pure noise
    return img


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(1)


a = fake_image(300, 400)
vec = extract_features(a)
check(f"length is {FEATURE_LENGTH}", vec.shape == (FEATURE_LENGTH,))
check("no NaN or inf", bool(np.isfinite(vec).all()))
check("same image gives identical numbers", np.array_equal(vec, extract_features(a.copy())))
check("different images give different numbers", not np.array_equal(vec, extract_features(fake_image(300, 400))))

for shape in [(50, 50), (1000, 700), (256, 256), (3000, 4000)]:
    check(f"length is fixed for size {shape}", extract_features(fake_image(*shape)).shape == (FEATURE_LENGTH,))

flat = np.full((200, 200, 3), 128, dtype=np.uint8)  # zero-texture image must not crash (divide by zero)
check("plain gray image works", np.isfinite(extract_features(flat)).all())
check("grayscale (H, W) input works", extract_features(a[:, :, 0]).shape == (FEATURE_LENGTH,))
check("RGBA input works", extract_features(np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])).shape == (FEATURE_LENGTH,))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "test image é.png"  # space and non-ASCII on purpose
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    path.write_bytes(buf.tobytes())
    from_path = extract_features_from_path(path)
    from_bytes = extract_features_from_bytes(buf.tobytes())
    check("path and bytes give identical numbers", np.array_equal(from_path, from_bytes))
    check("lossless PNG round-trip matches the array", np.allclose(from_path, vec))

try:
    extract_features_from_bytes(b"not an image")
    check("bad bytes raise an error", False)
except ValueError:
    check("bad bytes raise ValueError", True)

start = time.perf_counter()
for _ in range(20):
    extract_features(a)
print(f"speed: {(time.perf_counter() - start) / 20 * 1000:.0f} ms per image")
print("all checks passed")
