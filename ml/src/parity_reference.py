"""Reference answers from Python, for checking that the phone app's offline code (lib/offline/)
gives the same features and model answers. Used by test/offline_parity_test.dart.

Run after export_models.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/parity_reference.py
    flutter test test/offline_parity_test.dart

Writes build/parity/ (not on GitHub):
    raw/<n>.rgb + raw.json   40 test photos already decoded by OpenCV (raw RGB bytes) with
                             Python's features: checks the Dart image steps and features exactly
    e2e.json                 800 test photos (100 of each kind): path, Python's features and each
                             model's probabilities; the Dart test decodes the JPEG itself with
                             Flutter's decoder, as the phone does
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import decode_image, extract_features
from make_split import SPLIT_DIR
from predict import predict_proba

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "parity"
MODELS = {"corn_check": "corn_check", "detector": "corn_detector", "health": "corn_health", "pest": "corn_pest"}


def group(row):
    if row["crop"] == "not_corn":
        return "other plant"
    if row["pest_class"] in ("healthy", "unidentified_damage"):
        return row["pest_class"]
    return row["pest_class"] + (" insect" if row["bug_visible"] == "yes" else " damage")


def main():
    X = np.load(ROOT / "ml/results/features.npy")
    meta = pd.read_csv(ROOT / "ml/results/features_meta.csv", keep_default_na=False)
    test = meta[(meta["split"] == "test") & (meta["aug"] == "") & meta["path"].str.lower().str.endswith(".jpg")].copy()
    test["group"] = test.apply(group, axis=1)
    rng = np.random.default_rng(0)
    picks = []
    for _, g in test.groupby("group"):
        picks.extend(rng.choice(g.index.to_numpy(), min(100, len(g)), replace=False))
    picks = np.array(sorted(picks))
    bundles = {k: joblib.load(ROOT / "backend/models" / f"{v}_model.joblib") for k, v in MODELS.items()}
    probs = {k: predict_proba(b, X[picks]) for k, b in bundles.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, i in enumerate(picks):
        rows.append({
            "path": str(SPLIT_DIR / "test" / meta.loc[i, "path"]),
            "group": test.loc[i, "group"],
            "features": [float(v) for v in X[i]],
            "probs": {k: [float(v) for v in probs[k][n]] for k in MODELS},
            "classes": {k: list(b["classes"]) for k, b in bundles.items()},
        })
    (OUT / "e2e.json").write_text(json.dumps(rows), encoding="utf-8")

    raw_dir = OUT / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw = []
    for n, i in enumerate(rng.choice(picks, 40, replace=False)):
        rgb = decode_image(np.fromfile(str(SPLIT_DIR / "test" / meta.loc[i, "path"]), dtype=np.uint8).tobytes())
        (raw_dir / f"{n}.rgb").write_bytes(np.ascontiguousarray(rgb).tobytes())
        raw.append({"file": f"{n}.rgb", "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                    "features": [float(v) for v in extract_features(rgb)]})
    (OUT / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    print(f"wrote {len(rows)} end-to-end and {len(raw)} raw-pixel references to {OUT}")


if __name__ == "__main__":
    main()
