"""Version 2 experiment: does averaging the network's numbers of a photo and of its mirror image
(left-right flip) make the trained models more accurate? It costs a second network run per photo
on the phone. Choices are made on the validation photos; with --test the pest classifier (the one
that uses it, train.PREDICT_WITH) is then scored once on the test photos too.

Run from the project root after train.py:
    ml/.venv/Scripts/python.exe ml/src/flip_experiment.py [--test]
Writes ml/results/flip_experiment.txt; keeps the mirrored numbers (per photo) in
ml/results/cnn_flip.npy + cnn_flip_keys.csv.
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from cnn_features import Network, key, load, split_file
from flow import MODEL_FILES
from predict import predict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
MODELS = ROOT / "backend" / "models"
CACHE, CACHE_KEYS = RESULTS / "cnn_flip.npy", RESULTS / "cnn_flip_keys.csv"


def mirrored(meta, rows):
    """The mirror image's numbers for these rows of features_meta (cached per photo)."""
    have = {}
    if CACHE.exists() and CACHE_KEYS.exists():
        have = dict(zip(pd.read_csv(CACHE_KEYS, keep_default_na=False)["key"], np.load(CACHE)))
    keys = [key(meta.loc[i]) for i in rows]
    todo = [i for i, k in zip(rows, keys) if k not in have]
    if todo:
        net = Network()
        for b in range(0, len(todo), 64):
            part = todo[b: b + 64]
            rgbs = Parallel(n_jobs=8, prefer="threads")(delayed(load)(split_file(meta.loc[i])) for i in part)
            for i, v in zip(part, net.vectors([cv2.flip(r, 1) for r in rgbs])):
                have[key(meta.loc[i])] = v
            if b % 1600 == 0:
                print(f"  mirrored {b}/{len(todo)}", flush=True)
        np.save(CACHE, np.array(list(have.values()), np.float32))
        pd.DataFrame({"key": list(have)}).to_csv(CACHE_KEYS, index=False)
    return np.array([have[k] for k in keys], np.float32)


def main():
    test = "--test" in sys.argv
    meta = pd.read_csv(RESULTS / "features_meta.csv", keep_default_na=False)
    X = np.load(RESULTS / "cnn_features.npy")
    meta["health"] = np.where(meta["pest_class"] == "healthy", "healthy", "damaged")
    corn, real = meta["crop"] == "corn", meta["aug"] == ""
    tasks = {  # same photos as train.py
        "corn_check": ((corn & (meta["pest_class"] != "other_insect")) | (meta["crop"] == "not_corn"), "crop"),
        "pest": (corn & (meta["pest_class"] != "unidentified_damage")
                 & ((meta["bug_visible"] == "yes") | (meta["pest_class"] == "healthy")), "pest_class"),
        "detector": (corn & (meta["pest_class"] != "healthy"), "bug_visible"),
        "health": (corn & (meta["pest_class"].isin(["healthy", "unidentified_damage"]) | (meta["bug_visible"] == "no")),
                   "health"),
    }
    lines = ["Mirror-image averaging: macro F1 / accuracy, plain -> averaged"]
    for split in ("val", "test") if test else ("val",):
        for name, (rows, col) in tasks.items():
            if split == "test" and name != "pest":
                continue  # only the model that uses it is scored on test
            bundle = joblib.load(MODELS / MODEL_FILES[name])
            idx = meta.index[rows & real & (meta["split"] == split)].to_numpy()
            y = meta.loc[idx, col].to_numpy()
            plain = predict(bundle, X[idx])
            avg = predict(bundle, (X[idx] + mirrored(meta, idx)) / np.float32(2))
            line = (f"  {split:<4} {name:<11} F1 {f1_score(y, plain, average='macro'):.4f} -> "
                    f"{f1_score(y, avg, average='macro'):.4f}, accuracy {(plain == y).mean():.4f} -> "
                    f"{(avg == y).mean():.4f}  ({len(idx)} photos)")
            print(line, flush=True)
            lines.append(line)
    (RESULTS / "flip_experiment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
