"""Version 2 photo features: a ready-made image network (trained on ImageNet, not retrained) turns
each photo into a list of numbers (NETWORK below: EfficientNet-B2, 1408 numbers, since 25 Nov 2026;
MobileNetV3-Large, 960, before). The SVMs are then trained on these numbers instead of (or next to)
the hand-crafted features.py vector.

Same photos and order as extract_all.py: one row per row of ml/results/features_meta.csv.
Run after extract_all.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/cnn_features.py
Writes ml/results/cnn_features.npy (float32, shape (N, LENGTH)) and cnn_features_network.txt
(which network made them). Rows already computed by the same network (same set, path and
augmentation kind) are reused, also those from backbone_experiment.py.

Photo preparation (the phone app must do exactly the same): decode as features.decode_image,
resize to SIZE x SIZE with OpenCV INTER_AREA, scale to 0-1, subtract the ImageNet mean
(0.485, 0.456, 0.406) and divide by the ImageNet std (0.229, 0.224, 0.225) per RGB channel.
The network's output after global average pooling (before its classifier) is the vector.

Which network (24 Nov 2026, backbone_experiment.py, same photos and SVM for each): the pest
classifier named 91.0% of test photos right with EfficientNet-B2 against 86.2% with
MobileNetV3-Large, and 74.7% of 300 web photos against 70.3%. DINOv2 ViT-S/14 did better still
(93.3% / 76.0%) but would be about 3 times slower than B2 on the phone and much bigger, and it is
a different kind of network to run on the phone. The MobileNetV3 models are kept in
_archive/1_old_models/version2_mobilenet_2026-11-24/.
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features import decode_image
from make_split import AUG_PREFIX, SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"  # cnn_features.npy is about 300 MB: kept off GitHub (*.npy)
OUT_FILE = RESULTS / "cnn_features.npy"
KEYS_FILE = RESULTS / "cnn_features_keys.csv"
NETWORK_FILE = RESULTS / "cnn_features_network.txt"
# name: (input size, numbers per photo, torchvision weights)
NETWORKS = {"mobilenet_v3_large": (224, 960, "MobileNet_V3_Large_Weights.IMAGENET1K_V2"),
            "efficientnet_b2": (288, 1408, "EfficientNet_B2_Weights.IMAGENET1K_V1")}
NETWORK = "efficientnet_b2"
SIZE, LENGTH, WEIGHTS = NETWORKS[NETWORK]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare(rgb):
    """uint8 RGB (H, W, 3) -> float32 (3, SIZE, SIZE), as the phone does it."""
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def torch_model():
    """NETWORK with its ImageNet weights (torchvision; downloaded once to ~/.cache/torch)."""
    import torchvision
    group, name = WEIGHTS.split(".")
    return getattr(torchvision.models, NETWORK)(weights=getattr(torchvision.models, group)[name]).eval()


class Network:
    def __init__(self):
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
        self.torch = torch
        self.model = torch_model()

    def vectors(self, rgbs):
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack([prepare(r) for r in rgbs]))
            return self.model.avgpool(self.model.features(x)).flatten(1).numpy().astype(np.float32)


def split_file(row):
    """The photo file in split/ for one features_meta row."""
    crop, folder, name = row["path"].split("/")
    if row["aug"]:
        name = f"{AUG_PREFIX}{row['aug']}_{name}"
    return SPLIT_DIR / row["split"] / crop / folder / name


def load(path):
    return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())


def key(row):
    return f"{row['split']}|{row['path']}|{row['aug']}"


def reusable(meta):
    """{key: vector} made by NETWORK: from an earlier run, and from backbone_experiment.py (or, for
    MobileNetV3, pest_feature_experiment.py), which did the real pest photos the same way."""
    found = {}
    made_by = NETWORK_FILE.read_text(encoding="utf-8").strip() if NETWORK_FILE.exists() else "mobilenet_v3_large"
    if OUT_FILE.exists() and KEYS_FILE.exists() and made_by == NETWORK:
        X = np.load(OUT_FILE)
        for i, k in enumerate(pd.read_csv(KEYS_FILE, keep_default_na=False)["key"]):
            found[k] = X[i]
    rows = meta[(meta["crop"] == "corn") & (meta["aug"] == "") & (meta["pest_class"] != "unidentified_damage")
                & ((meta["bug_visible"] == "yes") | (meta["pest_class"] == "healthy"))]
    for exp in (RESULTS / "backbones" / f"{NETWORK}.npy",
                RESULTS / "pest_experiment" / "cnn_vectors.npy" if NETWORK == "mobilenet_v3_large" else None):
        if exp is not None and exp.exists():
            X = np.load(exp, mmap_mode="r")
            if len(X) >= len(rows) and X.shape[1] == LENGTH:  # backbones/: then the 300 web photos
                for v, (_, r) in zip(X, rows.iterrows()):
                    found.setdefault(key(r), np.array(v))
    return found


def main():
    meta = pd.read_csv(RESULTS / "features_meta.csv", keep_default_na=False)
    keys = [key(r) for _, r in meta.iterrows()]
    done = reusable(meta)
    X = np.zeros((len(meta), LENGTH), np.float32)
    have = np.zeros(len(meta), bool)
    for i, k in enumerate(keys):
        if k in done:
            X[i], have[i] = done[k], True
    todo = list(np.flatnonzero(~have))
    print(f"{len(meta)} photos: reusing {len(meta) - len(todo)}, computing {len(todo)}", flush=True)
    if todo:
        net = Network()
        files = [split_file(meta.iloc[i]) for i in todo]
        t = time.time()
        for b in range(0, len(todo), 32):
            rgbs = Parallel(n_jobs=6, prefer="threads")(delayed(load)(f) for f in files[b: b + 32])
            X[todo[b: b + 32]] = net.vectors(rgbs)
            have[todo[b: b + 32]] = True
            if (b // 32) % 50 == 0:
                done_n = b + len(rgbs)
                print(f"  {done_n}/{len(todo)}  {time.time() - t:.0f}s", flush=True)
            if (b // 32) % 200 == 199:  # save progress now and then (only the photos done)
                save(X[have], [k for k, h in zip(keys, have) if h])
    if len(todo) and len(done):  # check: a reused vector equals a fresh one
        net = net if todo else Network()
        sample = np.random.default_rng(0).choice(np.flatnonzero([k in done for k in keys]), 8, replace=False)
        fresh = net.vectors([load(split_file(meta.iloc[i])) for i in sample])
        print(f"check: 8 reused vectors vs fresh, largest difference {np.abs(fresh - X[sample]).max():.2e}")
    save(X, keys)
    print(f"saved {X.shape} to {OUT_FILE}")


def save(X, keys):
    np.save(OUT_FILE, X)
    pd.DataFrame({"key": keys}).to_csv(KEYS_FILE, index=False)
    NETWORK_FILE.write_text(NETWORK + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
