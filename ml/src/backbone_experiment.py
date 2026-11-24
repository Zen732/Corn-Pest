"""Version 2 experiment: would another ready-made network make the pest classifier more accurate?
Compares the current MobileNetV3-Large with EfficientNet-B0, EfficientNet-B2 (torchvision,
ImageNet) and DINOv2 ViT-S/14 (Meta, self-supervised; torch.hub facebookresearch/dinov2). Nothing
in the app changes; it only measures.

For every network, the same photos and the same SVM:
    photos   the pest classifier's real photos (5 pests + healthy + other insect; train / val /
             test as in make_split.py, no augmented copies) and the 300 web test photos
    numbers  the network's output for the whole photo resized to its input size (OpenCV
             INTER_AREA, ImageNet mean / std), averaged over the image (EfficientNet, MobileNet),
             or DINOv2's class token next to the average of its patch tokens
    SVM      scaler + RBF SVM, C 3 / 10, class weights none / balanced, chosen on validation
             macro F1; scored once on test and on the web photos
    speed    milliseconds per photo on this PC (one photo at a time, PyTorch)

Web photos are scored on the label alone: one of the five pests is right if named; any other
pest is right if it is NOT named as one of the five (the 50% rule is not applied here).

Run from the project root (one network at a time; safe to stop and run again, it resumes):
    ml/.venv/Scripts/python.exe ml/src/backbone_experiment.py [--only efficientnet_b0 ...]
Writes ml/results/backbones/<network>.npy (photo numbers, kept off GitHub) and
ml/results/backbones/summary.txt + results.json.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cnn_features import load, split_file

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
OUT = RESULTS / "backbones"
WEB = ROOT / "ml" / "dataset"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FIVE = {"fall_armyworm", "common_cutworm", "armyworm", "corn_aphid", "corn_planthopper"}
NETWORKS = {  # name: input size
    "mobilenet_v3_large": 224,
    "efficientnet_b0": 224,
    "efficientnet_b2": 288,
    "dinov2_vits14": 224,
}
BATCH = 32


def build(name):
    """(model, function: batch tensor -> photo numbers)."""
    import torch
    import torchvision.models as tv
    torch.set_num_threads(os.cpu_count() or 4)
    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval()

        def run(x):
            o = m.forward_features(x)
            return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)
        return m, run
    m = {"mobilenet_v3_large": lambda: tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
         "efficientnet_b0": lambda: tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1),
         "efficientnet_b2": lambda: tv.efficientnet_b2(weights=tv.EfficientNet_B2_Weights.IMAGENET1K_V1)}[name]().eval()
    return m, lambda x: m.avgpool(m.features(x)).flatten(1)


def prepare(rgb, size):
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def photo_sets():
    """(rows of features_meta for the pest photos, web photos [(file, kind, expected)])."""
    meta = pd.read_csv(RESULTS / "features_meta.csv", keep_default_na=False)
    rows = meta[(meta["crop"] == "corn") & (meta["aug"] == "") & (meta["pest_class"] != "unidentified_damage")
                & ((meta["bug_visible"] == "yes") | (meta["pest_class"] == "healthy"))]
    with open(WEB / "test_web_credits.csv", newline="", encoding="utf-8") as f:
        web = [(r["file"], r["kind"], r["expected"]) for r in csv.DictReader(f)]
    return rows, web


def numbers(name, rows, web):
    """Photo numbers for the pest photos then the web photos (cached, resumable)."""
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / f"{name}.npy"
    n = len(rows) + len(web)
    if final.exists():
        X = np.load(final)
        if len(X) == n:
            return X
    paths = [split_file(r) for _, r in rows.iterrows()] + [WEB / f for f, _, _ in web]
    part, done = OUT / f"{name}_partial.npy", 0
    X = None
    if part.exists():
        X = np.load(part)
        done = int((np.abs(X).sum(1) > 0).sum()) // BATCH * BATCH
        print(f"{name}: resuming at {done}/{n}", flush=True)
    import torch
    model, run = build(name)
    size = NETWORKS[name]
    t = time.time()
    with torch.no_grad():
        for b in range(done, n, BATCH):
            rgbs = Parallel(n_jobs=6, prefer="threads")(delayed(load)(p) for p in paths[b: b + BATCH])
            v = run(torch.from_numpy(np.stack([prepare(r, size) for r in rgbs]))).numpy().astype(np.float32)
            if X is None:
                X = np.zeros((n, v.shape[1]), np.float32)
            X[b: b + len(v)] = v
            if (b // BATCH) % 50 == 0:
                print(f"  {name}: {b + len(v)}/{n}  {time.time() - t:.0f}s", flush=True)
                np.save(part, X)
    np.save(final, X)
    part.unlink(missing_ok=True)
    return X


def speed(name):
    """Milliseconds per photo, one photo at a time (after a warm-up)."""
    import torch
    model, run = build(name)
    size = NETWORKS[name]
    x = torch.from_numpy(prepare(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8), size)[None])
    with torch.no_grad():
        run(x)
        t = time.time()
        for _ in range(10):
            run(x)
    return round((time.time() - t) * 100, 1)


def _fit(Xtr, ytr, Xva, yva, C, cw):
    m = SVC(C=C, gamma="scale", class_weight=cw, cache_size=1500).fit(Xtr, ytr)
    return f1_score(yva, m.predict(Xva), average="macro"), C, cw, m


def score(name, X, rows, web):
    y, split = rows["pest_class"].to_numpy(), rows["split"].to_numpy()
    k = len(rows)
    tr, va, te = split == "train", split == "val", split == "test"
    sc = StandardScaler().fit(X[:k][tr])
    Z = sc.transform(X)
    tried = Parallel(n_jobs=4)(delayed(_fit)(Z[:k][tr], y[tr], Z[:k][va], y[va], C, cw)
                               for C in (3, 10) for cw in (None, "balanced"))
    f1, C, cw, m = max(tried, key=lambda t: t[0])
    pred = m.predict(Z[:k][te])
    web_pred = m.predict(Z[k:])
    expected = np.array([e for _, _, e in web])
    five = np.isin(expected, list(FIVE))
    right = np.where(five, web_pred == expected, ~np.isin(web_pred, list(FIVE)))
    return {
        "dims": int(X.shape[1]), "C": C, "class_weight": cw, "val_macro_f1": round(f1, 4),
        "test_accuracy": round(accuracy_score(y[te], pred), 4), "test_macro_f1": round(f1_score(y[te], pred, average="macro"), 4),
        "test_right_by_class": {c: round(float((pred[y[te] == c] == c).mean()), 3) for c in sorted(set(y))},
        "web_five_named_right": round(float(right[five].mean()), 3),
        "web_others_not_named": round(float(right[~five].mean()), 3),
        "web_all": round(float(right.mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(NETWORKS))
    names = ap.parse_args().only or list(NETWORKS)
    rows, web = photo_sets()
    print(f"{len(rows)} pest photos ({', '.join(f'{s} {(rows.split == s).sum()}' for s in ('train', 'val', 'test'))}), "
          f"{len(web)} web photos", flush=True)
    res_file = OUT / "results.json"
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    for name in names:
        X = numbers(name, rows, web)
        print(f"{name}: training the SVM on {X.shape[1]} numbers per photo ...", flush=True)
        r = score(name, X, rows, web)
        r["ms_per_photo_pc"] = speed(name)
        results[name] = r
        res_file.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"{name}: {r}", flush=True)
    lines = ["PEST CLASSIFIER with different ready-made networks (same photos, same SVM)", "",
             f"{'network':<20}{'numbers':>8}{'ms/photo':>9}{'val F1':>8}{'test acc':>9}{'test F1':>8}"
             f"{'web: 5 pests':>13}{'web: others':>12}{'web: all':>9}"]
    for name, r in results.items():
        lines.append(f"{name:<20}{r['dims']:>8}{r['ms_per_photo_pc']:>9}{r['val_macro_f1']:>8.3f}"
                     f"{r['test_accuracy']:>9.1%}{r['test_macro_f1']:>8.3f}{r['web_five_named_right']:>13.0%}"
                     f"{r['web_others_not_named']:>12.0%}{r['web_all']:>9.1%}")
    lines += ["", "Test photos named right, by class:"]
    classes = sorted(next(iter(results.values()))["test_right_by_class"])
    lines.append(f"{'network':<20}" + "".join(f"{c[:12]:>13}" for c in classes))
    for name, r in results.items():
        lines.append(f"{name:<20}" + "".join(f"{r['test_right_by_class'][c]:>13.1%}" for c in classes))
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
