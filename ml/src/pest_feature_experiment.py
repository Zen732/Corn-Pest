"""Experiment: can the pest classifier name pests better with other photo features?

Nothing here changes the app; it only measures. Four ways to describe a photo are compared with
the same classifier (SVM, RBF kernel; C and class weights chosen on validation photos), the same
training photos and the same test photos as the pest classifier in train.py (real photos with
the insect in view, plus healthy; augmented copies are not used here, for every method alike):

    current     the app's features.py vector of the whole photo (406 numbers)
    crop        classic: find the insect (the part that is not green leaf or white background),
                crop to it, features.py on the crop + 7 shape numbers (size, elongation,
                solidity, ...); falls back to the whole photo when no insect-like region is found
    cnn         a ready-made image network (MobileNetV3-Large, trained on ImageNet) turns the
                photo into 960 numbers; the SVM still decides
    cnn+current both side by side

Reported on the test photos: accuracy, macro F1, each pest's share named right, and with the
app's 50% rule: how many pest names would be shown and how many of those are right.

Run from the project root (needs torch + torchvision for "cnn"):
    ml/.venv/Scripts/python.exe ml/src/pest_feature_experiment.py
Writes ml/results/pest_experiment/summary.txt, results.json, and the computed vectors (*.npy,
reused on later runs).
"""
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

from features import decode_image, extract_features
from make_split import SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
OUT = RESULTS / "pest_experiment"
PEST_AT_LEAST = 0.50
CATERPILLARS = {"fall_armyworm", "common_cutworm", "armyworm"}
SCREENSHOTS = Path(os.environ.get("PEST_SCREENSHOTS", "."))  # folder with 7.jpg, 8.jpg, 9.jpg (optional)


# ---------------- the photos ----------------

def pest_rows():
    meta = pd.read_csv(RESULTS / "features_meta.csv", keep_default_na=False)
    rows = meta[(meta["crop"] == "corn") & (meta["aug"] == "") & (meta["pest_class"] != "unidentified_damage")
                & ((meta["bug_visible"] == "yes") | (meta["pest_class"] == "healthy"))]
    return rows


def load_rgb(path):
    return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())


# ---------------- classic: crop to the insect ----------------

def insect_crop(rgb):
    """(crop, 7 shape numbers). The insect = the biggest central region that is neither green
    leaf nor white/very pale background."""
    h, w = rgb.shape[:2]
    scale = 512 / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    green = (hh >= 25) & (hh <= 95) & (ss > 50) & (vv > 40)
    pale = (ss < 35) & (vv > 190)
    mask = (~(green | pale)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    sh, sw = small.shape[:2]
    total = sh * sw
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (0.005 * total <= area <= 0.70 * total):
            continue
        dx, dy = (cents[i][0] - sw / 2) / sw, (cents[i][1] - sh / 2) / sh
        score = area * (1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5))
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return rgb, np.zeros(7, np.float32)
    x, y, bw, bh, area = stats[best]
    comp = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c)) or 1.0
    perim = cv2.arcLength(c, True) or 1.0
    if len(c) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(c)
        elong = max(a1, a2) / max(1e-6, min(a1, a2))
    else:
        elong = max(bw, bh) / max(1, min(bw, bh))
    shape = np.array([1.0, area / total, max(bw, bh) / max(1, min(bw, bh)), min(elong, 20.0),
                      area / hull, area / max(1, bw * bh), min(1.0, 4 * np.pi * area / perim ** 2)], np.float32)
    pad = 0.2
    x0, y0 = max(0, int((x - pad * bw) / scale)), max(0, int((y - pad * bh) / scale))
    x1, y1 = min(w, int((x + (1 + pad) * bw) / scale)), min(h, int((y + (1 + pad) * bh) / scale))
    crop = rgb[y0:y1, x0:x1]
    return (crop if crop.shape[0] >= 16 and crop.shape[1] >= 16 else rgb), shape


def crop_vector(rgb):
    crop, shape = insect_crop(rgb)
    return np.concatenate([extract_features(crop), shape]).astype(np.float32)


def _crop_one(path):
    try:
        return crop_vector(load_rgb(path))
    except Exception as e:
        print(f"  failed {path}: {e}")
        return None


# ---------------- ready-made network ----------------

class Cnn:
    def __init__(self):
        import torch
        import torchvision
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
        w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.torch = torch
        self.model = torchvision.models.mobilenet_v3_large(weights=w).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def prep(self, rgb):
        x = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    def vectors(self, rgbs):
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack([self.prep(r) for r in rgbs]))
            f = self.model.avgpool(self.model.features(x)).flatten(1)
        return f.numpy().astype(np.float32)


def cnn_vectors(paths, cnn, batch=64):
    out = []
    t = time.time()
    for i in range(0, len(paths), batch):
        rgbs = Parallel(n_jobs=8, prefer="threads")(delayed(load_rgb)(p) for p in paths[i: i + batch])
        out.append(cnn.vectors(rgbs))
        if (i // batch) % 20 == 0:
            print(f"  cnn {i + len(rgbs)}/{len(paths)} ({time.time() - t:.0f}s)", flush=True)
    return np.vstack(out)


# ---------------- train and score ----------------

def fit_and_score(name, Xtr, ytr, Xva, yva, Xte, yte, classes):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in (1, 10):
        for cw in (None, "balanced"):
            m = SVC(C=C, gamma="scale", class_weight=cw, cache_size=2000).fit(Xtr, ytr)
            f1 = f1_score(yva, m.predict(Xva), average="macro")
            print(f"  {name:12s} C={C:<3} class_weight={str(cw):9s} val macro F1 {f1:.3f}", flush=True)
            if best is None or f1 > best[0]:
                best = (f1, C, cw)
    _, C, cw = best
    m = SVC(C=C, gamma="scale", class_weight=cw, probability=True, random_state=42, cache_size=2000).fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = m.classes_[proba.argmax(1)]
    conf = proba.max(1)
    pests = yte != "healthy"
    named = pests & (pred != "healthy") & (conf >= PEST_AT_LEAST)
    r = {"val_macro_f1": round(best[0], 4), "C": C, "class_weight": cw,
         "test_accuracy": round(accuracy_score(yte, pred), 4),
         "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
         "pest_photos_right": round(float((pred[pests] == yte[pests]).mean()), 4),
         "caterpillars_right": round(float((pred[np.isin(yte, list(CATERPILLARS))] ==
                                            yte[np.isin(yte, list(CATERPILLARS))]).mean()), 4),
         "by_class_right": {c: round(float((pred[yte == c] == c).mean()), 4) for c in classes},
         "rule50_names_shown_share": round(float(named.sum() / pests.sum()), 4),
         "rule50_names_right": round(float((pred[named] == yte[named]).mean()), 4),
         "rule50_pest_photos_named_right": round(float(((pred == yte) & named)[pests].mean()), 4)}
    return r, (sc, m)


def screenshot_photos():
    """The three photos from the user's phone screenshots (beetle, caterpillar, aphid)."""
    out = {}
    for name, label in (("7.jpg", "beetle (not one of the 5)"), ("8.jpg", "caterpillar"), ("9.jpg", "aphid")):
        p = SCREENSHOTS / name
        if p.exists():
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            out[label] = img[200:610, 48:672]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pest_rows()
    paths = [SPLIT_DIR / s / p for s, p in zip(rows["split"], rows["path"])]
    y = rows["pest_class"].values
    split = rows["split"].values
    classes = sorted(set(y))
    print(f"{len(rows)} photos: " + ", ".join(f"{s} {(split == s).sum()}" for s in ("train", "val", "test")))

    X_all = np.load(RESULTS / "features.npy")
    feats = {"current": X_all[rows.index.values]}

    f = OUT / "crop_vectors.npy"
    if not f.exists():
        print("crop: finding the insect in every photo...", flush=True)
        vecs = Parallel(n_jobs=-1, verbose=2)(delayed(_crop_one)(p) for p in paths)
        np.save(f, np.vstack([v if v is not None else np.zeros(413, np.float32) for v in vecs]))
    feats["crop"] = np.load(f)

    cnn = None
    f = OUT / "cnn_vectors.npy"
    try:
        cnn = Cnn()
        if not f.exists():
            print("cnn: describing every photo with MobileNetV3...", flush=True)
            np.save(f, cnn_vectors(paths, cnn))
        feats["cnn"] = np.load(f)
        feats["cnn+current"] = np.hstack([feats["cnn"], feats["current"]])
    except ImportError:
        print("torch/torchvision not installed: skipping the ready-made network")

    results, models = {}, {}
    for name, X in feats.items():
        tr, va, te = split == "train", split == "val", split == "test"
        results[name], models[name] = fit_and_score(name, X[tr], y[tr], X[va], y[va], X[te], y[te], classes)
        print(f"  -> {name}: {results[name]}", flush=True)

    shots = screenshot_photos()
    if shots:
        for label, rgb in shots.items():
            vec = {"current": extract_features(rgb)[None], "crop": crop_vector(rgb)[None]}
            if cnn is not None:
                vec["cnn"] = cnn.vectors([rgb])
                vec["cnn+current"] = np.hstack([vec["cnn"], vec["current"]])
            for name, (sc, m) in models.items():
                p = m.predict_proba(sc.transform(vec[name]))[0]
                results[name].setdefault("your_screenshots", {})[label] = f"{m.classes_[p.argmax()]} {p.max():.0%}"

    lines = ["PEST CLASSIFIER FEATURE EXPERIMENT (test photos; real photos only, same SVM for all)", ""]
    head = f"{'method':12s} {'accuracy':>9s} {'macroF1':>8s} {'pests right':>12s} {'caterpillars':>13s} " \
           f"{'50%: names shown':>17s} {'...right':>9s}"
    lines.append(head)
    for name, r in results.items():
        lines.append(f"{name:12s} {r['test_accuracy']:9.1%} {r['test_macro_f1']:8.3f} {r['pest_photos_right']:12.1%} "
                     f"{r['caterpillars_right']:13.1%} {r['rule50_names_shown_share']:17.1%} {r['rule50_names_right']:9.1%}")
    lines.append("")
    for name, r in results.items():
        lines.append(f"{name}: by class " + ", ".join(f"{k} {v:.0%}" for k, v in r["by_class_right"].items()))
        if "your_screenshots" in r:
            lines.append(f"    your 3 screenshots: " + "; ".join(f"{k} -> {v}" for k, v in r["your_screenshots"].items()))
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
