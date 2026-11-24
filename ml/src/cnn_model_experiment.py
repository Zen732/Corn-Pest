"""Version 2 experiment: do the corn check, the insect detector and the healthy-vs-damaged model
also get better with the ready-made network's features (cnn_features.py) than with the
hand-crafted ones (features.py)? Nothing here changes the app; it only measures.

Each model uses the same photos as in train.py (real photos only), the same SVM search on the
validation photos (C 1 or 10, class weights none or balanced), and is scored once on the test
photos. The corn check is also scored on the 64 random phone photos (ml/dataset/test_random/,
never trained on) and the bowl screenshot (PEST_SCREENSHOTS folder, 6.png), at 50/60/70% "sure".

Run from the project root after cnn_features.py:
    ml/.venv/Scripts/python.exe ml/src/cnn_model_experiment.py
Writes ml/results/cnn_experiment/summary.txt and results.json.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cnn_features import Network
from features import decode_image, extract_features

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
OUT = RESULTS / "cnn_experiment"
RANDOM_DIR = ROOT / "ml" / "dataset" / "test_random"
SCREENSHOTS = Path(os.environ.get("PEST_SCREENSHOTS", "."))
LEVELS = (0.5, 0.6, 0.7)


def _one(Xtr, ytr, Xva, yva, C, cw):
    m = SVC(C=C, gamma="scale", class_weight=cw, cache_size=2000).fit(Xtr, ytr)
    return f1_score(yva, m.predict(Xva), average="macro"), C, cw, m


def fit(Xtr, ytr, Xva, yva, probability=True):
    """The 4 settings are tried side by side; probabilities (slow) only when asked for."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    tried = Parallel(n_jobs=4)(delayed(_one)(Xtr, ytr, Xva, yva, C, cw) for C in (1, 10) for cw in (None, "balanced"))
    for f1, C, cw, _ in tried:
        print(f"    C={C:<3} class_weight={str(cw):9s} val macro F1 {f1:.4f}", flush=True)
    best = max(tried, key=lambda t: t[0])
    m = best[3]
    if probability:
        m = SVC(C=best[1], gamma="scale", class_weight=best[2], probability=True, random_state=42,
                cache_size=3000).fit(Xtr, ytr)
    return sc, m, {"val_macro_f1": round(best[0], 4), "C": best[1], "class_weight": best[2]}


def group_of(path):
    return "everyday" if "/everyday/" in path else "ph_plants" if "/ph_plants/" in path else "other_crops"


def outside_photos():
    """Feature vectors (current, cnn) of the 64 random photos and the bowl, or None."""
    rgbs = {}
    if RANDOM_DIR.exists():
        for p in sorted(RANDOM_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rgbs[p.name] = decode_image(np.fromfile(str(p), dtype=np.uint8).tobytes())
    bowl = SCREENSHOTS / "6.png"
    if bowl.exists():
        rgbs["bowl"] = cv2.cvtColor(cv2.imread(str(bowl)), cv2.COLOR_BGR2RGB)[200:610, 48:672]
    if not rgbs:
        return None
    net = Network()
    names = list(rgbs)
    cur = np.array([extract_features(rgbs[n]) for n in names])
    cnn = np.vstack([net.vectors([rgbs[n]]) for n in names])
    return names, {"current": cur, "cnn": cnn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["corn_check", "detector", "health"])
    ap.add_argument("--both", action="store_true", help="also try the two feature sets side by side")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RESULTS / "features_meta.csv", keep_default_na=False)
    feats = {"current": np.load(RESULTS / "features.npy"), "cnn": np.load(RESULTS / "cnn_features.npy")}
    if args.both:
        feats["cnn+current"] = np.hstack([feats["cnn"], feats["current"]])
    assert all(len(f) == len(meta) for f in feats.values()), "run extract_all.py and cnn_features.py first"
    real, corn = meta["aug"] == "", meta["crop"] == "corn"
    meta["health"] = np.where(meta["pest_class"] == "healthy", "healthy", "damaged")
    tasks = {
        # other_insect photos (iNaturalist, mostly not on corn) are left out of the corn check
        "corn_check": (real & ((corn & (meta["pest_class"] != "other_insect")) | (meta["crop"] == "not_corn")), "crop"),
        "detector": (real & corn & (meta["pest_class"] != "healthy"), "bug_visible"),
        "health": (real & corn & (meta["pest_class"].isin(["healthy", "unidentified_damage"])
                                  | (meta["bug_visible"] == "no")), "health"),
    }
    if args.only:
        tasks = {k: v for k, v in tasks.items() if k in args.only}
    outside = outside_photos() if "corn_check" in tasks else None
    results, lines = {}, ["V2 EXPERIMENT: hand-crafted (current) vs ready-made network (cnn) features, same SVM", ""]
    for task, (rows, col) in tasks.items():
        sub = meta[rows]
        y, split = sub[col].values, sub["split"].values
        tr, va, te = split == "train", split == "val", split == "test"
        lines.append(f"== {task}: train {tr.sum()}, val {va.sum()}, test {te.sum()}")
        for name, X in feats.items():
            print(f"{task} / {name}", flush=True)
            if task == "corn_check" and name == "cnn+current":
                continue
            Xs = X[sub.index.values]
            sc, m, info = fit(Xs[tr], y[tr], Xs[va], y[va], probability=task == "corn_check")
            if task == "corn_check":
                proba = m.predict_proba(sc.transform(Xs[te]))
                pred = m.classes_[proba.argmax(1)]
            else:
                pred = m.predict(sc.transform(Xs[te]))
            r = {**info, "test_accuracy": round(accuracy_score(y[te], pred), 4),
                 "test_macro_f1": round(f1_score(y[te], pred, average="macro"), 4),
                 "by_class_right": {c: round(float((pred[y[te] == c] == c).mean()), 4) for c in m.classes_}}
            line = (f"  {name:8s} accuracy {r['test_accuracy']:.1%}, macro F1 {r['test_macro_f1']:.3f}; "
                    + ", ".join(f"{k} {v:.1%}" for k, v in r["by_class_right"].items()))
            if task == "corn_check":
                p_corn = proba[:, list(m.classes_).index("corn")]
                groups = sub["path"].values[te]
                is_corn = y[te] == "corn"
                r["levels"] = {}
                for lv in LEVELS:
                    row = {"corn_accepted": round(float((p_corn[is_corn] >= lv).mean()), 4)}
                    for g in ("other_crops", "ph_plants", "everyday"):
                        mask = (~is_corn) & np.array([group_of(p) == g for p in groups])
                        row[f"{g}_rejected"] = round(float((p_corn[mask] < lv).mean()), 4)
                    if outside:
                        names, ox = outside
                        po = m.predict_proba(sc.transform(ox[name]))[:, list(m.classes_).index("corn")]
                        rnd = np.array([n != "bowl" for n in names])
                        row["your_64_rejected"] = int((po[rnd] < lv).sum())
                        if not rnd.all():
                            row["bowl"] = f"{'rejected' if po[~rnd][0] < lv else 'accepted'} (corn {po[~rnd][0]:.0%})"
                    r["levels"][str(lv)] = row
                    line += (f"\n      need {lv:.0%}: corn accepted {row['corn_accepted']:.1%}, rejected: other crops "
                             f"{row['other_crops_rejected']:.1%}, PH plants {row['ph_plants_rejected']:.1%}, everyday "
                             f"{row['everyday_rejected']:.1%}"
                             + (f", your photos {row['your_64_rejected']}/64" if "your_64_rejected" in row else "")
                             + (f", bowl {row['bowl']}" if "bowl" in row else ""))
            results.setdefault(task, {})[name] = r
            lines.append(line)
            print(line, flush=True)
        lines.append("")
    text = "\n".join(lines)
    tag = "_".join(args.only) if args.only else "all"
    (OUT / f"summary_{tag}.txt").write_text(text, encoding="utf-8")
    (OUT / f"results_{tag}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
