"""Experiment: can a model trained ONLY on corn photos say "this isn't a corn photo"?

The corn check (train.py) learned corn vs other plants, so everyday things it never saw (a bowl,
wood, cloth) can land on the corn side. A one-class model learns only what our corn photos look
like and rejects photos too far from them. Nothing here changes the app; it only measures.

Methods compared (all on the same 406 photo features, standardized):
    ocsvm        One-Class SVM, RBF kernel
    ocsvm_pca    One-Class SVM after PCA (fewer, smoother features)
    knn          mean distance to the 5 nearest corn training photos (after PCA)
    iforest      Isolation Forest
Each learns from real corn TRAINING photos only (no augmented copies, no non-corn photos).
Its cutoff is set on corn VALIDATION photos so a chosen share of them is accepted (98% by default);
everything is then scored on the TEST photos, which no choice used.

Non-corn photos are used only to measure:
    ml/dataset/not_corn (test split)  rice, banana, tomato, cassava, cashew
    stand-ins                         the bowl from the app screenshot + made-up textures
    ml/dataset/test_random/           the user's own phone photos of random things, if present

Run from the project root:
    ml/.venv/Scripts/python.exe ml/src/one_class_experiment.py [--bowl path/to/screenshot.png]
Writes ml/results/one_class/summary.txt and results.json.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import decode_image, extract_features
from predict import predict_one

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
OUT = RESULTS / "one_class"
RANDOM_DIR = ROOT / "ml" / "dataset" / "test_random"
MODELS = ROOT / "backend" / "models"
SEED = 42
TRAIN_SIZE = 8000  # corn training photos per model (One-Class SVM time grows with the square)
ACCEPT = [0.95, 0.98, 0.99]  # share of corn validation photos the cutoff lets through
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def stand_ins(bowl_screenshot):
    """Photos of things that are not plants: the bowl from the app, plus made-up textures."""
    rng = np.random.default_rng(0)
    h, w = 480, 640

    def noisy(rgb, sd):
        return np.clip(np.full((h, w, 3), rgb, np.float32) + rng.normal(0, sd, (h, w, 3)), 0, 255).astype(np.uint8)

    def grain(rgb):  # wood-like stripes
        x = np.linspace(0, 40, w)[None, :] + rng.normal(0, 2, (h, 1)).cumsum(0) * 0.05
        s = (np.sin(x) * 25)[..., None]
        return np.clip(np.full((h, w, 3), rgb, np.float32) + s + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // 40) + (xx // 40)) % 2 * 200 + 30).astype(np.uint8)
    out = {
        "brown wood": grain((95, 62, 40)), "light wood": grain((190, 150, 105)),
        "blue cloth": noisy((40, 70, 160), 20), "red cloth": noisy((170, 40, 45), 20),
        "black cloth": noisy((30, 30, 32), 10), "grey concrete": noisy((140, 138, 135), 25),
        "white tiles": np.clip(noisy((225, 225, 220), 6) - (((yy % 120) < 3) | ((xx % 120) < 3))[..., None] * 80,
                               0, 255).astype(np.uint8),
        "skin tone": noisy((205, 160, 130), 8), "brown soil": noisy((110, 80, 55), 30),
        "checkerboard": np.repeat(checker[..., None], 3, 2),
        "sky gradient": np.dstack([np.linspace(120, 200, h)[:, None].repeat(w, 1)] * 2
                                  + [np.full((h, w), 235.0)]).astype(np.uint8),
    }
    if bowl_screenshot and Path(bowl_screenshot).exists():
        img = cv2.imread(str(bowl_screenshot))
        out["bowl (app screenshot)"] = cv2.cvtColor(img[200:610, 48:672], cv2.COLOR_BGR2RGB)
    return out


def random_photos():
    if not RANDOM_DIR.exists():
        return {}
    files = sorted(p for p in RANDOM_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    out = {}
    for p in files:
        try:
            out[str(p.relative_to(RANDOM_DIR))] = decode_image(np.fromfile(p, dtype=np.uint8).tobytes())
        except Exception as e:  # unreadable file: skip it, say so
            print(f"  skipped {p.name}: {e}")
    return out


def make_methods(Xtr):
    """Fit each method; return name -> score function (higher = more like our corn photos)."""
    methods = {}
    t = time.time()
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr)
    methods["ocsvm"] = (m.decision_function, {"support_vectors": int(m.support_vectors_.shape[0])})
    print(f"  ocsvm      {time.time() - t:.0f}s, {m.support_vectors_.shape[0]} support vectors", flush=True)

    pca = PCA(n_components=0.95, random_state=SEED).fit(Xtr)
    Ptr = pca.transform(Xtr)
    t = time.time()
    m2 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Ptr)
    methods["ocsvm_pca"] = (lambda X, m2=m2: m2.decision_function(pca.transform(X)),
                            {"support_vectors": int(m2.support_vectors_.shape[0]), "pca_dims": int(pca.n_components_)})
    print(f"  ocsvm_pca  {time.time() - t:.0f}s, {pca.n_components_} dims, {m2.support_vectors_.shape[0]} SVs", flush=True)

    nn = NearestNeighbors(n_neighbors=5).fit(Ptr)
    methods["knn"] = (lambda X: -nn.kneighbors(pca.transform(X))[0].mean(1), {"stored_photos": len(Ptr)})

    t = time.time()
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr)
    methods["iforest"] = (iso.decision_function, {})
    print(f"  iforest    {time.time() - t:.0f}s", flush=True)
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bowl", help="the app screenshot with the bowl (optional)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X = np.load(RESULTS / "features.npy")
    meta = pd.read_csv(RESULTS / "features_meta.csv")
    real = meta["aug"].isna()
    corn, other = meta["crop"] == "corn", meta["crop"] == "not_corn"
    tr_idx = meta.index[corn & real & (meta["split"] == "train")]
    tr_idx = np.random.default_rng(SEED).choice(tr_idx, min(TRAIN_SIZE, len(tr_idx)), replace=False)
    va, te_c, te_o = (meta.index[corn & real & (meta["split"] == "val")],
                      meta.index[corn & real & (meta["split"] == "test")],
                      meta.index[other & (meta["split"] == "test")])
    scaler = StandardScaler().fit(X[tr_idx])
    S = lambda rows: scaler.transform(X[rows])
    print(f"train {len(tr_idx)} corn photos | val {len(va)} corn | test {len(te_c)} corn, {len(te_o)} other plants")

    extra = {"stand-in": stand_ins(args.bowl), "your photos": random_photos()}
    extra_X = {g: (list(d), scaler.transform(np.array([extract_features(v) for v in d.values()])) if d else None)
               for g, d in extra.items()}

    # The current corn check on the same photos, to see what the one-class gate adds.
    cc = joblib.load(MODELS / "corn_check_model.joblib")
    cur = lambda Xraw: np.array([predict_one(cc, x)[0] == "corn" for x in Xraw])
    cur_te_c, cur_te_o = cur(X[te_c]), cur(X[te_o])
    cur_extra = {g: cur(np.array([extract_features(v) for v in d.values()])) if d else None for g, d in extra.items()}

    print("fitting...", flush=True)
    methods = make_methods(S(tr_idx))
    results, lines = {}, []
    sub = meta.loc[te_c, "pest_class"].values
    for name, (score, info) in methods.items():
        s_va, s_c, s_o = score(S(va)), score(S(te_c)), score(S(te_o))
        s_x = {g: score(v[1]) if v[1] is not None else None for g, v in extra_X.items()}
        r = {"info": info, "by_accept": {}}
        for a in ACCEPT:
            cut = float(np.quantile(s_va, 1 - a))
            ok_c, ok_o = s_c >= cut, s_o >= cut
            row = {"cutoff": cut,
                   "corn_test_accepted": float(ok_c.mean()),
                   "corn_test_accepted_by_class": {k: float(ok_c[sub == k].mean()) for k in sorted(set(sub))},
                   "other_plants_rejected": float(1 - ok_o.mean()),
                   "with_current_check": {"corn_test_accepted": float((ok_c & cur_te_c).mean()),
                                          "other_plants_rejected": float(1 - (ok_o & cur_te_o).mean())}}
            for g, sx in s_x.items():
                if sx is None:
                    continue
                okx = sx >= cut
                row[f"{g}_rejected"] = float(1 - okx.mean())
                row[f"{g}_rejected_with_current_check"] = float(1 - (okx & cur_extra[g]).mean())
                row[f"{g}_each"] = {n: ("accepted" if o else "rejected") for n, o in zip(extra_X[g][0], okx)}
            r["by_accept"][str(a)] = row
        results[name] = r

    lines.append("ONE-CLASS EXPERIMENT (trained on corn photos only)\n")
    lines.append(f"current corn check alone: corn test accepted {cur_te_c.mean():.1%}, "
                 f"other plants rejected {1 - cur_te_o.mean():.1%}, "
                 + ", ".join(f"{g} rejected {1 - v.mean():.0%}" for g, v in cur_extra.items() if v is not None))
    for name, r in results.items():
        lines.append(f"\n== {name} {r['info']}")
        for a, row in r["by_accept"].items():
            parts = [f"corn accepted {row['corn_test_accepted']:.1%}",
                     f"other plants rejected {row['other_plants_rejected']:.1%}"]
            parts += [f"{g} rejected {row[f'{g}_rejected']:.0%}" for g in extra if f"{g}_rejected" in row]
            both = row["with_current_check"]
            parts.append(f"| + current check: corn {both['corn_test_accepted']:.1%}, "
                         f"plants rej {both['other_plants_rejected']:.1%}"
                         + "".join(f", {g} rej {row[f'{g}_rejected_with_current_check']:.0%}"
                                   for g in extra if f"{g}_rejected" in row))
            lines.append(f"  accept {float(a):.0%} of val corn: " + ", ".join(parts[:-1]) + " " + parts[-1])
            low = {k: v for k, v in row["corn_test_accepted_by_class"].items() if v < float(a) - 0.03}
            if low:
                lines.append("     corn classes accepted less: " + ", ".join(f"{k} {v:.0%}" for k, v in low.items()))
            if "stand-in_each" in row:
                lines.append("     stand-ins: " + ", ".join(f"{k}={v[0]}" for k, v in row["stand-in_each"].items()))
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
