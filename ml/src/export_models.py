"""Export the four trained models for the phone app, so it can check photos offline.

The app (lib/offline/) runs the same math as scikit-learn: StandardScaler, then an RBF SVM
(libsvm one-vs-one), then probabilities (libsvm's pairwise method, or a sigmoid calibration
for the yes/no models), then the pest sub-class merge and cutoffs from predict.py.
This script first re-computes every model's probabilities with plain numpy, following those
steps exactly, and checks they match scikit-learn on validation photos. Then it writes the
numbers the app needs.

Run after train.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/export_models.py

Writes assets/models/<name>.json (settings) and <name>.bin (support vectors as float16, then
dual coefficients as float64, little-endian).

Half-size support vectors (25 Nov 2026, Version 2 with EfficientNet-B2): with 1408 numbers per
photo the support vectors were 130 MB as float32 and made the app about 165 MB. They are rounded to
float16 (about 3 significant digits) here, also in backend/models/*.joblib, so the backend, the
whole-app test and the phone keep giving exactly the same answers. The rounding changes each kernel
value by about 0.003%; the script prints how many test photos get a different answer (rounding an
already rounded model changes nothing, so running it again is safe).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from predict import apply_cutoffs, predict_proba as sklearn_proba

SV_DTYPE = "float16"  # how the support vectors are stored ("float32" keeps them exact)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "backend" / "models"
RESULTS_DIR = ROOT / "ml" / "results"
OUT = ROOT / "assets" / "models"
NAMES = {"corn_check": "corn_check", "detector": "corn_detector", "health": "corn_health", "pest": "corn_pest"}


def _sigmoid_predict(dec, a, b):
    """libsvm's sigmoid_predict: probability that the pair's first class is right."""
    f = dec * a + b
    return np.where(f >= 0, np.exp(-f) / (1.0 + np.exp(-f)), 1.0 / (1.0 + np.exp(f)))


def _multiclass_probability(r):
    """libsvm's multiclass_probability (pairwise coupling) for one sample; r[i][j] = P(i | i or j)."""
    k = r.shape[0]
    p = np.full(k, 1.0 / k)
    q = np.zeros((k, k))
    for t in range(k):
        for j in range(t):
            q[t, t] += r[j, t] * r[j, t]
            q[t, j] = q[j, t]
        for j in range(t + 1, k):
            q[t, t] += r[j, t] * r[j, t]
            q[t, j] = -r[j, t] * r[t, j]
    eps = 0.005 / k
    for _ in range(max(100, k)):
        qp = q @ p
        pqp = p @ qp
        if np.max(np.abs(qp - pqp)) < eps:
            break
        for t in range(k):
            diff = (-qp[t] + pqp) / q[t, t]
            p[t] += diff
            pqp = (pqp + diff * (diff * q[t, t] + 2 * qp[t])) / (1 + diff) / (1 + diff)
            qp = (qp + diff * q[t]) / (1 + diff)
            p /= 1 + diff
    return p


def svm_parts(svc):
    """libsvm's own numbers (scikit-learn's private, un-flipped copies)."""
    return {
        "gamma": float(svc._gamma),
        "n_support": [int(n) for n in svc._n_support],
        "dual_coef": np.asarray(svc._dual_coef_, dtype=np.float64),   # (k-1, n_SV)
        "intercept": np.asarray(svc._intercept_, dtype=np.float64),   # one per pair (i < j)
        "support_vectors": np.asarray(svc.support_vectors_, dtype=np.float64),
        "prob_a": np.asarray(getattr(svc, "_probA", []), dtype=np.float64),
        "prob_b": np.asarray(getattr(svc, "_probB", []), dtype=np.float64),
    }


def ovo_decisions(parts, x):
    """libsvm decision value for every pair (i, j), i < j, for one scaled sample x."""
    sv, coef, n_sv = parts["support_vectors"], parts["dual_coef"], parts["n_support"]
    kernel = np.exp(-parts["gamma"] * np.sum((sv - x) ** 2, axis=1))
    start = np.concatenate([[0], np.cumsum(n_sv)])
    k = len(n_sv)
    dec, p = [], 0
    for i in range(k):
        for j in range(i + 1, k):
            si, sj = slice(start[i], start[i + 1]), slice(start[j], start[j + 1])
            s = coef[j - 1, si] @ kernel[si] + coef[i, sj] @ kernel[sj]
            dec.append(s + parts["intercept"][p])
            p += 1
    return np.array(dec)


def numpy_proba(bundle, x_raw):
    """Probabilities in bundle['classes'] order, computed the way the app will."""
    scaler, model = bundle["scaler"], bundle["model"]
    x = (np.asarray(x_raw, dtype=np.float32) - scaler.mean_) / scaler.scale_
    x = x.astype(np.float32).astype(np.float64)  # scikit-learn scales float32 features in float32
    if isinstance(model, CalibratedClassifierCV):
        cc = model.calibrated_classifiers_[0]
        parts = svm_parts(cc.estimator)
        df = -ovo_decisions(parts, x)[0]  # scikit-learn's binary decision_function is -libsvm's
        cal = cc.calibrators[0]
        p1 = 1.0 / (1.0 + np.exp(cal.a_ * df + cal.b_))
        proba = np.array([1 - p1, p1])
    else:
        parts = svm_parts(model)
        dec = ovo_decisions(parts, x)
        k = len(parts["n_support"])
        r = np.zeros((k, k))
        p = 0
        for i in range(k):
            for j in range(i + 1, k):
                rij = float(np.clip(_sigmoid_predict(dec[p], parts["prob_a"][p], parts["prob_b"][p]), 1e-7, 1 - 1e-7))
                r[i, j], r[j, i] = rij, 1 - rij
                p += 1
        proba = _multiclass_probability(r)  # libsvm uses it for 2 classes too
    merge = bundle.get("merge") or {}
    out = np.zeros(len(bundle["classes"]))
    for j, c in enumerate(model.classes_):
        out[list(bundle["classes"]).index(merge.get(str(c), str(c)))] += proba[j]
    return out


def the_svc(bundle):
    model = bundle["model"]
    return model.calibrated_classifiers_[0].estimator if isinstance(model, CalibratedClassifierCV) else model


def round_support_vectors(bundle):
    """Rounds the SVM's support vectors to SV_DTYPE in place (scikit-learn predicts with
    support_vectors_). True if anything changed."""
    svc = the_svc(bundle)
    rounded = svc.support_vectors_.astype(SV_DTYPE).astype(np.float64)
    if np.array_equal(rounded, svc.support_vectors_):
        return False
    svc.support_vectors_ = rounded
    return True


def export(name, bundle):
    model = bundle["model"]
    calibrated = isinstance(model, CalibratedClassifierCV)
    svc = the_svc(bundle)
    if not isinstance(svc, SVC) or svc.kernel != "rbf":
        raise SystemExit(f"{name}: only RBF SVMs can be exported (got {type(svc).__name__}); retrain or extend the app")
    parts = svm_parts(svc)
    meta = {
        "name": name,
        "kind": "svm_sigmoid_calibrated" if calibrated else "svm_libsvm_probability",
        "feature_length": int(len(bundle["scaler"].mean_)),
        "features": bundle.get("features", "current"),  # V2: "cnn" = the image network's numbers
        "sv_dtype": SV_DTYPE,
        "scaler_mean": bundle["scaler"].mean_.tolist(),
        "scaler_scale": bundle["scaler"].scale_.tolist(),
        "model_classes": [str(c) for c in model.classes_],
        "classes": list(bundle["classes"]),
        "merge": bundle.get("merge") or {},
        "cutoffs": bundle.get("cutoffs") or {},
        "gamma": parts["gamma"],
        "n_support": parts["n_support"],
        "intercept": parts["intercept"].tolist(),
        "prob_a": parts["prob_a"].tolist(),
        "prob_b": parts["prob_b"].tolist(),
    }
    if calibrated:
        cal = model.calibrated_classifiers_[0].calibrators[0]
        meta["calibrator"] = {"a": float(cal.a_), "b": float(cal.b_)}
    sv = parts["support_vectors"].astype(SV_DTYPE)
    assert np.array_equal(sv.astype(np.float64), parts["support_vectors"]), f"support vectors are not {SV_DTYPE}-exact"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(OUT / f"{name}.bin", "wb") as f:
        f.write(sv.astype("<f2" if SV_DTYPE == "float16" else "<f4").tobytes())
        f.write(parts["dual_coef"].astype("<f8").tobytes())
    return sv.nbytes + parts["dual_coef"].nbytes


def main():
    cur, cnn = np.load(RESULTS_DIR / "features.npy"), np.load(RESULTS_DIR / "cnn_features.npy")
    sets = {"current": cur, "cnn": cnn, "cnn+current": np.hstack([cnn, cur]),
            "cnn_mirror": cnn}  # the SVM math check only needs numbers of the right length
    meta = pd.read_csv(RESULTS_DIR / "features_meta.csv", keep_default_na=False)
    val = meta.index[(meta["split"] == "val") & (meta["aug"] == "")].to_numpy()
    test = meta.index[(meta["split"] == "test") & (meta["aug"] == "")].to_numpy()
    sample = np.random.default_rng(0).choice(val, 200, replace=False)
    total = 0
    version = (MODELS_DIR / "model_version.txt").read_text(encoding="utf-8").strip()
    for key, name in NAMES.items():
        bundle = joblib.load(MODELS_DIR / f"{name}_model.joblib")
        X = sets[bundle.get("features", "current")]
        if SV_DTYPE == "float16":
            before = sklearn_proba(bundle, X[test])
            if round_support_vectors(bundle):
                after = sklearn_proba(bundle, X[test])
                cut = bundle.get("cutoffs")
                same = apply_cutoffs(before, bundle["classes"], cut) == apply_cutoffs(after, bundle["classes"], cut)
                print(f"{key:<10} support vectors rounded to float16: {len(test) - int(same.sum())} of {len(test)} "
                      f"test photos answered differently, largest probability change "
                      f"{float(np.abs(after - before).max()):.1e}", flush=True)
                joblib.dump(bundle, MODELS_DIR / f"{name}_model.joblib")
        ours = np.array([numpy_proba(bundle, X[i]) for i in sample])
        theirs = sklearn_proba(bundle, X[sample])
        diff = float(np.abs(ours - theirs).max())
        same = float((apply_cutoffs(ours, bundle["classes"], bundle.get("cutoffs"))
                      == apply_cutoffs(theirs, bundle["classes"], bundle.get("cutoffs"))).mean())
        print(f"{key:<10} numpy vs scikit-learn on 200 validation photos: max probability difference {diff:.2e}, "
              f"same answer {100 * same:.1f}%", flush=True)
        if diff > 1e-6 or same < 1:
            raise SystemExit(f"{key}: the app's math doesn't match scikit-learn; not exporting")
        total += export(key, bundle)
    (OUT / "model_version.txt").write_text(version + "\n", encoding="utf-8")
    if not (OUT / "network.bin").exists():
        print("note: assets/models/network.* missing - run network_export.py (the V2 models need it)")
    print(f"exported 4 models ({total / 1e6:.1f} MB of numbers), model version {version}, to {OUT}")


if __name__ == "__main__":
    main()
