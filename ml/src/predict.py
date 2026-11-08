"""Make a prediction with a saved model bundle, applying its per-class confidence cutoffs.

Shared with the backend like features.py and severity.py (backend/app/services/predict.py
must be an exact copy), so training-time evaluation and the app give the same answers.

    import joblib
    from predict import predict_one
    bundle = joblib.load("backend/models/corn_pest_model.joblib")
    label, confidence, probabilities = predict_one(bundle, feature_vector)

A bundle is {"model", "scaler", "classes", "cutoffs"}. "cutoffs" maps a class to the minimum
probability the model needs before that class may be the answer, e.g. {"asian_corn_borer": 0.40}:
below it, the next most likely class is given instead. Cutoffs are set by train.py for
classes with few real photos, where an uncut model raises too many false alarms.
"""
import numpy as np


def apply_cutoffs(proba, classes, cutoffs):
    """proba: (n, n_classes) probabilities in `classes` order. Returns predicted labels."""
    proba = np.array(proba, dtype=float, copy=True)
    for cls, cut in (cutoffs or {}).items():
        if cls in classes:
            j = list(classes).index(cls)
            proba[proba[:, j] < cut, j] = -1.0
    return np.asarray(classes)[proba.argmax(axis=1)]


def predict_proba(bundle, X):
    """Probabilities for raw (unscaled) feature vectors X, columns in bundle["classes"] order."""
    model = bundle["model"]
    proba = model.predict_proba(bundle["scaler"].transform(np.atleast_2d(X)))
    order = [list(model.classes_).index(c) for c in bundle["classes"]]
    return proba[:, order]


def predict(bundle, X):
    return apply_cutoffs(predict_proba(bundle, X), bundle["classes"], bundle.get("cutoffs"))


def predict_one(bundle, x):
    """Label, its probability, and all class probabilities for one feature vector."""
    proba = predict_proba(bundle, x)[0]
    label = apply_cutoffs(proba[None, :], bundle["classes"], bundle.get("cutoffs"))[0]
    probabilities = {c: float(p) for c, p in zip(bundle["classes"], proba)}
    return str(label), probabilities[label], probabilities
