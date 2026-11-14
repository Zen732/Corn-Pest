"""Train the four corn models: the corn check, the pest classifier, the bug-visible detector,
and the healthy-vs-damaged model (for photos with no insect in view).

Uses the fixed train / val / test sets made by make_split.py (70 / 15 / 15, ml/dataset/split/).
Run after make_split.py and extract_all.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/train.py
    ml/.venv/Scripts/python.exe ml/src/train.py --only corn_check    (just one model)

For each model:
    train  Random Forest, SVM and KNN, each with a few settings, learn from these photos.
    val    picks the best model and settings (macro F1). For the pest classifier it also picks
           per-class confidence cutoffs for the classes padded with augmented copies (see
           predict.py).
    test   scored once at the end, with the chosen model. Never used for any choice, so the
           test score is an honest estimate for new photos.
The saved model is the one trained on the training set.

The pest classifier learns only from photos with the insect in view (bug_visible = yes), plus
healthy; damage-only photos train the detector and the corn check.

Augmented copies (only in split/train/) are used by the pest classifier only. The corn check
and the detector use real photos: padding the small pest folders would only add more "insect
visible" photos and tilt the detector's yes/no balance. Val and test are real photos only.

Saves:
    backend/models/corn_check_model.joblib       {"model", "scaler", "classes", "cutoffs"}
    backend/models/corn_pest_model.joblib
    backend/models/corn_detector_model.joblib
    backend/models/corn_health_model.joblib
    ml/results/metrics/<name>.json                full metrics, for evaluate.py
    ml/results/metrics_summary.csv                one row per model
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from predict import apply_cutoffs, merge_proba

warnings.filterwarnings("ignore", category=FutureWarning)  # scikit-learn deprecation notices

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"
RANDOM_STATE = 42

CROP = "corn"
NOT_CORN = "not_corn"  # other plants (rice, banana, ...) in ml/dataset/not_corn/, for the corn check

# Folders that hold real photos but are NOT a pest classifier output: "unidentified_damage"
# is generic damage with no known pest, used for the detector ("no bug visible") and the
# corn check. See docs/classification_flow.md.
EXCLUDE_FROM_PEST_TASK = {"unidentified_damage"}

# class_weight is searched, not fixed: on an earlier dataset, "balanced" made a small class
# over-predicted (precision 0.38) while None did better (0.52).
MODEL_GRID = {
    "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                      {"n_estimators": [200, 400], "max_depth": [None, 20], "class_weight": ["balanced", None]}),
    # No probability=True here: comparing settings only needs predictions. The winner gets
    # probabilities in finalize(). cache_size (MB) speeds up kernel computations.
    "svm": (SVC(kernel="rbf", cache_size=1000, random_state=RANDOM_STATE),
            {"C": [1, 10], "gamma": ["scale"], "class_weight": ["balanced", None]}),
    "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}),
}
CUTOFF_GRID = np.round(np.arange(0.20, 0.81, 0.05), 2)


def load_data():
    X = np.load(RESULTS_DIR / "features.npy")
    meta = pd.read_csv(RESULTS_DIR / "features_meta.csv", keep_default_na=False)
    assert len(X) == len(meta), "features.npy and features_meta.csv are out of sync - rerun extract_all.py"
    if "split" not in meta:
        raise SystemExit("features_meta.csv has no split column - run make_split.py, then extract_all.py")
    # Safety check: photos of one iNaturalist observation (inat_<id>_p2.jpg ...) must share a set.
    group = meta["path"].str.replace(r"(inat_\d+)_p\d+(\.\w+)$", r"\1\2", regex=True)
    leaks = int((meta.groupby(group)["split"].nunique() > 1).sum())
    if leaks:
        raise SystemExit(f"{leaks} photo group(s) are in more than one set - rerun make_split.py")
    meta["health"] = np.where(meta["pest_class"] == "healthy", "healthy", "damaged")
    # Caterpillar pests are about half adult moths, which look nothing like their caterpillars:
    # the pest classifier learns them as separate sub-classes (fall_armyworm_moth, ...) and adds
    # them back up when it answers. +1.3 points macro F1 on validation (14 Nov 2026).
    meta["pest_sub"] = np.where(meta["path"].isin(adult_moth_paths()), meta["pest_class"] + "_moth",
                                meta["pest_class"])
    return X, meta


def adult_moth_paths():
    """Adult moth photos in the caterpillar-pest folders ("corn/<class>/<file>"): iNaturalist
    photos credited as adults, and moths moved back from the reserve (add_moths.py, filter_moths.py)."""
    paths = set()
    with open(DATASET_DIR / "inaturalist_credits.csv", newline="", encoding="utf-8") as f:
        paths |= {r["file"].replace("\\", "/") for r in csv.DictReader(f) if "adult" in r.get("species", "")}
    with open(DATASET_DIR / "reserve" / "corn" / "moth_moves.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            moved_to = r["moved_to"].replace("\\", "/")
            if "/raw/" in f"/{moved_to}/" and "reserve" not in moved_to:
                paths.add(f"corn/{r['class']}/{r['file']}")
    return paths


def sets(meta, rows, augment):
    """Row numbers of the train / val / test photos among `rows` (a boolean mask)."""
    real = meta["aug"] == ""
    pick = lambda s, extra=False: meta.index[rows & (meta["split"] == s) & (real | extra)].to_numpy()
    return pick("train", augment), pick("val"), pick("test")


def _merged_f1(y_true, y_pred, merge):
    """Macro F1 after mapping sub-classes to their class (fall_armyworm_moth -> fall_armyworm)."""
    to_class = np.vectorize(lambda v: merge.get(v, v), otypes=[object])
    return f1_score(to_class(y_true), to_class(y_pred), average="macro")


def select_best_model(X_train, y_train, X_val, y_val, merge=None):
    """Every model and setting is trained on the training set and scored (macro F1, on the
    final classes when the labels are sub-classes) on the validation set; the best one is
    retrained on the training set and returned."""
    X_both = np.vstack([X_train, X_val])
    y_both = np.concatenate([y_train, y_val])
    fold = PredefinedSplit(np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)])
    scoring = make_scorer(_merged_f1, merge=merge) if merge else "f1_macro"
    best, val_scores = None, {}
    for name, (estimator, grid) in MODEL_GRID.items():
        search = GridSearchCV(estimator, grid, cv=fold, scoring=scoring, n_jobs=-1, refit=False)
        search.fit(X_both, y_both)
        val_scores[name] = round(float(search.best_score_), 4)
        print(f"      {name:<14} val f1_macro={search.best_score_:.4f}  params={search.best_params_}", flush=True)
        if best is None or search.best_score_ > best[2]:
            best = (name, clone(estimator).set_params(**search.best_params_), search.best_score_)
    name, model, _ = best
    return name, finalize(name, model, len(np.unique(y_train))).fit(X_train, y_train), val_scores


def finalize(name, model, n_classes):
    """The winning model, set up for its final fit. The SVM needs probabilities:
    - yes/no models: 5-fold Platt scaling (sigmoid) with the folds run in parallel, fast;
    - models with more classes (the pest classifier): SVC(probability=True), whose pairwise
      method is slower (one core) but more accurate there: the parallel one-vs-rest
      calibration lowered the pest classifier's validation F1 from 0.669 to 0.657 (14 Nov 2026)."""
    if name == "svm":
        if n_classes > 2:
            return model.set_params(probability=True)
        return CalibratedClassifierCV(model, method="sigmoid", cv=5, ensemble=False, n_jobs=-1)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=-1)
    return model


def tune_cutoffs(proba_val, y_val, classes, candidates):
    """Per-class confidence cutoffs for `candidates` (classes padded with augmented copies),
    chosen on the validation photos (never the test set). A cutoff is kept only if it raises
    macro F1 on the validation set."""
    score = lambda cuts: f1_score(y_val, apply_cutoffs(proba_val, classes, cuts), average="macro")
    cutoffs = {}
    for cls in candidates:
        best_cut, best = None, score(cutoffs)
        for cut in CUTOFF_GRID:
            s = score({**cutoffs, cls: float(cut)})
            if s > best + 1e-9:
                best_cut, best = float(cut), s
        if best_cut is not None:
            cutoffs[cls] = best_cut
    return cutoffs


def scores(y_true, y_pred):
    return {"accuracy": round(float((y_pred == y_true).mean()), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4)}


def train_task(X, meta, rows, y_col, name, augment, cutoffs_for_short=False, sub_col=None):
    """sub_col: a column of finer labels to train on (e.g. pest_sub: fall_armyworm and
    fall_armyworm_moth); predictions are added back up into y_col's classes."""
    train_idx, val_idx, test_idx = sets(meta, rows, augment)
    y_train, y_val, y_test = (meta.loc[i, y_col].to_numpy() for i in (train_idx, val_idx, test_idx))
    fit_col = sub_col or y_col
    merge = {s: c for s, c in zip(meta.loc[rows, fit_col], meta.loc[rows, y_col]) if s != c}
    n_aug = int((meta.loc[train_idx, "aug"] != "").sum())
    print(f"    train {len(train_idx)} (incl. {n_aug} augmented copies), val {len(val_idx)}, test {len(test_idx)}"
          + (f"; sub-classes {sorted(merge)}" if merge else ""), flush=True)

    scaler = StandardScaler().fit(X[train_idx])
    X_train, X_val, X_test = (scaler.transform(X[i]) for i in (train_idx, val_idx, test_idx))
    best_name, model, val_scores = select_best_model(
        X_train, meta.loc[train_idx, fit_col].to_numpy(), X_val, meta.loc[val_idx, fit_col].to_numpy(), merge)
    classes = sorted({merge.get(str(c), str(c)) for c in model.classes_})
    proba_val, proba_test = (merge_proba(model.predict_proba(x), model.classes_, classes, merge)
                             for x in (X_val, X_test))

    cutoffs = {}
    if cutoffs_for_short and n_aug:
        short = sorted(set(meta.loc[train_idx[meta.loc[train_idx, "aug"].to_numpy() != ""], y_col]))
        cutoffs = tune_cutoffs(proba_val, y_val, classes, short)
        if cutoffs:
            print(f"      cutoffs (chosen on val): {cutoffs}", flush=True)
    y_plain = apply_cutoffs(proba_test, classes, {})
    y_pred = apply_cutoffs(proba_test, classes, cutoffs)
    test = scores(y_test, y_pred)

    real = meta.loc[np.concatenate([train_idx, val_idx, test_idx])]
    real = real[real["aug"] == ""]
    metrics = {
        "task": name,
        "best_model": best_name,
        "best_params": {k: v for k, v in (model.estimator if isinstance(model, CalibratedClassifierCV)
                                          else model).get_params().items()
                        if k in ("n_estimators", "max_depth", "class_weight", "C", "n_neighbors")},
        "val_f1_macro": val_scores,
        "cutoffs": cutoffs,
        "merge": merge,
        "trained_on": [str(c) for c in model.classes_],
        "val": scores(y_val, apply_cutoffs(proba_val, classes, cutoffs)),
        "without_cutoffs": {"test_accuracy": scores(y_test, y_plain)["accuracy"],
                            "test_f1_macro": scores(y_test, y_plain)["f1_macro"]},
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        "classes": classes,
        "n_train_original": int(len(train_idx) - n_aug),
        "n_train_augmented": n_aug,
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "class_counts": {c: {s: int(((real[y_col] == c) & (real["split"] == s)).sum()) for s in ("train", "val", "test")}
                         for c in classes},
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
    }
    return model, scaler, classes, metrics, train_idx, test_idx, y_pred


def save(name, model, scaler, classes, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "classes": classes, "cutoffs": metrics["cutoffs"],
                 "merge": metrics.get("merge") or {}},
                MODELS_DIR / f"{name}_model.joblib")
    with open(METRICS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"    saved {MODELS_DIR / (name + '_model.joblib')}  val f1 {metrics['val']['f1_macro']:.3f}, "
          f"test acc {metrics['test_accuracy']:.3f}, test f1 {metrics['test_f1_macro']:.3f}", flush=True)


def background_check(X, meta, train_idx, test_idx, y_pred):
    """Is the corn check learning backgrounds? Two tests:
    1. accuracy on test photos grouped by background type;
    2. a model that sees ONLY the photo's border colour - if it tells corn from not-corn well,
       backgrounds alone give the answer away and the real model may be using that."""
    test = meta.loc[test_idx].assign(pred=y_pred)
    by_background = {bg: {"n": int(len(g)), "accuracy": round(float((g["pred"] == g["crop"]).mean()), 4)}
                     for bg, g in test.groupby("background")}
    mix = (meta.loc[np.concatenate([train_idx, test_idx])]
           .groupby("crop")["background"].value_counts(normalize=True).round(3).unstack(fill_value=0))
    cols = ["border_v", "border_s"]
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(meta.loc[train_idx, cols].astype(float), meta.loc[train_idx, "crop"])
    border_only = float((rf.predict(meta.loc[test_idx, cols].astype(float)) == meta.loc[test_idx, "crop"]).mean())
    return {"accuracy_by_background": by_background,
            "background_share_per_crop": mix.to_dict(orient="index"),
            "border_colour_only_accuracy": round(border_only, 4),
            "chance_accuracy": round(1 / meta["crop"].nunique(), 4)}


MODEL_NAMES = ["corn_check", f"{CROP}_pest", f"{CROP}_detector", f"{CROP}_health"]


def summary_row(metrics):
    return {"model": metrics["task"], "best_model": metrics["best_model"], "val_f1_macro": metrics["val"]["f1_macro"],
            **{k: metrics[k] for k in ("test_accuracy", "test_f1_macro", "n_train_original",
                                       "n_train_augmented", "n_val", "n_test")}}


def main():
    ap = argparse.ArgumentParser(description="Train the corn models (see the top of this file).")
    ap.add_argument("--only", choices=MODEL_NAMES, help="train just this model; the others keep their saved files")
    only = ap.parse_args().only
    X, meta = load_data()
    in_crop = meta["crop"] == CROP

    # Corn check: every corn photo (healthy, diseased, damaged, and close-ups of the five pests,
    # which farmers photograph too) against other plants. Without the pest photos it rejected
    # three of four pest photos as "not corn" (whole-app test, 11 Nov 2026).
    if only in (None, "corn_check"):
        print("\n=== corn check (corn vs not corn), all corn photos incl. pest close-ups ===", flush=True)
        rows = in_crop | (meta["crop"] == NOT_CORN)
        model, scaler, classes, metrics, tr, te, pred = train_task(X, meta, rows, "crop", "corn_check", augment=False)
        metrics["background_check"] = background_check(X, meta, tr, te, pred)
        save("corn_check", model, scaler, classes, metrics)

    # Only photos with the insect in view, plus healthy (as designed): in the app this model runs
    # only after the detector says "insect visible". Damage-only photos (mostly fall armyworm
    # field shots) would teach it "chewed corn field = fall armyworm".
    if only in (None, f"{CROP}_pest"):
        print(f"\n=== {CROP}: pest classifier (insect-visible photos + healthy) ===", flush=True)
        rows = in_crop & ~meta["pest_class"].isin(EXCLUDE_FROM_PEST_TASK) & (
            (meta["bug_visible"] == "yes") | (meta["pest_class"] == "healthy"))
        model, scaler, classes, metrics, *_ = train_task(X, meta, rows, "pest_class", f"{CROP}_pest", augment=True,
                                                         cutoffs_for_short=True, sub_col="pest_sub")
        save(f"{CROP}_pest", model, scaler, classes, metrics)

    if only in (None, f"{CROP}_detector"):
        print(f"\n=== {CROP}: bug-visible detector (damaged photos only) ===", flush=True)
        rows = in_crop & (meta["pest_class"] != "healthy")
        model, scaler, classes, metrics, *_ = train_task(X, meta, rows, "bug_visible", f"{CROP}_detector",
                                                         augment=False)
        save(f"{CROP}_detector", model, scaler, classes, metrics)

    # Healthy vs damaged, for photos with no insect in view. Replaced "damage % under 5% =
    # healthy": on field photos the damage % counts soil and weeds, and that rule got 56% of
    # validation photos right; this model got 91% (14 Nov 2026). K-Means still grades the damage.
    if only in (None, f"{CROP}_health"):
        print(f"\n=== {CROP}: healthy vs damaged (photos with no insect in view) ===", flush=True)
        rows = in_crop & (meta["pest_class"].isin(["healthy", "unidentified_damage"]) | (meta["bug_visible"] == "no"))
        model, scaler, classes, metrics, *_ = train_task(X, meta, rows, "health", f"{CROP}_health", augment=False)
        save(f"{CROP}_health", model, scaler, classes, metrics)

    # Summary of every saved model (also those not retrained this run).
    table = pd.DataFrame([summary_row(json.load(open(METRICS_DIR / f"{n}.json", encoding="utf-8")))
                          for n in MODEL_NAMES if (METRICS_DIR / f"{n}.json").exists()])
    table.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("\n" + table.to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
