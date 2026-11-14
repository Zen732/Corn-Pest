"""The app's decision for one photo. Shared with the backend like features.py, severity.py and
predict.py (backend/app/services/flow.py must be an exact copy), so the whole-app test
(evaluate_flow.py) and the app give the same answers.

    photo -> corn check -> insect detector -> healthy-vs-damaged model -> severity -> outcome

    not corn                                       -> "not_corn"       "doesn't look like a corn plant"
    corn, insect visible                           -> "pest"           pest name + severity
    corn, no insect, health model says healthy     -> "healthy"
    corn, no insect, damaged                       -> "severity_only"  Low / Moderate / High
    corn, no insect, damaged, plant not found      -> "unclear"        ask for a clearer photo

The insect check comes before "healthy", so a pest on a barely damaged plant is still named
(see docs/classification_flow.md). If the pest classifier answers "healthy", the detector was
probably wrong, and the photo is treated as "no insect". Healthy vs damaged is decided by a
trained model, not by the damage % (which counts soil and weeds in field photos as damage);
the K-Means damage % then grades the damage, at least "Low".

    from flow import assess_photo, load_models
    models = load_models("backend/models")
    result = assess_photo(rgb, models)        # rgb: uint8 array (H, W, 3)
    result["outcome"], result["pest"], result["severity"]
"""
from pathlib import Path

import joblib

from features import extract_features
from predict import predict_one
from severity import assess

MODEL_FILES = {"corn_check": "corn_check_model.joblib",
               "detector": "corn_detector_model.joblib",
               "pest": "corn_pest_model.joblib",
               "health": "corn_health_model.joblib"}


def load_models(folder):
    return {key: joblib.load(Path(folder) / name) for key, name in MODEL_FILES.items()}


def decide(is_corn, insect_visible, healthy, damage_percent):
    """The outcome rule on its own (see the table at the top)."""
    if not is_corn:
        return "not_corn"
    if insect_visible:
        return "pest"
    if healthy:
        return "healthy"
    return "unclear" if damage_percent is None else "severity_only"


def assess_photo(rgb, models, features=None):
    """Everything the app shows for one photo. `features` may be passed if already computed
    (extract_features(rgb)); severity is always measured from the photo."""
    x = extract_features(rgb) if features is None else features
    corn, corn_conf, _ = predict_one(models["corn_check"], x)
    result = {"outcome": "not_corn", "corn_confidence": round(corn_conf, 3), "insect_visible": None,
              "insect_confidence": None, "pest": None, "pest_confidence": None,
              "healthy": None, "health_confidence": None, "damage_percent": None, "severity": None}
    if corn != "corn":
        return result

    insect, insect_conf, _ = predict_one(models["detector"], x)
    insect_visible = insect == "yes"
    if insect_visible:
        pest, pest_conf, _ = predict_one(models["pest"], x)
        if pest == "healthy":  # detector and pest classifier disagree: trust "no insect"
            insect_visible = False
        else:
            result.update(pest=pest, pest_confidence=round(pest_conf, 3))

    healthy = None
    if not insect_visible:
        health, health_conf, _ = predict_one(models["health"], x)
        healthy = health == "healthy"
        result.update(healthy=healthy, health_confidence=round(health_conf, 3))

    sev = assess(rgb)
    level = sev["level"]
    if healthy:
        level = "Healthy"
    elif level == "Healthy":
        level = "Low"  # damaged or a pest present: never graded "Healthy"
    result.update(outcome=decide(True, insect_visible, healthy, sev["damage_percent"]),
                  insect_visible=insect_visible, insect_confidence=round(insect_conf, 3),
                  damage_percent=sev["damage_percent"], severity=level)
    return result
