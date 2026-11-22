"""The app's decision for one photo. Shared with the backend like features.py, severity.py and
predict.py (backend/app/services/flow.py must be an exact copy), so the whole-app test
(evaluate_flow.py) and the app give the same answers.

    Is it a corn plant?  -> Is there an insect?  yes -> which pest?   (pest name only)
                                                 no  -> healthy or damaged? -> damage grade

    not corn (corn check < CORN_AT_LEAST sure)     -> "not_corn"       "doesn't look like a corn plant"
    corn, insect, pest model >= PEST_AT_LEAST sure -> "pest"           pest name only, no severity
    corn, insect, pest model less sure             -> "unknown_pest"   "insect not recognized"
    corn, no insect, health model says healthy     -> "healthy"
    corn, no insect, damaged                       -> "severity_only"  Low / Moderate / High
    corn, no insect, damaged, plant not found      -> "unclear"        ask for a clearer photo

A photo with an insect gets the pest's name and nothing else: no damage grade (22 Nov 2026; on
insect close-ups the "damage" was mostly the insect and the background). If the pest classifier
answers "healthy", the detector was probably wrong, and the photo is treated as "no insect".
Healthy vs damaged is decided by a trained model, not by the damage % (which counts soil and
weeds in field photos as damage); the K-Means damage % then grades the damage, at least "Low".

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

# The pest classifier must be at least this sure to name a pest; below it the app says the insect
# is not recognized (it may not be one of the five pests). On test photos with one of the five
# pests in view, 33% are under 50% and only 41% of those were named right (22 Nov 2026).
PEST_AT_LEAST = 0.50

# The corn check must be at least this sure (probability of "corn") to go on; below it the photo
# is "not_corn". Chosen on validation photos (22 Nov 2026), after real-world "not corn" photos
# were added: 50% -> 60% kept 92% of corn photos while rejecting 80% of Philippine garden plants
# and 90% of everyday scenes (was 71% / 84%); 70% would turn away 1 in 7 real corn photos.
CORN_AT_LEAST = 0.60

MODEL_FILES = {"corn_check": "corn_check_model.joblib",
               "detector": "corn_detector_model.joblib",
               "pest": "corn_pest_model.joblib",
               "health": "corn_health_model.joblib"}


def load_models(folder):
    return {key: joblib.load(Path(folder) / name) for key, name in MODEL_FILES.items()}


def decide(is_corn, insect_visible, pest_sure, healthy, damage_percent):
    """The outcome rule on its own (see the table at the top)."""
    if not is_corn:
        return "not_corn"
    if insect_visible:
        return "pest" if pest_sure else "unknown_pest"
    if healthy:
        return "healthy"
    return "unclear" if damage_percent is None else "severity_only"


def assess_photo(rgb, models, features=None):
    """Everything the app shows for one photo. `features` may be passed if already computed
    (extract_features(rgb)); severity is always measured from the photo."""
    x = extract_features(rgb) if features is None else features
    _, corn_conf, corn_probs = predict_one(models["corn_check"], x)
    result = {"outcome": "not_corn", "corn_confidence": round(corn_conf, 3), "insect_visible": None,
              "insect_confidence": None, "pest": None, "pest_confidence": None,
              "healthy": None, "health_confidence": None, "damage_percent": None, "severity": None}
    if corn_probs.get("corn", 0.0) < CORN_AT_LEAST:
        return result

    insect, insect_conf, _ = predict_one(models["detector"], x)
    insect_visible = insect == "yes"
    result.update(insect_confidence=round(insect_conf, 3))
    if insect_visible:
        pest, pest_conf, _ = predict_one(models["pest"], x)
        if pest == "healthy":  # detector and pest classifier disagree: trust "no insect"
            insect_visible = False
        else:
            # An insect: the pest's name only, no damage grade.
            sure = pest_conf >= PEST_AT_LEAST
            result.update(outcome=decide(True, True, sure, None, None), insect_visible=True,
                          pest=pest if sure else None, pest_confidence=round(pest_conf, 3))
            return result

    health, health_conf, _ = predict_one(models["health"], x)
    healthy = health == "healthy"
    sev = assess(rgb)
    level = "Healthy" if healthy else ("Low" if sev["level"] == "Healthy" else sev["level"])
    result.update(outcome=decide(True, False, False, healthy, sev["damage_percent"]), insect_visible=False,
                  healthy=healthy, health_confidence=round(health_conf, 3),
                  damage_percent=sev["damage_percent"], severity=level)
    return result
