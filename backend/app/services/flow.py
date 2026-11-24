"""The app's decision for one photo. Shared with the backend like features.py, severity.py and
predict.py (backend/app/services/flow.py must be an exact copy), so the whole-app test
(evaluate_flow.py) and the app give the same answers.

    Is it a corn plant?  -> Is there an insect?  yes -> which pest?   (pest name only)
                                                 no  -> healthy or damaged? -> damage grade

    not corn (corn check < CORN_AT_LEAST sure)     -> "not_corn"       "doesn't look like a corn plant"
    corn, insect, pest model >= PEST_AT_LEAST sure -> "pest"           pest name only, no severity
    corn, insect, less sure or "other_insect"      -> "unknown_pest"   "insect not recognized"
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

Version 2 (23 Nov 2026): each model says which photo numbers it learned from (its bundle's
"features": "cnn" = the image network's numbers, network.py, 1408 for EfficientNet-B2;
"current" = the hand-crafted features.py vector; or both side by side). Each set is computed
once per photo, only if needed.
An insect that is not one of the five pests is named "other_insect" by the pest classifier and
shown as "unknown_pest" ("Insect not recognized"), like a pest the classifier is unsure of.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import extract_features
from network import NumpyNetwork
from predict import predict_one
from severity import assess

# The pest classifier must be at least this sure to name a pest; below it the app says the insect
# is not recognized (it may not be one of the five pests). On test photos with one of the five
# pests in view, 33% are under 50% and only 41% of those were named right (22 Nov 2026).
PEST_AT_LEAST = 0.50

# The corn check must be at least this sure (probability of "corn") to go on; below it the photo
# is "not_corn". Chosen on validation photos:
# - 22 Nov 2026 (Version 1), after real-world "not corn" photos were added: 50% -> 60% kept 92% of
#   corn photos while rejecting 80% of Philippine garden plants and 90% of everyday scenes.
# - 23 Nov 2026 (Version 2), after close-ups of small objects (rice grains, coffee beans, pebbles,
#   shells) were added because a grain of rice passed: 60% -> 50% (the user's choice). At 50% the
#   new corn check accepts pest close-ups and corn like the old one at 60% (96.1% / 97.5%) and
#   rejects 97.9% of small objects (old: 56.7%); at 60% it would accept only 92.9% of pest close-ups.
# - 25 Nov 2026 (EfficientNet-B2): kept at 50%. The new corn check accepts 97.7% of pest close-ups
#   and 98.5% of corn and rejects 98.3% of small objects and 94.9% of Philippine plants.
CORN_AT_LEAST = 0.50

MODEL_FILES = {"corn_check": "corn_check_model.joblib",
               "detector": "corn_detector_model.joblib",
               "pest": "corn_pest_model.joblib",
               "health": "corn_health_model.joblib"}


# The pest classifier's answer for "an insect, but not one of the five pests".
OTHER_INSECT = "other_insect"


def load_models(folder):
    models = {key: joblib.load(Path(folder) / name) for key, name in MODEL_FILES.items()}
    if any("cnn" in b.get("features", "current") for b in models.values()):
        models["network"] = NumpyNetwork(folder)
    return models


def photo_numbers(rgb, models, given=None):
    """numbers(bundle) -> the photo numbers a model is given, each set computed once:
        "cnn"          the image network's numbers (1408 for EfficientNet-B2)
        "cnn_mirror"   the average of "cnn" for the photo and for its mirror image (left-right);
                       used by the pest classifier (+0.7 points on validation photos, 23 Nov 2026)
        "current"      the hand-made features (features.py)
        "a+b"          two sets side by side
    `given`: already computed sets, {"cnn": ..., "current": ...} (or one hand-crafted vector); a
    dict passed here also receives the sets computed, so a caller can reuse them."""
    cache = given if isinstance(given, dict) else ({} if given is None else {"current": given})

    def one(k):
        if k not in cache:
            if k == "cnn":
                cache[k] = models["network"].vector(rgb)
            elif k == "cnn_mirror":
                cache[k] = (one("cnn") + models["network"].vector(cv2.flip(rgb, 1))) / np.float32(2)
            else:
                cache[k] = extract_features(rgb)
        return cache[k]

    def numbers(bundle):
        kind = bundle.get("features", "current")
        if kind not in cache:
            cache[kind] = np.concatenate([one(k) for k in kind.split("+")])
        return cache[kind]
    return numbers


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
    (see photo_numbers); severity is always measured from the photo."""
    x = photo_numbers(rgb, models, features)
    _, corn_conf, corn_probs = predict_one(models["corn_check"], x(models["corn_check"]))
    result = {"outcome": "not_corn", "corn_confidence": round(corn_conf, 3), "insect_visible": None,
              "insect_confidence": None, "pest": None, "pest_confidence": None,
              "healthy": None, "health_confidence": None, "damage_percent": None, "severity": None}
    if corn_probs.get("corn", 0.0) < CORN_AT_LEAST:
        return result

    insect, insect_conf, _ = predict_one(models["detector"], x(models["detector"]))
    insect_visible = insect == "yes"
    result.update(insect_confidence=round(insect_conf, 3))
    if insect_visible:
        pest, pest_conf, _ = predict_one(models["pest"], x(models["pest"]))
        if pest == "healthy":  # detector and pest classifier disagree: trust "no insect"
            insect_visible = False
        else:
            # An insect: the pest's name only, no damage grade. Another insect, or not sure enough:
            # "not recognized" (pest_confidence is then how sure the classifier was of its answer).
            sure = pest != OTHER_INSECT and pest_conf >= PEST_AT_LEAST
            result.update(outcome=decide(True, True, sure, None, None), insect_visible=True,
                          pest=pest if sure else None, pest_confidence=round(pest_conf, 3))
            return result

    health, health_conf, _ = predict_one(models["health"], x(models["health"]))
    healthy = health == "healthy"
    sev = assess(rgb)
    level = "Healthy" if healthy else ("Low" if sev["level"] == "Healthy" else sev["level"])
    result.update(outcome=decide(True, False, False, healthy, sev["damage_percent"]), insect_visible=False,
                  healthy=healthy, health_confidence=round(health_conf, 3),
                  damage_percent=sev["damage_percent"], severity=level)
    return result
