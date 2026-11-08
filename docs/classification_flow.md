# Classification Flow

How one scan of a corn photo becomes a result, and why each model exists.

## Step 0: is this a corn plant?

Right after the photo is taken, a `corn_check` model (`backend/models/corn_check_model.joblib`,
see `train.py`) decides whether the photo shows a corn plant.

- Corn, with reasonable confidence: continue to the pipeline below.
- Not corn, or low confidence: stop and show a message ("This doesn't look like a corn
  plant - check your photo.") instead of forcing a result.

It is trained on corn plant photos (healthy, diseased, and damage with no insect in view)
against 2,000 rice and banana leaf photos in `ml/dataset/not_corn/`.

**Limitation to state in the paper:** the only "not corn" examples are rice and banana
leaves. A photo of something unrelated (a person, a wall, a different plant) was never
seen in training, so the check may still call it corn. A true "is this any known plant"
rejector would need a much wider set of unrelated photos.

## The three outcomes

Every scan that passes the corn check ends in exactly one of these. Nothing else is shown:
no treatment advice, no product names, no "what to do" text, anywhere in the app.

| Outcome | When | What the app shows |
|---|---|---|
| **Healthy** | No damage detected on the plant | "Healthy" |
| **Severity only** | Damage is visible, but no insect is visible in the photo | Severity: Low / Moderate / High (+ damage %) |
| **Pest + Severity** | Damage is visible AND an insect is visible in the photo | Pest name + confidence + Severity |

## Pipeline (per scan)

```
photo -> features.py -> corn_check model
  -> not corn, or low confidence ---> outcome = Message ("doesn't look like corn"), stop
  -> corn ---> continue:

  1. Severity module (K-Means damage %)
       -> damage % below the "healthy" cutoff?  ---yes---> outcome = Healthy, stop
       -> no: keep the damage % and Low/Moderate/High label

  2. Insect detector (binary classifier: bug_visible yes / no)
       -> no  ---> outcome = Severity only (from step 1)
       -> yes ---> pest classifier (fall armyworm / corn planthopper / corn aphid / armyworm /
                   common cutworm / healthy)
                ---> outcome = Pest name + Severity (from step 1)
```

## Why a detector, not just classifier confidence

We could have skipped a separate model and used "low confidence in the pest
classifier -> treat as no insect visible." We didn't, because an unsure-but-correct
call would then wrongly hide a real pest name. A dedicated binary model (insect visible
or not), using the same `features.py` vectors, needs its own labels but is more reliable.

## Severity is photo-only

Severity is estimated from the photo alone: the share of the plant's area that is
damaged (see [severity_thresholds.md](severity_thresholds.md)). For fall armyworm (ragged
holes in whorl leaves) and Asian corn borer (pinholes, stalk holes), that damage is
visible in a photo.

**Limitation to state in the paper:** DA field guidelines measure infestation across a
field (for example, the share of plants with larvae). A single photo measures how much of
*this plant part* is damaged. They are related but not the same, so field thresholds
don't transfer directly to photo damage %.

## Labeling for the detector

`ml/dataset/raw/corn/<class>/` encodes the pest class, not whether an insect is visible in
each photo. Two photos in `fall_armyworm/` might show the caterpillar, or only its damage.
Both are correct for that class, but need different detector labels.

Labels are kept in `ml/dataset/labels.csv` (path, crop, pest_class, bug_visible), filled
in with `ml/src/label_bug_visible.py`: it shows one photo at a time and you press a key
(insect visible / damage only / skip). `healthy/` photos are auto-labeled `no`.

### `unidentified_damage`: extra "no insect visible" examples

Pest datasets are mostly photos of the insect itself, so "damage only" examples are hard to
find pre-labeled. `ml/dataset/raw/corn/unidentified_damage/` holds diseased or damaged corn
leaves (blight, rust, leaf spot, streak virus) with no insect in view.

It is **not** a pest and is excluded from the pest classifier (`train.py`'s
`EXCLUDE_FROM_PEST_TASK`). It **is** used by the detector, labeled `no`, and by the corn
check as corn plant photos.

## What this removes from the original plan

- Rice and banana (project narrowed to corn, 15 October 2026).
- Crop selection screen: there is only one crop.
- `backend/app/services/recommend.py` and any recommendations table or seed data.
- Field-question screen; no "recommendation" or "what to do" text on the result screen.
