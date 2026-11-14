# Classification Flow

How one scan of a corn photo becomes a result, and why each model exists.

## Step 0: is this a corn plant?

Right after the photo is taken, a `corn_check` model (`backend/models/corn_check_model.joblib`,
see `train.py`) decides whether the photo shows a corn plant.

- Corn, with reasonable confidence: continue to the pipeline below.
- Not corn, or low confidence: stop and show a message ("This doesn't look like a corn
  plant - check your photo.") instead of forcing a result.

It is trained on every corn photo (healthy, diseased, damaged, and close-ups of the five
pests) against 10,000 photos of other plants in `ml/dataset/not_corn/` (rice, banana,
tomato, cassava, cashew). The pest close-ups were added on 11 November 2026: trained on corn
plant photos only, it rejected three of four pest photos as "not corn", because a close-up
of a caterpillar or aphid doesn't look like a corn plant.

**Limitations to state in the paper:** the only "not corn" examples are other plants. A
photo of something unrelated (a person, a wall, a different insect) was never seen in
training, so the check may still call it corn, and an insect from another crop would then
be named as one of the five corn pests. A true "is this any known plant" rejector would
need a much wider set of unrelated photos.

## The three outcomes

Every scan that passes the corn check ends in exactly one of these. Nothing else is shown:
no treatment advice, no product names, no "what to do" text, anywhere in the app.

| Outcome | When | What the app shows |
|---|---|---|
| **Healthy** | No insect visible, and the healthy-vs-damaged model says healthy | "Healthy" |
| **Severity only** | No insect visible, and the model says damaged | Severity: Low / Moderate / High (+ damage %) |
| **Pest + Severity** | An insect is visible in the photo (any damage level) | Pest name + confidence + Severity (at least "Low") |

## Pipeline (per scan)

Code: [ml/src/flow.py](../ml/src/flow.py) (the backend uses an exact copy).

```
photo -> features.py -> corn_check model
  -> not corn ---> outcome = Message ("doesn't look like corn"), stop
  -> corn ---> continue:

  1. Insect detector (binary classifier: bug_visible yes / no)
       -> yes ---> pest classifier (fall armyworm / corn planthopper / corn aphid / armyworm /
                   common cutworm / healthy)
                     -> a pest  ---> outcome = Pest name + Severity (step 2), stop
                     -> healthy ---> the detector was probably wrong: treat as "no insect"

  2. Healthy-vs-damaged model (binary classifier, same features)
       -> healthy ---> outcome = Healthy
       -> damaged ---> step 3

  3. Severity module (K-Means damage %)
       -> plant not found ---> outcome = Unclear (ask for a clearer photo)
       -> otherwise       ---> outcome = Severity only (Low/Moderate/High, at least Low)
```

**Healthy vs damaged is decided by a trained model, not by the damage %** (changed 14 November
2026). The first rule was "damage under 5% = Healthy", but the K-Means damage % counts soil,
weeds and sunny patches around the leaf as damage, so on field photos healthy and diseased
leaves both measured about 10%. On the validation photos the 5% rule was right for 56% and the
model for 91% (92% of healthy and 87% of diseased Doctor Maiz field photos). The model is trained
like the others (Random Forest, SVM and KNN compared on the validation set) on the corn photos
with no insect in view: `healthy/` against `unidentified_damage/` and pest-folder photos labeled
"damage only". K-Means still measures how much is damaged, for the Low / Moderate / High grade.

The insect check comes **before** the healthy cutoff (changed 11 November 2026). The first
version decided "Healthy" first, so a pest on a barely damaged plant was never named: in the
whole-app test, 7-19% of pest photos (aphids the most) stopped at "Healthy". With a pest
present the severity is shown as at least "Low", never "Healthy".

## Why a detector, not just classifier confidence

We could have skipped a separate model and used "low confidence in the pest
classifier -> treat as no insect visible." We didn't, because an unsure-but-correct
call would then wrongly hide a real pest name. A dedicated binary model (insect visible
or not), using the same `features.py` vectors, needs its own labels but is more reliable.

## The pest classifier learns from insect photos only

The pest classifier only runs after the detector says "insect visible", so it is trained
only on photos labeled `bug_visible = yes` (plus `healthy/`). Damage-only photos in the pest
folders (1,484, mostly fall armyworm field photos from AgML and Roboflow) are left out of it.
Otherwise it could learn "chewed corn field = fall armyworm" and pull other pests toward
that answer. They still train the detector (as "no") and the corn check.

## Caterpillars and moths are learned separately

About half of each caterpillar pest's photos are adult moths (fall armyworm 2,000, common
cutworm 2,142, armyworm 1,182), which look nothing like the caterpillars. The pest classifier
learns them as separate sub-classes (`fall_armyworm_moth`, ...) and adds the two back up when
it answers, so the app still says "Fall armyworm". On the validation photos this raised macro
F1 from 0.670 to 0.676; on the test set it scored the same as before (0.668), so the gain is
small.

**Tried and not kept (validation photos, 14 November 2026):**

| Pest classifier change | Validation macro F1 | Caterpillar pests right |
|---|---|---|
| None (current features, SVM) | 0.667 | 54.8% |
| Stronger SVM settings (C = 30 or 100) | 0.645-0.646 | 51.9% |
| Add dense-SIFT "bag of visual words" (256 words, 1x1 + 2x2 grid) | 0.653 | 53.1% |
| Dense-SIFT visual words alone | 0.541 | 39.9% |

SIFT patches of caterpillar skin, moth wings and leaf look alike across the three caterpillar
species, so counting them did not separate the pests. Telling these caterpillars apart reliably
probably needs a deep-learning model or a photo where the head and tail markings are visible;
this is stated as a limitation.

## Whole-app test

`ml/src/evaluate_flow.py` runs every test photo (5,709, never used for any choice) through
the full decision in `flow.py` and checks what the app would show. Results are in
`ml/results/flow_test.csv` and `flow_summary.csv`.

| Kind of test photo | Should show | 11 Nov (first) | 12 Nov | 14 Nov (now) |
|---|---|---|---|---|
| Fall armyworm (insect in view) | fall armyworm | 12.2% | 50.5% | 49.0% |
| Common cutworm | common cutworm | 11.1% | 59.4% | 58.1% |
| Corn aphid | corn aphid | 12.4% | 75.5% | 74.9% |
| Armyworm | armyworm | 11.8% | 45.4% | 46.1% |
| Corn planthopper | corn planthopper | 3.9% | 63.1% | 69.9% |
| Damaged corn, no insect | Severity only | 59.6% | 60.2% | 84.9% |
| Healthy corn | Healthy | 41.8% | 41.5% | 83.0% |
| Other plants | Not corn | 96.4% | 90.8% | 90.8% |
| **All test photos** | | **45.6%** | **65.7%** | **75.4%** |

What changed:
- 12 Nov: the insect check comes before "Healthy", and the corn check also learned from pest
  close-ups (it had rejected three of four pest photos as "not corn"). The cost: the corn
  check now lets through 9% of other plants (was 4%), most of which are then named as a pest
  or given a severity.
- 14 Nov: "Healthy" is decided by the healthy-vs-damaged model instead of "damage under 5%",
  and the pest classifier learns caterpillars and moths separately.

Remaining weak spots, for the paper:
- **Caterpillar pest names** (46-58%): fall armyworm, common cutworm and armyworm are confused
  with each other most. Stronger SVM settings and SIFT features were tried and did not help
  (see "Caterpillars and moths are learned separately").
- **Other plants let through** (9%): the price of accepting pest close-ups as corn.
- **Healthy corn shown as damaged** (13%) and **damaged shown as healthy** (6%).

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
