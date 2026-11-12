# Severity: Method, Thresholds and Limitations

Code: [ml/src/severity.py](../ml/src/severity.py). Checks: [ml/src/check_severity.py](../ml/src/check_severity.py).

## Method

Severity is measured from the photo alone; nothing is trained. For each corn photo:

1. **Remove the background.** White or grey tables, black backgrounds and sky are removed,
   but only where they touch the photo's edge. Dark lesions or pale spots *inside* the
   plant are kept.
2. **K-Means clustering.** The remaining plant pixels are grouped into 4 colour clusters
   in Lab colour space (fitted on a sample of 20,000 pixels, then applied to all).
3. **Colour rule.** Each cluster is called healthy or damaged: green (Lab hue 105-220) is
   healthy; yellow, brown, black and grey/white dead tissue is damage.
4. **Holes**: enclosed background-coloured areas, such as holes chewed by caterpillars,
   count as damage and are reported separately.
5. **Edge band:** a 2-pixel band along the plant's outline is left out, because those
   pixels blend leaf and background colours.
6. **Damage %** = damaged area / whole plant area, then graded:

| Healthy | Low | Moderate | High |
|---|---|---|---|
| < 5% | 5-20% | 21-40% | > 40% |

These bands are **placeholders**: no photo-based severity standard for corn was found.
The 5% "Healthy" cutoff is a project choice: below it, small marks from lighting or dirt
are not reported as damage. Final bands should be justified with expert input (for
example the Cabadbaran City Agriculture Office) or field validation.

## Results

**Synthetic photos with a known damage %** (`check_severity.py`): measured within about
3 percentage points for brown spots, yellowing and chewed holes, on white or black backgrounds.

**Real corn photos**, 100 random per class (14 October 2026):

| Class | Median damage | Graded Healthy | Low | Moderate | High |
|---|---|---|---|---|---|
| corn/healthy | 1.3% | 82 | 13 | 1 | 4 |
| corn/unidentified_damage | 20.9% | 21 | 26 | 33 | 20 |

Healthy vs damaged was right for 80.5% of these 200 photos. Some "Healthy" grades on
diseased photos may be genuinely tiny lesions under 5%.

**Validation photos of the current dataset** (11 November 2026, 1,718 corn photos with no
insect in view; share graded correctly at the 5% cutoff):

| Photos | Healthy read as Healthy | Damaged read as damaged |
|---|---|---|
| Close-up single leaves (PlantVillage) | 86% (median 1.2%) | 86% (median 22.4%) |
| Field photos with soil and weeds around the leaf (Doctor Maiz) | 34% (median 10.4%) | 60% (median 12.0%) |
| Field photos of fall armyworm damage (AgML, Roboflow) | - | 82-83% (median about 25%) |

On close-up leaves the method works. On field photos the leaf itself is measured correctly,
but soil, weeds and sunny patches around it are counted as damage, so healthy and damaged
Doctor Maiz photos both measure about 10%; no cutoff separates them (best: 61% at 15%).

**Two fixes were tried and not kept** (same validation photos, balanced accuracy at 5%):

| Method | Overall | Healthy field photos | Damaged field photos | Healthy close-ups |
|---|---|---|---|---|
| Current (kept) | 56% | 34% | 60% | 86% |
| Measure only the central 60% of the photo | 66% | 64% | 58% | 77% |
| Cut out the main leaf first (GrabCut) | 66% | 85% | 46% | 62% |

GrabCut removed the background well, but also cut away brown and dead parts of damaged
leaves (median damage on damaged field photos fell from 12% to 3%) and did worse on
close-ups. Neither fix clearly helped, so the photo tips ("fill the frame with one leaf")
are the main defence against background, and this goes in the limitations.

## Limitations (state these in the paper)

- **Thresholds** are placeholders, and photo-based severity measures *how much of this
  photo is damaged*, not field infestation as DA guidelines do.
- **Corn ears** are not supported: yellow kernels are healthy but read as damage.
- **Soil, mud, water, weeds and dry straw** in the photo count as damage. On field photos
  with background, a healthy leaf typically measures about 10% (graded "Low"). The app's photo
  tips ask users to fill the frame with one leaf. Leaf segmentation (GrabCut) was tested and
  not kept, because it also removed real damage (see Results).
- **Mature plants.** Dry, brown mature leaves read as damage. Photograph green, growing plants.
- **Lighting and colour casts.** Olive-looking healthy leaves under warm light can be
  partly marked as yellowing. A looser green rule fixed some healthy photos but missed more
  real yellowing, so the stricter rule was kept: a false "damaged" is safer than a false
  "Healthy" for a pest app.
- **Coloured backgrounds** that aren't white, grey, black or sky count as damage.
- **Grade boundaries.** A photo near 20% or 40% can land on either side between two photos
  of the same plant.
