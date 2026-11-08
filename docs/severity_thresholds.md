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

## Limitations (state these in the paper)

- **Thresholds** are placeholders, and photo-based severity measures *how much of this
  photo is damaged*, not field infestation as DA guidelines do.
- **Corn ears** are not supported: yellow kernels are healthy but read as damage.
- **Soil, mud, water and dry straw** in the photo count as damage. The app's photo tips ask
  users to fill the frame with the plant.
- **Mature plants.** Dry, brown mature leaves read as damage. Photograph green, growing plants.
- **Lighting and colour casts.** Olive-looking healthy leaves under warm light can be
  partly marked as yellowing. A looser green rule fixed some healthy photos but missed more
  real yellowing, so the stricter rule was kept: a false "damaged" is safer than a false
  "Healthy" for a pest app.
- **Coloured backgrounds** that aren't white, grey, black or sky count as damage.
- **Grade boundaries.** A photo near 20% or 40% can land on either side between two photos
  of the same plant.
