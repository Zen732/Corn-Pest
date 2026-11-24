# Version 2 (ready-made image network + SVM)

Version 2 is the app from **23 November 2026** on (`main` branch). Version 1 (hand-crafted
features) is saved as the branch `version-1` / tag `v1.0`; see [version1.md](version1.md) to go
back to it.

Version 2 had two image networks:

- **23-24 Nov: MobileNetV3-Large** (960 numbers per photo, app 2.0.0).
- **From 25 Nov: EfficientNet-B2** (1408 numbers per photo, app 2.1.0). It is more accurate at
  every step (see "Which image network" below).

The MobileNetV3 version is kept in `_archive/1_old_models/version2_mobilenet_2026-11-24/`.

## What changed from Version 1

| | Version 1 | Version 2 |
|---|---|---|
| Photo numbers the models learn from | 406 hand-crafted features (colour, GLCM, LBP, HOG) | **1408 numbers from EfficientNet-B2**, a network already trained on ImageNet (not retrained), run on the phone in plain Dart (960 from MobileNetV3-Large, 23-24 Nov) |
| Models | 4 SVMs (+ K-Means severity) | the same 4 SVMs, retrained on the network's numbers (+ the same K-Means severity) |
| Unknown insects | only "less than 50% sure" said "Insect not recognized" | the pest classifier also learned **"other insect"** (5,026 photos of insects and spiders that are not one of the five pests); "other insect" is shown as "Insect not recognized" |
| Pest step | one look at the photo | one look (with MobileNetV3 the photo and its mirror image were averaged; with B2 that no longer helps) |
| Offline checker | models copied into a new background isolate for every scan | **one background isolate** loads the models once and keeps them |
| Model files | float32 | support vectors stored as **16-bit floats** (half the size, same answers) |

The decision rules are the same with two exceptions:

- The corn check needs **50%** instead of 60%, since it also learned close-ups of small objects
  (23 Nov 2026, see below).
- A pest is still named only if the pest classifier is 50% sure
  ([classification_flow.md](classification_flow.md)).

## Scores

### Test photos from the dataset (never used for training or any choice)

| What | Version 1 | V2 MobileNetV3 (23 Nov) | **V2 EfficientNet-B2 (now)** |
|---|---|---|---|
| Corn check (corn vs not corn) | 91.2% | 97.7% | **98.3%** |
| Pest classifier (5 pests + healthy + other insect) | 62.9% (5 pests + healthy) | 86.8% | **90.8%** |
| Insect detector | 96.0% | 98.4% | **98.7%** |
| Healthy vs damaged | 90.8% | 96.2% | **98.0%** |
| **Whole app, same 6,459 photos as Version 1** | **70.2%** | **90.2%** | **93.2%** |
| Whole app, all 7,453 test photos (incl. 754 other insects, 240 small objects) | | 90.1% | **93.1%** |
| When a pest name is shown, it is right | 69.5% | 89.1% | **91.9%** |
| Photos of the 5 pests: right name shown | 43.7% | 79.8% | **86.1%** |
| Other insects named as one of the 5 pests | (no such photos) | 12.9% | **9.5%** |
| Real corn photos turned away as "not corn" | 8.5% | 2.6% | **1.9%** |
| Other crops / Philippine plants / everyday scenes rejected as "not corn" | 93.4% / 76.3% / 89.6% | 99.2% / 92.5% / 100% | **99.3% / 94.1% / 99.7%** |
| Close-ups of small objects (rice grains, coffee beans, pebbles, shells) rejected | (not tested) | 97.1% | **99.6%** (all rice grains, coffee beans and pebbles; 98% of shells) |

The corn check row includes the small objects in its test set; before the small-object fix about
56% of small objects were rejected and 33% of single rice grains.

By pest (whole app, insect in view, right name shown), MobileNetV3 -> B2:

| Pest | MobileNetV3 | B2 |
|---|---|---|
| Fall armyworm | 78.8% | 84.8% |
| Common cutworm | 81.7% | 88.3% |
| Corn aphid | 84.9% | 89.9% |
| Armyworm | 67.0% | 76.8% |
| Corn planthopper | 80.1% | 84.6% |

The per-model scores are from `ml/results/metrics/`, the whole-app scores from
`ml/results/flow_summary.csv` / `flow_test.csv` (`ml/src/evaluate_flow.py`).

### Web test: 30 kinds of crop pests the app never saw (300 iNaturalist photos)

`dataset_sorter/fill_web_test.py` downloaded 10 recent photos of each kind;
`ml/src/web_test.py` checks them (`ml/results/web_test/summary.txt`).

- The 5 corn pests should be named.
- The 25 other pests (Asian corn borer, cotton bollworm, rice bug, fruit flies, whitefly,
  beetles, ...) should **not** be named as one of the five.
- None of these photos, and (from round 2 on) none of the 25 other species, were used for
  training.

| Whole app | Version 1 | V2 MobileNetV3 | **V2 EfficientNet-B2** |
|---|---|---|---|
| The 5 corn pests named right (50 photos) | 24% | 56% | **70%** |
| Other pests NOT named as a corn pest (250 photos) | 57% | 78% | **78%** |
| **All 300** | 51.7% | 74.7% | **76.7%** |

Version 1 is scored at its own 60% corn level here. `web_test.py` checks every version with
today's 50% rule, which gives Version 1 48.7%.

What changed with B2:

- **Better:** common cutworm 70 -> 100%, corn aphid 20 -> 50%, corn planthopper 40 -> 50%;
  green stink bug, rice bug and locust 80-90 -> 100% not misnamed; eggplant fruit borer 40 -> 70%.
- **Worse:** Asian corn borer 70 -> 40% not misnamed (its caterpillars and moths are taken for
  armyworms); black cutworm 20 -> 0% (a close relative of the pests, named common cutworm or
  armyworm); silverleaf whitefly 90 -> 70% (taken for corn aphids).

Notes for reading it honestly:

- Most of the newest web photos of the three caterpillar pests show the **adult moth**, not the
  caterpillar on corn.
- 19 of the 20 corn aphid and corn planthopper photos are "needs ID" on iNaturalist (not yet
  confirmed by others), and several are blurry or far away. At least one "aphid" photo seems to
  show orange spots rather than aphids. Their scores say little: on the dataset's test photos B2
  names aphids right 90% of the time and planthoppers 85%.
- Still hard: moths and caterpillars that look almost the same as the pests' (black cutworm,
  Asian corn borer, cotton bollworm, diamondback moth) and spider mites.
- 29 photos (1-3 each) of 18 of the 25 other species were already in the first "other insect"
  download, before the web test existed; round 2 left those species out.

### Your own photos

| | Version 1 | V2 MobileNetV3 | **V2 EfficientNet-B2** |
|---|---|---|---|
| 64 random phone photos (not corn) rejected as "not corn" | 36 of 64 | 60 of 64 | **61 of 64** (3 pass at 52-63% "corn": see below) |
| The bowl screenshot | "Insect not recognized" | "Doesn't look like a corn plant" (66% sure) | **"Doesn't look like a corn plant"** (81% sure) |
| A grain of rice (phone, 23 Nov) | | "Insect not recognized" before the small-object fix | all 87 test rice grains rejected |
| The beetle on a corn leaf | "Corn aphid, 94%" | "Insect not recognized" | **"Insect not recognized"** |
| Screenshot of a blue-green aphid (Version 1 showed "Corn planthopper, 40%") | | "Corn aphid" (99%) | **"Corn aphid"** (100%) |
| Screenshot of a curled caterpillar (Version 1 showed "Corn aphid, 74%") | | "Armyworm" (70%) | "Armyworm" (73%) |

The three random photos B2 lets through:

- a variegated leaf plant, named "common cutworm";
- a pink flower, named "corn aphid";
- a floor with slippers and a kitten, shown as "Insect not recognized".

The curled caterpillar looks like a cutworm, but its species can't be told from the screenshot.

## How it was improved

### 23 Nov 2026 (MobileNetV3-Large)

1. **Measured first** (`ml/src/cnn_model_experiment.py`): with the network's numbers every model
   got better (corn check 91 -> 98%, detector 96 -> 98%, healthy-vs-damaged 91 -> 97%; pest 63 ->
   88% in `pest_feature_experiment.py`). Adding the hand-crafted features next to them gained only
   0.2-0.3 points, so Version 2 uses the network's numbers alone (faster on the phone).
2. **"Other insect", round 1:** 3,326 iNaturalist photos (caterpillars in Asia, Philippine
   beetles, true bugs, grasshoppers, other insects, spiders), leaving out the five pests' genera
   and all aphids and planthoppers (`dataset_sorter/fill_other_insects.py`).
3. **Retrained all four models** (SVM only, C 1/3/10/30, chosen on validation photos).
4. **Web test** showed other moths named armyworm / fall armyworm and whiteflies / mites named
   corn aphid.
5. **"Other insect", round 2:** +1,000 adult moths (Asia), +400 whiteflies / mealybugs / scale
   insects / thrips / psyllids, +300 leafhoppers. 150 mite photos were downloaded, then archived
   (mostly leaf galls, no animal in view; `_archive/2_unused_photos/other_insect_mites_leaf_galls/`).
   The pest classifier and the detector were retrained. Web test 68.0% -> 74.7%; dataset test
   about the same (90.5% -> 90.2% on the 6,459 photos).
6. **Mirror image** for the pest step (`ml/src/flip_experiment.py`): +0.5 points on validation,
   +0.1 on test. Not used for the other models (+0.1 or less).
7. **Small objects** (after a grain of rice passed, 23 Nov):
   - Added 1,600 "not corn" close-ups (rice grains, coffee beans, pebbles, shells;
     `dataset_sorter/fill_small_objects.py`) and retrained the corn check.
   - At 60% it would now turn away more pest close-ups (92.9% kept on validation), so the level
     went to **50%** (the user's choice).
   - At 50%, pest close-ups and corn are accepted as before (96.1% / 97.5%) and 97.9% of small
     objects are rejected (was 56.7%).
   - The old corn check is kept in
     `_archive/1_old_models/version2_corn_check_before_small_objects_2026-11-23/`.

### 24-25 Nov 2026 (EfficientNet-B2)

8. **Which image network** (`ml/src/backbone_experiment.py`, `ml/results/backbones/`). The same
   pest photos and the same SVM were tried with four ready-made networks (plain SVM, no 50% rule):

   | Network | Numbers per photo | PC time per photo (PyTorch) | Pest test accuracy | Web: all 300 |
   |---|---|---|---|---|
   | MobileNetV3-Large | 960 | 21 ms | 86.2% | 70.3% |
   | EfficientNet-B0 | 1280 | 29 ms | 88.9% | 72.7% |
   | **EfficientNet-B2** | 1408 | 52 ms | **91.0%** | **74.7%** |
   | DINOv2 ViT-S/14 | 768 | 67 ms | 93.3% | 76.0% |

   B2 was chosen. DINOv2 is a little more accurate still, but it is a different kind of network
   (a vision transformer) and would make each scan about 3 times slower than B2 on the phone.
9. **Photo numbers for all 55,590 photos** with B2 (`ml/src/cnn_features.py`, `NETWORK =
   "efficientnet_b2"`, 288 x 288 input; 25 minutes) and **all four models retrained** (about
   1.5 hours with 3 SVMs at a time; 4 left this laptop with 0.2 GB of free memory).
   Every model improved on test photos (table above).
10. **Mirror image dropped:** with B2 it adds only +0.05 points on validation (2 of 3,995 photos)
    and -0.1 on test, and it would double the phone's time for insect photos. With B2 no model
    uses it (`train.PREDICT_WITH = {}`).
11. **Half-size model files:** with 1408 numbers per photo the support vectors were 132 MB and
    the APK about 165 MB. `export_models.py` now stores them as 16-bit floats, and rounds the
    backend's models the same way so every part gives the same answers. This changed 0 of 7,453
    test answers in each model (largest probability change 0.1%). The model files are 66 MB.
12. **Corn check level** checked again: at 50% the B2 corn check is better than MobileNetV3's
    in every group, so the user's 50% was kept (table in
    [classification_flow.md](classification_flow.md)).

## How it runs on the phone

`lib/offline/network.dart` runs EfficientNet-B2 in plain Dart (no plugin), four channels at a
time (Float32x4). The weights come from `ml/src/network_export.py`: batch-norm is folded into the
convolutions, and the results are `assets/models/network.json` + `network.bin` (31 MB). The same
code still runs MobileNetV3: the layer list says which activations to use.

Checks:

- Against PyTorch on 24 photos: largest difference 2.4e-5 of the largest value.
- The numpy copy for the backend (`ml/src/network.py`, no PyTorch needed there): 2.3e-6.

**Speed.** B2 needs about 5 times the arithmetic of MobileNetV3, so the two slowest steps were
rewritten for speed (24 Nov), making them about 1.7 times faster:

- The 1 x 1 convolution does 4 pixels x 8 channels at a time, with the running sums in local
  variables.
- The depthwise convolution keeps its sums in local variables.

The sums are added in the same order as before, so the numbers did not change. Compiled as on
the phone (`dart compile exe`), one B2 pass takes **0.29 s on this PC** (MobileNetV3 with the
same code: 0.055 s). A phone may need about 1-1.5 s per scan, and there is one pass per scan
(MobileNetV3 needed two for insect photos). Still to be tried on a real phone.

The whole phone path (Flutter's decoder, Dart network, Dart SVMs, Dart decision) was compared
with Python on 1,003 test photos (25 Nov):

- 1,003 of 1,003 gave the same outcome and the same pest.
- Severity grades matched on 300 of 301.

See [offline_checking.md](offline_checking.md).

**Size:** the model files are 93 MB (SVMs 66 MB, network 31 MB). The arm64 APK is **108 MB**
(MobileNetV3 build 124 MB, Version 1 72 MB).

## How to retrain Version 2

From the project root, after adding or changing photos:

```
ml/.venv/Scripts/python.exe ml/src/make_split.py
ml/.venv/Scripts/python.exe ml/src/extract_all.py --reuse
ml/.venv/Scripts/python.exe ml/src/cnn_features.py          (network numbers; reuses earlier ones)
ml/.venv/Scripts/python.exe ml/src/train.py                 (or --only corn_pest, ...; about 1.5 hours)
ml/.venv/Scripts/python.exe ml/src/export_models.py         (also rounds the support vectors to float16)
ml/.venv/Scripts/python.exe ml/src/evaluate_flow.py         (whole-app test)
ml/.venv/Scripts/python.exe ml/src/web_test.py              (web test, your photos; V1, V2 MobileNetV3, V2 B2)
ml/.venv/Scripts/python.exe ml/src/parity_reference.py && flutter test
backend/.venv/Scripts/python.exe backend/sync_shared.py
flutter build apk --release --split-per-abi --dart-define-from-file=env.json
```

Notes:

- `network_export.py` only needs running again if the network itself changes (`NETWORK` in
  `cnn_features.py`).
- PyTorch is only needed on the training PC (`ml/.venv`).
- Keep the number of parallel workers low (`JOBS = 3` in `train.py`, 6 in `evaluate_flow.py` and
  `parity_reference.py`): too many crashed this laptop once.

**Going back to MobileNetV3:** copy
`_archive/1_old_models/version2_mobilenet_2026-11-24/backend_models/*` to `backend/models/` and
`app_assets_models/*` to `assets/models/`. Those are float32 files; the app still reads them. To
retrain with it:

1. Set `NETWORK = "mobilenet_v3_large"` in `cnn_features.py`.
2. Set `PREDICT_WITH = {"corn_pest": "cnn_mirror"}` in `train.py`.
3. Put its photo numbers back (`photo_numbers/` in that folder), or run `cnn_features.py` again.

## What Version 2 still needs

1. **Step 10: photos from Cabadbaran corn fields** (`ml/dataset/test_local/`). Every score above
   is on internet photos.
2. **Try it on a real phone** for speed and memory (about 1-1.5 s per scan expected).
3. **Look-alike moths and caterpillars** (black cutworm, Asian corn borer). The next step up
   would be one of these:
   - DINOv2 (93.3% on the pest step), if a phone can run it fast enough;
   - fine-tuning the network on the pest photos (Version 3);
   - more photos of those species as "other insect".
4. **"Was this right?" button**, publishing items and backups: as in [version1.md](version1.md)
   (items 6-9).
