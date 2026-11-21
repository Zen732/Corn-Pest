# Offline checking (on the phone)

The app checks photos **on the phone itself**, with no internet and no server. The four trained
models and the whole decision (corn check, insect detector, pest classifier or healthy-vs-damaged,
severity) were ported from Python to Dart in [lib/offline/](../lib/offline/).

The backend ([backend/](../backend/)) still runs the original Python and is used to test and
compare. Only creating an account and the first sign-in need internet; after that the app
stays signed in offline.

## History works offline too

Results are saved **on the phone first** in the app's private folder: the photo, plus a small
file with the result ([lib/services/history_store.dart](../lib/services/history_store.dart)).

**Uploading to Supabase** happens whenever there is internet:
- right after each scan;
- when the app opens;
- every minute while it is open;
- when the user pulls down on the History screen.

**Retries are safe.** Each scan gets its own id on the phone, which is also its row id in
Supabase. An upload that stopped halfway is simply retried, and an already-uploaded photo or row
counts as done, so nothing is saved twice.

**History** lists the user's scans newest first and marks each one "Uploaded" or "Waiting to
upload". Scans saved on another phone appear once online, and their photo is downloaded once.

**Deleting** removes a scan from History straight away; while offline, the deletion reaches
Supabase on the next upload. Each account only sees its own scans, even on a shared phone.

Tested in [test/history_test.dart](../test/history_test.dart) with a stand-in for Supabase
that can be switched offline.

## How the models get onto the phone

```
ml/src/train.py          -> backend/models/*.joblib       (scikit-learn models)
ml/src/export_models.py  -> assets/models/*.json + *.bin  (numbers the app reads, about 50 MB)
```

`export_models.py` first recomputes every model's probabilities in plain numpy, following
scikit-learn and libsvm step by step:

1. Scaling.
2. The RBF SVM, one-vs-one.
3. libsvm's pairwise-coupling probabilities, or a sigmoid calibration for the yes/no models.
4. The pest classifier's moth sub-classes merged back into their pest.

It exports only if the numpy results match scikit-learn (largest probability difference under
0.0000002, same answers 100%). Run it after every retraining, then rebuild the app.

## How the Dart code was checked

Each image step was first written in plain numpy and checked pixel for pixel against OpenCV 5
and scikit-image, then ported line by line to Dart.

- **Resize:** OpenCV's INTER_AREA, including its whole-number shrink, area tables and
  fixed-point enlarging.
- **Color conversions:** gray (OpenCV 5's 15-bit constants), HSV, and Lab. Lab was checked on
  all 16,777,216 RGB colors; one table entry where OpenCV's software cube root rounds a tie
  down had to be corrected.
- **Blur:** the 3x3 Gaussian blur.
- **Features:** GLCM, uniform LBP and HOG.

[test/offline_parity_test.dart](../test/offline_parity_test.dart) runs 903 test photos (100 of
each kind) through the phone path: Flutter's own photo decoder, then the Dart features, Dart
models and Dart decision. It compares the results with Python (`ml/src/parity_reference.py`):

| Check | Agreement with Python |
|---|---|
| Features (raw pixels in) | GLCM and LBP exact, HOG within float32 rounding, colour statistics within 0.0001 |
| Each model's answer (corn check, detector, healthy-vs-damaged, pest) | 100% |
| Outcome shown (healthy / damage / pest / not corn / unclear) | **100%** (903/903) |
| Pest name | **100%** |
| Plant area found (background, holes) | 100% |
| Severity grade (Low / Moderate / High) | **98.7%** (787/797); damage % median difference 0, 90% within 0.4 points |

The severity grade is the only thing that can differ. scikit-learn's K-Means picks its starting
points with Python's random generator, which cannot be reproduced in Dart, so the app uses its
own fixed-seed K-Means (k-means++ starts, 4 runs, the same tolerance). On a few borderline
photos the colour groups come out differently and the grade changes.

## Speed and size

About 0.3 s per photo on the training PC, run in a background isolate so the screen stays
responsive. The release APK grows from 51.5 MB to 95.7 MB because of the models; building with
`flutter build apk --split-per-abi` makes each phone's APK smaller.
