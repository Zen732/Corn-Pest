# Models for offline checking

The app checks photos on the phone with the four trained models (`corn_check`, `detector`,
`health`, `pest`: `.json` settings + `.bin` numbers each, plus `model_version.txt`) and, since
Version 2, the ready-made image network that turns a photo into numbers for them
(`network.json` + `network.bin`: EfficientNet-B2, 1408 numbers, 31 MB, since 25 Nov 2026;
MobileNetV3-Large before). The models' support vectors are stored as 16-bit floats. They are made by

    ml/.venv/Scripts/python.exe ml/src/network_export.py     (the network; only once)
    ml/.venv/Scripts/python.exe ml/src/export_models.py      (after every retraining)

and are not on GitHub (about 93 MB). Run the export again after every retraining, then rebuild
the app.
