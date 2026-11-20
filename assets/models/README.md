# Models for offline checking

The app checks photos on the phone with the four trained models. Their files are made by

    ml/.venv/Scripts/python.exe ml/src/export_models.py

and are not on GitHub (about 50 MB): `corn_check`, `detector`, `health` and `pest`
(`.json` settings + `.bin` numbers each), plus `model_version.txt`. Run the export again
after every retraining, then rebuild the app.
