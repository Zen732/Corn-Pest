"""Exact copies of ml/src/features.py, severity.py, predict.py and flow.py (made by
backend/sync_shared.py; never edit them here). They import each other by plain name
(`from features import ...`), as in ml/src, so this folder is put on the import path."""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
