"""Copy the code shared with training from ml/src into backend/app/services.

The backend must give exactly the same answers as the whole-app test (ml/src/evaluate_flow.py),
so these files are exact copies, never edited here. After changing any of them in ml/src, run
(from the project root):
    backend/.venv/Scripts/python.exe backend/sync_shared.py            copy them
    backend/.venv/Scripts/python.exe backend/sync_shared.py --check    only report differences
The backend tests also fail if the copies differ.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ml" / "src"
DEST = ROOT / "backend" / "app" / "services"
SHARED = ["features.py", "severity.py", "predict.py", "flow.py"]


def _text(path):
    return path.read_bytes().replace(b"\r\n", b"\n")  # git may change line endings on Windows


def differences():
    """Shared files whose backend copy is missing or differs from ml/src."""
    return [name for name in SHARED if not (DEST / name).exists() or _text(SOURCE / name) != _text(DEST / name)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report differences (exit code 1 if any)")
    check = ap.parse_args().check
    diff = differences()
    if check:
        print("backend copies match ml/src" if not diff else f"different from ml/src: {', '.join(diff)}")
        sys.exit(1 if diff else 0)
    for name in diff:
        shutil.copyfile(SOURCE / name, DEST / name)
    print(f"copied {', '.join(diff)}" if diff else "already up to date")


if __name__ == "__main__":
    main()
