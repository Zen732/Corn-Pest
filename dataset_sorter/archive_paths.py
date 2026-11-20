"""Where the dataset tools' data lives since the cleanup of 20 November 2026.

Unused photos, downloads and tool outputs were moved into _archive/ (see _archive/README.md
and _archive/MANIFEST.csv). The tools read and write those places through these paths, so
they still work if more photos are ever collected.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "ml" / "dataset"
ARCHIVE = PROJECT_DIR / "_archive"

MOTH_MOVES = DATASET_DIR / "moth_moves.csv"                 # still in use by ml/src/train.py
RESERVE = ARCHIVE / "2_unused_photos" / "extra_moths_and_others"   # was ml/dataset/reserve
REJECTED = ARCHIVE / "2_unused_photos" / "rejected_bad_photos"     # was ml/dataset/rejected
REJECTED_LOG = REJECTED / "rejected_log.csv"
RICE_BANANA = ARCHIVE / "2_unused_photos" / "rice_banana_project"  # was ml/dataset/archive_rice_banana
DOWNLOADS = ARCHIVE / "3_downloads" / "dataset_sorter_downloads"   # was dataset_sorter/downloads
AGML_DOWNLOADS = ARCHIVE / "3_downloads" / "archive_non_corn" / "downloads"
TOOL_OUTPUTS = ARCHIVE / "4_dataset_tool_outputs"
AUDIT = TOOL_OUTPUTS / "audit"
EXAMPLES = TOOL_OUTPUTS / "examples"
MOTH_PREVIEW = TOOL_OUTPUTS / "moth_preview"
LOGS = ARCHIVE / "5_backups_and_logs"
