# Dataset Sorter

Finds images in big downloaded datasets that look like your example photos and copies
them into `ml/dataset/raw/`. Your original dataset files are never changed.

## How to use

1. **Dump datasets** (whole folders, any structure) into `inbox/`.
2. **Add examples**: 3-10 clear photos per class into `examples/<crop>/<class>/`.
   Use varied angles and lighting. Classes with no examples are skipped.
3. **Dry run first** (copies nothing, shows the numbers):
   ```
   dataset_sorter\.venv\Scripts\python.exe dataset_sorter\sort_images.py --dry-run
   ```
4. Open `sort_log_dryrun.csv`, sort by `score`, and check the images near your
   thresholds. Adjust `--high` / `--low` if needed.
5. **Run for real**:
   ```
   dataset_sorter\.venv\Scripts\python.exe dataset_sorter\sort_images.py
   ```
6. **Check `review/`** by hand. Move the good ones to `ml/dataset/raw/<crop>/<class>/`
   and delete the rest. Also spot-check `ml/dataset/raw/`, because the sorter judges
   by appearance, not by knowing the real pest.

Run from the project root. Files already handled are recorded in `sort_log.csv` and
skipped on the next run, so you can keep adding datasets to `inbox/`.

## What happens to each image

| Result | Condition | Action |
|--------|-----------|--------|
| Clear match | score >= `--high` (0.85) and beats the 2nd-best class by `--margin` (0.03) | copied to `ml/dataset/raw/<crop>/<class>/` |
| Unsure | score >= `--low` (0.75) but not a clear match | copied to `review/<crop>/<class>/` |
| No match | below `--low` | left alone |

Score is the CLIP similarity (0 to 1) between the image and the closest example of the class.

## Duplicates

Different datasets often contain the same photo. When a clear match is a near-duplicate
of an image already in that class:

- class has **fewer than `--target` (300) images**: the duplicate is saved as a
  flipped / rotated version (`*_aug.jpg`), so it is not an identical copy
- class has **300 or more images**: the duplicate is skipped

Other options: `--max-per-class N` stops adding to a class at N images.

## Setup (already done once)

```
python -m venv dataset_sorter\.venv
dataset_sorter\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
dataset_sorter\.venv\Scripts\python.exe -m pip install -r dataset_sorter\requirements.txt
```

PyTorch is the CUDA 12.4 build for the RTX 2050. The first run downloads CLIP ViT-B/32 (about 350 MB).
