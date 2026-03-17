# prepare_datasets.py
"""
Prepare training and test datasets by unzipping archives and remapping paths.
Run this in Colab after mounting Google Drive.
"""

import os
import zipfile
from pathlib import Path
import shutil

DRIVE_ROOT = "/content/drive/MyDrive/GeoAware_project/datasets"
LOCAL_ROOT = "/content/data"

os.makedirs(LOCAL_ROOT, exist_ok=True)

DATASETS = [
    # Train
    ("train/art_train.zip",        "train/art_train.txt",        "train/art_train.txt",        "art_train"),
    ("train/iiit5k_train.zip",     "train/iiit5k_train.txt",     "train/iiit5k_train.txt",     "iiit5k_train"),
    ("train/mjsynth.zip",          "train/mjsynth_path_label.txt","train/mjsynth_path_label.txt","mjsynth"),
    ("train/total_train.zip",      "train/total_train.txt",      "train/totaltext_train_gt.txt","totaltext_train_gt"),
    # Test
    ("test/art_test.zip",          "test/art_test_gt.txt",       "test/art_test_gt.txt",       "art_test"),
    ("test/iiit5k_test.zip",       "test/iiit5k_test.txt",       "test/iiit5k_test.txt",       "iiit5k_test"),
    ("test/totaltext_test.zip",    "test/totaltext_test_gt.txt", "test/totaltext_test_gt.txt", "totaltext_test"),
]

def find_img_dir(extract_to):
    entries = os.listdir(extract_to)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_to, entries[0])):
        return os.path.join(extract_to, entries[0])
    return extract_to

def remap_txt(gt_path, out_path, img_dir):
    if not os.path.exists(gt_path):
        print(f"GT file missing: {gt_path}")
        return 0, 0

    remapped = missing = 0
    lines_out = []
    with open(gt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        filename = parts[0].replace('\\', '/').split('/')[-1]
        label = parts[-1].strip()
        local_path = os.path.join(img_dir, filename)
        if os.path.exists(local_path):
            lines_out.append(f"{local_path}\t{label}")
            remapped += 1
        else:
            missing += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines_out))

    return remapped, missing

print("="*70)
print("Starting dataset preparation...")
print("="*70)

results = []

for zip_rel, gt_rel, out_rel, extract_rel in DATASETS:
    zip_path  = os.path.join(DRIVE_ROOT, zip_rel)
    gt_path   = os.path.join(DRIVE_ROOT, gt_rel)
    out_path  = os.path.join(LOCAL_ROOT, out_rel)
    extract_to = os.path.join(LOCAL_ROOT, extract_rel)

    name = os.path.basename(zip_path)

    if not os.path.exists(zip_path):
        print(f"[SKIP] {name} — zip not found")
        results.append((name, None, None))
        continue

    os.makedirs(extract_to, exist_ok=True)
    if len(os.listdir(extract_to)) > 1:
        print(f"[SKIP] {name} — already extracted")
    else:
        print(f"[UNZIP] {name} ...", end=" ")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"done ({len(os.listdir(extract_to))} entries)")

    img_dir = find_img_dir(extract_to)
    remapped, missing = remap_txt(gt_path, out_path, img_dir)
    print(f"  remapped={remapped:,}  missing={missing:,}  → {out_path}")
    results.append((name, remapped, missing))

print("\n" + "="*70)
print(f"{'Dataset':<35} {'Remapped':>10} {'Missing':>10}")
print("-"*70)
for name, r, m in results:
    if r is None:
        print(f"{name:<35} {'NOT FOUND':>10}")
    else:
        print(f"{name:<35} {r:>10,} {m:>10,}")
print("="*70)

print("\nDone. Cleaned files are in:", LOCAL_ROOT)