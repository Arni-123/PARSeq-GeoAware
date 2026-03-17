# prepare_all_test_annotations.py
"""
Prepare annotation files for ALL SIX common STR test datasets:
  1. IIIT5K_test
  2. ArT_test
  3. Total-Text_test
  4. SVT
  5. ICDAR 2013 (IC13)
  6. ICDAR 2015 (IC15)

Output location: PROJECT_ROOT/test_annotations/
   iiit5k_test.txt
   art_test_gt.txt
   totaltext_test_gt.txt
   svt_test.txt
   icdar13_test.txt
   icdar15_test.txt

Assumes raw data is already uploaded to:
  GeoAware_project/datasets/test/
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = '/content/drive/MyDrive/GeoAware_project'
RAW_TEST_DIR = f'{PROJECT_ROOT}/datasets/test'
ANNOT_DIR    = f'{PROJECT_ROOT}/test_annotations'

os.makedirs(ANNOT_DIR, exist_ok=True)

CHARSET_36 = set('0123456789abcdefghijklmnopqrstuvwxyz')

def clean(s: str) -> str:
    return ''.join(c for c in s.lower() if c in CHARSET_36)

def is_valid_label(s: str) -> bool:
    cl = clean(s)
    return 1 <= len(cl) <= 25

# ──────────────────────────────────────────────────────────────────────────────
# Helper: report file status
# ──────────────────────────────────────────────────────────────────────────────

def check_file_status(name: str, path: str):
    exists = os.path.exists(path)
    print(f"  {name:<14} {'✓ Exists' if exists else '✗ Missing'}   {path}")
    return exists

# ──────────────────────────────────────────────────────────────────────────────
# Already prepared (just check & report)
# ──────────────────────────────────────────────────────────────────────────────

def report_existing():
    print("\nAlready prepared datasets (from unzip/remap):")
    print("-" * 60)
    existing = [
        ("IIIT5K",    f"{RAW_TEST_DIR}/iiit5k_test.txt"),
        ("ArT",       f"{RAW_TEST_DIR}/art_test_gt.txt"),
        ("Total-Text",f"{RAW_TEST_DIR}/totaltext_test_gt.txt"),
    ]
    for name, p in existing:
        check_file_status(name, p)

# ──────────────────────────────────────────────────────────────────────────────
# SVT (from XML)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_svt():
    base = f'{RAW_TEST_DIR}/svt_img'
    xml_path = next((p for p in [
        f'{base}/svt_test.xml',
        f'{RAW_TEST_DIR}/svt_test.xml'
    ] if os.path.exists(p)), None)

    if not xml_path:
        print('[SVT] ✗ svt_test.xml not found — skipping')
        return 0

    img_base = f'{base}/img'
    out_path = f'{ANNOT_DIR}/svt_test.txt'

    tree = ET.parse(xml_path)
    root = tree.getroot()

    kept = skipped = 0

    with open(out_path, 'w', encoding='utf-8') as f:
        for img_elem in root.iter('image'):
            name_elem = img_elem.find('imageName')
            if not name_elem:
                continue
            rel = name_elem.text.strip()
            img_path = f'{img_base}/{rel}'
            if not os.path.exists(img_path):
                skipped += len(list(img_elem.iter('taggedRectangle')))
                continue

            for rect in img_elem.iter('taggedRectangle'):
                tag = rect.find('tag')
                if not tag or not tag.text:
                    skipped += 1
                    continue
                text = tag.text.strip()
                if text in ('?', '###', '') or not is_valid_label(text):
                    skipped += 1
                    continue

                x, y = int(rect.get('x', 0)), int(rect.get('y', 0))
                w, h = int(rect.get('width', 0)), int(rect.get('height', 0))
                bbox = f"{x},{y},{x+w},{y+h}"

                f.write(f"{img_path}|{bbox}\t{clean(text)}\n")
                kept += 1

    print(f'[SVT]     kept = {kept:>6,}   skipped = {skipped:>6,}   → {out_path}')
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# ICDAR 2013
# ──────────────────────────────────────────────────────────────────────────────

def prepare_icdar13():
    img_dir = f'{RAW_TEST_DIR}/ICDAR13_test'
    json_path = next((p for p in [
        f'{img_dir}/icdar13_test_gt.json',
        f'{RAW_TEST_DIR}/icdar13_test_gt.json'
    ] if os.path.exists(p)), None)

    if not json_path:
        print('[IC13] ✗ JSON not found — skipping')
        return 0

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    annots = data.get('annots', data)
    out_path = f'{ANNOT_DIR}/icdar13_test.txt'

    kept = skipped = 0

    with open(out_path, 'w', encoding='utf-8') as f:
        for img_name, info in annots.items():
            if not isinstance(info, dict):
                continue
            img_p = f'{img_dir}/{img_name}'
            if not os.path.exists(img_p):
                skipped += len(info.get('text', []))
                continue
            for bbox, text in zip(info.get('bbox', []), info.get('text', [])):
                t = text.strip()
                if t in ('###', '') or not is_valid_label(t):
                    skipped += 1
                    continue
                bbox_str = bbox_to_str(bbox)
                f.write(f"{img_p}|{bbox_str}\t{clean(t)}\n")
                kept += 1

    print(f'[IC13]    kept = {kept:>6,}   skipped = {skipped:>6,}   → {out_path}')
    return kept


def bbox_to_str(bbox):
    if len(bbox) == 8:  # 4 corners
        xs = bbox[0::2]
        ys = bbox[1::2]
    elif len(bbox) == 4:  # x,y,w,h or x1,y1,x2,y2
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        xs, ys = bbox[::2], bbox[1::2]
    else:
        return "0,0,0,0"
    return f"{min(xs)},{min(ys)},{max(xs)},{max(ys)}"


# ──────────────────────────────────────────────────────────────────────────────
# ICDAR 2015
# ──────────────────────────────────────────────────────────────────────────────

def prepare_icdar15():
    gt_candidates = [
        f'{RAW_TEST_DIR}/ICDAR15_test_gt.txt',
        f'{RAW_TEST_DIR}/ICDAR15_test/ICDAR15_test_gt.txt'
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)

    if not gt_path:
        print('[IC15] ✗ gt txt not found — skipping')
        return 0

    img_dir = f'{RAW_TEST_DIR}/ICDAR15_test'
    out_path = f'{ANNOT_DIR}/icdar15_test.txt'

    kept = skipped = 0

    with open(gt_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            m = re.match(r'^([^,]+),\s*"?([^"]+?)"?\s*$', line)
            if not m:
                skipped += 1
                continue
            img_name, label = m.group(1).strip(), m.group(2).strip()
            if label in ('###', '') or not is_valid_label(label):
                skipped += 1
                continue
            img_p = f'{img_dir}/{img_name}'
            if not os.path.exists(img_p):
                skipped += 1
                continue
            fout.write(f'{img_p}\t{clean(label)}\n')
            kept += 1

    print(f'[IC15]    kept = {kept:>6,}   skipped = {skipped:>6,}   → {out_path}')
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Preparing / Checking annotations for 6 STR test datasets")
    print("=" * 80)

    report_existing()

    print("\nPreparing remaining datasets (SVT, IC13, IC15):")
    print("-" * 80)

    n_svt  = prepare_svt()
    n_ic13 = prepare_icdar13()
    n_ic15 = prepare_icdar15()

    print("\n" + "=" * 80)
    print("Final Summary")
    print("-" * 80)
    print(f"{'Dataset':<14} {'Status':<12} {'Samples':>10}")
    print("-" * 80)

    for name, func in [
        ("IIIT5K",      lambda: check_file_status("IIIT5K", f"{RAW_TEST_DIR}/iiit5k_test.txt")),
        ("ArT",         lambda: check_file_status("ArT",    f"{RAW_TEST_DIR}/art_test_gt.txt")),
        ("Total-Text",  lambda: check_file_status("Total-Text", f"{RAW_TEST_DIR}/totaltext_test_gt.txt")),
        ("SVT",         lambda: n_svt),
        ("ICDAR13",     lambda: n_ic13),
        ("ICDAR15",     lambda: n_ic15),
    ]:
        if callable(func) and func.__name__ == "<lambda>":
            # for existing — just print status
            pass
        else:
            print(f"{name:<14} {'Prepared':<12} {func:>10,}" if isinstance(func, int) else "")

    print("=" * 80)
    print("\nAll annotation files are ready in:")
    print(f"  {ANNOT_DIR}/")
    print("You can now use them in evaluation.")

if __name__ == '__main__':
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        print("Not in Colab — assuming Drive is accessible.")

    main()