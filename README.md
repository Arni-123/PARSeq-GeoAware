# PARSeq-GeoAware

Enhanced PARSeq model with geometric feature extractor and adaptive rectification for robust scene text recognition.

This project improves performance on irregular, curved, and incidental text (ArT, Total-Text, IC15, etc.) through:

* Explicit geometric supervision (boundary, orientation, curvature)
* Adaptive rectification (affine + optional TPS)
* Progressive training (synthetic → regular → irregular/curved)

\---

## Installation

```bash
git clone https://github.com/Arni-123/PARSeq-GeoAware.git
cd PARSeq-GeoAware
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
torch torchvision timm Pillow pyyaml editdistance numpy
```

\---

## Pre-trained Checkpoint (\~340 MB)

A trained checkpoint (`stage3\_best.pth`) is available via Google Drive:

* [**Download stage3\_best.pth**](https://drive.google.com/file/d/1G6OBZN9h9FXj5iK_JckXAKvmacc1cI5T/view?usp=sharing)

One-liner in terminal:

```bash
mkdir -p checkpoints
wget -O checkpoints/stage3\_best.pth \\
  "https://drive.google.com/uc?export=download\&id=1G6OBZN9h9FXj5iK\_JckXAKvmacc1cI5T"
```

> \*\*Note:\*\* For large files, Google Drive may show a confirmation page. Use the web link above if `wget` fails.

\---

## Quick Test / Demo

Run inference on your own cropped word image — no full dataset needed.

```bash
# Single image
python demo.py --checkpoint checkpoints/stage3\_best.pth \\
  --image path/to/your\_word\_crop.jpg

# Folder of images
python demo.py --checkpoint checkpoints/stage3\_best.pth \\
  --image\_folder path/to/your\_images/ \\
  --output predictions.txt
```

**Example output:**

```
your\_image.jpg          → hello123
sign\_crop.png           → open
menu\_item.png           → coffee
```

\---

## Dataset Preparation

> This repo contains \*\*code only\*\*. Datasets must be prepared locally for training or full evaluation.

### Expected Directory Structure

```
datasets/
├── train/
│   ├── mjsynth/          # MJ cropped words
│   ├── iiit5k/           # IIIT5K cropped words
│   ├── art/              # ArT cropped words
│   └── totaltext/        # Total-Text cropped words
└── test/
    ├── iiit5k/
    ├── art/
    ├── totaltext/
    ├── svt/
    ├── icdar13/
    └── icdar15/
```

Annotation files should be **tab-separated**: `image\_path\\tlabel`

Common annotation filenames:

* `data/train/mjsynth/mjsynth\_path\_label.txt`
* `data/train/iiit5k/iiit5k\_train.txt`
* `data/train/art/art\_train\_clean.txt` *(filtered: a–z0–9, length 2–15)*
* `data/train/totaltext/totaltext\_train\_gt.txt`

### Dataset Sources

|Dataset|Link|
|-|-|
|MJSynth|https://www.robots.ox.ac.uk/\~vgg/data/text/mjsynth/|
|IIIT5K|https://cvit.iiit.ac.in/projects/mb/ocrcv/iiit5k.html|
|ArT|https://rrc.cvc.uab.es/?ch=14|
|Total-Text|https://github.com/cs-chan/Total-Text-Dataset|
|SVT / ICDAR13 / ICDAR15|https://github.com/clovaai/deep-text-recognition-benchmark or https://github.com/baudm/parseq|

\---

## Training

Training follows a three-stage progressive curriculum that prevents catastrophic forgetting (**+14.8 pp over joint training** on IIIT5K).

### Stage 1 — Regular Initialization (15 epochs)

```bash
python train.py --stage 1 --epochs 12 --lr 1e-4 \\
  --data mjsynth,iiit5k --freeze-vit-epochs 3
```

MJSynth (134,828) + IIIT5K-train (2,000). ViT encoder frozen for first 3 epochs. Rectification **OFF**. λ\_geo = 0.05.

### Stage 2 — Mixed Regularization (15 epochs)

```bash
python train.py --stage 2 --epochs 12 --lr 5e-5 \\
  --data iiit5k,art --affine
```

Affine rectification activated. Regular and irregular batches interleaved at step level (\~8,862 batches/epoch). λ\_geo = 0.03, λ\_reg = 0.05.

### Stage 3 — Irregular Specialization (25 epochs)

```bash
python train.py --stage 3 --epochs 25 --lr 2e-5 \\
  --data art,totaltext --affine --tps --strong-aug
```

Full rectification (affine + TPS). Strong augmentation: perspective 0.5, rotation ±30°, random erasing p = 0.4. λ\_geo = λ\_reg = 0.0.

\---

## Pretrained Checkpoints

Download pretrained weights from Google Drive:

|Stage|Dataset|Checkpoint|Size|
|-|-|-|-|
|Stage 1|MJSynth + IIIT5K|[Download](https://drive.google.com/file/d/1G6OBZN9h9FXj5iK_JckXAKvmacc1cI5T/view?usp=sharing)|\~319 MB|
|Stage 2|+ ArT|[Download](https://drive.google.com/file/d/1G6OBZN9h9FXj5iK_JckXAKvmacc1cI5T/view?usp=sharing)|\~319 MB|
|Stage 3 ★|+ Total-Text|[Download](https://drive.google.com/file/d/1G6OBZN9h9FXj5iK_JckXAKvmacc1cI5T/view?usp=sharing)|\~319 MB|

Place downloaded files under `checkpoints/`:

```
checkpoints/
├── stage1\_best.pth
├── stage2\_best.pth
└── stage3\_best.pth
```

\---

## Reproducing Paper Results

1. Download test sets for IIIT5K, SVT, ICDAR13, ICDAR15, ArT, and Total-Text
2. Place them under `data/test/`
3. Download the Stage 3 checkpoint (see [Pretrained Checkpoints](#pretrained-checkpoints))
4. Run:

```bash
# Full six-benchmark evaluation
python evaluate.py --checkpoint checkpoints/stage3\_best.pth --all

# Single dataset
python evaluate.py --checkpoint checkpoints/stage3\_best.pth --dataset art

# Cross-domain validation (Stage 1 checkpoint, zero ArT/Total-Text exposure)
python evaluate.py --checkpoint checkpoints/stage1\_best.pth --dataset art,totaltext
```

\---

## Train Datasets

|Dataset|Type|Samples|Stage|Notes|
|-|-|-|-|-|
|MJSynth|Synthetic, regular|134,828|1|Benchmark-standard subset of 9M|
|IIIT5K|Real, regular|2,000|1, 2|Regular anchor for step interleaving|
|ArT|Real, irregular|30,520|2, 3|Latin-only; 85% data from ArT train samples|
|Total-Text|Real, curved|9,282|3|Extracted from 2,211 train images|

See [`configs/default.yaml`](configs/default.yaml) for batch size, epochs, learning rates, and other hyperparameters.

\---

## Folder Structure

```
PARSeq-GeoAware/
├── README.md
├── requirements.txt
├── train.py
├── demo.py                     ← quick inference on your images
├── models/
│   └── model.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── logging.py
├── configs/
│   └── default.yaml
├── data/                       ← .gitignore – create locally
│   ├── train/
│   └── test/
└── checkpoints/                ← .gitignore – where models are saved
```

\---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

