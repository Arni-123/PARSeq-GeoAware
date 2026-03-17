# evaluate.py
"""
Basic evaluation script for PARSeq-GeoAware (CTC version)
Usage:
    python evaluate.py --checkpoint path/to/model.pth --test_txt path/to/test.txt
"""

import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from models.model import PARSeqGeoAware, improved_ctc_decode, set_charset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--test_txt", type=str, required=True)
parser.add_argument("--charset", type=int, default=36, choices=[36,64])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--beam", action="store_true", help="use beam search")
args = parser.parse_args()

set_charset(args.charset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PARSeqGeoAware(num_chars=args.charset+1, use_geometric=True, use_fusion=True,
                       use_rectification=True, use_tps=True, use_attention=False)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((64, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

def load_image(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)

print("Loading test set...")
lines = open(args.test_txt).read().splitlines()
paths = []
gts = []
for line in lines:
    if "\t" not in line:
        continue
    p, gt = line.split("\t", 1)
    paths.append(p)
    gts.append(gt.lower().strip())

print(f"Found {len(paths)} samples")

correct = 0
total = 0

with torch.no_grad():
    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i:i+args.batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = load_image(p)
                batch_imgs.append(img)
            except:
                continue

        if not batch_imgs:
            continue

        batch_imgs = torch.cat(batch_imgs).to(device)
        log_probs = model(batch_imgs)
        preds = improved_ctc_decode(log_probs)

        for pred, gt in zip(preds, gts[i:i+len(preds)]):
            if pred == gt:
                correct += 1
            total += 1

acc = 100.0 * correct / total if total > 0 else 0
print(f"Accuracy: {acc:.2f}%  ({correct}/{total})")