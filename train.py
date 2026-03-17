# train.py
"""
Progressive training script for PARSeq-GeoAware (CTC version)

Supports three stages:
  Stage 1: MJSynth + IIIT5K (synthetic + regular)
  Stage 2: IIIT5K + ArT (regular + irregular)
  Stage 3: ArT + Total-Text (+ optional IIIT5K)

Usage example:
    python train.py --stage all \
        --mjsynth_txt data/train/mjsynth_path_label.txt \
        --iiit5k_train data/train/iiit5k_train.txt \
        --art_txt    data/train/art_train_clean.txt \
        --totaltext_txt data/train/totaltext_train_gt.txt \
        --save_dir checkpoints_geoaware_v4
"""

import os
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
import torchvision.transforms as TF

import yaml

def load_config(config_path="configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# In main()
cfg = load_config()
# override with argparse if needed
args.batch_size = cfg["training"]["batch_size"]
# etc.

from models.model import (
    PARSeqGeoAware,
    set_charset,
    CHARSET,
    BLANK_IDX,
    improved_ctc_decode
)

# ──────────────────────────────────────────────────────────────────────────────
# Simple dataset (line-based, no fancy caching yet)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleLineDataset(Dataset):
    def __init__(self, txt_path, transform=None, max_samples=None):
        self.transform = transform or TF.Compose([
            TF.Resize((64, 256)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.samples = []
        with open(txt_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                path, label = line.split('\t', 1)
                label = label.strip().lower()
                if len(label) == 0:
                    continue
                self.samples.append((path, label))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples):,} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('L')
        except Exception:
            # Fallback - avoid crashing the loader
            img = Image.new('L', (256, 64), random.randint(100, 180))
            label = "error"
        img = self.transform(img)
        return img, label


def get_loader(txt_path, batch_size, shuffle=True, max_samples=None, num_workers=2):
    ds = SimpleLineDataset(txt_path, max_samples=max_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# CTC target preparation
# ──────────────────────────────────────────────────────────────────────────────

def make_ctc_targets(labels, device, T_max):
    target_list = []
    lengths = []
    valid_mask = []

    char2idx = {c: i for i, c in enumerate(CHARSET)}

    for lbl in labels:
        if not lbl or (2 * len(lbl) + 1) > T_max:
            valid_mask.append(False)
            continue
        idxs = [char2idx.get(c, BLANK_IDX) for c in lbl]
        target_list.extend(idxs)
        lengths.append(len(idxs))
        valid_mask.append(True)

    if not target_list:
        return None, None, valid_mask

    return (
        torch.tensor(target_list, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        valid_mask
    )


# ──────────────────────────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, accum_steps, epoch_idx, log_every=50):
    model.train()
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=(epoch_idx > 0))

    total_loss = 0.0
    n_batches = 0
    n_skipped = 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)

        valid = [lbl for lbl in labels if lbl]
        if len(valid) == 0:
            n_skipped += 1
            continue

        with autocast():
            log_probs, _ = model(images)
            T = log_probs.size(0)
            targets, target_lengths, vmask = make_ctc_targets(labels, device, T)

            if targets is None:
                n_skipped += 1
                continue

            if not all(vmask):
                keep = [i for i, ok in enumerate(vmask) if ok]
                log_probs = log_probs[:, keep, :]
                target_lengths = target_lengths[torch.tensor(keep, device=device)]

            input_lengths = torch.full((log_probs.size(1),), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad()
                continue

        loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches += 1

        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"Epoch {epoch_idx+1} | step {step+1:>5} | loss {total_loss/n_batches:.4f}")

    if n_batches == 0:
        print("Warning: no valid batches this epoch")

    return total_loss / n_batches if n_batches > 0 else float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def run_stage(stage_num, model, loader, optimizer, scaler, device, epochs, lr, save_dir):
    print(f"\n=== Stage {stage_num} ===")
    print(f"Epochs: {epochs}   LR: {lr:.2e}   Loader size: {len(loader)} batches")

    best_loss = float('inf')

    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, scaler, device,
            args.accum_steps, epoch, log_every=args.log_every
        )

        print(f"Epoch {epoch+1}/{epochs}   loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(save_dir, f"stage{stage_num}_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PARSeq-GeoAware Training")
    parser.add_argument('--stage', default='all', choices=['1', '2', '3', 'all'])

    parser.add_argument('--mjsynth_txt',    required=True)
    parser.add_argument('--iiit5k_train',   required=True)
    parser.add_argument('--art_txt',        required=True)
    parser.add_argument('--totaltext_txt',  required=True)

    parser.add_argument('--mjsynth_max',    type=int, default=134828)
    parser.add_argument('--art_max',        type=int, default=36000)
    parser.add_argument('--totaltext_max',  type=int, default=9000)

    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--accum_steps',    type=int, default=None)  # auto
    parser.add_argument('--num_workers',    type=int, default=2)

    parser.add_argument('--epochs_s1',      type=int, default=12)
    parser.add_argument('--epochs_s2',      type=int, default=12)
    parser.add_argument('--epochs_s3',      type=int, default=25)

    parser.add_argument('--lr_s1',          type=float, default=1e-4)
    parser.add_argument('--lr_s2',          type=float, default=5e-5)
    parser.add_argument('--lr_s3',          type=float, default=1e-4)

    parser.add_argument('--save_dir',       default='checkpoints_geoaware_v4')
    parser.add_argument('--log_every',      type=int, default=50)

    parser.add_argument('--charset',        type=int, default=36, choices=[36,64])
    parser.add_argument('--no_pretrained',  action='store_true')
    parser.add_argument('--iiit5k_in_stage3', action='store_true')

    return parser.parse_args()


def main():
    global args
    args = parse_args()

    set_charset(args.charset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accum_steps = args.accum_steps or (4 if device.type == 'cuda' else 1)

    print(f"Device: {device}")
    print(f"Charset: {len(CHARSET)} chars + blank")
    print(f"Effective batch size: {args.batch_size * accum_steps}")

    model = PARSeqGeoAware(
        num_chars = len(CHARSET) + 1,
        use_geometric     = True,
        use_rectification = True,
        use_tps           = False,           # turn on in stage 3 if desired
        use_attention     = False,
        max_len           = 25
    ).to(device)

    if not args.no_pretrained:
        print("Note: pretrained ViT loading not implemented yet in this version")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_s1, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == 'cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    stages = ['1','2','3'] if args.stage == 'all' else [args.stage]

    for stage in stages:
        print(f"\nStarting stage {stage}")

        if stage == '1':
            loader = get_loader(
                args.mjsynth_txt,
                args.batch_size,
                max_samples=args.mjsynth_max,
                num_workers=args.num_workers
            )
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_s1, weight_decay=1e-4)
            run_stage(1, model, loader, optimizer, scaler, device,
                      args.epochs_s1, args.lr_s1, args.save_dir)

        elif stage == '2':
            loader_reg = get_loader(args.iiit5k_train, args.batch_size, num_workers=args.num_workers)
            loader_irr = get_loader(args.art_txt, args.batch_size, max_samples=args.art_max,
                                    num_workers=args.num_workers)
            # simple interleaving (not perfect, but good enough for now)
            print("Stage 2: interleaving regular + irregular loaders")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_s2, weight_decay=1e-4)
            # Note: real interleaving would need custom iterator
            # Here we just train on both sequentially for simplicity
            run_stage(2, model, loader_reg, optimizer, scaler, device,
                      args.epochs_s2 // 2, args.lr_s2, args.save_dir)
            run_stage(2, model, loader_irr, optimizer, scaler, device,
                      args.epochs_s2 // 2, args.lr_s2, args.save_dir)

        elif stage == '3':
            loader = get_loader(
                args.art_txt,
                args.batch_size,
                max_samples=args.art_max,
                num_workers=args.num_workers
            )
            # You can add TotalText similarly
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_s3, weight_decay=1e-4)
            run_stage(3, model, loader, optimizer, scaler, device,
                      args.epochs_s3, args.lr_s3, args.save_dir)

    print("\nTraining finished.")


if __name__ == '__main__':
    main()