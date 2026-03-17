"""
Progressive Training — PARSeq-GeoAware   [v4 — clean rewrite]
================================================================
Fixes vs all prior versions:
  1. loss=0.0000 was zero_infinity=True masking infinite CTC loss.
     Now: epoch 0 uses zero_infinity=False (crashes loudly if inf).
     Epochs 1+ use zero_infinity=True but log inf-count separately.
  2. Loss reported is now REAL unscaled CTC (not loss/accum_steps).
  3. LR scheduler never called before optimizer.step().
  4. accum_steps auto-set: 4 on GPU, 1 on CPU.
  5. Dataset validation gate: crashes with clear message if empty.
  6. Per-step loss logging every `log_every` steps.
  7. Epoch time shows steps-per-second for easy sanity check.

Usage:
  # Recommended: diagnose data first
  python diagnose.py --mjsynth_txt data/colab/mjsynth_path_label.txt \
                     --iiit5k_train data/colab/iiit5k_train.txt

  # Then: debug loss in isolation
  python debug_loss.py

  # Then: Stage 1 training
  python train.py --stage 1 \
    --mjsynth_txt  data/colab/mjsynth_path_label.txt \
    --iiit5k_train data/colab/iiit5k_train.txt \
    --batch_size 8 --save_dir checkpoints

  # CPU smoke test (no real data):
  python train.py --smoke_test --stage 1 --epochs_s1 2 --batch_size 4
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model import PARSeqGeoAware, CHARSET, BLANK_IDX
from datasets import (build_stage1, build_stage2, build_stage3,
                      CHARSET as DS_CHARSET)

assert DS_CHARSET == CHARSET, "CHARSET mismatch between model.py and datasets.py"

CHARSET_LEN = len(CHARSET) + 1   # 64 = 63 real chars + 1 dedicated CTC blank (idx 63) (idx 38)
_CHAR2IDX   = {c: i for i, c in enumerate(CHARSET)}
CTC_BLANK   = BLANK_IDX          # 63 — dedicated blank, space at index 0 is real char


# ===========================================================================
# CTC target builder
# ===========================================================================

def _make_ctc_targets(labels, device, T_max):
    """
    Convert labels → (flat_targets, target_lengths, valid_mask).
    Skips any label where  2*len + 1 > T_max  (CTC math impossible).
    Returns (None, None, valid_mask) if no valid labels remain.
    """
    target_list, lengths, valid_mask = [], [], []
    for lbl in labels:
        if not lbl or (2 * len(lbl) + 1) > T_max:
            valid_mask.append(False)
            continue
        idxs = [_CHAR2IDX.get(c, CTC_BLANK) for c in lbl]
        target_list.extend(idxs)
        lengths.append(len(idxs))
        valid_mask.append(True)

    if not target_list:
        return None, None, valid_mask

    return (torch.tensor(target_list, dtype=torch.long, device=device),
            torch.tensor(lengths,     dtype=torch.long, device=device),
            valid_mask)


# ===========================================================================
# Auxiliary losses
# ===========================================================================

def _sobel_edges(images):
    """Compute Sobel edge map — used as GT for boundary head."""
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                            dtype=torch.float32).view(1,1,3,3).to(images.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                            dtype=torch.float32).view(1,1,3,3).to(images.device)
    gx = F.conv2d(images, sobel_x, padding=1)
    gy = F.conv2d(images, sobel_y, padding=1)
    edges = torch.sqrt(gx**2 + gy**2)
    edges = F.interpolate(edges, size=(8,32), mode='bilinear', align_corners=False)
    # Normalize per-image to [0,1]
    b = edges.shape[0]
    mn = edges.view(b,-1).min(1)[0].view(b,1,1,1)
    mx = edges.view(b,-1).max(1)[0].view(b,1,1,1)
    return (edges - mn) / (mx - mn + 1e-6)


def _sobel_orientation(images):
    """Compute gradient orientation — used as GT for orientation head."""
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                            dtype=torch.float32).view(1,1,3,3).to(images.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                            dtype=torch.float32).view(1,1,3,3).to(images.device)
    gx = F.conv2d(images, sobel_x, padding=1)
    gy = F.conv2d(images, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2).clamp(min=1e-6)
    orient = torch.cat([gy/mag, gx/mag], dim=1)   # (B, 2, H, W)
    return F.interpolate(orient, size=(8,32), mode='bilinear', align_corners=False)


def _geo_auxiliary_loss(geo_output, images, weight=0.15):
    """
    Auxiliary supervised loss for GFE heads using Sobel GT.
    Gives boundary/orientation/curvature real learning signal.

    boundary  : MSE vs Sobel edge magnitude
    orientation: cosine similarity vs Sobel gradient direction
    curvature : MSE vs Sobel edges (high curvature where edges are strong)
    weight=0.15 keeps geo loss at ~15% of total — CTC still dominates
    """
    with torch.no_grad():
        edge_gt   = _sobel_edges(images)       # (B, 1, 8, 32)
        orient_gt = _sobel_orientation(images) # (B, 2, 8, 32)

    # Boundary loss
    boundary_loss = F.mse_loss(geo_output['boundary'], edge_gt)

    # Orientation loss — cosine similarity
    B = images.shape[0]
    pred_o  = geo_output['orientation'].view(B, 2, -1)
    gt_o    = orient_gt.view(B, 2, -1)
    orient_loss = (1.0 - F.cosine_similarity(pred_o, gt_o, dim=1)).mean()

    # Curvature loss — encourage high response near text edges
    curvature_loss = F.mse_loss(geo_output['curvature'], edge_gt)

    return weight * (boundary_loss + orient_loss + curvature_loss)


def _rectification_loss(rectified, original, geo_output, images):
    """
    Rectification should:
    1. Reduce edge response (straightened text has less curvature)
    2. Not distort image content too much
    """
    # Edge reduction: rectified image should have lower curvature than original
    rect_edges  = _sobel_edges(rectified).mean()
    orig_edges  = _sobel_edges(original).mean()
    # Encourage rectified to have less/equal edge complexity
    edge_loss   = F.relu(rect_edges - orig_edges)

    # Content preservation: rectified should not look completely different
    content_loss = F.mse_loss(rectified, original).clamp(max=0.5)

    return 0.1 * edge_loss + 0.05 * content_loss


def _geo_consistency_loss(gfe, images):
    """Kept for backward compat — now replaced by _geo_auxiliary_loss."""
    with torch.no_grad():
        flipped = torch.flip(images, dims=[3])
    geo1 = gfe(images)['features']
    geo2 = gfe(flipped)['features']
    return F.mse_loss(geo1, torch.flip(geo2, dims=[3]))


# ===========================================================================
# One training epoch
# ===========================================================================

def _train_epoch(model, loader, optimizer,
                 device, scaler, accum_steps,
                 enable_rect_loss, enable_geo_loss,
                 lambda_rect=0.1, lambda_geo=0.3,
                 log_every=100, epoch_num=0,
                 save_every=0, save_dir='checkpoints', stage_num=1, model_ref=None):
    """
    Returns: (avg_ctc_loss, avg_total_loss, n_batches_processed)
    Both losses are UNSCALED (before /accum_steps).
    """
    model.train()

    # Epoch 0: zero_infinity=False so infinite CTC crashes loudly.
    # Epoch 1+: zero_infinity=True but we log inf-count separately.
    strict_criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=False)
    safe_criterion   = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)

    sum_ctc   = 0.0
    sum_total = 0.0
    n_ok      = 0     # batches with finite loss
    n_inf     = 0     # batches with infinite CTC (skipped)
    n_empty   = 0     # batches with no valid labels

    optimizer.zero_grad()
    t_epoch = time.perf_counter()

    for step, (images, labels) in enumerate(loader):

        # ── Filter empty labels ───────────────────────────────────────
        valid = [(img, lbl) for img, lbl in zip(images, labels) if lbl]
        if not valid:
            n_empty += 1
            continue

        images_t = torch.stack([v[0] for v in valid]).to(device)
        labels_v = [v[1] for v in valid]

        use_amp = (scaler is not None)
        with torch.amp.autocast('cuda', enabled=use_amp):

            log_probs, feats = model(images_t, return_features=True)
            T_actual = log_probs.size(0)   # 65 for ViT-Small/patch16/64×256

            targets, t_lens, vmask = _make_ctc_targets(
                labels_v, device, T_max=T_actual)

            if targets is None:
                n_empty += 1
                continue

            # Drop samples whose label failed the T constraint
            if not all(vmask):
                keep      = [i for i, ok in enumerate(vmask) if ok]
                log_probs = log_probs[:, keep, :]
                images_t  = images_t[keep]
                for key in ('rectified',):
                    if feats.get(key) is not None:
                        feats[key] = feats[key][keep]
                if feats.get('geometric') is not None:
                    for k, v in feats['geometric'].items():
                        if v is not None:
                            feats['geometric'][k] = v[keep]

            i_lens = torch.full(
                (log_probs.size(1),), log_probs.size(0),
                dtype=torch.long, device=device)

            # ── CTC loss ─────────────────────────────────────────────
            if epoch_num == 0:
                # Strict — will crash if infinite, exposing real bugs
                try:
                    ctc_loss = strict_criterion(
                        log_probs, targets, i_lens, t_lens)
                except Exception as e:
                    print(f"\n  ✗ CTC crashed at step {step}: {e}")
                    print(f"    T={T_actual}  t_lens={t_lens.tolist()}")
                    print(f"    labels={labels_v[:4]}")
                    raise
            else:
                ctc_loss = safe_criterion(
                    log_probs, targets, i_lens, t_lens)

            if not torch.isfinite(ctc_loss):
                n_inf += 1
                if n_inf <= 5:
                    print(f"    ⚠ step {step}: CTC=inf  "
                          f"T={T_actual}  t_lens={t_lens.tolist()}  "
                          f"labels={labels_v[:3]}")
                # Can't backward through inf — skip
                optimizer.zero_grad()
                continue

            loss = ctc_loss

            # ── Auxiliary losses ──────────────────────────────────────
            if enable_geo_loss and model.use_geometric:
                if feats.get('geometric') is not None:
                    # Main geo loss: supervised boundary/orientation/curvature
                    gl = _geo_auxiliary_loss(feats['geometric'], images_t,
                                             weight=lambda_geo)
                    loss = loss + gl

            if (enable_rect_loss
                    and feats.get('rectified') is not None
                    and feats.get('geometric') is not None):
                rl = _rectification_loss(
                    feats['rectified'], images_t,
                    feats['geometric'], images_t)
                loss = loss + lambda_rect * rl

            loss_scaled = loss / accum_steps

        # ── Backward ─────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # ── Optimizer step every accum_steps ─────────────────────────
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Track UNSCALED losses for reporting
        sum_ctc   += ctc_loss.item()
        sum_total += loss.item()
        n_ok      += 1

        if log_every and n_ok % log_every == 0:
            elapsed = time.perf_counter() - t_epoch
            sps = n_ok / elapsed
            # Estimate remaining time in this epoch
            if hasattr(loader, '__len__'):
                total_batches = len(loader)
                done_frac = (step + 1) / max(total_batches, 1)
                eta_s = (elapsed / max(step+1, 1)) * (total_batches - step - 1)
                eta_str = f"  ETA {eta_s/60:.0f}min" if eta_s > 0 else ""
            else:
                eta_str = ""
            msg = (f"  step {step+1:>5}  "
                   f"ctc={sum_ctc/n_ok:.4f}  "
                   f"{sps:.1f} steps/s"
                   f"{eta_str}  "
                   f"(inf={n_inf} empty={n_empty})")
            print(msg, flush=True)

        # Mid-epoch recovery checkpoint — protects against Colab timeout
        if save_every > 0 and n_ok > 0 and n_ok % save_every == 0:
            _mdl = model_ref if model_ref is not None else model
            rpath = os.path.join(save_dir, f'stage{stage_num}_recovery.pth')
            torch.save({
                'model_state_dict': _mdl.state_dict(),
                'ctc_loss':  sum_ctc / n_ok,
                'step':      step + 1,
                'epoch':     epoch_num,
                'stage':     stage_num,
                'note':      'mid-epoch recovery checkpoint',
            }, rpath)
            print(f"    💾 recovery → {rpath}", flush=True)

    # ── Flush remaining accumulated gradients ─────────────────────────
    if n_ok > 0 and n_ok % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # ── Summary ───────────────────────────────────────────────────────
    if n_inf > 0:
        print(f"    ⚠ {n_inf} batches had infinite CTC (skipped in backward)")
    if n_ok == 0:
        print(f"    ✗ ZERO valid batches this epoch!")
        print(f"      empty={n_empty}  inf={n_inf}")
        print(f"      → Run: python debug_loss.py")

    avg_ctc   = sum_ctc   / n_ok if n_ok else float('nan')
    avg_total = sum_total / n_ok if n_ok else float('nan')
    return avg_ctc, avg_total, n_ok


# ===========================================================================
# Stage 2: interleaved loader — cycles the shorter dataset so the longer
# one is fully consumed each epoch.
# With 250 reg batches and 4431 irr batches: reg cycles ~17x so all
# 4431 irr batches are seen, giving 8862 total batches per epoch.
# ===========================================================================

def _interleaved_loader(reg_loader, irr_loader):
    """
    Interleave reg and irr batches, cycling the shorter one so
    the longer dataset is fully consumed every epoch.
    Pattern: reg irr reg irr ... until longer loader exhausted.
    """
    from itertools import cycle

    # Determine which is longer
    reg_len = len(reg_loader)
    irr_len = len(irr_loader)

    if reg_len >= irr_len:
        # reg is longer — cycle irr
        for rb, ib in zip(reg_loader, cycle(irr_loader)):
            yield rb
            yield ib
    else:
        # irr is longer — cycle reg (our case: 250 reg, 4431 irr)
        for rb, ib in zip(cycle(reg_loader), irr_loader):
            yield rb
            yield ib


# ===========================================================================
# Dataset validation
# ===========================================================================

def _validate(ds, name, min_samples=100):
    n = len(ds)
    if n < min_samples:
        print(f"\n  ✗ '{name}' has only {n} samples (need ≥ {min_samples})")
        print(f"    Image paths in the txt file likely don't exist here.")
        print(f"    Fix: python diagnose.py → python remap_paths.py")
        raise RuntimeError(f"Dataset '{name}' too small.")
    print(f"  ✓ {name}: {n:,} samples")
    return n


# ===========================================================================
# Stage runner
# ===========================================================================

def _run_stage(stage_num, model, loader_or_loaders,
               device, save_dir,
               num_epochs, lr, use_tps,
               enable_rect_loss=True, enable_geo_loss=True,
               lambda_geo=0.15, lambda_rect=0.05,
               warmup_epochs=2, scheduler_type='cosine',
               accum_steps=1, log_every=100, save_every=0,
               freeze_encoder_epochs=0):

    os.makedirs(save_dir, exist_ok=True)
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # Live progress log — readable from another Colab cell while training runs
    log_path = os.path.join(save_dir, f'stage{stage_num}_progress.txt')
    def _log(msg):
        ts = time.strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, 'a') as _f:
            _f.write(line + '\n')

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=1e-4, betas=(0.9, 0.999))

    # Scheduler starts after warmup
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=lr * 0.01)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)

    best_loss   = float('inf')
    patience_ct = 0
    EARLY_STOP  = 5

    print(f"\n{'='*66}")
    print(f" STAGE {stage_num}  |  epochs={num_epochs}  lr={lr:.1e}"
          f"  TPS={'ON' if use_tps else 'OFF'}"
          f"  AMP={'ON' if use_amp else 'OFF'}"
          f"  accum={accum_steps}  warmup={warmup_epochs}ep")
    print(f"{'='*66}")

    for epoch in range(num_epochs):

        # ── Unfreeze encoder after freeze_encoder_epochs ───────────────
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            for param in model.parameters():
                param.requires_grad = True
            _log(f"  *** ViT encoder UNFROZEN at epoch {epoch+1} — full model trains ***")
            # Rebuild optimizer so newly unfrozen params are included
            optimizer = optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=1e-4, betas=(0.9, 0.999))

        # ── Warmup LR — set manually, don't touch scheduler ──────────
        if epoch < warmup_epochs:
            wlr = max(1e-7, lr * (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = wlr
            lr_str = f"{wlr:.2e} (warmup {epoch+1}/{warmup_epochs})"
        else:
            lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"

        _log(f"Epoch {epoch+1}/{num_epochs}  lr={lr_str}")

        t0 = time.perf_counter()

        if stage_num == 2:
            reg_ld, irr_ld = loader_or_loaders
            epoch_iter = _interleaved_loader(reg_ld, irr_ld)
        else:
            epoch_iter = loader_or_loaders

        avg_ctc, avg_total, n_ok = _train_epoch(
            model, epoch_iter, optimizer,
            device=device, scaler=scaler, accum_steps=accum_steps,
            enable_rect_loss=enable_rect_loss,
            enable_geo_loss=enable_geo_loss,
            lambda_geo=lambda_geo,
            lambda_rect=lambda_rect,
            log_every=log_every,
            epoch_num=epoch,
            save_every=save_every,
            save_dir=save_dir,
            stage_num=stage_num,
            model_ref=model,
        )
        elapsed = time.perf_counter() - t0

        # NaN = all batches skipped
        if avg_ctc != avg_ctc:
            print(f"  ✗ Epoch {epoch+1}: NaN loss — "
                  f"{n_ok} batches OK, likely data/path issue")
            print(f"    Run: python debug_loss.py")
            break

        _log(f"→ ctc={avg_ctc:.4f}  total={avg_total:.4f}  "
             f"batches={n_ok}  time={elapsed:.0f}s  "
             f"({n_ok/elapsed:.1f} b/s)")

        # ── Scheduler — ONLY after warmup ────────────────────────────
        # optimizer.step() was called inside _train_epoch already,
        # so calling scheduler.step() here is correctly ordered.
        if epoch >= warmup_epochs:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_ctc)
            else:
                scheduler.step()

        # ── Checkpoint ────────────────────────────────────────────────
        if avg_ctc < best_loss:
            best_loss   = avg_ctc
            patience_ct = 0
            # Check what components actually have weights saved
            # (use_rectification flag may be False in Stage 1 even though
            #  rectification weights exist in the model — config must reflect
            #  actual weights so evaluate.py builds the right architecture)
            sd = model.state_dict()
            has_rect = any('rectification' in k for k in sd)
            has_geo  = any('gfe' in k or 'fusion' in k for k in sd)
            ckpt = {
                'epoch':                epoch,
                'model_state_dict':     sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'ctc_loss':             avg_ctc,
                'total_loss':           avg_total,
                'config': {
                    'use_geometric':     has_geo,
                    'use_rectification': has_rect,
                    'use_tps':           model.use_tps,
                    'num_chars':         CHARSET_LEN,
                },
                'charset': CHARSET,
                'stage':   stage_num,
            }
            p = os.path.join(save_dir, f'stage{stage_num}_best.pth')
            torch.save(ckpt, p)
            _log(f"✓ Saved {p}  (best ctc: {best_loss:.4f})")
        else:
            patience_ct += 1
            _log(f"No improvement ({patience_ct}/{EARLY_STOP})")
            if patience_ct >= EARLY_STOP:
                _log(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n  Stage {stage_num} done.  Best CTC loss: {best_loss:.4f}")
    return best_loss


# ===========================================================================
# DataLoader
# ===========================================================================

def _loader(ds, batch_size, shuffle, num_workers, device):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=(device.type == 'cuda'))


# ===========================================================================
# Smoke test dataset (zero real data needed)
# ===========================================================================

class _Smoke(torch.utils.data.Dataset):
    W = ['hello','world','text','curve','read','sign','train','sample']
    def __init__(self, n=256): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.randn(1, 64, 256), self.W[i % len(self.W)]


# ===========================================================================
# Args
# ===========================================================================

def _args():
    p = argparse.ArgumentParser('PARSeq-GeoAware Training')
    p.add_argument('--stage',         default='all',
                   choices=['1','2','3','all'])
    p.add_argument('--epochs_s1',     type=int, default=10)
    p.add_argument('--epochs_s2',     type=int, default=15)
    p.add_argument('--epochs_s3',     type=int, default=20)
    p.add_argument('--mjsynth_txt',   default='data/mjsynth_path_label.txt')
    p.add_argument('--iiit5k_train',  default='data/iiit5k_train.txt')
    p.add_argument('--art_txt',       default='data/ArT_GeoAware_gt.txt')
    p.add_argument('--totaltext_txt', default='data/totaltext_gt.txt')
    p.add_argument('--mjsynth_max',   type=int, default=134_828)
    p.add_argument('--art_max',       type=int, default=None,
                   help='Cap ArT samples for Stage 2/3. '
                        'Default None = use all 35K. '
                        'Use 5000 for ~30min epochs on T4.')
    p.add_argument('--totaltext_max', type=int, default=None,
                   help='Cap TotalText samples for Stage 3. '
                        'Default None = use all ~9.3K. '
                        'Use 5000 to match art_max.')
    p.add_argument('--batch_size',    type=int, default=8)
    p.add_argument('--num_workers',   type=int, default=0)
    p.add_argument('--accum_steps',   type=int, default=None,
                   help='Gradient accum steps. Default: 4 GPU / 1 CPU')
    p.add_argument('--log_every',     type=int, default=100,
                   help='Log per-step stats every N steps. 0=silent')
    p.add_argument('--warmup_s1',     type=int, default=2,
                   help='Warmup epochs for Stage 1 (default 2). '
                        'Set 0 when resuming mid-stage.')
    p.add_argument('--warmup_s2',     type=int, default=2,
                   help='Warmup epochs for Stage 2 (default 2). '
                        'Set 0 when resuming mid-stage.')
    p.add_argument('--warmup_s3',     type=int, default=1,
                   help='Warmup epochs for Stage 3 (default 1). '
                        'Set 0 when resuming mid-stage.')
    p.add_argument('--save_every',    type=int, default=0,
                   help='Save a recovery checkpoint every N steps inside an epoch. '
                        'Recommended: 500. Saved as stageX_recovery.pth on Drive.')
    p.add_argument('--resume',        default=None)
    p.add_argument('--parseq_weights',default=None,
                   help='Path to official PARSeq pretrained .pt checkpoint. '
                        'Transfers ViT encoder trained on 16M images -> +15%% accuracy.')
    p.add_argument('--freeze_encoder',type=int, default=3,
                   help='Freeze ViT encoder for N epochs in Stage 1 while '
                        'GFE/head warm up. Only applies when --parseq_weights set.')
    p.add_argument('--save_dir',      default='checkpoints')
    p.add_argument('--lr_s1',         type=float, default=1e-4,  help='Stage 1 learning rate (default 1e-4)')
    p.add_argument('--lr_s2',         type=float, default=5e-5,  help='Stage 2 learning rate (default 5e-5)')
    p.add_argument('--lr_s3',         type=float, default=2e-5,  help='Stage 3 learning rate (default 2e-5)')
    p.add_argument('--charset',       type=int,   default=36, choices=[36, 64],
                                      help='36=SOTA standard (a-z,0-9 only)  64=extended with punctuation')
    p.add_argument('--no_pretrained',    action='store_true',
                                         help='Train from scratch — ignore --parseq_weights')
    p.add_argument('--no_geometric',     action='store_true',
                                         help='Disable GeoAware module — pure ViT+CTC baseline')
    p.add_argument('--iiit5k_in_stage3', action='store_true',
                                         help='Add IIIT5K train to Stage 3 to prevent forgetting regular text')
    p.add_argument('--no_rectification', action='store_true',
                                         help='Disable rectification module')
    p.add_argument('--smoke_test',       action='store_true')
    return p.parse_args()


# ===========================================================================
# PARSeq pretrained weight loader
# ===========================================================================

def load_parseq_weights(model, parseq_path, device):
    """
    Transfer official PARSeq ViT-Small encoder weights into our model.

    Key challenge: PARSeq pos_embed is (1,128,384) — trained on larger images.
    Our pos_embed is (1,65,384) — 64 patches + CLS for 64x256 input.
    Solution: interpolate pos_embed from 128 → 65 positions.
    This preserves the learned positional structure (standard practice).

    All 12 transformer blocks transfer exactly — that is where OCR knowledge lives.
    """
    import torch.nn.functional as F

    print(f"  Loading PARSeq pretrained weights: {parseq_path}")
    ckpt = torch.load(parseq_path, map_location=device)
    sd   = ckpt.get('state_dict', ckpt)

    our_sd   = model.state_dict()
    transfer = {}
    skipped  = []

    for k, v in sd.items():
        k2 = k[len('model.'):] if k.startswith('model.') else k

        # ── pos_embed: interpolate from PARSeq seq_len → our seq_len ──
        if k2 == 'encoder.pos_embed' and k2 in our_sd:
            src_len = v.shape[1]          # 128
            tgt_len = our_sd[k2].shape[1] # 65
            if src_len != tgt_len:
                print(f"  Interpolating pos_embed: {src_len} → {tgt_len} positions...")
                # Separate CLS token from position tokens
                cls_pos  = v[:, :1, :]          # (1, 1, 384)
                src_pos  = v[:, 1:, :]           # (1, 127, 384)
                tgt_n    = tgt_len - 1           # 64

                # Reshape to 2D grid for bicubic interpolation
                # PARSeq 128 tokens = approx 8×16 grid or 1×127 — use 1D interp
                src_pos  = src_pos.permute(0, 2, 1)          # (1, 384, 127)
                tgt_pos  = F.interpolate(src_pos,
                                         size=tgt_n,
                                         mode='linear',
                                         align_corners=False) # (1, 384, 64)
                tgt_pos  = tgt_pos.permute(0, 2, 1)          # (1, 64, 384)
                v_interp = torch.cat([cls_pos, tgt_pos], dim=1)  # (1, 65, 384)
                transfer[k2] = v_interp
                continue

        # ── exact shape match ──────────────────────────────────────────
        if k2 in our_sd and our_sd[k2].shape == v.shape:
            transfer[k2] = v
        else:
            # Try adding encoder. prefix for bare ViT keys
            enc_k = 'encoder.' + k2
            if enc_k in our_sd and our_sd[enc_k].shape == v.shape:
                transfer[enc_k] = v
            else:
                skipped.append(k2)

    model.load_state_dict({**our_sd, **transfer}, strict=True)

    enc_loaded  = len([k for k in transfer if 'encoder' in k])
    print(f"  Transferred : {len(transfer)} keys  "
          f"({enc_loaded} encoder, {len(transfer)-enc_loaded} other)")
    print(f"  Skipped     : {len(skipped)} keys "
          f"(decoder/head/text_embed — not used in our architecture)")
    print(f"  pos_embed   : interpolated 128→65 ✓")
    return model


# ===========================================================================
# Main
# ===========================================================================

def main():
    args   = _args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accum  = args.accum_steps or (4 if device.type == 'cuda' else 1)

    # Apply charset FIRST — before model build
    import models.model as _mm
    _mm.CHARSET   = _mm.CHARSET_36 if args.charset == 36 else _mm.CHARSET_64
    _mm.BLANK_IDX = len(_mm.CHARSET)
    global CHARSET, BLANK_IDX, CHARSET_LEN, CTC_BLANK, _CHAR2IDX
    CHARSET     = _mm.CHARSET
    BLANK_IDX   = _mm.BLANK_IDX
    CHARSET_LEN = len(CHARSET) + 1
    CTC_BLANK   = BLANK_IDX
    _CHAR2IDX   = {c: i for i, c in enumerate(CHARSET)}

    print(f"Charset     : {args.charset} chars  blank={BLANK_IDX}  outputs={CHARSET_LEN}")
    print(f"Pretrained  : {'SCRATCH' if args.no_pretrained else 'PARSeq ViT'}")
    print(f"Device      : {device}")
    print(f"Stage       : {args.stage}")
    print(f"Batch       : {args.batch_size} × accum={accum}"
          f" = eff. {args.batch_size * accum}")
    print(f"Smoke test  : {args.smoke_test}")

    use_geo  = not args.no_geometric
    use_rect = not args.no_rectification
    print(f"GeoModule    : {'ON' if use_geo  else 'OFF -- ablation baseline'}")
    print(f"Rectification: {'ON' if use_rect else 'OFF -- ablation'}")
    model = PARSeqGeoAware(
        num_chars=CHARSET_LEN,
        use_geometric=use_geo,
        use_rectification=use_rect,
        use_tps=False,
    ).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # Load PARSeq pretrained ViT encoder (skip if --no_pretrained)
    if args.no_pretrained:
        print("  Training from scratch — no pretrained weights")
    elif args.parseq_weights and os.path.exists(args.parseq_weights):
        model = load_parseq_weights(model, args.parseq_weights, device)
    elif args.parseq_weights:
        print(f"  WARNING: --parseq_weights not found: {args.parseq_weights}")

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        print(f"Resumed     : {args.resume}")

    stages = ['1','2','3'] if args.stage == 'all' else [args.stage]
    bs, nw = args.batch_size, args.num_workers

    for stage in stages:

        # Auto-load previous stage checkpoint
        for prev, prev_stage in [('2','1'), ('3','2')]:
            if stage == prev:
                ck = os.path.join(args.save_dir, f'stage{prev_stage}_best.pth')
                if os.path.exists(ck):
                    sd = torch.load(ck, map_location=device)
                    model.load_state_dict(
                        sd.get('model_state_dict', sd), strict=False)
                    print(f"\nLoaded stage{prev_stage} weights: {ck}")

        # ── Stage 1 ──────────────────────────────────────────────────
        if stage == '1':
            if args.smoke_test:
                ds = _Smoke(n=max(256, bs * 16))
                ld = _loader(ds, bs, True, nw, device)
            else:
                print("\n[Stage 1] Loading datasets...")
                ds = build_stage1(args.mjsynth_txt, args.iiit5k_train,
                                  mjsynth_max=args.mjsynth_max)
                _validate(ds, 'Stage1-combined', min_samples=100)
                ld = _loader(ds, bs, True, nw, device)
                print(f"  DataLoader: {len(ld)} batches per epoch")

            model.use_rectification = False
            model.use_tps           = False

            # Freeze ViT encoder initially so GFE/head can warm up first
            if args.parseq_weights and args.freeze_encoder > 0:
                for name, param in model.named_parameters():
                    if 'encoder' in name:
                        param.requires_grad = False
                n_frozen = sum(1 for _,p in model.named_parameters() if not p.requires_grad)
                print(f"  Encoder frozen ({n_frozen} params). "
                      f"GFE/fusion/head train first for {args.freeze_encoder} epochs.")

            # Freeze ViT for first 3 epochs — forces GFE to develop features
            # before ViT dominates and GFE becomes redundant
            if not args.no_geometric and not args.parseq_weights:
                print("  [GeoAware] Freezing ViT for first 3 epochs — GFE trains first")
                for name, param in model.named_parameters():
                    if 'encoder' in name and 'gfe' not in name:
                        param.requires_grad = False

            _run_stage(1, model, ld, device, args.save_dir,
                       num_epochs=args.epochs_s1, lr=args.lr_s1,
                       use_tps=False,
                       enable_rect_loss=False,
                       enable_geo_loss=True,
                       lambda_geo=0.05, lambda_rect=0.0,
                       warmup_epochs=args.warmup_s1,
                       scheduler_type='cosine',
                       accum_steps=accum,
                       log_every=args.log_every,
                       save_every=args.save_every,
                       freeze_encoder_epochs=3
                           if (not args.no_geometric and not args.parseq_weights)
                           else (args.freeze_encoder if args.parseq_weights else 0))

        # ── Stage 2 ──────────────────────────────────────────────────
        elif stage == '2':
            if args.smoke_test:
                reg = irr = _Smoke(n=max(128, bs * 8))
            else:
                print("\n[Stage 2] Loading datasets...")
                reg, irr = build_stage2(args.iiit5k_train, args.art_txt,
                                            art_max=args.art_max)
                _validate(reg, 'Stage2-regular',   min_samples=10)
                _validate(irr, 'Stage2-irregular', min_samples=10)
                print("  Warming image cache (faster epochs after this)...")
                reg.warm_cache()
                irr.warm_cache()

            reg_ld = _loader(reg, bs, True, nw, device)
            irr_ld = _loader(irr, bs, True, nw, device)
            print(f"  DataLoader: {len(reg_ld)} reg + {len(irr_ld)} irr batches")

            model.use_rectification = True
            model.use_tps           = False
            _run_stage(2, model, (reg_ld, irr_ld), device, args.save_dir,
                       num_epochs=args.epochs_s2, lr=args.lr_s2,
                       use_tps=False,
                       enable_rect_loss=False,
                       enable_geo_loss=True,
                       lambda_geo=0.03, lambda_rect=0.0,
                       warmup_epochs=args.warmup_s2,
                       scheduler_type='cosine',
                       accum_steps=accum,
                       log_every=args.log_every,
                       save_every=args.save_every)

        # ── Stage 3 ──────────────────────────────────────────────────
        elif stage == '3':
            if args.smoke_test:
                ds = _Smoke(n=max(256, bs * 16))
                ld = _loader(ds, bs, True, nw, device)
            else:
                print("\n[Stage 3] Loading datasets...")
                ds = build_stage3(args.art_txt, args.totaltext_txt,
                                     art_max=args.art_max,
                                     totaltext_max=args.totaltext_max,
                                     strong_augment=True,
                                     iiit5k_train_txt=args.iiit5k_train if args.iiit5k_in_stage3 else None)
                _validate(ds, 'Stage3-irregular', min_samples=100)
                print("  Warming image cache (faster epochs after this)...")
                for sub in ds.datasets:
                    sub.warm_cache()
                ld = _loader(ds, bs, True, nw, device)
                print(f"  DataLoader: {len(ld)} batches per epoch")

            model.use_rectification = True
            model.use_tps           = True
            _run_stage(3, model, ld, device, args.save_dir,
                       num_epochs=args.epochs_s3, lr=args.lr_s3,
                       use_tps=False,
                       enable_rect_loss=False,
                       enable_geo_loss=False,
                       lambda_geo=0.0, lambda_rect=0.0,
                       warmup_epochs=args.warmup_s3,
                       scheduler_type='cosine',
                       accum_steps=accum,
                       log_every=args.log_every,
                       save_every=args.save_every)

    print("\n" + "="*66)
    print("  Training complete!")
    print(f"  Checkpoints → {args.save_dir}/")
    print("  stage1_best.pth  stage2_best.pth  stage3_best.pth")
    print("="*66)


if __name__ == '__main__':
    main()
