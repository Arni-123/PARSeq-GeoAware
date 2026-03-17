"""
demo.py — PARSeq-GeoAware single-image and batch inference
===========================================================

Usage:
  # Single image
  python demo.py --checkpoint checkpoints/stage3_best.pth --image path/to/img.jpg

  # Folder of images
  python demo.py --checkpoint checkpoints/stage3_best.pth --image_folder path/to/imgs/

  # Save results to txt
  python demo.py --checkpoint checkpoints/stage3_best.pth --image_folder imgs/ --output predictions.txt

  # Use 64-char charset
  python demo.py --checkpoint checkpoints/stage3_best.pth --image img.jpg --charset 64

Architecture confirmed from stage3_best.pth (281 tensors):
  encoder.*        ViT-Small, 16×16 patches, pos_embed [1,65,384] → 64 tokens + CLS
  gfe.*            4-stage ResNet + boundary/orientation/curvature heads + geo_attention
  fusion.*         Cross-attention + gate_proj, v_proj [256,384], g_proj [256,260]
  rectification.*  affine_mlp + tps_mlp + ctrl_pts [8,2]
  head.*           Linear(384→37): 36 chars + 1 CTC blank
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as TF
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import (
    PARSeqGeoAware,
    set_charset,
    CHARSET,
    BLANK_IDX,
    improved_ctc_decode,
)

# ── Image transform — must match training exactly ────────────────────────────
# SimpleLineDataset in train__20_.py: Resize(64,256), grayscale, ToTensor,
# Normalize(mean=0.5, std=0.5)
_TRANSFORM = TF.Compose([
    TF.Resize((64, 256)),
    TF.Grayscale(1),
    TF.ToTensor(),
    TF.Normalize(mean=[0.5], std=[0.5]),
])


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device,
               charset: int = 36) -> PARSeqGeoAware:
    """
    Build PARSeqGeoAware and load weights from checkpoint.

    Architecture flags are read from the checkpoint's 'config' key so the
    model is always built to match what was actually trained.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ── Read config saved by train__20_.py _run_stage ────────────────────
    cfg = ckpt.get('config', {})
    use_geo  = cfg.get('use_geometric',     True)
    use_rect = cfg.get('use_rectification', True)
    use_tps  = cfg.get('use_tps',           True)
    n_chars  = cfg.get('num_chars',         len(CHARSET) + 1)

    print(f"  Stage          : {ckpt.get('stage', '?')}")
    print(f"  Epoch          : {ckpt.get('epoch', '?')}")
    print(f"  CTC loss       : {ckpt.get('ctc_loss', '?')}")
    print(f"  use_geometric  : {use_geo}")
    print(f"  use_rect       : {use_rect}")
    print(f"  use_tps        : {use_tps}")
    print(f"  num_chars      : {n_chars}")

    model = PARSeqGeoAware(
        num_chars         = n_chars,
        use_geometric     = use_geo,
        use_rectification = use_rect,
        use_tps           = use_tps,
        use_attention     = False,
        max_len           = 25,
    ).to(device)

    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  ⚠  Missing keys  ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print(f"  ✅ Weights loaded — strict match")

    model.eval()
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> torch.Tensor:
    """Load image → (1, 1, 64, 256) tensor in [-1, 1]."""
    img = Image.open(image_path).convert('RGB')
    return _TRANSFORM(img).unsqueeze(0)   # (1, 1, 64, 256)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: PARSeqGeoAware,
            image_tensor: torch.Tensor,
            device: torch.device) -> str:
    """
    Run one forward pass and decode the CTC output.

    train__20_.py calls model(images, return_features=True) which returns
    (log_probs, feats) where log_probs is already stacked (T, B, C).

    Fallback: if model doesn't support return_features, stack the raw tuple
    manually (skip CLS token at index 0).
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        try:
            # Primary path — matches train__20_.py forward API
            log_probs, _ = model(image_tensor, return_features=True)
            # log_probs: (T, B, C) already stacked inside model.forward()

        except TypeError:
            # Fallback for older model.py without return_features arg
            out = model(image_tensor)
            if isinstance(out, (list, tuple)):
                if isinstance(out[0], torch.Tensor) and out[0].dim() == 2:
                    # Raw tuple of N × (B, C) — skip CLS token (index 0)
                    log_probs = torch.stack(list(out[1:]), dim=0)  # (64, B, C)
                else:
                    log_probs = out[0]
            else:
                log_probs = out

        log_probs = log_probs.log_softmax(2)   # (T, B, C)
        predictions = improved_ctc_decode(log_probs)

    return predictions[0]   # batch index 0


# ── Single image helper ───────────────────────────────────────────────────────

def run_single(model, image_path, device):
    img_tensor = preprocess(image_path)
    pred = predict(model, img_tensor, device)
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Prediction: {pred}")
    return pred


# ── Visualise (optional) ──────────────────────────────────────────────────────

def visualise(results, save_path='inference_results.png'):
    """Plot images with predictions. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualisation")
        return

    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, (path, pred) in zip(axes, results):
        img = Image.open(path).convert('RGB')
        ax.imshow(img)
        ax.set_title(f'"{pred}"', fontsize=14, fontweight='bold',
                     color='#27ae60', pad=8)
        ax.set_xlabel(os.path.basename(path), fontsize=9, color='gray')
        ax.axis('off')

    plt.suptitle('PARSeq-GeoAware — Inference Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='PARSeq-GeoAware — single image or batch inference')
    p.add_argument('--checkpoint',    required=True,
                   help='Path to .pth checkpoint (e.g. checkpoints/stage3_best.pth)')
    p.add_argument('--image',         default=None,
                   help='Path to a single image')
    p.add_argument('--image_folder',  default=None,
                   help='Folder containing .jpg/.png images')
    p.add_argument('--output',        default='predictions.txt',
                   help='Where to save results (one image\\tprediction per line)')
    p.add_argument('--charset',       type=int, default=36, choices=[36, 64],
                   help='Character set size (default 36)')
    p.add_argument('--visualise',     action='store_true',
                   help='Plot images with predictions using matplotlib')
    p.add_argument('--device',        default=None,
                   help='Force device: cuda / cpu (default: auto)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Charset ───────────────────────────────────────────────────────────
    set_charset(args.charset)
    print(f"Charset: {args.charset} ({len(CHARSET)} chars, blank={BLANK_IDX})")

    # ── Model ─────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device, args.charset)

    # ── Collect images ────────────────────────────────────────────────────
    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)

    if args.image_folder:
        if not os.path.isdir(args.image_folder):
            print(f"❌ Folder not found: {args.image_folder}")
            sys.exit(1)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths += sorted([
            os.path.join(args.image_folder, f)
            for f in os.listdir(args.image_folder)
            if os.path.splitext(f)[1].lower() in exts
        ])

    if not image_paths:
        print("❌ No images provided. Use --image or --image_folder.")
        sys.exit(1)

    print(f"\nRunning inference on {len(image_paths)} image(s)...\n")

    # ── Inference loop ────────────────────────────────────────────────────
    results = []
    for img_path in image_paths:
        try:
            pred = run_single(model, img_path, device)
            results.append((img_path, pred))
        except Exception as e:
            print(f"  ⚠  Failed on {img_path}: {e}")
            results.append((img_path, '<error>'))

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for path, pred in results:
            f.write(f"{path}\t{pred}\n")
    print(f"\nResults saved to: {args.output}")

    # ── Optional visualisation ────────────────────────────────────────────
    if args.visualise:
        visualise(results)


if __name__ == '__main__':
    main()
