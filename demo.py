# demo.py
"""
Quick demo: Run inference on one or more local images using a trained checkpoint.

Usage examples:

Single image:
    python demo.py --checkpoint checkpoints/stage3_best.pth --image path/to/your_image.jpg

Multiple images (folder):
    python demo.py --checkpoint checkpoints/stage3_best.pth --image_folder path/to/folder --output predictions.txt

Show help:
    python demo.py --help
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as TF

from models.model import PARSeqGeoAware, set_charset, improved_ctc_decode, CHARSET


def load_model(checkpoint_path: str, device: torch.device):
    set_charset(36)  # change to 64 if your checkpoint uses extended charset

    model = PARSeqGeoAware(
        num_chars=len(CHARSET) + 1,
        use_geometric=True,
        use_rectification=True,
        use_tps=False,
        use_attention=False,
        max_len=25
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def preprocess_image(image_path: str):
    transform = TF.Compose([
        TF.Resize((64, 256)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)  # add batch dim


def predict(model, image_tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        log_probs, _ = model(image_tensor)
        pred = improved_ctc_decode(log_probs)
    return pred[0]  # take first (and only) prediction


def main():
    parser = argparse.ArgumentParser(description="PARSeq-GeoAware single image inference")
    parser.add_argument('--checkpoint', required=True, help="Path to .pth checkpoint")
    parser.add_argument('--image', help="Path to single image")
    parser.add_argument('--image_folder', help="Folder with multiple images")
    parser.add_argument('--output', default="predictions.txt", help="Where to save results (txt)")
    parser.add_argument('--charset', type=int, default=36, choices=[36,64])

    args = parser.parse_args()

    if not args.image and not args.image_folder:
        parser.error("You must provide either --image or --image_folder")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    results = []

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found → {args.image}")
            return
        img_tensor = preprocess_image(args.image)
        pred = predict(model, img_tensor, device)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {pred}")
        results.append(f"{args.image}\t{pred}")

    elif args.image_folder:
        folder = Path(args.image_folder)
        if not folder.is_dir():
            print(f"Error: Folder not found → {folder}")
            return

        image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not image_paths:
            print("No .jpg or .png images found in folder.")
            return

        print(f"Found {len(image_paths)} images in {folder}")

        for img_path in image_paths:
            try:
                img_tensor = preprocess_image(str(img_path))
                pred = predict(model, img_tensor, device)
                print(f"{img_path.name:30} → {pred}")
                results.append(f"{img_path}\t{pred}")
            except Exception as e:
                print(f"Failed on {img_path.name}: {e}")

    # Save results
    if results:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()