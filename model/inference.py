"""Inference script -- generate inbetween frames from a trained model.

Usage:
    python -m model.inference \
        --checkpoint training_output/checkpoints/latest.pt \
        --key-first frame_a.png \
        --key-last frame_b.png \
        --num-frames 8 \
        --output ./generated/
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .generator import UNetGenerator


def load_model(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    """Load trained generator from checkpoint."""
    generator = UNetGenerator(in_channels=7, out_channels=3, base_features=64)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    generator.eval()
    return generator


def generate_inbetweens(
    generator: UNetGenerator,
    key_first_path: str,
    key_last_path: str,
    num_frames: int,
    image_size: int,
    device: torch.device,
) -> list:
    """Generate N inbetween frames between two keyframes.

    Args:
        generator: Trained generator model.
        key_first_path: Path to first keyframe image.
        key_last_path: Path to last keyframe image.
        num_frames: Number of inbetween frames to generate.
        image_size: Processing size (should match training size).
        device: Torch device.

    Returns:
        List of PIL Images (generated inbetweens).
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    key_first = transform(Image.open(key_first_path).convert('RGB')).unsqueeze(0).to(device)
    key_last = transform(Image.open(key_last_path).convert('RGB')).unsqueeze(0).to(device)

    generated_frames = []

    with torch.no_grad():
        for i in range(num_frames):
            t_val = (i + 1) / (num_frames + 1)
            t = torch.tensor([[t_val]], dtype=torch.float32).to(device)

            output = generator(key_first, key_last, t)

            # Convert to PIL
            img = output.squeeze(0).cpu()
            img = (img + 1) / 2  # [-1,1] -> [0,1]
            img = img.clamp(0, 1)
            img = transforms.ToPILImage()(img)
            generated_frames.append(img)

    return generated_frames


def main():
    parser = argparse.ArgumentParser(
        description="Generate inbetween frames using a trained model."
    )
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--key-first', type=str, required=True,
                        help='Path to first keyframe image')
    parser.add_argument('--key-last', type=str, required=True,
                        help='Path to last keyframe image')
    parser.add_argument('--num-frames', '-n', type=int, default=8,
                        help='Number of inbetween frames to generate (default: 8)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Processing image size, must match training (default: 256)')
    parser.add_argument('--output', '-o', type=str, default='./generated',
                        help='Output directory for generated frames')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    generator = load_model(args.checkpoint, device)
    print(f"Loaded model from: {args.checkpoint}")

    # Generate
    frames = generate_inbetweens(
        generator, args.key_first, args.key_last,
        args.num_frames, args.image_size, device,
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        path = output_dir / f"inbetween_{i + 1:04d}.png"
        frame.save(path)

    print(f"Generated {len(frames)} frames -> {output_dir}/")


if __name__ == '__main__':
    main()
