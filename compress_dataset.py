#!/usr/bin/env python3
"""Compress an existing dataset to fit a target size.

Resizes all PNG images to a lower resolution and optionally converts
to JPEG to dramatically reduce file size.

Usage:
    python compress_dataset.py --input ./output --target-size 5
    python compress_dataset.py --input ./output --target-size 5 --height 256
    python compress_dataset.py --input ./output --target-size 5 --format jpeg --quality 90
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def get_dataset_stats(root_dir: Path):
    """Get current dataset size and image count."""
    total_size = 0
    image_files = list(root_dir.rglob("*.png")) + list(root_dir.rglob("*.jpg"))
    for f in image_files:
        total_size += f.stat().st_size
    return image_files, total_size


def estimate_target_height(
    sample_files: list,
    current_total_size: int,
    target_size_bytes: int,
    output_format: str,
    jpeg_quality: int,
) -> int:
    """Estimate the height needed to hit the target total size.

    Samples a few images at different heights and extrapolates.
    """
    if not sample_files:
        return 256

    # Sample at different heights to find the relationship
    test_heights = [128, 192, 256, 320, 384]
    size_per_image_at_height = {}

    sample = sample_files[:20]  # Sample 20 images

    for test_h in test_heights:
        sizes = []
        for f in sample:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = test_h / h
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, test_h), interpolation=cv2.INTER_AREA)

            # Encode to estimate compressed size
            if output_format == "jpeg":
                _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                _, buf = cv2.imencode('.png', resized, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            sizes.append(len(buf))

        if sizes:
            size_per_image_at_height[test_h] = np.mean(sizes)

    # Find height that gets us closest to target
    num_images = len(list(Path(sample_files[0]).parent.parent.parent.parent.rglob("*.png"))) if sample_files else 0
    # Use the number from argument
    num_images = len(sample_files)

    target_per_image = target_size_bytes / num_images

    best_height = 256
    best_diff = float('inf')
    for h, avg_size in size_per_image_at_height.items():
        diff = abs(avg_size - target_per_image)
        if diff < best_diff:
            best_diff = diff
            best_height = h

    return best_height


def compress_dataset(
    root_dir: Path,
    target_height: int,
    output_format: str = "png",
    jpeg_quality: int = 90,
    target_size_gb: float = 5.0,
    dry_run: bool = False,
):
    """Compress all images in the dataset in-place.

    Args:
        root_dir: Dataset root directory.
        target_height: Target image height in pixels.
        output_format: 'png' or 'jpeg'.
        jpeg_quality: JPEG quality (1-100). Only used if format is jpeg.
        target_size_gb: Target total size in GB (for reporting).
        dry_run: If True, only print what would be done without modifying files.
    """
    image_files, current_size = get_dataset_stats(root_dir)
    current_gb = current_size / (1024 ** 3)

    print(f"Current dataset: {len(image_files)} images, {current_gb:.2f} GB")
    print(f"Target: ~{target_size_gb:.1f} GB")
    print(f"Resize to: {target_height}p height (aspect ratio preserved)")
    print(f"Output format: {output_format.upper()}" + (f" (quality={jpeg_quality})" if output_format == "jpeg" else ""))
    print()

    if dry_run:
        # Estimate with a sample
        sample_sizes = []
        for f in image_files[:50]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
            if output_format == "jpeg":
                _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                _, buf = cv2.imencode('.png', resized, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            sample_sizes.append(len(buf))

        avg_size = np.mean(sample_sizes)
        estimated_total = avg_size * len(image_files) / (1024 ** 3)
        print(f"[DRY RUN] Estimated size after compression: {estimated_total:.2f} GB")
        print(f"[DRY RUN] Average image size: {avg_size / 1024:.1f} KB")
        print(f"[DRY RUN] Compression ratio: {current_gb / estimated_total:.1f}x")
        return

    # Process all images
    new_total_size = 0
    ext = ".jpg" if output_format == "jpeg" else ".png"

    for img_path in tqdm(image_files, desc="Compressing", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Cannot read {img_path}, skipping.")
            continue

        h, w = img.shape[:2]

        # Skip if already smaller than target
        if h <= target_height:
            resized = img
        else:
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

        # Determine output path
        if output_format == "jpeg" and img_path.suffix == ".png":
            new_path = img_path.with_suffix(".jpg")
            # Write new file
            cv2.imwrite(
                str(new_path), resized,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            # Remove old PNG
            img_path.unlink()
        else:
            # Overwrite in-place (PNG with higher compression)
            cv2.imwrite(
                str(img_path), resized,
                [cv2.IMWRITE_PNG_COMPRESSION, 6]
            )
            new_path = img_path

        new_total_size += new_path.stat().st_size

    new_gb = new_total_size / (1024 ** 3)
    compression_ratio = current_gb / new_gb if new_gb > 0 else 0

    print(f"\nDone!")
    print(f"Before: {current_gb:.2f} GB")
    print(f"After:  {new_gb:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Saved: {current_gb - new_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Compress an existing dataset by resizing images to a lower resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate compression without modifying (dry run):
  python compress_dataset.py --input ./output --target-size 5 --dry-run

  # Compress to ~5GB using PNG at 256p:
  python compress_dataset.py --input ./output --target-size 5 --height 256

  # Compress aggressively using JPEG for smaller size:
  python compress_dataset.py --input ./output --target-size 5 --format jpeg --quality 85

  # Custom height:
  python compress_dataset.py --input ./output --height 192
        """,
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--target-size', type=float, default=5.0,
                        help='Target total size in GB (default: 5.0)')
    parser.add_argument('--height', type=int, default=None,
                        help='Target image height in pixels. If not set, auto-calculated from target-size.')
    parser.add_argument('--format', choices=['png', 'jpeg'], default='png',
                        help='Output image format (default: png)')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality 1-100 (default: 90). Only used with --format jpeg.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Estimate result without modifying files.')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist.")
        sys.exit(1)

    # Auto-calculate height if not specified
    if args.height is None:
        print("Auto-calculating target height...")
        image_files, current_size = get_dataset_stats(args.input)
        target_bytes = args.target_size * (1024 ** 3)
        height = estimate_target_height(
            image_files, current_size, target_bytes, args.format, args.quality
        )
        print(f"Estimated optimal height: {height}p")
        args.height = height

    compress_dataset(
        root_dir=args.input,
        target_height=args.height,
        output_format=args.format,
        jpeg_quality=args.quality,
        target_size_gb=args.target_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
