#!/usr/bin/env python3
"""Validate an exported dataset for integrity.

Checks that every segment has key_first.png, key_last.png, and at least
one inbetween frame. Reports any missing files or structural issues.

Usage:
    python -m src.validate_dataset ./output/video_name
    python -m src.validate_dataset ./output
"""

import argparse
import json
import sys
from pathlib import Path


def validate_segment(segment_dir: Path) -> list:
    """Validate a single segment directory. Returns list of error strings."""
    errors = []

    # Support both PNG and JPEG formats
    key_first_exists = (segment_dir / "key_first.png").exists() or (segment_dir / "key_first.jpg").exists()
    key_last_exists = (segment_dir / "key_last.png").exists() or (segment_dir / "key_last.jpg").exists()
    inbetweens_dir = segment_dir / "inbetweens"

    if not key_first_exists:
        errors.append(f"Missing key_first.png/jpg in {segment_dir}")
    if not key_last_exists:
        errors.append(f"Missing key_last.png/jpg in {segment_dir}")
    if not inbetweens_dir.exists():
        errors.append(f"Missing inbetweens/ directory in {segment_dir}")
    elif not any(inbetweens_dir.iterdir()):
        errors.append(f"Empty inbetweens/ directory in {segment_dir}")

    return errors


def validate_shot(shot_dir: Path) -> list:
    """Validate a single shot directory. Returns list of error strings."""
    errors = []

    metadata_path = shot_dir / "metadata.json"
    if not metadata_path.exists():
        errors.append(f"Missing metadata.json in {shot_dir}")

    segment_dirs = sorted(
        d for d in shot_dir.iterdir()
        if d.is_dir() and d.name.startswith("segment_")
    )

    if not segment_dirs:
        errors.append(f"No segment directories in {shot_dir}")
        return errors

    for seg_dir in segment_dirs:
        errors.extend(validate_segment(seg_dir))

    # Cross-check with metadata if present
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        expected_segments = meta.get("num_segments", 0)
        if len(segment_dirs) != expected_segments:
            errors.append(
                f"{shot_dir}: metadata says {expected_segments} segments, "
                f"found {len(segment_dirs)} directories."
            )

    return errors


def validate_video_dataset(video_dir: Path) -> list:
    """Validate a video-level dataset directory. Returns list of error strings."""
    errors = []

    summary_path = video_dir / "dataset_summary.json"
    if not summary_path.exists():
        errors.append(f"Missing dataset_summary.json in {video_dir}")

    shot_dirs = sorted(
        d for d in video_dir.iterdir()
        if d.is_dir() and d.name.startswith("shot_")
    )

    if not shot_dirs:
        errors.append(f"No shot directories in {video_dir}")
        return errors

    for shot_dir in shot_dirs:
        errors.extend(validate_shot(shot_dir))

    return errors


def validate_dataset(root_dir: Path) -> list:
    """Validate a dataset root. Auto-detects if it's a video dir or multi-video dir."""
    root_dir = Path(root_dir)
    errors = []

    if not root_dir.exists():
        return [f"Path does not exist: {root_dir}"]

    # Check if this is a video-level dir (has shot_* subdirs)
    has_shots = any(
        d.name.startswith("shot_") for d in root_dir.iterdir() if d.is_dir()
    )
    if has_shots:
        return validate_video_dataset(root_dir)

    # Otherwise try each subdirectory as a video dataset
    video_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    if not video_dirs:
        return [f"No video dataset directories found in {root_dir}"]

    for video_dir in video_dirs:
        errors.extend(validate_video_dataset(video_dir))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a shot segregation dataset for integrity."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to dataset root, video directory, or shot directory.",
    )
    args = parser.parse_args()

    errors = validate_dataset(args.path)

    if errors:
        print(f"\nValidation FAILED — {len(errors)} error(s):\n")
        for err in errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print("\nValidation PASSED — dataset is intact.")
        sys.exit(0)


if __name__ == "__main__":
    main()
