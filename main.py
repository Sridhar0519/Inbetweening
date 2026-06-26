#!/usr/bin/env python3
"""Shot Segregation — CLI tool to create a keyframe dataset from animated videos.

Takes animated MP4 videos as input, detects shot boundaries, identifies
keyframe pairs within each shot via SSIM-based visual difference analysis,
and exports a structured PNG dataset suitable for training a GAN model
to generate inbetween frames.

Usage:
    python main.py --input video.mp4 --output ./output
    python main.py --input ./videos/ --output ./output --shot-threshold 25
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.shot_detector import detect_shots
from src.frame_extractor import extract_frames
from src.keyframe_analyzer import detect_keyframes
from src.dataset_writer import write_shot, write_dataset_summary
from src.shot_classifier import classify_shot, ShotType
from src.review_gui import review_dataset

logger = logging.getLogger("shot_segregation")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with a human-readable format."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def process_video(
    video_path: Path,
    output_dir: Path,
    shot_threshold: float,
    keyframe_threshold: float,
    min_inbetweens: int,
    min_scene_len: int,
    characters_only: bool = False,
    character_threshold: float = 0.35,
    output_height: int = 480,
    image_format: str = "png",
    jpeg_quality: int = 90,
) -> None:
    """Process a single video file through the full pipeline.

    1. Detect shot boundaries
    2. For each shot: extract frames → detect keyframes → build segments
    3. Write dataset to disk with metadata
    """
    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Processing: %s", video_path.name)
    logger.info("=" * 60)

    # Step 1: Detect shots
    shots, fps, total_frames = detect_shots(
        str(video_path),
        threshold=shot_threshold,
        min_scene_len=min_scene_len,
    )
    logger.info("Found %d shot(s) in %d frames (%.2f fps).", len(shots), total_frames, fps)

    if not shots:
        logger.warning("No shots detected in %s — skipping.", video_path.name)
        return

    # Step 2-3: Process each shot
    all_shot_metadata = []
    shot_progress = tqdm(
        enumerate(shots, start=1),
        total=len(shots),
        desc=f"  Shots ({video_name})",
        unit="shot",
    )

    for shot_idx, (start_frame, end_frame) in shot_progress:
        shot_progress.set_postfix(frames=f"{start_frame}-{end_frame}")

        # Extract frames for this shot
        frames = extract_frames(str(video_path), start_frame, end_frame)

        if len(frames) < 3:
            logger.debug(
                "Shot %03d has only %d frames — skipping.", shot_idx, len(frames)
            )
            continue

        # Classify shot content if filtering is enabled
        if characters_only:
            shot_type, confidence = classify_shot(
                frames, character_threshold=character_threshold
            )
            if shot_type != ShotType.CHARACTER:
                logger.info(
                    "Shot %03d classified as %s (confidence=%.2f) — skipping.",
                    shot_idx,
                    shot_type.value,
                    confidence,
                )
                continue
            logger.info(
                "Shot %03d classified as CHARACTER (confidence=%.2f) — keeping.",
                shot_idx,
                confidence,
            )

        # Detect keyframes and create segments
        segments = detect_keyframes(
            frames,
            keyframe_threshold=keyframe_threshold,
            min_inbetweens=min_inbetweens,
        )

        if not segments:
            logger.debug("Shot %03d produced no valid segments — skipping.", shot_idx)
            continue

        # Write to disk
        shot_meta = write_shot(
            shot_index=shot_idx,
            segments=segments,
            shot_start_frame=start_frame,
            shot_end_frame=end_frame,
            fps=fps,
            video_output_dir=video_output_dir,
            target_height=output_height,
            img_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        all_shot_metadata.append(shot_meta)

    # Step 4: Write summary
    if all_shot_metadata:
        summary_path = write_dataset_summary(
            video_name=video_name,
            shot_metadata=all_shot_metadata,
            fps=fps,
            total_video_frames=total_frames,
            video_output_dir=video_output_dir,
        )
        logger.info("Dataset written to: %s", video_output_dir)
        logger.info("Summary: %s", summary_path)
    else:
        logger.warning("No valid segments produced for %s.", video_path.name)


def find_videos(input_path: Path) -> list:
    """Find all MP4 video files from an input path (file or directory)."""
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    if input_path.is_file():
        if input_path.suffix.lower() in video_extensions:
            return [input_path]
        else:
            logger.error("Unsupported file format: %s", input_path.suffix)
            return []

    if input_path.is_dir():
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in video_extensions
        )
        if not videos:
            logger.error("No video files found in: %s", input_path)
        return videos

    logger.error("Input path does not exist: %s", input_path)
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a keyframe+inbetween dataset from animated videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input video.mp4 --output ./output
  python main.py --input ./videos/ --output ./output
  python main.py --input video.mp4 --output ./output --shot-threshold 25 --keyframe-threshold 0.82

Threshold tuning:
  --shot-threshold    Controls shot/scene boundary sensitivity.
                      Lower = more shots detected. Default: 27.0
                      For animated content, try 20-30.

  --keyframe-threshold  SSIM threshold for keyframe detection within shots.
                        Lower = only very large visual changes count.
                        Higher = more keyframes detected. Default: 0.85
                        For animation, try 0.80-0.92.

  --min-inbetweens    Minimum inbetween frames per segment. Segments with
                      fewer inbetweens are discarded. Default: 1
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to a video file or directory of video files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Output directory for the dataset.",
    )
    parser.add_argument(
        "--shot-threshold",
        type=float,
        default=27.0,
        help="Scene-change detection threshold (lower=more sensitive). Default: 27.0",
    )
    parser.add_argument(
        "--keyframe-threshold",
        type=float,
        default=0.85,
        help="SSIM threshold for keyframe detection (0.0-1.0). Default: 0.85",
    )
    parser.add_argument(
        "--min-inbetweens",
        type=int,
        default=1,
        help="Minimum inbetween frames per segment. Default: 1",
    )
    parser.add_argument(
        "--min-scene-len",
        type=int,
        default=6,
        help="Minimum frames per detected scene/shot. Default: 6",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output image height in pixels (aspect ratio preserved). Default: 480",
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpeg"],
        default="png",
        help="Output image format: png (lossless) or jpeg (smaller). Default: png",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality 1-100 (only used with --image-format jpeg). Default: 90",
    )
    parser.add_argument(
        "--characters-only",
        action="store_true",
        help="Only keep shots containing characters. Filters out blank, text, and background-only shots.",
    )
    parser.add_argument(
        "--character-threshold",
        type=float,
        default=0.35,
        help="Character detection sensitivity (0.0-1.0). Lower=more permissive. Default: 0.35",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Launch interactive GUI after export to review and keep/delete segments.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Shot Segregation — Dataset Creator")
    logger.info("Input:  %s", args.input.resolve())
    logger.info("Output: %s", args.output.resolve())

    videos = find_videos(args.input)
    if not videos:
        logger.error("No videos to process. Exiting.")
        sys.exit(1)

    logger.info("Found %d video(s) to process.", len(videos))

    args.output.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        process_video(
            video_path=video_path,
            output_dir=args.output,
            shot_threshold=args.shot_threshold,
            keyframe_threshold=args.keyframe_threshold,
            min_inbetweens=args.min_inbetweens,
            min_scene_len=args.min_scene_len,
            characters_only=args.characters_only,
            character_threshold=args.character_threshold,
            output_height=args.height,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )

    logger.info("Done. Dataset written to: %s", args.output.resolve())

    if args.review:
        logger.info("Launching interactive review GUI...")
        review_dataset(args.output)


if __name__ == "__main__":
    main()
