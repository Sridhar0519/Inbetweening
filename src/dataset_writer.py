"""Dataset writer — exports shot segments to a structured folder of PNG images.

Creates the output directory structure:
    output/
    └── video_name/
        ├── shot_001/
        │   ├── segment_001/
        │   │   ├── key_first.png
        │   │   ├── key_last.png
        │   │   └── inbetweens/
        │   │       ├── frame_001.png
        │   │       └── ...
        │   └── metadata.json
        └── dataset_summary.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .keyframe_analyzer import Segment

logger = logging.getLogger(__name__)

# Default export settings
DEFAULT_HEIGHT = 480
DEFAULT_FORMAT = "png"
DEFAULT_JPEG_QUALITY = 90


def _resize_frame(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Resize a frame to target height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h == target_height:
        return frame
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)


def _save_image(path: str, frame: np.ndarray, img_format: str, jpeg_quality: int) -> None:
    """Save an image in the specified format."""
    if img_format == "jpeg":
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_segment(
    segment: Segment,
    segment_dir: Path,
    target_height: int = DEFAULT_HEIGHT,
    img_format: str = DEFAULT_FORMAT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Dict:
    """Save a single segment (keyframes + inbetweens) to disk.

    Args:
        segment: The Segment object to write.
        segment_dir: Folder path to create for this segment.
        target_height: Output image height in pixels.
        img_format: 'png' or 'jpeg'.
        jpeg_quality: JPEG quality (1-100).

    Returns:
        Metadata dict for this segment.
    """
    segment_dir.mkdir(parents=True, exist_ok=True)
    inbetweens_dir = segment_dir / "inbetweens"
    inbetweens_dir.mkdir(exist_ok=True)

    ext = ".jpg" if img_format == "jpeg" else ".png"

    # Save keyframes
    key_first_resized = _resize_frame(segment.key_first, target_height)
    key_last_resized = _resize_frame(segment.key_last, target_height)
    _save_image(str(segment_dir / f"key_first{ext}"), key_first_resized, img_format, jpeg_quality)
    _save_image(str(segment_dir / f"key_last{ext}"), key_last_resized, img_format, jpeg_quality)

    # Save inbetweens
    for i, (frame_idx, frame) in enumerate(segment.inbetweens, start=1):
        filename = f"frame_{i:04d}{ext}"
        resized = _resize_frame(frame, target_height)
        _save_image(str(inbetweens_dir / filename), resized, img_format, jpeg_quality)

    return {
        "key_first_frame_idx": segment.key_first_idx,
        "key_last_frame_idx": segment.key_last_idx,
        "num_inbetweens": len(segment.inbetweens),
        "inbetween_frame_indices": [idx for idx, _ in segment.inbetweens],
    }


def write_shot(
    shot_index: int,
    segments: List[Segment],
    shot_start_frame: int,
    shot_end_frame: int,
    fps: float,
    video_output_dir: Path,
    target_height: int = DEFAULT_HEIGHT,
    img_format: str = DEFAULT_FORMAT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Dict:
    """Save all segments of a shot to disk with metadata.

    Args:
        shot_index: 1-based index of this shot.
        segments: List of Segment objects for this shot.
        shot_start_frame: First frame index of the shot.
        shot_end_frame: Last frame index of the shot.
        fps: Video frame rate.
        video_output_dir: Root output directory for this video.

    Returns:
        Metadata dict for this shot.
    """
    shot_name = f"shot_{shot_index:03d}"
    shot_dir = video_output_dir / shot_name
    shot_dir.mkdir(parents=True, exist_ok=True)

    segment_metadata = []
    for seg_i, segment in enumerate(segments, start=1):
        seg_name = f"segment_{seg_i:03d}"
        seg_dir = shot_dir / seg_name
        seg_meta = write_segment(
            segment, seg_dir,
            target_height=target_height,
            img_format=img_format,
            jpeg_quality=jpeg_quality,
        )
        seg_meta["segment_name"] = seg_name
        segment_metadata.append(seg_meta)

    total_frames = shot_end_frame - shot_start_frame + 1
    shot_meta = {
        "shot_name": shot_name,
        "start_frame": shot_start_frame,
        "end_frame": shot_end_frame,
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps, 3),
        "fps": fps,
        "num_segments": len(segments),
        "segments": segment_metadata,
    }

    # Write per-shot metadata
    meta_path = shot_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(shot_meta, f, indent=2)

    logger.info(
        "Wrote %s: %d segment(s), %d total frames.",
        shot_name,
        len(segments),
        total_frames,
    )

    return shot_meta


def write_dataset_summary(
    video_name: str,
    shot_metadata: List[Dict],
    fps: float,
    total_video_frames: int,
    video_output_dir: Path,
) -> Path:
    """Write a top-level summary JSON for the entire video dataset.

    Args:
        video_name: Name of the source video (without extension).
        shot_metadata: List of per-shot metadata dicts.
        fps: Video frame rate.
        total_video_frames: Total frames in the source video.
        video_output_dir: Root output directory for this video.

    Returns:
        Path to the written summary JSON.
    """
    total_segments = sum(s["num_segments"] for s in shot_metadata)
    total_inbetweens = sum(
        seg["num_inbetweens"]
        for shot in shot_metadata
        for seg in shot["segments"]
    )

    summary = {
        "video_name": video_name,
        "fps": fps,
        "total_video_frames": total_video_frames,
        "total_shots": len(shot_metadata),
        "total_segments": total_segments,
        "total_inbetween_frames": total_inbetweens,
        "shots": shot_metadata,
    }

    summary_path = video_output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Dataset summary: %d shots, %d segments, %d inbetween frames.",
        len(shot_metadata),
        total_segments,
        total_inbetweens,
    )

    return summary_path
