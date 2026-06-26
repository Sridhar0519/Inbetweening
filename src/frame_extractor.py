"""Frame extraction from video files using OpenCV.

Reads frames within a given range from a video, returning them as
in-memory numpy arrays. Processes one shot at a time to keep memory
usage bounded.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
) -> List[Tuple[int, np.ndarray]]:
    """Extract frames from a video within a given range.

    Args:
        video_path: Path to the input video file.
        start_frame: First frame index to extract (inclusive).
        end_frame: Last frame index to extract (inclusive).

    Returns:
        List of (frame_index, frame_array) tuples. frame_array is a
        BGR numpy array as returned by OpenCV.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video cannot be opened.
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[Tuple[int, np.ndarray]] = []

    try:
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "Failed to read frame %d (expected up to %d). Stopping early.",
                    frame_idx,
                    end_frame,
                )
                break
            frames.append((frame_idx, frame))
    finally:
        cap.release()

    logger.debug(
        "Extracted %d frames (range %d-%d) from %s",
        len(frames),
        start_frame,
        end_frame,
        Path(video_path).name,
    )

    return frames
