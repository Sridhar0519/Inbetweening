"""Keyframe detection within a shot using structural similarity (SSIM).

Analyzes consecutive frame pairs to find significant visual changes
(keyframes). Groups frames into segments where each segment contains
a start keyframe, inbetween frames, and an end keyframe.

Designed for animated content where keyframes represent distinct poses
and inbetweens are the transitional drawings.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A keyframe pair and its inbetween frames.

    Attributes:
        key_first_idx: Frame index of the first keyframe.
        key_last_idx: Frame index of the last keyframe.
        key_first: The first keyframe image (BGR numpy array).
        key_last: The last keyframe image (BGR numpy array).
        inbetweens: List of (frame_index, frame_array) for frames
            between the two keyframes.
    """
    key_first_idx: int
    key_last_idx: int
    key_first: np.ndarray
    key_last: np.ndarray
    inbetweens: List[Tuple[int, np.ndarray]] = field(default_factory=list)


def _compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute SSIM between two BGR frames.

    Converts to grayscale first for efficiency and robustness.
    Returns a value in [0, 1] where 1 = identical.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


def detect_keyframes(
    frames: List[Tuple[int, np.ndarray]],
    keyframe_threshold: float = 0.85,
    min_inbetweens: int = 1,
) -> List[Segment]:
    """Detect keyframes and group frames into segments within a shot.

    Computes SSIM between consecutive frames. When the similarity drops
    below `keyframe_threshold`, the current frame is marked as a keyframe
    boundary (indicating a significant pose/visual change).

    Args:
        frames: List of (frame_index, frame_array) ordered by index.
            These are all frames within a single shot.
        keyframe_threshold: SSIM value below which a frame transition
            is considered a keyframe change. Lower values = only very
            large changes count as keyframes. Range 0.0-1.0.
            For animation: 0.80-0.90 typically works well.
        min_inbetweens: Minimum number of inbetween frames required
            for a segment to be kept. Segments with fewer inbetweens
            are discarded (merged into adjacent segments).

    Returns:
        List of Segment objects, each containing a keyframe pair and
        its inbetween frames. Returns empty list if the shot has
        fewer than 3 frames.
    """
    if len(frames) < 3:
        logger.debug("Shot has fewer than 3 frames — skipping keyframe detection.")
        return []

    # Step 1: Compute per-frame SSIM differences
    ssim_scores = []
    for i in range(len(frames) - 1):
        score = _compute_ssim(frames[i][1], frames[i + 1][1])
        ssim_scores.append(score)

    logger.debug(
        "SSIM range: min=%.4f, max=%.4f, mean=%.4f",
        min(ssim_scores),
        max(ssim_scores),
        sum(ssim_scores) / len(ssim_scores),
    )

    # Step 2: Identify keyframe indices
    # The first frame is always a keyframe.
    # Frames where the preceding transition has low SSIM are keyframes.
    # The last frame is always a keyframe.
    keyframe_positions = [0]  # first frame is always a keyframe

    for i, score in enumerate(ssim_scores):
        if score < keyframe_threshold:
            # The frame AFTER the low-similarity transition is a keyframe
            keyframe_positions.append(i + 1)

    # Ensure last frame is a keyframe
    last_pos = len(frames) - 1
    if keyframe_positions[-1] != last_pos:
        keyframe_positions.append(last_pos)

    # Deduplicate and sort
    keyframe_positions = sorted(set(keyframe_positions))

    logger.debug(
        "Found %d keyframe(s) at positions: %s",
        len(keyframe_positions),
        keyframe_positions,
    )

    # Step 3: Build segments from consecutive keyframe pairs
    segments: List[Segment] = []

    for i in range(len(keyframe_positions) - 1):
        kf_start_pos = keyframe_positions[i]
        kf_end_pos = keyframe_positions[i + 1]

        # Inbetweens are all frames strictly between the two keyframes
        inbetween_positions = list(range(kf_start_pos + 1, kf_end_pos))

        if len(inbetween_positions) < min_inbetweens:
            logger.debug(
                "Skipping segment (keyframes at positions %d-%d): "
                "only %d inbetween(s), minimum is %d.",
                kf_start_pos,
                kf_end_pos,
                len(inbetween_positions),
                min_inbetweens,
            )
            continue

        segment = Segment(
            key_first_idx=frames[kf_start_pos][0],
            key_last_idx=frames[kf_end_pos][0],
            key_first=frames[kf_start_pos][1],
            key_last=frames[kf_end_pos][1],
            inbetweens=[
                (frames[pos][0], frames[pos][1]) for pos in inbetween_positions
            ],
        )
        segments.append(segment)

    logger.info(
        "Shot segmented into %d segment(s) from %d keyframe(s) "
        "(%d frames total, threshold=%.2f).",
        len(segments),
        len(keyframe_positions),
        len(frames),
        keyframe_threshold,
    )

    # If no segments were created but we have enough frames, fall back to
    # treating the entire shot as one segment (first frame → last frame).
    if not segments and len(frames) >= 3:
        logger.info(
            "No segments met the criteria — falling back to full-shot segment."
        )
        segment = Segment(
            key_first_idx=frames[0][0],
            key_last_idx=frames[-1][0],
            key_first=frames[0][1],
            key_last=frames[-1][1],
            inbetweens=[(idx, frame) for idx, frame in frames[1:-1]],
        )
        segments.append(segment)

    return segments
