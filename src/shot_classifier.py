"""Shot classifier — filters shots to keep only those containing characters.

Uses a combination of heuristics to classify shot frames as:
- BLANK: Nearly uniform color, no meaningful content
- TEXT: Dominated by text (title cards, credits, subtitles)
- BACKGROUND: Scenery/environment without characters
- CHARACTER: Contains animated characters

Classification strategy:
1. Blank detection: very low pixel variance across the frame
2. Text detection: high edge density in narrow horizontal bands +
   low color saturation (typical of title/credit cards)
3. Character detection via edge complexity, color diversity in
   foreground regions, and contour shape analysis. Characters tend
   to have complex, rounded contours with varied colors, while
   pure backgrounds have smoother, less contour-dense regions.
"""

import logging
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ShotType(Enum):
    CHARACTER = "character"
    TEXT = "text"
    BLANK = "blank"
    BACKGROUND = "background"


def _is_blank(frame: np.ndarray, variance_threshold: float = 100.0) -> bool:
    """Check if a frame is essentially blank (uniform color).

    Computes the variance of grayscale pixel intensities. A nearly
    uniform frame (solid color, black, white) has very low variance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray)) < variance_threshold


def _is_text_dominant(frame: np.ndarray) -> bool:
    """Detect if a frame is primarily a text card (titles, credits, etc.).

    Text frames tend to have:
    - Low color saturation (often white/black text on solid bg)
    - Edge patterns that form horizontal line clusters
    - High contrast between text and background
    - Low overall color diversity
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Check saturation — text cards are typically low-saturation
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    if mean_saturation > 60:
        # Colorful frame — unlikely to be a text card
        return False

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)

    # Text cards typically have moderate edge density (0.02-0.15)
    # Very low = blank, very high = complex scene
    if edge_density < 0.01 or edge_density > 0.20:
        return False

    # Analyze horizontal distribution of edges — text creates
    # clusters of edges at specific vertical positions (lines of text)
    row_edge_counts = np.count_nonzero(edges, axis=1)
    nonzero_rows = row_edge_counts[row_edge_counts > 0]
    if len(nonzero_rows) == 0:
        return False

    # Text is concentrated in bands — high std relative to mean
    # compared to scattered edges from a scene
    row_std = float(np.std(row_edge_counts))
    row_mean = float(np.mean(row_edge_counts))
    if row_mean > 0:
        coefficient_of_variation = row_std / row_mean
        # Text tends to have CV > 1.5 (clustered in lines)
        if coefficient_of_variation > 1.5:
            return True

    # Check unique color count — text frames have very few unique colors
    small = cv2.resize(frame, (64, 48))
    quantized = (small // 32) * 32  # Quantize to reduce noise
    pixels = quantized.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    if unique_colors < 20:
        return True

    return False


def _compute_character_score(frame: np.ndarray) -> float:
    """Compute a score (0-1) indicating likelihood of character presence.

    Characters in animation tend to have:
    - Complex contours with varied shapes (not just straight lines)
    - Higher color diversity in localized regions
    - More contour density in mid-frame regions (characters are usually centered)
    - Enclosed/rounded contour shapes (heads, bodies) vs open edges (landscapes)

    Returns a score from 0.0 (background) to 1.0 (strong character presence).
    """
    h, w = frame.shape[:2]
    scores = []

    # --- 1. Contour complexity analysis ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Count contours of meaningful size (not tiny noise)
    min_area = (h * w) * 0.0005  # 0.05% of frame area
    max_area = (h * w) * 0.5  # 50% of frame area
    meaningful_contours = [
        c for c in contours
        if min_area < cv2.contourArea(c) < max_area
    ]

    # Characters produce many meaningful contours
    contour_count_score = min(1.0, len(meaningful_contours) / 30.0)
    scores.append(contour_count_score)

    # --- 2. Circularity — characters have rounded shapes ---
    circularity_scores = []
    for cnt in meaningful_contours[:50]:  # Limit for performance
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity_scores.append(circularity)

    if circularity_scores:
        # Higher mean circularity = more rounded shapes = likely characters
        mean_circularity = float(np.mean(circularity_scores))
        circ_score = min(1.0, mean_circularity / 0.4)
        scores.append(circ_score)

    # --- 3. Edge density in center vs edges of frame ---
    # Characters are typically in the center/foreground
    center_y1, center_y2 = h // 4, 3 * h // 4
    center_x1, center_x2 = w // 4, 3 * w // 4

    center_edges = edges[center_y1:center_y2, center_x1:center_x2]
    center_density = float(np.count_nonzero(center_edges)) / center_edges.size

    border_mask = np.zeros_like(edges)
    border_mask[:h // 4, :] = edges[:h // 4, :]
    border_mask[3 * h // 4:, :] = edges[3 * h // 4:, :]
    border_mask[:, :w // 4] = edges[:, :w // 4]
    border_mask[:, 3 * w // 4:] = edges[:, 3 * w // 4:]
    border_density = float(np.count_nonzero(border_mask)) / max(1, np.count_nonzero(border_mask > -1))

    if border_density > 0:
        center_ratio = center_density / max(border_density, 0.001)
        center_score = min(1.0, center_ratio / 3.0)
    else:
        center_score = 0.5 if center_density > 0.01 else 0.0
    scores.append(center_score)

    # --- 4. Color diversity in foreground ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Check saturation — characters usually have more saturated colors
    # than plain backgrounds
    center_region = hsv[center_y1:center_y2, center_x1:center_x2]
    mean_sat = float(np.mean(center_region[:, :, 1]))
    sat_score = min(1.0, mean_sat / 80.0)
    scores.append(sat_score)

    # Hue diversity in center region
    center_hues = center_region[:, :, 0].flatten()
    # Only consider pixels with meaningful saturation
    center_sats = center_region[:, :, 1].flatten()
    saturated_hues = center_hues[center_sats > 30]

    if len(saturated_hues) > 100:
        hue_std = float(np.std(saturated_hues))
        hue_score = min(1.0, hue_std / 40.0)
    else:
        hue_score = 0.2
    scores.append(hue_score)

    # --- 5. Enclosed contour areas (filled shapes = character bodies) ---
    filled_area = sum(cv2.contourArea(c) for c in meaningful_contours)
    filled_ratio = filled_area / (h * w)
    fill_score = min(1.0, filled_ratio / 0.15)
    scores.append(fill_score)

    final_score = float(np.mean(scores))

    logger.debug(
        "Character score: %.3f (contours=%.2f, circ=%.2f, center=%.2f, "
        "sat=%.2f, hue=%.2f, fill=%.2f)",
        final_score,
        contour_count_score,
        circ_score if circularity_scores else 0.0,
        center_score,
        sat_score,
        hue_score,
        fill_score,
    )

    return final_score


def classify_shot(
    frames: List[Tuple[int, np.ndarray]],
    character_threshold: float = 0.35,
    sample_count: int = 5,
) -> Tuple[ShotType, float]:
    """Classify a shot by sampling frames and analyzing their content.

    Samples multiple frames across the shot (to account for variation)
    and returns the dominant classification.

    Args:
        frames: List of (frame_index, frame_array) for the shot.
        character_threshold: Minimum character score to classify as
            CHARACTER. Range 0.0-1.0. Lower = more permissive.
        sample_count: Number of frames to sample across the shot.

    Returns:
        Tuple of (ShotType, confidence_score).
    """
    if not frames:
        return ShotType.BLANK, 1.0

    # Sample frames evenly across the shot
    n = len(frames)
    if n <= sample_count:
        sampled = frames
    else:
        indices = np.linspace(0, n - 1, sample_count, dtype=int)
        sampled = [frames[i] for i in indices]

    blank_count = 0
    text_count = 0
    char_scores = []

    for _, frame in sampled:
        # Check blank first (fastest)
        if _is_blank(frame):
            blank_count += 1
            continue

        # Check text
        if _is_text_dominant(frame):
            text_count += 1
            continue

        # Compute character score
        score = _compute_character_score(frame)
        char_scores.append(score)

    total = len(sampled)

    # Majority rules for blank/text
    if blank_count > total / 2:
        return ShotType.BLANK, blank_count / total

    if text_count > total / 2:
        return ShotType.TEXT, text_count / total

    # For remaining frames, use character score
    if char_scores:
        mean_char_score = float(np.mean(char_scores))
        if mean_char_score >= character_threshold:
            return ShotType.CHARACTER, mean_char_score
        else:
            return ShotType.BACKGROUND, 1.0 - mean_char_score

    # Fallback: if all frames were blank/text but not majority,
    # treat as background
    return ShotType.BACKGROUND, 0.5
