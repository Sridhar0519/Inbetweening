"""Shot boundary detection using PySceneDetect's ContentDetector.

Detects scene/shot transitions in animated videos by measuring
frame-to-frame content changes. Returns shot boundaries as
(start_frame, end_frame) pairs.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


def detect_shots(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 6,
) -> Tuple[List[Tuple[int, int]], float, int]:
    """Detect shot boundaries in a video file.

    Args:
        video_path: Path to the input video file.
        threshold: Sensitivity for scene change detection. Lower values
            detect more subtle changes (good for animation). Range ~15-40.
        min_scene_len: Minimum number of frames for a valid scene/shot.

    Returns:
        Tuple of:
            - List of (start_frame, end_frame) tuples for each shot.
            - FPS of the video.
            - Total frame count.
    """
    video_path = str(Path(video_path).resolve())
    logger.info("Opening video: %s", video_path)

    video = open_video(video_path)
    fps = video.frame_rate
    total_frames = video.duration.get_frames()

    logger.info("Video info: %.2f fps, %d total frames", fps, total_frames)

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )

    logger.info(
        "Detecting shots (threshold=%.1f, min_scene_len=%d)...",
        threshold,
        min_scene_len,
    )
    scene_manager.detect_scenes(video, show_progress=True)

    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # Entire video is one shot
        logger.info("No scene changes detected — treating entire video as one shot.")
        shots = [(0, total_frames - 1)]
    else:
        shots = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames() - 1  # inclusive end
            if end_frame > start_frame:
                shots.append((start_frame, end_frame))

        logger.info("Detected %d shot(s).", len(shots))

    # Log shot details
    for i, (start, end) in enumerate(shots):
        frame_count = end - start + 1
        logger.debug(
            "  Shot %03d: frames %d-%d (%d frames, %.2fs)",
            i + 1,
            start,
            end,
            frame_count,
            frame_count / fps,
        )

    return shots, fps, total_frames
