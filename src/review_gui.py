#!/usr/bin/env python3
#python -m src.review_gui ./output
"""GUI-based dataset review tool using OpenCV.

Displays each segment's keyframe pair (key_first + key_last) side by side
and lets the user keep or delete the segment with keyboard controls.

Usage:
    python -m src.review_gui ./output/video_name
    python -m src.review_gui ./output

Controls:
    K / Enter   — Keep the current segment
    D / Delete  — Delete the current segment
    ← (Left)    — Go back to the previous segment
    Q / Esc     — Quit review (unapproved segments are kept by default)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _load_segment_list(root_dir: Path) -> List[Tuple[Path, str]]:
    """Discover all segment directories under the root.

    Returns list of (segment_path, display_label) sorted by path.
    """
    segments = []

    # Auto-detect: is this a video dir (has shot_*) or multi-video root?
    shot_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("shot_"))

    if shot_dirs:
        # Single video directory
        video_dirs = [root_dir]
    else:
        # Multi-video root
        video_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())

    for video_dir in video_dirs:
        video_name = video_dir.name
        for shot_dir in sorted(video_dir.iterdir()):
            if not shot_dir.is_dir() or not shot_dir.name.startswith("shot_"):
                continue
            for seg_dir in sorted(shot_dir.iterdir()):
                if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
                    continue
                label = f"{video_name} / {shot_dir.name} / {seg_dir.name}"
                segments.append((seg_dir, label))

    return segments


def _render_review_frame(
    key_first: np.ndarray,
    key_last: np.ndarray,
    label: str,
    index: int,
    total: int,
    num_inbetweens: int,
    decision: str,
) -> np.ndarray:
    """Compose a display frame showing both keyframes + info overlay."""
    h1, w1 = key_first.shape[:2]
    h2, w2 = key_last.shape[:2]

    # Scale both to same height, max 400px
    display_h = min(400, h1, h2)
    scale1 = display_h / h1
    scale2 = display_h / h2
    img1 = cv2.resize(key_first, (int(w1 * scale1), display_h))
    img2 = cv2.resize(key_last, (int(w2 * scale2), display_h))

    # Gap between images
    gap = 20
    gap_col = np.ones((display_h, gap, 3), dtype=np.uint8) * 40

    # Side by side
    combined = np.hstack([img1, gap_col, img2])
    ch, cw = combined.shape[:2]

    # Add header and footer bars
    header_h = 70
    footer_h = 80
    header = np.ones((header_h, cw, 3), dtype=np.uint8) * 30
    footer = np.ones((footer_h, cw, 3), dtype=np.uint8) * 30

    # Header text
    cv2.putText(header, f"Segment {index + 1} / {total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(header, label, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Labels on images
    label_bar_h = 30
    label_bar1 = np.ones((label_bar_h, img1.shape[1], 3), dtype=np.uint8) * 50
    label_bar2 = np.ones((label_bar_h, img2.shape[1], 3), dtype=np.uint8) * 50
    label_gap = np.ones((label_bar_h, gap, 3), dtype=np.uint8) * 40
    cv2.putText(label_bar1, "KEY FIRST", (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    cv2.putText(label_bar2, "KEY LAST", (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    label_row = np.hstack([label_bar1, label_gap, label_bar2])

    # Footer: controls + decision
    cv2.putText(footer, "[K/Enter] Keep    [D/Del] Delete    [<-] Back    [Q/Esc] Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(footer, f"Inbetweens: {num_inbetweens}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Decision badge
    if decision == "keep":
        color = (0, 200, 0)
        text = "KEEPING"
    elif decision == "delete":
        color = (0, 0, 220)
        text = "DELETING"
    else:
        color = (100, 100, 100)
        text = "PENDING"

    badge_x = cw - 180
    cv2.rectangle(footer, (badge_x, 35), (badge_x + 160, 65), color, -1)
    cv2.putText(footer, text, (badge_x + 15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    canvas = np.vstack([header, label_row, combined, footer])
    return canvas


def review_dataset(root_dir: Path) -> dict:
    """Launch interactive review GUI. Returns summary of decisions."""
    segments = _load_segment_list(root_dir)

    if not segments:
        print(f"No segments found under {root_dir}")
        return {"kept": 0, "deleted": 0}

    print(f"\nFound {len(segments)} segment(s) to review.")
    print("Opening review window...\n")

    window_name = "Shot Segregation - Segment Review"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    decisions = {}  # index -> "keep" | "delete"
    current = 0
    quit_review = False

    while not quit_review:
        seg_dir, label = segments[current]

        key_first_path = seg_dir / "key_first.png"
        key_last_path = seg_dir / "key_last.png"
        inbetweens_dir = seg_dir / "inbetweens"

        # Load keyframe images
        key_first = cv2.imread(str(key_first_path))
        key_last = cv2.imread(str(key_last_path))

        if key_first is None or key_last is None:
            print(f"  WARNING: Cannot load keyframes for {label} — skipping.")
            decisions[current] = "keep"
            current = min(current + 1, len(segments) - 1)
            continue

        # Count inbetweens
        num_inbetweens = 0
        if inbetweens_dir.exists():
            num_inbetweens = len(list(inbetweens_dir.glob("*.png")))

        decision = decisions.get(current, "pending")

        # Render display
        canvas = _render_review_frame(
            key_first, key_last, label,
            current, len(segments), num_inbetweens, decision,
        )
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('k'), ord('K'), 13):  # K or Enter = keep
            decisions[current] = "keep"
            print(f"  [{current + 1}/{len(segments)}] KEEP: {label}")
            if current < len(segments) - 1:
                current += 1
            else:
                # Last segment — show updated status then prompt to finish
                canvas = _render_review_frame(
                    key_first, key_last, label,
                    current, len(segments), num_inbetweens, "keep",
                )
                cv2.imshow(window_name, canvas)
                cv2.waitKey(300)
                quit_review = True

        elif key in (ord('d'), ord('D'), 255, 0):  # D or Delete = delete
            decisions[current] = "delete"
            print(f"  [{current + 1}/{len(segments)}] DELETE: {label}")
            if current < len(segments) - 1:
                current += 1
            else:
                canvas = _render_review_frame(
                    key_first, key_last, label,
                    current, len(segments), num_inbetweens, "delete",
                )
                cv2.imshow(window_name, canvas)
                cv2.waitKey(300)
                quit_review = True

        elif key == 81 or key == 2:  # Left arrow
            current = max(0, current - 1)

        elif key in (ord('q'), ord('Q'), 27):  # Q or Esc
            quit_review = True

    cv2.destroyAllWindows()

    # Apply deletions
    kept = 0
    deleted = 0
    for idx, (seg_dir, label) in enumerate(segments):
        if decisions.get(idx) == "delete":
            shutil.rmtree(seg_dir)
            deleted += 1
            print(f"  Deleted: {label}")
        else:
            kept += 1

    # Clean up empty shot/video dirs and update metadata
    _cleanup_empty_dirs(root_dir)
    _update_metadata(root_dir)

    print(f"\nReview complete: {kept} kept, {deleted} deleted.")
    return {"kept": kept, "deleted": deleted}


def _cleanup_empty_dirs(root_dir: Path) -> None:
    """Remove shot directories that have no remaining segments."""
    for video_dir in _iter_video_dirs(root_dir):
        for shot_dir in sorted(video_dir.iterdir()):
            if not shot_dir.is_dir() or not shot_dir.name.startswith("shot_"):
                continue
            remaining = [
                d for d in shot_dir.iterdir()
                if d.is_dir() and d.name.startswith("segment_")
            ]
            if not remaining:
                shutil.rmtree(shot_dir)
                print(f"  Removed empty shot dir: {shot_dir.relative_to(root_dir)}")


def _iter_video_dirs(root_dir: Path):
    """Yield video-level directories from root."""
    has_shots = any(
        d.name.startswith("shot_") for d in root_dir.iterdir() if d.is_dir()
    )
    if has_shots:
        yield root_dir
    else:
        for d in sorted(root_dir.iterdir()):
            if d.is_dir():
                yield d


def _update_metadata(root_dir: Path) -> None:
    """Regenerate metadata.json and dataset_summary.json after deletions."""
    for video_dir in _iter_video_dirs(root_dir):
        shot_metadata_list = []

        for shot_dir in sorted(video_dir.iterdir()):
            if not shot_dir.is_dir() or not shot_dir.name.startswith("shot_"):
                continue

            # Read original metadata if it exists
            meta_path = shot_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    shot_meta = json.load(f)

                # Update segment list to only include remaining segments
                remaining_segs = sorted(
                    d.name for d in shot_dir.iterdir()
                    if d.is_dir() and d.name.startswith("segment_")
                )
                shot_meta["segments"] = [
                    s for s in shot_meta.get("segments", [])
                    if s.get("segment_name") in remaining_segs
                ]
                shot_meta["num_segments"] = len(shot_meta["segments"])

                with open(meta_path, "w") as f:
                    json.dump(shot_meta, f, indent=2)

                shot_metadata_list.append(shot_meta)

        # Update dataset summary
        summary_path = video_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            total_segments = sum(s["num_segments"] for s in shot_metadata_list)
            total_inbetweens = sum(
                seg["num_inbetweens"]
                for shot in shot_metadata_list
                for seg in shot["segments"]
            )
            summary["total_shots"] = len(shot_metadata_list)
            summary["total_segments"] = total_segments
            summary["total_inbetween_frames"] = total_inbetweens
            summary["shots"] = shot_metadata_list

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review dataset segments — keep or delete each keyframe pair.",
        epilog="""
Controls:
  K / Enter   Keep the current segment
  D / Delete  Delete the current segment
  Left arrow  Go back to the previous segment
  Q / Esc     Quit (unreviewed segments are kept)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to dataset root or video directory to review.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: path does not exist: {args.path}")
        sys.exit(1)

    result = review_dataset(args.path)
    sys.exit(0 if result["deleted"] >= 0 else 1)


if __name__ == "__main__":
    main()
