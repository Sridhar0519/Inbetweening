"""PyTorch Dataset for loading keyframe-inbetween training pairs.

Reads from the output directory structure produced by the dataset creator:
    output/video_name/shot_NNN/segment_NNN/
        key_first.png
        key_last.png
        inbetweens/frame_NNNN.png

Each sample is: (key_first, key_last, t, target_frame)
where t is the temporal position of the inbetween frame.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class InbetweenDataset(Dataset):
    """Dataset of (keyframe_pair, t, inbetween_frame) samples.

    Each segment contributes N samples (one per inbetween frame),
    each with t = frame_position / (num_inbetweens + 1).
    """

    def __init__(self, root_dir: str, image_size: int = 256):
        """Initialize dataset.

        Args:
            root_dir: Path to dataset root (output directory).
            image_size: Target size for square resize. Default 256.
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples: List[Tuple[Path, Path, Path, float]] = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._scan_dataset()

    def _scan_dataset(self):
        """Walk the dataset directory and collect all training samples."""
        for video_dir in sorted(self.root_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for shot_dir in sorted(video_dir.iterdir()):
                if not shot_dir.is_dir() or not shot_dir.name.startswith("shot_"):
                    continue
                for seg_dir in sorted(shot_dir.iterdir()):
                    if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
                        continue
                    self._add_segment(seg_dir)

    def _find_keyframe(self, seg_dir: Path, name: str) -> Path:
        """Find a keyframe file (supports .png and .jpg)."""
        for ext in (".png", ".jpg"):
            path = seg_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _add_segment(self, seg_dir: Path):
        """Add all inbetween frames from a segment as training samples."""
        key_first_path = self._find_keyframe(seg_dir, "key_first")
        key_last_path = self._find_keyframe(seg_dir, "key_last")
        inbetweens_dir = seg_dir / "inbetweens"

        if not (key_first_path and key_last_path and inbetweens_dir.exists()):
            return

        inbetween_files = sorted(
            f for f in inbetweens_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        n = len(inbetween_files)
        if n == 0:
            return

        for i, inbetween_path in enumerate(inbetween_files):
            # t ranges from 1/(n+1) to n/(n+1), exclusive of 0 and 1
            t = (i + 1) / (n + 1)
            self.samples.append((key_first_path, key_last_path, inbetween_path, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key_first_path, key_last_path, inbetween_path, t = self.samples[idx]

        key_first = self.transform(Image.open(key_first_path).convert('RGB'))
        key_last = self.transform(Image.open(key_last_path).convert('RGB'))
        target = self.transform(Image.open(inbetween_path).convert('RGB'))
        t_tensor = torch.tensor([t], dtype=torch.float32)

        return {
            'key_first': key_first,
            'key_last': key_last,
            'target': target,
            't': t_tensor,
        }
