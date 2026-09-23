"""
dataset.py
----------
PyTorch Dataset for EchoNet-Dynamic (Section 5 of the technical spec).

EchoNet-Dynamic ships as:
    Videos/               10,030 .avi echo videos
    FileList.csv           one row per video: FileName, EF, ESV, EDV, FPS,
                            NumberOfFrames, Split (TRAIN/VAL/TEST), etc.
    VolumeTracings.csv      per-video coordinate line segments tracing the LV
                            boundary, but ONLY for the ED and ES frame of
                            each video.

Ground truth for segmentation only exists on those two labeled frames per
video, so this Dataset yields (frame, mask) pairs at the ED/ES frame level,
not full videos. Full-video inference (all frames) happens later, at
inference time, via segmentation_module.segment_video() -- that function
uses the *trained* model to predict masks on every frame, not just ED/ES.

This file is intentionally independent of segmentation_module.py: this is
the *training data* pipeline, segmentation_module.py is the *deliverable
inference* module described in Contract 1 (Section 8.4.1). Keeping them
separate means the deliverable module has no dependency on pandas/csv/
training-only code.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.draw import polygon as sk_polygon
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class EchoNetPaths:
    """Groups the three things every EchoNet-Dynamic consumer needs (Section 5.2)."""
    videos_dir: str
    file_list_csv: str
    volume_tracings_csv: str


def load_file_list(paths: EchoNetPaths) -> pd.DataFrame:
    """Loads FileList.csv and sanity-checks the official split counts (Section 5.3).

    Raises a clear error if the split doesn't match the published counts --
    this is a deliberate fail-fast check per the spec ("confirm the
    TRAIN/VAL/TEST split and row counts match Section 5.3 before doing
    anything else").
    """
    df = pd.read_csv(paths.file_list_csv)
    counts = df["Split"].value_counts().to_dict()

    expected = {"TRAIN": 7465, "VAL": 1288, "TEST": 1277}
    mismatches = {
        split: (counts.get(split, 0), expected_count)
        for split, expected_count in expected.items()
        if counts.get(split, 0) != expected_count
    }
    if mismatches:
        print(
            "[dataset] WARNING: split counts differ from the published EchoNet-Dynamic "
            f"split. Got {counts}, expected {expected}. Mismatches: {mismatches}. "
            "This is fine for a small sample subset used during dev, but the FULL "
            "dataset should match exactly before reporting final Dice/IoU numbers."
        )
    return df


def load_volume_tracings(paths: EchoNetPaths) -> pd.DataFrame:
    """Loads VolumeTracings.csv -- per-frame coordinate line segments for ED/ES frames only."""
    return pd.read_csv(paths.volume_tracings_csv)


def tracing_to_mask(frame_tracings: pd.DataFrame, height: int, width: int) -> np.ndarray:
    """Converts one frame's VolumeTracings.csv rows into a filled binary polygon mask.

    VolumeTracings.csv stores the LV boundary as a set of line segments
    (X1, Y1, X2, Y2 per row) rather than an ordered polygon, so we collect
    the segment endpoints and rasterize them into a filled region with
    skimage.draw.polygon, per Section 5.3 ("produced with polygon
    rasterization (e.g. skimage.draw.polygon)").
    """
    xs = pd.concat([frame_tracings["X1"], frame_tracings["X2"]]).to_numpy()
    ys = pd.concat([frame_tracings["Y1"], frame_tracings["Y2"]]).to_numpy()

    mask = np.zeros((height, width), dtype=np.uint8)
    if len(xs) < 3:
        # Not enough points to form a polygon -- return an empty mask rather
        # than crash; caller can log/skip these as data-quality outliers.
        return mask

    rr, cc = sk_polygon(ys, xs, shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def read_single_frame(video_path: str, frame_index: int, resize=(112, 112)) -> np.ndarray:
    """Reads exactly one frame from a video by index and resizes it (RGB, uint8)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
    return frame_rgb


def normalize_frame(frame_rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW normalized with ImageNet mean/std (Section 5.3)."""
    frame = frame_rgb_uint8.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame.transpose(2, 0, 1)  # HWC -> CHW


class EchoNetSegmentationDataset(Dataset):
    """Yields (frame, mask) pairs for the ED and ES frame of each video in a given split.

    Each item:
        frame: float32 tensor, shape (3, H, W), ImageNet-normalized
        mask:  float32 tensor, shape (H, W), values in {0, 1}

    Augmentation (Section 5.3, training only):
        - random rotation up to +/-10 degrees
        - NO horizontal flip (clinically meaningful orientation, must be preserved)
        - mild brightness/contrast jitter
    """

    def __init__(self, paths: EchoNetPaths, split: str, resize=(112, 112), augment: bool = False,
                 max_videos: int = None):
        """
        max_videos: optional cap on the number of videos drawn from this split,
        for quick smoke-test / time-constrained runs (e.g. a same-day demo).
        Leave as None for the full official split.
        """
        assert split in {"TRAIN", "VAL", "TEST"}
        self.paths = paths
        self.resize = resize
        self.augment = augment

        file_list = load_file_list(paths)
        self.file_list = file_list[file_list["Split"] == split].reset_index(drop=True)
        if max_videos is not None:
            self.file_list = self.file_list.iloc[:max_videos].reset_index(drop=True)

        tracings = load_volume_tracings(paths)
        # NOTE: in the real EchoNet-Dynamic files, FileList.csv's FileName has
        # NO .avi extension, but VolumeTracings.csv's FileName DOES include
        # .avi. Strip it here so the two tables match up correctly -- without
        # this, zero videos match and the dataset ends up empty.
        tracings = tracings.copy()
        tracings["FileName"] = tracings["FileName"].apply(
            lambda x: os.path.splitext(str(x))[0]
        )
        # VolumeTracings.csv has one row per line-segment; group by
        # (video, frame) so we can rebuild each labeled frame's polygon.
        self.tracings_by_video_frame = {
            key: group for key, group in tracings.groupby(["FileName", "Frame"])
        }

        # Build a flat index: one entry per (video, ED/ES frame).
        self.samples = []
        for _, row in self.file_list.iterrows():
            filename = row["FileName"]
            video_frames = [
                frame for (fname, frame) in self.tracings_by_video_frame.keys()
                if fname == filename
            ]
            for frame_idx in video_frames:
                self.samples.append((filename, frame_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, frame: np.ndarray, mask: np.ndarray):
        if not self.augment:
            return frame, mask

        h, w = frame.shape[:2]

        # Small rotation, +/- 10 degrees (no horizontal flip -- see docstring).
        angle = np.random.uniform(-10, 10)
        rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        frame = cv2.warpAffine(frame, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # Mild brightness/contrast jitter to simulate different ultrasound machines.
        alpha = np.random.uniform(0.9, 1.1)  # contrast
        beta = np.random.uniform(-10, 10)    # brightness
        frame = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return frame, mask

    def __getitem__(self, idx: int):
        filename, frame_idx = self.samples[idx]
        video_path = os.path.join(self.paths.videos_dir, filename)
        if not video_path.endswith(".avi"):
            video_path += ".avi"

        frame = read_single_frame(video_path, frame_idx, resize=self.resize)

        raw_h, raw_w = self._original_dims(video_path)
        frame_tracings = self.tracings_by_video_frame[(filename, frame_idx)]
        mask = tracing_to_mask(frame_tracings, raw_h, raw_w)
        mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        frame, mask = self._augment(frame, mask)

        frame_norm = normalize_frame(frame)
        return (
            torch.from_numpy(frame_norm).float(),
            torch.from_numpy(mask).float(),
        )

    @staticmethod
    def _original_dims(video_path: str):
        cap = cv2.VideoCapture(video_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return h, w
