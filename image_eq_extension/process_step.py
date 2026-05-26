"""
Image histogram equalization extension for INCAR imitation learning.

Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) to the
luminance channel (L in LAB colorspace) of camera images. This normalizes
contrast variation caused by autofocus and lighting changes without altering
hue or saturation.

Dual-use design:
  - Training (DATASET): bakes CLAHE into the preprocessed video files once,
    so no per-sample overhead during the training dataloader.
  - Inference (OBSERVATION): applies the same transform to each incoming live
    camera frame before it reaches the policy.

Both paths use identical parameters, guaranteeing that inference images land
in the same distribution the policy was trained on.
"""
import fractions
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, TYPE_CHECKING

import av
import cv2
import numpy as np
import torch
from tqdm import tqdm

from incar.common import FeatureType, ProcessHook
from incar.extensions.processing_step import ProcessStep

if TYPE_CHECKING:
    from incar.config.dataset_config import DatasetConfig


def _to_hwc_uint8(image: np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool, torch.dtype | None, object]:
    """Convert any supported image format to HWC uint8 numpy array.

    Returns (hwc_uint8, was_torch, original_torch_dtype, original_device).
    """
    if isinstance(image, torch.Tensor):
        dtype = image.dtype
        device = image.device
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)   # CHW -> HWC
        return arr.clip(0, 255).astype(np.uint8), True, dtype, device
    else:
        return image.clip(0, 255).astype(np.uint8), False, None, None


def _from_hwc_uint8(result: np.ndarray, was_torch: bool, original_dtype: torch.dtype | None,
                    original_device: object, original: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert HWC uint8 back to the original format, preserving device and dtype."""
    if was_torch:
        chw = result.transpose(2, 0, 1).copy()         # HWC -> CHW
        return torch.from_numpy(chw).to(dtype=original_dtype, device=original_device)
    else:
        return result.astype(original.dtype)


def apply_clahe(image: np.ndarray | torch.Tensor, clahe: cv2.CLAHE) -> np.ndarray | torch.Tensor:
    """Apply CLAHE to the L channel of an RGB image in LAB colorspace.

    Accepts:
      np.ndarray  (H, W, 3) uint8 or float32 [0–255]  → returns same type/shape/device
      torch.Tensor (3, H, W) float32 [0–255]           → returns same type/shape/device
    """
    hwc, was_torch, orig_dtype, orig_device = _to_hwc_uint8(image)
    lab = cv2.cvtColor(hwc, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _from_hwc_uint8(rgb, was_torch, orig_dtype, orig_device, image)


def _rewrite_mp4_with_clahe(mp4_path: Path, clahe: cv2.CLAHE, fps: int) -> None:
    """Read mp4, apply CLAHE to every frame, write back with correct timestamps.

    Uses Fraction(1, fps) as the frame time_base so each PTS unit equals
    one frame duration (1/fps seconds). This avoids the ×fps inflation that
    happens when time_base is mistakenly set to Fraction(fps).
    """
    # Read all frames and the source codec name before closing the container
    in_container = av.open(str(mp4_path))
    codec_name = in_container.streams.video[0].codec_context.name
    raw_frames = [f.to_ndarray(format="rgb24") for f in in_container.decode()]
    in_container.close()

    if not raw_frames:
        return

    tmp_path = str(mp4_path) + ".eq_tmp.mp4"
    rate = fractions.Fraction(fps)
    time_base = fractions.Fraction(1, fps)   # 1/fps seconds per PTS unit — CORRECT

    out_container = av.open(tmp_path, "w")
    stream = out_container.add_stream(codec_name, rate)
    first = apply_clahe(raw_frames[0].copy(), clahe)
    stream.height = first.shape[0]
    stream.width = first.shape[1]

    for i, frame_arr in enumerate(raw_frames):
        transformed = apply_clahe(frame_arr, clahe)
        video_frame = av.VideoFrame.from_ndarray(transformed, format="rgb24")
        video_frame.pts = i
        video_frame.time_base = time_base
        for packet in stream.encode(video_frame):
            out_container.mux(packet)

    for packet in stream.encode():   # flush encoder
        out_container.mux(packet)
    out_container.close()

    os.replace(tmp_path, str(mp4_path))


@ProcessStep.register_subclass("histogram_equalize")
@dataclass
class HistogramEqualize(ProcessStep):
    """Apply CLAHE on the luminance channel to normalize contrast across observations.

    Hooks:
      DATASET     — bakes CLAHE into the preprocessed video files once so the
                    training dataloader has zero per-sample overhead.
      OBSERVATION — applies CLAHE to each live camera frame at inference so
                    images match the trained distribution.

    Parameters
    ----------
    features : list[str]
        Camera feature names to equalize (e.g. ["oak_cam", "wrist_cam"]).
    clip_limit : float
        CLAHE contrast-limiting threshold.  Higher values allow more contrast
        enhancement at the cost of amplifying noise.  2.0 is a safe default.
    tile_grid_size : list[int]
        [rows, cols] grid for adaptive histogram computation.  Larger grids
        produce finer local adaptation at slightly higher compute cost.
    """

    hooks: List[ProcessHook] = field(
        default_factory=lambda: [
            ProcessHook.DATASET,
            ProcessHook.OBSERVATION,
        ]
    )
    features: List[str] = field(default_factory=list)
    clip_limit: float = 2.0
    tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])

    def __post_init__(self):
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=tuple(self.tile_grid_size),
        )

    # ------------------------------------------------------------------
    # DATASET hook — bake CLAHE into the preprocessed video files once
    # ------------------------------------------------------------------
    def process_dataset(self, root_path: str, config: "DatasetConfig") -> None:
        for f in self.features:
            if f not in config.features:
                raise ValueError(
                    f"[histogram_equalize] feature '{f}' not found in dataset. "
                    f"Available features: {list(config.features.keys())}"
                )
            if config.features[f].type != FeatureType.VISUAL:
                raise ValueError(
                    f"[histogram_equalize] feature '{f}' has type "
                    f"{config.features[f].type}, expected VISUAL."
                )
        visual_features = self.features

        demos = sorted(
            [f.name for f in Path(root_path).iterdir() if f.is_dir()],
            key=lambda x: int(x.split("_")[-1]),
        )

        for feature in visual_features:
            for demo in tqdm(demos, desc=f"histogram_equalize [{feature}]"):
                mp4_path = Path(root_path) / demo / feature / "data.mp4"
                if not mp4_path.exists():
                    continue
                _rewrite_mp4_with_clahe(mp4_path, self._clahe, config.video_fps)

    # ------------------------------------------------------------------
    # OBSERVATION hook — apply to live camera frames at inference
    # frame[key] is np.ndarray (H, W, C) uint8  OR  torch.Tensor (C, H, W)
    # ------------------------------------------------------------------
    def process_single_frame(self, frame: dict) -> None:
        for key in [k for k in frame if k in self.features]:
            frame[key] = apply_clahe(frame[key], self._clahe)

    # Not used with DATASET / OBSERVATION hooks, but required by the ABC
    def process_sequence(self, frames: dict) -> None:
        pass
