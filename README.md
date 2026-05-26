# image_eq_extension

INCAR preprocessing extension that normalizes image contrast via **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) applied to the luminance channel of camera observations.

---

## Motivation

Visuomotor diffusion policies are sensitive to distribution shift in visual observations. Two camera phenomena are the primary culprits in real deployments:

- **Autofocus**: when the camera hunts for focus between demonstrations and during inference, the global contrast of the frame changes even though the scene content is identical.
- **Lighting variation**: ambient light changes (time of day, reflections, operator shadows) shift pixel intensities across the entire image.

Both effects push incoming frames outside the training distribution without changing anything about the task. This extension applies a fast, deterministic contrast normalization step that absorbs these variations before the policy sees the image.

**Why CLAHE over global histogram equalization:**
Global HE stretches the entire histogram to fill [0, 255], which over-amplifies flat regions and introduces visible halo artefacts near edges. CLAHE limits contrast enhancement per local tile (`clip_limit`) and blends tile boundaries, producing more uniform and natural-looking results on robot scene imagery.

**Why the L channel in LAB:**
Operating on luminance only leaves hue and saturation untouched. Colour-dependent task cues — "pick the red clutch disk", "place on the blue holder" — are fully preserved.

---

## How It Works

```
RGB  →  LAB  →  CLAHE on L channel  →  LAB  →  RGB
```

CLAHE divides the image into a grid of `tile_grid_size` tiles, computes a local histogram for each tile, clips histogram bins above `clip_limit × average_bin_count` (redistributing the excess uniformly), and applies the resulting mapping. Tile boundaries are blended via bilinear interpolation.

The transform is applied identically at two points in the INCAR pipeline:

```
dataset preprocessing (DATASET hook)
  └─ rewrites every demo's mp4 in-place with CLAHE applied
     → zero per-sample overhead during the training dataloader

live inference (OBSERVATION hook)
  └─ applies CLAHE to each incoming camera frame before the policy sees it
     → images land in the same contrast distribution as the training data
```

Both paths use the same `clip_limit` and `tile_grid_size`, guaranteeing that inference images fall inside the distribution the policy was trained on.

---

## Quick start

### 1. Install

```bash
source ~/incar_env/bin/activate
pip install -e /path/to/image_eq_extension
```

### 2. Add to the `steps` list in your training config

```json
{
    "type": "histogram_equalize",
    "features": ["wrist_cam"],
    "clip_limit": 2.0,
    "tile_grid_size": [8, 8]
}
```

A single entry in the policy `steps` list handles **both** training and inference. The `DATASET` hook fires during dataset preprocessing; the `OBSERVATION` hook fires automatically during inference when the policy is loaded — no `workspace_config.json` change required.

> `OBSERVATION` is a policy preprocessing hook, not a workspace command hook. Only `TELEOP_COMMAND` / `INFERENCE_COMMAND` steps belong in `workspace_config.json`.

---

## Placement in the preprocessing pipeline

Place **after** `downsample_video` so the CLAHE tile grid aligns with the final image resolution, and **before** `image_transform` so stochastic colour jitter is applied on top of the already-equalized frames:

```json
"steps": [
    { "type": "sample_dt", "dt": 0.1 },
    { "type": "downsample_video", "features": ["wrist_cam"], "new_size": [240, 320] },
    { "type": "filter_takeover", "..." },
    { "type": "filter_by_buttons", "..." },
    { "type": "histogram_equalize", "features": ["wrist_cam"], "clip_limit": 2.0, "tile_grid_size": [8, 8] },
    { "type": "image_transform", "features": ["wrist_cam"], "transforms": { "brightness": [0.6, 1.4], "contrast": [0.6, 1.4] } }
]
```

The `DATASET` hook rewrites the mp4 files once and shows a progress bar per feature:

```
histogram_equalize [wrist_cam]: 100%|████████████████| 31/31 [00:08<00:00]
```

---

## Configuration reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `list[str]` | `[]` | Camera feature names to equalize. Must match the feature keys in the dataset and workspace configs (e.g. `"wrist_cam"`, `"oak_cam"`). |
| `clip_limit` | `float` | `2.0` | CLAHE contrast-limiting threshold. Higher values allow stronger local enhancement but amplify sensor noise. Typical range: `[1.0, 4.0]`. |
| `tile_grid_size` | `list[int]` | `[8, 8]` | `[rows, cols]` grid for adaptive histogram computation. Larger grids give finer local adaptation at slightly higher compute cost. |
| `hooks` | `list[ProcessHook]` | `DATASET, OBSERVATION` | INCAR hooks. Override only if you need to skip dataset baking or inference application. |

### Typical configurations

**Conservative — wrist camera, stable lab lighting:**
```json
{ "type": "histogram_equalize", "features": ["wrist_cam"], "clip_limit": 1.5, "tile_grid_size": [4, 4] }
```

**Default — general use:**
```json
{ "type": "histogram_equalize", "features": ["wrist_cam"], "clip_limit": 2.0, "tile_grid_size": [8, 8] }
```

**Aggressive — overhead camera with harsh or variable lighting:**
```json
{ "type": "histogram_equalize", "features": ["overhead_cam"], "clip_limit": 3.5, "tile_grid_size": [8, 8] }
```

---

## Performance

CLAHE on a single 240×320 RGB frame (8×8 grid) takes **~0.3 ms** on CPU, using OpenCV's optimised C++ backend (`cv2.createCLAHE`). At 10 Hz inference this adds < 0.3 % latency overhead. The dataset baking step processes one demo in roughly 250 ms; a 31-demo dataset completes in under 10 seconds.

---

## Implementation notes

### Video rewrite

The `DATASET` hook rewrites each mp4 in-place using a custom encoding loop rather than the INCAR utility `apply_transformation_to_all_mp4_frames`. The native utility sets `video_frame.time_base = Fraction(fps)` (10 seconds per PTS unit at 10 fps) instead of `Fraction(1, fps)` (0.1 seconds per PTS unit), which inflates the apparent video duration by a factor of `fps²`. The custom loop sets the correct time_base and reads the codec name from the source video so the output container format is always consistent with the input.

### Colour-space round-trip

The transform converts `RGB → LAB → RGB` using OpenCV's `COLOR_RGB2LAB` and `COLOR_LAB2RGB` constants (not `BGR`). INCAR stores video frames as RGB throughout the pipeline, so no channel swap is needed.

---

## Repository structure

```
image_eq_extension/
├── pyproject.toml
├── README.md
└── image_eq_extension/
    ├── __init__.py
    └── process_step.py    # HistogramEqualize ProcessStep, _rewrite_mp4_with_clahe, CLAHE helpers
```

---

## Background and related work

This extension targets the specific failure mode identified in visuomotor diffusion policy deployments where autofocus and lighting variation shift the camera observation out of the training distribution. The approach is consistent with the broader literature on training-time augmentation (RAD, DrQ-v2) but adds a deterministic inference-time counterpart — a path that is largely unexplored in the robot learning literature, where most prior work relies on wide domain randomisation or robust visual pre-training (DINOv2) rather than explicit test-time normalisation.

Key design decisions:

- **CLAHE over global HE** — avoids over-amplification and halo artefacts; performs better on scenes with mixed dark and bright regions (e.g. a robot arm against a bright tabletop).
- **L channel in LAB** — decouples contrast from chrominance; colour-based task cues are fully preserved.
- **DATASET hook, not GET_ITEM** — bakes the transform into the video files once so the training dataloader has zero per-sample overhead. `clip_limit` changes require a dataset re-run, but this is the correct trade-off for a deterministic, non-stochastic transform.
- **Identical parameters at training and inference** — ensures the policy's learned prior over image contrast matches what it sees at deployment time.
