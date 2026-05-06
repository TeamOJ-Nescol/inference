# Dartz Inference Service

Real-time dartboard scoring service. A FastAPI WebSocket server runs an [RF-DETR](https://github.com/roboflow/rf-detr) detector over incoming camera frames, locates the dartboard via four alignment markers, projects each detected dart tip into board coordinates with a perspective transform, and returns annotated frames + scores to the client.

Built as a college project alongside the [`Auth`](../Auth) backend and the [`website`](../website) frontend.

## What it does

1. Client streams JPEG frames (or a video file) over a WebSocket.
2. RF-DETR detects two classes per frame: `align` (board calibration markers) and `dart_tip`.
3. The four alignment markers are deduplicated, ordered (top / left / right / bottom) and used to compute a perspective homography from image space → canonical board space.
4. Each dart tip is mapped into board space and scored against the dartboard geometry (singles, doubles, trebles, bull, double-bull).
5. A temporal tracker dedupes darts across frames, locking three darts per round.
6. Annotated frame + scores are streamed back to the client.

A full debug post-mortem covering the homography, angle, resolution, and calibration fixes lives in [`CORRECTION.md`](./CORRECTION.md).

## Stack

- **Python:** 3.10–3.13 (3.12+ preferred)
- **Detection:** RF-DETR Small (`rfdetr`), PyTorch
- **Serving:** FastAPI + Uvicorn, `websockets`
- **Vision:** OpenCV, NumPy, Pillow, SciPy
- **Tracking:** `supervision`

## Repository layout

```
src/
├── server.py              # FastAPI app: /ws (live frames), /video/ws (file upload)
├── prediction.py          # Per-frame inference, scoring, overlay drawing
├── detector_detr.py       # RF-DETR wrapper: detect() + calibrate()
├── detector_detectron.py  # Alternative detector (fallback / experimental)
├── darttracker.py         # Temporal dedup of dart tips across frames
├── camera.py              # Camera capture wrapper
├── validation.py          # Score sanity checks
├── dataclass/
│   └── dartboard.py       # DartboardScorer geometry (radii, sector angles)
├── helpers/
│   ├── filter.py          # filter_reasonable_scores, dual-camera reconciliation
│   ├── points_order.py    # Order calibration points by angle
│   ├── vect.py            # Vector utilities
│   ├── logger.py
│   └── counter.py
└── rf-detr/               # git submodule: roboflow/rf-detr

model/
└── checkpoint_best_ema.pth   # 127 MB EMA weights (download separately, see below)

training/
└── get_photos.py          # Dual-camera capture utility for dataset collection
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd inference
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

**macOS / Linux:**
```bash
./install.sh
```

**Windows:**
```cmd
install.cmd
```

The scripts auto-select a compatible Python (3.10–3.14), create `.venv`, and install `requirements.txt`.

Manual alternative:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download model weights

The trained checkpoint is hosted on Hugging Face:

> https://huggingface.co/struan-mclean1/DartBoard/tree/main

Place `checkpoint_best_ema.pth` at `model/checkpoint_best_ema.pth` (or set `MODEL_PATH`).

### 4. Run the server

```bash
./run.sh        # macOS / Linux
run.cmd         # Windows
```

The server listens on `http://0.0.0.0:8000`. CORS is open to `http://localhost:3000` for the frontend.

## WebSocket API

### `/ws` — live camera frames

Client sends:
- A JSON metadata message: `{"cam_id": 0, "frame_skip": 2}` (optional; defaults shown).
- Then binary JPEG frames.

Server replies with alternating:
- Binary JPEG frames with overlays (calibration markers, scoring rings, dart highlights).
- JSON: `{"type": "scores", "cam_id": 0, "scores": [[20, "Double 20"], [51, "T17"]], "frame": 42}`.

### `/video/ws` — uploaded video file

Client sends a single binary blob (MP4, MOV, …); the server processes it frame by frame and streams back the same payload types.

## Environment variables

| Variable        | Default                             | Purpose                                 |
| --------------- | ----------------------------------- | --------------------------------------- |
| `MODEL_PATH`    | `model/checkpoint_best_ema.pth`     | RF-DETR checkpoint location             |
| `CAM_FOV_DEG`   | `0` (off)                           | Lens undistortion field of view         |
| `CAM_K1`, `CAM_K2` | `0`                              | Radial-distortion coefficients          |

Lens-undistortion scaffolding is in place but disabled by default (see `CORRECTION.md`).

## The model

- **Architecture:** RF-DETR Small (transformer-based DETR variant from Roboflow)
- **Classes:** `align`, `dart_tip`, `unknown` (filtered out)
- **Checkpoint:** EMA snapshot at best validation, ~127 MB
- **Inference device:** CPU by default in `detector_detr.py`; switch to `cuda` if available

## Dataset collection

`training/get_photos.py` is a small dual-camera capture utility used to build the training set. It opens cameras `0` and `1`, negotiates the highest supported resolution, and lets the operator save synchronized frame pairs into `images/`. Actual training happens externally; only the final checkpoint is committed (well, downloaded — see above).

## Authors

Struan McLean · Callum Stewart · Cristian · C
