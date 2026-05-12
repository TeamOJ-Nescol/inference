<<<<<<< HEAD
# Inference Server

This service runs dart detection, board calibration, and score inference for the project. It exposes:

- a live frame-by-frame WebSocket API for camera streaming
- a video-file WebSocket API for uploaded recordings
- HTTP endpoints for single-frame inference and calibration

The server is implemented in [src/server.py](/Users/cris/Documents/college/darts/inference/src/server.py) and is started with [run.sh](/Users/cris/Documents/college/darts/inference/run.sh).

## What It Does

For each incoming frame, the server:

1. undistorts the frame using a lightweight lens model
2. detects calibration markers on the dartboard
3. solves a homography from image space to board space
4. detects darts
5. maps dart tip positions into board coordinates
6. scores each dart against the board geometry
7. returns an annotated JPEG plus structured score data

If a calibration has already been cached for a camera, the server reuses that homography instead of recalibrating every frame.

## Repository Layout

- [src/server.py](/Users/cris/Documents/college/darts/inference/src/server.py): FastAPI app, WebSocket handlers, calibration cache
- [src/prediction.py](/Users/cris/Documents/college/darts/inference/src/prediction.py): board geometry, homography mapping, frame annotation
- [src/detector_detr.py](/Users/cris/Documents/college/darts/inference/src/detector_detr.py): RF-DETR dart detector wrapper
- [src/helpers/points_order.py](/Users/cris/Documents/college/darts/inference/src/helpers/points_order.py): calibration-point ordering
- [src/helpers/filter.py](/Users/cris/Documents/college/darts/inference/src/helpers/filter.py): final score filtering
- [install.sh](/Users/cris/Documents/college/darts/inference/install.sh): local environment setup
- [run.sh](/Users/cris/Documents/college/darts/inference/run.sh): local dev server startup

## Requirements

- Python 3.10, 3.11, or 3.12
- a model checkpoint file
- enough CPU/RAM to run the detector locally

The install script creates a local virtual environment and installs dependencies from [requirements.txt](/Users/cris/Documents/college/darts/inference/requirements.txt).

## Setup

From the `inference/` directory:

```bash
./install.sh
```

This script:

- creates `.venv/`
- upgrades `pip`, `setuptools`, and `wheel`
- installs Python dependencies
- creates the `model/` directory if it does not already exist

## Model Checkpoint

The server will not start without a model file.

Default expected path:

```text
inference/model/checkpoint_best_ema.pth
```

You can either:

- place the checkpoint at that path, or
- point `MODEL_PATH` at another checkpoint location

The original README referenced this Hugging Face repository for model download:

- <https://huggingface.co/struan-mclean1/DartBoard/tree/main>

## Running The Server

From the `inference/` directory:

```bash
./run.sh
```

What `run.sh` does:

- activates `.venv`
- sets `PYTHONPATH` for the repo layout
- resolves `MODEL_PATH`
- starts Uvicorn on `0.0.0.0:8000`
- enables `--reload` for development
- raises the WebSocket size limit to handle uploaded video files

Default local URL:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/
```

Expected response:

```json
{"message":"Running"}
```

## Environment Variables

### `MODEL_PATH`

Overrides the checkpoint path used by the detector.

Example:

```bash
MODEL_PATH=/absolute/path/to/checkpoint_best_ema.pth ./run.sh
```

### `CAM_FOV_DEG`

Approximate horizontal camera field of view in degrees. Default: `90`.

This is used to build the synthetic camera matrix for lens undistortion.

### `CAM_K1`

First radial distortion coefficient. Default: `0.0`.

Negative values help correct barrel distortion.

### `CAM_K2`

Second radial distortion coefficient. Default: `0.0`.

Use this to fine-tune distortion after `CAM_K1` is close.

Example:

```bash
CAM_FOV_DEG=90 CAM_K1=-0.10 CAM_K2=0.02 ./run.sh
```

## API Summary

### `GET /`

Simple health endpoint.

Response:

```json
{"message":"Running"}
```

### `POST /frame/{cam_id}`

Runs inference on a single uploaded image.

Request:

- multipart form upload
- field name: `file`

Response:

```json
{
  "scores": [[20, "20"], [60, "T20"]]
}
```

This endpoint returns scores only. It does not return the annotated frame.

### `POST /calibration/frame`

Runs calibration-marker detection on a single uploaded image and returns an annotated JPEG response.

Form fields:

- `file`: image payload
- `cam_id`: optional camera identifier, default `"0"`

Response body:

- JPEG image with detected calibration markers and status overlay

Response headers:

- `X-Calibration-Cam-Id`
- `X-Calibration-Points`
- `X-Calibration-Ready`
- `X-Calibration-Cache-Active`

This endpoint is useful when tuning camera position or checking whether four distinct markers are visible.

### `POST /calibration/cache`

Caches the latest successful homography for a camera.

Form fields:

- `cam_id`: optional camera identifier, default `"0"`

Success response:

```json
{
  "cam_id": "0",
  "cached": true,
  "points_detected": 4
}
```

If no successful calibration exists yet, the endpoint returns HTTP `409`.

### `POST /calibration/cache/clear`

Clears the currently cached homography for a camera.

Form fields:

- `cam_id`: optional camera identifier, default `"0"`

Response:

```json
{
  "cam_id": "0",
  "cleared": true
}
```

### `WS /ws`

Live frame-streaming endpoint for browser or camera clients.

Expected session flow:

1. connect WebSocket
2. send an initial JSON message with `cam_id`
3. repeatedly send JPEG frame bytes
4. receive one annotated JPEG frame
5. receive one JSON score payload
6. optionally send `{"type":"stop"}` to end the session

Initial metadata example:

```json
{"cam_id": 0}
```

Score response example:

```json
{
  "type": "scores",
  "cam_id": 0,
  "scores": [[20, "20"], [50, "Double Bull"]]
}
```

Error response example:

```json
{
  "type": "error",
  "message": "Invalid frame payload"
}
```

The server resets per-camera predictor state when the WebSocket closes.

### `WS /video/ws`

Video-upload endpoint intended for browser playback fallback or offline testing.

Expected session flow:

1. connect WebSocket
2. send JSON metadata
3. send the full video file as bytes
4. receive `ready`
5. receive repeated annotated JPEG frames and score JSON messages
6. receive `done` when processing completes

Initial metadata example:

```json
{
  "cam_id": 0,
  "frame_skip": 2,
  "ext": ".mp4"
}
```

Ready response example:

```json
{
  "type": "ready",
  "total_frames": 742
}
```

Per-frame score response example:

```json
{
  "type": "scores",
  "cam_id": 0,
  "frame": 18,
  "scores": [[19, "19"]]
}
```

Completion response example:

```json
{
  "type": "done",
  "processed_frames": 371
}
```

The server writes the uploaded video to a temporary file, decodes it with OpenCV, and removes the temp file when the session ends.

## Calibration Workflow

The scoring pipeline depends on four visible calibration markers. Without them, the server cannot reliably map detections into board coordinates.

Typical workflow:

1. aim the camera so all four calibration markers are visible
2. call `POST /calibration/frame` until `X-Calibration-Ready: true`
3. call `POST /calibration/cache` for that `cam_id`
4. start streaming frames through `WS /ws`
5. call `POST /calibration/cache/clear` if the camera moves or the board shifts

If a cached homography is active, the live endpoint skips repeated calibration solves and uses the stored mapping for that camera ID.

## Score Filtering

Before score data is returned to clients, the server runs a final sanity filter in [filter.py](/Users/cris/Documents/college/darts/inference/src/helpers/filter.py).

Current rules:

- discard scores below `0`
- discard scores above `60`
- keep at most three darts per response

This is intentionally conservative and only meant to strip obviously impossible outputs.

## Development Notes

- The detector is currently constructed with `device="cpu"` in `FramePredict`, so local performance is CPU-bound unless the code is changed.
- CORS is enabled for local frontend origins on ports `3000` and `3001`.
- `run.sh` uses Uvicorn `--reload`, which is useful for development but not ideal for production.
- JPEG responses are encoded at quality `80`.
- Video WebSocket uploads rely on a large max WebSocket payload size configured in `run.sh`.

## Quick Test Client

A simple example client lives at [src/ws_test_client.py](/Users/cris/Documents/college/darts/inference/src/ws_test_client.py).

It:

- opens `ws://localhost:8000/ws`
- reads frames from a local video file
- JPEG-encodes each frame
- sends metadata plus frame bytes
- displays the annotated frame in an OpenCV window
- prints the returned score payloads

The script is useful for local protocol testing, but note that it sends metadata before each frame, while the server only requires an initial metadata message and then frame/control messages after that.

## Troubleshooting

### `Model checkpoint not found`

Either:

- place `checkpoint_best_ema.pth` under `inference/model/`, or
- export `MODEL_PATH` before starting the server

### Calibration never becomes ready

Check:

- all four calibration markers are visible
- the board is well lit
- the camera is not too far away
- distortion is not severe enough to break marker geometry

If the image is visibly warped near the edges, tune `CAM_K1` and `CAM_K2`.

### Live stream connects but no scores appear

Check:

- the client is sending valid JPEG bytes
- the camera ID is stable across the session
- the board has a usable calibration
- the detector checkpoint actually matches the expected model architecture

### Video upload connects but stalls

Check:

- the uploaded file extension in metadata matches the actual container
- the video format can be opened by local OpenCV codecs
- the WebSocket client is prepared to alternate between image frames and JSON messages

## Authors

- Struan McLean
- Callum Stewert
- Cristian
- C
=======
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
>>>>>>> main
