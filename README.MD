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
