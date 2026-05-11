from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import tempfile
import asyncio
from src.dataclass.dartboard import DartboardScorer
import numpy as np
import cv2
import math
import traceback

from helpers.filter import filter_reasonable_scores
from helpers.logger import logger

from prediction import Prediction, Dartboard, HomographyMapper, draw_calibration
from detector_detr import DetrDartDetector

class FramePredict:
    def __init__(self, model_path: str):
        self.detector = DetrDartDetector(
            checkpoint_path=model_path, 
            device="cpu"
        )

        self.board = Dartboard(
            bull_radius=DartboardScorer.bull_radius,
            double_bull_radius=DartboardScorer.double_bull_radius,
            treble_inner_radius=DartboardScorer.treble_inner_radius,
            treble_outer_radius=DartboardScorer.treble_outer_radius,
            double_inner_radius=DartboardScorer.double_inner_radius,
            double_outer_radius=DartboardScorer.double_outer_radius
        )

        self.predictor = None

        self._fov_deg = float(os.environ.get("CAM_FOV_DEG", "90"))
        self._k1 = float(os.environ.get("CAM_K1", "0.0"))
        self._k2 = float(os.environ.get("CAM_K2", "0.0"))
        logger.info(
            f"Lens undistortion params: FOV={self._fov_deg}°, "
            f"k1={self._k1}, k2={self._k2}"
        )

        # Cached per-frame-size undistort maps
        self._undistort_size = None
        self._undistort_map1 = None
        self._undistort_map2 = None

        # Per-camera calibration cache (cam_id -> (top, left, right, bottom, mapper, outer_radius))
        self._calibration_cache = {}

    def _prepare_undistort(self, w: int, h: int):
        if self._undistort_size == (w, h) and self._undistort_map1 is not None:
            return
        fov_rad = math.radians(self._fov_deg)
        fx = w / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # assume square pixels
        cx, cy = w / 2.0, h / 2.0
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        D = np.array([self._k1, self._k2, 0.0, 0.0, 0.0], dtype=np.float64)
        # alpha=1 keeps the full field with black corners allowed, so you can
        # visually judge whether the correction is actually rectifying the
        # image. Switch to 0 once the k1/k2 values are locked in if you want
        # to crop the black borders away.
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2
        )
        self._undistort_size = (w, h)

    def _undistort(self, frame):
        h, w = frame.shape[:2]
        self._prepare_undistort(w, h)
        return cv2.remap(
            frame,
            self._undistort_map1,
            self._undistort_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def _build_predictor(self, top, left, right, bottom, outer_radius):
        mapper = HomographyMapper.from_calibration(
            top=top, left=left, right=right, bottom=bottom, outer_radius=outer_radius,
        )
        if self.predictor is None:
            self.predictor = Prediction(self.detector, mapper, self.board)
        else:
            self.predictor.mapper = mapper
        return mapper

    def reset_calibration(self, cam_id: int):
        self._calibration_cache.pop(cam_id, None)
        if self.predictor is not None:
            self.predictor.reset_camera(cam_id)

    def main(self, frame, cam_id: int):
        # Drop colour information up front: the detector and calibration
        # markers are shape/contrast based, and luminance-only input removes
        # colour-cast confounders (lighting tint, board paint variation).
        # We expand back to 3 channels so the detector (which expects
        # 3-channel BGR/RGB) and downstream annotators continue to work.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Undistort first: the homography is a pinhole projective model, so
        # any residual radial lens distortion causes the projected scoring
        # plane to agree with the calibration points but deviate everywhere
        # else. Doing this before detection+calibration makes every downstream
        # step operate in the (approximately) rectified pinhole view.
        frame = self._undistort(frame)

        cached = None
        if cached is None:
            # Calibrate once per session; the detector raises if the markers
            # can't be located, in which case the cache stays empty and the
            # next frame retries.
            top, left, right, bottom = self.detector.calibrate(frame)

            # The alignment markers sit on the outer edge of the double ring, so
            # the board-plane radius they correspond to is exactly
            # `double_outer_radius`. Using this value (rather than an average of
            # pixel distances) keeps the homography consistent with the fixed
            # scoring thresholds in `Dartboard`.
            outer_radius = float(DartboardScorer.double_outer_radius)

            mapper = self._build_predictor(top, left, right, bottom, outer_radius)
            cached = (top, left, right, bottom, mapper, outer_radius)
            self._calibration_cache[cam_id] = cached
        else:
            top, left, right, bottom, mapper, outer_radius = cached
            if self.predictor is None or self.predictor.mapper is not mapper:
                self._build_predictor(top, left, right, bottom, outer_radius)

        annotated_frame, scores = self.predictor.process_frame(frame, cam_id)

        draw_calibration(
            frame=annotated_frame,
            top=top, left=left, right=right, bottom=bottom,
            outer_radius=outer_radius,
            mapper=mapper,
            board=self.board,
        )

        return annotated_frame, scores


# App setup

app = FastAPI()

# Configure CORS to allow WebSocket connections from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "checkpoint_best_ema.pth"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model checkpoint not found. Set MODEL_PATH or place the checkpoint at "
        f"{DEFAULT_MODEL_PATH}"
    )

predicter = FramePredict(str(MODEL_PATH))


# WebSocket: frame-based inference
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket connection attempt from {client_host}")
    logger.info(f"Headers: {dict(websocket.headers)}")

    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted from {client_host}")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}")
        return

    # Receive initial metadata to get camera ID
    try:
        data = await websocket.receive_text()
        meta = json.loads(data)
        cam_id = meta.get("cam_id", 0)
        logger.info(f"Client requested camera ID: {cam_id}")
    except Exception as e:
        logger.error(f"Failed to receive camera metadata: {e}")
        cam_id = 0

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client {client_host} disconnected from camera {cam_id}")
                break

            text = message.get("text")
            if text:
                control = json.loads(text)
                if control.get("type") == "stop":
                    logger.info(f"Received stop signal for camera {cam_id} from {client_host}")
                    break

                if "cam_id" in control:
                    cam_id = control["cam_id"]
                continue

            frame_bytes = message.get("bytes")
            if not frame_bytes:
                await websocket.send_json({"type": "error", "message": "Missing frame bytes"})
                continue

            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is None:
                await websocket.send_json({"type": "error", "message": "Invalid frame payload"})
                continue

            # Process frame
            try:
                annotated_frame, scores = predicter.main(frame, cam_id)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            # Send annotated frame
            ok, jpg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                await websocket.send_bytes(jpg.tobytes())

            # Send scores
            filtered_scores = filter_reasonable_scores(scores)
            await websocket.send_json({
                "type": "scores",
                "cam_id": cam_id,
                "scores": filtered_scores
            })

            if filtered_scores:
                logger.info(f"Detected scores: {filtered_scores}")

    except Exception as e:
        logger.error(f"WebSocket error from {client_host}: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
    finally:
        predicter.reset_calibration(cam_id)

# WebSocket: video file inference (OpenCV decodes server-side, bypasses browser codec limits)
@app.websocket("/video/ws")
async def video_websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Video WebSocket connection from {client_host}")

    await websocket.accept()

    try:
        meta_text = await websocket.receive_text()
        meta = json.loads(meta_text)
        cam_id = meta.get("cam_id", 0)
        frame_skip = max(1, int(meta.get("frame_skip", 2)))
        logger.info(f"Video session: cam_id={cam_id} frame_skip={frame_skip}")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Invalid metadata: {e}"})
        return

    try:
        video_message = await websocket.receive()
        video_bytes = video_message.get("bytes")
        if not video_bytes:
            try:
                await websocket.send_json({"type": "error", "message": "Expected video bytes"})
            except Exception:
                pass
            return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": f"Failed to receive video: {e}"})
        except Exception:
            pass
        return

    tmp_path = None
    try:
        suffix = meta.get("ext", ".mp4")
        if not suffix.startswith("."):
            suffix = "." + suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            await websocket.send_json({"type": "error", "message": "OpenCV could not open video file"})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        await websocket.send_json({"type": "ready", "total_frames": total_frames})

        frame_idx = 0
        processed = 0

        while True:
            # Check for stop signal (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.001)
                if msg.get("type") == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text:
                    ctrl = json.loads(text)
                    if ctrl.get("type") == "stop":
                        logger.info(f"Stop signal received for video session from {client_host}")
                        break
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"type": "done", "processed_frames": processed})
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            try:
                annotated_frame, scores = predicter.main(frame, cam_id)
            except Exception as e:
                logger.error(f"Inference error on video frame {frame_idx}: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            ok, jpg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                await websocket.send_bytes(jpg.tobytes())

            filtered_scores = filter_reasonable_scores(scores)
            await websocket.send_json({
                "type": "scores",
                "cam_id": cam_id,
                "frame": frame_idx,
                "scores": filtered_scores
            })

            processed += 1

        cap.release()

    except Exception as e:
        logger.error(f"Video WebSocket error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        predicter.reset_calibration(cam_id)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp video file: {tmp_path}")


# Health check
@app.get("/")
async def root():
    return {"message": "Running"}
