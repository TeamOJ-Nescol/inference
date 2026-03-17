from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from src.dataclass.dartboard import DartboardScorer
import numpy as np
import cv2
import math
import traceback

from camera import Camera
from helpers.filter import filter_reasonable_scores
from helpers.logger import logger

from prediction import Prediction, Dartboard, HomographyMapper, draw_calibration
from detector_detr import DetrDartDetector

class FramePredict:
    def __init__(self, model_path: str):
        self.detector = DetrDartDetector(checkpoint_path=model_path, device="cpu")

        self.board = Dartboard(
            bull_radius=DartboardScorer.bull_radius,
            double_bull_radius=DartboardScorer.double_bull_radius,
            treble_inner_radius=DartboardScorer.treble_inner_radius,
            treble_outer_radius=DartboardScorer.treble_outer_radius,
            double_inner_radius=DartboardScorer.double_inner_radius,
            double_outer_radius=DartboardScorer.double_outer_radius
        )

        self.predictor = None

    def _build_predictor(self, top, left, right, bottom, outer_radius):
        mapper = HomographyMapper.from_calibration(
            top=top, left=left, right=right, bottom=bottom, outer_radius=outer_radius,
        )
        self.predictor = Prediction(self.detector, mapper, self.board)
        return mapper

    def main(self, frame, cam_id: int):
        # Calibrate on every frame
        top, left, right, bottom = self.detector.calibrate(frame)

        cx = (left.x + right.x) / 2
        cy = (top.y + bottom.y) / 2
        radii = [
            math.hypot(top.x    - cx, top.y    - cy),
            math.hypot(bottom.x - cx, bottom.y - cy),
            math.hypot(left.x   - cx, left.y   - cy),
            math.hypot(right.x  - cx, right.y  - cy),
        ]
        outer_radius = sum(radii) / len(radii)

        mapper = self._build_predictor(top, left, right, bottom, outer_radius)

        annotated_frame, scores = self.predictor.process_frame(frame, cam_id)

        draw_calibration(
            frame=annotated_frame,
            top=top, left=left, right=right, bottom=bottom,
            outer_radius=outer_radius,
            mapper=mapper
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

    # Initialize camera
    camera = None
    try:
        logger.info(f"Initializing camera {cam_id}...")
        camera = Camera(cam_id)
        logger.info(f"Camera {cam_id} initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize camera {cam_id}: {e}")
        await websocket.send_json({"type": "error", "message": f"Failed to initialize camera: {str(e)}"})
        return

    try:
        while True:
            # Capture frame from camera
            frame = camera.get_cur_frame()
            if frame is None:
                logger.warning("Failed to capture frame")
                await websocket.send_json({"type": "error", "message": "Failed to capture frame"})
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
        if camera:
            logger.info(f"Releasing camera {cam_id}")
            camera.cam.release()


# HTTP: single-frame inference

@app.post("/frame/{cam_id}")
async def upload_frame(cam_id: int, file: UploadFile = File(...)):
    image_bytes = await file.read()
    frame = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    annotated_frame, scores = predicter.main(frame, cam_id)
    return {"scores": filter_reasonable_scores(scores)}


# Health check

@app.get("/")
async def root():
    return {"message": "Running"}
