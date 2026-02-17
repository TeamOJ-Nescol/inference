from typing import Tuple
from fastapi import FastAPI, WebSocket, UploadFile, File
import json
from src.dataclass.dartboard import DartboardScorer
import numpy as np
import cv2
import math

from camera import Camera
from helpers.filter import filter_reasonable_scores
from helpers.logger import logger
from helpers.vect import Vector
import os
from pathlib import Path

from prediction import Prediction, Dartboard, HomographyMapper, save_calibration, draw_calibration
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


# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------

app = FastAPI()

MODEL_DIR = "/Users/struanmclean/Documents/SpazzyDarts/model/checkpoint_best_ema.pth"
predicter = FramePredict(str(MODEL_DIR))


# -------------------------------------------------------------------
# WebSocket: frame-based inference
# -------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            meta = json.loads(await websocket.receive_text())
            cam_id = meta.get("cam_id", 0)

            frame_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            frame = cv2.resize(frame, (640, 480))

            try:
                annotated_frame, scores = predicter.main(frame, cam_id)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            ok, jpg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                await websocket.send_bytes(jpg.tobytes())

            await websocket.send_json({
                "type": "scores",
                "cam_id": cam_id,
                "scores": filter_reasonable_scores(scores)
            })

    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# -------------------------------------------------------------------
# HTTP: single-frame inference
# -------------------------------------------------------------------

@app.post("/frame/{cam_id}")
async def upload_frame(cam_id: int, file: UploadFile = File(...)):
    image_bytes = await file.read()
    frame = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    annotated_frame, scores = predicter.main(frame, cam_id)
    return {"scores": filter_reasonable_scores(scores)}


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Running"}