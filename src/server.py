from typing import Tuple
from fastapi import FastAPI, WebSocket, UploadFile, File
import json
import numpy as np
import cv2
import math

from camera import Camera
from helpers.filter import filter_reasonable_scores
from helpers.logger import logger
from model.download import download_model
from helpers.vect import Vector
#from prediction import Predict
import os
from pathlib import Path

# -------------------------------------------------------------------
# Model handling
# -------------------------------------------------------------------

from prediction import Prediction, Dartboard, HomographyMapper, load_calibration, save_calibration, draw_calibration
from detector_detr import DetrDartDetector
from model.download import download_model

CALIBRATION_PATH = (
    Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
    / "dartboard"
    / "calibration.json"
)
CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)

class FramePredict:
    def __init__(self, model_path: str):
        detector = DetrDartDetector(checkpoint_path = model_path, device = "cuda")
        self.detector = detector

        # load calibration once
        # Re-implement dynamic calibration
        top, left, right, bottom, outer_radius = load_calibration(
            CALIBRATION_PATH
        )

        mapper = HomographyMapper.from_calibration(
            top=top,
            left=left,
            right=right,
            bottom=bottom,
            outer_radius=outer_radius
        )

        board = Dartboard(
            bull_radius=6.35,
            double_bull_radius=3.175,
            treble_inner_radius=99,
            treble_outer_radius=107,
            double_inner_radius=162,
            double_outer_radius=170
        )

        self.predictor = Prediction(detector, mapper, board)
        self.calibration = None

    def calibrate(self, frame):
        """
        Detect alignment points, compute homography, and save calibration.
        """
        # ---- Detect alignment points
        top, left, right, bottom = self.detector.calibrate(frame)

        # ---- Estimate board center
        cx = (left.x + right.x) / 2
        cy = (top.y + bottom.y) / 2
        center = Vector(cx, cy)

        # ---- Estimate outer radius (average distance)
        radii = [
            math.hypot(top.x - cx, top.y - cy),
            math.hypot(bottom.x - cx, bottom.y - cy),
            math.hypot(left.x - cx, left.y - cy),
            math.hypot(right.x - cx, right.y - cy),
        ]
        outer_radius = sum(radii) / len(radii)

        # ---- Create new mapper
        mapper = HomographyMapper.from_calibration(
            top=top,
            left=left,
            right=right,
            bottom=bottom,
            outer_radius=outer_radius,
        )

        # ---- Hot-swap mapper
        self.predictor.mapper = mapper

        # ---- Persist calibration
        save_calibration(
            path=CALIBRATION_PATH,
            top=top,
            left=left,
            right=right,
            bottom=bottom,
            outer_radius=outer_radius,
        )

        self.calibration = {
            "top": top,
            "left": left,
            "right": right,
            "bottom": bottom,
            "outer_radius": outer_radius,
        }

        return self.calibration

    def main(self, frame, cam_id: int):

        annotated_frame, scores = self.predictor.process_frame(frame, cam_id)

        # ---- DRAW CALIBRATION OVERLAY (if available)
        if self.calibration:
            logger.info(f"Drawing calibration.")
            draw_calibration(
                frame=annotated_frame,
                top=self.calibration["top"],
                left=self.calibration["left"],
                right=self.calibration["right"],
                bottom=self.calibration["bottom"],
                outer_radius=self.calibration["outer_radius"],
                mapper=self.predictor.mapper
            )

        return annotated_frame, scores

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------

app = FastAPI()

MODEL_DIR = "/home/cris/Documents/detr-tune/output/checkpoint_best_ema.pth"
MODEL_LOADED = True
predicter = FramePredict(str(MODEL_DIR))


# -------------------------------------------------------------------
# WebSocket: frame-based inference
# -------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    calibrated = False

    try:
        while True:
            meta = json.loads(await websocket.receive_text())
            cam_id = meta["cam_id"]

            frame_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            frame = cv2.resize(frame, (640, 480)) # Change to global value

            if not calibrated:
                logger.info(f"Computing calibration.")
                predicter.calibrate(frame)
                logger.info(f"Calibration finished.")
                calibrated = True

            annotated_frame, scores = predicter.main(frame, cam_id)

            # Encode annotated frame
            ok, jpg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                await websocket.send_bytes(jpg.tobytes())

            # Send scores separately (or together if you want)
            await websocket.send_json({
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

    scores = predicter.main(frame, cam_id)
    return {"scores": filter_reasonable_scores(scores)}

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Running"}

