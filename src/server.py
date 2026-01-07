from typing import Tuple
from fastapi import FastAPI, WebSocket, UploadFile, File
import json
import numpy as np
import cv2

from camera import Camera
from helpers.filter import filter_reasonable_scores
from helpers.logger import logger
from model.download import download_model
#from prediction import Predict

# -------------------------------------------------------------------
# Model handling
# -------------------------------------------------------------------

from prediction import Prediction, Dartboard, HomographyMapper, load_calibration
from detector_detr import DetrDartDetector
from model.download import download_model

class FramePredict:
    def __init__(self, model_path: str):
        detector = DetrDartDetector(model_path)

        # load calibration once
        # Re-implement dynamic calibration
        top, left, right, bottom, outer_radius = load_calibration(
            "calibration.json"
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

    def main(self, frame, cam_id: int):
        return self.predictor.process_frame(frame, cam_id)

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------

app = FastAPI()

MODEL_LOADED, MODEL_DIR = "/home/cris/Documents/detr-tune/output/checkpoint_best_ema.pth"
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
            cam_id = meta["cam_id"]

            frame_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

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

