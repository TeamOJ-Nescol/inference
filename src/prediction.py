#!prediction.py

from helpers.logger import logger

# ---- Board-specific logic + scoring

import math
from helpers.vect import Vector

class Dartboard:
    NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
               3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    def __init__(
        self,
        bull_radius: float,
        double_bull_radius: float,
        treble_inner_radius: float,
        treble_outer_radius: float,
        double_inner_radius: float,
        double_outer_radius: float
    ):
        self.bull_radius = bull_radius
        self.double_bull_radius = double_bull_radius
        self.treble_inner_radius = treble_inner_radius
        self.treble_outer_radius = treble_outer_radius
        self.double_inner_radius = double_inner_radius
        self.double_outer_radius = double_outer_radius

    def score(self, p: Vector) -> tuple[int, str]:
        r = math.hypot(p.x, p.y)

        if r > self.double_outer_radius:
            return 0, "Miss"

        angle = (90 - math.degrees(math.atan2(-p.y, p.x))) % 360
        index = int(angle // 18) % 20
        number = self.NUMBERS[index]

        if r <= self.double_bull_radius:
            return 50, "Double Bull"
        if r <= self.bull_radius:
            return 25, "Bull"
        if r <= self.treble_inner_radius:
            return number, str(number)
        if r <= self.treble_outer_radius:
            return number * 3, f"T{number}"
        if r <= self.double_inner_radius:
            return number, str(number)
        return number * 2, f"D{number}"


# ---- Calibration handling

import json

def load_calibration(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    return (
        Vector(**data["top"]),
        Vector(**data["left"]),
        Vector(**data["right"]),
        Vector(**data["bottom"]),
        data["outer_radius"]
    )


# ---- Homography mapping


import cv2
import numpy as np
from helpers.vect import Vector

class HomographyMapper:
    def __init__(self, H: np.ndarray):
        self.H = H

    def map(self, p: Vector) -> Vector:
        src = np.array([[p.x, p.y, 1.0]], dtype=np.float32).T
        dst = self.H @ src
        dst /= dst[2]
        return Vector(dst[0][0], dst[1][0])

    @staticmethod
    def from_calibration(
        top: Vector,
        left: Vector,
        right: Vector,
        bottom: Vector,
        outer_radius: float
    ):
        image_pts = np.array([
            [top.x, top.y],
            [left.x, left.y],
            [right.x, right.y],
            [bottom.x, bottom.y]
        ], dtype=np.float32)

        board_pts = np.array([
            [0, -outer_radius],
            [-outer_radius, 0],
            [outer_radius, 0],
            [0, outer_radius]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(image_pts, board_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Homography computation failed")

        return HomographyMapper(H)

# ---- Debug display
import cv2
from helpers.vect import Vector

def draw_detections(
    frame,
    detected_darts: list[Vector],
    new_darts: list[Vector],
    scores: list[tuple[int, str]] | None = None
):
    """
    Draws detection + scoring overlays on frame (in-place).
    """

    # Raw detections (yellow)
    for p in detected_darts:
        cv2.circle(frame, (int(p.x), int(p.y)), 6, (0, 255, 255), 2)

    # Newly scored darts (green)
    for i, p in enumerate(new_darts):
        cv2.circle(frame, (int(p.x), int(p.y)), 8, (0, 255, 0), -1)

        if scores:
            score, label = scores[i]
            cv2.putText(
                frame,
                label,
                (int(p.x) + 10, int(p.y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


# ---- Orchestration

from helpers.vect import Vector
from darttracker import DartTracker
from detector import DartDetector

class Prediction:
    def __init__(
        self,
        detector: DartDetector,
        mapper: HomographyMapper,
        board: Dartboard
    ):
        self.detector = detector
        self.mapper = mapper
        self.board = board

        # one tracker per camera
        self.trackers = {}

    def _get_tracker(self, cam_id: int) -> DartTracker:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = DartTracker()
        return self.trackers[cam_id]

    def process_frame(self, frame, cam_id: int):
        """
        Returns:
            annotated_frame,
            List[(score:int, label:str)]
        """
        tracker = self._get_tracker(cam_id)

        h, w = frame.shape[:2]
        logger.info(f"Frame shape: {w}x{h}")
        frame = cv2.resize(frame, (640, 480))

        darts = self.detector.detect(frame)
        new_darts = tracker.update(darts)

        scores = []
        for dart in new_darts:
            board_pt = self.mapper.map(dart)
            score = self.board.score(board_pt)
            scores.append(score)
            tracker.mark_as_scored(dart)

        #!TODO make this optional
        # ---- DRAW OVERLAYS
        draw_detections(
            frame=frame,
            detected_darts=darts,
            new_darts=new_darts,
            scores=scores,
        )
        logger.info(f"Annotated frame mean: {frame.mean()}")
        return frame, scores


    def reset_camera(self, cam_id: int):
        self.trackers.pop(cam_id, None)
