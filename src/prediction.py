from typing import Optional
from helpers.logger import logger
import json
import math
from helpers.vect import Vector
import cv2
import numpy as np
from helpers.vect import Vector
from helpers.vect import Vector
from darttracker import DartTracker
import cv2
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

def save_calibration(
    path: str,
    top: Vector,
    left: Vector,
    right: Vector,
    bottom: Vector,
    outer_radius: float,
):
    data = {
        "top": {"x": float(top.x), "y": float(top.y)},
        "left": {"x": float(left.x), "y": float(left.y)},
        "right": {"x": float(right.x), "y": float(right.y)},
        "bottom": {"x": float(bottom.x), "y": float(bottom.y)},
        "outer_radius": float(outer_radius),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

class HomographyMapper:
    def __init__(self, H: np.ndarray):
        self.H = H

    def map(self, p: Vector) -> Vector:
        src = np.array([[p.x, p.y, 1.0]], dtype=np.float32).T
        dst = self.H @ src
        dst /= dst[2]
        return Vector(dst[0][0], dst[1][0])

    def unmap(self, p: Vector) -> Vector:
        """Board → image"""
        src = np.array([[p.x, p.y, 1.0]], dtype=np.float32).T
        dst = self.H_inv @ src
        dst /= dst[2]
        return Vector(dst[0][0], dst[1][0])

    @staticmethod
    def from_calibration(
        top: Vector,
        right: Vector,
        bottom: Vector,
        left: Vector,
        outer_radius: float
    ):
        # image points (order does not matter as long as it matches angles)
        image_pts = np.array([
            [top.x, top.y],
            [right.x, right.y],
            [bottom.x, bottom.y],
            [left.x, left.y],
        ], dtype=np.float32)

        # measured real board angles (left = 0°)
        angles_deg = [
            81,   # top
            189,  # right
            261,  # bottom
            9,    # left
        ]

        board_pts = np.array([
            [
                outer_radius * math.cos(math.radians(a)),
                outer_radius * math.sin(math.radians(a)),
            ]
            for a in angles_deg
        ], dtype=np.float32)

        H, _ = cv2.findHomography(image_pts, board_pts, cv2.RANSAC, 3.0)
        if H is None:
            raise RuntimeError("Homography computation failed")

        mapper = HomographyMapper(H)
        mapper.H_inv = np.linalg.inv(H)
        return mapper

def draw_detections(
    frame,
    detected_darts: list[Vector],
    new_darts: list[Vector],
    scores: Optional[list[tuple[int, str]]] = None
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

def _project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    Projects Nx2 points with homography H.
    Returns Nx2 float array.
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])  # Nx3

    dst_h = (H @ pts_h.T).T  # Nx3
    dst_h[:, 0] /= dst_h[:, 2]
    dst_h[:, 1] /= dst_h[:, 2]
    return dst_h[:, :2]


def draw_calibration(
    frame,
    top: Vector,
    left: Vector,
    right: Vector,
    bottom: Vector,
    outer_radius: float,
    mapper,  # HomographyMapper (needs .H)
    ring_color=(255, 0, 0),
    ring_thickness=2,
    ring_segments=180,
):
    """
    Draw calibration landmarks + estimated board geometry.
    Outer ring is drawn as the projection of a board-plane circle (radius=outer_radius)
    through the inverse homography, matching the scoring space.
    """

    pts = {"T": top, "L": left, "R": right, "B": bottom}

    # Draw cardinal points (red)
    for label, p in pts.items():
        cv2.circle(frame, (int(p.x), int(p.y)), 6, (0, 0, 255), -1)
        cv2.putText(
            frame, label,
            (int(p.x) + 6, int(p.y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 2
        )

    # --- Project board-plane center (0,0) and outer circle back into image ---
    Hinv = np.linalg.inv(mapper.H)

    # Center in image = projection of board origin
    c_img = _project_points(Hinv, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    cx, cy = int(round(c_img[0])), int(round(c_img[1]))

    # Center point (blue)
    cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    cv2.putText(
        frame, "C",
        (cx + 6, cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 0, 0), 2
    )

    # Sample points on board-plane circle: (R cos t, R sin t)
    thetas = np.linspace(0, 2 * np.pi, ring_segments, endpoint=False, dtype=np.float32)
    circle_board = np.stack(
        [outer_radius * np.cos(thetas), outer_radius * np.sin(thetas)],
        axis=1
    )  # Nx2

    circle_img = _project_points(Hinv, circle_board)  # Nx2 float

    # Convert to polyline points and draw (this will look like an ellipse if tilted)
    poly = np.round(circle_img).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [poly], isClosed=True, color=ring_color, thickness=ring_thickness)

class Prediction:
    def __init__(
        self,
        mapper: HomographyMapper,
        board: Dartboard
    ):
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
