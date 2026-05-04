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
from dataclass.dartboard import DartboardScorer

DARTS_PER_ROUND = 3

class Dartboard:
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

        angle = math.degrees(math.atan2(-p.y, p.x))
        angle = (90 - angle) % 360
        angle = (angle + 9) % 360 

        index = int(angle // 18) % 20
        number = DartboardScorer.dartboard_numbers[index]

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
        angles = DartboardScorer.calibration_angles
        markers = [top, right, bottom, left]

        image_pts = np.array(
            [[p.x, p.y] for p in markers],
            dtype=np.float32,
        )

        # Compass angle θ → board (x, y) with +x right, +y down, θ=0 at top.
        board_pts = np.array(
            [
                [
                    outer_radius * math.sin(math.radians(a)),
                    -outer_radius * math.cos(math.radians(a)),
                ]
                for a in angles
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(image_pts, board_pts, cv2.RANSAC, 5.0)

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
    board: Optional["Dartboard"] = None,
):
    """
    Draw calibration landmarks + estimated board geometry.
    Outer ring is drawn as the projection of a board-plane circle (radius=outer_radius)
    through the inverse homography, matching the scoring space.
    """

    pts = {"T": top, "L": left, "R": right, "B": bottom}

    # Draw cardinal points (red, with white outline + large label for visibility)
    for label, p in pts.items():
        cx_p, cy_p = int(p.x), int(p.y)
        cv2.circle(frame, (cx_p, cy_p), 12, (255, 255, 255), 2)  # white halo
        cv2.circle(frame, (cx_p, cy_p), 8, (0, 0, 255), -1)      # red dot
        cv2.putText(
            frame, label,
            (cx_p + 12, cy_p - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 4
        )
        cv2.putText(
            frame, label,
            (cx_p + 12, cy_p - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
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

    def _draw_ring(radius, color, thickness=ring_thickness):
        pts_board = np.stack(
            [radius * np.cos(thetas), radius * np.sin(thetas)], axis=1
        )
        pts_img = _project_points(Hinv, pts_board)
        poly = np.round(pts_img).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)

    # Calibration / outer ring (this should trace the outer edge of the double ring)
    _draw_ring(outer_radius, ring_color, ring_thickness)

    # Additional scoring bands, if the Dartboard is provided. These let you
    # visually compare the resolved scoring plane against the physical board.
    if board is not None:
        _draw_ring(board.double_inner_radius,  (0, 255, 255))  # yellow
        _draw_ring(board.treble_outer_radius,  (0, 255,   0))  # green
        _draw_ring(board.treble_inner_radius,  (0, 255,   0))  # green
        _draw_ring(board.bull_radius,          (0, 165, 255))  # orange
        _draw_ring(board.double_bull_radius,   (0,   0, 255))  # red

class Prediction:
    def __init__(
        self,
        detector,
        mapper: HomographyMapper,
        board: Dartboard
    ):
        self.detector = detector
        self.mapper = mapper
        self.board = board

        # one tracker per camera
        self.trackers = {}
        # per-camera round state: {"scores": [...], "locked": bool}
        self.rounds = {}

    def _get_tracker(self, cam_id: int) -> DartTracker:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = DartTracker()
        return self.trackers[cam_id]

    def _get_round(self, cam_id: int) -> dict:
        if cam_id not in self.rounds:
            self.rounds[cam_id] = {"scores": [], "locked": False}
        return self.rounds[cam_id]

    def process_frame(self, frame, cam_id: int):
        tracker = self._get_tracker(cam_id)
        round_state = self._get_round(cam_id)

        h, w = frame.shape[:2]
        logger.info(f"Frame shape: {w}x{h}")

        darts = self.detector.detect(frame)
        new_darts = tracker.update(darts)

        new_scores = []

        # store scored dart positions (per camera)
        if not hasattr(self, "scored_positions"):
            self.scored_positions = {}

        if cam_id not in self.scored_positions:
            self.scored_positions[cam_id] = []

        scored_positions = self.scored_positions[cam_id]

        def is_duplicate(pos: Vector, threshold: float = 20.0) -> bool:
            for p in scored_positions:
                if math.hypot(p.x - pos.x, p.y - pos.y) < threshold:
                    return True
            return False

        if not round_state["locked"]:
            for dart in new_darts:
                board_pt = self.mapper.map(dart)

                if is_duplicate(dart):
                    continue

                score = self.board.score(board_pt)

                round_state["scores"].append(score)
                new_scores.append(score)

                tracker.mark_as_scored(dart)

                # remember this position so it won't be double counted
                scored_positions.append(dart)

                if len(round_state["scores"]) >= DARTS_PER_ROUND:
                    round_state["locked"] = True
                    break

        # reset when board is empty
        if round_state["locked"] and not tracker.tracked_darts:
            round_state["scores"] = []
            round_state["locked"] = False
            self.scored_positions[cam_id] = []  # reset memory

        draw_detections(
            frame=frame,
            detected_darts=darts,
            new_darts=new_darts,
            scores=new_scores,
        )

        return frame, list(round_state["scores"])

    def reset_camera(self, cam_id: int):
        self.trackers.pop(cam_id, None)
        self.rounds.pop(cam_id, None)
