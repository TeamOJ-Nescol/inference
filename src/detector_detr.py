from typing import List, Tuple, Optional
from PIL import Image
import supervision as sv

from helpers.logger import logger
from helpers.vect import Vector
from helpers.points_order import order_calibration_points
from rfdetr import RFDETRSmall

import math
import itertools

"""
def order_calibration_points(points: list[Vector]) -> tuple[Vector, Vector, Vector, Vector]:
    ""
    Orders 4 corner points as: (top_left, top_right, bottom_right, bottom_left)
    ""
    if len(points) != 4:
        raise RuntimeError("Expected exactly 4 calibration points")

    # For image coords: x right, y down
    sums = [(p.x + p.y, p) for p in points]
    diffs = [(p.x - p.y, p) for p in points]

    top_left = min(sums, key=lambda t: t[0])[1]
    bottom_right = max(sums, key=lambda t: t[0])[1]
    top_right = max(diffs, key=lambda t: t[0])[1]
    bottom_left = min(diffs, key=lambda t: t[0])[1]

    ordered = (top_left, top_right, bottom_right, bottom_left)

    if len({id(p) for p in ordered}) != 4:
        raise RuntimeError("Calibration points are not uniquely identifiable")

    return ordered
"""



class DetrDartDetector:
    """
    RF-DETR based dart detector.

    Similar role to DetectronDartDetector, but using RF-DETR.
    """

    DEFAULT_CLASSES = ["unknown", "align", "dart_tip"]

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or self.DEFAULT_CLASSES

        self.model = RFDETRSmall(
            num_classes=len(self.class_names),
            pretrain_weights=checkpoint_path,
            device=device,
        )

        # Uncomment once tensor shape issue is resolved
        # self.model.optimize_for_inference()

    # ------------------------------------------------------------
    # Public API required by DartDetector
    # ------------------------------------------------------------
    def detect(self, frame) -> List[Vector]:
        """
        `frame` can be a NumPy array (OpenCV) or PIL.Image.
        Always returns a list of Vector(x, y) *centres* of dart tips.
        """
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame[..., ::-1])  # BGR → RGB

        detections = self.model.predict(
            frame,
            threshold=self.confidence_threshold,
        )

        dart_vectors: List[Vector] = []
        for xyxy, cid in zip(detections.xyxy, detections.class_id):
            if self.class_names[cid] != "dart_tip":
                continue
            x1, y1, x2, y2 = xyxy
            dart_vectors.append(Vector((x1 + x2) / 2, (y1 + y2) / 2))

        return dart_vectors

    def detect_dart_tips(self, image: Image.Image) -> List[Vector]:
        """
        Detect dart tips and return them as Vector(x, y) points.
        """
        detections = self.detect(image)

        dart_vectors: List[Vector] = []

        for xyxy, class_id in zip(detections.xyxy, detections.class_id):
            if self.class_names[class_id] != "dart_tip":
                continue

            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            dart_vectors.append(Vector(cx, cy))

        return dart_vectors

    def detect_with_metadata(
        self, image: Image.Image
    ) -> List[Tuple[str, float, Vector]]:
        """
        Returns (class_name, confidence, center_vector).
        Useful for debugging or analytics.
        """
        detections = self.detect(image)

        results = []

        for xyxy, class_id, confidence in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
        ):
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            results.append(
                (
                    self.class_names[class_id],
                    float(confidence),
                    Vector(cx, cy),
                )
            )

        return results

    def annotate(self, image: Image.Image) -> Image.Image:
        """
        Draw boxes + labels for visualization.
        """
        detections = self.detect(image)

        labels = [
            f"{self.class_names[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = image.copy()
        annotated = sv.BoxAnnotator().annotate(annotated, detections)
        annotated = sv.LabelAnnotator().annotate(
            annotated, detections, labels
        )
        return annotated

    def calibrate(self, image: Image.Image) -> List[Vector]:
        """
        Detects board alignment markers and returns them as:
        [top, left, right, bottom]

        Raises:
            RuntimeError if calibration markers are missing or ambiguous.
        """
        detections = self.model.predict(
            image,
            threshold=self.confidence_threshold,
        )

        points: List[Vector] = []

        for xyxy, cid in zip(detections.xyxy, detections.class_id):
            if self.class_names[cid] != "align":
                continue

            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            points.append(Vector(cx, cy))

        if len(points) != 4:
            raise RuntimeError(
                f"Expected 4 calibration points, got {len(points)}"
            )

        ordered = order_calibration_points(points)

        return ordered
