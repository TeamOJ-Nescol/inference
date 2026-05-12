from typing import List, Tuple, Optional
from PIL import Image
import supervision as sv

from helpers.logger import logger
from helpers.vect import Vector
from helpers.points_order import order_calibration_points
from rfdetr import RFDETRSmall

import math
import itertools

class DetrDartDetector:
    DEFAULT_CLASSES = ["unknown", "align", "dart_tip"]

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float = 0.2,
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

    def get_calibration_candidates(
        self, image: Image.Image
    ) -> List[Tuple[float, Vector]]:
        detections = self.model.predict(
            image,
            threshold=self.confidence_threshold,
        )

        candidates: List[Tuple[float, Vector]] = []
        for xyxy, cid, conf in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            if self.class_names[cid] != "align":
                continue
            x1, y1, x2, y2 = xyxy
            candidates.append(
                (float(conf), Vector((x1 + x2) / 2, (y1 + y2) / 2))
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates

    def get_distinct_calibration_points(
        self,
        image: Image.Image,
        max_points: Optional[int] = None,
        merge_dist_px: float = 30.0,
    ) -> List[Vector]:
        candidates = self.get_calibration_candidates(image)

        merge_dist_sq = merge_dist_px * merge_dist_px
        kept: List[Vector] = []
        for _conf, p in candidates:
            if all(
                (p.x - q.x) ** 2 + (p.y - q.y) ** 2 > merge_dist_sq
                for q in kept
            ):
                kept.append(p)
            if max_points is not None and len(kept) >= max_points:
                break

        return kept

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
            RuntimeError if fewer than 4 calibration markers are detected.

        If more than 4 candidates are detected (common after lens
        undistortion or with overlapping DETR proposals), near-duplicates
        are merged and the 4 highest-confidence remaining points are used.
        """
        candidates = self.get_calibration_candidates(image)
        if len(candidates) < 4:
            raise RuntimeError(
                f"Expected at least 4 calibration points, got {len(candidates)}"
            )
        kept = self.get_distinct_calibration_points(image, max_points=4)

        if len(kept) < 4:
            raise RuntimeError(
                f"Expected 4 distinct calibration points, "
                f"got {len(kept)} after de-duplication "
                f"(from {len(candidates)} raw detections)"
            )

        ordered = order_calibration_points(kept)

        return ordered
