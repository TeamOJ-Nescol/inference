from typing import List, Tuple
from PIL import Image
import supervision as sv

from helpers.vect import Vector
from rf_detr.rfdetr import RFDETRSmall


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
        class_names: List[str] | None = None,
        device: str | None = None,
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

    def detect(self, image: Image.Image) -> sv.Detections:
        """
        Run inference and return raw Supervision Detections.
        """
        detections = self.model.predict(
            image,
            threshold=self.confidence_threshold,
        )
        return detections

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

