from typing import List
import numpy as np
import cv2

from helpers.vect import Vector
from detector import DartDetector

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class DetectronDartDetector(DartDetector):
    """
    Detects dart impact points using Detectron2 and returns
    image-space coordinates as Vector(x, y).
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        device: str = "cpu"
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # darts only
        cfg.MODEL.DEVICE = device

        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.predictor = DefaultPredictor(cfg)

    def detect(self, frame) -> List[Vector]:
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        # Convert boxes to centers
        centers = []
        confidences = []

        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue

            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            centers.append(Vector(cx, cy))
            confidences.append(float(score))

        if not centers:
            return []

        return self._apply_nms(centers, confidences)

    def _apply_nms(
        self,
        points: List[Vector],
        scores: List[float],
        box_size: int = 20
    ) -> List[Vector]:
        """
        Applies NMS to point detections by wrapping them
        in fixed-size boxes.
        """

        if len(points) <= 1:
            return points

        boxes = []
        for p in points:
            x = int(p.x - box_size // 2)
            y = int(p.y - box_size // 2)
            boxes.append([x, y, box_size, box_size])

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.0,
            nms_threshold=self.nms_threshold
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [points[i] for i in indices]

