from typing import List
from helpers.vect import Vector

class DartDetector:
    def detect(self, frame) -> List[Vector]:
        """
        Return dart impact points in image coordinates.
        """
        raise NotImplementedError
