import math
from typing import Optional, Tuple, List
from helpers.logger import logger
from helpers.vect import Vector


def order_calibration_points(points):
    if len(points) != 4:
        raise RuntimeError("Expected exactly 4 calibration points")

    cx = sum(p.x for p in points) / 4
    cy = sum(p.y for p in points) / 4

    def angle(p):
        # y increases downward, so invert dy for standard math orientation
        return math.atan2(-(p.y - cy), p.x - cx)

    # Sorts CCW: [Right, Top, Left, Bottom]
    pts = sorted(points, key=angle)

    # Re-assigning to shift the order 90 degrees CCW
    # Old: right, top, left, bottom = pts
    # New (Shifted):
    bottom, right, top, left = pts

    return top, left, right, bottom

