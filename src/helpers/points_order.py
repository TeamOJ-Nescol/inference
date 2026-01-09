import math
from typing import Optional, Tuple, List
from helpers.logger import logger
from helpers.vect import Vector

# Assumes Vector has .x and .y and Vector(x, y) constructor.
# If your Vector is different, adjust construction accordingly.

EPS = 1e-9

def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx

def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by

def _norm(ax: float, ay: float) -> float:
    return math.hypot(ax, ay)

def _segment_intersection(
    p1, p2, p3, p4,
    eps: float = 1e-9,
    require_proper: bool = True
) -> Optional[Tuple[object, float, float]]:
    """
    Returns (intersection_point, t, u) where:
      p1 + t*(p2-p1) == p3 + u*(p4-p3)
    If require_proper=True, intersection must be strictly inside both segments.
    """
    r_x, r_y = (p2.x - p1.x), (p2.y - p1.y)
    s_x, s_y = (p4.x - p3.x), (p4.y - p3.y)

    denom = _cross(r_x, r_y, s_x, s_y)
    if abs(denom) < eps:
        return None  # parallel or colinear

    qpx, qpy = (p3.x - p1.x), (p3.y - p1.y)

    t = _cross(qpx, qpy, s_x, s_y) / denom
    u = _cross(qpx, qpy, r_x, r_y) / denom

    if require_proper:
        inside = (eps < t < 1 - eps) and (eps < u < 1 - eps)
    else:
        inside = (-eps <= t <= 1 + eps) and (-eps <= u <= 1 + eps)

    if not inside:
        return None

    ix = p1.x + t * r_x
    iy = p1.y + t * r_y
    return (type(p1)(ix, iy), t, u)

def _find_diagonals_and_center(points: List[object]):
    if len(points) != 4:
        raise RuntimeError("Expected exactly 4 calibration points")

    # The 3 perfect matchings of 4 items:
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    candidates = []
    for (i, j), (k, l) in pairings:
        hit = _segment_intersection(points[i], points[j], points[k], points[l], require_proper=True)
        if hit is not None:
            center, t, u = hit
            # Prefer intersections "deep" inside segments (more stable)
            margin = min(t, 1 - t, u, 1 - u)
            candidates.append((margin, center, (i, j), (k, l)))

    if not candidates:
        # Fallback: allow endpoint intersections (less ideal), else average
        for (i, j), (k, l) in pairings:
            hit = _segment_intersection(points[i], points[j], points[k], points[l], require_proper=False)
            if hit is not None:
                center, t, u = hit
                margin = min(max(0.0, t), max(0.0, 1 - t), max(0.0, u), max(0.0, 1 - u))
                candidates.append((margin, center, (i, j), (k, l)))
                break

    if not candidates:
        # Last resort: arithmetic mean
        cx = sum(p.x for p in points) / 4.0
        cy = sum(p.y for p in points) / 4.0
        center = type(points[0])(cx, cy)
        # No reliable diagonal info
        return center, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, center, pair1, pair2 = candidates[0]
    return center, pair1, pair2

def order_calibration_points(points: list) -> tuple:
    if len(points) != 4:
        raise RuntimeError("Expected exactly 4 calibration points")

    center, pair1, pair2 = _find_diagonals_and_center(points)

    # If we couldn't identify diagonals, just classify by x/y around the mean
    if pair1 is None or pair2 is None:
        top = min(points, key=lambda p: p.y)
        bottom = max(points, key=lambda p: p.y)
        left = min(points, key=lambda p: p.x)
        right = max(points, key=lambda p: p.x)
        ordered = (top, left, right, bottom)
        if len({id(p) for p in ordered}) != 4:
            raise RuntimeError("Calibration points are not uniquely identifiable")
        return ordered

    # Choose which diagonal is "more vertical" (aligned with screen Y axis).
    def diag_vec(pair):
        a, b = pair
        return (points[b].x - points[a].x, points[b].y - points[a].y)

    d1x, d1y = diag_vec(pair1)
    d2x, d2y = diag_vec(pair2)

    def vertical_alignment(dx, dy) -> float:
        n = _norm(dx, dy)
        return 0.0 if n < EPS else abs(dy) / n  # dot with (0,1)

    if vertical_alignment(d2x, d2y) > vertical_alignment(d1x, d1y):
        vdx, vdy = d2x, d2y
    else:
        vdx, vdy = d1x, d1y

    # Build a stable (v_axis, h_axis) basis.
    vn = _norm(vdx, vdy)
    vax, vay = (vdx / vn, vdy / vn)

    # Orient v_axis "downwards" (so top has smaller projection, bottom larger) in screen coords (y increasing down).
    if vay < 0:
        vax, vay = -vax, -vay

    # h_axis is perpendicular; orient it "rightwards" (positive x)
    hax, hay = (-vay, vax)
    if hax < 0:
        hax, hay = -hax, -hay

    # Project each point relative to center
    def proj(p):
        rx, ry = (p.x - center.x), (p.y - center.y)
        v = _dot(rx, ry, vax, vay)
        h = _dot(rx, ry, hax, hay)
        return v, h

    projections = {p: proj(p) for p in points}

    top = min(points, key=lambda p: projections[p][0])
    bottom = max(points, key=lambda p: projections[p][0])
    left = min(points, key=lambda p: projections[p][1])
    right = max(points, key=lambda p: projections[p][1])

    ordered = (top, left, right, bottom)
    if len({id(p) for p in ordered}) != 4:
        raise RuntimeError("Calibration points are not uniquely identifiable")

    return ordered

