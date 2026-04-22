# Board-Homography Correction — Debug Post-Mortem

This document records the sequence of defects that caused the projected
dartboard scoring plane (and therefore every dart score) to be wrong, and
explains what the final working configuration reveals about which of those
defects was actually the dominant one.

## Final working configuration

- `cv2.undistort` is wired into the pipeline but runs with
  `k1 = 0, k2 = 0`, i.e. it is effectively an identity transform.
  Lens undistortion is **not** doing any work in the current setup.
- `cv2.getPerspectiveTransform` (not `findHomography` / RANSAC) is used to
  fit the 4-point homography.
- The board-plane calibration radius is fixed at `double_outer_radius`
  (1700 units = 170 mm), matching the real outer edge of the double ring.
- `calibrate()` now (a) keeps per-detection confidence, (b) de-duplicates
  raw detections within 30 px of a higher-confidence one, and (c) takes
  the top 4 remaining points.
- All downstream processing (detection, mapping, drawing) runs in a single
  coordinate space at the input frame's native resolution.

When all of those are in place, every scoring ring (outer double, inner
double, treble-outer, treble-inner, bull, double-bull) sits precisely on
its painted counterpart on the physical board, across the full field.

## What the final fix proves about the earlier theories

The intermediate hypothesis was that the blue outer ring diverging from the
physical board ring — while still passing through the 4 calibration
markers — indicated significant radial lens distortion. We added a tunable
`cv2.undistort` stage and expected non-zero `k1/k2` to be required for the
rings to line up.

The rings ended up lining up with `k1 = k2 = 0`. Therefore the lens
distortion across the region of the image the board occupies is small
enough to be explained perfectly well by a single planar homography. The
camera has visible barrel curvature near the frame edges, but that area
is well outside the board's footprint in the image, so the pinhole +
homography model is sufficient for scoring.

What this means in turn is that the earlier "ring doesn't match the real
board" symptom was **not** caused by lens distortion. It was caused by
the 4 image points being fed into the homography not actually being the
positions we thought they were.

## The real root cause of the final symptom

Three things were wrong with the calibration-point pipeline prior to the
last fix:

1. `DetrDartDetector.calibrate` iterated over every detection classified
   as `align` and hard-required exactly 4 of them. DETR-family models
   often emit multiple overlapping proposals for the same physical
   landmark. In frames where the raw `align` count was 5+ the call
   raised and the frame was dropped; in frames where the raw count
   happened to be exactly 4, the four points could include near-duplicate
   detections of a single physical marker — leaving one marker
   represented twice and another not represented at all. The homography
   would then be built from an inconsistent correspondence set.
2. The detector consumed detections in the order the model returned them,
   with no tie-breaking by confidence. Duplicates always displaced real
   markers non-deterministically from frame to frame.
3. Because the inconsistent detection set was nevertheless exactly 4
   points, `getPerspectiveTransform` still produced a homography and
   `draw_calibration` still drew "a ring through the 4 points." The ring
   passed through whichever 4 points were picked and therefore looked
   geometrically plausible but did not trace the real outer-double-ring
   circle — because two of its anchors weren't on it.

The fix in `@/Users/cris/Documents/college/darts/inference/src/detector_detr.py:161-222`:

- Keep confidence alongside each `align` detection.
- Sort descending by confidence.
- Greedily accept detections, skipping any within 30 px of an already
  accepted (higher-confidence) point — collapsing the DETR duplicates.
- Take the top 4 post-dedup.
- Fail explicitly if fewer than 4 distinct markers survive dedup, with
  an error that distinguishes raw vs post-dedup counts.

Once the four homography anchors genuinely are the four physical
boundary markers (and only those), the homography is exact, the
outer-double ring is drawn on the real outer-double ring, and every
other scoring band — being a concentric circle of known radius in the
same board plane — is drawn on its real painted counterpart.

## Full list of bugs fixed over the debugging session

Listed in the order they were found, each worsening the visual symptom
differently. None of them individually produced the final perfect
alignment; all were necessary.

1. **`outer_radius` passed to the homography was in pixels, not board
   units.** `server.main` computed the mean pixel distance from the
   frame's centroid to the 4 calibration points and fed that as the
   board-plane radius of the marker circle. The scoring thresholds in
   `DartboardScorer` (`double_outer_radius = 1700`, `bull_radius = 159`,
   etc.) are in real dartboard units (0.1 mm). Feeding a ~200 px value
   where the scorer expects 1700 caused every dart to be mapped to
   `r ≤ 200` and therefore always classified in the innermost bands.
   Fix: use `DartboardScorer.double_outer_radius` directly.
2. **`angles_deg` in `HomographyMapper.from_calibration` did not match
   the physical marker layout.** The values rotated the scoring plane
   ~180° relative to reality because they were defined in a coordinate
   convention different from the one `Dartboard.score` uses
   (`+y` down, 20-sector at `(0, −R)`, clockwise from "up"). The
   physical markers sit at the sector boundaries 5/20, 6/10, 3/17,
   11/14 (scoring angles 351°, 99°, 171°, 279°); with image labels
   T / R / B / L assigned by image pixel extremes, the correct
   board-plane angles `a = θ − 90°` are `[189°, 261°, 9°, 81°]`.
3. **Calibration ran on the original-resolution frame but detection
   and drawing ran on a resized 640×480 frame** inside
   `Prediction.process_frame`. The homography was built in 1920×1080
   pixel coords and then used on 640×480 detections and overlays,
   which translated everything by a factor of 3 and pushed the
   projected board centroid off-screen to the bottom-right. Fix:
   remove the internal resize so the whole pipeline shares one
   coordinate space.
4. **`cv2.findHomography(..., cv2.RANSAC, 3.0)` with only 4 points.**
   With exactly 4 correspondences, RANSAC can classify a valid point
   as an outlier at the 3 px threshold and return a degenerate fit.
   Fix: switch to `cv2.getPerspectiveTransform`, the canonical 4-point
   solver.
5. **`DetrDartDetector.calibrate` was fragile and non-deterministic.**
   Described in detail above. This was the last blocker and the one
   whose removal produced the perfect alignment.

A lens-undistortion scaffolding (`FramePredict._undistort`, env-var-tunable
`CAM_FOV_DEG / CAM_K1 / CAM_K2`) was added while investigating (4) and
(5), but is not contributing to the final result and currently runs in
pass-through mode. It is left in place so that if a future camera with
more aggressive fisheye distortion is used, correction can be enabled
without further code changes, and so that a proper `cv2.calibrateCamera`
result can be dropped in the same slot.

## Takeaways

- With 4-point correspondences, always use `getPerspectiveTransform`
  rather than RANSAC-backed `findHomography`.
- Never trust "exactly N detections" as a success condition for a
  DETR-family model. Always de-duplicate and rank by confidence.
- When an overlay "passes through the anchors but deviates between
  them," the first suspicion should be **wrong anchors**, not lens
  distortion. Lens distortion was a plausible-sounding but incorrect
  explanation for the symptom we were seeing; the anchors themselves
  were the problem.
- Keep physical-units (mm) and pixel-units strictly separated at API
  boundaries. The pixel-mean `outer_radius` bug only survived as long
  as it did because nothing in the type system or naming made it
  obvious that the receiver expected a board-plane value.
