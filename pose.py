#!/usr/bin/env python3
"""
pose.py - Pallet pose offset solve
FANUC Epoxy Dispense Vision System - Stage 4 (Pose)

Given the nominal fiducial layout and where those fiducials actually landed (both
in robot-plane millimetres), find the rigid 2D transform between them - that
transform IS the pallet offset the PLC validates and the robot applies as a
position-register offset (PR[50] Side A, PR[51] Side B).

  nominal fiducials (mm)  --[ best-fit rigid R,t ]-->  measured fiducials (mm)

Output: dX, dY (mm), dR (deg), per-fiducial residual, RMS / max residual, and a
confidence score. No scale and no reflection are allowed - the pallet is rigid;
a fit that wanted scale would mean a calibration or correspondence error, which
shows up as a large residual (low confidence) rather than a silently absorbed
one.

Typical flow:
    detections = detect.detect_fiducials(gray)          # pixels, per side, IDs
    measured   = {d.fid_id: cal.pixel_to_world([(d.cx, d.cy)])[0] for d in side}
    nominal    = {i+1: xy for i, xy in enumerate(cfg.corners_mm())}
    result     = pose.solve_pose(nominal, measured)

Pure numpy - no OpenCV / Qt - so it is trivially testable.

Conventions (match CLAUDE.md / the design doc):
  +X right, +Y up, +R counter-clockwise in the overhead view. dX/dY are the
  displacement of the nominal-layout CENTROID (the lid/CAD frame origin), and dR
  is the rotation about that centroid - which is what a FANUC position-register
  offset applied in the lid user frame expects. Referencing translation to the
  centroid (not some far world origin) keeps dX/dY from being amplified by a
  rotation lever arm. Confirm the rotation SIGN physically before trusting it.

    python3 pose.py --mock        # synthetic self-test (recovers an injected offset)
"""

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np


# Fit-quality tolerance: how large an RMS residual drives confidence to zero.
# This is NOT the 0.5 mm system budget - it is "does this rigid fit explain the
# four measured points". A clean pinned pallet should sit well under this.
DEFAULT_TOL_MM = 0.25
MIN_POINTS = 2          # rigid needs 2; 3+ gives a meaningful residual


@dataclass
class PoseResult:
    ok: bool
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dr_deg: float = 0.0
    rms_residual_mm: float = 0.0
    max_residual_mm: float = 0.0
    n_points: int = 0
    confidence: float = 0.0
    per_fiducial_mm: dict = field(default_factory=dict)   # id -> residual mm
    reason: str = ""

    def summary(self):
        if not self.ok:
            return f"pose: NO SOLUTION ({self.reason})"
        return (f"dX={self.dx_mm:+.3f} mm  dY={self.dy_mm:+.3f} mm  "
                f"dR={self.dr_deg:+.3f} deg  | RMS {self.rms_residual_mm:.3f} mm  "
                f"MAX {self.max_residual_mm:.3f} mm  conf {self.confidence:.2f}  "
                f"(n={self.n_points})")


# --------------------------------------------------------------------------- #
# Core rigid fit (Kabsch / Umeyama, no scale, no reflection)
# --------------------------------------------------------------------------- #
def solve_rigid(nominal, measured):
    """Best-fit rotation R (2x2) and translation t (2,) with measured ~= R@nominal + t.

    Returns (R, t, theta_rad). Inputs are (N,2) arrays in the same units/frame.
    """
    A = np.asarray(nominal, dtype=np.float64)
    B = np.asarray(measured, dtype=np.float64)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - ca, B - cb

    Hcov = AA.T @ BB                      # 2x2 covariance
    U, _S, Vt = np.linalg.svd(Hcov)
    d = np.sign(np.linalg.det(Vt.T @ U.T))    # guard against a reflection
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    theta = math.atan2(R[1, 0], R[0, 0])
    return R, t, theta


def _confidence(rms_mm, tol_mm, n_points):
    """0..1 fit-quality score. Falls off linearly with RMS up to tol; a bare
    2-point fit (no redundancy to detect error) is capped below 1."""
    base = max(0.0, 1.0 - rms_mm / max(tol_mm, 1e-9))
    if n_points < 3:
        base = min(base, 0.5)
    return float(max(0.0, min(1.0, base)))


# --------------------------------------------------------------------------- #
# High-level solve over matched fiducial IDs
# --------------------------------------------------------------------------- #
def solve_pose(nominal_by_id, measured_by_id, tol_mm=DEFAULT_TOL_MM):
    """Solve the pallet offset from matched fiducials.

    nominal_by_id  : {fid_id: (X_mm, Y_mm)} the nominal layout in the robot plane
    measured_by_id : {fid_id: (X_mm, Y_mm)} the measured fiducials (post-calibration)

    Only IDs present in both are used. Returns a PoseResult.
    """
    ids = sorted(set(nominal_by_id) & set(measured_by_id))
    if len(ids) < MIN_POINTS:
        return PoseResult(ok=False, n_points=len(ids),
                          reason=f"need >= {MIN_POINTS} matched fiducials, got {len(ids)}")

    A = np.array([nominal_by_id[i] for i in ids], dtype=np.float64)
    B = np.array([measured_by_id[i] for i in ids], dtype=np.float64)

    R, t, theta = solve_rigid(A, B)
    pred = (A @ R.T) + t                  # R@A_i + t for each row
    resid = np.linalg.norm(B - pred, axis=1)

    # Report translation as the displacement of the nominal centroid (the lid
    # frame origin), with rotation about that same centroid - not the raw Kabsch
    # t, which is referenced to the coordinate origin and would couple a tiny
    # rotation error into a large dX/dY via the lever arm.
    centroid_shift = B.mean(axis=0) - A.mean(axis=0)

    rms = float(np.sqrt(np.mean(resid ** 2)))
    return PoseResult(
        ok=True,
        dx_mm=float(centroid_shift[0]), dy_mm=float(centroid_shift[1]),
        dr_deg=math.degrees(theta),
        rms_residual_mm=rms,
        max_residual_mm=float(np.max(resid)),
        n_points=len(ids),
        confidence=_confidence(rms, tol_mm, len(ids)),
        per_fiducial_mm={i: float(r) for i, r in zip(ids, resid)},
    )


def apply_offset(points, dx_mm, dy_mm, dr_deg, center=(0.0, 0.0)):
    """Apply (dX, dY, dR) to points (N,2): rotate by dR about `center`, then
    translate by (dX, dY). The inverse of what solve_pose recovers when `center`
    is the nominal centroid. Handy for tests and for previewing an offset."""
    th = math.radians(dr_deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]])
    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    return ((pts - c) @ R.T) + c + np.array([dx_mm, dy_mm])


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _self_test_isolated():
    """Inject a known offset on a nominal layout and recover it."""
    from fiducial_config import FiducialConfig
    cfg = FiducialConfig()
    nominal = {i + 1: xy for i, xy in enumerate(cfg.corners_mm())}

    dx, dy, dr = 1.250, -0.800, 2.500
    rng = np.random.default_rng(1)
    measured = {}
    for i, xy in nominal.items():
        m = apply_offset([xy], dx, dy, dr)[0] + rng.normal(0, 0.01, 2)  # 10 um noise
        measured[i] = tuple(m)

    res = solve_pose(nominal, measured)
    print("[INFO] injected   dX=+1.250 dY=-0.800 dR=+2.500")
    print("[INFO] recovered ", res.summary())
    ok = (res.ok and abs(res.dx_mm - dx) < 0.02 and abs(res.dy_mm - dy) < 0.02
          and abs(res.dr_deg - dr) < 0.05 and res.confidence > 0.8)
    print("[TEST] isolated pose:", "PASS" if ok else "FAIL")
    return ok


def _self_test_end_to_end():
    """Full chain: render a frame at a known offset, detect, calibrate, solve.
    Skipped (not failed) if OpenCV / detect / calibrate are unavailable."""
    try:
        import cv2
        import detect
        import calibrate
        from fiducial_config import FiducialConfig
    except Exception as e:
        print(f"[SKIP] end-to-end chain needs OpenCV/detect/calibrate ({e})")
        return True

    w, h = 1280, 1024
    cfg = FiducialConfig()

    # Nominal fiducial world positions for ONE pallet, placed at a positive
    # origin in the robot plane (mm). IDs 1..4.
    origin = np.array([250.0, 200.0])
    nominal = {i + 1: tuple(np.array(xy) + origin)
               for i, xy in enumerate(cfg.corners_mm())}

    # Ground-truth world->pixel homography that puts the NOMINAL fiducials onto a
    # pad in the image; fit it from those 4 correspondences so the calibration is
    # self-consistent with the rendered scene.
    pad_px = {1: (520, 430), 2: (760, 430), 3: (520, 600), 4: (760, 600)}
    wpts = np.array([nominal[i] for i in (1, 2, 3, 4)], dtype=np.float64)
    ppts = np.array([pad_px[i] for i in (1, 2, 3, 4)], dtype=np.float64)
    Hwp, _ = cv2.findHomography(wpts, ppts, 0)          # world mm -> pixel
    cal = calibrate.fit_homography(ppts, wpts, (w, h), cal_id="cal_e2e")  # pixel -> mm

    # Inject a pallet offset (about the lid centroid, matching how solve_pose
    # reports it), push the fiducials through it in mm, project to pixels, and
    # RENDER a real frame so detection has to find them.
    dx, dy, dr = 0.900, 1.300, -1.800
    centroid = tuple(np.mean([nominal[i] for i in nominal], axis=0))
    measured_world_truth = {i: tuple(pose_apply_single(nominal[i], dx, dy, dr, centroid))
                            for i in nominal}
    mw = np.array([measured_world_truth[i] for i in (1, 2, 3, 4)])
    mp = cv2.perspectiveTransform(mw.reshape(-1, 1, 2), Hwp).reshape(-1, 2)

    frame = _render_pad(w, h, mp)
    dets = detect.detect_fiducials(detect.to_gray8(frame))
    if len(dets) != 4:
        print(f"[TEST] end-to-end: FAIL (detected {len(dets)}/4 fiducials)")
        return False

    measured = {}
    for d in dets:
        measured[d.fid_id] = tuple(cal.pixel_to_world([(d.cx, d.cy)])[0])

    res = solve_pose(nominal, measured)
    print("[INFO] e2e injected   dX=+0.900 dY=+1.300 dR=-1.800")
    print("[INFO] e2e recovered ", res.summary())
    ok = (res.ok and abs(res.dx_mm - dx) < 0.15 and abs(res.dy_mm - dy) < 0.15
          and abs(res.dr_deg - dr) < 0.2 and res.confidence > 0.5)
    print("[TEST] end-to-end pose:", "PASS" if ok else "FAIL")
    return ok


def pose_apply_single(xy, dx, dy, dr, center=(0.0, 0.0)):
    return apply_offset([xy], dx, dy, dr, center)[0]


def _render_pad(w, h, centers_px, ss=4):
    """Bright pad with dark inserts at the given pixel centres (for the e2e test).

    Rendered supersampled (ss x) then area-downsampled so the insert edges are
    anti-aliased - otherwise integer-rasterised disks quantise the fiducial
    centres to whole pixels and the test would measure that, not the detector.
    """
    import cv2
    big = np.full((h * ss, w * ss), 28, np.uint8)
    xs, ys = centers_px[:, 0] * ss, centers_px[:, 1] * ss
    x0, y0 = int(xs.min() - 60 * ss), int(ys.min() - 60 * ss)
    x1, y1 = int(xs.max() + 60 * ss), int(ys.max() + 60 * ss)
    big[max(0, y0):min(h * ss, y1), max(0, x0):min(w * ss, x1)] = 170
    # INTER_AREA maps output coord X to the centre of input block [ss*X, ss*X+ss),
    # i.e. input (ss-1)/2 ; offset the drawn centre by that so the downsampled
    # disk lands exactly on the requested sub-pixel coordinate.
    off = (ss - 1) / 2.0
    for cx, cy in centers_px:
        detect_draw_disk(big, int(round(cx * ss + off)), int(round(cy * ss + off)),
                         22 * ss, 18)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(2)
    return np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape),
                   0, 255).astype(np.uint8)


def detect_draw_disk(img, cx, cy, r, val):
    h, w = img.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val


def _self_test():
    a = _self_test_isolated()
    b = _self_test_end_to_end()
    print("[TEST] PASS" if (a and b) else "[TEST] FAIL")
    return 0 if (a and b) else 1


def main():
    ap = argparse.ArgumentParser(description="Pallet pose offset solve")
    ap.add_argument("--mock", action="store_true", help="Run the synthetic self-test")
    args = ap.parse_args()
    if args.mock:
        sys.exit(_self_test())
    ap.print_help()


if __name__ == "__main__":
    main()
