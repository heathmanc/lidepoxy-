#!/usr/bin/env python3
"""
calibrate.py - Pixel -> robot-plane calibration
FANUC Epoxy Dispense Vision System - Stage 3 (Calibration)

Turn detected fiducial pixel centres into millimetres in the robot work plane at
lid height. This is the bridge between detection (pixels) and the pose solve
(mm offset the PLC / robot consume).

Model
-----
A planar homography (cv2.findHomography) maps image pixels to the robot plane.
This is the right first model because the lid surface is planar and the camera
is fixed: a single 3x3 projective transform absorbs scale, the small overhead
tilt, and mild keystone. If edge residuals ever eat into the 0.5 mm budget, the
next step is full intrinsics + distortion (cv2.calibrateCamera) - the Calibration
container here is written so that swap is localised.

Every calibration is persisted with:
  - an ID + timestamp (so a stale calibration can't be silently reused),
  - the image size it was fit for,
  - RMS and MAX reprojection error in mm (the numbers you gate on).

The pose stage records the Calibration ID alongside the offset it sends, so the
PLC can reject an offset computed against the wrong calibration.

Usage
-----
    python3 calibrate.py --mock                 # synthetic self-test
    python3 calibrate.py --points pts.json --out calibration/cal.json
        pts.json: {"image_size": [w, h],
                   "pairs": [{"px": [x, y], "mm": [X, Y]}, ...]}   (>= 4 pairs)
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


SCHEMA = "lidepoxy.calibration/1"


@dataclass
class Calibration:
    """A pixel -> robot-plane (mm) homography plus the metadata you gate on."""
    H: np.ndarray                      # 3x3, maps pixel [x,y,1] -> world [X,Y,1]
    image_size: tuple                  # (width, height) the fit was made for
    rms_error_mm: float
    max_error_mm: float
    n_points: int
    cal_id: str = ""
    created: str = ""
    method: str = "homography"
    schema: str = SCHEMA

    # ---- transforms ---------------------------------------------------- #
    def pixel_to_world(self, pts):
        """pixels (N,2) -> robot-plane mm (N,2)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def world_to_pixel(self, pts):
        """robot-plane mm (N,2) -> pixels (N,2)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, np.linalg.inv(self.H))
        return out.reshape(-1, 2)

    # ---- persistence --------------------------------------------------- #
    def to_dict(self):
        return {
            "schema": self.schema,
            "cal_id": self.cal_id,
            "created": self.created,
            "method": self.method,
            "image_size": list(self.image_size),
            "n_points": self.n_points,
            "rms_error_mm": self.rms_error_mm,
            "max_error_mm": self.max_error_mm,
            "H": self.H.tolist(),
        }

    def save(self, path):
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d):
        return cls(
            H=np.asarray(d["H"], dtype=np.float64),
            image_size=tuple(d.get("image_size", (0, 0))),
            rms_error_mm=float(d.get("rms_error_mm", 0.0)),
            max_error_mm=float(d.get("max_error_mm", 0.0)),
            n_points=int(d.get("n_points", 0)),
            cal_id=d.get("cal_id", ""),
            created=d.get("created", ""),
            method=d.get("method", "homography"),
            schema=d.get("schema", SCHEMA),
        )

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self):
        w, h = self.image_size
        return (f"Calibration {self.cal_id or '(unnamed)'}  {self.method}  "
                f"{self.n_points} pts  {w}x{h}  "
                f"RMS {self.rms_error_mm:.4f} mm  MAX {self.max_error_mm:.4f} mm")


# --------------------------------------------------------------------------- #
# Fitting / validation
# --------------------------------------------------------------------------- #
def _make_id():
    return "cal_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def reprojection_error_mm(H, pixel_pts, world_pts):
    """Per-point pixel->world error in mm, given a fitted pixel->world H."""
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 1, 2)
    world_pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 2)
    pred = cv2.perspectiveTransform(pixel_pts, H).reshape(-1, 2)
    return np.linalg.norm(pred - world_pts, axis=1)


def fit_homography(pixel_pts, world_pts, image_size, cal_id=None, robust=False):
    """Fit a pixel -> robot-plane (mm) homography from >= 4 correspondences.

    pixel_pts : (N,2) detected fiducial / grid centres in pixels
    world_pts : (N,2) the same points' known robot-plane positions in mm
    robust    : use RANSAC (tolerates a bad correspondence) instead of plain LS
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required: pip3 install opencv-python")
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64)
    world_pts = np.asarray(world_pts, dtype=np.float64)
    if len(pixel_pts) < 4 or len(pixel_pts) != len(world_pts):
        raise ValueError("need >= 4 matched pixel/world points")

    method = cv2.RANSAC if robust else 0
    H, _mask = cv2.findHomography(pixel_pts, world_pts, method)
    if H is None:
        raise RuntimeError("findHomography failed (degenerate point layout?)")

    err = reprojection_error_mm(H, pixel_pts, world_pts)
    return Calibration(
        H=H,
        image_size=tuple(image_size),
        rms_error_mm=float(np.sqrt(np.mean(err ** 2))),
        max_error_mm=float(np.max(err)),
        n_points=len(pixel_pts),
        cal_id=cal_id or _make_id(),
        created=datetime.now().isoformat(timespec="seconds"),
        method="homography" + ("+ransac" if robust else ""),
    )


def validate(cal, pixel_pts, world_pts):
    """Reprojection stats for held-out check points (mm). Returns (rms, max, n)."""
    err = reprojection_error_mm(cal.H, pixel_pts, world_pts)
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(err)), len(err)


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #
def _synthetic_truth(w=1280, h=1024):
    """A plausible world(mm) -> pixel homography for the mock geometry: ~0.5
    mm/px, a few degrees of overhead tilt, and mild keystone."""
    mm_per_px = 0.55
    # world->pixel: scale + rotation, then a touch of perspective.
    th = math.radians(3.0)
    s = 1.0 / mm_per_px
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]])
    A = s * R
    Hwp = np.array([[A[0, 0], A[0, 1], w / 2.0],
                    [A[1, 0], A[1, 1], h / 2.0],
                    [2e-5,    1e-5,    1.0]])
    return Hwp


def _self_test():
    w, h = 1280, 1024
    Hwp = _synthetic_truth(w, h)            # world mm -> pixel (ground truth)

    # A 6x5 mm grid spanning roughly the work area, centred on the origin.
    xs = np.linspace(-300, 300, 6)
    ys = np.linspace(-200, 200, 5)
    world = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), Hwp).reshape(-1, 2)
    rng = np.random.default_rng(0)
    pixel = proj + rng.normal(0, 0.1, proj.shape)     # 0.1 px detection noise

    # Fit on a training split, validate on the held-out split.
    idx = np.arange(len(world))
    train, test = idx[idx % 3 != 0], idx[idx % 3 == 0]
    cal = fit_homography(pixel[train], world[train], (w, h), cal_id="cal_selftest")
    print("[INFO]", cal.summary())
    v_rms, v_max, v_n = validate(cal, pixel[test], world[test])
    print(f"[INFO] hold-out validation: RMS {v_rms:.4f} mm  MAX {v_max:.4f} mm  "
          f"({v_n} pts)")

    # Round-trip persistence.
    import os, tempfile
    p = os.path.join(tempfile.gettempdir(), "cal_selftest.json")
    cal.save(p)
    back = Calibration.load(p)
    rt = float(np.max(np.abs(back.H - cal.H)))

    ok = cal.rms_error_mm < 0.1 and v_max < 0.2 and rt < 1e-12
    print(f"[TEST] persistence round-trip Δ = {rt:.2e}")
    print("[TEST] PASS" if ok else "[TEST] FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    if cv2 is None:
        sys.exit("OpenCV not found. Install with: pip3 install opencv-python")

    ap = argparse.ArgumentParser(description="Pixel -> robot-plane calibration")
    ap.add_argument("--points", help="JSON of correspondences (see module docstring)")
    ap.add_argument("--out", help="Write the fitted calibration JSON here")
    ap.add_argument("--robust", action="store_true", help="RANSAC fit")
    ap.add_argument("--mock", action="store_true", help="Run the synthetic self-test")
    args = ap.parse_args()

    if args.mock or not args.points:
        sys.exit(_self_test())

    with open(args.points) as f:
        data = json.load(f)
    pairs = data["pairs"]
    pixel = [p["px"] for p in pairs]
    world = [p["mm"] for p in pairs]
    image_size = tuple(data.get("image_size", (0, 0)))
    cal = fit_homography(pixel, world, image_size, robust=args.robust)
    print("[INFO]", cal.summary())
    if args.out:
        cal.save(args.out)
        print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
