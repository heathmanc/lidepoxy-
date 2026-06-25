#!/usr/bin/env python3
"""
detect.py - Fiducial detection
FANUC Epoxy Dispense Vision System - Stage 2 (Detection)

Find the bored-circle fiducials (dark insert on a bright backing) in a captured
frame and return a sub-pixel centre per fiducial, plus quality metrics that flag
chips / occlusions. These centres are the input to calibration (pixel -> robot
plane) and the pose solve.

What this stage does (per the roadmap in CLAUDE.md):

  - crop the expected ROI per fiducial  (detect_in_roi / detect_fiducials)
  - threshold the dark insert against the bright backing  (Otsu, inverted)
  - find the insert contour and fit it  (ellipse fit -> sub-pixel centre)
  - quality-check it: circularity, fill ratio, and a concentricity test
    (intensity centroid vs. fitted geometric centre) so a chipped or occluded
    insert is caught instead of silently shifting the measured centre

There are two ways to drive it:

  detect_fiducials(gray)                         auto: find the two bright pads,
                                                 then the (up to) four inserts on
                                                 each, ID them 1..4 per side.
                                                 Use this before calibration and
                                                 for the live GUI overlay.

  detect_fiducials(gray, expected_centers=...)   ROI-guided: you already know the
                                                 approximate pixel location of
                                                 each fiducial (from calibration),
                                                 so search a tight window around
                                                 each one.

Geometry (diameter / spacing) comes from fiducial_config.FiducialConfig - the
same numbers the setup GUI edits. Nothing here duplicates those constants.

Run standalone:
    python3 detect.py path/to/frame.png          # detect + print + annotate
    python3 detect.py --mock                      # synthetic scene self-test
    python3 detect.py frame.png --save out.png    # write an annotated overlay
"""

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # detection genuinely needs OpenCV
    cv2 = None


# --------------------------------------------------------------------------- #
# Tunables - first-cut values, validate against real captured frames.
# --------------------------------------------------------------------------- #
MIN_FID_AREA_PX = 40       # ignore specks below this contour area
MAX_FID_AREA_FRAC = 0.20   # ignore blobs bigger than this fraction of the ROI/pad
MIN_CIRCULARITY = 0.55     # 4*pi*A / P^2 ; 1.0 is a perfect circle
MIN_FILL_RATIO = 0.65      # contour area / enclosing-circle area ; ~1 for a disk
MIN_PAD_AREA_FRAC = 0.01   # a "bright pad" must be at least this much of the frame
GAUSS_KSIZE = 3            # pre-threshold blur (odd); knocks down sensor noise


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One detected fiducial. Centre is sub-pixel, in full-image pixels."""
    cx: float
    cy: float
    radius_px: float
    circularity: float        # 4*pi*A/P^2 ; closer to 1 = rounder
    fill_ratio: float         # area / enclosing-circle area
    concentricity_px: float   # intensity centroid vs fitted centre (chip flag)
    axis_ratio: float         # ellipse minor/major ; 1 = circular
    confidence: float         # 0..1 combined quality score
    side: str = ""            # "A" / "B" when grouped by pad, else ""
    fid_id: int = 0           # 1..4 when assigned, else 0

    def label(self):
        if self.side and self.fid_id:
            return f"{self.side}{self.fid_id}"
        if self.fid_id:
            return str(self.fid_id)
        return "?"


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_gray8(img):
    """Return a contiguous single-channel uint8 view for processing.

    Mono12 frames arrive as uint16 (12 significant bits); shift them down to
    8-bit. Detection thresholds are intensity-relative (Otsu), so 8-bit is plenty
    for finding centres; the full-depth frame is still what gets saved/measured.
    """
    a = np.asarray(img)
    if a.dtype == np.uint16:
        a = (a >> 4).astype(np.uint8) if a.max() > 255 else a.astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(a)


def _clip01(v):
    return float(max(0.0, min(1.0, v)))


# --------------------------------------------------------------------------- #
# Single-blob measurement (the heart of the detector)
# --------------------------------------------------------------------------- #
def _refine_center(contour, fallback_cx, fallback_cy, fallback_r):
    """Sub-pixel centre + axis ratio from an ellipse fit (>=5 points needed).

    Falls back to the moment centroid / enclosing radius for tiny contours.
    """
    if len(contour) >= 5:
        try:
            (ex, ey), (d1, d2), _ang = cv2.fitEllipse(contour)
            major, minor = max(d1, d2), min(d1, d2)
            axis = (minor / major) if major > 0 else 0.0
            return float(ex), float(ey), (major + minor) / 4.0, axis
        except cv2.error:
            pass
    return float(fallback_cx), float(fallback_cy), float(fallback_r), 1.0


def _measure(contour, roi_area, expected_r_px=None):
    """Measure one candidate contour. Returns a Detection or None if rejected."""
    area = cv2.contourArea(contour)
    if area < MIN_FID_AREA_PX or area > MAX_FID_AREA_FRAC * roi_area:
        return None
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return None

    circularity = 4.0 * math.pi * area / (perim * perim)
    (_mx, _my), r_enc = cv2.minEnclosingCircle(contour)
    if r_enc <= 0:
        return None
    fill_ratio = area / (math.pi * r_enc * r_enc)

    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    cgx, cgy = m["m10"] / m["m00"], m["m01"] / m["m00"]   # intensity centroid

    cx, cy, radius, axis_ratio = _refine_center(contour, cgx, cgy, r_enc)

    # Concentricity: a clean disk has its area centroid on the fitted geometric
    # centre. A chip / occlusion biases the centroid off-centre - that gap (in
    # pixels, normalised by radius for the score) is our tamper flag.
    concentricity = math.hypot(cgx - cx, cgy - cy)

    if circularity < MIN_CIRCULARITY or fill_ratio < MIN_FILL_RATIO:
        return None
    if expected_r_px is not None and not (0.5 * expected_r_px <= radius <= 1.7 * expected_r_px):
        return None

    confidence = (
        _clip01(circularity)
        * _clip01(axis_ratio)
        * _clip01(fill_ratio)
        * _clip01(1.0 - concentricity / max(radius, 1.0))
    )
    return Detection(
        cx=cx, cy=cy, radius_px=radius,
        circularity=circularity, fill_ratio=fill_ratio,
        concentricity_px=concentricity, axis_ratio=axis_ratio,
        confidence=confidence,
    )


def _dark_blobs(gray_roi):
    """Threshold the dark inserts out of a (mostly bright) ROI and return contours."""
    k = GAUSS_KSIZE if GAUSS_KSIZE % 2 == 1 else GAUSS_KSIZE + 1
    g = cv2.GaussianBlur(gray_roi, (k, k), 0)
    # Inverted Otsu: dark insert -> white foreground, bright backing -> black.
    _t, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts


# --------------------------------------------------------------------------- #
# ROI-guided detection (one fiducial expected near the ROI centre)
# --------------------------------------------------------------------------- #
def detect_in_roi(gray, x0, y0, w, h, expected_diameter_px=None):
    """Detect the single best fiducial inside the pixel rectangle (x0,y0,w,h).

    Use when you already know roughly where each fiducial is (post-calibration).
    Returns a Detection in full-image coordinates, or None.
    """
    gray = to_gray8(gray)
    H, W = gray.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = gray[y0:y1, x0:x1]
    roi_area = crop.shape[0] * crop.shape[1]
    exp_r = (expected_diameter_px / 2.0) if expected_diameter_px else None
    cxr, cyr = (x1 - x0) / 2.0, (y1 - y0) / 2.0

    best, best_score = None, -1.0
    for c in _dark_blobs(crop):
        m = _measure(c, roi_area, exp_r)
        if m is None:
            continue
        # Prefer the most confident blob nearest the ROI centre.
        dist = math.hypot(m.cx - cxr, m.cy - cyr)
        score = m.confidence - 0.001 * dist
        if score > best_score:
            best, best_score = m, score
    if best is None:
        return None
    best.cx += x0
    best.cy += y0
    return best


# --------------------------------------------------------------------------- #
# Auto detection (no prior positions): find pads, then inserts on each
# --------------------------------------------------------------------------- #
def find_pads(gray, max_pads=2):
    """Locate the bright pad regions; return up to `max_pads` boxes, left->right."""
    gray = to_gray8(gray)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _t, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = gray.shape[0] * gray.shape[1]
    pads = [cv2.boundingRect(c) for c in cnts
            if cv2.contourArea(c) >= MIN_PAD_AREA_FRAC * frame_area]
    pads.sort(key=lambda r: r[2] * r[3], reverse=True)   # biggest first
    pads = pads[:max_pads]
    pads.sort(key=lambda r: r[0])                         # then left -> right
    return pads


def _assign_ids(dets):
    """Number up to four detections 1..4 (TL, TR, BL, BR) by their positions."""
    if not dets:
        return
    mx = float(np.median([d.cx for d in dets]))
    my = float(np.median([d.cy for d in dets]))
    for d in dets:
        top, left = d.cy < my, d.cx < mx
        d.fid_id = 1 if (top and left) else 2 if (top and not left) \
            else 3 if (not top and left) else 4


def detect_in_pad(gray, pad, max_count=4, expected_diameter_px=None):
    """Detect the (up to) `max_count` inserts inside one pad box. No IDs assigned."""
    gray = to_gray8(gray)
    x, y, w, h = pad
    crop = gray[y:y + h, x:x + w]
    roi_area = crop.shape[0] * crop.shape[1]
    exp_r = (expected_diameter_px / 2.0) if expected_diameter_px else None

    cands = []
    for c in _dark_blobs(crop):
        m = _measure(c, roi_area, exp_r)
        if m is None:
            continue
        m.cx += x
        m.cy += y
        cands.append(m)
    cands.sort(key=lambda d: d.confidence, reverse=True)
    return cands[:max_count]


def detect_fiducials(gray, expected_centers=None, roi_half_px=80,
                     expected_diameter_px=None):
    """Top-level detection.

    expected_centers given -> ROI-guided: a dict {fid_id: (x_px, y_px)} or a
        sequence of (x_px, y_px); each is searched in a +/- roi_half_px window.
    expected_centers None  -> auto: find the two bright pads (Side A = left,
        Side B = right), detect up to four inserts on each, and number them 1..4.

    Returns a list of Detection in full-image pixel coordinates.
    """
    gray = to_gray8(gray)

    if expected_centers is not None:
        items = (expected_centers.items() if isinstance(expected_centers, dict)
                 else enumerate(expected_centers, start=1))
        out = []
        for fid_id, (x, y) in items:
            d = detect_in_roi(gray, x - roi_half_px, y - roi_half_px,
                              2 * roi_half_px, 2 * roi_half_px,
                              expected_diameter_px)
            if d is not None:
                d.fid_id = int(fid_id)
                out.append(d)
        return out

    out = []
    for side, pad in zip(("A", "B"), find_pads(gray)):
        dets = detect_in_pad(gray, pad, max_count=4,
                             expected_diameter_px=expected_diameter_px)
        _assign_ids(dets)
        for d in dets:
            d.side = side
        out.extend(dets)
    return out


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def annotate(img, detections):
    """Return a BGR copy of `img` with detections drawn on it (for CLI / saves)."""
    bgr = to_gray8(img)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    for d in detections:
        good = d.confidence >= 0.6
        color = (80, 220, 80) if good else (60, 170, 235)   # BGR green / amber
        c = (int(round(d.cx)), int(round(d.cy)))
        cv2.circle(bgr, c, int(round(d.radius_px)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(bgr, c, color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.putText(bgr, f"{d.label()} {d.confidence:.2f}",
                    (c[0] + int(d.radius_px) + 4, c[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr


# --------------------------------------------------------------------------- #
# Synthetic scene (matches MockBackend) - lets detection be tested without a camera
# --------------------------------------------------------------------------- #
def _disk(img, cx, cy, r, val):
    h, w = img.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val


def synthetic_frame(w=1280, h=1024, seed=0):
    """Two bright pads, four dark inserts each - the MockBackend scene. Also
    returns the ground-truth fiducial centres so the self-test can measure error."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 28, np.uint8)
    truth = []   # (side, x, y)
    for side, cx in (("A", w // 4), ("B", 3 * w // 4)):
        pad_w, pad_h = w // 5, h // 3
        x0, y0 = cx - pad_w // 2, h // 2 - pad_h // 2
        img[y0:y0 + pad_h, x0:x0 + pad_w] = 170
        fx, fy = pad_w // 4, pad_h // 4
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            px, py = cx + sx * fx, h // 2 + sy * fy
            _disk(img, px, py, 22, 18)
            truth.append((side, px, py))
    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, truth


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_table(dets):
    print(f"{'id':>4} {'cx':>9} {'cy':>9} {'r_px':>7} {'circ':>6} "
          f"{'fill':>6} {'axis':>6} {'conc':>6} {'conf':>6}")
    for d in sorted(dets, key=lambda d: (d.side, d.fid_id)):
        print(f"{d.label():>4} {d.cx:9.2f} {d.cy:9.2f} {d.radius_px:7.2f} "
              f"{d.circularity:6.2f} {d.fill_ratio:6.2f} {d.axis_ratio:6.2f} "
              f"{d.concentricity_px:6.2f} {d.confidence:6.2f}")


def main():
    if cv2 is None:
        sys.exit("OpenCV not found. Install with: pip3 install opencv-python")

    ap = argparse.ArgumentParser(description="Fiducial detection")
    ap.add_argument("image", nargs="?", help="PNG frame to detect (omit with --mock)")
    ap.add_argument("--mock", action="store_true",
                    help="Run on a synthetic scene and report centre error")
    ap.add_argument("--save", metavar="PATH", help="Write an annotated overlay PNG")
    args = ap.parse_args()

    if args.mock or not args.image:
        gray, truth = synthetic_frame()
        print(f"[INFO] Synthetic scene {gray.shape[1]}x{gray.shape[0]} "
              f"with {len(truth)} fiducials.")
    else:
        gray = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if gray is None:
            sys.exit(f"Could not read image: {args.image}")
        truth = None
        print(f"[INFO] Loaded {args.image} {gray.shape[1]}x{gray.shape[0]} ({gray.dtype})")

    dets = detect_fiducials(gray)
    print(f"[INFO] Detected {len(dets)} fiducial(s).")
    _print_table(dets)

    if truth is not None:
        # Match each detection to the nearest ground-truth centre and report error.
        errs = []
        for d in dets:
            dx_dy = [(math.hypot(d.cx - tx, d.cy - ty)) for _s, tx, ty in truth]
            errs.append(min(dx_dy))
        if errs:
            print(f"[TEST] max centre error {max(errs):.2f} px, "
                  f"mean {sum(errs) / len(errs):.2f} px over {len(errs)} match(es)")
        ok = len(dets) == len(truth) and (not errs or max(errs) < 2.0)
        print("[TEST] PASS" if ok else "[TEST] FAIL")
        if not ok:
            sys.exit(1)

    if args.save:
        cv2.imwrite(args.save, annotate(gray, dets))
        print(f"[SAVE] annotated overlay -> {args.save}")


if __name__ == "__main__":
    main()
