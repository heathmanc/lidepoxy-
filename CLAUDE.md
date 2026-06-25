# CLAUDE.md

Context for Claude Code working in this repository.

## What this project is

Vision-guided epoxy dispensing for a FANUC robot. An overhead Basler camera sees
two pallets (Side A, Side B), each carrying a pinned battery lid. Bored circular
fiducials on each pallet let the vision software measure pallet pose (X, Y,
rotation). That offset goes to an Allen-Bradley PLC, which validates it and
forwards it to the FANUC robot, which runs a CAD-defined epoxy path through a
fixed nominal user frame plus a position-register offset — so no point reteaching
per pallet position.

Authoritative design: `docs/fanuc_epoxy_vision_system_design.pdf`.

## Architecture & ownership (do not violate)

- **PLC = cell master.** Owns sequencing, safety permissives, stops, clamps,
  robot start, and validation of vision results.
- **Python (this repo) = measurement device only.** Detects fiducials, computes
  Side A/B X/Y/R offset + residual, reports to the PLC. It must **never** own
  safety or robot-motion permission.
- **FANUC = motion + dispense.** Holds nominal user frames, applies the vision
  offset as a position register, runs CAD paths, controls dispense I/O.

Data flow: camera → Python (pose) → PLC (validate) → FANUC (PR offset → path).

## Decisions already locked (design review)

- **Total placement tolerance: 0.5 mm.** This is the *whole* budget (calibration
  + pallet stop + lid pin + robot UF/TCP + epoxy process). Target the "excellent
  controlled system" numbers from the design doc, not "good practical". Build an
  RSS error budget before trusting it.
- **Controller: Allen-Bradley CompactLogix** (tag-based). PLC comms use
  **pylogix**; the design doc's UDT-per-side tag layout applies as written.
- **Lid is reliably pinned**, but a **lid-datum cross-check** is planned to
  future-proof: pallet fiducials drive the offset; the lid datum is also measured
  and the lid-vs-pallet difference is reported as a residual (log-only first).
- **Compute: Jetson Orin AGX, JetPack 6.2.1** (Ubuntu 22.04, Python 3.10).

## Repo layout

- `vision_gui.py` — PyQt5 setup GUI (device, exposure/gain, fiducial geometry +
  live Side A/B schematics, PNG save). Camera abstraction: `CameraBackend` →
  `BaslerBackend` (pypylon) / `MockBackend`. Acquisition runs on a
  `CameraThread` (QThread). **All camera calls happen on that thread** — UI
  changes are queued via `CameraThread.queue()` and applied at the top of the
  grab loop. Preserve this invariant; do not call the camera from the UI thread.
  `--mock` runs the whole GUI with a synthetic camera (no hardware).
- `capture.py` — standalone CLI capture (live + headless), same camera config.
- `docs/` — design document.

## Conventions

- **Python 3.10**; PyQt5 from apt (`python3-pyqt5`, not pip), pypylon, OpenCV,
  numpy.
- **Images are saved as lossless PNG** (8- or 16-bit). Never JPEG — they feed
  calibration and metrology.
- Camera defaults: **mono, full-frame ROI, auto exposure/gain OFF** (locked) for
  stable, repeatable images.
- Fiducials: **10 mm bored circle, dark insert on bright backing**, 4 per side
  near the lid-nest corners, numbered **1 = top-left, 2 = top-right,
  3 = bottom-left, 4 = bottom-right** (matches the design doc and the GUI
  schematic). Frame convention: **right-hand rule**, +Z out of the lid toward the
  camera, +R counter-clockwise in the overhead view. Confirm the rotation sign
  physically (jog a known rotation) before trusting pose output.

## Build roadmap (what to work on next)

1. **Detection** — `detect.py`: crop the expected ROI per fiducial, threshold /
   edge, find contour, fit circle/ellipse, return a sub-pixel center. Validate
   that the dark insert and bright backing ring are concentric (chip / occlusion
   check). Tune against real frames captured with the tools here. Wire as a live
   overlay in the GUI.
2. **Calibration** — pixel → robot-plane transform at lid height. Start with a
   planar homography (`cv2.findHomography`); move to full intrinsics + distortion
   (`cv2.calibrateCamera`) if edge residuals eat the budget. Persist calibration
   with an ID, RMS error, and max validation error.
3. **Pose** — best-fit rigid transform (nominal fiducial layout → measured);
   output X/Y/R offset + per-fiducial residual + confidence.
4. **PLC** — pylogix to CompactLogix. **Write the offset payload first, then set
   the data-valid / Seq_ID bit last** so the PLC never latches a half-written
   offset. Use the UDT-per-side structure from the design doc; keep Seq_ID and
   Calibration_ID so stale data can't be reused.
5. **FANUC offset** — PLC → robot over EtherNet/IP; PR[50] = Side A, PR[51] =
   Side B vision offset.
6. **Lid-datum cross-check** — as above, log-only first; promote to a reject
   limit only once pin behavior is understood.

The fiducial geometry fields in `vision_gui.py` (diameter, spacing, corner
positions) are the **source** of the detection ROIs and the nominal layout —
reuse them, don't duplicate the numbers elsewhere.

## How to run / test without hardware

```bash
python3 vision_gui.py --mock
```
The mock synthesizes a live scene (two bright pads with dark fiducials), so UI
and (later) detection-overlay work can proceed before the camera is wired.

## Safety

Vision software is non-safety. The PLC and robot safety circuits are independent
of this code. Never enable epoxy without a dry-run path verification.
