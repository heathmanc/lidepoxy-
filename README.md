# lidepoxy

Vision-guided epoxy dispensing for a FANUC robot on battery lids. An overhead
Basler camera measures the X / Y / rotation pose of two pallets (Side A and
Side B) from bored-circle fiducials. A Python program reports that offset to an
Allen-Bradley PLC, which validates it and hands it to the FANUC robot so the
robot can run a CAD-defined dispense path through a fixed nominal user frame
plus a position-register offset — no reteaching points per pallet position.

Full concept design: [`docs/fanuc_epoxy_vision_system_design.pdf`](docs/fanuc_epoxy_vision_system_design.pdf)

> **Status: early build.** The camera **capture** utility and the **setup GUI**
> are in place. Detection, calibration, pose solve, and PLC / robot comms are
> the next stages — see [Roadmap](#roadmap).

---

## Contents

| File | What it is |
|------|------------|
| `vision_gui.py` | PyQt5 setup GUI: device selection, exposure/gain sliders + auto toggles with a **live camera-status readout** (actual exposure/gain/fps even in auto), live view, fiducial settings with live Side A / Side B pallet schematics, an optional **live fiducial-detection overlay**, lossless PNG save. Runs with `--mock` (no camera needed). |
| `capture.py` | Standalone Basler capture utility: full-res grab, live view with exposure/gain tuning keys, lossless PNG save, headless batch mode. |
| `detect.py` | Fiducial detector: dark-insert-on-bright-backing → threshold → contour → ellipse fit → sub-pixel centre, with circularity / fill / concentricity quality checks. Auto (find pads → inserts) or ROI-guided. `python3 detect.py --mock` self-tests against a synthetic scene. |
| `fiducial_config.py` | Shared fiducial geometry (diameter / spacing / corner layout) — the single source used by the GUI and the detector. |
| `requirements.txt` | Python pip dependencies. |
| `CLAUDE.md` | Project context for Claude Code. |
| `docs/` | Design document. |

---

## Hardware

- **Camera:** Basler ace 2 a2A5328-15umPRO — mono, USB 3.0, 24.4 MP, Sony IMX540 global shutter
- **Lens:** Basler C12-1624-25M — 16 mm, C-mount
- **Compute:** NVIDIA Jetson Orin AGX, JetPack 6.2.1 (Ubuntu 22.04, Python 3.10)
- **Geometry:** camera ~36 in above the lid plane; useful view ~28 x 14 in

---

## Setup (Jetson Orin AGX / JetPack 6.2.1)

Run these once on the Jetson.

### 1. System packages
```bash
sudo apt update
sudo apt install -y python3-pip python3-pyqt5
```
> PyQt5 is installed via **apt on purpose** — a pip build of PyQt5 can fail on
> aarch64. It is intentionally **not** in `requirements.txt`.

### 2. Basler pylon SDK (ARM64)
Download the **Linux ARM 64-bit** pylon Camera Software Suite (`.deb`) from
Basler, then:
```bash
sudo dpkg -i pylon_*_arm64.deb        # installs to /opt/pylon
```

### 3. Python dependencies
```bash
pip3 install --upgrade pip            # pypylon wheels need pip >= 20.3
pip3 install -r requirements.txt
```

### 4. USB permissions + buffer (do not skip)
```bash
# udev rules so the camera is accessible without root:
sudo /opt/pylon/share/pylon/setup-usb.sh

# Raise the USB filesystem buffer. The default 16 MB is far too small for
# 24 MP USB3 frames and causes torn/incomplete images that look like a flaky
# camera but are actually a buffer problem:
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
```
Make the buffer permanent by adding `usbcore.usbfs_memory_mb=1000` to the kernel
boot arguments (the `APPEND` line in `/boot/extlinux/extlinux.conf`) and
rebooting.

### 5. Verify the camera
```bash
/opt/pylon/bin/pylonviewer            # should see and stream the camera
```

---

## Running

### Setup GUI
```bash
python3 vision_gui.py                 # real camera (falls back to mock if none found)
python3 vision_gui.py --mock          # force synthetic camera — no hardware needed
```
Needs a display (a monitor on the AGX, or remote X / VNC). Pick the device, tune
exposure/gain (with auto toggles), set fiducial geometry (diameter / spacing /
arrangement) and preview both pallets live, and save full-resolution PNGs.

### Capture utility (headless-friendly)
```bash
python3 capture.py                                        # live view + tuning keys
python3 capture.py --no-display --count 10 --interval 2   # unattended grabs
python3 capture.py --exposure 8000 --gain 2 --outdir captures
```
Live-view keys: `s` save · `+` / `-` exposure · `]` / `[` gain · `i` info · `q` quit.

Both tools save **lossless PNG** (never JPEG — these images feed calibration and
metrology).

---

## Roadmap

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Capture | done — `capture.py`, `vision_gui.py` | Image acquisition + setup |
| 2. Fiducial detection | in progress — `detect.py` | 10 mm dark-on-bright: ROI → threshold → contour → ellipse fit → sub-pixel centre; circularity / fill / concentricity quality checks. Wired as a live GUI overlay; passes a synthetic self-test. **Next: tune thresholds against real captured frames.** |
| 3. Grid calibration | todo | pixel → robot-plane mapping at lid height (planar homography, or full intrinsics + distortion if edge residuals are high) |
| 4. Pose solve | todo | best-fit rigid transform (nominal layout → measured); output X/Y/R + residual |
| 5. PLC comms | todo | pylogix → CompactLogix; write payload first, set valid/Seq_ID bit last |
| 6. FANUC offset | todo | PLC → robot over EtherNet/IP; PR[50]/PR[51] vision offset |
| 7. Lid-datum cross-check | todo | detect lid datum, report lid-vs-pallet residual (log-only first) |

See `CLAUDE.md` and the design PDF for the full architecture and the decisions
already locked (0.5 mm total tolerance, CompactLogix controller, lid-datum
future-proofing).

---

## Troubleshooting

- **No Basler devices found** — check the USB3 cable/port and camera power, and
  that `setup-usb.sh` was run. Confirm with `/opt/pylon/bin/pylonviewer`.
- **Torn / incomplete / dropped frames** — raise `usbfs_memory_mb` (step 4).
- **GUI won't start / "cannot connect to X server"** — PyQt needs a display;
  attach a monitor or use remote X / VNC. For headless work use
  `capture.py --no-display`.
- **A camera property errors on connect** — node names can vary slightly by
  firmware; the error names the property and it's a one-line fix in the backend.
- **`pip install` tries to build PyQt5 from source / fails** — don't pip-install
  PyQt5 on the Jetson; use `sudo apt install python3-pyqt5`.

---

## Notes / safety

Safety and motion permission live in the **PLC and robot only** — the Python
vision program is a measurement device that reports pose and validity. The PLC
and robot safety circuits stay independent of this software. Always prove
dispense paths with a dry run before enabling epoxy.
