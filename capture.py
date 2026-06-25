#!/usr/bin/env python3
"""
capture.py - Basler ace 2 a2A5328-15umPRO capture utility
FANUC Epoxy Dispense Vision System - Stage 1 (Capture)

Target platform: Jetson Orin AGX / JetPack 6.2.1 (Ubuntu 22.04, Python 3.10).

Purpose
-------
Acquire FULL-RESOLUTION images for: setup, glare/lighting tuning, calibration
grid capture, and fiducial-detection development. This is the tool you use to
PRODUCE the representative images that unblock the detection stage.

The camera is configured mono, full-frame ROI, with auto exposure/gain OFF
(locked), per the design doc's calibration requirements (section 10.1). Images
are saved as lossless PNG - never JPEG for metrology targets.

Live-view keys
--------------
  s        save full-resolution frame (lossless PNG, timestamped)
  + / =    increase exposure   |   -        decrease exposure   (glare tuning)
  ] / [    increase / decrease gain
  i        print current camera settings
  q / ESC  quit

One-time Jetson setup (do this BEFORE running)
----------------------------------------------
  1. Install pylon SDK (ARM64 .deb from Basler):
       sudo dpkg -i pylon_*_arm64.deb
  2. Install Python deps:
       pip3 install --upgrade pip
       pip3 install pypylon opencv-python numpy
  3. USB device permissions (udev rules, so you don't need root):
       sudo /opt/pylon/share/pylon/setup-usb.sh
  4. usbfs buffer - CRITICAL for 24 MP USB3. The Linux default (16 MB) is far
     too small and causes incomplete/torn frames that look like a flaky camera:
       sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
     Make it permanent by adding  usbcore.usbfs_memory_mb=1000  to the kernel
     boot args (extlinux.conf APPEND line), then reboot.

Notes
-----
- Live view (imshow) needs a display attached and a GUI-enabled OpenCV build.
  Headless? Use --no-display to grab N frames on an interval and save them.
- Node names below are standard pypylon/SFNC for the ace 2 (a2A) family; if a
  property name differs on your exact firmware, the error will name it - the
  fix is a one-word change.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found. Install with: pip3 install opencv-python")

try:
    from pypylon import pylon
except ImportError:
    sys.exit("pypylon not found. Install the pylon SDK (.deb) then: pip3 install pypylon")


USBFS_PATH = "/sys/module/usbcore/parameters/usbfs_memory_mb"


# --------------------------------------------------------------------------- #
# Setup checks
# --------------------------------------------------------------------------- #
def check_usbfs(min_mb: int = 256) -> None:
    """Warn if the USB filesystem buffer is too small for full-res USB3 grabs."""
    try:
        with open(USBFS_PATH) as f:
            mb = int(f.read().strip())
    except FileNotFoundError:
        return  # not Linux / not applicable
    except Exception:
        return
    if mb < min_mb:
        print(f"[WARN] usbfs_memory_mb = {mb} MB - likely too small for 24 MP USB3.")
        print(f"       Torn/incomplete frames? Raise it:")
        print(f"       sudo sh -c 'echo 1000 > {USBFS_PATH}'")
    else:
        print(f"[OK]   usbfs_memory_mb = {mb} MB")


# --------------------------------------------------------------------------- #
# Camera selection / configuration
# --------------------------------------------------------------------------- #
def open_camera(serial: str | None):
    """Open the requested Basler device (by serial), or the first one found."""
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    if not devices:
        sys.exit("No Basler devices found. Check USB connection, power, and udev rules.")

    print(f"[INFO] Found {len(devices)} Basler device(s):")
    for d in devices:
        print(f"         {d.GetModelName()}  SN={d.GetSerialNumber()}")

    chosen = None
    if serial:
        for d in devices:
            if d.GetSerialNumber() == serial:
                chosen = d
                break
        if chosen is None:
            sys.exit(f"Serial {serial} not found among connected devices.")
    else:
        chosen = devices[0]

    cam = pylon.InstantCamera(tlf.CreateDevice(chosen))
    cam.Open()
    print(f"[INFO] Opened {cam.GetDeviceInfo().GetModelName()} "
          f"SN={cam.GetDeviceInfo().GetSerialNumber()}")
    return cam


def configure_camera(cam, pixel_format: str, exposure_us: float, gain_db: float) -> None:
    """Full-frame mono ROI with auto exposure/gain locked off."""
    # Maximize ROI: zero the offsets first, then push width/height to max.
    for setter in (("OffsetX", 0), ("OffsetY", 0)):
        try:
            getattr(cam, setter[0]).Value = setter[1]
        except Exception:
            pass
    cam.Width.Value = cam.Width.Max
    cam.Height.Value = cam.Height.Max

    cam.PixelFormat.Value = pixel_format

    # Lock exposure & gain - stable, repeatable images are required for
    # calibration and consistent fiducial detection.
    for auto_node in ("ExposureAuto", "GainAuto"):
        try:
            getattr(cam, auto_node).Value = "Off"
        except Exception:
            pass
    set_exposure(cam, exposure_us)
    set_gain(cam, gain_db)

    print_settings(cam)


def _clamp(node, value):
    return max(node.Min, min(node.Max, value))


def set_exposure(cam, us: float) -> float:
    val = _clamp(cam.ExposureTime, float(us))
    cam.ExposureTime.Value = val
    return val


def set_gain(cam, db: float) -> float:
    val = _clamp(cam.Gain, float(db))
    cam.Gain.Value = val
    return val


def print_settings(cam) -> None:
    w, h = cam.Width.Value, cam.Height.Value
    pf = cam.PixelFormat.Value
    exp = cam.ExposureTime.Value
    gain = cam.Gain.Value
    line = f"[CAM]  {w}x{h}  {pf}  exposure={exp:.0f} us  gain={gain:.1f} dB"
    # Resulting frame rate node name varies; show it if available.
    for node in ("ResultingFrameRate", "BslResultingAcquisitionFrameRate",
                 "AcquisitionFrameRate"):
        try:
            line += f"  ~{getattr(cam, node).Value:.1f} fps"
            break
        except Exception:
            continue
    print(line)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_display(img: np.ndarray, max_w: int = 1280) -> np.ndarray:
    """Downscale + 8-bit for on-screen view (24 MP is too big to show natively)."""
    disp = img
    if disp.dtype == np.uint16:
        # Mono12 packed into uint16 -> shift down to 8-bit for display only.
        disp = (disp >> 4).astype(np.uint8)
    h, w = disp.shape[:2]
    if w > max_w:
        s = max_w / w
        disp = cv2.resize(disp, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    return disp


def save_frame(img: np.ndarray, outdir: str) -> str:
    """Save the FULL-resolution array as lossless PNG (8- or 16-bit as captured)."""
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(outdir, f"frame_{ts}.png")
    cv2.imwrite(path, img)  # cv2 writes 16-bit PNG for uint16, 8-bit for uint8
    print(f"[SAVE] {path}  ({img.shape[1]}x{img.shape[0]}, {img.dtype})")
    return path


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_live(cam, outdir: str) -> None:
    """Interactive live view with save + exposure/gain tuning."""
    win = "Basler capture - s:save  +/-:exposure  ]/[:gain  i:info  q:quit"
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("[INFO] Live view started.")
    try:
        while cam.IsGrabbing():
            grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                print(f"[WARN] Grab failed: {grab.ErrorCode} {grab.ErrorDescription}")
                grab.Release()
                continue

            img = grab.Array
            disp = to_display(img)
            label = (f"{img.shape[1]}x{img.shape[0]}  "
                     f"exp={cam.ExposureTime.Value:.0f}us  "
                     f"gain={cam.Gain.Value:.1f}dB")
            cv2.putText(disp, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            grab.Release()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):
                save_frame(img, outdir)
            elif key in (ord('+'), ord('=')):
                set_exposure(cam, cam.ExposureTime.Value * 1.25); print_settings(cam)
            elif key == ord('-'):
                set_exposure(cam, cam.ExposureTime.Value * 0.8); print_settings(cam)
            elif key == ord(']'):
                set_gain(cam, cam.Gain.Value + 1.0); print_settings(cam)
            elif key == ord('['):
                set_gain(cam, cam.Gain.Value - 1.0); print_settings(cam)
            elif key == ord('i'):
                print_settings(cam)
    finally:
        cam.StopGrabbing()
        cv2.destroyAllWindows()


def run_headless(cam, outdir: str, count: int, interval_s: float) -> None:
    """No display: grab `count` frames `interval_s` apart and save each."""
    import time
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print(f"[INFO] Headless capture: {count} frame(s), {interval_s}s apart.")
    saved = 0
    try:
        while saved < count and cam.IsGrabbing():
            grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                save_frame(grab.Array, outdir)
                saved += 1
                grab.Release()
                if saved < count:
                    time.sleep(interval_s)
            else:
                print(f"[WARN] Grab failed: {grab.ErrorCode} {grab.ErrorDescription}")
                grab.Release()
    finally:
        cam.StopGrabbing()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Basler a2A5328 capture utility")
    ap.add_argument("--serial", default=None, help="Camera serial (default: first found)")
    ap.add_argument("--pixel-format", default="Mono8", choices=["Mono8", "Mono12"],
                    help="Mono8 (default) or Mono12 for more dynamic range")
    ap.add_argument("--exposure", type=float, default=10000.0,
                    help="Exposure time in microseconds (default 10000)")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="Gain in dB (default 0)")
    ap.add_argument("--outdir", default="captures",
                    help="Output folder for saved PNGs (default: ./captures)")
    ap.add_argument("--no-display", action="store_true",
                    help="Headless: grab --count frames and save, no live window")
    ap.add_argument("--count", type=int, default=1,
                    help="(headless) number of frames to grab")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="(headless) seconds between frames")
    args = ap.parse_args()

    check_usbfs()

    cam = open_camera(args.serial)
    try:
        configure_camera(cam, args.pixel_format, args.exposure, args.gain)
        if args.no_display:
            run_headless(cam, args.outdir, args.count, args.interval)
        else:
            run_live(cam, args.outdir)
    finally:
        if cam.IsOpen():
            cam.Close()
        print("[INFO] Camera closed.")


if __name__ == "__main__":
    main()
