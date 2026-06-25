#!/usr/bin/env python3
"""
vision_gui.py - Setup & capture GUI
FANUC Epoxy Dispense Vision System

A PyQt5 desktop GUI for the Basler ace 2 a2A5328-15umPRO on a Jetson Orin AGX
(JetPack 6.2.1 / Ubuntu 22.04 / Python 3.10). Provides:

  - Camera device selection (lists connected Basler cameras; mock always offered)
  - Live view via a background grab thread (UI never blocks)
  - Exposure & gain sliders, with auto-exposure / auto-gain toggles
  - Pixel format (Mono8 / Mono12)
  - Fiducial marker settings (diameter, X/Y spacing, arrangement) that drive a
    LIVE schematic of Side A and Side B pallets, in the cell +X/+Y frame
  - Save full-resolution frame as lossless PNG

This is a foundation for the production HMI - extend the right-hand panel and
the pallet views as detection / calibration stages come online.

------------------------------------------------------------------------------
Install (JetPack 6 / Ubuntu 22.04)
------------------------------------------------------------------------------
  sudo apt update
  sudo apt install python3-pyqt5          # reliable Qt on aarch64 via apt
  pip3 install pypylon opencv-python numpy # (already done for the capture stage)

  Camera setup reminders (same as capture.py):
    sudo /opt/pylon/share/pylon/setup-usb.sh
    sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'

------------------------------------------------------------------------------
Run
------------------------------------------------------------------------------
  python3 vision_gui.py            # real camera (falls back to mock if none)
  python3 vision_gui.py --mock     # force synthetic camera, no hardware needed

Needs a display attached (or remote X / VNC). Headless? Use capture.py instead.

NOTE: The fiducial settings here are for laying out / recording your pallet
geometry and previewing it. They are not yet wired to detection - that comes in
the next stage, where these same numbers define the search ROIs and the nominal
fiducial layout for the pose solve.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    import cv2  # only needed for saving PNGs
except ImportError:
    cv2 = None

from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QMutex, QMutexLocker,
                          QPointF, QRectF, QTimer)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QCheckBox, QComboBox,
    QPushButton, QDoubleSpinBox, QGroupBox, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFileDialog, QSizePolicy
)

try:
    from pypylon import pylon
    HAVE_PYLON = True
except ImportError:
    HAVE_PYLON = False


# ===========================================================================
#  Fiducial configuration model
# ===========================================================================
@dataclass
class FiducialConfig:
    diameter_mm: float = 10.0      # bored-hole diameter (design doc: 10 mm)
    spacing_x_mm: float = 220.0    # left<->right distance (fiducial 1 <-> 2)
    spacing_y_mm: float = 150.0    # top<->bottom distance (fiducial 1 <-> 3)
    arrangement: str = "Rectangle (4-corner)"

    def corners_mm(self):
        """Fiducial centres in mm, origin-centred. Numbering matches the doc."""
        sx, sy = self.spacing_x_mm / 2.0, self.spacing_y_mm / 2.0
        return [
            (-sx,  sy),   # 1 top-left
            ( sx,  sy),   # 2 top-right
            (-sx, -sy),   # 3 bottom-left
            ( sx, -sy),   # 4 bottom-right
        ]

    def diagonal_mm(self):
        return math.hypot(self.spacing_x_mm, self.spacing_y_mm)


# ===========================================================================
#  Camera backends (real Basler + synthetic mock), one interface
# ===========================================================================
class CameraBackend:
    name = "base"
    def open(self): ...
    def grab(self, timeout_ms=2000): return None
    def set_pixel_format(self, fmt): pass
    def set_auto_exposure(self, on): pass
    def set_exposure(self, us): pass
    def set_auto_gain(self, on): pass
    def set_gain(self, db): pass
    def close(self): pass


def _draw_disk(img, cx, cy, r, val):
    """Filled circle in a 2D uint8 array (no OpenCV dependency)."""
    h, w = img.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[y0:y1, x0:x1][mask] = val


class MockBackend(CameraBackend):
    """Synthetic camera: two bright pads, each with 4 dark fiducials + noise.
    Brightness follows the exposure setting so the sliders visibly do something."""
    name = "Mock camera (synthetic)"

    def __init__(self, w=1280, h=1024):
        self.w, self.h = w, h
        self._exp = 10000.0
        self._gain = 0.0
        self._auto_exp = False

    def open(self):
        pass

    def set_auto_exposure(self, on): self._auto_exp = bool(on)
    def set_exposure(self, us): self._exp = float(us)
    def set_auto_gain(self, on): pass
    def set_gain(self, db): self._gain = float(db)
    def set_pixel_format(self, fmt): pass

    def grab(self, timeout_ms=2000):
        time.sleep(0.03)  # ~30 fps synthetic
        e = max(self._exp, 50.0)
        bright = 170 if self._auto_exp else int(np.clip(
            40 + 60 * math.log10(e / 50.0), 30, 235))
        img = np.full((self.h, self.w), 28, np.uint8)
        img = np.clip(img.astype(np.int16) + int(self._gain * 2),
                      0, 255).astype(np.uint8)
        for cx in (self.w // 4, 3 * self.w // 4):
            pad_w, pad_h = self.w // 5, self.h // 3
            x0, y0 = cx - pad_w // 2, self.h // 2 - pad_h // 2
            img[y0:y0 + pad_h, x0:x0 + pad_w] = bright
            fx, fy = pad_w // 4, pad_h // 4
            for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                _draw_disk(img, cx + sx * fx, self.h // 2 + sy * fy, 22, 18)
        noise = np.random.normal(0, 4, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class BaslerBackend(CameraBackend):
    """pypylon wrapper. All camera calls happen on the grab thread."""
    def __init__(self, device_info, pixel_format="Mono8"):
        self.device_info = device_info
        self.pixel_format = pixel_format
        self.cam = None
        self.name = f"{device_info.GetModelName()} ({device_info.GetSerialNumber()})"

    def open(self):
        tlf = pylon.TlFactory.GetInstance()
        self.cam = pylon.InstantCamera(tlf.CreateDevice(self.device_info))
        self.cam.Open()
        for off in ("OffsetX", "OffsetY"):
            try:
                getattr(self.cam, off).Value = 0
            except Exception:
                pass
        self.cam.Width.Value = self.cam.Width.Max
        self.cam.Height.Value = self.cam.Height.Max
        try:
            self.cam.PixelFormat.Value = self.pixel_format
        except Exception:
            pass
        for a in ("ExposureAuto", "GainAuto"):
            try:
                getattr(self.cam, a).Value = "Off"
            except Exception:
                pass
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def set_pixel_format(self, fmt):
        was = self.cam.IsGrabbing()
        if was:
            self.cam.StopGrabbing()
        try:
            self.cam.PixelFormat.Value = fmt
        except Exception:
            pass
        if was:
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def set_auto_exposure(self, on):
        try:
            self.cam.ExposureAuto.Value = "Continuous" if on else "Off"
        except Exception:
            pass

    def set_exposure(self, us):
        try:
            lo, hi = self.cam.ExposureTime.Min, self.cam.ExposureTime.Max
            self.cam.ExposureTime.Value = float(np.clip(us, lo, hi))
        except Exception:
            pass

    def set_auto_gain(self, on):
        try:
            self.cam.GainAuto.Value = "Continuous" if on else "Off"
        except Exception:
            pass

    def set_gain(self, db):
        try:
            lo, hi = self.cam.Gain.Min, self.cam.Gain.Max
            self.cam.Gain.Value = float(np.clip(db, lo, hi))
        except Exception:
            pass

    def grab(self, timeout_ms=2000):
        res = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        if res is None:
            return None
        try:
            return res.Array.copy() if res.GrabSucceeded() else None
        finally:
            res.Release()

    def close(self):
        try:
            if self.cam:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                if self.cam.IsOpen():
                    self.cam.Close()
        except Exception:
            pass


# ===========================================================================
#  Grab thread - keeps acquisition off the UI thread
# ===========================================================================
class CameraThread(QThread):
    frameReady = pyqtSignal(object)
    error = pyqtSignal(str)

    # UI -> thread setting changes are queued and applied on the grab thread,
    # so the camera object is only ever touched from one thread.
    _APPLY_ORDER = ["pixel_format", "auto_exposure", "exposure",
                    "auto_gain", "gain"]

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._running = False
        self._mutex = QMutex()
        self._pending = {}

    def queue(self, key, value):
        with QMutexLocker(self._mutex):
            self._pending[key] = value

    def _apply_pending(self):
        with QMutexLocker(self._mutex):
            pending, self._pending = self._pending, {}
        for k in self._APPLY_ORDER:
            if k in pending:
                try:
                    getattr(self.backend, "set_" + k)(pending[k])
                except Exception as e:
                    self.error.emit(f"{k}: {e}")

    def run(self):
        self._running = True
        try:
            self.backend.open()
        except Exception as e:
            self.error.emit(f"Open failed: {e}")
            return
        while self._running:
            self._apply_pending()
            try:
                frame = self.backend.grab(2000)
            except Exception as e:
                self.error.emit(f"Grab error: {e}")
                break
            if frame is not None:
                self.frameReady.emit(frame)
            else:
                self.msleep(5)
        self.backend.close()

    def stop(self):
        self._running = False
        self.wait(3000)


# ===========================================================================
#  Pallet schematic widget (one per side)
# ===========================================================================
class PalletView(QWidget):
    def __init__(self, side_label, cfg: FiducialConfig):
        super().__init__()
        self.side = side_label
        self.cfg = cfg
        self.setMinimumSize(300, 230)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setConfig(self, cfg):
        self.cfg = cfg
        self.update()

    def paintEvent(self, _ev):
        cfg = self.cfg
        if cfg.spacing_x_mm <= 0 or cfg.spacing_y_mm <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#0e141c"))

        pad, title_h, footer_h = 16, 22, 28
        ax0, ay0 = pad, pad + title_h
        aw, ah = W - 2 * pad, H - (pad + title_h) - footer_h
        cx_px, cy_px = ax0 + aw / 2.0, ay0 + ah / 2.0

        ext_x, ext_y = cfg.spacing_x_mm * 1.35, cfg.spacing_y_mm * 1.35
        scale = min(aw / ext_x, ah / ext_y) * 0.92

        def to_px(xmm, ymm):
            return QPointF(cx_px + xmm * scale, cy_px - ymm * scale)  # +Y up

        # pallet body
        pal_w, pal_h = cfg.spacing_x_mm * 1.30 * scale, cfg.spacing_y_mm * 1.30 * scale
        p.setPen(QPen(QColor("#3a6ea5"), 2))
        p.setBrush(QColor("#16202c"))
        p.drawRoundedRect(QRectF(cx_px - pal_w / 2, cy_px - pal_h / 2, pal_w, pal_h), 10, 10)

        # lid nest
        lid_w, lid_h = cfg.spacing_x_mm * 0.60 * scale, cfg.spacing_y_mm * 0.60 * scale
        lid_rect = QRectF(cx_px - lid_w / 2, cy_px - lid_h / 2, lid_w, lid_h)
        p.setPen(QPen(QColor("#8aa0b4"), 1, Qt.DashLine))
        p.setBrush(QColor("#1d2a38"))
        p.drawRoundedRect(lid_rect, 6, 6)
        p.setPen(QColor("#7fa8d0"))
        f = QFont(); f.setPointSize(8); p.setFont(f)
        p.drawText(lid_rect, Qt.AlignCenter | Qt.TextWordWrap, "LID\nCAD FRAME")

        # fiducials: bright backing ring + dark insert + label
        fid_r = max(4.0, (cfg.diameter_mm / 2.0) * scale)
        for i, (xmm, ymm) in enumerate(cfg.corners_mm(), start=1):
            c = to_px(xmm, ymm)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#e8e8e8"))
            p.drawEllipse(c, fid_r * 1.7, fid_r * 1.7)
            p.setBrush(QColor("#101010"))
            p.drawEllipse(c, fid_r, fid_r)
            p.setPen(QColor("#cfe3f7"))
            p.drawText(QRectF(c.x() - fid_r * 1.7, c.y() - fid_r * 1.7 - 14,
                              fid_r * 3.4, 12),
                       Qt.AlignCenter, f"{self.side}{i}")

        # cell axes (+X right red, +Y up green)
        ox, oy = ax0 + 18, ay0 + ah - 12
        p.setPen(QPen(QColor("#e05a4d"), 2)); p.drawLine(int(ox), int(oy), int(ox + 26), int(oy))
        p.drawText(int(ox + 28), int(oy + 4), "+X")
        p.setPen(QPen(QColor("#4db26a"), 2)); p.drawLine(int(ox), int(oy), int(ox), int(oy - 26))
        p.drawText(int(ox - 6), int(oy - 30), "+Y")

        # title
        p.setPen(QColor("#dfe9f5"))
        ft = QFont(); ft.setPointSize(11); ft.setBold(True); p.setFont(ft)
        p.drawText(QRectF(0, 4, W, title_h), Qt.AlignCenter, f"SIDE {self.side}")

        # footer
        p.setPen(QColor("#90a4b8"))
        ff = QFont(); ff.setPointSize(8); p.setFont(ff)
        p.drawText(QRectF(0, H - footer_h + 2, W, footer_h - 2), Qt.AlignCenter,
                   f"dia {cfg.diameter_mm:.1f} mm   "
                   f"spacing {cfg.spacing_x_mm:.0f}x{cfg.spacing_y_mm:.0f} mm   "
                   f"diag {cfg.diagonal_mm():.0f} mm")
        p.end()


# ===========================================================================
#  Main window
# ===========================================================================
class MainWindow(QMainWindow):
    def __init__(self, force_mock=False):
        super().__init__()
        self.setWindowTitle("FANUC Epoxy Vision - Capture & Setup")
        self.resize(1300, 840)
        self.cfg = FiducialConfig()
        self.cam_thread = None
        self.last_frame = None
        self.outdir = os.path.abspath("captures")
        self.force_mock = force_mock
        self._exp_guard = False
        self._gain_guard = False
        self._build_ui()
        self._populate_devices()

    # ---- UI construction ----------------------------------------------- #
    def _build_ui(self):
        # Left: live view + two pallet schematics
        left = QWidget()
        lv = QVBoxLayout(left)
        self.view = QLabel("Camera disconnected - pick a device and press Connect")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(
            "background:#0b0f14; color:#7fa8d0; border:1px solid #233; font-size:14px;")
        self.view.setMinimumSize(640, 420)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lv.addWidget(self.view, 1)

        pallets = QHBoxLayout()
        self.pallet_a = PalletView("A", self.cfg)
        self.pallet_b = PalletView("B", self.cfg)
        pallets.addWidget(self.pallet_a)
        pallets.addWidget(self.pallet_b)
        lv.addLayout(pallets)

        # Right: control panel
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.addWidget(self._camera_group())
        pv.addWidget(self._exposure_group())
        pv.addWidget(self._gain_group())
        pv.addWidget(self._fiducial_group())
        pv.addWidget(self._capture_group())
        pv.addStretch(1)
        panel.setFixedWidth(340)

        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(left, 1)
        root.addWidget(panel)
        self.setCentralWidget(central)

        self.status = QLabel("Ready.")
        self.statusBar().addWidget(self.status)

    def _camera_group(self):
        box = QGroupBox("Camera")
        l = QVBoxLayout(box)
        self.dev_combo = QComboBox()
        l.addWidget(QLabel("Device"))
        l.addWidget(self.dev_combo)
        row = QHBoxLayout()
        b_con = QPushButton("Connect"); b_con.clicked.connect(self._connect)
        b_dis = QPushButton("Disconnect"); b_dis.clicked.connect(self._disconnect)
        row.addWidget(b_con); row.addWidget(b_dis)
        l.addLayout(row)
        l.addWidget(QLabel("Pixel format"))
        self.pf_combo = QComboBox(); self.pf_combo.addItems(["Mono8", "Mono12"])
        self.pf_combo.currentTextChanged.connect(self._on_pf)
        l.addWidget(self.pf_combo)
        return box

    def _exposure_group(self):
        box = QGroupBox("Exposure")
        l = QVBoxLayout(box)
        self.auto_exp = QCheckBox("Auto exposure")
        self.auto_exp.toggled.connect(self._on_auto_exp)
        l.addWidget(self.auto_exp)
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(50, 50000); self.exp_slider.setValue(10000)
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(50, 50000); self.exp_spin.setDecimals(0)
        self.exp_spin.setSuffix(" us"); self.exp_spin.setValue(10000)
        self.exp_slider.valueChanged.connect(self._on_exp_slider)
        self.exp_spin.valueChanged.connect(self._on_exp_spin)
        l.addWidget(self.exp_slider)
        l.addWidget(self.exp_spin)
        return box

    def _gain_group(self):
        box = QGroupBox("Gain")
        l = QVBoxLayout(box)
        self.auto_gain = QCheckBox("Auto gain")
        self.auto_gain.toggled.connect(self._on_auto_gain)
        l.addWidget(self.auto_gain)
        self.gain_slider = QSlider(Qt.Horizontal)   # 0..360 -> 0.0..36.0 dB
        self.gain_slider.setRange(0, 360); self.gain_slider.setValue(0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 36.0); self.gain_spin.setDecimals(1)
        self.gain_spin.setSingleStep(0.5); self.gain_spin.setSuffix(" dB")
        self.gain_slider.valueChanged.connect(self._on_gain_slider)
        self.gain_spin.valueChanged.connect(self._on_gain_spin)
        l.addWidget(self.gain_slider)
        l.addWidget(self.gain_spin)
        return box

    def _fiducial_group(self):
        box = QGroupBox("Fiducial markers")
        l = QFormLayout(box)
        self.dia_spin = QDoubleSpinBox()
        self.dia_spin.setRange(1, 50); self.dia_spin.setDecimals(1)
        self.dia_spin.setValue(self.cfg.diameter_mm); self.dia_spin.setSuffix(" mm")
        self.sx_spin = QDoubleSpinBox()
        self.sx_spin.setRange(10, 800); self.sx_spin.setDecimals(0)
        self.sx_spin.setValue(self.cfg.spacing_x_mm); self.sx_spin.setSuffix(" mm")
        self.sy_spin = QDoubleSpinBox()
        self.sy_spin.setRange(10, 800); self.sy_spin.setDecimals(0)
        self.sy_spin.setValue(self.cfg.spacing_y_mm); self.sy_spin.setSuffix(" mm")
        self.arr_combo = QComboBox()
        self.arr_combo.addItems(["Rectangle (4-corner)"])
        for w in (self.dia_spin, self.sx_spin, self.sy_spin):
            w.valueChanged.connect(self._on_fid_changed)
        self.arr_combo.currentTextChanged.connect(self._on_fid_changed)
        l.addRow("Diameter", self.dia_spin)
        l.addRow("Spacing X (1<->2)", self.sx_spin)
        l.addRow("Spacing Y (1<->3)", self.sy_spin)
        l.addRow("Arrangement", self.arr_combo)
        return box

    def _capture_group(self):
        box = QGroupBox("Capture")
        l = QVBoxLayout(box)
        b_save = QPushButton("Save frame (PNG)")
        b_save.clicked.connect(self._save)
        l.addWidget(b_save)
        row = QHBoxLayout()
        self.outdir_label = QLabel(self.outdir); self.outdir_label.setWordWrap(True)
        b_dir = QPushButton("Folder...")
        b_dir.clicked.connect(self._browse)
        row.addWidget(self.outdir_label, 1); row.addWidget(b_dir)
        l.addLayout(row)
        return box

    # ---- device handling ----------------------------------------------- #
    def _populate_devices(self):
        self.dev_combo.clear()
        self.devices = []
        if HAVE_PYLON and not self.force_mock:
            try:
                for d in pylon.TlFactory.GetInstance().EnumerateDevices():
                    self.devices.append(d)
                    self.dev_combo.addItem(
                        f"{d.GetModelName()} ({d.GetSerialNumber()})")
            except Exception as e:
                self.status.setText(f"Enumerate error: {e}")
        self.dev_combo.addItem(MockBackend.name)
        if not self.devices:
            self.dev_combo.setCurrentText(MockBackend.name)

    def _connect(self):
        self._disconnect()
        idx = self.dev_combo.currentIndex()
        text = self.dev_combo.currentText()
        if text == MockBackend.name or idx >= len(self.devices):
            backend = MockBackend()
        else:
            backend = BaslerBackend(self.devices[idx], self.pf_combo.currentText())
        self.cam_thread = CameraThread(backend)
        self.cam_thread.frameReady.connect(self._on_frame)
        self.cam_thread.error.connect(self._on_cam_error)
        self.cam_thread.start()
        QTimer.singleShot(300, self._sync_controls_to_camera)
        self.status.setText(f"Connected: {backend.name}")

    def _disconnect(self):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
            self.view.setText("Camera disconnected")
            self.status.setText("Disconnected")

    def _sync_controls_to_camera(self):
        t = self.cam_thread
        if not t:
            return
        t.queue("pixel_format", self.pf_combo.currentText())
        t.queue("auto_exposure", self.auto_exp.isChecked())
        t.queue("exposure", float(self.exp_spin.value()))
        t.queue("auto_gain", self.auto_gain.isChecked())
        t.queue("gain", float(self.gain_spin.value()))

    # ---- frame display -------------------------------------------------- #
    def _on_frame(self, frame):
        self.last_frame = frame
        disp = (frame >> 4).astype(np.uint8) if frame.dtype == np.uint16 else frame
        disp = np.ascontiguousarray(disp)
        h, w = disp.shape[:2]
        if disp.ndim == 2:
            qimg = QImage(disp.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            rgb = np.ascontiguousarray(disp[:, :, ::-1])
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_cam_error(self, msg):
        self.status.setText(f"Camera: {msg}")

    # ---- control handlers ---------------------------------------------- #
    def _on_pf(self, text):
        if self.cam_thread:
            self.cam_thread.queue("pixel_format", text)

    def _on_auto_exp(self, on):
        self.exp_slider.setEnabled(not on)
        self.exp_spin.setEnabled(not on)
        if self.cam_thread:
            self.cam_thread.queue("auto_exposure", on)

    def _on_exp_slider(self, v):
        if self._exp_guard:
            return
        self._exp_guard = True
        self.exp_spin.setValue(float(v))
        self._exp_guard = False
        if self.cam_thread:
            self.cam_thread.queue("exposure", float(v))

    def _on_exp_spin(self, v):
        if self._exp_guard:
            return
        self._exp_guard = True
        self.exp_slider.setValue(int(v))
        self._exp_guard = False
        if self.cam_thread:
            self.cam_thread.queue("exposure", float(v))

    def _on_auto_gain(self, on):
        self.gain_slider.setEnabled(not on)
        self.gain_spin.setEnabled(not on)
        if self.cam_thread:
            self.cam_thread.queue("auto_gain", on)

    def _on_gain_slider(self, v):
        if self._gain_guard:
            return
        self._gain_guard = True
        self.gain_spin.setValue(v / 10.0)
        self._gain_guard = False
        if self.cam_thread:
            self.cam_thread.queue("gain", v / 10.0)

    def _on_gain_spin(self, v):
        if self._gain_guard:
            return
        self._gain_guard = True
        self.gain_slider.setValue(int(round(v * 10)))
        self._gain_guard = False
        if self.cam_thread:
            self.cam_thread.queue("gain", float(v))

    def _on_fid_changed(self, *_):
        self.cfg.diameter_mm = self.dia_spin.value()
        self.cfg.spacing_x_mm = self.sx_spin.value()
        self.cfg.spacing_y_mm = self.sy_spin.value()
        self.cfg.arrangement = self.arr_combo.currentText()
        self.pallet_a.setConfig(self.cfg)
        self.pallet_b.setConfig(self.cfg)

    # ---- capture -------------------------------------------------------- #
    def _save(self):
        if self.last_frame is None:
            self.status.setText("No frame to save yet.")
            return
        if cv2 is None:
            self.status.setText("Saving needs OpenCV: pip3 install opencv-python")
            return
        os.makedirs(self.outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(self.outdir, f"frame_{ts}.png")
        cv2.imwrite(path, self.last_frame)
        self.status.setText(f"Saved {path}  "
                            f"({self.last_frame.shape[1]}x{self.last_frame.shape[0]})")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.outdir)
        if d:
            self.outdir = d
            self.outdir_label.setText(d)

    def closeEvent(self, e):
        self._disconnect()
        e.accept()


# ===========================================================================
#  Entry point
# ===========================================================================
DARK_QSS = """
QWidget { background:#10161e; color:#d6e2f0; font-size:12px; }
QGroupBox { border:1px solid #29384a; border-radius:6px; margin-top:9px; padding-top:6px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; color:#7fa8d0; }
QPushButton { background:#1c2a3a; border:1px solid #2f4a66; border-radius:4px; padding:5px 10px; }
QPushButton:hover { background:#243span; }
QPushButton:pressed { background:#152030; }
QComboBox, QDoubleSpinBox { background:#16202c; border:1px solid #2f4a66; border-radius:4px; padding:3px; }
QSlider::groove:horizontal { height:6px; background:#1d2a38; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#4d8fd0; border-radius:7px; margin:-5px 0; }
QCheckBox { spacing:6px; }
QStatusBar { background:#0d131b; color:#9fb4c8; }
"""


def main():
    ap = argparse.ArgumentParser(description="FANUC epoxy vision setup GUI")
    ap.add_argument("--mock", action="store_true",
                    help="Force the synthetic camera (no Basler hardware needed)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS.replace("#243span", "#243648"))  # guard against typo
    win = MainWindow(force_mock=args.mock)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
