#!/usr/bin/env python3
"""
fiducial_config.py - Shared fiducial geometry model
FANUC Epoxy Dispense Vision System

The single source of truth for pallet fiducial geometry: diameter, X/Y spacing,
arrangement, and the resulting nominal corner layout. The setup GUI edits these
values, the detector uses them to size search ROIs and as the nominal layout for
the pose solve, and any future calibration / pose code reuses the same numbers.

Kept deliberately free of PyQt / OpenCV so it can be imported anywhere (headless
detection, tests) without dragging in a GUI toolkit.

Numbering matches the design doc and the GUI schematic:
    1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right
Frame: right-hand rule, +X right, +Y up in the overhead view.
"""

import math
from dataclasses import dataclass


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
