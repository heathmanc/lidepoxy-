#!/usr/bin/env python3
"""
plc.py - PLC comms (vision -> Allen-Bradley CompactLogix)
FANUC Epoxy Dispense Vision System - Stage 5 (PLC comms)

Report the measured Side A / Side B offset to the PLC over EtherNet/IP using
pylogix. The PLC is the cell master: it VALIDATES the offset and decides whether
the robot may run. This module only *reports* a measurement - it owns no safety
and no motion permission (see CLAUDE.md / the design doc).

Write ordering (the important part)
-----------------------------------
The PLC must never latch a half-written offset, so for each side we:

  1. clear Data_Valid       (PLC ignores the side while we write)
  2. write the payload       (X, Y, R, residual, confidence, fiducials, Cal_ID)
  3. stamp Seq_ID            (monotonic - lets the PLC detect a fresh result and
                              reject a stale / re-read one)
  4. set Data_Valid LAST     (the latch: payload is now complete and consistent)

Carrying Seq_ID and Calibration_ID means the PLC can reject a stale offset or one
computed against the wrong calibration.

Tag layout
----------
UDT-per-side, as in the design doc. The base tag and member names are
configurable here (DEFAULT_SIDE_BASE / DEFAULT_MEMBERS) so they can be matched to
the actual controller UDT without touching logic. Full tag = "<base>.<member>",
e.g. "Vision_SideA.Offset_X".

Usage
-----
    from plc import OffsetPublisher, PylogixBackend
    pub = OffsetPublisher(PylogixBackend("192.168.1.10"))
    pub.publish("A", pose_result_a, calibration_id="cal_20260625_101500")
    pub.publish("B", pose_result_b, calibration_id="cal_20260625_101500")

    python3 plc.py --mock      # in-memory self-test (no PLC / no pylogix needed)
"""

import argparse
import sys
from dataclasses import dataclass

try:
    from pylogix import PLC
    HAVE_PYLOGIX = True
except ImportError:
    HAVE_PYLOGIX = False


# UDT-per-side base tags and member names. Override to match the controller UDT.
DEFAULT_SIDE_BASE = {"A": "Vision_SideA", "B": "Vision_SideB"}
DEFAULT_MEMBERS = {
    "x": "Offset_X",                 # REAL, mm
    "y": "Offset_Y",                 # REAL, mm
    "r": "Offset_R",                 # REAL, deg
    "residual": "Residual_mm",       # REAL, mm (worst per-fiducial residual)
    "confidence": "Confidence",      # REAL, 0..1
    "fiducials": "Fiducials_Found",  # DINT
    "calibration_id": "Calibration_ID",  # STRING
    "seq_id": "Seq_ID",              # DINT, monotonic
    "valid": "Data_Valid",           # BOOL, set last
}


def tag(base, member):
    return f"{base}.{member}"


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
@dataclass
class SideOffset:
    side: str            # "A" / "B"
    x_mm: float
    y_mm: float
    r_deg: float
    residual_mm: float
    confidence: float
    fiducials_found: int
    calibration_id: str
    valid: bool = True   # PLC-visible validity (cleared if the measure is no good)


def from_pose(side, pose_result, calibration_id, min_confidence=0.0):
    """Build a SideOffset from a pose.PoseResult.

    `valid` is set only when the solve succeeded and cleared the confidence gate -
    otherwise the offset is still written (for logging/visibility) but Data_Valid
    stays False so the PLC won't act on it.
    """
    ok = bool(getattr(pose_result, "ok", False))
    conf = float(getattr(pose_result, "confidence", 0.0))
    return SideOffset(
        side=side,
        x_mm=float(getattr(pose_result, "dx_mm", 0.0)),
        y_mm=float(getattr(pose_result, "dy_mm", 0.0)),
        r_deg=float(getattr(pose_result, "dr_deg", 0.0)),
        residual_mm=float(getattr(pose_result, "max_residual_mm", 0.0)),
        confidence=conf,
        fiducials_found=int(getattr(pose_result, "n_points", 0)),
        calibration_id=str(calibration_id),
        valid=(ok and conf >= min_confidence),
    )


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class PLCBackend:
    """Interface: connect, write a list of (tag, value), read one tag, close."""
    name = "base"
    def connect(self): ...
    def write_tags(self, pairs): return []
    def read_tag(self, tag_name): return None
    def close(self): pass
    def __enter__(self):
        self.connect()
        return self
    def __exit__(self, *exc):
        self.close()


class MockPLCBackend(PLCBackend):
    """In-memory PLC: stores tag values and records write order, so tests can
    assert that Data_Valid really was written last."""
    name = "Mock PLC (in-memory)"

    def __init__(self):
        self.tags = {}
        self.writes = []        # ordered list of tag names, in write order

    def write_tags(self, pairs):
        results = []
        for t, v in pairs:
            self.tags[t] = v
            self.writes.append(t)
            results.append((t, True, "Success"))
        return results

    def read_tag(self, tag_name):
        return self.tags.get(tag_name)


class PylogixBackend(PLCBackend):
    """Real CompactLogix over EtherNet/IP via pylogix."""

    def __init__(self, ip_address, processor_slot=0):
        if not HAVE_PYLOGIX:
            raise RuntimeError("pylogix not installed: pip3 install pylogix")
        self.ip_address = ip_address
        self.processor_slot = processor_slot
        self.comm = None
        self.name = f"CompactLogix @ {ip_address}"

    def connect(self):
        self.comm = PLC()
        self.comm.IPAddress = self.ip_address
        self.comm.ProcessorSlot = self.processor_slot

    def write_tags(self, pairs):
        pairs = list(pairs)
        resp = self.comm.Write(pairs)            # pylogix accepts a list of tuples
        if not isinstance(resp, list):
            resp = [resp]
        results = []
        for r in resp:
            status = getattr(r, "Status", "Unknown")
            results.append((getattr(r, "TagName", None), status == "Success", status))
        return results

    def read_tag(self, tag_name):
        r = self.comm.Read(tag_name)
        return getattr(r, "Value", None)

    def close(self):
        try:
            if self.comm:
                self.comm.Close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Publisher (enforces the write ordering)
# --------------------------------------------------------------------------- #
def write_offset(backend, offset, seq_id, side_base=None, members=None):
    """Write one side's offset with the safe ordering. Returns the list of
    (tag, ok, status) results from every underlying write."""
    base = (side_base or DEFAULT_SIDE_BASE)[offset.side]
    M = members or DEFAULT_MEMBERS
    valid_tag = tag(base, M["valid"])
    seq_tag = tag(base, M["seq_id"])

    results = []
    # 1. invalidate while we write
    results += backend.write_tags([(valid_tag, False)])
    # 2. payload
    results += backend.write_tags([
        (tag(base, M["x"]), float(offset.x_mm)),
        (tag(base, M["y"]), float(offset.y_mm)),
        (tag(base, M["r"]), float(offset.r_deg)),
        (tag(base, M["residual"]), float(offset.residual_mm)),
        (tag(base, M["confidence"]), float(offset.confidence)),
        (tag(base, M["fiducials"]), int(offset.fiducials_found)),
        (tag(base, M["calibration_id"]), str(offset.calibration_id)),
    ])
    # 3. stamp Seq_ID, then 4. set Data_Valid LAST
    results += backend.write_tags([(seq_tag, int(seq_id))])
    results += backend.write_tags([(valid_tag, bool(offset.valid))])
    return results


class OffsetPublisher:
    """Holds the PLC backend, per-side Seq_ID counters, and the tag layout."""

    def __init__(self, backend, side_base=None, members=None, min_confidence=0.0):
        self.backend = backend
        self.side_base = side_base or DEFAULT_SIDE_BASE
        self.members = members or DEFAULT_MEMBERS
        self.min_confidence = min_confidence
        self._seq = {"A": 0, "B": 0}

    def connect(self):
        self.backend.connect()
        return self

    def close(self):
        self.backend.close()

    def next_seq(self, side):
        return self._seq.get(side, 0) + 1

    def publish(self, side, pose_result, calibration_id):
        """Convert a pose result to an offset and write it. Returns (results, offset)."""
        self._seq[side] = self.next_seq(side)
        offset = from_pose(side, pose_result, calibration_id, self.min_confidence)
        results = write_offset(self.backend, offset, self._seq[side],
                               self.side_base, self.members)
        return results, offset


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test():
    import pose

    backend = MockPLCBackend()
    pub = OffsetPublisher(backend, min_confidence=0.5)

    good = pose.PoseResult(ok=True, dx_mm=0.900, dy_mm=1.300, dr_deg=-1.800,
                           rms_residual_mm=0.059, max_residual_mm=0.081,
                           n_points=4, confidence=0.76)
    results, offset = pub.publish("A", good, "cal_20260625_101500")
    w = backend.writes
    base = "Vision_SideA"

    checks = []
    checks.append(("payload X", backend.read_tag(f"{base}.Offset_X") == 0.900))
    checks.append(("payload Y", backend.read_tag(f"{base}.Offset_Y") == 1.300))
    checks.append(("payload R", backend.read_tag(f"{base}.Offset_R") == -1.800))
    checks.append(("cal id", backend.read_tag(f"{base}.Calibration_ID") == "cal_20260625_101500"))
    checks.append(("fiducials", backend.read_tag(f"{base}.Fiducials_Found") == 4))
    checks.append(("seq == 1", backend.read_tag(f"{base}.Seq_ID") == 1))
    checks.append(("valid latched True", backend.read_tag(f"{base}.Data_Valid") is True))
    # Ordering guarantees
    checks.append(("invalidate first", w[0] == f"{base}.Data_Valid"))
    checks.append(("valid written last", w[-1] == f"{base}.Data_Valid"))
    checks.append(("payload before seq",
                   w.index(f"{base}.Offset_X") < w.index(f"{base}.Seq_ID")))
    checks.append(("seq before final valid",
                   w.index(f"{base}.Seq_ID") < len(w) - 1))
    checks.append(("all writes ok", all(ok for _t, ok, _s in results)))

    # Seq increments on the next publish
    pub.publish("A", good, "cal_20260625_101500")
    checks.append(("seq == 2", backend.read_tag(f"{base}.Seq_ID") == 2))

    # Low-confidence result: written for visibility, but Data_Valid stays False
    bad = pose.PoseResult(ok=True, dx_mm=0.1, dy_mm=0.1, dr_deg=0.0,
                          rms_residual_mm=0.9, max_residual_mm=1.4,
                          n_points=4, confidence=0.10)
    pub.publish("B", bad, "cal_20260625_101500")
    checks.append(("low-conf not valid",
                   backend.read_tag("Vision_SideB.Data_Valid") is False))
    checks.append(("low-conf payload still written",
                   backend.read_tag("Vision_SideB.Offset_X") == 0.1))

    # No-solution result: invalid
    nores = pose.PoseResult(ok=False, reason="too few fiducials")
    pub.publish("B", nores, "cal_20260625_101500")
    checks.append(("no-solution not valid",
                   backend.read_tag("Vision_SideB.Data_Valid") is False))

    ok_all = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok_all = ok_all and passed
    print("[TEST] PASS" if ok_all else "[TEST] FAIL")
    return 0 if ok_all else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Vision -> CompactLogix offset publisher")
    ap.add_argument("--mock", action="store_true", help="Run the in-memory self-test")
    ap.add_argument("--ip", help="PLC IP address (real publish; requires pylogix)")
    ap.add_argument("--slot", type=int, default=0, help="Processor slot (default 0)")
    args = ap.parse_args()

    if args.mock or not args.ip:
        sys.exit(_self_test())

    # Tiny smoke publish to a real PLC (zeros), to prove connectivity / tag names.
    import pose
    backend = PylogixBackend(args.ip, args.slot)
    with backend:
        pub = OffsetPublisher(backend)
        demo = pose.PoseResult(ok=True, dx_mm=0.0, dy_mm=0.0, dr_deg=0.0,
                               rms_residual_mm=0.0, max_residual_mm=0.0,
                               n_points=4, confidence=1.0)
        for side in ("A", "B"):
            results, _off = pub.publish(side, demo, "cal_smoke")
            for t, ok, status in results:
                print(f"  {t}: {status}")


if __name__ == "__main__":
    main()
