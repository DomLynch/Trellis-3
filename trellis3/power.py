"""Power-source gate: a GPU run must not start on battery.

On the ROG Flow Z13 a power-source change mid-run throttled the GPU (86 -> 140 s/step) and then aborted the HIP
runtime (c10 SetDevice) at 12:30, 2026-11-02. generate.py refuses to start a GPU run while the machine
discharges; override with --allow-battery or TRELLIS3_ALLOW_BATTERY=1. Non-Windows hosts and desktops report
"unknown" and are never blocked.
"""
import os
import sys


def on_battery():
    """True when running on battery, False on AC, None when unknown."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        if s.ACLineStatus == 255:
            return None
        return s.ACLineStatus == 0
    except Exception:
        return None


def refuse_on_battery(allow=False, probe=on_battery):
    """Message to print when the run must not start, else None."""
    if allow or os.environ.get("TRELLIS3_ALLOW_BATTERY") == "1":
        return None
    if probe() is True:
        return ("Refusing to start a GPU run on battery: a power-source change throttles the GPU and has aborted the HIP "
                "runtime mid-run. Plug in the charger (performance power plan), or pass --allow-battery.")
    return None
