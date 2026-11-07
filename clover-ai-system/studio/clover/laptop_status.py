"""This laptop's free disk space, battery, memory and the apps using the most memory, for the laptop_status tool.
LFM3-Omni-2B wrote its own PowerShell for these and got it wrong (sums of file sizes in the home folder for "free disk
space", an invented `get-batpower` for the battery; 2026-11-01), so the readings are fixed code that takes no input."""
import ctypes
import os
import shutil
import string

import local_terminal

TOP_APPS = ("Get-Process | Group-Object ProcessName | ForEach-Object { [pscustomobject]@{ Name = $_.Name; "
            "MB = [int](($_.Group | Measure-Object WorkingSet64 -Sum).Sum / 1MB) } } | "
            "Sort-Object MB -Descending | Select-Object -First 6 | ForEach-Object { \"$($_.Name) $($_.MB) MB\" }")


def _gb(n):
    return f"{n / 1024 ** 3:.1f} GB"


def disks():
    rows = []
    mask = ctypes.windll.kernel32.GetLogicalDrives() if os.name == "nt" else 0
    for i, letter in enumerate(string.ascii_uppercase):
        if not mask >> i & 1 or ctypes.windll.kernel32.GetDriveTypeW(f"{letter}:\\") != 3:  # 3: a fixed disk
            continue
        try:
            u = shutil.disk_usage(f"{letter}:\\")
        except OSError:
            continue
        rows.append(f"Drive {letter}: {_gb(u.free)} free of {_gb(u.total)}")
    return rows


def battery():
    class PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong)]
    st = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return None
    if st.BatteryFlag == 128 or st.BatteryLifePercent == 255:
        return "No battery found (desktop or battery removed)."
    plug = {0: "on battery", 1: "plugged in"}.get(st.ACLineStatus, "power source unknown")
    left = (f", about {st.BatteryLifeTime // 60} minutes left" if st.ACLineStatus == 0 and st.BatteryLifeTime != 0xFFFFFFFF
            else "")
    return f"Battery {st.BatteryLifePercent}%, {plug}{' (charging)' if st.BatteryFlag & 8 else ''}{left}"


def memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MemoryStatus()
    st.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return (f"Memory (RAM): {_gb(st.ullTotalPhys - st.ullAvailPhys)} in use of {_gb(st.ullTotalPhys)}, "
            f"{_gb(st.ullAvailPhys)} free")


def report(cancel=None):
    if os.name != "nt":
        lines = []
        try:
            u = shutil.disk_usage("/")
            lines.append(f"Storage (/): {_gb(u.free)} free of {_gb(u.total)}")
        except OSError:
            pass
        try:
            import hardware
            tot, free = hardware.android_memory_gb()
            if tot is not None and free is not None:
                lines.append(f"Memory (RAM): {tot - free:.1f} GB in use of {tot:.1f} GB, {free:.1f} GB free")
        except Exception:
            pass
        try:
            import device_actions
            act = device_actions.read_active_activity()
            if act.get("open_windows"):
                lines.append("Running apps / processes: " + "; ".join(act["open_windows"][:6]))
        except Exception:
            pass
        return "\n".join(lines) or "Could not read device status."
    lines = disks()
    for part in (battery, memory):
        try:
            line = part()
        except (AttributeError, OSError):
            line = None
        if line:
            lines.append(line)
    out, code, _ = local_terminal.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                                       local_terminal.PS_HEAD + TOP_APPS], timeout_s=20, cancel=cancel)
    apps = [a.strip() for a in out.splitlines() if a.strip()] if code == 0 else []
    if apps:
        lines.append("Apps using the most memory: " + "; ".join(apps))
    try:
        import device_actions
        act = device_actions.read_active_activity()
        if act.get("foreground_window"):
            lines.append("Active window: " + act["foreground_window"])
        if act.get("open_windows"):
            lines.append("Open windows: " + "; ".join(act["open_windows"][:8]))
    except Exception:
        pass
    return "\n".join(lines) or "Could not read this laptop's status."
