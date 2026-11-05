"""This machine, read when the Studio starts: CPU (cores, threads, AVX2), memory, the GPUs llama.cpp can use (Vulkan),
and the settings chosen from them. Nothing is fixed to one machine: the engine (GPU or CPU), the context length, the
threads of the speech instance and of CSI follow what is found.

The minimum Clover AI Studio supports is the laptop it was built and measured on (13th Gen Intel Core i5-1335U: 10
cores / 12 threads with AVX2, 16 GB memory, Intel Iris Xe through Vulkan, Windows 11). A machine with more gets more
(context, threads); one without a usable GPU runs the model on the CPU."""
import ctypes
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import paths

try:
    from ctypes import wintypes
except (ImportError, ValueError):  # not on Windows
    wintypes = None

# the family laptop: 15.7 GB visible of 16 GB, 12 threads, AVX2, Windows 10 2004 or later
MINIMUM = {"ram_gb": 15.0, "threads": 8, "avx2": True, "windows_build": 19041, "disk_gb": 25}
SOFTWARE_GPU = re.compile(r"llvmpipe|swiftshader|basic render|microsoft basic|lavapipe", re.I)


def _k32():
    return ctypes.WinDLL("kernel32", use_last_error=True)


def cpu():
    info = {"name": "", "threads": os.cpu_count() or 1, "cores": None, "performance_cores": None, "avx2": None}
    if os.name != "nt":
        txt = _read("/proc/cpuinfo")
        m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
        if m:
            info["name"] = m.group(1).strip()
        mc = re.search(r"^cpu cores\s*:\s*(\d+)$", txt, re.M)
        if mc:
            info["cores"] = int(mc.group(1))
            info["performance_cores"] = info["cores"]
        if "flags" in txt:
            info["avx2"] = bool(re.search(r"\bavx2\b", txt))
        return info
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            info["name"] = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except OSError:
        pass
    k32 = _k32()
    info["avx2"] = bool(k32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
    # one record per physical core (RelationProcessorCore); EfficiencyClass is higher for performance cores
    size = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if size.value and k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        classes, off, raw = [], 0, buf.raw
        while off < size.value:
            rel, rec = struct.unpack_from("<II", raw, off)
            if rel == 0:
                classes.append(raw[off + 9])  # PROCESSOR_RELATIONSHIP: Flags (byte 8), EfficiencyClass (byte 9)
            off += rec or size.value
        if classes:
            info["cores"] = len(classes)
            info["performance_cores"] = sum(1 for c in classes if c == max(classes))
    return info


def memory_gb():
    if os.name != "nt":
        return android_memory_gb()

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    _k32().GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullTotalPhys / 2 ** 30, 1), round(m.ullAvailPhys / 2 ** 30, 1)


def gpus(base):
    """The Vulkan devices llama.cpp's Vulkan build sees: [{"name", "memory_mb", "free_mb", "software"}]."""
    exe = Path(base) / "tools" / "llama.cpp" / "vulkan" / "llama-server.exe"
    if not exe.is_file():
        return []
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60,
                             creationflags=0x08000000 if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for m in re.finditer(r"Vulkan\d+:\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", out.stdout + out.stderr):
        found.append({"name": m.group(1).strip(), "memory_mb": int(m.group(2)), "free_mb": int(m.group(3)),
                      "software": bool(SOFTWARE_GPU.search(m.group(1)))})
    return found


def choose(c, ram, gpu_list):
    """Settings for this machine. GPU when Vulkan finds a real one (the 2B model with its projector needs about 2 GB of
    the GPU's memory); context 32,768 tokens at the minimum's 16 GB, 65,536 from 30 GB; the speech instance and CSI use
    the physical cores the chat engine leaves."""
    usable = [g for g in gpu_list if not g["software"] and g["memory_mb"] >= 2048]
    best = max(usable, key=lambda g: g["memory_mb"]) if usable else None
    cores = c.get("cores") or max(1, (c.get("threads") or 2) // 2)
    return {"engine": "gpu" if best else "cpu", "gpu": best["name"] if best else None,
            "ctx": 65536 if (ram or 0) >= 30 else 32768,
            "voice_threads": max(4, min(16, cores)),
            "csi_threads": 2 if cores <= 10 else 4}


def meets_minimum(info, disk_free_gb=None):
    """-> list of what is below the minimum (empty: fine)."""
    low = []
    c = info.get("cpu") or {}
    if (info.get("ram_gb") or 0) < MINIMUM["ram_gb"]:
        low.append(f"memory {info.get('ram_gb')} GB (needs 16 GB)")
    if (c.get("threads") or 0) < MINIMUM["threads"]:
        low.append(f"{c.get('threads')} CPU threads (needs {MINIMUM['threads']})")
    if c.get("avx2") is False:
        low.append("the CPU has no AVX2")
    if os.name == "nt" and (info.get("windows_build") or 0) < MINIMUM["windows_build"]:
        low.append(f"Windows build {info.get('windows_build')} (needs Windows 10 2004 or later, 64-bit)")
    if disk_free_gb is not None and disk_free_gb < MINIMUM["disk_gb"]:
        low.append(f"{disk_free_gb:.0f} GB free disk (needs {MINIMUM['disk_gb']} GB)")
    return low


def probe(base, save_to=None):
    if paths.ANDROID:
        info = android_probe(base)
    else:
        c = cpu()
        ram, free = memory_gb()
        g = gpus(base)
        info = {"cpu": c, "ram_gb": ram, "ram_free_gb": free, "gpus": g,
                "windows_build": sys.getwindowsversion().build if os.name == "nt" else None,
                "chosen": choose(c, ram, g)}
        info["below_minimum"] = meets_minimum(info)
    if save_to:
        p = Path(save_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------- Android (the phone app)
def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def android_cpu():
    """Phone CPUs mix big and small cores: the big ones (within 70% of the fastest clock) run the model."""
    threads = os.cpu_count() or 1
    freqs = []
    for i in range(threads):
        f = _read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq").strip()
        if f.isdigit():
            freqs.append(int(f))
    big = sum(1 for f in freqs if f >= 0.7 * max(freqs)) if freqs else None
    m = re.search(r"^Hardware\s*:\s*(.+)$", _read("/proc/cpuinfo"), re.M)
    name = os.environ.get("CLOVER_SOC") or (m.group(1).strip() if m else "")
    return {"name": name, "threads": threads, "cores": threads, "performance_cores": big, "avx2": None,
            "max_mhz": [f // 1000 for f in freqs]}


def android_memory_gb():
    kb = dict(re.findall(r"^(\w+):\s+(\d+) kB", _read("/proc/meminfo"), re.M))
    total, free = kb.get("MemTotal"), kb.get("MemAvailable")
    return (round(int(total) / 2 ** 20, 1) if total else None), (round(int(free) / 2 ** 20, 1) if free else None)


def android_gpus(base):
    """The GPUs the app's llama.cpp sees (OpenCL); Adreno is what its OpenCL backend is made for."""
    import llamacpp_engine as L
    exe = L.server_exe(base, "vulkan")
    if not exe.is_file():
        return []
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60,
                             env=L.server_env(dict(os.environ)))
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for m in re.finditer(r"^\s*(GPUOpenCL\w*|OpenCL\w*|Vulkan\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)",
                         out.stdout + out.stderr, re.M):
        found.append({"id": m.group(1), "name": m.group(2).strip(), "memory_mb": int(m.group(3)),
                      "free_mb": int(m.group(4)), "software": False})
    return found


def android_choose(c, ram, gpu_list):
    """GPU when an Adreno is found (else the CPU); context 8,192 tokens up to 10 GB of memory, 16,384 above; chat and
    speech share one llama.cpp instance and use the big cores."""
    adreno = [g for g in gpu_list if "adreno" in g["name"].lower()]
    big = c.get("performance_cores") or max(1, (c.get("threads") or 4) // 2)
    return {"engine": "gpu" if adreno else "cpu", "gpu": adreno[0]["name"] if adreno else None,
            "ctx": 16384 if (ram or 0) >= 10.5 else 8192, "threads": big, "voice_threads": big, "csi_threads": 1}


def android_probe(base):
    c = android_cpu()
    ram, free = android_memory_gb()
    g = android_gpus(base)
    low = [f"memory {ram} GB (6 GB or more works best)"] if ram is not None and ram < 5.5 else []
    return {"cpu": c, "ram_gb": ram, "ram_free_gb": free, "gpus": g, "windows_build": None,
            "android_api": sys.getandroidapilevel(), "chosen": android_choose(c, ram, g), "below_minimum": low}


if __name__ == "__main__":  # the installer prints this (and keeps it in config\hardware.json)
    import shutil
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    out = probe(base, base / "config" / "hardware.json")
    out["below_minimum"] = meets_minimum(out, shutil.disk_usage(base).free / 2 ** 30)
    print(json.dumps(out, indent=2))
