"""Pictures from a text description on this laptop: stable-diffusion.cpp (MIT, Vulkan on the Iris Xe) with a
permissively licensed model (Direction 2026-11-01). The process is hidden, stops with the turn or at its time limit, and
the picture is saved in the media folder under a name made from its content.
Presets (files installed by Clover AI Studio setup, component "pictures"):
  sd15-lcm       Stable Diffusion 1.5 (CreativeML OpenRAIL-M) + LCM-LoRA (OpenRAIL++), 4 steps
  sd15           Stable Diffusion 1.5, 20 steps
"""
import hashlib
import random
import re
import time
from pathlib import Path

import local_terminal

TIME_LIMIT_S = 600
MAX_PROMPT = 900
# SD 1.5 is trained at 512 px; the model asked for 1024 x 1024 "for a wallpaper" and sd-cli took 11 GB and paged for
# 10 minutes on this 16 GB laptop (2026-11-01), so 768 is the most it gets
MAX_SIZE = 768
# sd-cli holds ~3 GB (measured peak 2.97 GB at 512 px); with 1 GB free Windows paged and a 33 s picture ran past 600 s
NEED_FREE_MB = 3500


def free_mb():
    """Physical memory free now, in MB (Windows), or None."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys // (1024 * 1024)
    except (AttributeError, OSError):
        pass
    return None


def presets(base):
    img = Path(base) / "data" / "image"
    sd15 = img / "sd15" / "v1-5-pruned-emaonly.safetensors"
    return {
        "sd15-lcm": {"files": [sd15, img / "lora" / "lcm-lora-sdv1-5.safetensors"],
                     "args": ["-m", str(sd15), "--type", "f16", "--lora-model-dir", str(img / "lora"), "--steps", "4",
                              "--cfg-scale", "1", "--sampling-method", "lcm"],
                     "suffix": "<lora:lcm-lora-sdv1-5:1>", "label": "Stable Diffusion 1.5 + LCM (4 steps)"},
        "sd15": {"files": [sd15], "args": ["-m", str(sd15), "--type", "f16", "--steps", "20", "--cfg-scale", "7"],
                 "suffix": "", "label": "Stable Diffusion 1.5 (20 steps)"},
    }


class ImageGen:
    def __init__(self, base, preset="sd15-lcm"):
        self.base = Path(base)
        self.exe = self.base / "tools" / "sd.cpp" / "vulkan" / "sd-cli.exe"
        self.preset = preset

    def available(self):
        p = presets(self.base).get(self.preset)
        if p is None:
            return False, f"unknown image model {self.preset!r}"
        missing = [f.name for f in [self.exe] + p["files"] if not f.is_file()]
        return (not missing), ("missing " + ", ".join(missing) + " (run Clover AI Studio setup again)") if missing else ""

    def generate(self, prompt, out_dir, size=512, seed=None, cancel=None, time_limit_s=TIME_LIMIT_S):
        """-> {"file", "seconds", "model", "seed", "width", "height"}; raises RuntimeError with the reason."""
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        text = re.sub(r"\s+", " ", str(prompt or "")).strip()[:MAX_PROMPT]
        if not text:
            raise RuntimeError("the picture needs a description")
        p = presets(self.base)[self.preset]
        size = max(256, min(MAX_SIZE, int(size) // 64 * 64))
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".gen-{seed}-{int(time.time() * 1000)}.png"
        args = [str(self.exe), *p["args"], "-p", text + p["suffix"], "-W", str(size), "-H", str(size), "-s", str(seed),
                "-o", str(tmp)]
        t0 = time.perf_counter()
        out, code, how = local_terminal.run(args, cwd=str(self.exe.parent), timeout_s=time_limit_s, cancel=cancel)
        secs = round(time.perf_counter() - t0, 1)
        if how != "exit" or code != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            tail = " ".join(out.strip().splitlines()[-3:])[-300:]
            raise RuntimeError({"timeout": f"stopped after {time_limit_s} s", "stopped": "stopped by the user"}.get(
                how, f"the image model failed (exit {code}): {tail}"))
        data = tmp.read_bytes()
        name = hashlib.sha256(data).hexdigest()[:32] + ".png"
        final = out_dir / name
        if final.exists():
            tmp.unlink()
        else:
            tmp.replace(final)
        return {"file": name, "seconds": secs, "model": p["label"], "seed": seed, "width": size, "height": size}
