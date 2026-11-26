"""Pictures from a text description on this laptop: stable-diffusion.cpp (MIT, Vulkan on the Iris Xe) with a
permissively licensed model (Direction 2026-11-01). The process is hidden, stops with the turn or at its time limit, and
the picture is saved in the media folder under a name made from its content.
Presets (files installed by Clover AI Studio setup, component "pictures"):
  sd15-lcm       Stable Diffusion 1.5 (CreativeML OpenRAIL-M) + LCM-LoRA (OpenRAIL++), 4 steps
  sd15           Stable Diffusion 1.5, 20 steps
"""
import base64
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import local_terminal
import paths

TIME_LIMIT_S = 600
MAX_PROMPT = 900
# SD 1.5 is trained at 512 px; the model asked for 1024 x 1024 "for a wallpaper" and sd-cli took 11 GB and paged for
# 10 minutes on this 16 GB laptop (2026-11-01), so 768 is the most it gets
MAX_SIZE = 768
# sd-cli holds ~3 GB (measured peak 2.97 GB at 512 px); with 1 GB free Windows paged and a 33 s picture ran past 600 s
NEED_FREE_MB = 3500
SD_SERVER_URL = os.environ.get("CLOVER_SD_SERVER_URL", "http://127.0.0.1:18084")


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


def _save_png_bytes(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(data).hexdigest()[:32] + ".png"
    final = out_dir / name
    if not final.exists():
        final.write_bytes(data)
    return name


def _generate_via_sd_server(text, out_dir, preset_cfg, size=512, seed=None, time_limit_s=TIME_LIMIT_S):
    """Generate an image via local resident sd-server (stable-diffusion.cpp HTTP daemon on 127.0.0.1:18084)."""
    t0 = time.perf_counter()
    seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
    steps = 4 if preset_cfg.get("suffix") else 20
    cfg_scale = 1.0 if preset_cfg.get("suffix") else 7.0
    payload = {
        "prompt": text + preset_cfg.get("suffix", ""),
        "width": int(size),
        "height": int(size),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_name": "lcm" if preset_cfg.get("suffix") else "euler_a",
    }
    req = urllib.request.Request(
        f"{SD_SERVER_URL.rstrip('/')}/sdapi/v1/txt2img",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=min(120, max(15, int(time_limit_s)))) as r:
        res = json.loads(r.read().decode("utf-8", "replace"))
    images = res.get("images") or []
    if not images:
        raise RuntimeError("sd-server returned no images")
    raw_b64 = str(images[0])
    if "," in raw_b64 and raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    data = base64.b64decode(raw_b64)
    if not data.startswith(b"\x89PNG"):
        raise RuntimeError("sd-server did not return a valid PNG")
    name = _save_png_bytes(data, out_dir)
    secs = round(time.perf_counter() - t0, 1)
    return {"file": name, "seconds": secs, "model": preset_cfg["label"], "seed": seed, "width": size, "height": size}


def _generate_via_k3_worker(text, out_dir, preset="sd15-lcm", size=512, seed=None, time_limit_s=TIME_LIMIT_S, base=None):
    """Delegate Stable Diffusion 1.5 image generation to the k3 server's /api/worker/image endpoint when the local
    device (e.g. Android phone) does not have the 4.2 GB SD 1.5 checkpoint installed locally."""
    import k3_host_engine
    key = k3_host_engine.k3_host_key(base)
    if not key:
        raise RuntimeError("no k3 host key configured for remote image generation")
    seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
    payload = json.dumps({"prompt": text, "preset": preset, "size": int(size), "seed": seed}).encode("utf-8")
    last_err = "no k3 URL configured"
    t0 = time.perf_counter()
    for _attempt in range(2):
        for su in k3_host_engine.K3_URLS:
            try:
                req = urllib.request.Request(
                    f"{su.rstrip('/')}/api/worker/image",
                    payload,
                    {
                        "Content-Type": "application/json",
                        "X-Clover-Key": key,
                        "Authorization": f"Bearer {key}",
                        "User-Agent": "CloverStudio/1.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=min(120, max(25, int(time_limit_s)))) as r:
                    res = json.loads(r.read().decode("utf-8", "replace"))
                if not isinstance(res, dict) or not res.get("ok") or not res.get("image_base64"):
                    last_err = str((res or {}).get("error") or "invalid response from k3 image worker")
                    continue
                data = base64.b64decode(str(res["image_base64"]))
                if not data.startswith(b"\x89PNG"):
                    last_err = "k3 image worker returned non-PNG payload"
                    continue
                name = _save_png_bytes(data, out_dir)
                secs = round(time.perf_counter() - t0, 1)
                return {
                    "file": name,
                    "seconds": res.get("seconds") or secs,
                    "model": res.get("model") or "Stable Diffusion 1.5 + LCM (4 steps)",
                    "seed": int(res.get("seed") if res.get("seed") is not None else seed),
                    "width": int(res.get("width") or size),
                    "height": int(res.get("height") or size),
                }
            except Exception as exc:
                last_err = str(exc)
    raise RuntimeError(last_err)


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
        """-> {"file", "seconds", "model", "seed", "width", "height"}; uses Stable Diffusion 1.5 (sd-server / sd-cli / k3 worker)."""
        text = re.sub(r"\s+", " ", str(prompt or "")).strip()[:MAX_PROMPT]
        if not text:
            raise RuntimeError("the picture needs a description")
        p = presets(self.base).get(self.preset)
        if p is None:
            raise RuntimeError(f"unknown image model {self.preset!r}")
        size = max(256, min(MAX_SIZE, int(size) // 64 * 64))
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        if paths.WEB:
            try:
                return _generate_via_sd_server(text, out_dir, p, size=size, seed=seed, time_limit_s=time_limit_s)
            except Exception:
                pass
        ok, why = self.available()
        if not ok:
            if paths.ANDROID or os.environ.get("CLOVER_REMOTE_IMAGE", "1") != "0":
                try:
                    return _generate_via_k3_worker(
                        text, out_dir, preset=self.preset, size=size, seed=seed,
                        time_limit_s=time_limit_s, base=self.base,
                    )
                except Exception as exc:
                    raise RuntimeError(f"{why}; remote k3 image worker failed: {exc}") from None
            raise RuntimeError(why)
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
