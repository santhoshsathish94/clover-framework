"""Clover Studio Image Generation Gateway (`image_gen.py`).

Provides three explicit, non-deceptive backend tiers with full provenance metadata:
  1. High-Quality Generative API (`gpt-image-1` / OpenAI-compatible Images API):
     - Supports high-detail generation, reference-image editing (`/v1/images/edits`), and transparent backgrounds
       when explicitly configured (`config/image_provider.json` or `CLOVER_IMAGE_API_KEY` / `OPENAI_API_KEY`).
  2. Local Diffusion (`sd15-lcm`, `sd15` via `stable-diffusion.cpp` `sd-cli`):
     - Runs locally on Vulkan / CPU when checkpoint files are installed under `<base>/data/image/`.
     - Supports both text-to-image (`generate`) and reference-guided `img2img` (`edit` via `-i`).
  3. Procedural Diagram (`diagram` — explicit opt-in ONLY):
     - Simple labeled geometric/diagrammatic renderer (`is_generative = False`, `backend_tier = "procedural_diagram"`).
     - NEVER silently substituted when a generative model (`sd15-lcm`, `sd15`, `gpt-image-1`, `auto`) is unavailable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import struct
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import local_terminal

TIME_LIMIT_S = 600
MAX_PROMPT = 900
# SD 1.5 is trained at 512 px; the model asked for 1024 x 1024 "for a wallpaper" and sd-cli took 11 GB and paged for
# 10 minutes on this 16 GB laptop (2026-11-01), so 768 is the most it gets
MAX_SIZE = 768
# sd-cli holds ~3 GB (measured peak 2.97 GB at 512 px); with 1 GB free Windows paged and a 33 s picture ran past 600 s
NEED_FREE_MB = 3500


class ImageBackendUnavailableError(RuntimeError):
    """Raised when the requested generative image backend is not installed or configured."""

    def __init__(self, message: str, *, backend: str = "", failure_layer: str = "backend_capability"):
        super().__init__(message)
        self.backend = backend
        self.failure_layer = failure_layer


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


def _api_provider_config(base: Path) -> dict | None:
    """Read optional user-configured high-quality image provider (`config/image_provider.json` or env vars)."""
    cfg_path = Path(base) / "config" / "image_provider.json"
    cfg: dict = {}
    if cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg = raw
        except (OSError, ValueError):
            cfg = {}
    api_key = (
        str(cfg.get("api_key") or "").strip()
        or os.environ.get("CLOVER_IMAGE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        str(cfg.get("base_url") or "").strip()
        or os.environ.get("CLOVER_IMAGE_API_URL", "").strip()
        or "https://api.openai.com/v1"
    ).rstrip("/")
    model = str(cfg.get("model") or os.environ.get("CLOVER_IMAGE_API_MODEL") or "gpt-image-1").strip()
    allow_keyless_local = bool(cfg.get("allow_keyless")) and (
        base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")
    )
    if not api_key and not allow_keyless_local:
        return None
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _png_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None, None


def _save_content_png(data: bytes, out_dir: Path) -> tuple[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    name = digest[:32] + ".png"
    final = out_dir / name
    if not final.exists():
        final.write_bytes(data)
    return name, digest


def _render_procedural_diagram(prompt: str, out_dir: Path, size: int = 512, seed: int | None = None,
                               transparent: bool = False) -> dict:
    """Explicitly labeled non-generative geometric/diagram renderer (`preset='diagram'` or `allow_procedural=True` ONLY)."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t0 = time.perf_counter()
    seed_val = int(seed) if seed is not None else (int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF)
    size = max(256, min(1024, int(size) // 64 * 64))
    dpi = 128
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if not transparent:
        fig.patch.set_facecolor("#F8FAFC")
        ax.add_patch(mpatches.Rectangle((0, 0), 100, 100, color="#F8FAFC", zorder=0))
    ax.add_patch(mpatches.FancyBboxPatch((8, 16), 84, 68, boxstyle="round,pad=2", edgecolor="#334155",
                                         facecolor="#E2E8F0" if not transparent else "none", linewidth=2, zorder=1))
    ax.text(50, 74, "PROCEDURAL DIAGRAM", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A", zorder=2)
    short = prompt[:90] + ("..." if len(prompt) > 90 else "")
    ax.text(50, 48, short, ha="center", va="center", fontsize=8.5, color="#334155", wrap=True, zorder=2)
    ax.text(50, 22, "(Non-generative fallback diagram)", ha="center", va="center", fontsize=7.5, color="#64748B", zorder=2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=bool(transparent))
    plt.close(fig)
    data = buf.getvalue()
    name, digest = _save_content_png(data, Path(out_dir))
    secs = round(max(0.05, time.perf_counter() - t0), 2)
    return {
        "file": name,
        "sha256": digest,
        "seconds": secs,
        "provider": "procedural_diagram",
        "backend_tier": "procedural_diagram",
        "is_generative": False,
        "operation": "generate",
        "model": "Clover Procedural Diagram (Non-Generative)",
        "seed": seed_val,
        "width": size,
        "height": size,
        "transparent": bool(transparent),
    }


def _call_images_api(
    cfg: dict,
    prompt: str,
    out_dir: Path,
    *,
    size: int = 1024,
    reference_image: Path | None = None,
    transparent: bool = False,
    time_limit_s: float = TIME_LIMIT_S,
) -> dict:
    """Execute generation or reference-guided editing against an OpenAI-compatible Images API."""
    t0 = time.perf_counter()
    dim = 1024 if size >= 896 else (512 if size <= 512 else 1024)
    size_str = f"{dim}x{dim}"
    headers = {"User-Agent": "CloverAIStudio/1.6"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    if reference_image is not None:
        ref_path = Path(reference_image)
        if not ref_path.is_file():
            raise RuntimeError(f"reference image {ref_path.name} does not exist")
        boundary = "----CloverImgBoundary" + uuid.uuid4().hex
        parts: list[bytes] = []

        def _add_field(k: str, v: str) -> None:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))

        _add_field("model", cfg["model"])
        _add_field("prompt", prompt)
        _add_field("size", size_str)
        if transparent:
            _add_field("background", "transparent")
        img_bytes = ref_path.read_bytes()
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{ref_path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n".encode("utf-8") + img_bytes + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        url = f"{cfg['base_url']}/images/edits"
        op_kind = "edit"
    else:
        payload: dict = {"model": cfg["model"], "prompt": prompt, "n": 1, "size": size_str, "response_format": "b64_json"}
        if transparent:
            payload["background"] = "transparent"
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        url = f"{cfg['base_url']}/images/generations"
        op_kind = "generate"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=min(float(time_limit_s), 180.0)) as resp:
            resp_json = json.loads(resp.read(32 * 1024 * 1024).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_txt = exc.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"image API error (HTTP {exc.code}): {err_txt[:240]}") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"image API request failed: {exc}") from None

    items = resp_json.get("data") or []
    if not items or not isinstance(items[0], dict):
        raise RuntimeError("image API returned no image data")
    b64 = items[0].get("b64_json")
    if b64:
        png_data = base64.b64decode(b64)
    elif items[0].get("url"):
        with urllib.request.urlopen(str(items[0]["url"]), timeout=60.0) as r_img:
            png_data = r_img.read(32 * 1024 * 1024)
    else:
        raise RuntimeError("image API response contained neither b64_json nor url")

    w, h = _png_dimensions(png_data)
    name, digest = _save_content_png(png_data, Path(out_dir))
    secs = round(max(0.05, time.perf_counter() - t0), 2)
    return {
        "file": name,
        "sha256": digest,
        "seconds": secs,
        "provider": "openai_images",
        "backend_tier": "high_quality",
        "is_generative": True,
        "operation": op_kind,
        "model": f"{cfg['model']} (Images API)",
        "seed": None,
        "width": w or dim,
        "height": h or dim,
        "transparent": bool(transparent),
    }


def capabilities(base) -> dict:
    """Report honest availability of all three Image Generation Gateway tiers on `base`."""
    base_p = Path(base)
    api_cfg = _api_provider_config(base_p)
    sd_lcm = ImageGen(base_p, "sd15-lcm").available()
    sd_20 = ImageGen(base_p, "sd15").available()
    return {
        "high_quality_api": {"available": api_cfg is not None, "model": api_cfg["model"] if api_cfg else None},
        "local_diffusion": {
            "sd15_lcm_available": sd_lcm[0],
            "sd15_available": sd_20[0],
            "reason": sd_lcm[1] if not sd_lcm[0] else "",
        },
        "procedural_diagram": {"available": True, "is_generative": False},
    }


class ImageGen:
    def __init__(self, base, preset="sd15-lcm", *, allow_procedural: bool = False):
        self.base = Path(base)
        vk = self.base / "tools" / "sd.cpp" / "vulkan"
        self.exe = vk / "sd-cli.exe" if (os.name == "nt" or (vk / "sd-cli.exe").is_file() or not (vk / "sd-cli").is_file()) else vk / "sd-cli"
        self.preset = preset
        self.allow_procedural = bool(allow_procedural)

    def available(self):
        if self.preset == "diagram":
            return True, ""
        if self.preset in ("gpt-image-1", "openai"):
            cfg = _api_provider_config(self.base)
            return (cfg is not None), ("" if cfg else "high-quality image API is not configured (set config/image_provider.json or OPENAI_API_KEY)")
        if self.preset == "auto":
            if _api_provider_config(self.base) is not None:
                return True, ""
            sd_ok, sd_why = ImageGen(self.base, "sd15-lcm").available()
            if sd_ok:
                return True, ""
            return False, f"no generative image backend available ({sd_why}; and no Images API configured)"
        p = presets(self.base).get(self.preset)
        if p is None:
            return False, f"unknown image model {self.preset!r}"
        missing = [f.name for f in [self.exe] + p["files"] if not f.is_file()]
        return (not missing), ("missing " + ", ".join(missing) + " (run Clover AI Studio setup again)") if missing else ""

    def _run_local_sd(self, text: str, preset_name: str, out_dir: Path, *, size=512, seed=None,
                      reference_image: Path | None = None, cancel=None, time_limit_s=TIME_LIMIT_S) -> dict:
        p = presets(self.base)[preset_name]
        size = max(256, min(MAX_SIZE, int(size) // 64 * 64))
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".gen-{seed}-{int(time.time() * 1000)}.png"
        args = [str(self.exe), *p["args"]]
        op_kind = "generate"
        if reference_image is not None:
            ref_p = Path(reference_image)
            if not ref_p.is_file():
                raise RuntimeError(f"reference image {ref_p.name} does not exist")
            args += ["-i", str(ref_p), "--strength", "0.65"]
            op_kind = "edit"
        args += ["-p", text + p["suffix"], "-W", str(size), "-H", str(size), "-s", str(seed), "-o", str(tmp)]
        t0 = time.perf_counter()
        out, code, how = local_terminal.run(args, cwd=str(self.exe.parent), timeout_s=time_limit_s, cancel=cancel)
        secs = round(time.perf_counter() - t0, 1)
        if how != "exit" or code != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            tail = " ".join(out.strip().splitlines()[-3:])[-300:]
            raise RuntimeError({"timeout": f"stopped after {time_limit_s} s", "stopped": "stopped by the user"}.get(
                how, f"the image model failed (exit {code}): {tail}"))
        data = tmp.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        name = digest[:32] + ".png"
        final = out_dir / name
        if final.exists():
            tmp.unlink()
        else:
            tmp.replace(final)
        return {
            "file": name,
            "sha256": digest,
            "seconds": secs,
            "provider": "local_diffusion",
            "backend_tier": "local_diffusion",
            "is_generative": True,
            "operation": op_kind,
            "model": p["label"],
            "seed": seed,
            "width": size,
            "height": size,
            "transparent": False,
        }

    def generate(self, prompt, out_dir, size=512, seed=None, cancel=None, time_limit_s=TIME_LIMIT_S,
                 *, reference_image: Path | str | None = None, transparent: bool = False):
        """Generate or edit an image through the explicit 3-tier Image Generation Gateway."""
        text = re.sub(r"\s+", " ", str(prompt or "")).strip()[:MAX_PROMPT]
        if not text:
            raise RuntimeError("the picture needs a description")

        ref_path = Path(reference_image) if reference_image else None

        if self.preset == "diagram":
            return _render_procedural_diagram(text, Path(out_dir), size=size, seed=seed, transparent=transparent)

        if self.preset in ("gpt-image-1", "openai", "auto"):
            api_cfg = _api_provider_config(self.base)
            if api_cfg is not None:
                return _call_images_api(
                    api_cfg, text, Path(out_dir), size=max(size, 1024) if size == 512 else size,
                    reference_image=ref_path, transparent=transparent, time_limit_s=time_limit_s,
                )
            if self.preset in ("gpt-image-1", "openai"):
                if self.allow_procedural:
                    return _render_procedural_diagram(text, Path(out_dir), size=size, seed=seed, transparent=transparent)
                raise ImageBackendUnavailableError(
                    "high-quality image API is not configured (configure config/image_provider.json or OPENAI_API_KEY)",
                    backend=self.preset,
                )

        local_preset = "sd15-lcm" if self.preset == "auto" else self.preset
        ok, why = ImageGen(self.base, local_preset).available()
        if ok:
            return self._run_local_sd(
                text, local_preset, Path(out_dir), size=size, seed=seed,
                reference_image=ref_path, cancel=cancel, time_limit_s=time_limit_s,
            )

        if self.allow_procedural:
            return _render_procedural_diagram(text, Path(out_dir), size=size, seed=seed, transparent=transparent)

        raise ImageBackendUnavailableError(
            f"image generation model unavailable: {why}",
            backend=local_preset,
        )

    def edit(self, prompt, reference_image: Path | str, out_dir, size=512, seed=None, cancel=None,
             time_limit_s=TIME_LIMIT_S, *, transparent: bool = False):
        """Edit an existing reference image while preserving unrequested details."""
        return self.generate(
            prompt, out_dir, size=size, seed=seed, cancel=cancel, time_limit_s=time_limit_s,
            reference_image=reference_image, transparent=transparent,
        )
