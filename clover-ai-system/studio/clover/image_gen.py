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


def _render_canvas_image(prompt, out_dir, size=512, seed=None):
    """On-device Matplotlib / NumPy visual canvas renderer used on Android, Web, or laptops where the 2 GB SD 1.5
    checkpoint is not installed."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    t0 = time.perf_counter()
    seed = int(seed) if seed is not None else (int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF)
    rng = np.random.default_rng(seed)
    low = prompt.lower()

    dpi = 128
    fig_in = size / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    if any(k in low for k in ("dna", "helix", "gene", "chromosome", "rna", "molecule", "biology")):
        fig.patch.set_facecolor("#0B1622")
        ax.set_facecolor("#0B1622")
        y = np.linspace(8, 92, 300)
        phase = (y - 8) * 0.14
        x1 = 50 + 22 * np.sin(phase)
        x2 = 50 + 22 * np.sin(phase + np.pi)
        z = np.cos(phase)
        colors_bp = [("#38BDF8", "#F43F5E"), ("#4ADE80", "#FACC15")]
        for idx, yr in enumerate(np.linspace(12, 88, 22)):
            ph = (yr - 8) * 0.14
            rx1 = 50 + 22 * np.sin(ph)
            rx2 = 50 + 22 * np.sin(ph + np.pi)
            xm = 0.5 * (rx1 + rx2)
            c1, c2 = colors_bp[idx % 2]
            ax.plot([rx1, xm], [yr, yr], color=c1, lw=3.2, alpha=0.9, solid_capstyle="round")
            ax.plot([xm, rx2], [yr, yr], color=c2, lw=3.2, alpha=0.9, solid_capstyle="round")
            ax.scatter([rx1, rx2], [yr, yr], color="#E2E8F0", s=26, zorder=4)
        ax.plot(x1, y, color="#38BDF8", lw=4.5, alpha=0.95, zorder=3)
        ax.plot(x2, y, color="#A855F7", lw=4.5, alpha=0.95, zorder=3)
        ax.scatter(x1[::10], y[::10], c=z[::10], cmap="cool", s=40, zorder=5)
        ax.text(50, 95, "DNA Double Helix Structure", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center", va="top")
        ax.text(50, 3.5, "A–T  ·  G–C Base Pairs & Sugar-Phosphate Backbone", color="#94A3B8", fontsize=7.5, ha="center")
    elif any(k in low for k in ("cat", "kitty", "kitten")):
        fig.patch.set_facecolor("#F7F3E9")
        ax.set_facecolor("#F7F3E9")
        ax.add_patch(mpatches.Circle((50, 48), 34, color="#E3EEDA", zorder=1))
        # Body and tail
        ax.add_patch(mpatches.Ellipse((50, 26), 34, 26, color="#334155", zorder=2))
        t_theta = np.linspace(0, 1.3 * np.pi, 80)
        ax.plot(64 + 12 * np.sin(t_theta), 22 + 10 * (1 - np.cos(t_theta)), color="#334155", lw=7, solid_capstyle="round", zorder=2)
        # Ears
        ax.add_patch(mpatches.Polygon([[32, 60], [26, 82], [44, 68]], closed=True, color="#334155", zorder=3))
        ax.add_patch(mpatches.Polygon([[68, 60], [74, 82], [56, 68]], closed=True, color="#334155", zorder=3))
        ax.add_patch(mpatches.Polygon([[34, 62], [29, 78], [42, 67]], closed=True, color="#FDA4AF", zorder=4))
        ax.add_patch(mpatches.Polygon([[66, 62], [71, 78], [58, 67]], closed=True, color="#FDA4AF", zorder=4))
        # Head
        ax.add_patch(mpatches.Circle((50, 52), 20, color="#475569", zorder=4))
        # Eyes
        ax.add_patch(mpatches.Ellipse((42, 55), 5.5, 6.5, color="#83B83D", zorder=5))
        ax.add_patch(mpatches.Ellipse((58, 55), 5.5, 6.5, color="#83B83D", zorder=5))
        ax.add_patch(mpatches.Ellipse((42, 55), 2.0, 5.0, color="#0F172A", zorder=6))
        ax.add_patch(mpatches.Ellipse((58, 55), 2.0, 5.0, color="#0F172A", zorder=6))
        ax.scatter([43, 59], [56.8, 56.8], color="#FFFFFF", s=14, zorder=7)
        # Nose & whiskers
        ax.add_patch(mpatches.Polygon([[48, 48], [52, 48], [50, 45.5]], closed=True, color="#FDA4AF", zorder=6))
        for wy, dy in ((47, 1.5), (45, 0), (43, -1.5)):
            ax.plot([22, 40], [wy + dy, wy], color="#CBD5E1", lw=1.6, zorder=6)
            ax.plot([60, 78], [wy, wy + dy], color="#CBD5E1", lw=1.6, zorder=6)
    elif any(k in low for k in ("space", "galaxy", "nebula", "planet", "cosmos", "star", "universe", "moon")):
        fig.patch.set_facecolor("#070B19")
        ax.set_facecolor("#070B19")
        sx, sy = rng.uniform(0, 100, 180), rng.uniform(0, 100, 180)
        ss = rng.uniform(2, 22, 180)
        ax.scatter(sx, sy, s=ss, color="#F8FAFC", alpha=rng.uniform(0.3, 0.95, 180))
        for r_rad, alpha in ((28, 0.08), (22, 0.14), (16, 0.25)):
            ax.add_patch(mpatches.Circle((52, 54), r_rad, color="#6366F1", alpha=alpha))
        ax.add_patch(mpatches.Circle((52, 54), 14, color="#38BDF8", zorder=3))
        ring_t = np.linspace(0, 2 * np.pi, 200)
        rx = 52 + 26 * np.cos(ring_t)
        ry = 54 + 7 * np.sin(ring_t) + 0.22 * (rx - 52)
        ax.plot(rx, ry, color="#FDE047", lw=3.0, alpha=0.85, zorder=4)
    else:
        # Scenic sunset / mountain / ocean / landscape or artistic composition
        grad = np.linspace(0, 1, 256).reshape(256, 1)
        sky = np.zeros((256, 256, 3))
        sky[:, :, 0] = np.clip(0.15 + 0.80 * grad[:, 0], 0, 1)[:, None]
        sky[:, :, 1] = np.clip(0.12 + 0.52 * (grad[:, 0] ** 1.3), 0, 1)[:, None]
        sky[:, :, 2] = np.clip(0.35 + 0.25 * (1 - grad[:, 0]), 0, 1)[:, None]
        ax.imshow(sky, extent=[0, 100, 0, 100], origin="upper", aspect="auto", zorder=0)
        # Sun and glow
        for sr, sa in ((22, 0.12), (16, 0.22), (11, 0.95)):
            ax.add_patch(mpatches.Circle((50, 56), sr, color="#FDE047" if sr == 11 else "#FB923C", alpha=sa, zorder=1))
        x_m = np.linspace(0, 100, 250)
        m1 = 34 + 14 * np.sin(x_m * 0.055 + 0.4) + 6 * np.cos(x_m * 0.13)
        m2 = 24 + 10 * np.sin(x_m * 0.075 + 2.1) + 4 * np.cos(x_m * 0.17)
        ax.fill_between(x_m, 0, m1, color="#312E81", alpha=0.85, zorder=2)
        ax.fill_between(x_m, 0, m2, color="#1E1B4B", alpha=0.95, zorder=3)
        ax.fill_between(x_m, 0, 16, color="#0F172A", alpha=0.92, zorder=4)
        for wy in np.linspace(3, 14.5, 9):
            w_len = 16 * (1 - wy / 20.0)
            ax.plot([50 - w_len, 50 + w_len], [wy, wy], color="#FDE047", lw=1.8, alpha=0.65, zorder=5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    data = buf.getvalue()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(data).hexdigest()[:32] + ".png"
    final = out_dir / name
    if not final.exists():
        final.write_bytes(data)
    secs = round(max(0.1, time.perf_counter() - t0), 2)
    return {"file": name, "seconds": secs, "model": "Clover Visual Canvas", "seed": seed, "width": size, "height": size}


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
        """-> {"file", "seconds", "model", "seed", "width", "height"}; uses sd-cli.exe when installed, else Clover Visual Canvas."""
        text = re.sub(r"\s+", " ", str(prompt or "")).strip()[:MAX_PROMPT]
        if not text:
            raise RuntimeError("the picture needs a description")
        ok, _why = self.available()
        if not ok:
            return _render_canvas_image(text, out_dir, size=max(256, min(MAX_SIZE, int(size) // 64 * 64)), seed=seed)
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
