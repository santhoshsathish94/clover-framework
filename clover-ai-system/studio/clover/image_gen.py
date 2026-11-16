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
    """On-device Matplotlib / NumPy compositional scene & object renderer used on Android, Web, or laptops where the
    2 GB SD 1.5 checkpoint is not installed. Parses subject, environment, lighting, and palette from the prompt."""
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

    is_night = any(k in low for k in ("night", "midnight", "moon", "starry", "dark", "neon", "cyberpunk"))
    is_sunset = any(k in low for k in ("sunset", "sunrise", "dusk", "dawn", "golden hour", "evening"))
    is_minimal = any(k in low for k in ("minimal", "minimalist", "minimalistic", "simple", "icon", "logo", "flat"))

    def _draw_sky_and_ground(ground_y=24, meadow=True):
        grad = np.linspace(0, 1, 256).reshape(256, 1)
        sky = np.zeros((256, 256, 3))
        if is_night:
            sky[:, :, 0] = np.clip(0.03 + 0.08 * grad[:, 0], 0, 1)[:, None]
            sky[:, :, 1] = np.clip(0.05 + 0.12 * grad[:, 0], 0, 1)[:, None]
            sky[:, :, 2] = np.clip(0.12 + 0.25 * grad[:, 0], 0, 1)[:, None]
            ax.imshow(sky, extent=[0, 100, 0, 100], origin="upper", aspect="auto", zorder=0)
            sx, sy = rng.uniform(4, 96, 90), rng.uniform(ground_y + 10, 96, 90)
            ax.scatter(sx, sy, s=rng.uniform(2, 14, 90), color="#F8FAFC", alpha=rng.uniform(0.35, 0.95, 90), zorder=1)
            ax.add_patch(mpatches.Circle((80, 82), 7.5, color="#FEF08A", alpha=0.92, zorder=1))
        elif is_sunset:
            sky[:, :, 0] = np.clip(0.22 + 0.75 * grad[:, 0], 0, 1)[:, None]
            sky[:, :, 1] = np.clip(0.14 + 0.50 * (grad[:, 0] ** 1.2), 0, 1)[:, None]
            sky[:, :, 2] = np.clip(0.38 + 0.22 * (1 - grad[:, 0]), 0, 1)[:, None]
            ax.imshow(sky, extent=[0, 100, 0, 100], origin="upper", aspect="auto", zorder=0)
            for sr, sa in ((18, 0.14), (12, 0.28), (8, 0.92)):
                ax.add_patch(mpatches.Circle((74, 68), sr, color="#FDE047" if sr == 8 else "#FB923C", alpha=sa, zorder=1))
        elif is_minimal:
            fig.patch.set_facecolor("#F6F5EE")
            ax.set_facecolor("#F6F5EE")
            ax.add_patch(mpatches.Rectangle((0, 0), 100, 100, color="#F6F5EE", zorder=0))
            ax.add_patch(mpatches.Circle((76, 78), 11, color="#FEF3C7", alpha=0.85, zorder=1))
            ax.add_patch(mpatches.Circle((76, 78), 6.5, color="#FDE68A", alpha=0.95, zorder=1))
        else:
            sky[:, :, 0] = np.clip(0.52 + 0.36 * grad[:, 0], 0, 1)[:, None]
            sky[:, :, 1] = np.clip(0.76 + 0.20 * grad[:, 0], 0, 1)[:, None]
            sky[:, :, 2] = np.clip(0.94 + 0.05 * grad[:, 0], 0, 1)[:, None]
            ax.imshow(sky, extent=[0, 100, 0, 100], origin="upper", aspect="auto", zorder=0)
            for sr, sa in ((16, 0.15), (10, 0.30), (6.5, 0.95)):
                ax.add_patch(mpatches.Circle((78, 80), sr, color="#FEF08A" if sr == 6.5 else "#FDE047", alpha=sa, zorder=1))
        if meadow:
            x_g = np.linspace(0, 100, 200)
            h1 = ground_y + 3.5 * np.sin(x_g * 0.06 + rng.uniform(0, 2))
            h2 = (ground_y - 4) + 2.5 * np.cos(x_g * 0.08 + rng.uniform(0, 2))
            c_back = "#1E3A2F" if is_night else ("#84A98C" if is_minimal else "#4ADE80")
            c_front = "#14281D" if is_night else ("#52796F" if is_minimal else "#22C55E")
            ax.fill_between(x_g, 0, h1, color=c_back, alpha=0.85, zorder=2)
            ax.fill_between(x_g, 0, h2, color=c_front, alpha=0.95, zorder=2)

    if any(k in low for k in ("tree", "oak", "maple", "pine", "evergreen", "forest", "woods", "bonsai", "sakura", "cherry blossom", "willow")):
        _draw_sky_and_ground(ground_y=22, meadow=True)
        is_pine = any(k in low for k in ("pine", "evergreen", "spruce", "fir", "christmas"))
        is_sakura = any(k in low for k in ("sakura", "cherry blossom", "pink", "blossom"))
        is_autumn = any(k in low for k in ("autumn", "fall", "golden", "orange", "red leaves"))
        # Sunbeams filtering through branches
        if not is_night:
            for ang in (-18, -6, 6, 18):
                rad = np.deg2rad(ang)
                bx = [50, 50 + 38 * np.sin(rad) - 6, 50 + 38 * np.sin(rad) + 6]
                by = [66, 18, 18]
                ax.add_patch(mpatches.Polygon(list(zip(bx, by)), closed=True, color="#FEF08A", alpha=0.14, zorder=2.5))
        # Shadow under tree
        ax.add_patch(mpatches.Ellipse((50, 18), 38, 6, color="#1B4332", alpha=0.32, zorder=2.8))
        if is_pine:
            ax.add_patch(mpatches.Rectangle((46.5, 16), 7, 18, color="#5C3A21", zorder=3))
            for idx_t, (yb, yt, hw) in enumerate(((26, 52, 24), (40, 68, 19), (55, 82, 13), (68, 91, 8))):
                col_p = ("#1B4332", "#2D6A4F", "#40916C", "#52B788")[idx_t % 4]
                ax.add_patch(mpatches.Polygon([[50 - hw, yb], [50 + hw, yb], [50, yt]], closed=True, color=col_p, zorder=4 + idx_t))
        else:
            # Sculpted trunk, roots, and spreading branches
            trunk_pts = [[41, 16], [45, 24], [46.5, 46], [53.5, 46], [55, 24], [59, 16]]
            ax.add_patch(mpatches.Polygon(trunk_pts, closed=True, color="#5C3A21", zorder=3))
            ax.add_patch(mpatches.Polygon([[47.5, 17], [48.5, 45], [50.5, 45], [50, 17]], closed=True, color="#7C4D2B", alpha=0.7, zorder=3.2))
            for b_pts, lw_b in (
                (([49, 42], [35, 56]), 6.5),
                (([51, 42], [65, 56]), 6.5),
                (([50, 45], [44, 64]), 5.5),
                (([50, 45], [57, 64]), 5.5),
                (([39, 51], [28, 60]), 4.0),
                (([61, 51], [72, 60]), 4.0),
            ):
                ax.plot([b_pts[0][0], b_pts[1][0]], [b_pts[0][1], b_pts[1][1]], color="#5C3A21", lw=lw_b, solid_capstyle="round", zorder=3.5)
            # Multi-layered leaf canopy
            if is_sakura:
                palette = ("#F472B6", "#F9A8D4", "#FBCFE8", "#FCE7F3")
            elif is_autumn:
                palette = ("#B45309", "#D97706", "#F59E0B", "#FBBF24")
            else:
                palette = ("#1B4332", "#2D6A4F", "#40916C", "#52B788", "#74C69D")
            canopy_blobs = [
                (50, 66, 22, 18, palette[0]),
                (34, 58, 15, 13, palette[0]),
                (66, 58, 15, 13, palette[0]),
                (40, 72, 16, 14, palette[1]),
                (60, 72, 16, 14, palette[1]),
                (50, 78, 17, 14, palette[1]),
                (31, 63, 12, 11, palette[2]),
                (69, 63, 12, 11, palette[2]),
                (45, 63, 15, 13, palette[2]),
                (56, 64, 15, 13, palette[2]),
                (50, 71, 16, 13, palette[3]),
                (42, 76, 11, 9, palette[-1]),
                (58, 75, 11, 9, palette[-1]),
                (50, 81, 11, 8, palette[-1]),
            ]
            for cx, cy, rx, ry, col_c in canopy_blobs:
                jx = cx + rng.uniform(-1.5, 1.5)
                jy = cy + rng.uniform(-1.2, 1.2)
                ax.add_patch(mpatches.Ellipse((jx, jy), rx * 2, ry * 2, color=col_c, alpha=0.94, zorder=4))
            # Dappled sunlit leaf highlights
            for _ in range(28):
                ang = rng.uniform(0, 2 * np.pi)
                rad = rng.uniform(2, 22)
                lx = 50 + rad * np.cos(ang) * 1.15
                ly = 66 + rad * np.sin(ang) * 0.85
                ax.add_patch(mpatches.Circle((lx, ly), rng.uniform(1.2, 2.6), color="#D8F3DC" if not is_sakura else "#FFF1F2", alpha=0.55, zorder=5))
    elif any(k in low for k in ("flower", "rose", "sunflower", "lotus", "tulip", "blossom", "bouquet", "daisy")):
        _draw_sky_and_ground(ground_y=22, meadow=True)
        ax.plot([50, 50], [18, 56], color="#15803D", lw=6, solid_capstyle="round", zorder=3)
        ax.add_patch(mpatches.Ellipse((41, 36), 14, 6, angle=28, color="#22C55E", zorder=3))
        ax.add_patch(mpatches.Ellipse((59, 42), 14, 6, angle=-28, color="#22C55E", zorder=3))
        p_col = "#F43F5E" if "rose" in low or "red" in low else ("#FACC15" if "sunflower" in low or "yellow" in low else "#EC4899")
        for a_deg in range(0, 360, 30):
            rad = np.deg2rad(a_deg)
            px, py = 50 + 13 * np.cos(rad), 62 + 13 * np.sin(rad)
            ax.add_patch(mpatches.Ellipse((px, py), 14, 8, angle=a_deg, color=p_col, alpha=0.92, zorder=4))
        ax.add_patch(mpatches.Circle((50, 62), 7.5, color="#F59E0B", zorder=5))
        ax.add_patch(mpatches.Circle((50, 62), 4.5, color="#78350F", zorder=6))
    elif any(k in low for k in ("dog", "puppy", "retriever", "hound", "husky", "corgi", "canine")):
        _draw_sky_and_ground(ground_y=24, meadow=True)
        fur, ear_c = ("#D97706", "#92400E") if "black" not in low else ("#334155", "#1E293B")
        ax.add_patch(mpatches.Ellipse((48, 34), 34, 22, color=fur, zorder=3))
        for lx in (36, 43, 55, 61):
            ax.add_patch(mpatches.FancyBboxPatch((lx - 2.5, 16), 5, 16, boxstyle="round,pad=1.2", color=fur, zorder=3))
        t_th = np.linspace(0, 1.0, 40)
        ax.plot(31 - 10 * t_th, 38 + 12 * (t_th ** 1.4), color=fur, lw=6, solid_capstyle="round", zorder=3)
        ax.add_patch(mpatches.Circle((62, 54), 14, color=fur, zorder=4))
        ax.add_patch(mpatches.Ellipse((52, 54), 8, 16, angle=18, color=ear_c, zorder=5))
        ax.add_patch(mpatches.Ellipse((72, 54), 8, 16, angle=-18, color=ear_c, zorder=5))
        ax.add_patch(mpatches.Ellipse((66, 49), 12, 9, color="#FDE68A", zorder=5))
        ax.add_patch(mpatches.Circle((70, 51), 2.8, color="#0F172A", zorder=6))
        ax.scatter([59, 66], [57, 57], color="#0F172A", s=42, zorder=6)
        ax.scatter([59.6, 66.6], [57.8, 57.8], color="#FFFFFF", s=10, zorder=7)
        ax.add_patch(mpatches.Ellipse((67, 45.2), 3.6, 4.5, color="#FB7185", zorder=5))
    elif any(k in low for k in ("cat", "kitty", "kitten", "feline")):
        fig.patch.set_facecolor("#F7F3E9")
        ax.set_facecolor("#F7F3E9")
        ax.add_patch(mpatches.Circle((50, 48), 34, color="#E3EEDA", zorder=1))
        ax.add_patch(mpatches.Ellipse((50, 26), 34, 26, color="#334155", zorder=2))
        t_theta = np.linspace(0, 1.3 * np.pi, 80)
        ax.plot(64 + 12 * np.sin(t_theta), 22 + 10 * (1 - np.cos(t_theta)), color="#334155", lw=7, solid_capstyle="round", zorder=2)
        ax.add_patch(mpatches.Polygon([[32, 60], [26, 82], [44, 68]], closed=True, color="#334155", zorder=3))
        ax.add_patch(mpatches.Polygon([[68, 60], [74, 82], [56, 68]], closed=True, color="#334155", zorder=3))
        ax.add_patch(mpatches.Polygon([[34, 62], [29, 78], [42, 67]], closed=True, color="#FDA4AF", zorder=4))
        ax.add_patch(mpatches.Polygon([[66, 62], [71, 78], [58, 67]], closed=True, color="#FDA4AF", zorder=4))
        ax.add_patch(mpatches.Circle((50, 52), 20, color="#475569", zorder=4))
        ax.add_patch(mpatches.Ellipse((42, 55), 5.5, 6.5, color="#83B83D", zorder=5))
        ax.add_patch(mpatches.Ellipse((58, 55), 5.5, 6.5, color="#83B83D", zorder=5))
        ax.add_patch(mpatches.Ellipse((42, 55), 2.0, 5.0, color="#0F172A", zorder=6))
        ax.add_patch(mpatches.Ellipse((58, 55), 2.0, 5.0, color="#0F172A", zorder=6))
        ax.scatter([43, 59], [56.8, 56.8], color="#FFFFFF", s=14, zorder=7)
        ax.add_patch(mpatches.Polygon([[48, 48], [52, 48], [50, 45.5]], closed=True, color="#FDA4AF", zorder=6))
        for wy, dy in ((47, 1.5), (45, 0), (43, -1.5)):
            ax.plot([22, 40], [wy + dy, wy], color="#CBD5E1", lw=1.6, zorder=6)
            ax.plot([60, 78], [wy, wy + dy], color="#CBD5E1", lw=1.6, zorder=6)
    elif any(k in low for k in ("bird", "eagle", "owl", "parrot", "swan", "dove", "sparrow")):
        _draw_sky_and_ground(ground_y=20, meadow=True)
        ax.plot([15, 85], [38, 42], color="#5C3A21", lw=5.5, solid_capstyle="round", zorder=3)
        ax.add_patch(mpatches.Ellipse((50, 50), 18, 22, color="#0284C7", zorder=4))
        ax.add_patch(mpatches.Ellipse((50, 47), 12, 16, color="#E0F2FE", zorder=4.5))
        ax.add_patch(mpatches.Circle((52, 64), 9, color="#0369A1", zorder=5))
        ax.add_patch(mpatches.Polygon([[59, 65], [68, 63], [59, 61]], closed=True, color="#F59E0B", zorder=6))
        ax.scatter([54], [66], color="#0F172A", s=36, zorder=6)
        ax.scatter([54.5], [66.6], color="#FFFFFF", s=10, zorder=7)
    elif any(k in low for k in ("house", "home", "cottage", "cabin", "building", "castle", "villa")):
        _draw_sky_and_ground(ground_y=24, meadow=True)
        ax.add_patch(mpatches.Rectangle((28, 22), 44, 32, color="#FDE68A", ec="#B45309", lw=1.5, zorder=3))
        ax.add_patch(mpatches.Rectangle((60, 50), 7, 16, color="#991B1B", zorder=2.8))
        ax.add_patch(mpatches.Polygon([[22, 54], [78, 54], [50, 76]], closed=True, color="#DC2626", ec="#7F1D1D", lw=1.5, zorder=4))
        ax.add_patch(mpatches.Rectangle((45, 22), 10, 18, color="#78350F", zorder=4))
        ax.add_patch(mpatches.Circle((53, 31), 0.9, color="#FACC15", zorder=5))
        for wx in (33, 59):
            ax.add_patch(mpatches.Rectangle((wx, 34), 8, 9, color="#FEF08A" if is_night else "#BAE6FD", ec="#78350F", lw=1.2, zorder=4))
    elif any(k in low for k in ("city", "skyline", "skyscraper", "metropolis", "cyberpunk", "futuristic", "downtown")):
        is_night = True
        _draw_sky_and_ground(ground_y=16, meadow=False)
        ax.add_patch(mpatches.Rectangle((0, 0), 100, 18, color="#090D16", zorder=2))
        buildings = [(6, 14, 52), (22, 16, 68), (40, 18, 78), (60, 15, 62), (77, 17, 56)]
        for bx, bw, bh in buildings:
            ax.add_patch(mpatches.Rectangle((bx, 16), bw, bh, color="#1E293B", ec="#38BDF8", lw=0.8, zorder=3))
            for wy in range(22, 16 + bh - 4, 6):
                for wx in range(bx + 2, bx + bw - 2, 4):
                    if rng.random() > 0.25:
                        w_col = "#22D3EE" if rng.random() > 0.4 else ("#F472B6" if "neon" in low or "cyberpunk" in low else "#FDE047")
                        ax.add_patch(mpatches.Rectangle((wx, wy), 2.2, 3.2, color=w_col, alpha=0.88, zorder=4))
    elif any(k in low for k in ("car", "vehicle", "sports car", "automobile", "sedan", "truck", "bus")):
        _draw_sky_and_ground(ground_y=26, meadow=True)
        ax.add_patch(mpatches.Rectangle((0, 10), 100, 16, color="#334155", zorder=2.5))
        for dx in range(6, 96, 18):
            ax.add_patch(mpatches.Rectangle((dx, 17.5), 10, 1.5, color="#FACC15", zorder=2.6))
        c_col = "#EF4444" if "blue" not in low else "#3B82F6"
        ax.add_patch(mpatches.FancyBboxPatch((20, 22), 60, 14, boxstyle="round,pad=2", color=c_col, zorder=4))
        ax.add_patch(mpatches.Polygon([[30, 36], [42, 48], [64, 48], [74, 36]], closed=True, color=c_col, zorder=4))
        ax.add_patch(mpatches.Polygon([[34, 37], [43, 46], [62, 46], [70, 37]], closed=True, color="#BAE6FD", zorder=4.5))
        for wx in (32, 68):
            ax.add_patch(mpatches.Circle((wx, 21), 8, color="#0F172A", zorder=5))
            ax.add_patch(mpatches.Circle((wx, 21), 4, color="#94A3B8", zorder=6))
    elif any(k in low for k in ("robot", "android", "bot", "cyborg", "mech")):
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")
        ax.add_patch(mpatches.Circle((50, 50), 38, color="#1E293B", zorder=1))
        ax.plot([50, 50], [74, 86], color="#94A3B8", lw=3.5, zorder=3)
        ax.add_patch(mpatches.Circle((50, 87), 3.2, color="#38BDF8", zorder=4))
        ax.add_patch(mpatches.FancyBboxPatch((32, 48), 36, 24, boxstyle="round,pad=3", color="#CBD5E1", ec="#475569", lw=2, zorder=4))
        ax.add_patch(mpatches.FancyBboxPatch((36, 55), 28, 11, boxstyle="round,pad=1.5", color="#0F172A", zorder=5))
        ax.add_patch(mpatches.Circle((43, 60.5), 3.5, color="#38BDF8", zorder=6))
        ax.add_patch(mpatches.Circle((57, 60.5), 3.5, color="#38BDF8", zorder=6))
        ax.add_patch(mpatches.FancyBboxPatch((30, 18), 40, 26, boxstyle="round,pad=2.5", color="#94A3B8", ec="#475569", lw=2, zorder=3))
        ax.add_patch(mpatches.Circle((50, 31), 6, color="#22D3EE", alpha=0.85, zorder=4))
    elif any(k in low for k in ("dna", "helix", "gene", "chromosome", "rna", "molecule", "biology")):
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
    elif any(k in low for k in ("space", "galaxy", "nebula", "planet", "cosmos", "star", "universe", "moon", "rocket", "astronaut")):
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
    elif any(k in low for k in ("heart", "love", "valentine")):
        fig.patch.set_facecolor("#FFF1F2")
        ax.set_facecolor("#FFF1F2")
        t_h = np.linspace(0, 2 * np.pi, 300)
        hx = 50 + 2.1 * (16 * (np.sin(t_h) ** 3))
        hy = 52 + 2.1 * (13 * np.cos(t_h) - 5 * np.cos(2 * t_h) - 2 * np.cos(3 * t_h) - np.cos(4 * t_h))
        ax.fill(hx, hy, color="#E11D48", alpha=0.95, zorder=3)
    elif any(k in low for k in ("coffee", "tea", "cup", "mug", "apple", "fruit", "cake", "pizza", "food")):
        fig.patch.set_facecolor("#FDF8F5")
        ax.set_facecolor("#FDF8F5")
        ax.add_patch(mpatches.Rectangle((0, 0), 100, 28, color="#D6C7B2", zorder=1))
        if "apple" in low or "fruit" in low:
            ax.add_patch(mpatches.Circle((45, 44), 16, color="#DC2626", zorder=3))
            ax.add_patch(mpatches.Circle((55, 44), 16, color="#EF4444", zorder=3))
            ax.plot([50, 52], [58, 67], color="#78350F", lw=4, solid_capstyle="round", zorder=4)
            ax.add_patch(mpatches.Ellipse((57, 65), 11, 5, angle=25, color="#22C55E", zorder=5))
        else:
            ax.add_patch(mpatches.Ellipse((50, 24), 46, 8, color="#E2E8F0", zorder=2))
            ax.add_patch(mpatches.FancyBboxPatch((32, 26), 34, 28, boxstyle="round,pad=2", color="#F8FAFC", ec="#94A3B8", lw=2, zorder=3))
            ax.add_patch(mpatches.Arc((66, 40), 14, 16, theta1=-90, theta2=90, color="#94A3B8", lw=4, zorder=2.5))
            sy = np.linspace(58, 82, 80)
            for sx_off in (-6, 0, 6):
                ax.plot(49 + sx_off + 2.5 * np.sin((sy - 58) * 0.25), sy, color="#CBD5E1", lw=2.5, alpha=0.75, zorder=4)
    else:
        # Scenic landscape with mountains, lake, sun, and lakeside trees
        grad = np.linspace(0, 1, 256).reshape(256, 1)
        sky = np.zeros((256, 256, 3))
        sky[:, :, 0] = np.clip(0.15 + 0.80 * grad[:, 0], 0, 1)[:, None]
        sky[:, :, 1] = np.clip(0.12 + 0.52 * (grad[:, 0] ** 1.3), 0, 1)[:, None]
        sky[:, :, 2] = np.clip(0.35 + 0.25 * (1 - grad[:, 0]), 0, 1)[:, None]
        ax.imshow(sky, extent=[0, 100, 0, 100], origin="upper", aspect="auto", zorder=0)
        for sr, sa in ((22, 0.12), (16, 0.22), (11, 0.95)):
            ax.add_patch(mpatches.Circle((50, 56), sr, color="#FDE047" if sr == 11 else "#FB923C", alpha=sa, zorder=1))
        x_m = np.linspace(0, 100, 250)
        m1 = 34 + 14 * np.sin(x_m * 0.055 + rng.uniform(0, 1.5)) + 6 * np.cos(x_m * 0.13)
        m2 = 24 + 10 * np.sin(x_m * 0.075 + rng.uniform(1.5, 3.0)) + 4 * np.cos(x_m * 0.17)
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
