"""Playwright-Grade Image & UI Visual Inspector for Clover AI Studio (`image_inspector.py`).

Provides deterministic S3 visual inspection for any attached image, screenshot, UI mockup, chart, receipt, or webpage:
1. Exact image geometry (`width`, `height`, `aspect_ratio`, `format`, `mode`, `file_bytes`).
2. Dominant color palette (`hex`, `rgb`, human color name, coverage percentage) and luminance/contrast metrics
   (including WCAG 2.1 contrast ratio checks between background and foreground regions).
3. High-accuracy OCR text & word/line bounding boxes `[x, y, w, h]`:
   - Linux / k3: `/usr/bin/tesseract` TSV (`--psm 6 tsv`) with exact pixel bounding boxes and confidence scores.
   - Windows: native `Windows.Media.Ocr.OcrEngine` WinRT API (or `tesseract` if installed) with line/word bounding boxes.
4. Pure-NumPy + Pillow UI Region & Component Detector (`[ref=e1]`, `[ref=e2]`, ...):
   - Detects interactive UI regions, buttons, cards, badges, headers, input boxes, and text bands using Sobel/gradient
     density and connected-row/column projection profiles.
   - Assigns each element a Playwright-style reference (`e1`, `e2`, ...), role (`header`, `button`, `card`, `text`,
     `panel`, `badge`, `visual`), bounding box `[x, y, w, h]`, spatial zone (`top-left`, `top-bar`, `center`, etc.),
     fill/foreground hex colors, WCAG contrast ratio, and matched OCR text.
5. Annotated Visual Overlay generator (`render_annotated_image`) that draws crisp numbered bounding boxes (`e1`, `e2`, ...)
   onto a copy of the image for visual debugging in chat.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageOps = None  # type: ignore[assignment]

NAMED_COLORS = (
    ("black", (18, 18, 18)),
    ("charcoal", (45, 52, 54)),
    ("slate gray", (100, 116, 139)),
    ("silver", (189, 195, 199)),
    ("white", (248, 249, 250)),
    ("warm cream", (247, 243, 233)),
    ("red", (220, 38, 38)),
    ("coral", (249, 115, 22)),
    ("amber", (245, 158, 11)),
    ("gold", (234, 179, 8)),
    ("emerald green", (16, 185, 129)),
    ("forest green", (23, 107, 44)),
    ("teal", (20, 184, 166)),
    ("sky blue", (14, 165, 233)),
    ("royal blue", (37, 99, 235)),
    ("indigo", (79, 70, 229)),
    ("purple", (147, 51, 234)),
    ("pink", (236, 72, 153)),
    ("brown", (120, 53, 15)),
)


def rgb_to_hex(rgb: tuple[int, int, int] | list[int]) -> str:
    r, g, b = [max(0, min(255, int(c))) for c in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"


def closest_color_name(rgb: tuple[int, int, int] | list[int]) -> str:
    r, g, b = [int(c) for c in rgb[:3]]
    best_name = "neutral"
    best_dist = 1e12
    for name, (cr, cg, cb) in NAMED_COLORS:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def relative_luminance(rgb: tuple[int, int, int] | list[int]) -> float:
    def ch(c: int) -> float:
        v = max(0.0, min(255.0, float(c))) / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:3]
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def wcag_contrast(rgb1: tuple[int, int, int] | list[int], rgb2: tuple[int, int, int] | list[int]) -> float:
    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return round((hi + 0.05) / (lo + 0.05), 2)


def spatial_zone(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    cx = (x + w * 0.5) / max(1.0, float(img_w))
    cy = (y + h * 0.5) / max(1.0, float(img_h))
    if w >= 0.72 * img_w and h <= 0.18 * img_h:
        if cy < 0.2:
            return "top-bar"
        if cy > 0.8:
            return "bottom-bar"
    v_part = "top" if cy < 0.33 else ("bottom" if cy > 0.67 else "middle")
    h_part = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    if v_part == "middle" and h_part == "center":
        return "center"
    if h_part == "center":
        return v_part
    if v_part == "middle":
        return h_part
    return f"{v_part}-{h_part}"


def extract_palette(img_rgb: Image.Image, max_colors: int = 6) -> list[dict[str, Any]]:
    small = img_rgb.copy()
    small.thumbnail((160, 160), Image.Resampling.BILINEAR)
    quant = small.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    pal = quant.getpalette() or []
    counts = sorted(quant.getcolors() or [], key=lambda x: -x[0])
    total = max(1, sum(c for c, _ in counts))
    out: list[dict[str, Any]] = []
    for count, idx in counts[:max_colors]:
        base = idx * 3
        if base + 2 >= len(pal):
            continue
        rgb = (int(pal[base]), int(pal[base + 1]), int(pal[base + 2]))
        pct = round(100.0 * count / total, 1)
        if pct < 1.0 and len(out) >= 3:
            continue
        out.append({
            "hex": rgb_to_hex(rgb),
            "rgb": list(rgb),
            "name": closest_color_name(rgb),
            "pct": pct,
        })
    return out


def _ocr_tesseract(img_path: Path) -> list[dict[str, Any]]:
    exe = shutil.which("tesseract")
    if not exe and os.name == "nt":
        for cand in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.isfile(cand):
                exe = cand
                break
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, str(img_path), "stdout", "--psm", "6", "tsv"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except Exception:
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    lines_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in proc.stdout.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12:
            continue
        try:
            lvl = int(parts[0])
            block_num, par_num, line_num = int(parts[2]), int(parts[3]), int(parts[4])
            left, top, width, height = int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])
            conf = float(parts[10])
        except ValueError:
            continue
        txt = parts[11].strip()
        if lvl != 5 or not txt or conf < 25.0:
            continue
        k = (block_num, par_num, line_num)
        cur = lines_by_key.get(k)
        if cur is None:
            lines_by_key[k] = {
                "x": left,
                "y": top,
                "r": left + width,
                "b": top + height,
                "words": [txt],
                "confs": [conf],
            }
        else:
            cur["x"] = min(cur["x"], left)
            cur["y"] = min(cur["y"], top)
            cur["r"] = max(cur["r"], left + width)
            cur["b"] = max(cur["b"], top + height)
            cur["words"].append(txt)
            cur["confs"].append(conf)
    out: list[dict[str, Any]] = []
    for item in lines_by_key.values():
        text_line = " ".join(item["words"]).strip()
        if not text_line:
            continue
        out.append({
            "text": text_line,
            "bbox": [int(item["x"]), int(item["y"]), int(item["r"] - item["x"]), int(item["b"] - item["y"])],
            "conf": round(sum(item["confs"]) / len(item["confs"]), 1),
        })
    out.sort(key=lambda r: (r["bbox"][1] // 14, r["bbox"][0]))
    return out


def _ocr_windows_winrt(img_path: Path) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    ps_script = (
        "$ErrorActionPreference='Stop'; "
        "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
        "$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]; "
        "$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]; "
        "$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]; "
        "function Await($op, $t) { "
        "  $m = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0].MakeGenericMethod($t); "
        "  $tk = $m.Invoke($null, @($op)); $tk.Wait(); return $tk.Result; "
        "} "
        f"$f = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync({json.dumps(str(img_path.resolve()))})) ([Windows.Storage.StorageFile]); "
        "$s = Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream]); "
        "$dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder]); "
        "$bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap]); "
        "$eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages(); "
        "if (-not $eng) { $eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('en-US'))) }; "
        "$res = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult]); "
        "$out = @(); "
        "foreach ($ln in $res.Lines) { "
        "  $x = 999999; $y = 999999; $r = 0; $b = 0; "
        "  foreach ($w in $ln.Words) { "
        "    $rc = $w.BoundingRect; "
        "    if ($rc.X -lt $x) { $x = [int]$rc.X }; "
        "    if ($rc.Y -lt $y) { $y = [int]$rc.Y }; "
        "    if (($rc.X + $rc.Width) -gt $r) { $r = [int]($rc.X + $rc.Width) }; "
        "    if (($rc.Y + $rc.Height) -gt $b) { $b = [int]($rc.Y + $rc.Height) }; "
        "  } "
        "  $out += @{ text = $ln.Text; bbox = @($x, $y, [Math]::Max(1, $r - $x), [Math]::Max(1, $b - $y)); conf = 95.0 }; "
        "} "
        "$s.Dispose(); "
        "ConvertTo-Json -InputObject @($out) -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            creationflags=0x08000000,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        raw = json.loads(proc.stdout.strip())
        if isinstance(raw, dict):
            raw = [raw]
        return [
            {"text": str(x.get("text") or "").strip(), "bbox": [int(v) for v in (x.get("bbox") or [0, 0, 1, 1])[:4]], "conf": 95.0}
            for x in (raw or [])
            if isinstance(x, dict) and str(x.get("text") or "").strip()
        ]
    except Exception:
        return []


def extract_ocr_lines(img_path: Path) -> list[dict[str, Any]]:
    lines = _ocr_tesseract(img_path)
    if not lines and os.name == "nt":
        lines = _ocr_windows_winrt(img_path)
    return lines


def detect_visual_regions(
    img_rgb: Image.Image,
    ocr_lines: list[dict[str, Any]],
    max_elements: int = 18,
) -> list[dict[str, Any]]:
    """Detect Playwright-style UI & visual elements (`[ref=e1]..[ref=eN]`) with bounding boxes, roles, colors, and contrast."""
    W, H = img_rgb.size
    arr = np.asarray(img_rgb, dtype=np.float32)  # [H, W, 3]
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Compute fast gradient magnitude to find visual boxes, buttons, cards, and UI bands
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = gx + gy

    candidate_boxes: list[dict[str, Any]] = []
    used_ocr: set[int] = set()

    # 1. Group nearby OCR lines into interactive/semantic UI elements (buttons, headings, inputs, labels, paragraphs)
    for idx, ln in enumerate(ocr_lines):
        bx, by, bw, bh = ln["bbox"]
        bx = max(0, min(W - 1, bx))
        by = max(0, min(H - 1, by))
        bw = max(1, min(W - bx, bw))
        bh = max(1, min(H - by, bh))
        pad_x = max(4, min(16, bw // 10))
        pad_y = max(3, min(10, bh // 4))
        rx0, ry0 = max(0, bx - pad_x), max(0, by - pad_y)
        rx1, ry1 = min(W, bx + bw + pad_x), min(H, by + bh + pad_y)
        txt = ln["text"].strip()
        # Infer Playwright-style role from geometry, text, and local border contrast
        role = "text"
        if bh >= max(24, int(H * 0.038)) and len(txt) <= 80:
            role = "heading"
        elif len(txt) <= 32 and bw <= int(W * 0.45) and bh <= int(H * 0.10):
            # Check if surrounded by a button/badge pill or high edge density
            border_grad = float(grad[ry0:ry1, rx0:rx1].mean()) if (ry1 > ry0 and rx1 > rx0) else 0.0
            if border_grad > 14.0 or re.match(
                r"^(?:sign\s*in|log\s*in|submit|send|save|cancel|delete|continue|next|back|search|download|install|open|add|apply|ok|yes|no|buy|start|try|connect|sync|review|confirm)\b",
                txt,
                re.I,
            ):
                role = "button"
            elif re.match(r"^(?:v?\d+\.\d+|beta|new|active|ready|pro|on|off|ok|error|warning)\b", txt, re.I):
                role = "badge"
        if re.search(r"https?://|www\.|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", txt, re.I):
            role = "link"
        candidate_boxes.append({
            "bbox": [bx, by, bw, bh],
            "role": role,
            "text": txt,
            "score": float(bw * bh) + (2500.0 if role in ("heading", "button", "link") else 500.0),
        })
        used_ocr.add(idx)

    # 2. Pure-NumPy connected visual band/box detector (finds headers, cards, panels, charts, or visual focal regions even without OCR)
    grid_h, grid_w = min(64, max(16, H // 8)), min(64, max(16, W // 8))
    small_grad = np.asarray(
        Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8)).resize((grid_w, grid_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    thresh = max(12.0, float(np.percentile(small_grad, 68)))
    active = small_grad >= thresh
    row_energy = active.mean(axis=1)
    in_band = False
    band_start = 0
    for r_i in range(grid_h + 1):
        on = r_i < grid_h and row_energy[r_i] >= 0.18
        if on and not in_band:
            in_band = True
            band_start = r_i
        elif not on and in_band:
            in_band = False
            if r_i - band_start >= 2:
                col_energy = active[band_start:r_i, :].mean(axis=0)
                c_in = False
                c_start = 0
                for c_i in range(grid_w + 1):
                    c_on = c_i < grid_w and col_energy[c_i] >= 0.20
                    if c_on and not c_in:
                        c_in = True
                        c_start = c_i
                    elif not c_on and c_in:
                        c_in = False
                        if c_i - c_start >= 3:
                            x0 = int(round(c_start * W / float(grid_w)))
                            y0 = int(round(band_start * H / float(grid_h)))
                            x1 = int(round(c_i * W / float(grid_w)))
                            y1 = int(round(r_i * H / float(grid_h)))
                            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                            if bw * bh >= 0.012 * W * H:
                                # Check overlap with existing OCR boxes
                                overlaps = any(
                                    not (x1 <= cb["bbox"][0] or x0 >= cb["bbox"][0] + cb["bbox"][2]
                                         or y1 <= cb["bbox"][1] or y0 >= cb["bbox"][1] + cb["bbox"][3])
                                    for cb in candidate_boxes
                                )
                                if not overlaps:
                                    role = "header" if (y0 < 0.15 * H and bw > 0.6 * W) else (
                                        "button" if (bw < 0.35 * W and bh < 0.12 * H) else "visual-region"
                                    )
                                    candidate_boxes.append({
                                        "bbox": [x0, y0, bw, bh],
                                        "role": role,
                                        "text": "",
                                        "score": float(bw * bh) * 0.5,
                                    })

    # If image has no OCR and no high-frequency bands (e.g. smooth photo/illustration), add primary focal quadrants
    if not candidate_boxes:
         candidate_boxes.append({
             "bbox": [int(W * 0.1), int(H * 0.1), int(W * 0.8), int(H * 0.8)],
             "role": "visual-subject",
             "text": "",
             "score": float(W * H),
         })

    # Sort top-to-bottom, left-to-right (Playwright reading order) after keeping top max_elements by importance
    candidate_boxes.sort(key=lambda b: -b["score"])
    selected = candidate_boxes[:max_elements]
    selected.sort(key=lambda b: (b["bbox"][1] // max(12, H // 25), b["bbox"][0]))

    elements: list[dict[str, Any]] = []
    for idx, box in enumerate(selected, start=1):
        x, y, bw, bh = box["bbox"]
        patch = arr[max(0, y) : min(H, y + bh), max(0, x) : min(W, x + bw)]
        if patch.size == 0:
            bg_rgb = (255, 255, 255)
            fg_rgb = (0, 0, 0)
        else:
            flat = patch.reshape(-1, 3)
            lums = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
            lo_mask = lums <= np.percentile(lums, 25)
            hi_mask = lums >= np.percentile(lums, 75)
            dark_rgb = tuple(int(round(v)) for v in (flat[lo_mask].mean(axis=0) if lo_mask.any() else flat.mean(axis=0)))
            light_rgb = tuple(int(round(v)) for v in (flat[hi_mask].mean(axis=0) if hi_mask.any() else flat.mean(axis=0)))
            # Whichever covers the border/majority is background
            if float((lums > np.median(lums)).mean()) >= 0.5:
                bg_rgb, fg_rgb = light_rgb, dark_rgb
            else:
                bg_rgb, fg_rgb = dark_rgb, light_rgb
        contrast = wcag_contrast(bg_rgb, fg_rgb)
        elements.append({
            "ref": f"e{idx}",
            "role": box["role"],
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "zone": spatial_zone(x, y, bw, bh, W, H),
            "text": box["text"][:160],
            "bg_hex": rgb_to_hex(bg_rgb),
            "fg_hex": rgb_to_hex(fg_rgb),
            "contrast": contrast,
            "wcag_aa": contrast >= 4.5,
        })
    return elements


def render_annotated_image(img_rgb: Image.Image, elements: list[dict[str, Any]], out_path: Path) -> Path:
    """Draw Playwright-style numbered bounding boxes (`e1`, `e2`, ...) onto a copy of `img_rgb` and save to `out_path`."""
    canvas = img_rgb.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    ROLE_COLORS = {
        "button": (16, 185, 129, 235),
        "heading": (37, 99, 235, 235),
        "link": (14, 165, 233, 235),
        "badge": (245, 158, 11, 235),
        "header": (147, 51, 234, 235),
    }
    for el in elements:
        x, y, bw, bh = el["bbox"]
        col = ROLE_COLORS.get(el["role"], (23, 107, 44, 220))
        fill_col = (col[0], col[1], col[2], 28)
        draw.rectangle([x, y, x + bw, y + bh], fill=fill_col, outline=col, width=2)
        tag = el["ref"]
        tw = max(18, 7 * len(tag) + 6)
        th = 15
        ty = max(0, y - th)
        draw.rectangle([x, ty, x + tw, ty + th], fill=col)
        draw.text((x + 3, ty + 1), tag, fill=(255, 255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def inspect_image(
    img_path: str | Path,
    *,
    annotate_dir: Path | None = None,
    crop_bbox: list[int] | tuple[int, int, int, int] | None = None,
    max_elements: int = 18,
    jepa_arch: str = "auto",
    skip_ocr: bool = False,
) -> dict[str, Any]:
    """Run full Playwright-style structured visual, layout, color, OCR, and JEPA visual latent inspection on `img_path`."""
    t0 = time.perf_counter()
    p = Path(img_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    cache_path = p.with_suffix(p.suffix + ".inspect.json")
    if crop_bbox is None and jepa_arch != "none" and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("width") and "elements" in cached and "jepa_visual" in cached:
                ann_ok = (
                    annotate_dir is None
                    or not cached.get("elements")
                    or (bool(cached.get("annotated_file")) and (Path(annotate_dir) / str(cached["annotated_file"])).is_file())
                )
                if ann_ok:
                    return cached
        except Exception:
            pass

    with Image.open(p) as raw_im:
        fmt = raw_im.format or p.suffix.lstrip(".").upper() or "PNG"
        orig_mode = raw_im.mode
        im = ImageOps.exif_transpose(raw_im).convert("RGB")

    if crop_bbox is not None and len(crop_bbox) == 4:
        cx, cy, cw, ch = [int(v) for v in crop_bbox]
        cx = max(0, min(im.width - 1, cx))
        cy = max(0, min(im.height - 1, cy))
        cw = max(1, min(im.width - cx, cw))
        ch = max(1, min(im.height - cy, ch))
        im = im.crop((cx, cy, cx + cw, cy + ch))
        crop_tmp = p.parent / f"{p.stem}_crop_{cx}_{cy}_{cw}_{ch}.png"
        im.save(crop_tmp, format="PNG")
        ocr_target = crop_tmp
    else:
        ocr_target = p

    W, H = im.size
    gcd_v = math.gcd(W, H) or 1
    aspect = f"{W // gcd_v}:{H // gcd_v}" if (W // gcd_v <= 64 and H // gcd_v <= 64) else f"{W / max(1.0, float(H)):.2f}:1"

    palette = extract_palette(im, max_colors=6)
    ocr_lines = [] if skip_ocr else extract_ocr_lines(ocr_target)
    elements = detect_visual_regions(im, ocr_lines, max_elements=max_elements)
    full_text = "\n".join(ln["text"] for ln in ocr_lines).strip()

    jepa_vis_summary = None
    try:
        import jepa_engine
        jv = jepa_engine.for_base().inspect_visual_latent(np.asarray(im), elements=elements, configured_arch=jepa_arch)
        if jv.get("enabled"):
            jepa_vis_summary = {
                "arch": jv["arch"],
                "scene_kind": jv["scene_kind"],
                "attn_flops_ratio": jv["attn_flops_ratio"],
                "mean_patch_surprise": jv["mean_patch_surprise"],
                "top_salient_elements": jv["top_salient_elements"],
                "z_jepa": [round(float(v), 5) for v in jv["z_jepa"].tolist()] if "z_jepa" in jv else None,
                "ms": jv["ms"],
            }
    except Exception:
        jepa_vis_summary = None

    annotated_file = None
    if annotate_dir is not None and elements:
        h_tag = hashlib.sha256(f"{p.name}:{W}x{H}:{len(elements)}".encode("utf-8")).hexdigest()[:16]
        ann_path = Path(annotate_dir) / f"annotated_{h_tag}.png"
        render_annotated_image(im, elements, ann_path)
        annotated_file = ann_path.name

    low_contrast = [el["ref"] for el in elements if el.get("text") and not el.get("wcag_aa", True)]
    ms = round((time.perf_counter() - t0) * 1000.0, 1)

    report = {
        "file": p.name,
        "path": str(p),
        "format": fmt,
        "mode": orig_mode,
        "width": W,
        "height": H,
        "aspect_ratio": aspect,
        "file_bytes": p.stat().st_size,
        "palette": palette,
        "ocr_lines": ocr_lines,
        "ocr_text": full_text,
        "elements": elements,
        "jepa_visual": jepa_vis_summary,
        "low_contrast_refs": low_contrast,
        "annotated_file": annotated_file,
        "ms": ms,
    }
    if crop_bbox is None and jepa_arch != "none":
        try:
            cache_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return report


def format_snapshot_for_prompt(report: dict[str, Any], max_chars: int = 3200) -> str:
    """Format `report` as a concise Playwright-style accessibility/visual snapshot for C1 (`LFM3-Omni-2B`) and S3 tools."""
    W, H = report.get("width", 0), report.get("height", 0)
    fmt = report.get("format", "IMAGE")
    aspect = report.get("aspect_ratio", "")
    pal = ", ".join(f"{c['hex']} ({c['name']} {c['pct']}%)" for c in (report.get("palette") or [])[:5])
    lines = [
        f"Playwright-Style Visual & UI Snapshot ({report.get('file', 'image')}: {W}x{H} px, {fmt}, aspect {aspect}):",
        f"- Dominant Palette: {pal or 'n/a'}",
    ]
    jv = report.get("jepa_visual")
    if isinstance(jv, dict) and jv.get("arch"):
        top_s = ", ".join(f"{x['ref']} ({x['role']} sal={x['salience']})" for x in (jv.get("top_salient_elements") or [])[:4])
        lines.append(
            f"- JEPA Visual Latent Prediction (arch={jv['arch']}, scene={jv.get('scene_kind')}, "
            f"sparse_flops={jv.get('attn_flops_ratio')}x, focal={top_s or 'n/a'})"
        )
    elements = report.get("elements") or []
    if elements:
        lines.append(f"- Detected Visual / UI Elements ({len(elements)} regions, top-to-bottom):")
        for el in elements[:16]:
            bx, by, bw, bh = el["bbox"]
            txt_part = f' "{el["text"]}"' if el.get("text") else ""
            wcag = f", contrast {el['contrast']}:1" + (" [LOW CONTRAST < 4.5:1]" if el.get("text") and not el.get("wcag_aa") else "")
            lines.append(
                f"  [{el['ref']}] {el['role']} ({el['zone']}, bbox=[{bx},{by},{bw},{bh}], bg={el['bg_hex']}, fg={el['fg_hex']}{wcag}){txt_part}"
            )
    ocr_txt = (report.get("ocr_text") or "").strip()
    if ocr_txt:
        lines.append("- Extracted OCR Text:")
        for ln in ocr_txt.splitlines()[:28]:
            lines.append("  " + ln)
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 3] + "..."
