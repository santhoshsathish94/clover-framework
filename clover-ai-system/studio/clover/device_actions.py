"""Cross-Platform Device & Activity Assistant for Clover AI Studio (`device_actions.py`).

Enables Clover to:
1. Open other applications, websites, files, and folders on Windows, Android, and Linux/Web (`open_target`),
   returning both the execution status and clickable action links (`app://...` / `https://...`) for the UI.
2. Capture live screenshots (`capture_screenshot`) and run Playwright-style Visual & OCR Inspection (`image_inspector`)
   so Clover can see and read what is on the user's screen.
3. Read active application window titles and clipboard text (`read_active_activity`, with secret redaction via
   `clover_repo.redact`) so Clover can help the user with their live activity across apps.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import clover_repo
import image_inspector
import paths

# Known cross-platform app targets:
#   key -> {"win": Windows exe/URI, "android_pkg": Android package, "android_uri": Android intent URI, "web": Web URL, "label": Display name}
KNOWN_APPS: dict[str, dict[str, str]] = {
    "calculator": {"win": "calc.exe", "android_pkg": "com.google.android.calculator", "android_uri": " app://calculator", "label": "Calculator"},
    "calc": {"win": "calc.exe", "android_pkg": "com.google.android.calculator", "label": "Calculator"},
    "notepad": {"win": "notepad.exe", "android_pkg": "com.samsung.android.app.notes", "label": "Notepad / Notes"},
    "notes": {"win": "notepad.exe", "android_pkg": "com.samsung.android.app.notes", "label": "Notes"},
    "paint": {"win": "mspaint.exe", "label": "Paint"},
    "snipping tool": {"win": "ms-screenclip:", "label": "Snipping Tool"},
    "settings": {"win": "ms-settings:", "android_uri": "android.settings.SETTINGS", "label": "Settings"},
    "wifi settings": {"win": "ms-settings:network-wifi", "android_uri": "android.settings.WIFI_SETTINGS", "label": "Wi-Fi Settings"},
    "bluetooth settings": {"win": "ms-settings:bluetooth", "android_uri": "android.settings.BLUETOOTH_SETTINGS", "label": "Bluetooth Settings"},
    "explorer": {"win": "explorer.exe", "android_pkg": "com.sec.android.app.myfiles", "label": "File Explorer"},
    "file explorer": {"win": "explorer.exe", "android_pkg": "com.sec.android.app.myfiles", "label": "File Explorer"},
    "files": {"win": "explorer.exe", "android_pkg": "com.google.android.apps.nbu.files", "label": "Files"},
    "task manager": {"win": "taskmgr.exe", "label": "Task Manager"},
    "terminal": {"win": "wt.exe", "label": "Terminal"},
    "powershell": {"win": "powershell.exe", "label": "PowerShell"},
    "command prompt": {"win": "cmd.exe", "label": "Command Prompt"},
    "cmd": {"win": "cmd.exe", "label": "Command Prompt"},
    "chrome": {"win": "chrome.exe", "android_pkg": "com.android.chrome", "web": "https://www.google.com", "label": "Google Chrome"},
    "google chrome": {"win": "chrome.exe", "android_pkg": "com.android.chrome", "web": "https://www.google.com", "label": "Google Chrome"},
    "browser": {"win": "https://www.google.com", "android_pkg": "com.android.chrome", "web": "https://www.google.com", "label": "Web Browser"},
    "edge": {"win": "msedge.exe", "android_pkg": "com.microsoft.emmx", "web": "https://www.bing.com", "label": "Microsoft Edge"},
    "microsoft edge": {"win": "msedge.exe", "android_pkg": "com.microsoft.emmx", "web": "https://www.bing.com", "label": "Microsoft Edge"},
    "firefox": {"win": "firefox.exe", "android_pkg": "org.mozilla.firefox", "label": "Firefox"},
    "brave": {"win": "brave.exe", "android_pkg": "com.brave.browser", "label": "Brave Browser"},
    "vscode": {"win": "code", "web": "https://vscode.dev", "label": "VS Code"},
    "vs code": {"win": "code", "web": "https://vscode.dev", "label": "VS Code"},
    "visual studio code": {"win": "code", "web": "https://vscode.dev", "label": "VS Code"},
    "word": {"win": "winword.exe", "android_pkg": "com.microsoft.office.word", "web": "https://office.live.com/start/Word.aspx", "label": "Microsoft Word"},
    "excel": {"win": "excel.exe", "android_pkg": "com.microsoft.office.excel", "web": "https://office.live.com/start/Excel.aspx", "label": "Microsoft Excel"},
    "powerpoint": {"win": "powerpnt.exe", "android_pkg": "com.microsoft.office.powerpoint", "web": "https://office.live.com/start/PowerPoint.aspx", "label": "PowerPoint"},
    "outlook": {"win": "outlook.exe", "android_pkg": "com.microsoft.office.outlook", "web": "https://outlook.live.com", "label": "Outlook"},
    "teams": {"win": "msteams:", "android_pkg": "com.microsoft.teams", "web": "https://teams.microsoft.com", "label": "Microsoft Teams"},
    "slack": {"win": "slack:", "android_pkg": "com.Slack", "web": "https://app.slack.com", "label": "Slack"},
    "spotify": {"win": "spotify:", "android_pkg": "com.spotify.music", "web": "https://open.spotify.com", "label": "Spotify"},
    "discord": {"win": "discord:", "android_pkg": "com.discord", "web": "https://discord.com/app", "label": "Discord"},
    "whatsapp": {"win": "whatsapp:", "android_pkg": "com.whatsapp", "web": "https://web.whatsapp.com", "label": "WhatsApp"},
    "telegram": {"win": "tg:", "android_pkg": "org.telegram.messenger", "web": "https://web.telegram.org", "label": "Telegram"},
    "zoom": {"win": "zoommtg:", "android_pkg": "us.zoom.videomeetings", "web": "https://zoom.us", "label": "Zoom"},
    "vlc": {"win": "vlc.exe", "android_pkg": "org.videolan.vlc", "label": "VLC Media Player"},
    "youtube": {"win": "https://www.youtube.com", "android_pkg": "com.google.android.youtube", "web": "https://www.youtube.com", "label": "YouTube"},
    "gmail": {"win": "https://mail.google.com", "android_pkg": "com.google.android.gm", "web": "https://mail.google.com", "label": "Gmail"},
    "email": {"win": "mailto:", "android_pkg": "com.google.android.gm", "web": "https://mail.google.com", "label": "Email"},
    "maps": {"win": "https://maps.google.com", "android_pkg": "com.google.android.apps.maps", "web": "https://maps.google.com", "label": "Google Maps"},
    "google maps": {"win": "https://maps.google.com", "android_pkg": "com.google.android.apps.maps", "web": "https://maps.google.com", "label": "Google Maps"},
    "calendar": {"win": "outlookcal:", "android_pkg": "com.google.android.calendar", "web": "https://calendar.google.com", "label": "Calendar"},
    "clock": {"win": "ms-clock:", "android_pkg": "com.google.android.deskclock", "label": "Clock"},
    "camera": {"win": "microsoft.windows.camera:", "android_uri": "android.media.action.IMAGE_CAPTURE", "label": "Camera"},
    "photos": {"win": "ms-photos:", "android_pkg": "com.google.android.apps.photos", "web": "https://photos.google.com", "label": "Photos"},
    "gallery": {"win": "ms-photos:", "android_pkg": "com.sec.android.gallery3d", "label": "Gallery"},
    "play store": {"win": "https://play.google.com", "android_pkg": "com.android.vending", "web": "https://play.google.com", "label": "Google Play Store"},
    "github": {"win": "https://github.com", "android_pkg": "com.github.android", "web": "https://github.com", "label": "GitHub"},
}

SCREEN_CAPTURE_ASK = re.compile(
    r"\b(?:take|capture|grab|snap)\s+(?:a\s+|my\s+|the\s+)?(?:screenshot|screen\s*shot|screen\s*capture|screen)\b|"
    r"\bwhat(?:'?s|\s+is)\s+(?:on|showing\s+on)\s+(?:my|the)\s+(?:screen|display|monitor|window)\b|"
    r"\b(?:read|look\s+at|inspect|check|analyze|see)\s+(?:what(?:'?s|\s+is)\s+on\s+)?(?:my|the)\s+(?:screen|active\s+window|open\s+window|open\s+app|current\s+window|current\s+app|display)\b|"
    r"\bhelp\s+(?:me\s+)?with\s+(?:what(?:'?s|\s+is)\s+on\s+)?(?:my|the)\s+(?:screen|open\s+window|active\s+window|current\s+activity)\b",
    re.I,
)

OPEN_APP_ASK = re.compile(
    r"^\W*(?:(?:please|can\s+you|could\s+you|would\s+you)\s+)?(?:open|launch|start)\s+(?:up\s+)?(?:the\s+|my\s+)?"
    r"([A-Za-z0-9_.\-:/\\ ]{2,80}?)(?:\s+(?:app|application|program|for\s+me|on\s+my\s+(?:laptop|phone|computer|device)|please))?\W*$",
    re.I,
)


def asks_for_screen_or_activity(text: str) -> bool:
    """True when the user asks Clover to take a screenshot or read/inspect their live screen / active window."""
    return bool(SCREEN_CAPTURE_ASK.search(str(text or "")))


def is_standalone_app_target(target: str) -> bool:
    """True when `target` names a standalone desktop/mobile app or web URL rather than a local code file for VS Code."""
    t = str(target or "").strip().lower()
    if not t:
        return False
    if t in KNOWN_APPS:
        return t not in ("vscode", "vs code", "visual studio code")
    if re.match(r"^(?:https?://|mailto:|ms-settings:|ms-photos:|ms-clock:|spotify:|whatsapp:|discord:|slack:|msteams:)", t):
        return True
    return False


def _find_windows_start_menu_shortcut(app_name: str) -> Path | None:
    if os.name != "nt":
        return None
    norm = re.sub(r"[^a-z0-9]+", "", app_name.lower())
    if not norm:
        return None
    roots = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    ]
    best: Path | None = None
    for r in roots:
        if not r.is_dir():
            continue
        for lnk in r.rglob("*.lnk"):
            stem_norm = re.sub(r"[^a-z0-9]+", "", lnk.stem.lower())
            if stem_norm == norm:
                return lnk
            if norm in stem_norm and best is None:
                best = lnk
    return best


def open_target(raw_target: str, base_dir: Path | None = None) -> dict[str, Any]:
    """Open an application, URL, file, or folder on the current device and return structured result metadata."""
    t = str(raw_target or "").strip().strip("\"'")
    if not t:
        raise ValueError("target application, URL, or path is required")

    low = re.sub(r"\s+(?:app|application|program)$", "", t.lower()).strip()
    spec = KNOWN_APPS.get(low)

    # Check if bare domain like "youtube.com" or "github.com/..."
    if not spec and re.match(r"^(?:www\.|[a-z0-9-]+\.(?:com|org|net|io|dev|ai|edu|gov|in|co\.uk))(?:/.*)?$", t, re.I):
        url = t if t.lower().startswith(("http://", "https://")) else f"https://{t}"
        return _open_url(url, label=t)

    if re.match(r"^https?://\S+$", t, re.I):
        return _open_url(t, label=t)

    # Local file or folder path
    expanded = os.path.expandvars(os.path.expanduser(t))
    p = Path(expanded)
    if p.exists():
        if base_dir and paths.read_blocked(base_dir, p.resolve()):
            raise ValueError("Clover's vault is off-limits.")
        if os.name == "nt":
            os.startfile(str(p.resolve()))
            return {
                "opened": True,
                "target": str(p.resolve()),
                "kind": "folder" if p.is_dir() else "file",
                "label": p.name or str(p),
                "action_link": f"file:///{p.resolve().as_posix()}",
            }

    if spec is not None:
        label = spec.get("label") or t.title()
        if os.name == "nt" and spec.get("win"):
            win_t = spec["win"]
            try:
                if win_t.endswith(":") or win_t.startswith(("http://", "https://", "ms-")):
                    os.startfile(win_t)
                else:
                    subprocess.Popen([win_t], shell=False, creationflags=0x08000000)
                return {
                    "opened": True,
                    "target": label,
                    "kind": "app",
                    "label": label,
                    "action_link": spec.get("web") or f"app://{low}",
                }
            except OSError:
                if spec.get("web"):
                    return _open_url(spec["web"], label=label)
        if paths.ANDROID:
            pkg = spec.get("android_pkg", "")
            uri = spec.get("android_uri", "")
            if pkg:
                subprocess.run(
                    ["am", "start", "-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER", "-p", pkg],
                    capture_output=True,
                    timeout=5,
                )
            elif uri:
                subprocess.run(["am", "start", "-a", uri], capture_output=True, timeout=5)
            return {
                "opened": True,
                "target": label,
                "kind": "app",
                "label": label,
                "android_pkg": pkg,
                "android_uri": uri,
                "action_link": f"app://{low}",
                "web_url": spec.get("web", ""),
            }
        # Web / Linux fallback: return action_link so client browser/phone can launch it
        return {
            "opened": True,
            "target": label,
            "kind": "app",
            "label": label,
            "action_link": spec.get("web") or f"app://{low}",
            "web_url": spec.get("web", ""),
        }

    # Windows Start Menu shortcut search
    if os.name == "nt":
        lnk = _find_windows_start_menu_shortcut(low)
        if lnk is not None:
            os.startfile(str(lnk))
            return {
                "opened": True,
                "target": lnk.stem,
                "kind": "app",
                "label": lnk.stem,
                "action_link": f"app://{low}",
            }

    raise ValueError(f"Could not find an installed application or URL matching {t!r}.")


def _open_url(url: str, label: str = "") -> dict[str, Any]:
    if os.name == "nt":
        try:
            os.startfile(url)
        except OSError:
            pass
    elif paths.ANDROID:
        try:
            subprocess.run(["am", "start", "-a", "android.intent.action.VIEW", "-d", url], capture_output=True, timeout=5)
        except Exception:
            pass
    return {
        "opened": True,
        "target": url,
        "kind": "url",
        "label": label or url,
        "url": url,
        "action_link": url,
    }


def read_active_activity() -> dict[str, Any]:
    """Read the user's open application window titles, foreground window, and text clipboard (secrets redacted)."""
    fg_title = ""
    open_windows: list[str] = []
    clipboard_text = ""

    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            hwnd_fg = user32.GetForegroundWindow()
            if hwnd_fg:
                length = user32.GetWindowTextLengthW(hwnd_fg)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd_fg, buf, length + 1)
                    fg_title = buf.value.strip()

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            seen_titles: set[str] = set()

            def _enum_cb(hwnd, _lparam):
                if user32.IsWindowVisible(hwnd):
                    ln = user32.GetWindowTextLengthW(hwnd)
                    if ln > 1:
                        b = ctypes.create_unicode_buffer(ln + 1)
                        user32.GetWindowTextW(hwnd, b, ln + 1)
                        title = b.value.strip()
                        if title and title not in ("Program Manager", "Settings", "Microsoft Text Input Application") and title not in seen_titles:
                            seen_titles.add(title)
                            open_windows.append(title)
                return True

            user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
        except Exception:
            pass

        # Read plain text clipboard via Win32 CF_UNICODETEXT (13) without spawning PowerShell
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            user32.GetClipboardData.restype = ctypes.c_void_p
            if user32.OpenClipboard(None):
                try:
                    if user32.IsClipboardFormatAvailable(13):
                        h_data = user32.GetClipboardData(13)
                        if h_data:
                            ptr = kernel32.GlobalLock(h_data)
                            if ptr:
                                try:
                                    raw_clip = ctypes.wstring_at(ptr)
                                    clipboard_text, _ = clover_repo.redact(raw_clip[:1200].strip())
                                finally:
                                    kernel32.GlobalUnlock(h_data)
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass
    else:
        try:
            proc = subprocess.run(
                ["ps", "-eo", "comm,args", "--sort=-%mem"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                for ln in proc.stdout.splitlines()[1:10]:
                    s = ln.strip()
                    if s:
                        open_windows.append(s[:100])
        except Exception:
            pass

    return {
        "foreground_window": fg_title,
        "open_windows": open_windows[:15],
        "clipboard_preview": clipboard_text[:800] if clipboard_text else "",
    }


def capture_screenshot(base_dir: Path, *, inspect: bool = True) -> dict[str, Any]:
    """Capture a live screenshot of the user's screen, save it into `<base_dir>/media/<sha256>.png`, and run
    `image_inspector.inspect_image` so Clover has both the visual PNG and Playwright-grade OCR + `[e1..eN]` elements."""
    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)
    img: Image.Image | None = None

    if os.name == "nt":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(all_screens=False)
        except Exception:
            img = None

    if img is None and paths.ANDROID:
        # Check if Android screencap works or pick most recent screenshot
        tmp_cap = media / "tmp_screencap.png"
        try:
            r = subprocess.run(["screencap", "-p", str(tmp_cap)], capture_output=True, timeout=5)
            if r.returncode == 0 and tmp_cap.is_file() and tmp_cap.stat().st_size > 1000:
                img = Image.open(tmp_cap).convert("RGB")
        except Exception:
            pass
        finally:
            if tmp_cap.is_file():
                tmp_cap.unlink(missing_ok=True)

    activity = read_active_activity()

    if img is None:
        # Fallback visual activity card when running on a headless Linux server (e.g., k3)
        im = Image.new("RGB", (960, 540), (247, 243, 233))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 960, 56], fill=(23, 107, 44))
        dr.text((24, 18), "Clover Live Screen & Activity Snapshot", fill=(255, 255, 255))
        y = 80
        if activity["foreground_window"]:
            dr.text((24, y), f"Active Window: {activity['foreground_window'][:80]}", fill=(18, 25, 21))
            y += 30
        dr.text((24, y), "Open Windows / Running Processes:", fill=(23, 107, 44))
        y += 26
        for w_title in (activity["open_windows"] or ["Clover AI Studio (active)"])[:10]:
            dr.rectangle([24, y, 936, y + 28], fill=(255, 255, 255), outline=(215, 210, 198))
            dr.text((34, y + 7), str(w_title)[:95], fill=(45, 52, 54))
            y += 34
        img = im

    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    fname = hashlib.sha256(png_bytes).hexdigest()[:32] + ".png"
    out_path = media / fname
    out_path.write_bytes(png_bytes)

    report = image_inspector.inspect_image(out_path, annotate_dir=media) if inspect else {
        "file": fname,
        "path": str(out_path),
        "width": img.width,
        "height": img.height,
        "elements": [],
        "ocr_lines": [],
        "ocr_text": "",
    }
    report["activity"] = activity
    return report


def format_activity_for_prompt(screen_report: dict[str, Any]) -> str:
    """Format the live screen capture + active window titles + clipboard + OCR/UI snapshot for C1."""
    act = screen_report.get("activity") or read_active_activity()
    lines = ["Live Device Activity & Screen Capture Context (captured in S3):"]
    if act.get("foreground_window"):
        lines.append(f"- Active Foreground Window: {act['foreground_window']}")
    if act.get("open_windows"):
        lines.append("- Open Application Windows: " + "; ".join(act["open_windows"][:10]))
    if act.get("clipboard_preview"):
        lines.append(f"- Clipboard Text Preview: {act['clipboard_preview'][:400]}")
    lines.append(image_inspector.format_snapshot_for_prompt(screen_report, max_chars=2600))
    return "\n".join(lines)
