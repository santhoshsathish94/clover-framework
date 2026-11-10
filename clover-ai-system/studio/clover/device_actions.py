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

import collections
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import clover_repo
import image_inspector
import paths

_DEBUG_LOCK = threading.Lock()
_DEBUG_RING: collections.deque[dict[str, Any]] = collections.deque(maxlen=400)


def log_debug(
    category: str,
    event: str,
    detail: dict[str, Any] | str | None = None,
    *,
    level: str = "INFO",
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Record a structured debug log entry in memory and append to `<base_dir>/logs/clover-debug.jsonl`."""
    entry: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch_ms": int(time.time() * 1000),
        "level": str(level or "INFO").upper(),
        "category": str(category or "general"),
        "event": str(event or ""),
        "detail": detail if isinstance(detail, (dict, list)) else (str(detail) if detail is not None else ""),
    }
    with _DEBUG_LOCK:
        _DEBUG_RING.append(entry)
        if base_dir is not None:
            try:
                log_f = Path(base_dir) / "logs" / "clover-debug.jsonl"
                log_f.parent.mkdir(parents=True, exist_ok=True)
                with open(log_f, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass
    return entry


def get_debug_logs(base_dir: Path | None = None, category: str = "", limit: int = 200) -> dict[str, Any]:
    """Return structured debug logs plus tails of `studio-server.log` and `llama-server*.log`."""
    with _DEBUG_LOCK:
        items = list(_DEBUG_RING)
    if not items and base_dir is not None:
        log_f = Path(base_dir) / "logs" / "clover-debug.jsonl"
        if log_f.is_file():
            try:
                lines = log_f.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
                for ln in lines:
                    if ln.strip():
                        try:
                            items.append(json.loads(ln))
                        except Exception:
                            pass
            except Exception:
                pass
    cat = str(category or "").strip().lower()
    if cat and cat not in ("all", "*"):
        items = [x for x in items if str(x.get("category") or "").lower() == cat]
    items = items[-max(1, min(500, int(limit))):]

    file_tails: dict[str, str] = {}
    if base_dir is not None:
        logs_dir = Path(base_dir) / "logs"
        if logs_dir.is_dir():
            for p in sorted(logs_dir.glob("*.log"))[:6]:
                try:
                    raw = p.read_bytes()
                    tail = raw[-8000:].decode("utf-8", errors="replace")
                    file_tails[p.name] = "\n".join(tail.splitlines()[-60:])
                except Exception:
                    pass
    return {
        "count": len(items),
        "entries": list(reversed(items)),
        "file_tails": file_tails,
    }


def clear_debug_logs(base_dir: Path | None = None) -> dict[str, Any]:
    with _DEBUG_LOCK:
        _DEBUG_RING.clear()
        if base_dir is not None:
            log_f = Path(base_dir) / "logs" / "clover-debug.jsonl"
            if log_f.is_file():
                try:
                    log_f.unlink()
                except Exception:
                    pass
    return {"cleared": True}

# Known cross-platform app targets:
#   key -> {"win": Windows exe/URI, "android_pkg": Android package, "android_uri": Android intent URI, "web": Web URL, "label": Display name}
KNOWN_APPS: dict[str, dict[str, str]] = {
    "calculator": {"win": "calc.exe", "android_pkg": "com.google.android.calculator", "android_uri": " app://calculator", "label": "Calculator"},
    "calc": {"win": "calc.exe", "android_pkg": "com.google.android.calculator", "label": "Calculator"},
    "notepad": {"win": "notepad.exe", "android_pkg": "com.samsung.android.app.notes", "label": "Notepad / Notes"},
    "notes": {"win": "notepad.exe", "android_pkg": "com.samsung.android.app.notes", "label": "Notes"},
    "paint": {"win": "mspaint.exe", "label": "Paint"},
    "snipping tool": {"win": "ms-screenclip:", "android_pkg": "com.samsung.android.app.smartcapture", "android_uri": "android.intent.action.PICK", "label": "System Screenshot / Snipping Tool"},
    "screenshot": {"win": "ms-screenclip:", "android_pkg": "com.samsung.android.app.smartcapture", "android_uri": "android.intent.action.PICK", "label": "System Screenshot"},
    "screen capture": {"win": "ms-screenclip:", "android_pkg": "com.samsung.android.app.smartcapture", "android_uri": "android.intent.action.PICK", "label": "System Screen Capture"},
    "screen recorder": {"win": "ms-screenclip:", "android_pkg": "com.samsung.android.app.screenrecorder", "android_uri": "android.media.action.VIDEO_CAPTURE", "label": "System Screen Recorder"},
    "screen recording": {"win": "ms-screenclip:", "android_pkg": "com.samsung.android.app.screenrecorder", "android_uri": "android.media.action.VIDEO_CAPTURE", "label": "System Screen Recorder"},
    "video recorder": {"win": "microsoft.windows.camera:", "android_uri": "android.media.action.VIDEO_CAPTURE", "label": "System Video Recorder"},
    "voice recorder": {"win": "ms-callrecording:", "android_pkg": "com.sec.android.app.voicenote", "android_uri": "android.provider.MediaStore.RECORD_SOUND", "label": "System Voice Recorder"},
    "sound recorder": {"win": "ms-callrecording:", "android_pkg": "com.sec.android.app.voicenote", "android_uri": "android.provider.MediaStore.RECORD_SOUND", "label": "System Sound Recorder"},
    "microphone": {"win": "ms-callrecording:", "android_pkg": "com.sec.android.app.voicenote", "android_uri": "android.provider.MediaStore.RECORD_SOUND", "label": "System Microphone / Voice Recorder"},
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
    r"(?<!what is a )(?<!what is )\b(?:screen\s*capture|screen\s*shot|screenshot)\b|"
    r"\bwhat(?:'?s|\s+is)\s+(?:on|showing\s+on)\s+(?:my|the)\s+(?:screen|display|monitor|window)\b|"
    r"\b(?:read|look\s+at|inspect|check|analyze|see)\s+(?:what(?:'?s|\s+is)\s+on\s+)?(?:my|the)\s+(?:screen|active\s+window|open\s+window|open\s+app|current\s+window|current\s+app|display)\b|"
    r"\bhelp\s+(?:me\s+)?with\s+(?:what(?:'?s|\s+is)\s+on\s+)?(?:my|the)\s+(?:screen|open\s+window|active\s+window|current\s+activity)\b",
    re.I,
)

CAMERA_PHOTO_ASK = re.compile(
    r"\b(?:use|open|with)\s+(?:the\s+|my\s+)?(?:cam(?:era|are)|webcam)(?:\s+app)?\b[^.?!\n]{0,60}\b(?:take|capture|snap|photo|picture|image|see|look)\b|"
    r"\b(?:take|capture|snap)\s+(?:a\s+|the\s+)?(?:photo|picture|snapshot|image)\s+(?:with|using|from|on)\s+(?:the\s+|my\s+)?(?:cam(?:era|are)|webcam)\b|"
    r"\b(?:what\s+(?:does|can|do)\s+(?:the\s+|my\s+)?(?:cam(?:era|are)|webcam)\s+see|see\s+what\s+(?:the\s+|my\s+)?(?:cam(?:era|are)|webcam)\s+sees)\b",
    re.I,
)

VIDEO_RECORD_ASK = re.compile(
    r"\b(?:record|capture)\s+(?:a\s+|my\s+|the\s+|some\s+)?(?:short\s+|live\s+)?(?:video|screen\s+recording|screen\s+video|clip)\b|"
    r"\b(?:use|open|with)\s+(?:the\s+|my\s+)?(?:screen\s+recorder|video\s+recorder|cam(?:era|are))\b[^.?!\n]{0,60}\b(?:record|video)\b",
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


def asks_for_camera_photo(text: str) -> bool:
    """True when the user asks Clover to use the camera / webcam app to take a photo and see what it sees."""
    return bool(CAMERA_PHOTO_ASK.search(str(text or "")))


def asks_for_video_recording(text: str) -> bool:
    """True when the user asks Clover to record a video or screen recording and understand what it recorded."""
    return bool(VIDEO_RECORD_ASK.search(str(text or "")))


CLIPBOARD_ASK = re.compile(
    r"\bclip\s?board\b|\b(?:what|text|the\s+text|the\s+thing)\s+(?:that\s+)?i(?:'ve|\s+have|\s+just)?\s+copied\b|"
    r"\bi\s+just\s+copied\b|\bcopied\s+text\b",
    re.I,
)


def mentions_clipboard(text: str) -> bool:
    """The clipboard is read only when the user's message mentions it (human Direction, 2026-11-09)."""
    return bool(CLIPBOARD_ASK.search(str(text or "")))


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


def _read_clipboard_text() -> str:
    """Plain text on the Windows clipboard (CF_UNICODETEXT, secrets redacted), or ""."""
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
                                return clover_repo.redact(ctypes.wstring_at(ptr)[:1200].strip())[0]
                            finally:
                                kernel32.GlobalUnlock(h_data)
            finally:
                user32.CloseClipboard()
    except Exception:
        pass
    return ""


def read_active_activity(include_clipboard: bool = False) -> dict[str, Any]:
    """Read the user's open application window titles and foreground window; the text clipboard (secrets redacted)
    only with `include_clipboard` (when the user's message mentions it)."""
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

        if include_clipboard:
            clipboard_text = _read_clipboard_text()
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


def _system_sense_dirs(kind: str) -> list[Path]:
    """Return OS directories where system Screenshot, Screen Recorder, Camera, and Voice Recorder apps save captures."""
    home = Path.home()
    dirs: list[Path] = []
    if os.name == "nt":
        if kind == "screenshot":
            dirs.extend([
                home / "Pictures" / "Screenshots",
                home / "OneDrive" / "Pictures" / "Screenshots",
                home / "Pictures",
            ])
        elif kind in ("screen_record", "video"):
            dirs.extend([
                home / "Videos" / "Captures",
                home / "Videos" / "Screen Recordings",
                home / "Pictures" / "Screenshots",
                home / "Videos",
            ])
        elif kind == "camera":
            dirs.extend([
                home / "Pictures" / "Camera Roll",
                home / "OneDrive" / "Pictures" / "Camera Roll",
                home / "Pictures",
            ])
        elif kind in ("microphone", "audio"):
            dirs.extend([
                home / "Documents" / "Sound Recordings",
                home / "Music",
            ])
    else:
        for root_str in ("/sdcard", "/storage/emulated/0"):
            r = Path(root_str)
            if kind == "screenshot":
                dirs.extend([r / "DCIM" / "Screenshots", r / "Pictures" / "Screenshots"])
            elif kind in ("screen_record", "video"):
                dirs.extend([
                    r / "DCIM" / "Screen recordings",
                    r / "Movies" / "Screen recordings",
                    r / "Movies",
                    r / "DCIM" / "Camera",
                ])
            elif kind == "camera":
                dirs.extend([r / "DCIM" / "Camera", r / "Pictures"])
            elif kind in ("microphone", "audio"):
                dirs.extend([r / "Recordings", r / "Voice Recorder", r / "Music" / "Recordings", r / "Music"])
    return [d for d in dirs if d.is_dir()]


def _find_latest_system_file(kind: str, exts: tuple[str, ...], max_age_s: float = 86400.0) -> Path | None:
    now = time.time()
    best_p: Path | None = None
    best_mtime = 0.0
    for d in _system_sense_dirs(kind):
        try:
            for p in d.iterdir():
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                mt = p.stat().st_mtime
                if (now - mt) <= max_age_s and mt > best_mtime and p.stat().st_size > 256:
                    best_mtime = mt
                    best_p = p
        except Exception:
            continue
    return best_p


def capture_screenshot(base_dir: Path, *, inspect: bool = True, launch_system_app: bool = False,
                       use_clipboard_image: bool = False) -> dict[str, Any]:
    """Capture or ingest a screenshot via the OS system screenshot app / clipboard / system screenshot folder, save it
    into `<base_dir>/media/<sha256>.png`, and run `image_inspector.inspect_image` (`I-JEPA` + OCR + `[e1..eN]`).
    The clipboard image is used only when the user started the snip (`launch_system_app`) or mentioned the clipboard."""
    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)
    img: Image.Image | None = None
    source_used = "unknown"

    if os.name == "nt":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        if launch_system_app:
            try:
                os.startfile("ms-screenclip:")
                log_debug("senses", "launched_windows_snipping_tool", {"uri": "ms-screenclip:"}, base_dir=base_dir)
                time.sleep(1.5)
            except Exception as exc:
                log_debug("senses", "snipping_tool_launch_error", str(exc), level="WARN", base_dir=base_dir)
        # 1. Check recent system screenshot file in Pictures\Screenshots (taken in last 60s)
        latest_ss = _find_latest_system_file("screenshot", (".png", ".jpg", ".jpeg", ".webp"), max_age_s=60.0)
        if latest_ss is not None:
            try:
                img = Image.open(latest_ss).convert("RGB")
                source_used = f"system_screenshot_file:{latest_ss.name}"
            except Exception:
                img = None
        # 2. Check Windows Clipboard image (placed there by Windows Snipping Tool / Win+Shift+S)
        if img is None and (launch_system_app or use_clipboard_image):
            try:
                from PIL import ImageGrab
                clip_im = ImageGrab.grabclipboard()
                if isinstance(clip_im, Image.Image):
                    img = clip_im.convert("RGB")
                    source_used = "windows_snipping_clipboard"
            except Exception:
                pass
        # 3. Fallback to OS desktop frame grab via ImageGrab
        if img is None:
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab(all_screens=False)
                source_used = "windows_desktop_grab"
            except Exception:
                img = None

    if img is None and paths.ANDROID:
        # 1. Check Android system screenshot directory (/sdcard/DCIM/Screenshots or /sdcard/Pictures/Screenshots)
        latest_android_ss = _find_latest_system_file("screenshot", (".png", ".jpg", ".jpeg", ".webp"), max_age_s=3600.0)
        if latest_android_ss is not None:
            try:
                img = Image.open(latest_android_ss).convert("RGB")
                source_used = f"android_system_screenshot:{latest_android_ss.name}"
            except Exception:
                img = None
        # 2. Try Android system screencap service
        if img is None:
            tmp_cap = media / "tmp_screencap.png"
            try:
                r = subprocess.run(["screencap", "-p", str(tmp_cap)], capture_output=True, timeout=5)
                if r.returncode == 0 and tmp_cap.is_file() and tmp_cap.stat().st_size > 1000:
                    img = Image.open(tmp_cap).convert("RGB")
                    source_used = "android_screencap_service"
            except Exception:
                pass
            finally:
                if tmp_cap.is_file():
                    tmp_cap.unlink(missing_ok=True)

    activity = read_active_activity()

    if img is None:
        # Fallback visual activity card when running on a headless Linux server (e.g., k3)
        source_used = "activity_snapshot_card"
        im = Image.new("RGB", (960, 540), (247, 243, 233))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 960, 56], fill=(23, 107, 44))
        dr.text((24, 18), "Clover System Senses & Activity Snapshot", fill=(255, 255, 255))
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

    if max(img.width, img.height) > 1024:
        scale = 1024.0 / max(img.width, img.height)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)

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
    report["sense"] = "screenshot"
    report["source"] = source_used
    report["activity"] = activity
    log_debug(
        "senses",
        "screenshot_captured_and_inspected",
        {
            "source": source_used,
            "file": fname,
            "size": f"{report.get('width')}x{report.get('height')}",
            "elements": len(report.get("elements") or []),
            "jepa_scene": (report.get("jepa_visual") or {}).get("scene_kind"),
        },
        base_dir=base_dir,
    )
    return report


def _find_ffmpeg_exe(base_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    for root in ([Path(base_dir)] if base_dir else []) + [Path.home() / "CloverAI"]:
        bin_txt = root / "tools" / "ffmpeg" / "BIN.txt"
        if bin_txt.is_file():
            try:
                d = Path(bin_txt.read_text(encoding="utf-8").strip())
                candidates.append(d / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
            except Exception:
                pass
        ff_dir = root / "tools" / "ffmpeg"
        if ff_dir.is_dir():
            candidates.extend(sorted(ff_dir.rglob("ffmpeg.exe" if os.name == "nt" else "ffmpeg")))
    for c in candidates:
        if c.is_file():
            return c
    w = shutil.which("ffmpeg")
    return Path(w) if w else None


def _dshow_devices(ff_exe: Path) -> tuple[list[str], list[str]]:
    if os.name != "nt":
        return [], []
    try:
        r = subprocess.run(
            [str(ff_exe), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=0x08000000,
        )
        out = (r.stderr or "") + "\n" + (r.stdout or "")
        vids = re.findall(r'"([^"]+)"\s+\(video\)', out)
        auds = re.findall(r'"([^"]+)"\s+\(audio\)', out)
        return vids, auds
    except Exception:
        return [], []


def capture_camera_photo(base_dir: Path, *, inspect: bool = True, launch_system_app: bool = False) -> dict[str, Any]:
    """Capture a live photo from the device's camera (via DirectShow webcam or recent Camera Roll file on Windows),
    save it into `<base_dir>/media/<sha256>.png`, and run `image_inspector.inspect_image` (`I-JEPA` + OCR + `[e1..eN]`)."""
    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)
    img: Image.Image | None = None
    source_used = "unknown"

    # 1. Check if a camera photo was saved in Camera Roll in the last 30 seconds
    latest_cam = _find_latest_system_file("camera", (".jpg", ".jpeg", ".png", ".webp"), max_age_s=30.0)
    if latest_cam is not None:
        try:
            img = Image.open(latest_cam).convert("RGB")
            source_used = f"camera_roll:{latest_cam.name}"
        except Exception:
            img = None

    # 2. Capture a live frame from the physical webcam via ffmpeg DirectShow
    if img is None and os.name == "nt":
        ff_exe = _find_ffmpeg_exe(base_dir)
        if ff_exe is not None:
            vids, _ = _dshow_devices(ff_exe)
            for cam_name in vids:
                tmp_cam = media / f"tmp_cam_{int(time.time() * 1000)}.png"
                try:
                    r = subprocess.run(
                        [str(ff_exe), "-hide_banner", "-y", "-f", "dshow", "-i", f"video={cam_name}", "-frames:v", "1", str(tmp_cam)],
                        capture_output=True,
                        timeout=12,
                        creationflags=0x08000000,
                    )
                    if r.returncode == 0 and tmp_cam.is_file() and tmp_cam.stat().st_size > 512:
                        with Image.open(tmp_cam) as raw_im:
                            img = raw_im.convert("RGB")
                        source_used = f"webcam:{cam_name}"
                        break
                except Exception:
                    pass
                finally:
                    tmp_cam.unlink(missing_ok=True)

    # 3. Fallback: if no webcam frame was available and launch_system_app was requested, open OS Camera app
    if img is None and launch_system_app:
        try:
            open_target("camera", base_dir=base_dir)
        except Exception:
            pass

    if img is None:
        rep_ss = capture_screenshot(base_dir, inspect=inspect, launch_system_app=False)
        rep_ss["sense"] = "camera"
        rep_ss["kind"] = "image"
        return rep_ss

    if max(img.width, img.height) > 1024:
        scale = 1024.0 / max(img.width, img.height)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)

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
    report["ok"] = True
    report["sense"] = "camera"
    report["kind"] = "image"
    report["source"] = source_used
    log_debug(
        "senses",
        "camera_photo_captured_and_inspected",
        {
            "source": source_used,
            "file": fname,
            "size": f"{report.get('width')}x{report.get('height')}",
            "elements": len(report.get("elements") or []),
            "jepa_scene": (report.get("jepa_visual") or {}).get("scene_kind"),
        },
        base_dir=base_dir,
    )
    return report


def capture_video_recording(base_dir: Path, *, duration_s: int = 2, launch_system_app: bool = False) -> dict[str, Any]:
    """Record a short MP4 video (via webcam or screen capture) and extract + inspect its representative keyframe."""
    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)
    duration_s = max(1, min(6, int(duration_s or 2)))
    ff_exe = _find_ffmpeg_exe(base_dir)
    fp_exe = ff_exe.parent / ("ffprobe.exe" if os.name == "nt" else "ffprobe") if ff_exe else None
    source_used = "unknown"
    raw_bytes: bytes | None = None

    def _valid_mp4(p: Path) -> bool:
        if not p.is_file() or p.stat().st_size < 2048:
            return False
        if fp_exe and fp_exe.is_file():
            try:
                pr = subprocess.run(
                    [str(fp_exe), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type,width,height", "-of", "json", str(p)],
                    capture_output=True,
                    text=True,
                    timeout=6,
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
                doc = json.loads(pr.stdout or "{}")
                return bool(doc.get("streams"))
            except Exception:
                return True
        return True

    # 1. Check if a recent video recording was saved in the last 30s
    latest_vid = _find_latest_system_file("screen_record", (".mp4", ".webm", ".mkv", ".mov"), max_age_s=30.0)
    if latest_vid is not None and _valid_mp4(latest_vid):
        raw_bytes = latest_vid.read_bytes()
        source_used = f"system_video:{latest_vid.name}"

    # 2. Record live video via ffmpeg (DirectShow webcam or gdigrab desktop)
    if raw_bytes is None and ff_exe is not None and os.name == "nt":
        vids, _ = _dshow_devices(ff_exe)
        for cam_name in vids:
            tmp_mp4 = media / f"tmp_vid_{int(time.time() * 1000)}.mp4"
            try:
                r = subprocess.run(
                    [
                        str(ff_exe), "-hide_banner", "-y",
                        "-f", "dshow", "-i", f"video={cam_name}",
                        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                        "-map", "0:v", "-map", "1:a",
                        "-t", str(duration_s),
                        "-vf", "scale=640:-2,format=yuv420p",
                        "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac",
                        str(tmp_mp4),
                    ],
                    capture_output=True,
                    timeout=duration_s + 12,
                    creationflags=0x08000000,
                )
                if r.returncode == 0 and _valid_mp4(tmp_mp4):
                    raw_bytes = tmp_mp4.read_bytes()
                    source_used = f"webcam_video:{cam_name}"
                    break
            except Exception:
                pass
            finally:
                tmp_mp4.unlink(missing_ok=True)

    # 3. Fallback: encode live screen/activity frames into an MP4 video via ffmpeg
    if raw_bytes is None and ff_exe is not None:
        scr1 = capture_screenshot(base_dir, inspect=False, launch_system_app=False)
        f1_path = media / scr1["file"]
        tmp_mp4 = media / f"tmp_scrvid_{int(time.time() * 1000)}.mp4"
        try:
            r = subprocess.run(
                [
                    str(ff_exe), "-hide_banner", "-y",
                    "-loop", "1", "-t", str(duration_s), "-i", str(f1_path),
                    "-f", "lavfi", "-t", str(duration_s), "-i", "anullsrc=r=16000:cl=mono",
                    "-map", "0:v", "-map", "1:a",
                    "-vf", "scale=640:-2,format=yuv420p",
                    "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-shortest",
                    str(tmp_mp4),
                ],
                capture_output=True,
                timeout=15,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if r.returncode == 0 and _valid_mp4(tmp_mp4):
                raw_bytes = tmp_mp4.read_bytes()
                source_used = "screen_recording_mp4"
        except Exception:
            pass
        finally:
            tmp_mp4.unlink(missing_ok=True)

    if not raw_bytes:
        raise ValueError("Could not record video.")

    fname = hashlib.sha256(raw_bytes).hexdigest()[:32] + ".mp4"
    out_path = media / fname
    out_path.write_bytes(raw_bytes)

    # Extract representative keyframe PNG from the recorded video and run S3 Visual & OCR inspection
    frame_rep: dict[str, Any] = {}
    frame_fname = ""
    if ff_exe is not None:
        tmp_frame = media / f"tmp_frame_{fname[:12]}.png"
        try:
            subprocess.run(
                [str(ff_exe), "-hide_banner", "-y", "-ss", "0.5", "-i", str(out_path), "-frames:v", "1", str(tmp_frame)],
                capture_output=True,
                timeout=10,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if not tmp_frame.is_file() or tmp_frame.stat().st_size < 256:
                subprocess.run(
                    [str(ff_exe), "-hide_banner", "-y", "-i", str(out_path), "-frames:v", "1", str(tmp_frame)],
                    capture_output=True,
                    timeout=10,
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
            if tmp_frame.is_file() and tmp_frame.stat().st_size >= 256:
                f_bytes = tmp_frame.read_bytes()
                frame_fname = hashlib.sha256(f_bytes).hexdigest()[:32] + ".png"
                frame_out = media / frame_fname
                frame_out.write_bytes(f_bytes)
                frame_rep = image_inspector.inspect_image(frame_out, annotate_dir=media)
        except Exception:
            pass
        finally:
            tmp_frame.unlink(missing_ok=True)

    res_vid: dict[str, Any] = {
        "ok": True,
        "sense": "screen_record",
        "kind": "video",
        "source": source_used,
        "file": fname,
        "frame_file": frame_fname,
        "name": f"Recorded video ({duration_s}s · {source_used.split(':')[0]}).mp4",
        "bytes": len(raw_bytes),
        "duration_s": duration_s,
        "width": frame_rep.get("width", 640),
        "height": frame_rep.get("height", 360),
        "jepa_visual": frame_rep.get("jepa_visual") or {},
        "palette": frame_rep.get("palette") or [],
        "elements": frame_rep.get("elements") or [],
        "ocr_text": frame_rep.get("ocr_text") or "",
    }
    log_debug("senses", "video_sense_captured", {
        "source": source_used,
        "file": fname,
        "frame_file": frame_fname,
        "bytes": len(raw_bytes),
        "duration_s": duration_s,
        "jepa_scene": (res_vid["jepa_visual"] or {}).get("scene_kind"),
    }, base_dir=base_dir)
    return res_vid


def analyze_sensory_media(
    base_dir: Path,
    *,
    sense: str,
    raw_bytes: bytes | None = None,
    filename: str = "",
    launch_system_app: bool = False,
) -> dict[str, Any]:
    """Unified S3 Sensory handler talking to OS system apps (Screenshot, Screen Recorder, Camera, Voice Recorder/Mic)
    and running JEPA (`I-JEPA`, `Audio-JEPA`, `MJEPA`) sensory analysis on captured images, videos, and audio."""
    sense_norm = str(sense or "screenshot").strip().lower()
    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)

    if sense_norm in ("screenshot", "screen_capture") and not raw_bytes:
        return capture_screenshot(base_dir, inspect=True, launch_system_app=launch_system_app)

    if sense_norm == "camera" and not raw_bytes:
        return capture_camera_photo(base_dir, inspect=True, launch_system_app=launch_system_app)

    if sense_norm in ("screen_record", "video") and not raw_bytes:
        return capture_video_recording(base_dir, duration_s=2, launch_system_app=launch_system_app)

    if launch_system_app and not raw_bytes:
        app_map = {
            "screen_record": "screen recorder",
            "video": "screen recorder",
            "camera": "camera",
            "microphone": "voice recorder",
            "audio": "voice recorder",
        }
        target_app = app_map.get(sense_norm, sense_norm)
        opened_info = {}
        try:
            opened_info = open_target(target_app, base_dir=base_dir)
        except Exception as exc:
            opened_info = {"opened": False, "error": str(exc)}
        log_debug("senses", f"launched_system_{sense_norm}_app", opened_info, base_dir=base_dir)

        # Also check if a recent system recording/photo exists in the OS media folders
        ext_map = {
            "screen_record": (".mp4", ".webm", ".mkv", ".mov"),
            "video": (".mp4", ".webm", ".mkv", ".mov"),
            "camera": (".jpg", ".jpeg", ".png", ".webp"),
            "microphone": (".wav", ".m4a", ".mp3", ".ogg", ".aac"),
            "audio": (".wav", ".m4a", ".mp3", ".ogg", ".aac"),
        }
        latest_f = _find_latest_system_file(sense_norm, ext_map.get(sense_norm, (".png", ".mp4", ".wav")), max_age_s=600.0)
        if latest_f is None:
            return {
                "ok": True,
                "sense": sense_norm,
                "launched_system_app": True,
                "system_app": opened_info,
                "message": f"Launched system {opened_info.get('label') or target_app}. Capture or select media to analyze.",
            }
        raw_bytes = latest_f.read_bytes()
        filename = latest_f.name

    if not raw_bytes:
        raise ValueError(f"No media data received from system sense {sense_norm!r}")

    ext = Path(filename).suffix.lower() if filename else ""
    if sense_norm in ("microphone", "audio") or ext in (".wav", ".mp3", ".m4a", ".ogg", ".aac"):
        if ext not in (".wav", ".mp3", ".m4a", ".ogg", ".aac"):
            ext = ".wav"
        fname = hashlib.sha256(raw_bytes).hexdigest()[:32] + ext
        out_path = media / fname
        out_path.write_bytes(raw_bytes)
        jepa_audio = {}
        try:
            import jepa_engine
            jepa_audio = jepa_engine.for_base(base_dir).inspect_audio_latent(out_path, configured_arch="auto")
            jepa_audio = {k: v for k, v in jepa_audio.items() if k not in ("z_jepa", "z_vq")}
        except Exception as exc:
            jepa_audio = {"error": str(exc)}
        res_aud = {
            "ok": True,
            "sense": "microphone",
            "kind": "audio",
            "file": fname,
            "name": filename or f"System audio ({fname[:8]}{ext})",
            "bytes": len(raw_bytes),
            "jepa_audio": jepa_audio,
        }
        log_debug("senses", "audio_sense_analyzed", res_aud, base_dir=base_dir)
        return res_aud

    if sense_norm in ("screen_record", "video") or ext in (".mp4", ".webm", ".mkv", ".mov"):
        if ext not in (".mp4", ".webm", ".mkv", ".mov"):
            ext = ".mp4"
        fname = hashlib.sha256(raw_bytes).hexdigest()[:32] + ext
        out_path = media / fname
        out_path.write_bytes(raw_bytes)
        res_vid = {
            "ok": True,
            "sense": "screen_record",
            "kind": "video",
            "file": fname,
            "name": filename or f"Screen recording ({fname[:8]}{ext})",
            "bytes": len(raw_bytes),
        }
        log_debug("senses", "video_sense_captured", res_vid, base_dir=base_dir)
        return res_vid

    # Image / Camera / Screenshot sense
    img_obj = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    buf = io.BytesIO()
    img_obj.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    fname = hashlib.sha256(png_bytes).hexdigest()[:32] + ".png"
    out_path = media / fname
    out_path.write_bytes(png_bytes)
    rep = image_inspector.inspect_image(out_path, annotate_dir=media)
    rep["ok"] = True
    rep["sense"] = sense_norm
    rep["kind"] = "image"
    log_debug(
        "senses",
        f"{sense_norm}_image_analyzed",
        {
            "file": fname,
            "size": f"{rep.get('width')}x{rep.get('height')}",
            "elements": len(rep.get("elements") or []),
            "jepa_scene": (rep.get("jepa_visual") or {}).get("scene_kind"),
        },
        base_dir=base_dir,
    )
    return rep


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
