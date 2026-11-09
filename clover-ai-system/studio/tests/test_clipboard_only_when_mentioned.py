"""The clipboard is read only when the user's message mentions it (human Direction, 2026-11-09). The real clipboard and
screen are never touched: the readers are stubs that record whether they were called.
Usage: python tests/test_clipboard_only_when_mentioned.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import device_actions as D  # noqa: E402
from PIL import Image, ImageGrab  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
for text in ("What's in my clipboard?", "Read what I copied and what's on my screen", "Look at the text I just copied",
             "check my clip board", "Summarize the copied text"):
    ok &= check(f"mentions the clipboard: {text!r}", D.mentions_clipboard(text))
for text in ("What's on my screen?", "Take a screenshot and help me with my activity",
             "I copied the folder to D: yesterday, what's on my screen?", "Copy this file to my desktop"):
    ok &= check(f"does not mention the clipboard: {text!r}", not D.mentions_clipboard(text))

calls = {"text": 0, "image": 0, "snip": 0}


def fake_text():
    calls["text"] += 1
    return "STUB-CLIPBOARD-TEXT"


def fake_clip_image():
    calls["image"] += 1
    return Image.new("RGB", (40, 30), (200, 0, 0))


D._read_clipboard_text = fake_text
act = D.read_active_activity()
ok &= check("screen question without a mention: clipboard text not read", calls["text"] == 0 and act["clipboard_preview"] == "")
act = D.read_active_activity(include_clipboard=True)
ok &= check("message mentions the clipboard: text read (the gate can open)",
            calls["text"] == 1 and act["clipboard_preview"] == "STUB-CLIPBOARD-TEXT")

if os.name == "nt":
    real_grab, real_clip, real_start = ImageGrab.grab, ImageGrab.grabclipboard, os.startfile
    ImageGrab.grabclipboard = fake_clip_image
    ImageGrab.grab = lambda **k: Image.new("RGB", (40, 30), (0, 0, 200))
    os.startfile = lambda uri: calls.__setitem__("snip", calls["snip"] + 1)
    D._find_latest_system_file = lambda *a, **k: None
    D.read_active_activity = lambda include_clipboard=False: {"foreground_window": "", "open_windows": [],
                                                              "clipboard_preview": ""}
    try:
        base = Path(tempfile.mkdtemp())
        rep = D.capture_screenshot(base, inspect=False)
        ok &= check("screenshot without a mention: clipboard image not read", calls["image"] == 0
                    and rep["source"] == "windows_desktop_grab")
        rep = D.capture_screenshot(base, inspect=False, use_clipboard_image=True)
        ok &= check("screenshot when the clipboard is mentioned: clipboard image used", calls["image"] == 1
                    and rep["source"] == "windows_snipping_clipboard")
        rep = D.capture_screenshot(base, inspect=False, launch_system_app=True)
        ok &= check("snip started by the user from Clover: its clipboard image used", calls["snip"] == 1
                    and calls["image"] == 2 and rep["source"] == "windows_snipping_clipboard")
    finally:
        ImageGrab.grab, ImageGrab.grabclipboard, os.startfile = real_grab, real_clip, real_start
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
