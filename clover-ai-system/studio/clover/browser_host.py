"""The browser hand-back on the desktop: the page the user sends from their own Chrome or Edge reaches Clover AI Studio
on this device. Nothing is read from the browser unless the user clicks the extension's button or menu item.

- The "Send to Clover" extension (../browser-extension) reads the current tab after that click (activeTab) and sends it
  over native messaging. The browser starts this program for that extension's ID only: allowed_origins in the host
  manifest, which register() writes for the current Windows user (no administrator rights). This program checks the
  caller's origin too, saves the page in the inbox (<base>/media/inbox), answers once and exits.
- Clover AI Studio lists the inbox above the message box; the user adds a page to a message (take(): an .md document
  with the page's readable text and its citation) or dismisses it. Pages are web content: never instructions.

Usage: python browser_host.py --register | --unregister
(the browser starts it through the launcher register() wrote: --base <folder> <caller origin> [--parent-window=...])
Standard library only, and no Clover modules: the browser waits while it starts.
"""
import base64
import datetime
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

HOST_NAME = "com.cloverframework.studio"
EXTENSION_ID = "eibfjkaeioghjapmaejcingainoiepdf"  # from the public key in browser-extension/manifest.json
ORIGIN = f"chrome-extension://{EXTENSION_ID}/"
MAX_MESSAGE = 16 << 20  # the browser allows 64 MiB towards a host
MAX_HTML, MAX_TEXT, MAX_SELECTION, MAX_TITLE = 3_000_000, 1_000_000, 100_000, 500
KEEP = 20  # the newest pages kept in the inbox
BASE = Path(__file__).resolve().parent.parent  # the --base folder the server runs with (clover/ is inside it)
REG_KEYS = ("Software\\Google\\Chrome\\NativeMessagingHosts\\" + HOST_NAME,
            "Software\\Microsoft\\Edge\\NativeMessagingHosts\\" + HOST_NAME)
PAGE_ID = re.compile(r"\d{14}-[0-9a-f]{12}")


def inbox_dir(base=BASE):
    return Path(base) / "media" / "inbox"


# ---------------- native messaging (one message in, one answer out) ----------------
def read_message(stream):
    head = stream.read(4)
    if len(head) < 4:
        raise ValueError("no message")
    n = struct.unpack("=I", head)[0]  # the browser's native byte order
    if not 0 < n <= MAX_MESSAGE:
        raise ValueError(f"message size {n} is outside 1..{MAX_MESSAGE}")
    data = stream.read(n)
    if len(data) < n:
        raise ValueError("message cut short")
    return json.loads(data.decode("utf-8"))


def write_message(stream, obj):
    data = json.dumps(obj).encode("utf-8")
    stream.write(struct.pack("=I", len(data)) + data)
    stream.flush()


def save_page(msg, base=BASE):
    """A page message -> its id in the inbox; ValueError when it is not a web page with content."""
    url = str(msg.get("url") or "").strip()
    if not re.match(r"https?://[^\s\x00-\x1f\x7f]+$", url, re.I) or len(url) > 4096:
        raise ValueError("only web pages (http and https) are sent to Clover")
    page = {"url": url, "title": " ".join(str(msg.get("title") or "").split())[:MAX_TITLE],
            "selection": str(msg.get("selection") or "")[:MAX_SELECTION], "html": str(msg.get("html") or "")[:MAX_HTML],
            "text": str(msg.get("text") or "")[:MAX_TEXT],
            "received": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    if not (page["html"].strip() or page["text"].strip() or page["selection"].strip()):
        raise ValueError("the page had no content")
    body = json.dumps(page, ensure_ascii=False)
    pid = time.strftime("%Y%m%d%H%M%S") + "-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    d = inbox_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (pid + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(d / (pid + ".json"))
    for old in sorted(d.glob("*.json"))[:-KEEP]:
        old.unlink(missing_ok=True)
    return pid


def serve(argv, stdin, stdout, base=BASE):
    """One exchange with the browser. argv: ["--base", folder] from the launcher register() wrote, then the caller's
    origin (the browser passes it, and on Windows --parent-window=...)."""
    if len(argv) >= 2 and argv[0] == "--base":
        base, argv = Path(argv[1]), argv[2:]
    if not argv or argv[0] != ORIGIN:
        write_message(stdout, {"ok": False, "error": "this program answers only the Send to Clover extension"})
        return 1
    try:
        msg = read_message(stdin)
        if not isinstance(msg, dict) or msg.get("type") != "page":
            raise ValueError("unknown message")
        write_message(stdout, {"ok": True, "id": save_page(msg, base)})
        return 0
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        write_message(stdout, {"ok": False, "error": str(exc)[:300]})
        return 1


# ---------------- the inbox, for Clover AI Studio ----------------
def _page_file(pid, base):
    if not PAGE_ID.fullmatch(str(pid or "")):
        raise ValueError("unknown page")
    p = inbox_dir(base) / f"{pid}.json"
    if not p.is_file():
        raise ValueError("that page is no longer in the inbox")
    return p


def pages(base=BASE):
    """The pages waiting, newest first: [{"id", "title", "url", "site", "received", "selection"}]."""
    out = []
    d = inbox_dir(base)
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else ():
        try:
            pg = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        host = re.sub(r"^www\.", "", (re.match(r"https?://([^/:?#]+)", pg.get("url", ""), re.I) or [None, ""])[1].lower())
        out.append({"id": p.stem, "title": pg.get("title") or host, "url": pg.get("url", ""), "site": host,
                    "received": pg.get("received", ""), "selection": bool(pg.get("selection"))})
    return out


def count(base=BASE):
    d = inbox_dir(base)
    return sum(1 for _ in d.glob("*.json")) if d.is_dir() else 0


def dismiss(pid, base=BASE):
    _page_file(pid, base).unlink(missing_ok=True)


def take(pid, base=BASE, flag=None):
    """The page as a document for the next message ({"name", "data" (base64 .md)}), removed from the inbox. Its text:
    the readable text of the page as the browser showed it (page_fetch.extract), under the lines a citation needs.
    flag: marks text that addresses the assistant (semantics.flag_injection)."""
    import page_fetch
    p = _page_file(pid, base)
    pg = json.loads(p.read_text(encoding="utf-8"))
    info = page_fetch.extract(pg["html"], pg["url"]) if pg.get("html") else {}
    shown = re.sub(r"\n{3,}", "\n\n", str(pg.get("text") or "").replace("\r\n", "\n")).strip()  # as the browser drew it
    text = (info.get("text") or "").strip()
    if not text or len(text) < 200 < len(shown):  # the markup held little readable text (a page drawn by scripts)
        text = shown
    title = info.get("title") or pg.get("title") or pg["url"]
    site = info.get("site") or re.sub(r"^www\.", "", (re.match(r"https?://([^/:?#]+)", pg["url"], re.I) or [None, ""])[1])
    when = info.get("published") or ""
    lines = [f"# {title}",
             f"Source: {site}, {'published ' + when if when else 'publication date not stated'}"
             + (f", updated {info['modified']}" if info.get("modified") and info["modified"] != when else "")
             + f"; sent from your browser on {str(pg.get('received', ''))[:10]}.",
             f"Link: {pg['url']}"]
    if pg.get("selection", "").strip():
        lines += ["", "The part you selected:", "> " + "\n> ".join(pg["selection"].strip().splitlines())]
    body = "\n".join(lines) + "\n\n" + (text or "(the page had no readable text)") + "\n"
    if flag is not None:
        body = flag(body)
    name = re.sub(r"[^\w .()\-+&,']", "_", title).strip(" ._")[:80] or "page"
    p.unlink(missing_ok=True)
    return {"name": name + ".md", "data": base64.b64encode(body.encode("utf-8")).decode("ascii")}


# ---------------- registration (current Windows user) ----------------
def _winreg(mod):
    if mod is not None:
        return mod
    import winreg
    return winreg


def register(base=BASE, python=None, winreg_mod=None):
    """Point Chrome and Edge at this host for the current user: a launcher and the manifest in <base>/browser-host, and
    a key under HKCU for each browser. -> what was written."""
    reg = _winreg(winreg_mod)
    d = Path(base).resolve() / "browser-host"
    d.mkdir(parents=True, exist_ok=True)
    py = Path(python or sys.executable)
    if py.name.lower() == "pythonw.exe":
        py = py.with_name("python.exe")  # the host talks over stdin and stdout
    launcher = d / "clover-browser-host.cmd"
    launcher.write_text(f'@echo off\n"{py}" "{Path(__file__).resolve()}" --base "{Path(base).resolve()}" %*\n',
                        encoding="oem" if os.name == "nt" else "utf-8")  # cmd.exe reads its console code page; CRLF from text mode
    manifest = d / f"{HOST_NAME}.json"
    manifest.write_text(json.dumps({"name": HOST_NAME, "description": "Clover AI Studio: the page you send from your browser",
                                    "path": str(launcher), "type": "stdio", "allowed_origins": [ORIGIN]}, indent=2),
                        encoding="utf-8")
    for sub in REG_KEYS:
        reg.SetValue(reg.HKEY_CURRENT_USER, sub, reg.REG_SZ, str(manifest))
    return {"manifest": str(manifest), "launcher": str(launcher), "keys": ["HKCU\\" + s for s in REG_KEYS]}


def unregister(base=BASE, winreg_mod=None):
    reg = _winreg(winreg_mod)
    for sub in REG_KEYS:
        try:
            reg.DeleteKey(reg.HKEY_CURRENT_USER, sub)
        except FileNotFoundError:
            pass
    for name in (f"{HOST_NAME}.json", "clover-browser-host.cmd"):
        (Path(base) / "browser-host" / name).unlink(missing_ok=True)


def registered(base=BASE, winreg_mod=None):
    """Whether Chrome or Edge would start this host: a key pointing at this base's manifest."""
    try:
        reg = _winreg(winreg_mod)
    except ImportError:
        return False
    want = str(Path(base).resolve() / "browser-host" / f"{HOST_NAME}.json")
    for sub in REG_KEYS:
        try:
            if reg.QueryValue(reg.HKEY_CURRENT_USER, sub) == want and Path(want).is_file():
                return True
        except OSError:
            continue
    return False


def main():
    if "--register" in sys.argv[1:]:
        print(json.dumps(register(), indent=2))
        return 0
    if "--unregister" in sys.argv[1:]:
        unregister()
        print("unregistered")
        return 0
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    return serve(sys.argv[1:], sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    sys.exit(main())
