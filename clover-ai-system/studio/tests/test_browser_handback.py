"""Offline checks for the browser hand-back (browser_host.py): the native-messaging exchange as Chrome and Edge make it
(a real subprocess, framed messages), who may call it, what it saves, the inbox Clover AI Studio offers, the page as
a document with its citation, and registration (against a stand-in registry: nothing is written to this PC's
registry).  Usage: python tests/test_browser_handback.py"""
import base64
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import browser_host as BH  # noqa: E402

ok, n = True, 0


def check(name, cond):
    global ok, n
    n += 1
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


def frame(obj):
    data = json.dumps(obj).encode("utf-8")
    return struct.pack("=I", len(data)) + data


def unframe(raw):
    if len(raw) < 4:
        return None
    k = struct.unpack("=I", raw[:4])[0]
    return json.loads(raw[4:4 + k].decode("utf-8")) if len(raw) >= 4 + k else None


ARTICLE = ("<html><head><title>Rivers of Pune</title><meta property='article:published_time' content='2026-10-30'>"
           "</head><body><nav>MENU</nav><article><h1>Rivers of Pune</h1><p>" + "The Mula and the Mutha meet in Pune. " * 20
           + "</p></article></body></html>")
PAGE = {"type": "page", "url": "https://news.example/pune-rivers", "title": "Rivers of Pune", "selection": "Mula and the Mutha",
        "html": ARTICLE, "text": "Rivers of Pune\nThe Mula and the Mutha meet in Pune."}

with tempfile.TemporaryDirectory() as d:
    base = Path(d) / "CloverAI"
    host = base / "clover" / "browser_host.py"
    host.parent.mkdir(parents=True)
    shutil.copy2(CLOVER / "browser_host.py", host)  # BASE is the folder above clover/, as in an install

    def run(args, payload, raw=None):
        r = subprocess.run([sys.executable, str(host)] + args, input=raw if raw is not None else frame(payload),
                           capture_output=True, timeout=60)
        return r.returncode, unframe(r.stdout), r.stdout

    # ---------------- the exchange, as the browser makes it ----------------
    code, reply, raw = run([BH.ORIGIN, "--parent-window=0"], PAGE)
    check("the extension's message is answered in one framed reply", code == 0 and reply and reply.get("ok") is True)
    check("... and nothing else is written to stdout", raw == frame(reply) if reply else False)
    saved = sorted((base / "media" / "inbox").glob("*.json"))
    check("the page is saved in <base>/media/inbox", len(saved) == 1 and saved[0].stem == reply.get("id"))
    stored = json.loads(saved[0].read_text(encoding="utf-8")) if saved else {}
    check("url, title, selection, markup and shown text are kept", stored.get("url") == PAGE["url"] and
          stored.get("selection") == PAGE["selection"] and "Mula" in stored.get("html", "") and stored.get("text"))
    code, reply, _ = run(["chrome-extension://abcdefghijklmnopabcdefghijklmnop/"], PAGE)
    check("another extension's origin is refused", code == 1 and reply and reply["ok"] is False and len(list((base / "media" / "inbox").glob("*.json"))) == 1)
    code, reply, _ = run([], PAGE)
    check("no origin at all is refused", code == 1 and reply and reply["ok"] is False)
    code, reply, _ = run([BH.ORIGIN], dict(PAGE, url="file:///C:/Users/you/secret.txt"))
    check("a file:// page is refused", reply and reply["ok"] is False and "http" in reply["error"])
    code, reply, _ = run([BH.ORIGIN], dict(PAGE, url="chrome://settings"))
    check("a browser page is refused", reply and reply["ok"] is False)
    code, reply, _ = run([BH.ORIGIN], dict(PAGE, html="", text="", selection=""))
    check("an empty page is refused", reply and reply["ok"] is False and "content" in reply["error"])
    code, reply, _ = run([BH.ORIGIN], {"type": "other"})
    check("an unknown message is refused", reply and reply["ok"] is False)
    code, reply, _ = run([BH.ORIGIN], None, raw=struct.pack("=I", BH.MAX_MESSAGE + 1) + b"{}")
    check("an oversized message is refused before it is read", reply and reply["ok"] is False and "size" in reply["error"])
    code, reply, _ = run([BH.ORIGIN], None, raw=struct.pack("=I", 50) + b'{"type": "page"')
    check("a message cut short is refused", reply and reply["ok"] is False)
    code, reply, _ = run([BH.ORIGIN], None, raw=b"")
    check("no message at all is answered with an error, not a crash", reply and reply["ok"] is False)
    other = Path(d) / "elsewhere"
    code, reply, _ = run(["--base", str(other), BH.ORIGIN], PAGE)
    check("the launcher's --base decides where the page goes", code == 0 and len(list((other / "media" / "inbox").glob("*.json"))) == 1)
    code, reply, _ = run([BH.ORIGIN], dict(PAGE, title="T" * 5000, html="x" * (BH.MAX_HTML + 10)))
    big = json.loads((base / "media" / "inbox" / (reply["id"] + ".json")).read_text(encoding="utf-8")) if reply and reply.get("ok") else {}
    check("long fields are capped", len(big.get("title", "")) == BH.MAX_TITLE and len(big.get("html", "")) == BH.MAX_HTML)

    # ---------------- the inbox ----------------
    listed = BH.pages(base)
    check("the inbox lists the pages, newest first, without their content",
          len(listed) == 2 and listed[0]["id"] == reply["id"] and "html" not in listed[0] and listed[1]["site"] == "news.example")
    check("count() agrees", BH.count(base) == 2)
    first = listed[1]["id"]
    doc = BH.take(first, base, flag=lambda t: "[FLAGGED] " + t)
    body = base64.b64decode(doc["data"]).decode("utf-8")
    check("take(): an .md document named after the page", doc["name"] == "Rivers of Pune.md")
    check("... with the citation lines first", body.startswith("[FLAGGED] # Rivers of Pune\nSource: news.example, published 2026-10-30")
          and "Link: https://news.example/pune-rivers" in body)
    check("... the user's selection quoted", "> Mula and the Mutha" in body)
    check("... the readable text, without the page's menu", "meet in Pune." in body and "MENU" not in body)
    check("... and the flag applied (pages are content, never instructions)", body.startswith("[FLAGGED]"))
    check("taken pages leave the inbox", BH.count(base) == 1 and all(p["id"] != first for p in BH.pages(base)))
    for bad in ("../../etc/passwd", first, "20261106-zzzz", ""):
        try:
            BH.take(bad, base)
            refused = False
        except ValueError:
            refused = True
        check(f"take({bad!r}) is refused", refused)
    outside = base / "media" / "outside.json"
    outside.write_text(json.dumps(dict(PAGE, html="<p>OUTSIDE-THE-INBOX</p>")), encoding="utf-8")
    try:
        BH.take("../outside", base)
        refused = False
    except ValueError:
        refused = True
    check("a file outside the inbox is not read or removed through an id", refused and outside.is_file())
    BH.dismiss(listed[0]["id"], base)
    check("dismiss() removes a page", BH.count(base) == 0)
    js = dict(PAGE, html="<html><body><div id='app'></div><script>render()</script></body></html>",
              text="A page drawn by scripts. " * 20)
    BH.save_page(js, base)
    drawn = base64.b64decode(BH.take(BH.pages(base)[0]["id"], base)["data"]).decode("utf-8")
    check("a page drawn by scripts keeps the text the browser showed", "A page drawn by scripts." in drawn)
    for i in range(25):  # more than KEEP (20)
        BH.save_page(dict(PAGE, text=PAGE["text"] + str(i)), base)
    check("the inbox keeps the newest 20", BH.KEEP == 20 and BH.count(base) == 20)

    # ---------------- registration (a stand-in registry) ----------------
    class FakeReg:
        HKEY_CURRENT_USER, REG_SZ = "HKCU", 1

        def __init__(self):
            self.keys = {}

        def SetValue(self, root, sub, kind, value):
            self.keys[(root, sub)] = value

        def QueryValue(self, root, sub):
            if (root, sub) not in self.keys:
                raise FileNotFoundError(sub)
            return self.keys[(root, sub)]

        def DeleteKey(self, root, sub):
            if self.keys.pop((root, sub), None) is None:
                raise FileNotFoundError(sub)

    reg = FakeReg()
    check("not registered at first", not BH.registered(base, winreg_mod=reg))
    got = BH.register(base, python=r"C:\CloverAI\venv\Scripts\pythonw.exe", winreg_mod=reg)
    man = json.loads(Path(got["manifest"]).read_text(encoding="utf-8"))
    check("the manifest names the host, stdio and exactly this extension",
          man["name"] == BH.HOST_NAME and man["type"] == "stdio" and man["allowed_origins"] == [BH.ORIGIN])
    check("both browsers' HKCU keys point at the manifest",
          sorted(reg.keys) == sorted(("HKCU", k) for k in BH.REG_KEYS) and set(reg.keys.values()) == {got["manifest"]})
    launcher = Path(man["path"]).read_text(encoding="oem")
    check("the launcher runs python.exe (stdio), not pythonw.exe", "python.exe" in launcher and "pythonw.exe" not in launcher)
    check("the launcher passes this base and the browser's arguments",
          f'--base "{base.resolve()}" %*' in launcher and launcher.startswith("@echo off"))
    check("registered() sees it", BH.registered(base, winreg_mod=reg))
    check("another base is not registered", not BH.registered(Path(d) / "elsewhere", winreg_mod=reg))
    BH.unregister(base, winreg_mod=reg)
    check("unregister() removes the keys and files", not reg.keys and not Path(got["manifest"]).exists() and not Path(man["path"]).exists())
    BH.unregister(base, winreg_mod=reg)
    check("unregister() twice is harmless", not reg.keys)

# ---------------- the extension ----------------
ext = CLOVER.parent / "browser-extension"
mf = json.loads((ext / "manifest.json").read_text(encoding="utf-8"))
import hashlib  # noqa: E402

der = base64.b64decode(mf["key"])
check("the manifest key gives the ID the host allows",
      "".join(chr(97 + int(c, 16)) for c in hashlib.sha256(der).hexdigest()[:32]) == BH.EXTENSION_ID)
check("the extension asks only for activeTab, scripting, nativeMessaging and contextMenus",
      sorted(mf["permissions"]) == ["activeTab", "contextMenus", "nativeMessaging", "scripting"]
      and "host_permissions" not in mf and "content_scripts" not in mf)
bg = (ext / "background.js").read_text(encoding="utf-8")
check("it reads a tab only from a click (action or menu)", "chrome.action.onClicked.addListener(send)" in bg
      and "chrome.contextMenus.onClicked.addListener" in bg and bg.count("executeScript") == 1)
check("it talks only to Clover's host", f"const HOST = '{BH.HOST_NAME}'" in bg and "fetch(" not in bg and "XMLHttpRequest" not in bg)
check("no private key is in the extension folder", not any(p.suffix in (".pem", ".key") for p in ext.iterdir()))

# ---------------- the server ----------------
import paths  # noqa: E402
import server as SV  # noqa: E402

platform_before, args_before = paths.PLATFORM, SV.S.args
try:
    with tempfile.TemporaryDirectory() as d:
        SV.S.args = type("A", (), {"base": d})()
        paths.PLATFORM = "windows"
        BH.save_page(PAGE, d)
        check("/api/status counts the pages waiting", SV.inbox_count() == 1)
        desc = SV.inbox_describe()
        check("/api/inbox lists them and names the extension", len(desc["pages"]) == 1 and desc["extension_id"] == BH.EXTENSION_ID
              and desc["extension_dir"] and desc["available"] is True)
        paths.PLATFORM = "android"
        check("the phone has no laptop inbox", SV.inbox_count() == 0 and SV.inbox_describe() == {"pages": [], "available": False})
finally:
    paths.PLATFORM, SV.S.args = platform_before, args_before
app = (CLOVER / "web" / "app.html").read_text(encoding="utf-8")
check("the page offers waiting pages above the message box", 'id="inbox"' in app and "From your browser: " in app
      and "/api/inbox/take" in app and "/api/inbox/dismiss" in app)
check("a taken page joins the message as a document", "state.docs.push(r.document)" in app)
check("Settings sets up and removes the hand-back (laptop only)", "handback-on" in app and "/api/browser/register" in app
      and 'class="group pad laptop-only"' in app)

print(f"{n} checks" + (" passed" if ok else ", SOME FAILED"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
