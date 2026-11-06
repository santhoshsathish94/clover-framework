"""Offline checks for browser-first search: the search address for the user's own browser, the engine choice kept in
search.json, and what Clover will open in the user's default browser (http/https only, Windows only; the phone app and
the Web page open links themselves).  Usage: python tests/test_browser_first.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import web_search as WS  # noqa: E402

ok, n = True, 0


def check(name, cond):
    global ok, n
    n += 1
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# ---------------- the search address ----------------
U = WS.browser_search_url
check("Google is the default engine", U("weather in Pune") == "https://www.google.com/search?q=weather+in+Pune")
check("the picked engine is used", U("weather in Pune", "duckduckgo") == "https://duckduckgo.com/?q=weather+in+Pune")
check("characters that end a query string are encoded", U("C# & .NET?", "bing") == "https://www.bing.com/search?q=C%23+%26+.NET%3F")
check("an unknown engine falls back to Google", U("x", "altavista").startswith("https://www.google.com/search?q="))
check("spaces and line breaks are collapsed", U("  two\n words ") == "https://www.google.com/search?q=two+words")
check("a very long query is cut", len(U("a" * 5000)) < 500)
check("every engine address is https", all(url.startswith("https://") for _, url in WS.BROWSERS.values()))

# ---------------- the engine choice is kept ----------------
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "search.json"
    cfg = WS.SearchConfig(p)
    check("a new configuration searches the browser with Google", cfg.browser_engine() == "google")
    desc = cfg.update(browser="duckduckgo")
    check("update(browser=) answers with the choice and every engine's address",
          desc["browser"] == "duckduckgo" and desc["browsers"]["brave"]["url"] == "https://search.brave.com/search?q=")
    check("it is saved beside the provider", json.loads(p.read_text(encoding="utf-8"))["browser"] == "duckduckgo")
    check("a new process reads it back", WS.SearchConfig(p).browser_engine() == "duckduckgo")
    try:
        cfg.update(browser="evil")
        check("an unknown engine is refused", False)
    except ValueError:
        check("an unknown engine is refused", cfg.browser_engine() == "duckduckgo")
    cfg.update(provider="wikipedia")
    check("changing the provider keeps the engine", cfg.browser_engine() == "duckduckgo")
    p.write_text(json.dumps({"provider": "wikipedia", "browser": "javascript:alert(1)"}), encoding="utf-8")
    check("a tampered file falls back to Google", WS.SearchConfig(p).browser_engine() == "google")

# ---------------- opening in the user's browser ----------------
import paths  # noqa: E402
import server as SV  # noqa: E402

opened = []
with tempfile.TemporaryDirectory() as d:
    search_before, platform_before = getattr(SV.S, "search", None), paths.PLATFORM
    try:
        SV.S.search = WS.SearchConfig(Path(d) / "search.json")
        SV.S.search.update(browser="bing")
        paths.PLATFORM = "windows"
        got = SV.open_in_browser({"query": "rivers of Pune"}, start=opened.append)
        check("a search opens with the engine picked in Settings", got == opened[-1] == "https://www.bing.com/search?q=rivers+of+Pune")
        got = SV.open_in_browser({"url": "https://en.wikipedia.org/wiki/Pune"}, start=opened.append)
        check("a page address opens as given", opened[-1] == "https://en.wikipedia.org/wiki/Pune")
        for bad in ("javascript:alert(1)", "file:///C:/Windows/win.ini", "ms-settings:privacy", "C:\\Windows\\notepad.exe",
                    "https://example.com/a b", "https://example.com/\x00", "http://" + "a" * 3000, "", "https://"):
            before = len(opened)
            try:
                SV.open_in_browser({"url": bad}, start=opened.append)
                refused = False
            except ValueError:
                refused = True
            check(f"not opened: {bad[:40]!r}", refused and len(opened) == before)
        paths.PLATFORM = "android"
        try:
            SV.open_in_browser({"query": "x"}, start=opened.append)
            check("the phone app opens links itself (no server open)", False)
        except ValueError:
            check("the phone app opens links itself (no server open)", True)
    finally:
        SV.S.search, paths.PLATFORM = search_before, platform_before

# ---------------- the page ----------------
app = (Path(__file__).resolve().parent.parent / "clover" / "web" / "app.html").read_text(encoding="utf-8")
check("Settings offers the six engines", all(f'data-v="{e}"' in app for e in WS.BROWSERS) and 'id="s-browser"' in app)
check("web search steps offer the user a browser search", "Search in your browser" in app and "browserLink(t)" in app)
check("Windows opens through this laptop's server, phone and Web as plain links",
      "/api/browser/open" in app and "if (PHONE || WEB) return;" in app)
check("nothing opens by itself: only a click handler calls /api/browser/open",
      app.count("/api/browser/open") == 1 and "a.addEventListener('click'" in app)

print(f"{n} checks" + (" passed" if ok else ", SOME FAILED"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
