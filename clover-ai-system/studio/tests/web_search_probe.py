"""Why web_search returns nothing: probe the Wikipedia and DuckDuckGo paths of the live search tool on this laptop."""
import importlib.util
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lws", BASE / "clover" / "live-web-search-tool.py")
lws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lws)
tool = lws.LiveWebSearchTool(max_results=5)
for q in ("CWE-79 definition prevention", "cross-site scripting prevention", "Liquid AI LFM3"):
    print(q, "| wiki:", [r[:70] for r in tool._search_wikipedia(q, max_items=3)], "| ddg:", [r[:70] for r in tool._search_ddg(q, max_items=3)])
url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote("cross-site scripting prevention")
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": tool.user_agent}), timeout=10) as r:
        page = r.read().decode("utf-8", "ignore")
        print("ddg status", r.status, "bytes", len(page), "result__a" in page, "anomaly" in page.lower(), "captcha" in page.lower())
except Exception as exc:  # report the failure kind
    print("ddg error", type(exc).__name__, exc)
print("user agent:", tool.user_agent)
