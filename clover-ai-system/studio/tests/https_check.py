"""Which HTTPS sites Python on this laptop can open, with Windows' default certificates and with certifi's bundle."""
import ssl
import urllib.request

try:
    import certifi
    CAFILE = certifi.where()
except ImportError:
    CAFILE = None

for url in ("https://example.com", "https://en.wikipedia.org/wiki/Main_Page", "https://www.python.org", "https://github.com"):
    ctxs = [("default", ssl.create_default_context())]
    if CAFILE:
        ctxs.append(("certifi", ssl.create_default_context(cafile=CAFILE)))
    for name, ctx in ctxs:
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "clover-check"}), timeout=15,
                                   context=ctx).read(100)
            result = "ok"
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            result = f"{type(exc).__name__}: {str(exc)[:100]}"
        print(f"{url:42} {name:8} {result}")
print("certifi:", CAFILE)
