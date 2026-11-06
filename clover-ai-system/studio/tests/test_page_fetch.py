"""Offline checks for page_fetch (the page reader behind fetch_url and web research) and its wiring into the tools:
the address guard (this device, private networks, redirects into them, only typed hosts allowed), robots.txt (RFC
9309), caps, encodings, readable-text extraction with what a citation needs. Uses throwaway local HTTP servers; the
keyless search fallbacks are stubbed, so nothing goes to the internet.  Usage: python tests/test_page_fetch.py"""
import gzip
import http.server
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import page_fetch as PF  # noqa: E402

ok, n = True, 0


def check(name, cond):
    global ok, n
    n += 1
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


def refused(fn, *words):
    """FetchError raised, its message holding every word."""
    try:
        fn()
    except PF.FetchError as e:
        return all(w.lower() in str(e).lower() for w in words)
    return False


# ---------------- addresses ----------------
for ip in ("127.0.0.1", "127.8.8.8", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254", "100.64.1.1",
           "0.0.0.0", "224.0.0.1", "255.255.255.255", "240.0.0.1", "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1",
           "::ffff:127.0.0.1", "::ffff:10.0.0.1", "2002:7f00:1::1", "2002:c0a8:101::1", "64:ff9b::a00:1",
           "64:ff9b::7f00:1"):
    check(f"address {ip} is refused", PF.blocked_reason(ip) != "")
for ip in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "2a00:1450:4001:80b::200e"):
    check(f"address {ip} is public", PF.blocked_reason(ip) == "")
check("the reason names this device for loopback", PF.blocked_reason("127.0.0.1") == "this device")
check("cloud metadata 169.254.169.254 is link-local", PF.blocked_reason("169.254.169.254") == "a link-local address")

# ---------------- what the user typed ----------------
T = PF.typed_hosts
check("a typed link gives its host and port", T("read http://192.168.1.10:8080/status please") == {("192.168.1.10", 8080)})
check("https default port, host lower-cased, trailing dot of the sentence dropped",
      T("check https://Example.com/a.") == {("example.com", 443)})
check("a bare local address allows ports 80 and 443", T("my router is 192.168.0.1") == {("192.168.0.1", 80), ("192.168.0.1", 443)})
check("localhost:3000 gives that port only", T("the app on localhost:3000 is slow") == {("localhost", 3000)})
check("printer.local is a local name", T("what does printer.local say") == {("printer.local", 80), ("printer.local", 443)})
check("a version number is not an address", T("version 1.2.3.4.5 is out") == frozenset())
check("an e-mail address is not a host", T("mail bob@192.168.1.1 now") == frozenset())
check("no text, no hosts", T("") == frozenset() and T(None) == frozenset())


# ---------------- local servers ----------------
class Site:
    """A local HTTP server; routes: path -> (status, headers, body) or a callable(handler)."""

    def __init__(self, routes):
        site = self
        self.routes, self.hits = routes, []

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):
                site.hits.append(self.path)
                r = site.routes.get(self.path.split("?")[0]) or site.routes.get("*")
                if r is None:
                    r = (404, {"Content-Type": "text/plain"}, b"not here")
                if callable(r):
                    return r(self)
                status, headers, body = r
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.me = {("127.0.0.1", self.port)}
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


HTML = "text/html; charset=utf-8"
ARTICLE = """<!doctype html><html><head><title>Plain title | Example News</title>
<meta property="og:title" content="Rivers of Pune &amp; their bridges">
<meta property="og:site_name" content="Example News">
<meta property="article:published_time" content="2026-10-30T08:15:00+05:30">
<meta property="article:modified_time" content="2026-11-02T10:00:00Z">
<meta name="description" content="Which rivers run through Pune.">
<link rel="canonical" href="/news/rivers">
<style>.x{color:red}</style><script>var secret = "SCRIPT-TEXT";</script></head>
<body><header><nav><a href="/">Home</a> <a href="/sports">NAV-LINK</a></nav></header>
<aside>ASIDE-TEXT related stories</aside>
<article><h1>Rivers of Pune</h1><p>The Mula and the Mutha meet in Pune &amp; form the Mula-Mutha river.</p>
<h2>Bridges</h2><ul><li>Sambhaji bridge</li><li>Yashwantrao Chavan bridge</li></ul>
<table><tr><th>River</th><th>Length</th></tr><tr><td>Mula</td><td>60 km</td></tr></table>
<p>More than <b>twenty</b> bridges cross the rivers in the city, according to the corporation's 2025 survey of the
riverfront and its crossings, which also counted footbridges and the older stone bridges of the old city.</p></article>
<footer>FOOTER-TEXT copyright</footer><noscript>NOSCRIPT-TEXT</noscript></body></html>"""

site = Site({
    "/article": (200, {"Content-Type": HTML}, ARTICLE.encode("utf-8")),
    "/plain": (200, {"Content-Type": "text/plain"}, b"line one\n\n\n\nline two"),
    "/latin": (200, {"Content-Type": "text/html; charset=iso-8859-1"},
               "<html><body><p>Caf\xe9 \x93quoted\x94 menu</p></body></html>".encode("latin-1", "ignore")),
    "/metacharset": (200, {"Content-Type": "text/html"},
                     '<html><head><meta charset="windows-1252"></head><body><p>na\xefve r\xe9sum\xe9</p></body></html>'.encode("cp1252")),
    "/gzip": (200, {"Content-Type": HTML, "Content-Encoding": "gzip"}, gzip.compress(b"<html><body><p>ZIPPED-OK text</p></body></html>")),
    "/big": (200, {"Content-Type": "text/plain"}, b"x" * 2_500_000),
    "/pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.7 ..."),
    "/missing": (404, {"Content-Type": "text/plain"}, b"gone"),
    "/loop": (302, {"Location": "/loop"}, b""),
    "/hop": (302, {"Location": "/article"}, b""),
    "/secret": (200, {"Content-Type": HTML}, b"<html><body><p>LOCAL-ONLY-TOKEN-1234</p></body></html>"),
})
other = Site({"/": (200, {"Content-Type": HTML}, b"<html><body><p>OTHER-PORT</p></body></html>")})
site.routes["/to-other"] = (302, {"Location": other.base + "/"}, b"")
site.routes["/to-localhost"] = (302, {"Location": f"http://localhost:{site.port}/secret"}, b"")


def slow(h):
    time.sleep(3)
    h.send_response(200)
    h.end_headers()


site.routes["/slow"] = slow

check("an untyped local address is refused, naming this device",
      refused(lambda: PF.read_page(site.base + "/secret"), "this device", "type it"))
check("localhost is refused the same way", refused(lambda: PF.read_page(f"http://localhost:{site.port}/secret")))
check("typed on another port: still refused", refused(lambda: PF.read_page(site.base + "/secret", typed={("127.0.0.1", 1)})))
check("no request reached the server for refused addresses", "/secret" not in site.hits)
page = PF.read_page(site.base + "/article", typed=site.me)
check("typed: the page is read", "Mula-Mutha" in page["text"])
check("article text kept; nav, aside, footer, script, style and noscript dropped",
      all(w not in page["text"] for w in ("NAV-LINK", "ASIDE-TEXT", "FOOTER-TEXT", "SCRIPT-TEXT", "color:red", "NOSCRIPT-TEXT")))
check("title from og:title with entities decoded", page["title"] == "Rivers of Pune & their bridges")
check("site, published and updated dates", (page["site"], page["published"], page["modified"]) ==
      ("Example News", "2026-10-30", "2026-11-02"))
check("canonical resolved against the page", page["canonical"] == site.base + "/news/rivers")
check("headings, list items and table cells keep their shape",
      "# Rivers of Pune" in page["text"] and "• Sambhaji bridge" in page["text"] and "Mula | 60 km" in page["text"])
check("robots not checked for a page the user named", page["robots"].startswith("not checked"))
cited = PF.cite_text(page)
check("cite_text: first line in the tools' format, then site and date",
      cited.startswith("[Live Webpage: Rivers of Pune & their bridges — Link: " + site.base + "/article]")
      and "Source: Example News, published 2026-10-30, updated 2026-11-02; read on this device" in cited)
check("a redirect within the typed host is followed", "Mula-Mutha" in PF.read_page(site.base + "/hop", typed=site.me)["text"])
check("a redirect to another local port is refused", refused(lambda: PF.read_page(site.base + "/to-other", typed=site.me), "this device"))
check("a redirect to localhost is refused", refused(lambda: PF.read_page(site.base + "/to-localhost", typed=site.me)))
check("a redirect loop stops", refused(lambda: PF.read_page(site.base + "/loop", typed=site.me), "redirects"))
check("plain text is tidied", PF.read_page(site.base + "/plain", typed=site.me)["text"] == "line one\n\nline two")
check("latin-1 is read the way browsers do (cp1252 quotes)",
      PF.read_page(site.base + "/latin", typed=site.me)["text"] == "Caf\xe9 \u201cquoted\u201d menu")
check("a <meta charset> is honoured", PF.read_page(site.base + "/metacharset", typed=site.me)["text"] == "na\xefve r\xe9sum\xe9")
check("gzip is unpacked", PF.read_page(site.base + "/gzip", typed=site.me)["text"] == "ZIPPED-OK text")
big = PF.read_page(site.base + "/big", typed=site.me, max_chars=5000)
check("a large page is capped and marked", big["truncated"] and len(big["text"]) == 5000)
check("a PDF is not read as a page", refused(lambda: PF.read_page(site.base + "/pdf", typed=site.me), "application/pdf"))
check("HTTP 404 is reported", refused(lambda: PF.read_page(site.base + "/missing", typed=site.me), "404"))
t0 = time.monotonic()
check("a slow site is given up on", refused(lambda: PF.read_page(site.base + "/slow", typed=site.me, timeout=1.0)))
check("... within the time limit", time.monotonic() - t0 < 2.5)
check("file:// is refused", refused(lambda: PF.read_page("file:///C:/Windows/win.ini"), "http"))
check("user:password@ is refused", refused(lambda: PF.read_page("http://user:pw@example.com/"), "password"))
check("a malformed port is refused", refused(lambda: PF.read_page("http://example.com:99999/")))

# ---------------- extraction ----------------
E = PF.extract
check("a page without <article> keeps all its text",
      E("<html><body><div><p>First para here.</p><p>Second para.</p></div></body></html>")["text"] == "First para here.\nSecond para.")
check("without <article>: script, style, nav, aside, footer, noscript, buttons and svg are still dropped",
      E("<html><head><style>.a{}</style><script>var x='SCRIPT'</script></head><body><nav>NAV</nav><div><p>Body text</p>"
        "<button>BTN</button><svg><text>SVG</text></svg></div><aside>ASIDE</aside><footer>FOOT</footer>"
        "<noscript>NOS</noscript></body></html>")["text"] == "Body text")
check("a page wrapped in one <form> (ASP.NET) is still read",
      "Visible body text" in E("<html><body><form id='f'><div><p>Visible body text</p></div></form></body></html>")["text"])
check("JSON-LD gives the date when no meta does",
      E('<html><head><script type="application/ld+json">{"@graph":[{"@type":"NewsArticle","datePublished":"2026-09-01T10:00:00Z",'
        '"publisher":{"name":"LD Times"}}]}</script></head><body><p>x</p></body></html>')["published"] == "2026-09-01")
check("the site is not taken from a JSON-LD publisher (the host name is clearer)",
      E('<script type="application/ld+json">{"publisher":{"name":"Wikimedia Foundation, Inc."}}</script>')["site"] == "")
check("zero-width characters are removed", E("<p>18.5\ufeff / \u200b73.8</p>")["text"] == "18.5 / 73.8")
check("<time datetime> inside the article is the last resort",
      E("<article><time datetime='2025-12-24'>Christmas Eve</time><p>text</p></article>")["published"] == "2025-12-24")
check("an RFC 2822 date is read", PF._day("Mon, 02 Nov 2026 10:00:00 GMT") == "2026-11-02")
check("an svg <title> is not the page title", E("<html><body><svg><title>icon</title></svg><p>t</p></body></html>")["title"] == "")
check("a header inside the article is kept (the headline)", "Headline" in E("<article><header><h1>Headline</h1></header><p>b</p></article>")["text"])
check("inline elements do not split words", E("<p>Hel<b>lo</b> <i>world</i></p>")["text"] == "Hello world")
check("a short <main> beside long body text is not chosen",
      E("<body><main><p>Menu</p></main><div><p>" + "Long body text. " * 60 + "</p></div></body>")["text"].count("Long body") > 10)

# furniture inside the content area, modelled on the markup seen on en.wikipedia.org/wiki/Pune and a GitHub repository
# page (2026-11-06): the language menu, [1] markers and edit links were read before the article
langs = "".join(f'<li class="interlanguage-link mw-list-item"><a>Language{i}</a></li>' for i in range(40))
wiki = E(f"""<html><body><main id="content" class="mw-body"><h1 id="firstHeading">Pune</h1>
<div id="p-lang-btn" class="vector-dropdown mw-portlet mw-portlet-lang"><span class="vector-dropdown-label-text">40 languages</span>
<div class="vector-dropdown-content"><ul class="vector-menu-content-list">{langs}</ul></div></div>
<div id="mw-content-text"><div role="note" class="hatnote navigation-not-searchable">For other uses, see Pune (disambiguation).</div>
<p>Pune is a city in Maharashtra<sup id="cite_ref-1" class="reference"><a><span class="cite-bracket">[</span>1<span class="cite-bracket">]</span></a></sup>, India.</p>
<h2>History<span class="mw-editsection"><span class="mw-editsection-bracket">[</span><a>edit</a><span class="mw-editsection-bracket">]</span></span></h2>
<p>{"The city grew along the Mula and Mutha rivers. " * 30}</p>
<div class="navbox"><table><tr><td>Cities of Maharashtra: Mumbai Nagpur Nashik</td></tr></table></div></div></main></body></html>""")["text"]
check("furniture: the language menu inside <main> is dropped", "Language7" not in wiki and "40 languages" not in wiki)
check("furniture: [1] markers go and the sentence stays whole", "Pune is a city in Maharashtra, India." in wiki)
check("furniture: edit links, hatnote and navbox are dropped",
      "# History" in wiki and "edit" not in wiki and "disambiguation" not in wiki and "Nagpur" not in wiki)
gh = E("""<html><body><main><h2 class="prc-src-InternalVisuallyHidden-2YaI6">Repository navigation</h2>
<table><tr><td>src</td><td>initial commit</td></tr><tr><td>docs</td><td>update docs</td></tr></table>
<article class="markdown-body entry-content"><h1>Clover</h1><p>""" + "Clover is a way of adopting AI into the system cycle. " * 12 +
       """</p></article></main></body></html>""")["text"]
check("furniture: a visually hidden heading is dropped", "Repository navigation" not in gh)
check("an <article> holding the text is preferred to the rest of <main>", gh.startswith("# Clover") and "initial commit" not in gh)
wrapped = E('<html><body><div class="layout with-sidebar"><p>' + "Body of the article. " * 30 + '</p></div>'
            '<div class="sidebar"><p>Popular now</p></div></body></html>')["text"]
check("safeguard: a 'with-sidebar' wrapper holding the article keeps it", wrapped.count("Body of the article.") == 30)
check("safeguard: ... while the real sidebar goes", "Popular now" not in wrapped)
check("a share price is content, not a share bar",
      "1,234.50" in E('<p>Price</p><div class="share-price"><span>1,234.50</span></div><p>' + "Company news. " * 40 + "</p>")["text"])
check("aria-hidden and hidden elements are dropped",
      E('<p>Seen text here</p><span aria-hidden="true">ICON</span><div hidden>LATER</div><p>' + "More. " * 50 + "</p>")["text"].count("ICON") +
      E('<p>Seen</p><div hidden>LATER</div><p>' + "More. " * 50 + "</p>")["text"].count("LATER") == 0)
check("a missing </p> does not swallow the page", "After" in E("<div class='menu'><p>Home<p>About</div><p>After the menu " + "x " * 300)["text"])

# ---------------- robots.txt (RFC 9309) ----------------
R, A = PF.robots_rules, PF.robots_allows
rules = R("User-agent: *\nDisallow: /private/\nAllow: /private/ok\n")
check("robots: disallowed path", not A(rules, "/private/x"))
check("robots: the longer allow wins", A(rules, "/private/ok/page"))
check("robots: unmatched path is allowed", A(rules, "/public"))
check("robots: longest match, not file order", not A(R("User-agent: *\nAllow: /\nDisallow: /a\n"), "/a/b"))
check("robots: an allow rule wins a tie", A(R("User-agent: *\nDisallow: /x\nAllow: /x\n"), "/x"))
wild = R("User-agent: *\nDisallow: /*.pdf$\n")
check("robots: * and $ patterns", not A(wild, "/docs/a.pdf") and A(wild, "/docs/a.pdf?x=1") and A(wild, "/docs/a.pdfx"))
check("robots: our own group overrides *",
      A(R("User-agent: *\nDisallow: /\n\nUser-agent: CloverAIStudio\nAllow: /\n"), "/page"))
check("robots: groups naming us are combined, case-insensitively",
      R("user-agent: cloveraistudio\ndisallow: /a\n\nUser-agent: CLOVERAISTUDIO\nDisallow: /b\n") == [(False, "/a"), (False, "/b")])
check("robots: several user-agent lines make one group",
      R("User-agent: other\nUser-agent: CloverAIStudio/1.0\nDisallow: /c\n") == [(False, "/c")])
check("robots: an empty Disallow allows everything", A(R("User-agent: *\nDisallow:\n"), "/anything"))
check("robots: comments and unknown lines are ignored",
      not A(R("# hi\nSitemap: https://x/s.xml\nUser-agent: * # all\nDisallow: /p # private\n"), "/p/1"))
check("robots: percent-encoding compared decoded", not A(R("User-agent: *\nDisallow: /caf%C3%A9\n"), "/caf\u00e9/menu"))

# end to end: the address guard is replaced for this part so that the local server stands in for a public site
real_blocked = PF.blocked_reason
PF.blocked_reason = lambda ip: ""
try:
    rsite = Site({"/robots.txt": (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /secret\n"),
                  "/open": (200, {"Content-Type": HTML}, b"<p>OPEN-PAGE</p>"),
                  "/secret": (200, {"Content-Type": HTML}, b"<p>ROBOTS-SECRET</p>")})
    check("robots e2e: an allowed page is read", PF.read_page(rsite.base + "/open")["text"] == "OPEN-PAGE")
    check("robots e2e: a disallowed page is refused, pointing to the browser",
          refused(lambda: PF.read_page(rsite.base + "/secret"), "robots.txt", "browser"))
    check("robots e2e: the disallowed page was never requested", "/secret" not in rsite.hits)
    check("robots e2e: robots.txt fetched once (cached)", rsite.hits.count("/robots.txt") == 1)
    check("robots e2e: a page the user named is read without the check",
          PF.read_page(rsite.base + "/secret", typed=rsite.me)["text"] == "ROBOTS-SECRET")
    gone = Site({"/page": (200, {"Content-Type": HTML}, b"<p>NO-ROBOTS-FILE</p>")})
    check("robots e2e: robots.txt 404 allows", PF.read_page(gone.base + "/page")["text"] == "NO-ROBOTS-FILE")
    down = Site({"/robots.txt": (503, {"Content-Type": "text/plain"}, b"busy"),
                 "/page": (200, {"Content-Type": HTML}, b"<p>SHOULD-NOT-READ</p>")})
    check("robots e2e: robots.txt 5xx means the site is not read", refused(lambda: PF.read_page(down.base + "/page"), "robots.txt"))
    for s in (rsite, gone, down):
        s.close()
finally:
    PF.blocked_reason = real_blocked
check("the address guard is back", PF.blocked_reason("127.0.0.1") == "this device")

# ---------------- the tools ----------------
import web_search  # noqa: E402

w = web_search.WebSearch(None)
online = []  # a refused link falls back to the keyless search sources: kept offline here, and counted
w.public()._search_wikipedia = lambda *a, **k: online.append("wikipedia") or []
w.public()._search_arxiv = lambda *a, **k: online.append("arxiv") or []
w._news_rss_rows = lambda *a, **k: online.append("news") or []
got = w.fetch_url_content(site.base + "/secret")
check("fetch_url: an untyped local page is not read", got.startswith("[URL Fetch Error") and "LOCAL-ONLY-TOKEN" not in got)
got = w.execute("summarise " + site.base + "/secret")
check("web_search: a local link in the query is not read", "LOCAL-ONLY-TOKEN" not in got)
check("web_search: it falls back to searching instead (stubbed here)", "wikipedia" in online)
got = w.fetch_url_content(site.base + "/article", typed=site.me)
check("fetch_url: a typed page comes with its citation line", "Source: Example News, published 2026-10-30" in got)
got = w.execute("summarise " + site.base + "/secret", typed=site.me)
check("web_search: a typed local link is read", "LOCAL-ONLY-TOKEN-1234" in got)
check("web_search: the untyped result was not cached for the typed request (and back)",
      "LOCAL-ONLY-TOKEN" not in w.execute("summarise " + site.base + "/secret"))

import lfm3_native as N  # noqa: E402

eng = object.__new__(N.LFM3Native)
eng._web, eng.turn_hosts = w, N.typed_hosts(f"what does {site.base}/secret say?")
check("engine: hosts come from the user's words", eng.turn_hosts == site.me)
check("engine: fetch_url reads the typed local page", "LOCAL-ONLY-TOKEN" in eng._run_tool("fetch_url", {"url": site.base + "/secret"}))
eng.turn_hosts = N.typed_hosts("File: notes.txt\n" + site.base + "/secret\n---" + N.ATTACHED + "summarise the attached file")
check("engine: a link inside an attached file is not typed", eng.turn_hosts == frozenset())
out = eng._run_tool("fetch_url", {"url": site.base + "/secret"})
check("engine: so the page it names is not read", "LOCAL-ONLY-TOKEN" not in out)
check("engine: a refused read shows as a failed step", N.tool_worked("fetch_url", True, out)[0] is False)
check("engine: a page read shows as worked", N.tool_worked("fetch_url", True, PF.cite_text(page))[0] is True)

site.close()
other.close()
print(f"{n} checks" + (" passed" if ok else ", SOME FAILED"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
