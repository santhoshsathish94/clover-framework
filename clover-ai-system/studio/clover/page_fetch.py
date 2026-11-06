"""Reads one web page on this device for Clover AI Studio (fetch_url, web research): the page's own text and what a
citation needs (title, site, date, the address read).

Browser-first search (2026-11-06): the user searches in their own browser and Clover reads pages. A page address can
come from the model, so this reader:
- reads http and https only, and only public addresses: every address the host name resolves to is checked (this
  device, private networks, link-local, carrier-grade NAT, multicast, reserved, and IPv6 forms that carry such an IPv4
  address are refused) unless the user typed that host and port in their own message;
- connects to the address it checked, with the host name for TLS (certificate check) and the Host header: a second
  DNS lookup could answer differently;
- follows redirects itself, at most 5, each checked the same way;
- caps size and time;
- honours robots.txt (RFC 9309) when Clover reads on its own; a page whose host the user typed is the user's request.
Standard library only: the phone app has no other packages for this. Connections go direct (no system proxy).
"""
import codecs
import datetime
import email.utils
import http.client
import ipaddress
import json
import re
import socket
import ssl
import threading
import time
import urllib.parse
import zlib
from html.parser import HTMLParser

AGENT = "CloverAIStudio"  # the product token robots.txt groups are matched against
USER_AGENT = f"Mozilla/5.0 (compatible; {AGENT}/0.5; +https://cloverframework.com)"
ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"
MAX_BYTES = 2_000_000
TIMEOUT_S = 20.0
MAX_REDIRECTS = 5
REDIRECTS = (301, 302, 303, 307, 308)
ROBOTS_BYTES = 512 * 1024  # RFC 9309: parse at least 500 KiB
ROBOTS_TTL_S, ROBOTS_RETRY_S = 24 * 3600, 600
HTML_TYPES = ("text/html", "application/xhtml+xml")
TEXT_TYPES = ("application/json", "application/xml", "application/rss+xml", "application/atom+xml")


class FetchError(Exception):
    """Why a page was not read, in words for the user and the model."""


# ---------------- addresses ----------------
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def blocked_reason(ip):
    """Why an address is not a public internet address; "" when it is one."""
    try:
        a = ipaddress.ip_address(str(ip).split("%", 1)[0])
    except ValueError:
        return "not an IP address"
    if a.version == 6:
        inner = a.ipv4_mapped or a.sixtofour or (ipaddress.IPv4Address(int(a) & 0xFFFFFFFF) if a in _NAT64 else None)
        if inner is not None and blocked_reason(inner):
            return blocked_reason(inner)
    if a.is_loopback:
        return "this device"
    if a.is_unspecified:
        return "an unspecified address"
    if a.is_link_local:
        return "a link-local address"
    if a.is_multicast:
        return "a multicast address"
    if a.is_private:
        return "a private network"
    if a.is_reserved or getattr(a, "is_site_local", False) or not a.is_global:
        return "not a public internet address"  # 100.64.0.0/10 (carrier-grade NAT) is neither private nor global
    return ""


def _ascii_host(host):
    h = (host or "").strip().lower().rstrip(".")
    try:
        h.encode("ascii")
        return h
    except UnicodeEncodeError:
        return h.encode("idna").decode("ascii")


def _port(parts):
    return parts.port or (443 if parts.scheme.lower() == "https" else 80)


_TYPED_URL = re.compile(r"https?://[^\s<>\"'`]+", re.I)
_TYPED_LOCAL = re.compile(r"(?<![\w@/.:-])(localhost|\d{1,3}(?:\.\d{1,3}){3}|[a-z0-9-]+\.(?:local|lan|internal|home\.arpa))"
                          r"(?::(\d{1,5}))?(?![\w.-])", re.I)


def typed_hosts(text):
    """(host, port) pairs the user wrote in their own message: links, and bare local addresses such as
    192.168.1.10:8080 or printer.local (ports 80 and 443 when none is written). Only these may be private addresses,
    and pages on them are read without the robots.txt check."""
    out, t = set(), str(text or "")
    for m in _TYPED_URL.finditer(t):
        try:
            p = urllib.parse.urlsplit(m.group(0).rstrip(".,;:!?)]}'\""))
            if p.hostname:
                out.add((_ascii_host(p.hostname), _port(p)))
        except (ValueError, UnicodeError):
            continue
    for m in _TYPED_LOCAL.finditer(_TYPED_URL.sub(" ", t)):
        out.update((m.group(1).lower(), p) for p in ((int(m.group(2)),) if m.group(2) else (80, 443)))
    return frozenset(out)


def _target(url, typed):
    """-> (scheme, host, port, addresses, request target), the addresses checked; FetchError when not readable."""
    try:
        p = urllib.parse.urlsplit(str(url or "").strip())
        scheme = p.scheme.lower()
        host = _ascii_host(p.hostname or "")
        port = _port(p)
    except (ValueError, UnicodeError):
        raise FetchError("not a valid web address") from None
    if scheme not in ("http", "https"):
        raise FetchError("only http and https addresses are read")
    if p.username is not None or p.password is not None:
        raise FetchError("addresses with a user name or password are not read")
    if not host:
        raise FetchError("not a valid web address")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise FetchError(f"could not find {host} ({getattr(e, 'strerror', None) or type(e).__name__})") from None
    ips = list(dict.fromkeys(i[4][0] for i in infos))
    if (host, port) not in typed:
        for ip in ips:
            why = blocked_reason(ip)
            if why:
                raise FetchError(f"{host} is {why}" + (f" ({ip})" if ip != host else "") +
                                 "; Clover opens such an address only when you type it in your message")
    path = urllib.parse.quote(p.path or "/", safe="/%:@!$&'()*+,;=~-._")
    if p.query:
        path += "?" + urllib.parse.quote(p.query, safe="/%:@!$&'()*+,;=?~-._")
    return scheme, host, port, ips, path


# ---------------- one request to a checked address ----------------
class _HTTP(http.client.HTTPConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _HTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, ip, timeout, context):
        super().__init__(host, port, timeout=timeout, context=context)
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


_TLS, _TLS_LOCK = None, threading.Lock()


def _tls():
    global _TLS
    with _TLS_LOCK:
        if _TLS is None:
            _TLS = ssl.create_default_context()
        return _TLS


def _read(resp, max_bytes, deadline):
    chunks, n = [], 0
    while n <= max_bytes:
        if time.monotonic() > deadline:
            raise FetchError("the page took too long to arrive")
        try:
            b = resp.read(min(65536, max_bytes + 1 - n))
        except (OSError, http.client.HTTPException) as e:
            raise FetchError(f"the connection broke while reading ({type(e).__name__})") from None
        if not b:
            break
        chunks.append(b)
        n += len(b)
    data = b"".join(chunks)
    return data[:max_bytes], len(data) > max_bytes


def _request(target, deadline, max_bytes, accept):
    scheme, host, port, ips, path = target
    last = None
    for ip in ips[:4]:
        left = deadline - time.monotonic()
        if left <= 0:
            raise FetchError("the page took too long to arrive")
        conn = _HTTPS(host, port, ip, min(10.0, left), _tls()) if scheme == "https" else _HTTP(host, port, ip, min(10.0, left))
        try:
            conn.request("GET", path, headers={"User-Agent": USER_AGENT, "Accept": accept,
                                               "Accept-Language": "en;q=0.9, *;q=0.5"})
            resp = conn.getresponse()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            if resp.status in REDIRECTS:
                return resp.status, headers, b"", False
            body, cut = _read(resp, max_bytes, deadline)
            return resp.status, headers, body, cut
        except ssl.SSLCertVerificationError as e:
            raise FetchError(f"{host}'s certificate could not be verified ({e.verify_message})") from None
        except (OSError, http.client.HTTPException) as e:
            last = e
        finally:
            conn.close()
    reason = "timed out" if isinstance(last, (socket.timeout, TimeoutError)) else type(last).__name__
    raise FetchError(f"could not connect to {host} ({reason})")


def _decompressed(body, headers, max_bytes):
    enc = (headers.get("content-encoding") or "").strip().lower()
    if enc in ("", "identity"):
        return body
    if enc not in ("gzip", "x-gzip", "deflate"):
        raise FetchError(f"the page is compressed as {enc}, which is not read")
    for wbits in ((16 + zlib.MAX_WBITS,) if "gzip" in enc else (zlib.MAX_WBITS, -zlib.MAX_WBITS)):
        try:
            return zlib.decompressobj(wbits).decompress(body, max_bytes)
        except zlib.error:
            continue
    raise FetchError("the page's compression could not be read")


def _host_port(url):
    try:
        p = urllib.parse.urlsplit(str(url or "").strip())
        return (_ascii_host(p.hostname or ""), _port(p))
    except (ValueError, UnicodeError):
        return None


def fetch(url, typed=(), robots=True, max_bytes=MAX_BYTES, timeout=TIMEOUT_S, accept=ACCEPT):
    """GET a page with the checks above. typed: typed_hosts() of the user's message. robots: check robots.txt unless
    the user typed the first page's host. -> {"url" (after redirects), "requested", "status", "headers", "body"
    (bytes, decompressed), "truncated", "user_named"}; FetchError when it is not read."""
    deadline = time.monotonic() + timeout
    typed = frozenset(typed or ())
    start, user_named = str(url or "").strip(), _host_port(url) in typed
    for _ in range(MAX_REDIRECTS + 1):
        target = _target(url, typed)
        if robots and not user_named:
            ok, why = ROBOTS.allowed(url, target, deadline)
            if not ok:
                raise FetchError(why)
        status, headers, body, cut = _request(target, deadline, max_bytes, accept)
        if status in REDIRECTS and headers.get("location"):
            url = urllib.parse.urljoin(url, headers["location"].strip())
            continue
        return {"url": url, "requested": start, "status": status, "headers": headers,
                "body": _decompressed(body, headers, max_bytes), "truncated": cut, "user_named": user_named}
    raise FetchError(f"more than {MAX_REDIRECTS} redirects")


# ---------------- robots.txt (RFC 9309) ----------------
def robots_rules(text, agent=AGENT):
    """The allow/disallow rules of the groups that name `agent` (combined), else of the * groups: [(allow, path)]."""
    groups, cur, rules_seen, me = [], None, False, agent.lower()
    for raw in str(text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if cur is None or rules_seen:
                cur, rules_seen = {"agents": set(), "rules": []}, False
                groups.append(cur)
            cur["agents"].add(val.split("/", 1)[0].strip().lower())
        elif key in ("allow", "disallow") and cur is not None:
            rules_seen = True
            if val:
                cur["rules"].append((key == "allow", val))
    mine = [g for g in groups if me in g["agents"]] or [g for g in groups if "*" in g["agents"]]
    return [r for g in mine for r in g["rules"]]


def _rule_matches(pattern, path):
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    rx = ".*".join(re.escape(urllib.parse.unquote(part)) for part in body.split("*"))
    return re.match(rx + ("$" if anchored else ""), urllib.parse.unquote(path), re.S) is not None


def robots_allows(rules, path):
    """RFC 9309: the longest matching rule decides; an allow rule wins a tie; no match allows."""
    best, allow = -1, True
    for is_allow, pattern in rules:
        if _rule_matches(pattern, path) and (len(pattern) > best or (len(pattern) == best and is_allow)):
            best, allow = len(pattern), is_allow
    return allow


class Robots:
    def __init__(self):
        self.lock, self.cache = threading.Lock(), {}

    def allowed(self, url, target, deadline):
        """-> (ok, why not)."""
        scheme, host, port, _, path = target
        if urllib.parse.urlsplit(url).path == "/robots.txt":
            return True, ""
        key = (scheme, host, port)
        with self.lock:
            hit = self.cache.get(key)
        if hit is None or hit[0] < time.monotonic():
            rules, why = self._load(scheme, host, port, deadline)
            with self.lock:
                if len(self.cache) > 256:
                    self.cache.clear()
                self.cache[key] = (time.monotonic() + (ROBOTS_TTL_S if rules is not None else ROBOTS_RETRY_S), rules, why)
        else:
            rules, why = hit[1], hit[2]
        if rules is None:  # unreachable robots.txt: complete disallow (RFC 9309 2.3.1.4)
            return False, (f"Clover could not read {host}'s robots.txt ({why}), so it does not open that site on its "
                           "own; open the page in your browser instead")
        if robots_allows(rules, path):
            return True, ""
        return False, (f"{host}'s robots.txt asks automated readers not to open this page; open it in your browser "
                       "instead")

    def _load(self, scheme, host, port, deadline):
        netloc = host if port == (443 if scheme == "https" else 80) else f"{host}:{port}"
        if ":" in host:
            netloc = f"[{host}]" if port == (443 if scheme == "https" else 80) else f"[{host}]:{port}"
        try:
            r = fetch(f"{scheme}://{netloc}/robots.txt", robots=False, max_bytes=ROBOTS_BYTES,
                      timeout=max(1.0, min(8.0, deadline - time.monotonic())), accept="text/plain,*/*;q=0.5")
        except FetchError as e:
            return None, str(e)
        if 200 <= r["status"] < 300:
            return robots_rules(_decode(r["body"], r["headers"].get("content-type", ""))), ""
        if 400 <= r["status"] < 500:  # unavailable: no rules (RFC 9309 2.3.1.3)
            return [], ""
        return None, f"HTTP {r['status']}"


ROBOTS = Robots()


# ---------------- text ----------------
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)


def _decode(body, content_type, html=False):
    if body.startswith(codecs.BOM_UTF8):
        return body[3:].decode("utf-8", "replace")
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return body.decode("utf-16", "replace")
    m = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", content_type or "", re.I)
    enc = m.group(1) if m else ""
    if not enc and html:
        m2 = _META_CHARSET.search(body[:4096])
        enc = m2.group(1).decode("ascii", "ignore") if m2 else ""
    if enc.lower() in ("iso-8859-1", "latin1", "latin-1", "us-ascii", "ascii"):
        enc = "cp1252"  # what browsers do (WHATWG Encoding)
    try:
        codecs.lookup(enc or "utf-8")
    except LookupError:
        enc = ""
    return body.decode(enc or "utf-8", "replace")


def _tidy(text):
    lines = [ln.rstrip() for ln in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


DATE_KEYS = ("article:published_time", "og:article:published_time", "datepublished", "date", "dc.date",
             "dc.date.issued", "dcterms.date", "dcterms.created", "pubdate", "publishdate", "publish-date",
             "sailthru.date", "parsely-pub-date", "citation_publication_date", "citation_date")
MODIFIED_KEYS = ("article:modified_time", "og:updated_time", "datemodified", "dcterms.modified", "last-modified")


def _day(s):
    """A date as YYYY-MM-DD when it can be read; "" when there is none."""
    s = " ".join(str(s or "").split())
    if not s:
        return ""
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    try:
        return email.utils.parsedate_to_datetime(s).date().isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    return s[:40] if re.search(r"\b\d{4}\b", s) else ""


def _ld_fields(blobs):
    out = {}

    def walk(o, depth=0):
        if depth > 6:
            return
        if isinstance(o, list):
            for x in o:
                walk(x, depth + 1)
        elif isinstance(o, dict):
            for k in ("datePublished", "dateModified", "headline"):
                if isinstance(o.get(k), str) and k not in out:
                    out[k] = o[k]
            if "@graph" in o:
                walk(o["@graph"], depth + 1)
    for b in blobs:
        try:
            walk(json.loads(b))
        except ValueError:
            continue
    return out


# page furniture inside the content area, by class or id (a whole word of the name): Wikipedia's language menu, edit
# links and navboxes, GitHub's hidden headings, share bars, cookie notices; removed only while it is a small part of
# the page (FURNITURE_SHARE), so a wrapper that happens to be called "with-sidebar" keeps the article it holds
FURNITURE = re.compile(
    r"(?:^|[\s_-])(?:nav|navbar|navbox|navigation|menu|menubar|dropdown|breadcrumbs?|sidebar|toc|toolbar|portlet|footer|"
    r"cookie-?(?:banner|consent|notice|bar)|consent|sharing|share-?(?:bar|buttons?|links?|tools?|icons?)|social|related|"
    r"newsletter|subscribe|promo|advert|ads?|sponsored|popup|modal|jump-link|skip-link|editsection|hatnote|catlinks|"
    r"reflist|references|noprint|printfooter)(?=$|[\s_-])", re.I)
HIDDEN_CLASS = re.compile(r"sr-only|visually-?hidden|screen-?reader-text|a11y-hidden", re.I)
FURNITURE_ROLES = frozenset("navigation banner contentinfo complementary menu menubar toolbar search dialog alertdialog "
                            "tablist".split())
FURNITURE_SHARE = 0.4
OPTIONAL_END = frozenset("html head body p li dt dd tr td th thead tbody tfoot colgroup caption option optgroup rb rt rtc "
                         "rp".split())  # their end tag may be left out: not tracked as regions
VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_ZERO_WIDTH = re.compile("[\u200b-\u200d\u2060\ufeff]")


class _Page(HTMLParser):
    """Readable text of an HTML page (its <article>, else <main>, when that holds the page's text) and its metadata."""
    SKIP = frozenset("script style noscript svg math template iframe object canvas button select textarea nav aside "
                     "footer dialog".split())  # not <form>: ASP.NET pages wrap the whole body in one
    BLOCK = frozenset("address article blockquote br caption dd details div dl dt fieldset figcaption figure h1 h2 h3 h4 "
                      "h5 h6 header hr li main ol p pre section summary table tbody tfoot thead tr ul".split())

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip, self.main, self.article = [], 0, 0
        self.depth, self.regions, self.region_n = {}, [], 0  # open regions: (tag, depth of that tag, region id)
        self.lines, self.cur, self.flags, self.prefix = [], [], (False, False), ""
        self.meta, self.canonical, self.times, self.ld = {}, "", [], []
        self.title, self.in_title, self.in_ld = "", False, False

    def _flush(self):
        if any(d.strip() for d, _ in self.cur):
            self.lines.append((self.prefix, self.cur, self.flags))
            self.prefix = ""
        self.cur = []

    def _furniture(self, tag, a):
        if "hidden" in a or a.get("aria-hidden", "").lower() == "true":
            return True
        if a.get("role", "").lower() in FURNITURE_ROLES:
            return True
        cls = a.get("class", "")
        if tag == "sup" and "reference" in cls.lower():
            return True  # Wikipedia's [1] citation markers
        return bool(HIDDEN_CLASS.search(cls) or FURNITURE.search(cls) or FURNITURE.search(a.get("id", "")))

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").strip().lower()
            if key and a.get("content", "").strip():
                self.meta.setdefault(key, a["content"].strip())
            return
        if tag == "link":
            if "canonical" in a.get("rel", "").lower().split() and a.get("href") and not self.canonical:
                self.canonical = a["href"].strip()
            return
        if self.skip:
            if tag in self.SKIP or tag == "header":
                self.skip.append(tag)
            return
        if tag == "script" and "ld+json" in a.get("type", "").lower():
            self.in_ld = True
            self.ld.append("")
            return
        if tag == "title" and not self.title:
            self.in_title = True
            return
        if tag in self.SKIP or (tag == "header" and not self.main):
            self._flush()
            self.skip.append(tag)
            return
        if tag not in OPTIONAL_END and tag not in VOID:
            d = self.depth[tag] = self.depth.get(tag, 0) + 1
            if self._furniture(tag, a):
                self.region_n += 1
                self.regions.append((tag, d, self.region_n))
        if tag in ("article", "main"):
            self.main += 1
            if tag == "article":
                self.article += 1
        if tag == "time" and self.main and a.get("datetime"):
            self.times.append(a["datetime"].strip())
        if tag in ("td", "th") and self.cur:
            self.cur.append((" | ", ()))
        if tag in self.BLOCK:
            self._flush()
            if tag == "li":
                self.prefix = "• "
            elif tag[0] == "h" and tag[1:].isdigit():
                self.prefix = "#" * int(tag[1:]) + " "

    def handle_endtag(self, tag):
        if self.in_title and tag == "title":
            self.in_title = False
            return
        if self.in_ld and tag == "script":
            self.in_ld = False
            return
        if self.skip:
            if tag in self.skip:
                while self.skip.pop() != tag:
                    pass
            return
        if tag in self.BLOCK:
            self._flush()
        d = self.depth.get(tag, 0)
        if d > 0:
            for i in range(len(self.regions) - 1, -1, -1):
                if self.regions[i][:2] == (tag, d):
                    del self.regions[i:]  # with any region left open inside it
                    break
            self.depth[tag] = d - 1
        if tag in ("article", "main") and self.main:
            self.main -= 1
            if tag == "article" and self.article:
                self.article -= 1

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.in_ld:
            self.ld[-1] += data
        elif not self.skip and (self.cur or data.strip()):
            if not self.cur:
                self.flags = (self.main > 0, self.article > 0)
            self.cur.append((data, tuple(r[2] for r in self.regions)))

    def result(self, base):
        self._flush()
        size, total = {}, 0
        for _, chunks, _ in self.lines:
            for data, regions in chunks:
                n = len(data.strip())
                total += n
                for rid in regions:
                    size[rid] = size.get(rid, 0) + n
        dropped = {rid for rid, n in size.items() if n < FURNITURE_SHARE * total}
        kept = []
        for prefix, chunks, flags in self.lines:
            text = " ".join("".join(d for d, regions in chunks if not dropped.intersection(regions)).split())
            if text:
                kept.append((prefix + text, flags))

        def joined(lines):
            out = []
            for ln in lines:
                if not out or out[-1] != ln:
                    out.append(ln)
            return "\n".join(out)
        every = joined(t for t, _ in kept)
        main = joined(t for t, (m, _) in kept if m)
        article = joined(t for t, (_, art) in kept if art)

        def holds_page(t):
            return bool(t) and (len(t) >= 500 or len(t) >= 0.3 * len(every))
        text = article if holds_page(article) else main if holds_page(main) else every
        text = _ZERO_WIDTH.sub("", text)
        ld, meta = _ld_fields(self.ld), self.meta
        clean = lambda s: " ".join(str(s or "").split())  # noqa: E731
        published = next((meta[k] for k in DATE_KEYS if meta.get(k)), "") or ld.get("datePublished") or (
            self.times[0] if self.times else "")
        modified = next((meta[k] for k in MODIFIED_KEYS if meta.get(k)), "") or ld.get("dateModified") or ""
        return {"title": _ZERO_WIDTH.sub("", clean(meta.get("og:title") or ld.get("headline") or self.title))[:300],
                "site": clean(meta.get("og:site_name"))[:120],  # else the host name (read_page): clearer than a publisher
                "published": _day(published), "modified": _day(modified),
                "description": clean(meta.get("description") or meta.get("og:description"))[:500],
                "canonical": urllib.parse.urljoin(base, self.canonical) if self.canonical else "", "text": text}


def extract(html_text, base=""):
    """Readable text and metadata of an HTML document (see _Page.result)."""
    p = _Page()
    p.feed(html_text)
    p.close()
    return p.result(base)


def _site(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def read_page(url, typed=(), robots=True, max_chars=20000, timeout=TIMEOUT_S):
    """A page's readable text and what a citation needs: {"url", "requested", "title", "site", "published",
    "modified", "description", "canonical", "text", "chars", "truncated", "kind", "fetched", "robots"};
    FetchError when it is not read."""
    r = fetch(url, typed=typed, robots=robots, timeout=timeout)
    if not 200 <= r["status"] < 300:
        raise FetchError(f"the site answered HTTP {r['status']}")
    ctype_full = r["headers"].get("content-type", "")
    ctype = ctype_full.split(";", 1)[0].strip().lower()
    body = r["body"]
    page = {"url": r["url"], "requested": r["requested"], "title": "", "site": "", "published": "", "modified": "",
            "description": "", "canonical": "", "kind": ctype or "unknown", "truncated": r["truncated"],
            "fetched": datetime.date.today().isoformat(),
            "robots": "not checked: you named this site" if r["user_named"] else ("checked" if robots else "not checked")}
    sniffed = body.lstrip()[:14].lower().startswith((b"<!doctype html", b"<html"))
    if ctype in HTML_TYPES or (ctype in ("", "text/plain") and sniffed):
        page.update(extract(_decode(body, ctype_full, html=True), base=r["url"]))
    elif ctype.startswith("text/") or ctype in TEXT_TYPES:
        page["text"] = _tidy(_decode(body, ctype_full))
    else:
        raise FetchError(f"it is a {ctype or 'binary'} file, not a web page; fetch_url reads web pages and text")
    page["site"] = page["site"] or _site(page["url"])
    page["chars"] = len(page["text"])
    if page["chars"] > max_chars:
        page["text"], page["truncated"] = page["text"][:max_chars], True
    return page


def cite_text(page, link=None):
    """The page for the model: a first line in the tools' format, a line with what a citation needs, then the text."""
    url = link or page["url"]
    when = page.get("published") or ""
    bits = [page.get("site") or _site(url), f"published {when}" if when else "publication date not stated"]
    if page.get("modified") and page["modified"] != when:
        bits.append(f"updated {page['modified']}")
    head = (f"[Live Webpage: {page.get('title') or url} — Link: {url}]\nSource: {', '.join(bits)}; read on this device "
            f"on {page.get('fetched') or datetime.date.today().isoformat()}.")
    if page.get("truncated"):
        head += " Only the start of the page is shown."
    return head + "\n" + (page.get("text") or "(the page has no readable text)")
