"""Web search for Clover AI Studio.

Searches go to a provider that allows automated use: Tavily or Brave Search with the user's own API key (Settings ->
Web search; kept in config/search.json on this laptop), otherwise Wikipedia and arXiv through their public APIs.
DuckDuckGo is not used: it answers automated requests with an anomaly page (HTTP 202), and working around that block
is not appropriate. research() is the browsing step of the Clover cycle: search, open the best pages, keep the
sentences that bear on the question. Results keep the tools' format (one "Reference Source" line per result).
Pages are read on this device by page_fetch: public addresses only unless the user typed the host, robots.txt honoured
when Clover reads on its own.
"""
import html
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROVIDERS = {
    "wikipedia": {"label": "Web, News, Wikipedia & arXiv", "key": False, "signup": None,
                  "detail": "Free public web search, live news, Wikipedia and arXiv, plus live page reading (no key required)."},
    "tavily": {"label": "Tavily", "key": True, "signup": "https://app.tavily.com/home",
               "detail": "Web search built for AI assistants. Free plan: 1,000 searches a month, no credit card."},
    "brave": {"label": "Brave Search", "key": True, "signup": "https://api-dashboard.search.brave.com/register",
              "detail": "Brave's own web index. $5 of free searches a month (about 1,000); a card is needed to sign up."},
}
UA = "CloverAIStudio/0.3 (local assistant on the user's laptop; +https://cloverframework.com)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
# the user's own browser does the whole-web searching (browser-first, 2026-11-06): Clover only opens the search there,
# with the engine the user picks; the user sends back the pages that matter
BROWSERS = {
    "google": ("Google", "https://www.google.com/search?q="),
    "bing": ("Bing", "https://www.bing.com/search?q="),
    "duckduckgo": ("DuckDuckGo", "https://duckduckgo.com/?q="),
    "brave": ("Brave Search", "https://search.brave.com/search?q="),
    "startpage": ("Startpage", "https://www.startpage.com/do/search?q="),
    "ecosia": ("Ecosia", "https://www.ecosia.org/search?q="),
}


def browser_search_url(query, engine="google"):
    """The address that shows `query` on the search engine the user picked, for the user's own browser."""
    return BROWSERS.get(engine, BROWSERS["google"])[1] + urllib.parse.quote_plus(" ".join(str(query or "").split())[:400])

_KEY = re.compile(r"^[\x21-\x7e]{8,200}$")
_STOP = set("""a an the of to in on at by for with from and or but is are was were be been being it its this that these
those what which who whom whose how many much does do did can could would should will shall may might must than then
there their they them he she his her you your we our i me my as if so not no yes into over under about between per
each any all some more most less least very just also only use using used typical round nearest answer question
following best describes state option letter brief explanation solve step end response number""".split())
_SHARED = {}
_SHARED_LOCK = threading.Lock()
_FILLER = set("""one take takes taken get gets give gives find tell explain true false means did done make makes like
heavier lighter bigger smaller larger longer shorter greater whole has have had when where why know both other according
based result results while during""".split())


class SearchError(Exception):
    pass


class SearchConfig:
    """The chosen provider (config/<device>/search.json in Clover's repository) and the user's keys (CloverAI\\vault,
    encrypted for this Windows account). A key leaves this laptop only to its provider; describe() reports whether one
    is set, never the key."""

    @classmethod
    def shared(cls, path, keys_path=None):
        with _SHARED_LOCK:
            p = str(Path(path).resolve())
            if p not in _SHARED:
                _SHARED[p] = cls(p, keys_path)
            return _SHARED[p]

    def __init__(self, path, keys_path=None):
        self.path, self.lock = Path(path), threading.Lock()
        self.keys_path = Path(keys_path) if keys_path else None
        self.provider, self.keys = "wikipedia", {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        self.browser = saved.get("browser") if saved.get("browser") in BROWSERS else "google"
        found = dict(saved.get("keys") or {})  # earlier versions kept keys beside the provider
        if self.keys_path is not None:
            try:
                import vault
                found.update(json.loads(vault.unprotect(json.loads(self.keys_path.read_text(encoding="utf-8"))["keys_dpapi"])))
            except (OSError, ValueError, KeyError):
                pass
        for name, key in found.items():
            if name in PROVIDERS and PROVIDERS[name]["key"] and isinstance(key, str) and _KEY.match(key.strip()):
                self.keys[name] = key.strip()
        if saved.get("provider") in PROVIDERS and (not PROVIDERS[saved["provider"]]["key"] or saved["provider"] in self.keys):
            self.provider = saved["provider"]
        if saved.get("keys") and self.keys_path is not None:
            self._save(self.provider, self.keys, self.browser)

    def get(self):
        with self.lock:
            return self.provider, dict(self.keys)

    def browser_engine(self):
        with self.lock:
            return self.browser

    def describe(self):
        provider, keys = self.get()
        return {"provider": provider,
                "providers": {n: dict(p, key_set=n in keys) for n, p in PROVIDERS.items()},
                "browser": self.browser_engine(),
                "browsers": {n: {"label": label, "url": url} for n, (label, url) in BROWSERS.items()}}

    def update(self, provider=None, key=None, forget=None, browser=None):
        """provider: switch to it (a provider that needs a key needs one, given now or saved before);
        key: the key for `provider`; forget: a provider whose key is removed (falls back to Wikipedia if in use);
        browser: the search engine Clover opens searches with in the user's own browser (BROWSERS)."""
        with self.lock:
            keys, chosen, engine = dict(self.keys), self.provider, self.browser
            if browser is not None:
                if browser not in BROWSERS:
                    raise ValueError(f"unknown search engine {browser!r}")
                engine = browser
            if forget is not None:
                keys.pop(str(forget), None)
                if chosen == forget:
                    chosen = "wikipedia"
            if provider is not None:
                if provider not in PROVIDERS:
                    raise ValueError(f"unknown search provider {provider!r}")
                if key is not None and PROVIDERS[provider]["key"]:
                    key = str(key).strip()
                    if not _KEY.match(key):
                        raise ValueError("that does not look like an API key (8-200 visible characters, no spaces)")
                    keys[provider] = key
                if PROVIDERS[provider]["key"] and provider not in keys:
                    raise ValueError(f"{PROVIDERS[provider]['label']} needs your API key")
                chosen = provider
            self._save(chosen, keys, engine)
            self.provider, self.keys, self.browser = chosen, keys, engine
        return self.describe()

    def _save(self, provider, keys, browser):
        def put(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)
        if self.keys_path is None:
            put(self.path, {"provider": provider, "browser": browser, "keys": keys})
            return
        import vault
        put(self.keys_path, {"keys_dpapi": vault.protect(json.dumps(keys))})
        put(self.path, {"provider": provider, "browser": browser})


def _text(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(s or ""))).split())


def _http_json(url, body, headers, timeout):
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, **headers},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read(3_000_000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        why = {401: "the key was refused", 403: "the key was refused", 402: "the account needs credit",
               429: "too many searches right now", 432: "the plan's search limit is used up",
               433: "the spending limit is reached"}.get(e.code, "the service answered with an error")
        raise SearchError(f"{why} (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise SearchError(f"could not reach the service ({type(e).__name__})") from None


def search_tavily(q, key, timeout=12):
    data = _http_json("https://api.tavily.com/search",
                      json.dumps({"query": q, "search_depth": "basic", "max_results": 5}).encode("utf-8"),
                      {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, timeout)
    return [{"title": _text(r.get("title")), "url": str(r.get("url")), "snippet": _text(r.get("content"))[:700]}
            for r in data.get("results") or [] if r.get("url")]


def search_brave(q, key, timeout=12):
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": 5})
    data = _http_json(url, None, {"Accept": "application/json", "X-Subscription-Token": key}, timeout)
    return [{"title": _text(r.get("title")), "url": str(r.get("url")), "snippet": _text(r.get("description"))[:700]}
            for r in (data.get("web") or {}).get("results") or [] if r.get("url")]


SEARCH = {"tavily": search_tavily, "brave": search_brave}

# the weather is live data no encyclopedia has: Open-Meteo answers automated requests without a key (free for
# non-commercial use, data CC BY 4.0); the phone's "What's the weather" had nothing to look it up with (2026-11-04)
WEATHER = re.compile(
    r"\bweather\b(?!\s+forecasts?\s+work)|"
    r"\b(?:temperature|rain(?:ing|fall)?|snow(?:ing|fall)?|forecast|humidity|aqi|air\s+quality)\b"
    r"(?![^.?!\n]{0,30}\b(?:sun|star|core|oven|furnace|water|body|room|fever|sales|revenue|budget|demand|work)\b)",
    re.I,
)
_NOT_PLACE = set("""weather forecast forecasts temperature temperatures rain raining rainfall snow snowing humidity wind
today tonight tomorrow tomorrows todays now right current currently latest live this week weekend morning afternoon
evening night what whats is are the like in at for of how hows will it be going to outside report update conditions
condition degrees celsius fahrenheit hourly daily next days day search find get show me tell please check local city
town place area location my here""".split())
WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
       73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers", 81: "rain showers",
       82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
       96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


def weather_place(q):
    """The place a weather query names ("weather in Pune today" -> "Pune"); "" when it names none."""
    words = [w for w in re.findall(r"[^\W\d_][\w.'-]*", q or "") if re.sub(r"'s?$", "", w.lower()) not in _NOT_PLACE]
    return " ".join(words[:4])


def _ip_location(timeout=6):
    """Approximate city and coordinates from the connection's public IP when a weather query names no place."""
    for url in ("https://get.geojs.io/v1/ip/geo.json", "https://ipwho.is/"):
        try:
            d = _http_json(url, None, {}, timeout)
            lat = d.get("latitude")
            lon = d.get("longitude")
            if lat is not None and lon is not None:
                return {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "name": d.get("city") or "",
                    "admin1": d.get("region") or "",
                    "country": d.get("country") or "",
                }
        except Exception:
            continue
    return None


def weather_rows(q, timeout=12):
    """Now and the next two days at the place the query names (or the user's IP location when none is named), from Open-Meteo."""
    place = weather_place(q)
    hit = None
    if place:
        cands = [place, place.split(",")[0].strip(), place.split()[0], place.split()[-1]]
        for name in dict.fromkeys(c for c in cands if c):
            try:
                data = _http_json("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
                    {"name": name, "count": 1, "language": "en", "format": "json"}), None, {}, timeout)
                if data.get("results"):
                    hit = data["results"][0]
                    break
            except SearchError:
                continue
    else:
        hit = _ip_location(min(timeout, 6))
    if hit is None:
        return [{"title": "Weather", "url": "", "snippet": "Could not determine the location for this weather lookup."}]
    lat, lon = hit["latitude"], hit["longitude"]
    w = _http_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}), None, {}, timeout)
    cur, d = w.get("current") or {}, w.get("daily") or {}
    where = ", ".join(x for x in (hit.get("name"), hit.get("admin1"), hit.get("country")) if x) or f"{lat:.2f}, {lon:.2f}"
    loc_note = f"User's current location: {where} (answer with this weather directly; do not ask for their city). " if not place else ""
    lines = [f"{loc_note}Now ({str(cur.get('time', '')).replace('T', ' ')} local time in {where}): {cur.get('temperature_2m')} °C, feels like "
             f"{cur.get('apparent_temperature')} °C, {WMO.get(cur.get('weather_code'), 'weather code ' + str(cur.get('weather_code')))}, "
             f"humidity {cur.get('relative_humidity_2m')}%, wind {cur.get('wind_speed_10m')} km/h."]
    for i, day in enumerate(d.get("time") or []):
        lines.append(f"{day}: {WMO.get(d['weather_code'][i], 'weather code ' + str(d['weather_code'][i]))}, "
                     f"{d['temperature_2m_min'][i]} to {d['temperature_2m_max'][i]} °C, chance of rain "
                     f"{d['precipitation_probability_max'][i]}%.")
    return [{"title": f"Weather in {where} (Open-Meteo.com, CC BY 4.0)",
             "url": f"https://open-meteo.com/en/docs?latitude={lat}&longitude={lon}", "snippet": " ".join(lines)}]


def _terms(text):
    words = re.findall(r"[a-z0-9][a-z0-9\-\.]*", (text or "").lower())
    words += [p for w in words if "-" in w for p in w.split("-")]  # "light-second" also matches "light"
    return {w.rstrip(".")[:-1] if len(w) > 4 and w.endswith("s") else w.rstrip(".") for w in words
            if w.rstrip(".") not in _STOP and len(w.rstrip(".")) > 1}


def relevant_rows(rows, query):
    """Results that clearly match the query: a page about something the query names (every title word in the query),
    or, over title and snippet together, at least two of its words and a third of them, with a title word in common
    (or one word more). Wikipedia answered the milk question with Calcium carbonate, and an access-matrix question
    with Matrix of domination (one word in common); the model took both as evidence."""
    want = _terms(query)
    need = max(2, -(-len(want) // 3))
    keep = []
    for r in rows:
        tt = _terms(r["title"])
        shared = len((tt | _terms(r["snippet"])) & want)
        if (tt and tt <= want) or (shared >= need and (tt & want or shared > need)):
            keep.append(r)
    return keep


def relevant_sentences(page_text, query, k=4, min_hits=2):
    """The sentences of a page that share the most terms with the query (numbers count double: answers are often
    a quantity). Order of the page kept."""
    want = _terms(query)
    if not want:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", page_text or "") if 30 <= len(s.strip()) <= 600]
    scored = []
    for i, s in enumerate(sents):
        got = _terms(s) & want
        score = len(got) + sum(1 for w in got if any(c.isdigit() for c in w)) + (0.5 if re.search(r"\d", s) else 0)
        if len(got) >= min_hits:
            scored.append((score, i, s))
    best = sorted(scored, key=lambda x: (-x[0], x[1]))[:k]
    return [s for _, _, s in sorted(best, key=lambda x: x[1])]


def keywords(q, n=8):
    """A short keyword query for Wikipedia's search, which matches whole questions poorly ("how many grams heavier is
    one liter of milk..." found an article on a nuclear test; "speed sound air 20 degrees" finds Speed of sound)."""
    out, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", q or ""):
        lw = w.lower().strip("'")
        if lw in _STOP or lw in _FILLER or len(lw) < 2 or lw in seen:
            continue
        seen.add(lw)
        out.append(w.strip("'"))
    return " ".join(out[:n])


class WebSearch:
    """web_search / fetch_url for the Studio's tools, and research() for the Clover cycle."""

    def __init__(self, config=None, timeout=12):
        self.config, self.timeout, self._public, self.cache = config, timeout, None, {}

    def public(self):
        if self._public is None:
            from live_web_search import LiveWebSearchTool
            self._public = LiveWebSearchTool(max_results=5)
        return self._public

    def provider(self):
        """(name, key) in use: a key provider only when its key is set."""
        if self.config is None:
            return "wikipedia", None
        name, keys = self.config.get()
        return (name, keys[name]) if PROVIDERS[name]["key"] and name in keys else ("wikipedia", None)

    def _live_finance_rows(self, q):
        out = []
        low = q.lower()
        for coin, sym in (("bitcoin", "BTC"), ("btc", "BTC"), ("ethereum", "ETH"), ("eth", "ETH"), ("solana", "SOL")):
            if re.search(rf"\b{coin}\b", low):
                try:
                    d = _http_json(f"https://api.coinbase.com/v2/prices/{sym}-USD/spot", None, {}, min(self.timeout, 6)).get("data") or {}
                    if d.get("amount"):
                        out.append({
                            "title": f"{sym}/USD Spot Price (Coinbase)",
                            "url": f"https://www.coinbase.com/price/{coin}",
                            "snippet": f"Current live {sym} spot price: ${float(d['amount']):,.2f} USD.",
                        })
                except Exception:
                    pass
                break
        m_fx = re.search(r"\b([A-Za-z]{3})\s+(?:to|in|vs\.?)\s+([A-Za-z]{3})\b", q)
        if m_fx or re.search(r"\bexchange\s+rates?\b", low):
            base = (m_fx.group(1).upper() if m_fx else "USD")
            target = (m_fx.group(2).upper() if m_fx else "INR")
            if base in ("USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SGD", "AED"):
                try:
                    er = _http_json(f"https://open.er-api.com/v6/latest/{base}", None, {}, min(self.timeout, 6))
                    rates = er.get("rates") or {}
                    if target in rates:
                        out.append({
                            "title": f"Exchange Rate {base} to {target} (Open-ER)",
                            "url": "https://open.er-api.com/",
                            "snippet": f"1 {base} = {rates[target]} {target} (updated {er.get('time_last_update_utc', 'today')}).",
                        })
                except Exception:
                    pass
        return out

    def _news_rss_rows(self, q, max_items=4):
        import xml.etree.ElementTree as ET
        clean_q = re.sub(
            r"\b(?:202[567]|january|february|march|april|may|june|july|august|september|october|november|december)\b",
            " ",
            q or "",
            flags=re.I,
        )
        clean_q = " ".join(clean_q.split()).strip()
        topic_left = re.sub(
            r"\b(?:top|latest|current|recent|breaking|today'?s?|today|news|headlines|updates?|summary|summari[sz]e|"
            r"world|in|the|of|for|browse|new|websites?|sites?|web\s*pages?|using|browser|take|capture|grab|snap|"
            r"screen\s*capture|screen\s*shot|screenshot|screen|analy[sz]e|data|and|get|fetch|read|look|at|check|show|me|please|from)\b",
            " ",
            clean_q,
            flags=re.I,
        )
        topic_left = " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", topic_left)).strip()
        rss_q = f"{topic_left} news" if topic_left else clean_q
        out = []
        seen_titles = set()
        # 1. Bing News RSS (returns real publisher URLs in the link query and full 1-2 sentence summaries)
        if topic_left:
            try:
                b_url = "https://www.bing.com/news/search?q=" + urllib.parse.quote(rss_q or q) + "&format=rss"
                req_b = urllib.request.Request(b_url, headers={"User-Agent": BROWSER_UA})
                with urllib.request.urlopen(req_b, timeout=min(self.timeout, 6)) as rb:
                    b_root = ET.fromstring(rb.read(1_500_000))
                for it in b_root.findall(".//item")[:max_items]:
                    title = _text(it.findtext("title") or "")
                    desc = _text(it.findtext("description") or "")
                    link = (it.findtext("link") or "").strip()
                    m_u = re.search(r"[?&]url=([^&]+)", link)
                    real_u = urllib.parse.unquote(m_u.group(1)) if m_u else link
                    if title and title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        snip = f"[Live News] {desc}" if desc and desc != title else f"[Live News] {title}"
                        out.append({"title": title, "url": real_u if real_u.startswith("http") else "", "snippet": snip[:550]})
            except Exception:
                pass
        if len(out) >= max_items:
            return out[:max_items]
        # 2. Google News RSS
        if not topic_left:
            url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        else:
            url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(rss_q or q) + "&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=min(self.timeout, 8)) as r:
                root = ET.fromstring(r.read(1_500_000))
            for it in root.findall(".//item")[:max_items]:
                title = _text(it.findtext("title") or "")
                desc = _text(it.findtext("description") or "")
                if title and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    snip = f"[Live News] {desc}" if desc and desc != title else f"[Live News] {title}"
                    out.append({"title": title, "url": "", "snippet": snip[:500]})
                    if len(out) >= max_items:
                        break
        except Exception:
            pass
        return out

    def _enrich_top_page(self, rows, q):
        """Read and parse the top live web page via page_fetch so the model gets extracted article sentences, not only short search snippets."""
        t = self.public()
        for r in rows[:2]:
            u = str(r.get("url") or "")
            if not u.startswith("http") or any(d in u for d in ("wikipedia.org", "arxiv.org", "news.google.com", "youtube.com")):
                continue
            try:
                page = t.fetch_url_content(u, max_chars=8000)
                if not page or page.startswith("[URL Fetch Error"):
                    continue
                sents = relevant_sentences(page, q, k=3, min_hits=1)
                if sents:
                    extra = " ".join(sents)[:450]
                    if extra and extra[:60].lower() not in (r.get("snippet") or "").lower():
                        r["snippet"] = (str(r.get("snippet") or "").rstrip(".") + ". " + extra)[:750]
                break
            except Exception:
                continue

    def _public_rows(self, q, typed=()):
        t = self.public()
        rows = []
        seen_urls = set()

        def add_row(title, url, snippet):
            u_norm = (url or "").rstrip("/").lower()
            if u_norm and u_norm in seen_urls:
                return
            if u_norm:
                seen_urls.add(u_norm)
            rows.append({"title": title, "url": url, "snippet": snippet})

        for u in re.findall(r"https?://[^\s<>\"')]+", q)[:2]:
            got = t.fetch_url_content(u, typed=typed)
            if got and not got.startswith("[URL Fetch Error"):
                add_row(u, u, got[:1500])
        if rows:
            return rows
        for r in self._live_finance_rows(q):
            add_row(r["title"], r["url"], r["snippet"])
        want_news = bool(re.search(
            r"\b(?:news|latest|today|yesterday|current|recent|202[567]|price|rate|score|won|match|launch|update|release|"
            r"headline|midterms?|elections?|polls?|results?|senate|congress|parliament|winner|standings)\b",
            q, re.I,
        ))
        want_places = bool(re.search(
            r"\b(?:restaurants?|caf[e\u00e9]s?|hotels?|places?|shops?|stores?|bars?|baker(?:y|ies)|diner|bistro|near\s+me|best\s+\w+\s+in)\b",
            q, re.I,
        ))
        search_web = getattr(t, "_search_web", None)
        if want_news:
            explicit_news = bool(re.search(r"\b(?:news|headlines)\b", q, re.I))
            if explicit_news:
                for r in self._news_rss_rows(q, max_items=4):
                    add_row(r["title"], r["url"], r["snippet"])
            if callable(search_web) and len(rows) < 4:
                for r in search_web(q, max_items=3):
                    add_row(r["title"], r["url"], r["snippet"])
            if not explicit_news or len(rows) < 3:
                for r in self._news_rss_rows(q, max_items=4):
                    add_row(r["title"], r["url"], r["snippet"])
            self._enrich_top_page(rows, q)
            if len(rows) >= 3:
                return rows[:5]
        if want_places and callable(search_web):
            for r in search_web(q, max_items=4):
                add_row(r["title"], r["url"], r["snippet"])
            self._enrich_top_page(rows, q)
        if re.search(r"\b(?:research|paper|papers|arxiv|study|studies|preprint|publication)\b", q, re.I):
            for line in t._search_arxiv(q, max_items=3):
                m = re.search(r"Link: (\S+)", line)
                add_row(line.split(" — ")[0], m.group(1) if m else "", line)
        if callable(search_web) and len(rows) < 3:
            for r in search_web(q, max_items=3):
                add_row(r["title"], r["url"], r["snippet"])
            self._enrich_top_page(rows, q)
        for line in t._search_wikipedia(keywords(q) or q, max_items=max(2, 5 - len(rows))):
            m = re.match(r"(.*?) \((https?://[^)]+)\): (.*)$", line)
            if m:
                add_row(m.group(1), m.group(2), m.group(3))
            else:
                add_row("", "", line)
        if not want_news and len(rows) < 4:
            for r in self._news_rss_rows(q, max_items=2):
                add_row(r["title"], r["url"], r["snippet"])
        return rows[:5]

    def search(self, q, typed=()):
        """-> (rows, provider label, note). A failing key provider falls back to Wikipedia and arXiv, with a note.
        typed: page_fetch.typed_hosts() of the user's message (a link in the query is read only if public or typed)."""
        name, key = self.provider()
        note = ""
        if WEATHER.search(q):
            try:
                rows = weather_rows(q, self.timeout)
                if rows:
                    return rows, "Open-Meteo", ""
            except (SearchError, KeyError, IndexError, TypeError):
                pass
        if name != "wikipedia":
            try:
                rows = SEARCH[name](q, key, self.timeout)
                if rows:
                    return rows, PROVIDERS[name]["label"], ""
                note = f"{PROVIDERS[name]['label']} found nothing; these results are from Wikipedia."
            except SearchError as exc:
                note = f"{PROVIDERS[name]['label']} search failed: {exc}; these results are from Wikipedia."
        return self._public_rows(q, typed), PROVIDERS["wikipedia"]["label"], note

    def execute(self, query, typed=()):
        q = " ".join(str(query or "").split())[:400]
        if not q:
            return "Empty search query."
        ck = ("search", self.provider()[0], q, tuple(sorted(typed or ())))
        if ck in self.cache:
            return self.cache[ck]
        rows, label, note = self.search(q, typed)
        if not rows:
            return (note + "\n" if note else "") + f"No results for '{q}'."
        out = (f"({note})\n" if note else "") + "\n".join(
            f"• Reference Source {i}: {r['title']} ({r['url']}): {r['snippet']}" if r["url"] else
            f"• Reference Source {i}: {r['snippet']}" for i, r in enumerate(rows[:5], 1))
        if label != "Open-Meteo":  # the weather changes; a chat may ask again later
            self.cache[ck] = out
        return out

    def fetch_url_content(self, url, max_chars=5000, typed=()):
        u = str(url or "").strip().rstrip(").,]\"'")
        if "news.google.com/rss/articles/" in u:
            return "The headline and summary are already provided in the web_search result above; answer directly from those results."
        return self.public().fetch_url_content(u, max_chars=max_chars, typed=typed)

    def research(self, query, pages=2, max_chars=3000):
        """Search, keep the results that match the query, read the top pages and keep the sentences that bear on it.
        Starts with "Searched " only when something relevant was found."""
        q = " ".join(str(query or "").split())[:400]
        if not q:
            return "Empty search query."
        ck = ("research", self.provider()[0], q)
        if ck in self.cache:
            return self.cache[ck]
        found, label, note = self.search(q)
        rows = relevant_rows(found, q)
        if not rows:
            out = (f"No results from {label} matched the question" + (f" ({note})" if note else "") +
                   (": " + "; ".join(r["title"] for r in found[:4] if r["title"]) if found else "") + ".")
            self.cache[ck] = out
            return out
        lines = [f"Searched {label} for: {q}" + (f" ({note})" if note else "")]
        lines += [f"• {r['title']} ({r['url']}): {r['snippet'][:500]}" if r["url"] else f"• {r['snippet'][:500]}"
                  for r in rows[:4]]
        read = 0
        for r in rows:
            if read >= pages or not r["url"].startswith("http") or "arxiv.org" in r["url"]:
                continue
            page = self.public().fetch_url_content(r["url"], max_chars=60000)
            if not page or page.startswith("[URL Fetch Error"):
                continue
            read += 1
            sents = relevant_sentences(page, q)
            if sents:
                lines.append(f"From {r['url']}:")
                lines += [f"- {s}" for s in sents]
        out = "\n".join(lines)[:max_chars]
        self.cache[ck] = out
        return out

    def test(self):
        """One search with the chosen provider, no fallback: (ok, message)."""
        name, key = self.provider()
        try:
            rows = SEARCH[name]("Clover framework", key, self.timeout) if name != "wikipedia" else self._public_rows("Clover plant")
        except SearchError as exc:
            return False, f"{PROVIDERS[name]['label']}: {exc}"
        return (bool(rows), f"{PROVIDERS[name]['label']}: {len(rows)} results" if rows else
                f"{PROVIDERS[name]['label']}: no results")
