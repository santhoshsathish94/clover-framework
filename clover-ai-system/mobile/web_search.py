#!/usr/bin/env python3
"""
S3 · Tools — Multi-Provider Web Search & Relevance Filter (`web_search.py`).

Ported from `laptop/clover/web_search.py` (`Step 15`):
  - Supports **Tavily** and **Brave Search** (with the user's own API key encrypted in `vault/search-keys.json`
    via `mobile_vault`), and **Wikipedia** (via keyword extraction `keywords(q)`) + **arXiv** without a key.
  - Implements `relevant_rows()` and `relevant_sentences()` inside `research()` so unrelated pages (e.g. a
    single-word match like "Calcium carbonate" for a milk density question) are filtered out and never poison
    `run_cycle()` or `run_retry_cycle()`.
"""

from __future__ import annotations

import html
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import mobile_vault

PROVIDERS = {
    "wikipedia": {
        "label": "Wikipedia and arXiv",
        "key": False,
        "signup": None,
        "detail": "Free public APIs, no key. Good for encyclopedic & scientific facts; weak for breaking news.",
    },
    "tavily": {
        "label": "Tavily",
        "key": True,
        "signup": "https://app.tavily.com/home",
        "detail": "Web search built for AI assistants. Free plan: 1,000 searches a month, no credit card.",
    },
    "brave": {
        "label": "Brave Search",
        "key": True,
        "signup": "https://api-dashboard.search.brave.com/register",
        "detail": "Brave's own web index. $5 of free searches a month (about 1,000).",
    },
}
UA = "CloverAIStudio/0.3 (local assistant on the user's device; +https://cloverframework.com)"
_KEY = re.compile(r"^[\x21-\x7e]{8,200}$")
_STOP = set(
    """a an the of to in on at by for with from and or but is are was were be been being it its this that these
those what which who whom whose how many much does do did can could would should will shall may might must than then
there their they them he she his her you your we our i me my as if so not no yes into over under about between per
each any all some more most less least very just also only use using used typical round nearest answer question
following best describes state option letter brief explanation solve step end response number""".split()
)
_SHARED: dict[str, "SearchConfig"] = {}
_SHARED_LOCK = threading.Lock()
_FILLER = set(
    """one take takes taken get gets give gives find tell explain true false means did done make makes like
heavier lighter bigger smaller larger longer shorter greater whole has have had when where why know both other according
based result results while during""".split()
)


class SearchError(Exception):
    pass


class SearchConfig:
    """Stores the chosen search provider (`config/<device>/search.json`) and encrypted API keys (`vault/search-keys.json`)."""

    @classmethod
    def shared(cls, path: Path, keys_path: Path | None = None) -> "SearchConfig":
        with _SHARED_LOCK:
            p = str(Path(path).resolve())
            if p not in _SHARED:
                _SHARED[p] = cls(path, keys_path)
            return _SHARED[p]

    def __init__(self, path: Path, keys_path: Path | None = None):
        self.path, self.lock = Path(path), threading.Lock()
        self.keys_path = Path(keys_path) if keys_path else None
        self.provider: str = "wikipedia"
        self.keys: dict[str, str] = {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        found = dict(saved.get("keys") or {})
        if self.keys_path is not None and self.keys_path.is_file():
            try:
                raw_k = json.loads(self.keys_path.read_text(encoding="utf-8"))
                blob = raw_k.get("keys_Protected") or raw_k.get("keys_dpapi")
                if blob:
                    found.update(json.loads(mobile_vault.unprotect(blob)))
            except (OSError, ValueError, KeyError):
                pass
        for name, key in found.items():
            if name in PROVIDERS and PROVIDERS[name]["key"] and isinstance(key, str) and _KEY.match(key.strip()):
                self.keys[name] = key.strip()
        if saved.get("provider") in PROVIDERS and (
            not PROVIDERS[saved["provider"]]["key"] or saved["provider"] in self.keys
        ):
            self.provider = saved["provider"]
        if saved.get("keys") and self.keys_path is not None:
            self._save(self.provider, self.keys)

    def get(self) -> tuple[str, dict[str, str]]:
        with self.lock:
            return self.provider, dict(self.keys)

    def describe(self) -> dict[str, Any]:
        provider, keys = self.get()
        return {
            "provider": provider,
            "providers": {n: dict(p, key_set=n in keys) for n, p in PROVIDERS.items()},
        }

    def update(self, provider: str | None = None, key: str | None = None, forget: str | None = None) -> dict[str, Any]:
        with self.lock:
            keys, chosen = dict(self.keys), self.provider
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
            self._save(chosen, keys)
            self.provider, self.keys = chosen, keys
        return self.describe()

    def _save(self, provider: str, keys: dict[str, str]) -> None:
        def put(path: Path, data: dict[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)

        if self.keys_path is None:
            put(self.path, {"provider": provider, "keys": keys})
            return
        put(self.keys_path, {"keys_Protected": mobile_vault.protect(json.dumps(keys))})
        put(self.path, {"provider": provider})


def _text(s: Any) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(s or ""))).split())


def _http_json(url: str, body: bytes | None, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": UA, **headers},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read(3_000_000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        why = {
            401: "the key was refused",
            403: "the key was refused",
            402: "the account needs credit",
            429: "too many searches right now",
            432: "the plan's search limit is used up",
            433: "the spending limit is reached",
        }.get(e.code, "the service answered with an error")
        raise SearchError(f"{why} (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise SearchError(f"could not reach the service ({type(e).__name__})") from None


def search_tavily(q: str, key: str, timeout: int = 12) -> list[dict[str, str]]:
    data = _http_json(
        "https://api.tavily.com/search",
        json.dumps({"query": q, "search_depth": "basic", "max_results": 5}).encode("utf-8"),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        timeout,
    )
    return [
        {"title": _text(r.get("title")), "url": str(r.get("url")), "snippet": _text(r.get("content"))[:700]}
        for r in data.get("results") or []
        if r.get("url")
    ]


def search_brave(q: str, key: str, timeout: int = 12) -> list[dict[str, str]]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": 5})
    data = _http_json(url, None, {"Accept": "application/json", "X-Subscription-Token": key}, timeout)
    return [
        {"title": _text(r.get("title")), "url": str(r.get("url")), "snippet": _text(r.get("description"))[:700]}
        for r in (data.get("web") or {}).get("results") or []
        if r.get("url")
    ]


SEARCH = {"tavily": search_tavily, "brave": search_brave}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-\.]*", (text or "").lower())
    words += [p for w in words if "-" in w for p in w.split("-")]
    return {
        w.rstrip(".")[:-1] if len(w) > 4 and w.endswith("s") else w.rstrip(".")
        for w in words
        if w.rstrip(".") not in _STOP and len(w.rstrip(".")) > 1
    }


def relevant_rows(rows: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    """Keeps only results that genuinely match the query (prevents single-word false-positive pages from poisoning answers)."""
    want = _terms(query)
    need = max(2, -(-len(want) // 3))
    keep = []
    for r in rows:
        tt = _terms(r["title"])
        shared = len((tt | _terms(r["snippet"])) & want)
        if (tt and tt <= want) or (shared >= need and (tt & want or shared > need)):
            keep.append(r)
    return keep


def relevant_sentences(page_text: str, query: str, k: int = 4, min_hits: int = 2) -> list[str]:
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


def keywords(q: str, n: int = 8) -> str:
    out, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", q or ""):
        lw = w.lower().strip("'")
        if lw in _STOP or lw in _FILLER or len(lw) < 2 or lw in seen:
            continue
        seen.add(lw)
        out.append(w.strip("'"))
    return " ".join(out[:n])


class WebSearch:
    """`web_search` / `fetch_url` for S3 tools, and `research()` with strict relevance filtering for the Clover cycle."""

    def __init__(self, config: SearchConfig | None = None, timeout: int = 12):
        self.config, self.timeout, self._public, self.cache = config, timeout, None, {}

    def public(self) -> Any:
        if self._public is None:
            from live_web_search import LiveWebSearchTool
            self._public = LiveWebSearchTool(max_results=5)
        return self._public

    def provider(self) -> tuple[str, str | None]:
        if self.config is None:
            return "wikipedia", None
        name, keys = self.config.get()
        return (name, keys[name]) if PROVIDERS[name]["key"] and name in keys else ("wikipedia", None)

    def _public_rows(self, q: str) -> list[dict[str, str]]:
        t = self.public()
        rows: list[dict[str, str]] = []
        for u in re.findall(r"https?://[^\s<>\"')]+", q)[:2]:
            got = t.fetch_url_content(u)
            if got and not got.startswith("[URL Fetch Error"):
                rows.append({"title": u, "url": u, "snippet": got[:1500]})
        if rows:
            return rows
        if re.search(r"\b(?:research|paper|papers|arxiv|study|studies|preprint|publication)\b", q, re.I):
            for line in t._search_arxiv(q, max_items=3):
                m = re.search(r"Link: (\S+)", line)
                rows.append({"title": line.split(" — ")[0], "url": m.group(1) if m else "", "snippet": line})
        for line in t._search_wikipedia(keywords(q) or q, max_items=4 - min(len(rows), 2)):
            m = re.match(r"(.*?) \((https?://[^)]+)\): (.*)$", line)
            rows.append(
                {"title": m.group(1), "url": m.group(2), "snippet": m.group(3)}
                if m
                else {"title": "", "url": "", "snippet": line}
            )
        return rows

    def search(self, q: str) -> tuple[list[dict[str, str]], str, str]:
        name, key = self.provider()
        note = ""
        if name != "wikipedia" and key:
            try:
                rows = SEARCH[name](q, key, self.timeout)
                if rows:
                    return rows, PROVIDERS[name]["label"], ""
                note = f"{PROVIDERS[name]['label']} found nothing; these results are from Wikipedia."
            except SearchError as exc:
                note = f"{PROVIDERS[name]['label']} search failed: {exc}; these results are from Wikipedia."
        return self._public_rows(q), PROVIDERS["wikipedia"]["label"], note

    def execute(self, query: str) -> str:
        q = " ".join(str(query or "").split())[:400]
        if not q:
            return "Empty search query."
        ck = ("search", self.provider()[0], q)
        if ck in self.cache:
            return self.cache[ck]
        rows, _, note = self.search(q)
        if not rows:
            return (note + "\n" if note else "") + f"No results for '{q}'."
        out = (f"({note})\n" if note else "") + "\n".join(
            f"• Reference Source {i}: {r['title']} ({r['url']}): {r['snippet']}"
            if r["url"]
            else f"• Reference Source {i}: {r['snippet']}"
            for i, r in enumerate(rows[:5], 1)
        )
        self.cache[ck] = out
        return out

    def fetch_url_content(self, url: str, max_chars: int = 5000) -> str:
        return self.public().fetch_url_content(url, max_chars=max_chars)

    def research(self, query: str, pages: int = 2, max_chars: int = 3000) -> str:
        """
        Searches, filters to results that genuinely match `query`, reads the top pages, and keeps
        the relevant sentences. Starts with `"Searched "` ONLY when something relevant was found.
        """
        q = " ".join(str(query or "").split())[:400]
        if not q:
            return "Empty search query."
        ck = ("research", self.provider()[0], q)
        if ck in self.cache:
            return self.cache[ck]
        found, label, note = self.search(q)
        rows = relevant_rows(found, q)
        if not rows:
            out = (
                f"No results from {label} matched the question"
                + (f" ({note})" if note else "")
                + ((": " + "; ".join(r["title"] for r in found[:4] if r["title"])) if found else "")
                + "."
            )
            self.cache[ck] = out
            return out
        lines = [f"Searched {label} for: {q}" + (f" ({note})" if note else "")]
        lines += [
            f"• {r['title']} ({r['url']}): {r['snippet'][:500]}" if r["url"] else f"• {r['snippet'][:500]}"
            for r in rows[:4]
        ]
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

    def test(self) -> tuple[bool, str]:
        name, key = self.provider()
        try:
            rows = (
                SEARCH[name]("Clover framework", key or "", self.timeout)
                if name != "wikipedia"
                else self._public_rows("Clover plant")
            )
        except SearchError as exc:
            return False, f"{PROVIDERS[name]['label']}: {exc}"
        return (
            bool(rows),
            f"{PROVIDERS[name]['label']}: {len(rows)} results" if rows else f"{PROVIDERS[name]['label']}: no results",
        )
