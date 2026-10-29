"""Optional on-disk cache of raw search responses, keyed by the exact query. Off unless
HYBRID_SEARCH_CACHE_DIR is set. Entries are content-addressed files written atomically,
only successful responses are stored, and any cache error falls back to the live API,
so results are identical with and without the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = "v1"
_ENV_DIR = "HYBRID_SEARCH_CACHE_DIR"
_MEM_MAX = int(os.environ.get("HYBRID_SEARCH_CACHE_MEM", "2048"))


def enabled() -> bool:
    return bool(os.environ.get(_ENV_DIR, "").strip())


def cache_dir() -> pathlib.Path:
    return pathlib.Path(os.environ[_ENV_DIR])


def _key(query: str, search_depth: str, max_results: int) -> str:
    raw = f"{SCHEMA}|{query}|{search_depth}|{int(max_results)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _path(h: str) -> pathlib.Path:
    return cache_dir() / h[:2] / h[2:4] / f"{h}.json"


class _Stats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self._lock = threading.Lock()

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def summary(self) -> str:
        tot = self.hits + self.misses
        rate = (self.hits / tot) if tot else 0.0
        return (f"search cache: {self.hits} hits / {tot} lookups ({rate:.1%}), "
                f"{self.writes} writes, {self.errors} errors")


STATS = _Stats()

# Bounded in-process LRU in front of the disk.
_mem: "OrderedDict[str, Any]" = OrderedDict()
_mem_lock = threading.Lock()


def get(query: str, search_depth: str, max_results: int) -> Optional[Any]:
    """Cached raw response, or None on miss. Never raises."""
    if not enabled():
        return None
    try:
        h = _key(query, search_depth, max_results)
        with _mem_lock:
            if h in _mem:
                _mem.move_to_end(h)
                STATS.bump("hits")
                return _mem[h]
        p = _path(h)
        if not p.is_file():
            STATS.bump("misses")
            return None
        with p.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            STATS.bump("misses")
            return None
        resp = rec.get("response")
        with _mem_lock:
            _mem[h] = resp
            _mem.move_to_end(h)
            while len(_mem) > _MEM_MAX:
                _mem.popitem(last=False)
        STATS.bump("hits")
        return resp
    except Exception as e:
        STATS.bump("errors")
        logger.debug(f"search cache read failed ({e}); falling through to live API")
        return None


def put(query: str, search_depth: str, max_results: int, response: Any) -> None:
    """Store a successful raw response. Never raises."""
    if not enabled() or response is None:
        return
    try:
        h = _key(query, search_depth, max_results)
        p = _path(h)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"schema": SCHEMA, "query": query, "search_depth": search_depth,
               "max_results": int(max_results), "response": response}
        # Write a temp file in the same directory, then rename: readers never see a partial file.
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        with _mem_lock:
            _mem[h] = response
            _mem.move_to_end(h)
            while len(_mem) > _MEM_MAX:
                _mem.popitem(last=False)
        STATS.bump("writes")
    except Exception as e:
        STATS.bump("errors")
        logger.debug(f"search cache write failed ({e}); continuing without caching")


def stats_summary() -> str:
    return STATS.summary()
