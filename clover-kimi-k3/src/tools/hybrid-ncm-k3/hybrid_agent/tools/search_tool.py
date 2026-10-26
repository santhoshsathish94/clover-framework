"""Live search tool backed by the Tavily API: returns the top-k result snippets as
numbered "Page i: ..." lines, the observation format the agent is trained on.
"""
import os
import time
import logging
from typing import Dict

from .base import BaseTool
from . import search_cache as _search_cache

logger = logging.getLogger(__name__)


class TavilySearchTool(BaseTool):

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 10,
        result_length: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY env var or pass api_key=."
            )
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError("tavily-python is not installed. Run: pip install tavily-python")

        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results
        self._result_length = result_length
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def execute(self, query: str, **kwargs) -> str:
        if not query or not query.strip():
            return "Empty search query."

        query = query.strip()
        if query in self._cache:
            return self._cache[query]

        # Optional on-disk cache of raw responses; a hit is formatted exactly like a live call.
        _cached = _search_cache.get(query, "basic", self._max_results)
        if _cached is not None:
            formatted = self._format_results(_cached)
            if formatted:
                self._cache[query] = formatted
                return formatted
            return "No search results found."

        for attempt in range(self._max_retries):
            try:
                response = self._client.search(
                    query=query,
                    max_results=self._max_results,
                    search_depth="basic",
                    timeout=60,
                )
                formatted = self._format_results(response)
                if formatted:
                    _search_cache.put(query, "basic", self._max_results, response)
                    self._cache[query] = formatted
                    return formatted
                return "No search results found."
            except Exception as e:
                logger.warning(f"Tavily search attempt {attempt + 1}/{self._max_retries} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)

        return f"Search failed after {self._max_retries} attempts for query: {query}"

    def _format_results(self, response: dict) -> str | None:
        results = response.get("results", [])
        if not results:
            return None

        formatted = []
        for idx, item in enumerate(results[: self._max_results], 1):
            snippet = (item.get("content") or "").strip()
            if snippet:
                snippet = snippet[: self._result_length]
                formatted.append(f"Page {idx}: {snippet}")

        return "\n".join(formatted) if formatted else None
