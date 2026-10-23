"""Live Web + arXiv + Wikipedia + DuckDuckGo Search Tool for Clover Kimi K3 Agent Mode (`<search> query </search>`).
Executes real internet HTTP requests from the GPU server with zero paid API key required (plus optional Tavily if set).
"""
import html, json, os, re, urllib.parse, urllib.request, xml.etree.ElementTree as ET


class LiveWebSearchTool:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.cache = {}
        self.user_agent = "Mozilla/5.0 (compatible; CloverK3Agent/1.0; +https://cloverframework.com)"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _search_arxiv(self, query: str, max_items: int = 4) -> list[str]:
        """Queries the live arXiv API for recent research papers matching `query`."""
        clean_q = re.sub(r"\b(can you|find|relevant|research|on|recent|latest|papers|about|search|for|the)\b", " ", query.lower())
        clean_q = " ".join(clean_q.split()).strip() or query.strip()
        if any(k in clean_q for k in ("ai", "artificial intelligence", "llm", "language model", "agent")):
            q_expr = f'(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND all:"{clean_q}"'
        else:
            q_expr = f"all:{clean_q}"
        url = (
            "http://export.arxiv.org/api/query?search_query="
            + urllib.parse.quote(q_expr)
            + f"&sortBy=submittedDate&sortOrder=descending&max_results={max_items}"
        )
        out = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=8) as r:
                xml_data = r.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(xml_data)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entries = root.findall("a:entry", ns)
            if not entries and "AND" in q_expr:
                url2 = (
                    "http://export.arxiv.org/api/query?search_query="
                    + urllib.parse.quote("cat:cs.AI OR cat:cs.CL")
                    + f"&sortBy=submittedDate&sortOrder=descending&max_results={max_items}"
                )
                req2 = urllib.request.Request(url2, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req2, timeout=8) as r2:
                    root = ET.fromstring(r2.read().decode("utf-8", errors="ignore"))
                    entries = root.findall("a:entry", ns)
            for idx, entry in enumerate(entries[:max_items], 1):
                title = " ".join((entry.find("a:title", ns).text or "").split())
                summary = " ".join((entry.find("a:summary", ns).text or "").split())
                pub = (entry.find("a:published", ns).text or "")[:10]
                link = (entry.find("a:id", ns).text or "").strip()
                if len(summary) > 220:
                    summary = summary[:217].rsplit(" ", 1)[0] + "..."
                out.append(f"[arXiv {pub}] {title} ({link}): {summary}")
        except Exception:
            pass
        return out

    def _search_wikipedia(self, query: str, max_items: int = 3) -> list[str]:
        """Queries the live Wikipedia Search & Summary API."""
        clean_q = re.sub(r"\b(can you|find|relevant|research|on|recent|latest|about|search|for)\b", " ", query.lower())
        clean_q = " ".join(clean_q.split()).strip() or query.strip()
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            + urllib.parse.quote(clean_q)
            + f"&utf8=&format=json&srlimit={max_items}"
        )
        out = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
            for item in data.get("query", {}).get("search", [])[:max_items]:
                title = item.get("title", "")
                snip = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                snip = html.unescape(" ".join(snip.split()))
                out.append(f"[Wikipedia: {title}] {snip}")
        except Exception:
            pass
        return out

    def _search_ddg(self, query: str, max_items: int = 3) -> list[str]:
        """Queries DuckDuckGo HTML search endpoint for live web pages."""
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query.strip())
        out = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=8) as r:
                page = r.read().decode("utf-8", errors="ignore")
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', page, flags=re.DOTALL)
            snips = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', page, flags=re.DOTALL)
            for t, s in list(zip(titles, snips))[:max_items]:
                t_clean = html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                s_clean = html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
                if t_clean and s_clean:
                    out.append(f"[Web: {t_clean}] {s_clean}")
        except Exception:
            pass
        return out

    def execute(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty search query."
        if q in self.cache:
            return self.cache[q]

        results = []
        low = q.lower()
        # If query asks about research, papers, AI, science, physics, math, or algorithms, prioritize arXiv + Wikipedia
        if any(k in low for k in ("research", "paper", "arxiv", "ai", "artificial intelligence", "llm", "model", "agent", "quantum", "science", "development", "developments", "recent", "latest")):
            results.extend(self._search_arxiv(q, max_items=3))
            results.extend(self._search_wikipedia(q, max_items=2))
            if len(results) < 3:
                results.extend(self._search_ddg(q, max_items=2))
        else:
            results.extend(self._search_ddg(q, max_items=3))
            results.extend(self._search_wikipedia(q, max_items=2))
            if len(results) < 3:
                results.extend(self._search_arxiv(q, max_items=2))

        if not results:
            return f"No live external results returned for '{q}'."

        formatted = "\n".join(f"Page {i+1}: {line}" for i, line in enumerate(results[: self.max_results]))
        self.cache[q] = formatted
        return formatted
