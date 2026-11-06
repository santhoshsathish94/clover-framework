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

    def fetch_url_content(self, url: str, max_chars: int = 1600, typed=()) -> str:
        """Readable text of a live http/https page, read on this device by page_fetch: public addresses only unless the
        user typed the host, robots.txt honoured when Clover reads on its own, with the site and date for a citation.
        GitHub `/blob/` links are read from `raw.githubusercontent.com`. typed: page_fetch.typed_hosts() of the user's
        message."""
        import page_fetch
        u = (url or "").strip().rstrip(".,;)>\"'")
        if not u:
            return ""
        if not re.match(r"https?://", u, re.I):
            u = "https://" + u

        fetch_target = u
        # Convert GitHub blob URLs (`https://github.com/<owner>/<repo>/blob/<branch>/<path>`) to raw content URLs
        m_gh = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", u)
        if m_gh:
            owner, repo, branch, path_part = m_gh.groups()
            fetch_target = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path_part}"

        try:
            page = page_fetch.read_page(fetch_target, typed=typed, max_chars=max_chars)
        except page_fetch.FetchError as e:
            return f"[URL Fetch Error for {u}: {e}]"
        return page_fetch.cite_text(page, link=u if fetch_target != u else None)

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
                link = (entry.find("a:id", ns).text or "").strip().replace("http://", "https://")
                if len(summary) > 260:
                    summary = summary[:257].rsplit(" ", 1)[0] + "..."
                out.append(f"[arXiv {pub}] {title} — Link: {link} — Summary: {summary}")
        except Exception:
            pass
        return out

    def _search_wikipedia(self, query: str, max_items: int = 3) -> list[str]:
        """Queries the live Wikipedia Search & Summary API and includes canonical article URLs."""
        clean_q = re.sub(
            r"\b(can you|please|do|a|an|google|web|internet|online|find|relevant|research|on|recent|latest|about|search|for|more|details|and|links)\b",
            " ",
            query.lower(),
        )
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
                wiki_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                try:
                    sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
                    req_s = urllib.request.Request(sum_url, headers={"User-Agent": self.user_agent})
                    with urllib.request.urlopen(req_s, timeout=5) as rs:
                        s_data = json.loads(rs.read().decode("utf-8", errors="ignore"))
                    extract_txt = " ".join((s_data.get("extract") or "").split()).strip()
                    if len(extract_txt) >= 30:
                        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", extract_txt) if s.strip()]
                        kept_sents = []
                        cur_chars = 0
                        for s_item in sents:
                            if s_item.count("(") != s_item.count(")"):
                                s_item = re.sub(r"\([^)]*$", "", s_item).strip()
                                if s_item and not s_item.endswith((".", "!", "?")):
                                    s_item += "."
                            if not s_item:
                                continue
                            if kept_sents and cur_chars + len(s_item) > 420:
                                break
                            kept_sents.append(s_item)
                            cur_chars += len(s_item)
                            if len(kept_sents) >= 3:
                                break
                        if kept_sents:
                            snip = " ".join(kept_sents)
                        elif len(extract_txt) > 320:
                            snip = extract_txt[:317].rsplit(" ", 1)[0].rstrip(" ,;(") + "."
                        else:
                            snip = extract_txt
                except Exception:
                    pass
                out.append(f"{title} ({wiki_url}): {snip}")
        except Exception:
            pass
        return out

    def _search_ddg(self, query: str, max_items: int = 3) -> list[str]:
        """Queries DuckDuckGo HTML search endpoint for live web pages and extracts destination URLs."""
        clean_q = re.sub(
            r"^\s*(?:can you\s+|please\s+|do\s+a?\s*)?(?:google\s+search|search\s+google|search\s+the\s+web|search|find|look\s+up|google)\s*(?:for|on|about)?\s*",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        ).strip() or query.strip()
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(clean_q)
        out = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=8) as r:
                page = r.read().decode("utf-8", errors="ignore")
            anchors = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.DOTALL)
            snips = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', page, flags=re.DOTALL)
            for (raw_href, t), s in list(zip(anchors, snips))[:max_items]:
                t_clean = html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                s_clean = html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
                href_clean = html.unescape(raw_href).strip()
                if "uddg=" in href_clean:
                    try:
                        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href_clean).query)
                        if "uddg" in qs and qs["uddg"]:
                            href_clean = qs["uddg"][0]
                    except Exception:
                        pass
                if t_clean and s_clean:
                    if href_clean.startswith("http"):
                        out.append(f"[Web: {t_clean} — Link: {href_clean}] {s_clean}")
                    else:
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

        # 0. If the query contains explicit HTTP/HTTPS URLs or known domain links, fetch those URLs directly first!
        urls_found = re.findall(r"https?://[^\s<>\"')]+", q)
        if not urls_found:
            bare_domains = re.findall(r"\b(?:github\.com|cloverframework\.com|arxiv\.org|en\.wikipedia\.org)/[^\s<>\"')]+", q)
            urls_found = ["https://" + d for d in bare_domains]
        for u in urls_found[:2]:
            fetched = self.fetch_url_content(u)
            if fetched and not fetched.startswith("[URL Fetch Error"):
                results.append(fetched)

        if results and urls_found:
            formatted_url = "\n\n".join(results)
            self.cache[q] = formatted_url
            return formatted_url

        # Only query arXiv when the prompt explicitly asks for research papers, studies, or arXiv;
        # otherwise use Wikipedia + DuckDuckGo so general science/topic queries get encyclopedic facts!
        if any(k in low for k in ("research", "paper", "papers", "arxiv", "study", "studies", "publication", "preprint")):
            results.extend(self._search_arxiv(q, max_items=3))
            results.extend(self._search_wikipedia(q, max_items=2))
            if len(results) < 3:
                results.extend(self._search_ddg(q, max_items=2))
        else:
            results.extend(self._search_wikipedia(q, max_items=3))
            results.extend(self._search_ddg(q, max_items=3))

        if not results:
            return f"No live external results returned for '{q}'."

        formatted = "\n".join(f"• Reference Source {i+1}: {line}" for i, line in enumerate(results[: self.max_results]))
        self.cache[q] = formatted
        return formatted
