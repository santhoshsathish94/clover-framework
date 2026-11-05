#!/usr/bin/env python3
"""Fetch IEEE 9617564 and IEEE 11675788 metadata and arXiv PDFs, and inspect the full code of Voyager, Reflexion,
MAML, and Swarm Learning."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/opt/clover-k3/meaning-core/collective-learning")
PAPERS = ROOT / "papers"
REPOS = ROOT / "repos"


def fetch_json(url, timeout=20):
    time.sleep(1.0)
    req = urllib.request.Request(url, headers={"User-Agent": "CloverResearch/1.0 (mailto:research@cloverframework.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def main():
    out = {}
    for key, q in [
        ("p2_fa_2022", "Federated Analytics Opportunities and Challenges"),
        ("p8_priv_kd_fl_2026", "Privacy and Security in Knowledge Distillation for Federated Learning"),
    ]:
        url = f"https://api.openalex.org/works?filter=title.search:{urllib.parse.quote(q)}&per-page=5"
        try:
            data = fetch_json(url)
            items = []
            for w in data.get("results", []):
                inv = w.get("abstract_inverted_index") or {}
                words = {}
                for word, pos_list in inv.items():
                    for pos in pos_list:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words))
                items.append({
                    "title": w.get("title"),
                    "doi": w.get("doi"),
                    "year": w.get("publication_year"),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "oa_url": (w.get("open_access") or {}).get("oa_url"),
                    "authors": [a.get("author", {}).get("display_name") for a in w.get("authorships", [])],
                    "abstract": abstract,
                })
            out[key] = items
        except Exception as ex:
            out[key] = {"error": str(ex)}

    (ROOT / "stage1_ieee_papers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
