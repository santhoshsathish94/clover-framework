#!/usr/bin/env python3
"""Inspect the actual source files of the cloned repositories and extracted paper texts on k3,
and query OpenAlex / Semantic Scholar for IEEE 9617564 and IEEE 11675788."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/opt/clover-k3/meaning-core/collective-learning")
REPOS = ROOT / "repos"
PAPERS = ROOT / "papers"
OUT = ROOT / "stage1_source_details.json"


def fetch_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "CloverResearch/1.0 (mailto:research@cloverframework.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def read_head(path: Path, max_lines=160):
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[:max_lines])


def main():
    details = {"paper_heads": {}, "repo_sources": {}, "openalex": {}}
    for txt in sorted(PAPERS.glob("*.txt")):
        details["paper_heads"][txt.name] = read_head(txt, 90)

    # Read exact source files from cloned repos
    source_paths = {
        "voyager_skill": REPOS / "voyager/voyager/agents/skill.py",
        "voyager_critic": REPOS / "voyager/voyager/agents/critic.py",
        "voyager_main": REPOS / "voyager/voyager/voyager.py",
        "reflexion_prog": REPOS / "reflexion/programming_runs/reflexion.py",
        "reflexion_alfworld": REPOS / "reflexion/alfworld_runs/generate_reflections.py",
        "maml_core": REPOS / "maml/maml.py",
        "swarm_client": REPOS / "swarm_learning/lib/src/python-client/swarmlearning/client/swarm.py",
    }
    for k, p in source_paths.items():
        details["repo_sources"][k] = p.read_text(errors="replace") if p.exists() else "MISSING"

    # Query OpenAlex for the exact 10 papers to get complete metadata & abstracts
    queries = {
        "p2_federated_analytics": "https://api.openalex.org/works?search=Federated+Analytics+Opportunities+and+Challenges&per-page=3",
        "p3_kd_fl_2025": "https://api.openalex.org/works/https://doi.org/10.1007/s10791-025-09657-4",
        "p7_swarm_2021": "https://api.openalex.org/works/https://doi.org/10.1038/s41586-021-03583-3",
        "p8_privacy_kd_fl_2026": "https://api.openalex.org/works?search=Privacy+and+Security+in+Knowledge+Distillation+for+Federated+Learning&per-page=3",
        "p9_robust_fl_2025": "https://api.openalex.org/works/https://doi.org/10.1145/3727643",
    }
    for k, url in queries.items():
        try:
            data = fetch_json(url)
            if "results" in data:
                items = []
                for w in data["results"]:
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
                        "authors": [a.get("author", {}).get("display_name") for a in w.get("authorships", [])],
                        "abstract": abstract[:1500],
                    })
                details["openalex"][k] = items
            else:
                inv = data.get("abstract_inverted_index") or {}
                words = {}
                for word, pos_list in inv.items():
                    for pos in pos_list:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words))
                details["openalex"][k] = {
                    "title": data.get("title"),
                    "doi": data.get("doi"),
                    "year": data.get("publication_year"),
                    "venue": ((data.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "authors": [a.get("author", {}).get("display_name") for a in data.get("authorships", [])],
                    "abstract": abstract[:1500],
                }
        except Exception as ex:
            details["openalex"][k] = {"error": str(ex)}

    OUT.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
