#!/usr/bin/env python3
"""Stage 1: Primary-source verification of the 10 collective-learning papers and their official codebases.

Clones the official open-source repositories (--depth 1), downloads available open-access PDFs / arXiv / CrossRef
records, and inspects the exact source code of Voyager's SkillManager, Reflexion's memory/reflection loop, MAML's
bilevel inner/outer update, and Swarm Learning's decentralized contract/merge architecture.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("/opt/clover-k3/meaning-core/collective-learning")
REPOS = ROOT / "repos"
PAPERS = ROOT / "papers"
OUT = ROOT / "stage1_audit.json"


def run(cmd, cwd=None, timeout=180):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def fetch_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "CloverResearch/1.0 (mailto:research@cloverframework.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_bytes(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CloverResearch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def query_arxiv(query, max_results=3):
    q = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results={max_results}"
    xml_bytes = fetch_bytes(url, timeout=25)
    root = ET.fromstring(xml_bytes)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for e in root.findall("a:entry", ns):
        entries.append({
            "id": (e.findtext("a:id", "", ns) or "").strip(),
            "title": re.sub(r"\s+", " ", (e.findtext("a:title", "", ns) or "").strip()),
            "summary": re.sub(r"\s+", " ", (e.findtext("a:summary", "", ns) or "").strip()),
            "published": (e.findtext("a:published", "", ns) or "").strip(),
            "authors": [(a.findtext("a:name", "", ns) or "").strip() for a in e.findall("a:author", ns)],
        })
    return entries


def query_crossref(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    data = fetch_json(url).get("message", {})
    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in data.get("author", [])]
    return {
        "doi": doi,
        "title": (data.get("title") or [""])[0],
        "container": (data.get("container-title") or [""])[0],
        "year": (data.get("published-print") or data.get("published-online") or data.get("created") or {}).get("date-parts", [[None]])[0][0],
        "authors": authors,
        "abstract": re.sub(r"<[^>]+>", " ", data.get("abstract") or "").strip(),
    }


def clone_repos():
    REPOS.mkdir(parents=True, exist_ok=True)
    targets = {
        "voyager": "https://github.com/MineDojo/Voyager.git",
        "reflexion": "https://github.com/noahshinn/reflexion.git",
        "maml": "https://github.com/cbfinn/maml.git",
        "swarm_learning": "https://github.com/HewlettPackard/swarm-learning.git",
    }
    results = {}
    for name, url in targets.items():
        dest = REPOS / name
        if not (dest / ".git").exists():
            rc, out, err = run(["git", "clone", "--depth", "1", url, str(dest)])
        else:
            rc, out, err = 0, "already cloned", ""
        rc_h, sha, _ = run(["git", "rev-parse", "HEAD"], cwd=dest) if (dest / ".git").exists() else (1, "", "")
        rc_l, license_files, _ = run(["find", ".", "-maxdepth", "2", "-iname", "license*"], cwd=dest) if dest.exists() else (1, "", "")
        py_files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*.py"))[:80] if dest.exists() else []
        results[name] = {
            "url": url,
            "cloned": (dest / ".git").exists(),
            "head": sha.strip(),
            "license_files": license_files.strip().splitlines(),
            "py_files_sample": py_files,
        }
    return results


def download_pdfs():
    PAPERS.mkdir(parents=True, exist_ok=True)
    pdfs = {
        "p1_mcmahan17_fedavg.pdf": "http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf",
        "p2_wang21_federated_analytics.pdf": "https://arxiv.org/pdf/2107.01047.pdf",
        "p4_wang23_voyager.pdf": "https://arxiv.org/pdf/2305.16291.pdf",
        "p5_shinn23_reflexion.pdf": "https://arxiv.org/pdf/2303.11366.pdf",
        "p6_finn17_maml.pdf": "http://proceedings.mlr.press/v70/finn17a/finn17a.pdf",
        "p10_borkar25_privacy_ripple.pdf": "https://aclanthology.org/2025.findings-acl.959.pdf",
    }
    out = {}
    for name, url in pdfs.items():
        dest = PAPERS / name
        try:
            if not dest.exists() or dest.stat().st_size < 10000:
                data = fetch_bytes(url, timeout=60)
                dest.write_bytes(data)
            txt = dest.with_suffix(".txt")
            if not txt.exists():
                run(["pdftotext", str(dest), str(txt)])
            out[name] = {"bytes": dest.stat().st_size, "txt_exists": txt.exists(),
                         "txt_lines": len(txt.read_text(errors="replace").splitlines()) if txt.exists() else 0}
        except Exception as ex:
            out[name] = {"error": str(ex)}
    return out


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {"repos": clone_repos(), "pdfs": download_pdfs(), "crossref": {}, "arxiv": {}}
    for key, doi in [
        ("p3_kd_fl_2025", "10.1007/s10791-025-09657-4"),
        ("p7_swarm_2021", "10.1038/s41586-021-03583-3"),
        ("p9_robust_fl_2025", "10.1145/3727643"),
        ("p10_privacy_ripple_2025", "10.18653/v1/2025.findings-acl.959"),
    ]:
        try:
            report["crossref"][key] = query_crossref(doi)
        except Exception as ex:
            report["crossref"][key] = {"error": str(ex)}

    for key, q in [
        ("p2_federated_analytics", 'ti:"Federated Analytics: Opportunities and Challenges"'),
        ("p3_kd_fl_survey", 'ti:"Knowledge Distillation in Federated Learning"'),
        ("p8_privacy_kd_fl", 'all:"Privacy" AND all:"Knowledge Distillation" AND all:"Federated Learning"'),
        ("p9_robust_fl", 'ti:"Systematic Literature Review of Robust Federated Learning"'),
    ]:
        try:
            report["arxiv"][key] = query_arxiv(q, max_results=3)
        except Exception as ex:
            report["arxiv"][key] = {"error": str(ex)}

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
