#!/usr/bin/env python3
"""Fetch and extract primary source texts and bibliographic/abstract records for the 17 papers on
Subjective Experience, Phenomenal Consciousness, and Artificial Consciousness.

Stores extracted texts and provenance manifest in:
  `research/consciousness-and-subjective-experience/papers_text/`
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEXT_DIR = HERE / "papers_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("/opt/clover-k3/consciousness-research/raw") if os.name != "nt" else (HERE / "_raw_downloads")
RAW_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def fetch_url(url: str, timeout: int = 35) -> tuple[bytes, str, int]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/pdf,application/json,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        status = getattr(resp, "status", 200)
        return resp.read(), ctype, status


def html_to_text(raw_html: str) -> str:
    txt = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
    txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
    txt = re.sub(r"</?(?:p|div|h[1-6]|li|tr|br|section|article|blockquote)\b[^>]*>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    lines = [" ".join(ln.split()) for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_pdf_to_text(pdf_bytes: bytes, stem: str) -> str:
    pdf_fp = RAW_DIR / f"{stem}.pdf"
    txt_fp = RAW_DIR / f"{stem}.txt"
    pdf_fp.write_bytes(pdf_bytes)
    subprocess.run(["pdftotext", str(pdf_fp), str(txt_fp)], check=False)
    if txt_fp.is_file():
        return txt_fp.read_text(encoding="utf-8", errors="replace")
    return ""


def fetch_crossref_and_semantic_metadata(doi: str, title: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"doi": doi, "title": title}
    if doi:
        try:
            u = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
            raw, _, _ = fetch_url(u, timeout=20)
            msg = json.loads(raw.decode("utf-8", errors="replace")).get("message", {})
            meta["crossref_title"] = (msg.get("title") or [None])[0]
            meta["container_title"] = (msg.get("container-title") or [None])[0]
            meta["published_year"] = (
                ((msg.get("published-print") or msg.get("published-online") or {}).get("date-parts") or [[None]])[0][0]
            )
            if msg.get("abstract"):
                meta["crossref_abstract"] = html_to_text(str(msg["abstract"]))
        except Exception as exc:
            meta["crossref_error"] = str(exc)

    # Also query OpenAlex for open-access PDF url and inverted-index abstract reconstruction
    try:
        q_url = (
            f"https://api.openalex.org/works/https://doi.org/{doi}"
            if doi
            else f"https://api.openalex.org/works?search={urllib.parse.quote(title)}&per-page=1"
        )
        raw_oa, _, _ = fetch_url(q_url, timeout=20)
        oa_data = json.loads(raw_oa.decode("utf-8", errors="replace"))
        if "results" in oa_data and oa_data["results"]:
            oa_data = oa_data["results"][0]
        meta["openalex_id"] = oa_data.get("id")
        meta["cited_by_count"] = oa_data.get("cited_by_count")
        oa_info = oa_data.get("open_access") or {}
        meta["oa_url"] = oa_info.get("oa_url")
        inv = oa_data.get("abstract_inverted_index")
        if isinstance(inv, dict) and inv:
            pos_map: dict[int, str] = {}
            for word, positions in inv.items():
                for p in positions:
                    pos_map[int(p)] = word
            meta["openalex_abstract"] = " ".join(pos_map[k] for k in sorted(pos_map))
    except Exception as exc:
        meta["openalex_error"] = str(exc)

    # Also query PubMed E-utilities by title/DOI to retrieve PubMed abstract and PMC ID if available
    try:
        term = f'"{title}"[Title]'
        es_u = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term={urllib.parse.quote(term)}"
        raw_es, _, _ = fetch_url(es_u, timeout=15)
        id_list = ((json.loads(raw_es.decode("utf-8", errors="replace")).get("esearchresult") or {}).get("idlist") or [])
        if id_list:
            pmid = id_list[0]
            ef_u = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            raw_ef, _, _ = fetch_url(ef_u, timeout=15)
            xml_txt = raw_ef.decode("utf-8", errors="replace")
            title_m = re.search(r"<ArticleTitle[^>]*>([\s\S]*?)</ArticleTitle>", xml_txt)
            pm_title = html_to_text(title_m.group(1)).lower() if title_m else ""
            target_words = {w for w in re.findall(r"[a-z0-9]{4,}", title.lower())}
            pm_words = {w for w in re.findall(r"[a-z0-9]{4,}", pm_title)}
            if target_words and len(target_words & pm_words) / len(target_words) >= 0.6:
                meta["pmid"] = pmid
                abs_matches = re.findall(r"<AbstractText[^>]*>([\s\S]*?)</AbstractText>", xml_txt)
                if abs_matches:
                    meta["pubmed_abstract"] = " ".join(html_to_text(a) for a in abs_matches)
                pmc_m = re.search(r'<ArticleId IdType="pmc">(PMC\d+)</ArticleId>', xml_txt)
                if pmc_m:
                    meta["pmcid"] = pmc_m.group(1)
    except Exception as exc:
        meta["pubmed_error"] = str(exc)

    return meta


PAPERS: list[dict[str, Any]] = [
    {
        "num": 1,
        "id": "01-nagel-1974-bat",
        "title": "What Is It Like to Be a Bat?",
        "authors": "Thomas Nagel",
        "year": 1974,
        "doi": "10.2307/2183914",
        "priority": 1,
        "urls": [
            "https://www.sas.upenn.edu/~cavitch/pdf-library/Nagel_Bat.pdf",
            "https://warwick.ac.uk/fac/cross_fac/iatl/study/ugmodules/humananimalstudies/lectures/32/nagel_bat.pdf",
        ],
    },
    {
        "num": 2,
        "id": "02-chalmers-1995-facing-up",
        "title": "Facing Up to the Problem of Consciousness",
        "authors": "David J. Chalmers",
        "year": 1995,
        "doi": "",
        "priority": 1,
        "urls": [
            "https://consc.net/papers/facing.html",
            "https://consc.net/papers/facing.pdf",
        ],
    },
    {
        "num": 3,
        "id": "03-levine-1983-explanatory-gap",
        "title": "Materialism and Qualia: The Explanatory Gap",
        "authors": "Joseph Levine",
        "year": 1983,
        "doi": "10.1111/j.1468-0114.1983.tb00207.x",
        "priority": 1,
        "urls": [
            "https://philpapers.org/archive/LEVMAQ.pdf",
            "https://courses.washington.edu/ab462/Levine1983.pdf",
            "https://www.umass.edu/philosophy/sites/default/files/Levine_Explanatory_Gap.pdf",
        ],
    },
    {
        "num": 4,
        "id": "04-jackson-1982-epiphenomenal-qualia",
        "title": "Epiphenomenal Qualia",
        "authors": "Frank Jackson",
        "year": 1982,
        "doi": "10.2307/2219321",
        "priority": 1,
        "urls": [
            "https://fitelson.org/proseminar/jackson.pdf",
            "https://philpapers.org/archive/JACEQ.pdf",
            "https://www.sfu.ca/~jennings/bio/Jackson-Epiphenomenal%20Qualia.pdf",
        ],
    },
    {
        "num": 5,
        "id": "05-block-1995-confusion-consciousness",
        "title": "On a Confusion about a Function of Consciousness",
        "authors": "Ned Block",
        "year": 1995,
        "doi": "10.1017/S0140525X00038188",
        "priority": 1,
        "urls": [
            "https://www.nyu.edu/gsas/dept/philo/faculty/block/papers/1995.bms.pdf",
            "https://www.nyu.edu/gsas/dept/philo/faculty/block/papers/1995.Function.html",
            "https://www.nyu.edu/gsas/dept/philo/faculty/block/papers/bbs1995.html",
            "https://philpapers.org/archive/BLOOAC.pdf",
        ],
    },
    {
        "num": 6,
        "id": "06-crick-koch-1990-neurobiological-theory",
        "title": "Towards a Neurobiological Theory of Consciousness",
        "authors": "Francis Crick, Christof Koch",
        "year": 1990,
        "doi": "",
        "priority": 2,
        "urls": [
            "https://profiles.nlm.nih.gov/ps/access/SCBCFD.pdf",
            "https://authors.library.caltech.edu/records/c969p-ebr26/files/Crick_1990p263.pdf",
        ],
    },
    {
        "num": 7,
        "id": "07-tononi-2004-iit",
        "title": "An Information Integration Theory of Consciousness",
        "authors": "Giulio Tononi",
        "year": 2004,
        "doi": "10.1186/1471-2202-5-42",
        "priority": 2,
        "urls": [
            "https://bmcneurosci.biomedcentral.com/counter/pdf/10.1186/1471-2202-5-42.pdf",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC543470/fullTextXML",
            "https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-5-42",
        ],
    },
    {
        "num": 8,
        "id": "08-dehaene-changeux-2011-gnw",
        "title": "Experimental and Theoretical Approaches to Conscious Processing",
        "authors": "Stanislas Dehaene, Jean-Pierre Changeux",
        "year": 2011,
        "doi": "10.1016/j.neuron.2011.03.018",
        "priority": 2,
        "urls": [
            "https://www.unicog.org/publications/DehaeneChangeux_ConsciousProcessing_Neuron2011.pdf",
            "https://hal.science/hal-00836718/document",
            "https://www.cell.com/neuron/fulltext/S0896-6273(11)00258-3",
        ],
    },
    {
        "num": 9,
        "id": "09-tsuchiya-2015-no-report-paradigms",
        "title": "No-Report Paradigms: Extracting the True Neural Correlates of Consciousness",
        "authors": "Naotsugu Tsuchiya, Max Wilke, Stefan Frässle, Victor A.F. Lamme",
        "year": 2015,
        "doi": "10.1016/j.tics.2015.10.002",
        "priority": 3,
        "urls": [
            "https://www.sciencedirect.com/science/article/pii/S1364661315002296",
        ],
    },
    {
        "num": 10,
        "id": "10-koch-2016-ncc-progress-problems",
        "title": "Neural Correlates of Consciousness: Progress and Problems",
        "authors": "Christof Koch, Marcello Massimini, Melanie Boly, Giulio Tononi",
        "year": 2016,
        "doi": "10.1038/nrn.2016.22",
        "priority": 3,
        "urls": [
            "https://www.nature.com/articles/nrn.2016.22",
        ],
    },
    {
        "num": 11,
        "id": "11-seth-bayne-2022-theories-of-consciousness",
        "title": "Theories of Consciousness",
        "authors": "Anil K. Seth, Tim Bayne",
        "year": 2022,
        "doi": "10.1038/s41583-022-00587-4",
        "priority": 2,
        "urls": [
            "https://www.nature.com/articles/s41583-022-00587-4",
        ],
    },
    {
        "num": 12,
        "id": "12-kronemer-2025-covert-measures",
        "title": "Sleuthing Subjectivity: A Review of Covert Measures of Consciousness",
        "authors": "Sharif I. Kronemer et al.",
        "year": 2025,
        "doi": "10.1038/s41583-025-00934-1",
        "priority": 3,
        "urls": [
            "https://www.nature.com/articles/s41583-025-00934-1",
        ],
    },
    {
        "num": 13,
        "id": "13-butlin-2023-consciousness-in-ai",
        "title": "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness",
        "authors": "Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, Rufin VanRullen",
        "year": 2023,
        "doi": "10.48550/arXiv.2308.08708",
        "priority": 4,
        "urls": [
            "https://arxiv.org/pdf/2308.08708.pdf",
            "https://arxiv.org/abs/2308.08708",
        ],
    },
    {
        "num": 14,
        "id": "14-butlin-2025-identifying-indicators-ai",
        "title": "Identifying Indicators of Consciousness in AI Systems",
        "authors": "Patrick Butlin, Robert Long, Yoshua Bengio, Jonathan Birch, Chris Frith, Ryota Kanai, Colin Klein, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Rufin VanRullen",
        "year": 2025,
        "doi": "10.1016/j.tics.2025.10.011",
        "priority": 4,
        "urls": [
            "https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4",
            "https://doi.org/10.1016/j.tics.2025.10.011",
        ],
    },
    {
        "num": 15,
        "id": "15-butlin-2026-indicators-mimicry-internal-variants",
        "title": "Consciousness Indicators, Mimicry, and Internal Variants",
        "authors": "Patrick Butlin et al.",
        "year": 2026,
        "doi": "10.1016/j.tics.2026.04.006",
        "priority": 4,
        "urls": [
            "https://doi.org/10.1016/j.tics.2026.04.006",
        ],
    },
    {
        "num": 16,
        "id": "16-carvalho-damasio-2021-interoception-feelings",
        "title": "Interoception and the Origin of Feelings: A New Synthesis",
        "authors": "Gil B. Carvalho, Antonio Damasio",
        "year": 2021,
        "doi": "10.1002/bies.202000261",
        "priority": 5,
        "urls": [
            "https://onlinelibrary.wiley.com/doi/10.1002/bies.202000261",
        ],
    },
    {
        "num": 17,
        "id": "17-varela-thompson-rosch-1991-embodied-mind",
        "title": "The Embodied Mind: Cognitive Science and Human Experience",
        "authors": "Francisco J. Varela, Evan Thompson, Eleanor Rosch",
        "year": 1991,
        "doi": "10.7551/mitpress/9739.001.0001",
        "priority": 5,
        "urls": [
            "https://mitpress.mit.edu/9780262720212/the-embodied-mind/",
        ],
    },
]


def main() -> int:
    manifest: list[dict[str, Any]] = []
    for p in PAPERS:
        pid = p["id"]
        print(f"=== Fetching Paper {p['num']:02d}: {p['title']} ===")
        meta = fetch_crossref_and_semantic_metadata(p.get("doi", ""), p["title"])
        cand_urls = list(p.get("urls") or [])
        if meta.get("oa_url") and meta["oa_url"] not in cand_urls:
            cand_urls.insert(0, str(meta["oa_url"]))

        best_text = ""
        best_source_url = ""
        best_ctype = ""
        raw_sha256 = ""
        raw_bytes_len = 0

        for u in cand_urls:
            try:
                raw_b, ctype, status = fetch_url(u, timeout=30)
                if status >= 400 or len(raw_b) < 400:
                    continue
                is_pdf = ("%PDF" in raw_b[:16].decode("latin1", errors="ignore")) or ("pdf" in ctype.lower())
                if is_pdf:
                    txt = extract_pdf_to_text(raw_b, pid)
                else:
                    txt = html_to_text(raw_b.decode("utf-8", errors="replace"))
                if len(txt) > len(best_text):
                    best_text = txt
                    best_source_url = u
                    best_ctype = "application/pdf" if is_pdf else ctype
                    raw_sha256 = hashlib.sha256(raw_b).hexdigest()
                    raw_bytes_len = len(raw_b)
                if len(best_text) >= 15000:
                    break
            except Exception as exc:
                print(f"  [warn] {u}: {exc}")

        # Combine extracted fulltext or abstract/metadata record
        header_lines = [
            f"PAPER_ID: {pid}",
            f"TITLE: {p['title']}",
            f"AUTHORS: {p['authors']}",
            f"YEAR: {p['year']}",
            f"DOI: {p.get('doi', '')}",
            f"SOURCE_URL_REACHED: {best_source_url or 'metadata_only'}",
            f"RAW_BYTES: {raw_bytes_len}",
            f"RAW_SHA256: {raw_sha256}",
            f"EXTRACTED_TEXT_CHARS: {len(best_text)}",
            f"OPENALEX_ID: {meta.get('openalex_id', '')}",
            f"CITED_BY_COUNT: {meta.get('cited_by_count', '')}",
            "---",
        ]
        if meta.get("crossref_abstract"):
            header_lines.append("CROSSREF_ABSTRACT:\n" + str(meta["crossref_abstract"]))
            header_lines.append("---")
        if meta.get("openalex_abstract"):
            header_lines.append("OPENALEX_ABSTRACT:\n" + str(meta["openalex_abstract"]))
            header_lines.append("---")
        if meta.get("pubmed_abstract"):
            header_lines.append("PUBMED_ABSTRACT:\n" + str(meta["pubmed_abstract"]))
            header_lines.append("---")
        if best_text:
            header_lines.append("EXTRACTED_SOURCE_TEXT:\n" + best_text)

        full_out = "\n".join(header_lines) + "\n"
        out_fp = TEXT_DIR / f"{pid}.txt"
        out_fp.write_text(full_out, encoding="utf-8")

        access_tier = (
            "full_text_extracted"
            if len(best_text) >= 12000
            else ("partial_or_landing_page" if len(best_text) >= 1500 else "metadata_and_abstract_only")
        )
        entry_info = {
            "num": p["num"],
            "id": pid,
            "title": p["title"],
            "authors": p["authors"],
            "year": p["year"],
            "doi": p.get("doi", ""),
            "priority": p["priority"],
            "access_tier": access_tier,
            "source_url_reached": best_source_url,
            "raw_bytes": raw_bytes_len,
            "raw_sha256": raw_sha256,
            "extracted_text_chars": len(best_text),
            "has_abstract": bool(
                meta.get("crossref_abstract") or meta.get("openalex_abstract") or meta.get("pubmed_abstract")
            ),
            "pmid": meta.get("pmid"),
            "pmcid": meta.get("pmcid"),
            "cited_by_count": meta.get("cited_by_count"),
        }
        manifest.append(entry_info)
        print(
            f"  -> tier={access_tier}, chars={len(best_text)}, abstract={entry_info['has_abstract']}, url={best_source_url}"
        )

    manifest_fp = TEXT_DIR / "manifest.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest with {len(manifest)} papers to {manifest_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
