#!/usr/bin/env python3
"""Fetch and extract full primary source text for the 13 Intrinsic Motivation, Curiosity & Developmental Learning papers.

Uses pyoudeyer.com/publications/, INRIA HAL open-access archives, Frontiers in Neurorobotics/Neuroscience,
PubMed Central (PMC6032944), and IDSIA (people.idsia.ch/~juergen/) to download PDFs/HTML and extract plain text via
`pdftotext` or HTML parsing.
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

OUT_DIR = Path("/opt/clover-k3/curiosity-research/papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def fetch_bytes(url: str, timeout: int = 45) -> tuple[bytes, str]:
    ua = "curl/8.5.0" if "hal." in url else UA
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        return resp.read(), ctype


def html_to_text(raw_html: str) -> str:
    txt = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
    txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
    txt = re.sub(r"</?(?:p|div|h[1-6]|li|tr|br|section|article)\b[^>]*>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    lines = [" ".join(ln.split()) for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_pdf_text(pdf_path: Path, txt_path: Path) -> int:
    subprocess.run(["pdftotext", str(pdf_path), str(txt_path)], check=False)
    if txt_path.is_file():
        return txt_path.stat().st_size
    return 0


def main() -> int:
    # 1. Fetch pyoudeyer.com/publications/ to discover exact direct PDF links hosted on pyoudeyer.com / HAL INRIA
    pub_html = ""
    try:
        raw_pub, _ = fetch_bytes("https://www.pyoudeyer.com/publications/")
        pub_html = raw_pub.decode("utf-8", errors="replace")
        (OUT_DIR / "pyoudeyer_publications.html").write_text(pub_html, encoding="utf-8")
    except Exception as exc:
        print("Warning fetching pyoudeyer publications:", exc)

    # Extract all hrefs from pyoudeyer_publications.html
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', pub_html, flags=re.I)
    pdf_links = [h for h in hrefs if ".pdf" in h.lower() or "hal." in h.lower() or "inria" in h.lower() or "frontiers" in h.lower()]
    (OUT_DIR / "pyoudeyer_links.json").write_text(json.dumps(pdf_links, indent=2), encoding="utf-8")
    print(f"Found {len(pdf_links)} publication links on pyoudeyer.com/publications/")

    # Direct candidate URLs for each of the 13 papers (ordered by preference: direct PDF -> HAL PDF -> HTML fulltext -> PubMed/EuropePMC)
    papers = [
        {
            "id": "01-oudeyer-kaplan-hafner-2007-iac",
            "title": "Intrinsic Motivation Systems for Autonomous Mental Development",
            "authors": "Pierre-Yves Oudeyer, Frdric Kaplan, Verena V. Hafner",
            "year": 2007,
            "urls": [
                "https://www.pyoudeyer.com/ims.pdf",
                "https://inria.hal.science/hal-00793610/document",
            ],
        },
        {
            "id": "02-twomey-westermann-2018-infant-curiosity",
            "title": "Curiosity-Based Learning in Infants: A Neurocomputational Approach",
            "authors": "Katherine E. Twomey, Gert Westermann",
            "year": 2018,
            "urls": [
                "https://eprints.lancs.ac.uk/id/eprint/88135/1/Twomey_Westermann_accepted.pdf",
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6032944/fullTextXML",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC6032944/",
            ],
        },
        {
            "id": "03-oudeyer-smith-2016-curiosity-development",
            "title": "How Evolution May Work Through Curiosity-Driven Developmental Process: Self-Organization of Vocalization and Attention",
            "authors": "Pierre-Yves Oudeyer, Linda B. Smith",
            "year": 2016,
            "urls": [
                "https://www.pyoudeyer.com/OudeyerSmithTopicsCogSci14.pdf",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=26969919&retmode=xml",
            ],
        },
        {
            "id": "04-oudeyer-kaplan-2007-typology",
            "title": "What Is Intrinsic Motivation? A Typology of Computational Approaches",
            "authors": "Pierre-Yves Oudeyer, Frdric Kaplan",
            "year": 2007,
            "urls": [
                "https://www.pyoudeyer.com/oudeyer-kaplan-neurorobotics.pdf",
                "https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/neuro.12.006.2007/pdf",
                "https://www.pyoudeyer.com/epirob08OudeyerKaplan.pdf",
            ],
        },
        {
            "id": "05-kaplan-oudeyer-2007-neural-circuits",
            "title": "In Search of the Neural Circuits of Intrinsic Motivation",
            "authors": "Frdric Kaplan, Pierre-Yves Oudeyer",
            "year": 2007,
            "urls": [
                "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.01.1.1.017.2007/pdf",
                "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.01.1.1.017.2007/full",
                "https://www.pyoudeyer.com/kaplan-oudeyer-sice09.pdf",
            ],
        },
        {
            "id": "06-gottlieb-2013-info-seeking",
            "title": "Information-Seeking, Curiosity, and Attention: Computational and Neural Mechanisms",
            "authors": "Jacqueline Gottlieb, Pierre-Yves Oudeyer, Manuel Lopes, Adrien Baranes",
            "year": 2013,
            "urls": [
                "https://www.pyoudeyer.com/TICSCuriosity2013.pdf",
                "https://www.pyoudeyer.com/GottliebOudeyerLopes_IntrinsicMotivationChapter16.pdf",
            ],
        },
        {
            "id": "07-gottlieb-oudeyer-2018-active-sampling",
            "title": "Towards a Neuroscience of Active Sampling and Curiosity",
            "authors": "Jacqueline Gottlieb, Pierre-Yves Oudeyer",
            "year": 2018,
            "urls": [
                "https://www.pyoudeyer.com/2018_GottliebOudeyer_ActiveSampling_NatNeuro.pdf",
                "http://pyoudeyer.com/2018_GottliebOudeyer_ActiveSampling_NatNeuro.pdf",
            ],
        },
        {
            "id": "08-oudeyer-gottlieb-lopes-2016-educational",
            "title": "Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies",
            "authors": "Pierre-Yves Oudeyer, Jacqueline Gottlieb, Manuel Lopes",
            "year": 2016,
            "urls": [
                "https://www.pyoudeyer.com/oudeyerGottliebLopesPBR16.pdf",
                "https://www.pyoudeyer.com/GottliebOudeyerLopes_IntrinsicMotivationChapter16.pdf",
            ],
        },
        {
            "id": "09-oudeyer-2017-baby-robots",
            "title": "What Can We Learn About Development from Baby Robots?",
            "authors": "Pierre-Yves Oudeyer",
            "year": 2017,
            "urls": [
                "https://www.pyoudeyer.com/oudeyerWiley16.pdf",
                "https://www.pyoudeyer.com/oudeyerBBS17.pdf",
            ],
        },
        {
            "id": "10-schmidhuber-1991-curious-control",
            "title": "Curious Model-Building Control Systems",
            "authors": "Jrgen Schmidhuber",
            "year": 1991,
            "urls": [
                "https://sferics.idsia.ch/pub/juergen/curioussingapore.pdf",
                "https://people.idsia.ch/~juergen/curioussingapore/curioussingapore.html",
                "https://people.idsia.ch/~juergen/interest.html",
            ],
        },
        {
            "id": "11-ivaldi-2014-object-learning",
            "title": "Object Learning Through Active Exploration",
            "authors": "Serena Ivaldi, Sao Mai Nguyen, Natalia Lyubova, Alain Droniou, Vincent Padois, David Filliat, Emmanuel Palagi, Pierre-Yves Oudeyer",
            "year": 2014,
            "urls": [
                "https://hal.science/hal-00919694/file/Ivaldi_IEEE_TAMD_2013.pdf",
            ],
        },
        {
            "id": "12-nguyen-oudeyer-2014-sgim",
            "title": "Socially Guided Intrinsic Motivation for Robot Learning of Motor Skills",
            "authors": "Sao Mai Nguyen, Pierre-Yves Oudeyer",
            "year": 2014,
            "urls": [
                "http://nguyensmai.free.fr/publication/Nguyen_2014Auro.pdf",
            ],
        },
        {
            "id": "13-santucci-2020-open-ended-learning",
            "title": "Intrinsically Motivated Open-Ended Learning in Autonomous Robots",
            "authors": "Vieri Giuliano Santucci, Pierre-Yves Oudeyer, Andrew Barto, Gianluca Baldassarre",
            "year": 2020,
            "urls": [
                "https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00115/pdf",
                "https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00115/full",
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6978709/fullTextXML",
            ],
        },
    ]

    summary = []
    for p in papers:
        pid = p["id"]
        pdir = OUT_DIR / pid
        pdir.mkdir(parents=True, exist_ok=True)
        fetched_ok = False
        for u in p["urls"]:
            try:
                data, ctype = fetch_bytes(u)
                if len(data) < 800:
                    continue
                sha = hashlib.sha256(data).hexdigest()
                is_pdf = data[:4] == b"%PDF" or "pdf" in ctype.lower()
                if is_pdf:
                    pdf_fp = pdir / f"{pid}.pdf"
                    txt_fp = pdir / f"{pid}.txt"
                    pdf_fp.write_bytes(data)
                    txt_sz = extract_pdf_text(pdf_fp, txt_fp)
                    if txt_sz > 500:
                        fetched_ok = True
                        summary.append({
                            "id": pid, "url": u, "kind": "pdf", "bytes": len(data), "sha256": sha, "text_bytes": txt_sz
                        })
                        print(f"[OK PDF] {pid}: {u} -> {len(data)} bytes (text {txt_sz} bytes)")
                        break
                else:
                    raw_s = data.decode("utf-8", errors="replace")
                    clean_s = html_to_text(raw_s)
                    if len(clean_s) > 4000:
                        raw_fp = pdir / f"{pid}.raw.html"
                        txt_fp = pdir / f"{pid}.txt"
                        raw_fp.write_text(raw_s, encoding="utf-8")
                        txt_fp.write_text(clean_s, encoding="utf-8")
                        fetched_ok = True
                        summary.append({
                            "id": pid, "url": u, "kind": "html_or_xml", "bytes": len(data), "sha256": sha, "text_bytes": len(clean_s)
                        })
                        print(f"[OK TEXT] {pid}: {u} -> {len(data)} bytes (text {len(clean_s)} chars)")
                        break
            except Exception as exc:
                print(f"[TRY FAIL] {pid}: {u} -> {exc}")
        if not fetched_ok:
            summary.append({"id": pid, "status": "missing"})
            print(f"[MISSING] {pid}")

    (OUT_DIR / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
