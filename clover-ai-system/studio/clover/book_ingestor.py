"""Real Textbook Chapter Ingestor & Section/Chunk Indexer (`book_ingestor.py`).

Downloads, verifies, and parses actual upstream textbook chapters for Experiment 5.2:
  1. **Allen B. Downey — *Think Python, 3rd Edition*** (`CC BY-NC-SA 3.0`):
     - `chap06.ipynb`: Chapter 6 (*Return Values*, Recursion with return values, Leap of faith, Fibonacci, Checking types)
     - `chap10.ipynb`: Chapter 10 (*Dictionaries*, Memos, Call graphs, `fibonacci_memo`, Debugging)
  2. **OpenStax — *University Physics, Volume 1*** (`CC BY 4.0` / `CC BY-NC-SA 4.0`):
     - `m58306/index.cnxml`: Section 6.4 (*Drag Force and Terminal Speed*)
     - `m58313/index.cnxml`: Section 8.2 (*Conservative and Non-Conservative Forces*)
  3. **OpenStax — *Biology 2e*** (`CC BY 4.0` / `CC BY-NC-SA 4.0`):
     - `m66459/index.cnxml`: Section 6.2 (*Potential, Kinetic, Free, and Activation Energy*)
     - `m66461/index.cnxml`: Section 6.3 (*The Laws of Thermodynamics*)
     - `m66462/index.cnxml`: Section 6.4 (*ATP: Adenosine Triphosphate*)

Preserves:
  - Exact upstream URL, byte length, and SHA-256 digest of every ingested source file.
  - Exact cell indices (`cells[87..117]`) for Jupyter notebook chapters and `<section>` IDs / line ranges for OpenStax CNXML modules.
  - License and non-commercial attribution metadata.
  - A deterministic BM25 chunk retriever (`retrieve_rag_chunks`) over the raw ingested book passages so Ordinary RAG
    can be evaluated head-to-head against Clover's model-driven concept & hypothesis learning.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = HERE / "books_corpus"


@dataclass
class IngestedChapterSection:
    book_id: str
    book_title: str
    author: str
    license_name: str
    commercial_use_allowed: bool
    source_url: str
    local_file: str
    raw_file_bytes: int
    raw_file_sha256: str
    section_id: str
    section_title: str
    location_span: str  # e.g. "chap06.ipynb#cells[87:117]" or "m58306/index.cnxml#lines[1:420]"
    verbatim_text: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verbatim_chars"] = len(self.verbatim_text)
        return d


UPSTREAM_BOOK_SOURCES: list[dict[str, Any]] = [
    {
        "source_key": "think_python_3e_ch06",
        "book_id": "think_python_3e",
        "book_title": "Think Python: How to Think Like a Computer Scientist (3rd Edition)",
        "author": "Allen B. Downey",
        "license_name": "CC BY-NC-SA 3.0",
        "commercial_use_allowed": False,
        "format": "ipynb",
        "url": "https://raw.githubusercontent.com/AllenDowney/ThinkPython/v3/chapters/chap06.ipynb",
        "filename": "think_python_3e_chap06.ipynb",
        "cell_Start": 87,
        "cell_end": 118,
        "section_id": "ch06_recursion_and_types",
        "section_title": "Chapter 6: Return Values — Recursion, Fibonacci, and Checking Types",
    },
    {
        "source_key": "think_python_3e_ch10",
        "book_id": "think_python_3e",
        "book_title": "Think Python: How to Think Like a Computer Scientist (3rd Edition)",
        "author": "Allen B. Downey",
        "license_name": "CC BY-NC-SA 3.0",
        "commercial_use_allowed": False,
        "format": "ipynb",
        "url": "https://raw.githubusercontent.com/AllenDowney/ThinkPython/v3/chapters/chap10.ipynb",
        "filename": "think_python_3e_chap10.ipynb",
        "cell_Start": 96,
        "cell_end": 108,
        "section_id": "ch10_memos_and_debugging",
        "section_title": "Chapter 10: Dictionaries — Call Graphs, Memos, and Self-Checks",
    },
    {
        "source_key": "university_physics_v1_m58306",
        "book_id": "university_physics_v1",
        "book_title": "University Physics, Volume 1 (OpenStax)",
        "author": "William Moebs, Samuel J. Ling, Jeff Sanny",
        "license_name": "CC BY 4.0 / OpenStax Attribution",
        "commercial_use_allowed": False,
        "format": "cnxml",
        "url": "https://raw.githubusercontent.com/openstax/osbooks-university-physics-bundle/main/modules/m58306/index.cnxml",
        "filename": "university_physics_v1_m58306_drag_terminal_speed.cnxml",
        "char_start": 0,
        "char_end": 6600,
        "section_id": "m58306_drag_and_terminal_speed",
        "section_title": "Section 6.4: Drag Force and Terminal Speed",
    },
    {
        "source_key": "university_physics_v1_m58314",
        "book_id": "university_physics_v1",
        "book_title": "University Physics, Volume 1 (OpenStax)",
        "author": "William Moebs, Samuel J. Ling, Jeff Sanny",
        "license_name": "CC BY 4.0 / OpenStax Attribution",
        "commercial_use_allowed": False,
        "format": "cnxml",
        "url": "https://raw.githubusercontent.com/openstax/osbooks-university-physics-bundle/main/modules/m58314/index.cnxml",
        "filename": "university_physics_v1_m58314_conservation_of_energy.cnxml",
        "char_start": 6800,
        "char_end": 11600,
        "section_id": "m58314_conservation_of_energy_and_nonconservative_work",
        "section_title": "Section 8.3: Conservation of Energy and Non-Conservative Work",
    },
    {
        "source_key": "biology_2e_m66459",
        "book_id": "biology_2e",
        "book_title": "Biology 2e (OpenStax)",
        "author": "Mary Ann Clark, Matthew Douglas, Jung Choi",
        "license_name": "CC BY 4.0 / OpenStax Attribution",
        "commercial_use_allowed": False,
        "format": "cnxml",
        "url": "https://raw.githubusercontent.com/openstax/osbooks-biology-bundle/main/modules/m66459/index.cnxml",
        "filename": "biology_2e_m66459_free_energy.cnxml",
        "char_start": 0,
        "char_end": 5200,
        "section_id": "m66459_free_and_activation_energy",
        "section_title": "Section 6.2: Potential, Kinetic, Free, and Activation Energy",
    },
    {
        "source_key": "biology_2e_m66461",
        "book_id": "biology_2e",
        "book_title": "Biology 2e (OpenStax)",
        "author": "Mary Ann Clark, Matthew Douglas, Jung Choi",
        "license_name": "CC BY 4.0 / OpenStax Attribution",
        "commercial_use_allowed": False,
        "format": "cnxml",
        "url": "https://raw.githubusercontent.com/openstax/osbooks-biology-bundle/main/modules/m66461/index.cnxml",
        "filename": "biology_2e_m66461_thermodynamics.cnxml",
        "char_start": 0,
        "char_end": 5200,
        "section_id": "m66461_laws_of_thermodynamics",
        "section_title": "Section 6.3: The Laws of Thermodynamics in Living Systems",
    },
    {
        "source_key": "biology_2e_m66462",
        "book_id": "biology_2e",
        "book_title": "Biology 2e (OpenStax)",
        "author": "Mary Ann Clark, Matthew Douglas, Jung Choi",
        "license_name": "CC BY 4.0 / OpenStax Attribution",
        "commercial_use_allowed": False,
        "format": "cnxml",
        "url": "https://raw.githubusercontent.com/openstax/osbooks-biology-bundle/main/modules/m66462/index.cnxml",
        "filename": "biology_2e_m66462_atp_coupling.cnxml",
        "char_start": 3500,
        "char_end": 8800,
        "section_id": "m66462_atp_energy_coupling",
        "section_title": "Section 6.4: ATP: Adenosine Triphosphate and Energy Coupling",
    },
]


def _extract_ipynb_cells(raw_bytes: bytes, cell_start: int, cell_end: int) -> str:
    nb = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    cells = nb.get("cells") or []
    parts: list[str] = []
    for idx in range(max(0, cell_start), min(len(cells), cell_end)):
        c = cells[idx]
        ct = c.get("cell_type", "")
        src = "".join(c.get("source") or []).strip()
        if not src:
            continue
        if "from diagram import" in src or "adjust(x, y, bbox)" in src or "arrowprops =" in src:
            continue
        if ct == "code":
            clean_lines = [
                (f"# [notebook magic: {ln}]" if ln.lstrip().startswith("%") else ln)
                for ln in src.splitlines()
            ]
            src_clean = "\n".join(clean_lines)
            parts.append(f"[Cell {idx} — Python Code]\n```python\n{src_clean}\n```")
        elif ct == "markdown":
            parts.append(f"[Cell {idx} — Text]\n{src}")
    return "\n\n".join(parts)


def _render_mathml(math_xml: str) -> str:
    s = math_xml
    # Convert msup, mfrac, then msqrt from inside out before stripping remaining tags
    for _ in range(3):
        s = re.sub(
            r"<m:msup>\s*<m:([^>\s]+)[^>]*>([\s\S]*?)</m:\1>\s*<m:([^>\s]+)[^>]*>([\s\S]*?)</m:\3>\s*</m:msup>",
            lambda m: f"{re.sub(r'<[^>]+>', '', m.group(2)).strip()}^{re.sub(r'<[^>]+>', '', m.group(4)).strip()}",
            s,
        )
        s = re.sub(
            r"<m:mfrac>\s*<m:([^>\s]+)[^>]*>([\s\S]*?)</m:\1>\s*<m:([^>\s]+)[^>]*>([\s\S]*?)</m:\3>\s*</m:mfrac>",
            lambda m: f"({re.sub(r'<[^>]+>', ' ', m.group(2)).strip()}) / ({re.sub(r'<[^>]+>', ' ', m.group(4)).strip()})",
            s,
        )
        s = re.sub(
            r"<m:msqrt>\s*([\s\S]*?)\s*</m:msqrt>",
            lambda m: f"sqrt({re.sub(r'<[^>]+>', ' ', m.group(1)).strip()})",
            s,
        )
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_cnxml_text(raw_bytes: bytes, char_start: int = 0, char_end: int = 9500) -> tuple[str, int]:
    xml_str = raw_bytes.decode("utf-8", errors="replace")
    n_lines = len(xml_str.splitlines())
    # Preserve titles, paragraphs, notes, and equations clearly
    cleaned = re.sub(r"<metadata[\s\S]*?</metadata>", " ", xml_str)
    cleaned = re.sub(r"<m:annotation-xml[\s\S]*?</m:annotation-xml>", " ", cleaned)
    cleaned = re.sub(
        r"<m:math[^>]*>([\s\S]*?)</m:math>",
        lambda m: " [Equation: " + _render_mathml(m.group(1)) + "] ",
        cleaned,
    )
    cleaned = re.sub(r"<title>([^<]+)</title>", r"\n### \1\n", cleaned)
    cleaned = re.sub(r"</para>", "\n\n", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    paras = []
    for p in cleaned.split("\n\n"):
        s = re.sub(r"[ \t]+", " ", p).strip()
        if len(s) >= 20:
            paras.append(s)
    full_txt = "\n\n".join(paras)
    return full_txt[char_start:char_end], n_lines


class RealBookIngestor:
    """Fetches and caches real upstream textbook files and parses them into `IngestedChapterSection` records."""

    def __init__(self, corpus_dir: str | Path | None = None) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.sections: dict[str, IngestedChapterSection] = {}

    def ingest_source(self, spec: dict[str, Any], allow_network: bool = True) -> IngestedChapterSection:
        fp = self.corpus_dir / spec["filename"]
        if not fp.is_file() and allow_network:
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "Clover-Learning-Engine/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            fp.write_bytes(data)
        if not fp.is_file():
            raise FileNotFoundError(f"Textbook source file not found at {fp} and network fetch disabled")

        raw_bytes = fp.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()

        if spec["format"] == "ipynb":
            cs, ce = int(spec["cell_Start"]), int(spec["cell_end"])
            verbatim = _extract_ipynb_cells(raw_bytes, cs, ce)
            loc_span = f"{spec['filename']}#cells[{cs}:{ce}]"
        else:
            c_start = int(spec.get("char_start", 0))
            c_end = int(spec.get("char_end", 9500))
            verbatim, n_lines = _extract_cnxml_text(raw_bytes, char_start=c_start, char_end=c_end)
            loc_span = f"{spec['filename']}#lines[1:{n_lines}](chars[{c_start}:{c_end}])"

        sec = IngestedChapterSection(
            book_id=spec["book_id"],
            book_title=spec["book_title"],
            author=spec["author"],
            license_name=spec["license_name"],
            commercial_use_allowed=bool(spec["commercial_use_allowed"]),
            source_url=spec["url"],
            local_file=str(fp),
            raw_file_bytes=len(raw_bytes),
            raw_file_sha256=raw_sha,
            section_id=spec["section_id"],
            section_title=spec["section_title"],
            location_span=loc_span,
            verbatim_text=verbatim,
        )
        self.sections[spec["source_key"]] = sec
        return sec

    def ingest_all(self, keys: list[str] | None = None, allow_network: bool = True) -> dict[str, IngestedChapterSection]:
        for spec in UPSTREAM_BOOK_SOURCES:
            if keys is None or spec["source_key"] in keys:
                self.ingest_source(spec, allow_network=allow_network)
        return dict(self.sections)

    def build_rag_chunks(self, chunk_chars: int = 1100) -> list[dict[str, Any]]:
        """Split all ingested verbatim book sections into overlapping passages for the Ordinary RAG baseline."""
        chunks: list[dict[str, Any]] = []
        for skey, sec in self.sections.items():
            blocks = [b.strip() for b in sec.verbatim_text.split("\n\n") if b.strip()]
            cur: list[str] = []
            cur_len = 0
            idx = 0
            for blk in blocks:
                if cur_len + len(blk) > chunk_chars and cur:
                    txt = "\n\n".join(cur)
                    chunks.append({
                        "chunk_id": f"{skey}#chunk_{idx}",
                        "book_id": sec.book_id,
                        "section_title": sec.section_title,
                        "location_span": sec.location_span,
                        "text": txt,
                    })
                    idx += 1
                    cur = [cur[-1], blk] if len(cur[-1]) < 400 else [blk]
                    cur_len = sum(len(x) for x in cur)
                else:
                    cur.append(blk)
                    cur_len += len(blk)
            if cur:
                chunks.append({
                    "chunk_id": f"{skey}#chunk_{idx}",
                    "book_id": sec.book_id,
                    "section_title": sec.section_title,
                    "location_span": sec.location_span,
                    "text": "\n\n".join(cur),
                })
        return chunks

    def retrieve_rag_chunks(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top-k raw textbook passages using BM25 term scoring (Ordinary RAG baseline)."""
        chunks = self.build_rag_chunks()
        if not chunks:
            return []
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
            "which", "how", "are", "was", "were", "will", "can", "into", "does", "not",
        }
        q_terms = [w for w in re.findall(r"[a-z0-9_]{3,}", (query or "").lower()) if w not in stop]
        if not q_terms:
            return chunks[:top_k]

        doc_tokens = [
            [w for w in re.findall(r"[a-z0-9_]{3,}", c["text"].lower()) if w not in stop]
            for c in chunks
        ]
        avg_dl = max(1.0, sum(len(d) for d in doc_tokens) / len(doc_tokens))
        n_docs = len(chunks)
        df = {t: sum(1 for d in doc_tokens if t in d) for t in set(q_terms)}

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for i, (c, d_toks) in enumerate(zip(chunks, doc_tokens)):
            dl = len(d_toks)
            score = 0.0
            for t in q_terms:
                tf = d_toks.count(t)
                if tf == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * dl / avg_dl))
            scored.append((score, -i, c))
        scored.sort(reverse=True)
        return [c for _, _, c in scored[:top_k]]
