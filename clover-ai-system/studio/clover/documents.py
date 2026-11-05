"""Documents the user attaches to a message: spreadsheets (.xlsx .xlsm .xls), tables (.csv .tsv), PDF, Word (.docx),
PowerPoint (.pptx), JSON and plain text. The browser's file type is not trusted: the first bytes decide.

Each document is saved under <media>/documents/<content hash>/<its own name> (the user's name kept, so Python and the
answer can refer to it), and read here into a block the model sees: for tables, every sheet's size, columns, first
rows and exact totals of the numeric columns over all rows; for text documents, the text of the first pages. The block
names the saved path, so Python (when the user allows it) can work on the whole file for figures the preview does not
give. Exact numbers come from here or from Python, never from the model reading digits.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

DOC_LIMIT = 25 << 20
DOC_ITEMS = 3
PREVIEW_CHARS = 6000
PREVIEW_ROWS = 12
MAX_SHEETS = 6
SCAN_ROWS = 200_000
KINDS = {".xlsx": "spreadsheet", ".xlsm": "spreadsheet", ".xls": "spreadsheet", ".csv": "table", ".tsv": "table",
         ".pdf": "PDF", ".docx": "Word document", ".pptx": "PowerPoint", ".json": "JSON",
         ".txt": "text", ".md": "text"}
DATA_KINDS = ("spreadsheet", "table", "JSON")
ZIP_INNER = {".xlsx": "xl/workbook.xml", ".xlsm": "xl/workbook.xml", ".docx": "word/document.xml",
             ".pptx": "ppt/presentation.xml"}
BLOCK_HEAD = re.compile(r"^Document: (.+?) \(([A-Za-z ]+), [\d.,]+ [KM]?B\) saved at (.+)$", re.M)


def safe_name(name):
    """The user's file name without folders or characters Windows or a shell would trip over."""
    n = re.split(r"[\\/]", str(name or ""))[-1].strip()
    n = re.sub(r"[^\w .()\-+&,']", "_", n).strip(" .")[:120]
    return n or "document"


def _kind(name, data):
    ext = Path(name).suffix.lower()
    if ext not in KINDS:
        raise ValueError(f"{name}: this file type is not read here (spreadsheets, CSV, PDF, Word, PowerPoint, JSON, text)")
    if ext in ZIP_INNER:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if ZIP_INNER[ext] not in z.namelist():
                    raise ValueError(f"{name} is not a valid {KINDS[ext]} file")
        except zipfile.BadZipFile:
            raise ValueError(f"{name} is not a valid {KINDS[ext]} file") from None
    elif ext == ".pdf" and not data.startswith(b"%PDF"):
        raise ValueError(f"{name} is not a PDF file")
    elif ext == ".xls" and not data.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError(f"{name} is not an Excel 97-2003 file")
    elif ext in (".csv", ".tsv", ".json", ".txt", ".md") and b"\x00" in data[:4096]:
        raise ValueError(f"{name} looks like a binary file, not {KINDS[ext]}")
    return KINDS[ext]


def save(items, folder):
    """[{"name", "data" (base64)}] -> [{"name", "path", "kind", "bytes"}], saved under `folder`."""
    out = []
    if not isinstance(items, list):
        return out
    if len(items) > DOC_ITEMS:
        raise ValueError(f"at most {DOC_ITEMS} documents per message")
    for it in items:
        name = safe_name((it or {}).get("name"))
        raw = str(it.get("data") or "")
        raw = raw.split(",", 1)[1] if raw.startswith("data:") else raw
        try:
            data = base64.b64decode(raw, validate=True)
        except ValueError:
            raise ValueError(f"{name} is not valid base64") from None
        if not data or len(data) > DOC_LIMIT:
            raise ValueError(f"{name}: documents must be 1 byte to {DOC_LIMIT >> 20} MB")
        kind = _kind(name, data)
        d = Path(folder) / hashlib.sha256(data).hexdigest()[:24]
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        if not p.is_file():
            p.write_bytes(data)
        out.append({"name": name, "path": str(p), "kind": kind, "bytes": len(data)})
    return out


# ---------------------------------------------------------------- tables
def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    s = str(v).strip().replace(",", "")
    s = re.sub(r"^[$\u20ac\u00a3\u20b9]|%$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:,.2f}".rstrip("0").rstrip(".") if abs(x) < 1e15 else str(x)
    return str(x)


def _cell(v):
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        return v.isoformat()[:19].replace("T", " ").removesuffix(" 00:00:00")
    s = str(v).replace("\n", " ").replace("|", "/").strip()
    return s[:40] + "..." if len(s) > 43 else s


def table_summary(title, rows):
    """rows: an iterable of tuples, the first is the header. -> text: size, columns, first rows, numeric totals."""
    it = iter(rows)
    header = None
    for r in it:
        if any(c not in (None, "") for c in r):
            header = [_cell(c) or f"column {i + 1}" for i, c in enumerate(r)]
            break
    if header is None:
        return f"{title}: empty"
    ncol = len(header)
    first, n = [], 0
    stats = [{"n": 0, "sum": 0.0, "min": None, "max": None, "text": 0} for _ in range(ncol)]
    for r in it:
        if n >= SCAN_ROWS:
            break
        r = list(r)[:ncol] + [None] * max(0, ncol - len(r))
        if all(c in (None, "") for c in r):
            continue
        n += 1
        if len(first) < PREVIEW_ROWS:
            first.append(r)
        for i, c in enumerate(r):
            x = _num(c)
            if x is None:
                if c not in (None, ""):
                    stats[i]["text"] += 1
                continue
            s = stats[i]
            s["n"] += 1
            s["sum"] += x
            s["min"] = x if s["min"] is None else min(s["min"], x)
            s["max"] = x if s["max"] is None else max(s["max"], x)
    lines = [f"{title}: {n} data rows x {ncol} columns" + (f" (first {SCAN_ROWS:,} rows read)" if n >= SCAN_ROWS else ""),
             "| " + " | ".join(header) + " |", "|" + "---|" * ncol]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in first]
    if n > len(first):
        lines.append(f"(... {n - len(first)} more rows)")
    nums = [(header[i], s) for i, s in enumerate(stats) if s["n"] and s["n"] >= max(1, s["text"])]
    if nums:
        lines.append("Numeric columns over all rows (exact):")
        lines += [f"- {h}: count {s['n']}, total {_fmt(s['sum'])}, min {_fmt(s['min'])}, max {_fmt(s['max'])}, "
                  f"mean {_fmt(s['sum'] / s['n'])}" for h, s in nums]
    return "\n".join(lines)


def _xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        names = wb.sheetnames
        parts = [f"Sheets: {', '.join(names)}"]
        for name in names[:MAX_SHEETS]:
            parts.append(table_summary(f"Sheet '{name}'", wb[name].iter_rows(values_only=True)))
        if len(names) > MAX_SHEETS:
            parts.append(f"({len(names) - MAX_SHEETS} more sheets not shown)")
        return "\n\n".join(parts)
    finally:
        wb.close()


def _xls(path):
    import xlrd
    book = xlrd.open_workbook(path, on_demand=True)
    names = book.sheet_names()
    parts = [f"Sheets: {', '.join(names)}"]
    for name in names[:MAX_SHEETS]:
        sh = book.sheet_by_name(name)
        parts.append(table_summary(f"Sheet '{name}'", (sh.row_values(i) for i in range(sh.nrows))))
    return "\n\n".join(parts)


def _csv(path, kind_ext):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    if kind_ext == ".tsv":
        dialect = csv.excel_tab
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
    return table_summary("Table", csv.reader(io.StringIO(text), dialect))


# ---------------------------------------------------------------- text documents
def _pdf(path):
    from pypdf import PdfReader
    rd = PdfReader(path)
    out, total = [f"Pages: {len(rd.pages)}"], 0
    for i, page in enumerate(rd.pages, 1):
        t = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if t:
            out.append(f"[page {i}]\n{t}")
            total += len(t)
        if total > PREVIEW_CHARS:
            break
    if len(out) == 1:
        out.append("(no text layer: this PDF holds scanned pages; the text cannot be read here)")
    return "\n".join(out)


def _docx(path):
    import docx
    d = docx.Document(path)
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for ti, tbl in enumerate(d.tables, 1):
        rows = [tuple(c.text for c in r.cells) for r in tbl.rows]
        if rows:
            out.append(table_summary(f"Table {ti}", rows))
    return "\n".join(out)


def _pptx(path):
    from pptx import Presentation
    pres = Presentation(path)
    out = []
    for i, slide in enumerate(pres.slides, 1):
        texts = [sh.text_frame.text.strip() for sh in slide.shapes if getattr(sh, "has_text_frame", False)
                 and sh.text_frame.text.strip()]
        if texts:
            out.append(f"[slide {i}] " + " / ".join(texts))
    return f"Slides: {len(pres.slides)}\n" + "\n".join(out)


def _json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data[:50]):
        keys = list(dict.fromkeys(k for x in data[:200] for k in x))
        return table_summary("Records", [tuple(keys)] + [tuple(x.get(k) for k in keys) for x in data if isinstance(x, dict)])
    return json.dumps(data, indent=1, ensure_ascii=False, default=str)


def preview(doc):
    """The text the model sees for one saved document (cut at PREVIEW_CHARS)."""
    path, ext = doc["path"], Path(doc["path"]).suffix.lower()
    try:
        if ext in (".xlsx", ".xlsm"):
            text = _xlsx(path)
        elif ext == ".xls":
            text = _xls(path)
        elif ext in (".csv", ".tsv"):
            text = _csv(path, ext)
        elif ext == ".pdf":
            text = _pdf(path)
        elif ext == ".docx":
            text = _docx(path)
        elif ext == ".pptx":
            text = _pptx(path)
        elif ext == ".json":
            text = _json(path)
        else:
            text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except ImportError as exc:
        return f"(this file type needs the Python package '{exc.name}', which is not installed: run Clover AI Studio setup again)"
    except Exception as exc:  # a damaged or unusual file: say so instead of failing the turn
        return f"(the file could not be read: {type(exc).__name__}: {str(exc)[:200]})"
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + f"\n(... cut here; {len(text):,} characters in all)"


def _size(n):
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def block(doc):
    """The attachment block for the prompt: what the document is, where it is saved, what is in it."""
    head = f"Document: {doc['name']} ({doc['kind']}, {_size(doc['bytes'])}) saved at {doc['path']}"
    text, hint = preview(doc), ""
    if doc["kind"] in DATA_KINDS:
        p = doc["path"]
        if doc["kind"] == "spreadsheet":
            # one table per sheet: sheet_name=None returns a dict of all sheets, and the model then called .loc and
            # .groupby on the dict ("'dict' object has no attribute 'loc'", Studio regression 2026-11-03)
            m = re.search(r"^Sheets: (.+)$", text, re.M)
            sheets = [s.strip() for s in m.group(1).split(",")] if m else []
            first = sheets[0] if sheets else "Sheet1"
            how = f"df = pandas.read_excel(r\"{p}\", sheet_name={first!r})"
            if len(sheets) > 1:
                how += " (the other sheets: " + ", ".join(f"sheet_name={s!r}" for s in sheets[1:6]) + ")"
        elif doc["kind"] == "JSON":
            how = f"df = pandas.read_json(r\"{p}\")"
        else:
            how = f"df = pandas.read_csv(r\"{p}\"{', sep=chr(9)' if p.lower().endswith('.tsv') else ''})"
        hint = ("\nThe totals above are exact. For any other figure (per group, per month, filtered, ranked), compute it "
                f"with run_python on the whole file: {how}, then use the column names shown above; a figure that is not "
                "a column (e.g. revenue) is computed from the columns. Do not add up numbers by reading them.")
        # a worked groupby/chart line here (tried 2026-11-03 for the chart turn) did not fix the chart and made the first
        # CSV question run four 2,048-token rounds of unfinished tool calls: 445 s and 414 s instead of 11-13 s
    return f"{head}\n{text}{hint}"


def in_history(messages):
    """[(name, kind)] of the documents attached earlier in a conversation (from the blocks in its user messages)."""
    out = []
    for m in messages or ():
        if m.get("role") == "user":
            out += [(g.group(1), g.group(2)) for g in BLOCK_HEAD.finditer(str(m.get("content") or ""))]
    return out
