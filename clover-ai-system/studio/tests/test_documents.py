"""Offline checks for documents.py (attached spreadsheets, tables, PDF, Word, PowerPoint, JSON): what is saved, what is
refused, and that the preview gives the exact figures. Needs openpyxl, pypdf, python-docx, python-pptx (the Studio's
runtime has them). Usage: python tests/test_documents.py"""
import base64
import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
sys.path.insert(0, str(HERE))
import documents  # noqa: E402
import natural_regression as NR  # noqa: E402  (its file writers and data)

ok = 0


def check(cond, what):
    global ok
    if not cond:
        print("FAIL", what)
        raise SystemExit(1)
    ok += 1
    print("PASS", what)


def item(path, name=None):
    return {"name": name or Path(path).name, "data": base64.b64encode(Path(path).read_bytes()).decode()}


tmp = Path(tempfile.mkdtemp(prefix="clover-docs-"))
try:
    src, store = tmp / "src", tmp / "media" / "documents"
    src.mkdir()
    NR.SANDBOX = tmp / "sandbox"
    inputs = NR.make_files()

    # ---- saving and refusing
    docs = documents.save([item(inputs / "q1-sales.xlsx"), item(inputs / "orders-october.csv")], store)
    check([d["kind"] for d in docs] == ["spreadsheet", "table"] and all(Path(d["path"]).is_file() for d in docs),
          "a workbook and a CSV are saved, kinds from the file")
    again = documents.save([item(inputs / "q1-sales.xlsx")], store)
    check(again[0]["path"] == docs[0]["path"] and len(list(store.iterdir())) == 2, "the same file is stored once")
    for bad, why in (([{"name": "x.exe", "data": "TVo="}], "not read here"),
                     ([{"name": "fake.xlsx", "data": base64.b64encode(b"hello").decode()}], "not a valid"),
                     ([{"name": "fake.pdf", "data": base64.b64encode(b"hello").decode()}], "not a PDF"),
                     ([{"name": "a.csv", "data": "!!"}], "base64"),
                     ([{"name": "a.csv", "data": ""}], "1 byte"),
                     ([item(inputs / "stats.py", "a.csv")] * 4, "at most")):
        try:
            documents.save(bad, store)
            check(False, f"refused: {why}")
        except ValueError as exc:
            check(why in str(exc), f"refused ({why}): {exc}")
    check(documents.safe_name(r"..\..\Windows\evil name?.xlsx") == "evil name_.xlsx", "folders and odd characters dropped")

    # ---- spreadsheet preview: sheets, size, exact totals
    b = documents.block(docs[0])
    rev_units = sum(s[3] for s in NR.SALES)
    check("Sheets: Sales, Targets" in b and "36 data rows x 5 columns" in b, "workbook: sheet names and size")
    check(f"Units: count 36, total {rev_units:,}" in b, f"workbook: exact total of Units ({rev_units:,})")
    check(f"Q1 revenue target: count 4, total {sum(NR.TARGETS.values()):,.0f}" in b, "second sheet totals")
    check(docs[0]["path"] in b and "pandas.read_excel" in b and "run_python" in b, "the block names the saved path and Python")
    check("sheet_name='Sales')" in b and "sheet_name='Targets'" in b and "sheet_name=None" not in b,
          "one sheet per read_excel (sheet_name=None gave the model a dict)")
    import pandas
    exec_ns = {"pandas": pandas}
    exec(b.split("on the whole file: ", 1)[1].split(" (the other", 1)[0], exec_ns)
    check(int(exec_ns["df"].loc[exec_ns["df"]["Month"] == "March", "Units"].sum()) == NR.MARCH_UNITS,
          f"the hint's own line reads a table Python can filter (March units {NR.MARCH_UNITS})")
    check("groupby" not in b and "savefig" not in b, "no worked groupby/chart line (it made the CSV turn run away)")
    c = documents.block(docs[1])
    total = round(sum(o[3] for o in NR.ORDERS), 2)
    check("180 data rows x 4 columns" in c and f"amount: count 180, total {total:,.2f}".rstrip("0").rstrip(".") in c,
          f"CSV: every row read, exact total ({total:,.2f})")
    check(f"mean {NR.AVG_ORDER:,.2f}".rstrip("0").rstrip(".") in c, f"CSV: exact mean ({NR.AVG_ORDER})")
    check("(... 168 more rows)" in c and c.count("\n| ORD-") == documents.PREVIEW_ROWS, "CSV: first rows only in the preview")

    # ---- a workbook as Excel writes it (shared strings, dates, formulas' cached values)
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Expenses"
    ws.append(["Date", "Item", "Amount"])
    for i, (it, amt) in enumerate((("Rent", 1200), ("Groceries", 310.45), ("Power", 88.1)), 1):
        ws.append([datetime.date(2026, 10, i), it, amt])
    wb.create_sheet("Empty")
    wb.save(src / "expenses.xlsx")
    e = documents.block(documents.save([item(src / "expenses.xlsx")], store)[0])
    check("Amount: count 3, total 1,598.55" in e and "2026-10-01" in e and "Sheet 'Empty': empty" in e,
          "Excel's own file: dates, totals, an empty sheet")

    # ---- PDF, Word, PowerPoint, JSON
    p = documents.block(documents.save([item(inputs / "service-agreement.pdf")], store)[0])
    check("Pages: 1" in p and "15 March 2027" in p and "Priya Raman" in p and "run_python" not in p, "PDF text, no Python hint")
    w = documents.block(documents.save([item(inputs / "team-meeting-notes.docx")], store)[0])
    check("Maya will send the revised budget" in w and "Ravi will fix the login bug" in w, "Word paragraphs")
    import docx
    d = docx.Document()
    d.add_paragraph("Budget review")
    t = d.add_table(rows=3, cols=2)
    for r, (k, v) in enumerate((("Item", "Cost"), ("Venue", "900"), ("Food", "450.5"))):
        t.cell(r, 0).text, t.cell(r, 1).text = k, v
    d.save(src / "budget.docx")
    wt = documents.block(documents.save([item(src / "budget.docx")], store)[0])
    check("Cost: count 2, total 1,350.5" in wt, "a table inside a Word document: exact total")
    from pptx import Presentation
    pr = Presentation()
    s = pr.slides.add_slide(pr.slide_layouts[1])
    s.shapes.title.text = "Launch plan"
    s.placeholders[1].text = "Ship on 4 November"
    pr.save(src / "plan.pptx")
    pp = documents.block(documents.save([item(src / "plan.pptx")], store)[0])
    check("Slides: 1" in pp and "Launch plan" in pp and "Ship on 4 November" in pp, "PowerPoint slide text")
    (src / "people.json").write_text(json.dumps([{"name": "A", "age": 30}, {"name": "B", "age": 41}]), encoding="utf-8")
    j = documents.block(documents.save([item(src / "people.json")], store)[0])
    check("age: count 2, total 71" in j and "pandas.read_json" in j, "JSON records read as a table")

    # ---- the conversation remembers which documents it has
    hist = [{"role": "user", "content": b + "\n\nUsing the attached content ...\nWhich region?"},
            {"role": "assistant", "content": "South."}, {"role": "user", "content": p}]
    check(documents.in_history(hist) == [("q1-sales.xlsx", "spreadsheet"), ("service-agreement.pdf", "PDF")],
          "documents earlier in the chat are found in its messages")

    # ---- a long text is cut
    (src / "long.txt").write_text("word " * 5000, encoding="utf-8")
    lt = documents.block(documents.save([item(src / "long.txt")], store)[0])
    check("cut here; 25,000 characters in all" in lt, "a long document is cut with a note")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok} checks passed")
