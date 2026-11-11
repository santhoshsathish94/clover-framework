"""Exact value counts for a table's text columns (documents.table_summary, 2026-11-11).

Observed on the full system: "Which customer ordered the most times?" on orders-october.csv was answered "Ben, with 24
orders" because the model's own code printed the second row (customer_counts.index[1]); its own table showed Chen 61.
Like the numeric totals, the counts now come from the Studio, over all rows.
"""
import base64
import collections
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
sys.path.insert(0, str(HERE))

import documents  # noqa: E402
import natural_regression as NR  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


inputs = NR.make_files()
store = Path(tempfile.mkdtemp()) / "documents"
csv_path = inputs / "orders-october.csv"
doc = documents.save([{"name": csv_path.name, "data": base64.b64encode(csv_path.read_bytes()).decode()}], store)[0]
blk = documents.block(doc)
rows = csv_path.read_text(encoding="utf-8").splitlines()[1:]
counts = collections.Counter(r.split(",")[1] for r in rows)
top, n_top = counts.most_common(1)[0]
line = next((ln for ln in blk.splitlines() if ln.startswith("- customer:")), "")
check("the customer column has exact counts, most first", line.startswith(f"- customer: {len(counts)} distinct; {top} {n_top},")
      and top == NR.TOP_CUSTOMER)
check("every customer's count is exact", all(f"{k} {v}" in line for k, v in counts.items()))
check("an ID column (one row per value) gets no counts", "- order_id:" not in blk)
check("the numeric totals are still there", "- amount: count 180, total" in blk)
check("the hint says the counts are exact", "totals and value counts above are exact" in blk)

xlsx = inputs / "q1-sales.xlsx"
xdoc = documents.save([{"name": xlsx.name, "data": base64.b64encode(xlsx.read_bytes()).decode()}], store)[0]
xblk = documents.block(xdoc)
check("a spreadsheet's text columns get counts too", "- Region:" in xblk and "- Product:" in xblk)
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
