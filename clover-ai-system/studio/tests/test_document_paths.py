"""The code hint for an attached table names its file with forward slashes (documents.block, 2026-11-11).

Observed on the full system: with the Studio's data under "...\\Temp\\te-study\\...", the model's Python opened
"...\\Temp\\e-study\\..." in 7 of 7 tries (it re-escaped the hint's backslashes and lost the "t"), so every spreadsheet
and CSV question ended with "the file is not present at the path you specified".
"""
import base64
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import documents  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


root = Path(tempfile.mkdtemp()) / "te-study" / "tables"  # a "\t" after a backslash, as in the failing runs
csv_bytes = b"order_id,customer,amount\nORD-1,Chen,12.5\nORD-2,Elif,8\nORD-3,Chen,4\n"
doc = documents.save([{"name": "orders.csv", "data": base64.b64encode(csv_bytes).decode()}], root)[0]
blk = documents.block(doc)
posix = Path(doc["path"]).as_posix()
code = next((ln.split("run_python on the whole file: ", 1)[1].split(", then use", 1)[0]
             for ln in blk.splitlines() if "run_python on the whole file:" in ln), "")
check("the code hint uses the forward-slash path", f'"{posix}"' in code and "\\" not in code)
check("the 'saved at' line still names the saved path", doc["path"] in blk)
try:
    import pandas  # noqa: F401
    ns = {"pandas": pandas}
    exec(code, ns)
    check("the hinted code opens the file as written", len(ns["df"]) == 3 and list(ns["df"].columns)[1] == "customer")
except ImportError:
    print("SKIP pandas not installed: the hinted code was not run")
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
