"""Clover's arithmetic check reads numbers with thousands separators whole and stays within a line, and checks a value
carried down lines that name it. Before 2026-11-09 it split "25,000" at the comma and read across a line break, so its
feedback on a wrong compound-interest answer was "you wrote ... = 25" and "000 x 1.225043 = 0".
Usage: python tests/test_arithmetic_findings_layout.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import clover_cycle as C  # noqa: E402

# a live LFM3-Omni-2B reply (consciousness integration, bridge A/B, "ask" mode B1), verbatim
REPLY = """Thank you for the verification. I'll correct the arithmetic and provide the accurate calculation.

**Corrected Calculation:**
- Principal (P): 25,000 rupees
- Annual interest rate (r): 7%
- Time (t): 3 years

Using the compound interest formula:
A = P × (1 + r)^t
A = 25,000 × (1 + 0.07)^3
A = 25,000 × (1.07)^3
A = 25,000 × 1.225043
A ≈ 28,251.30 rupees

**Answer:** After 3 years, your savings account will hold approximately **28,251 rupees**."""


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def findings(text):
    applied = []
    return C.arithmetic_findings(text, applied), applied


def split_number(f):  # "= 25, but" or "000 ×": a number cut at its thousands separator (not "25,000, but")
    return re.search(r"(?<![\d.,])\d{1,3},(?!\d{3})|(?<![\d,])000\b", f) is not None or "\n" in f


ok = True
got, applied = findings(REPLY)
ok &= check("the wrong carried value is found, with the whole numbers",
            len(got) == 1 and "25,000 × 1.225043" in got[0] and "28,251.30" in got[0] and "30,626.075" in got[0])
ok &= check("no fragment of a split number or a line break in the feedback", not any(map(split_number, got)))
fixed, applied2 = findings(REPLY.replace("28,251.30", "30,626.08").replace("28,251 rupees", "30,626 rupees"))
ok &= check("liveness: the same reply with the right value has no finding, and the check ran",
            fixed == [] and "arithmetic" in applied2)
ok &= check("inline, thousands separators: right", findings("So 25,000 × 2 = 50,000 in total.")[0] == [])
wrong = findings("So 25,000 × 2 = 40,000 in total.")[0]
ok &= check("inline, thousands separators: wrong, reported whole", len(wrong) == 1 and "25,000 × 2 = 50,000" in wrong[0])
ok &= check("a line break is not read as a unit word, and 25,000 is not cut to 25",
            findings("Rate = (1.07)^3\nA = 25,000 × 1.07")[0] == [])
ok &= check("a contradiction carried down a name is reported whole",
            [not split_number(f) for f in findings("A = 2 × (1.07)^3\nA = 25,000")[0]] == [True])
ok &= check("a result on the next line is still checked",
            findings("25,000 × 1.07^3\n= 30,626.08")[0] == [] and len(findings("25,000 × 1.07^3\n= 28,251.30")[0]) == 1)
ok &= check("reassigning a variable inside a code block is not a claim",
            findings("Here:\n```python\nA = 2 * 3\nA = 7\n```\nDone.")[0] == [])
ok &= check("an unrelated plain value later does not reuse an old step",
            findings("x = 3 + 4\nx = 7\nLater, with new data:\nx = 9")[0] == [])
near = findings("A = 25,000 × 1.225043\nA = 30,626.10")[0]
ok &= check("a near miss shows the value precisely enough to see the error (30,626.10 vs 30,626.075)",
            len(near) == 1 and near[0].endswith("= 30,626.075."))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
