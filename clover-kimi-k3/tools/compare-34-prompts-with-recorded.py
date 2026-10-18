#!/usr/bin/env python3
"""Compares a 34-prompt run (run-34-recorded-prompts.sh, run in work/results) with the recorded data in ../prompts.
(a) token ids our tokenizer gives for the prompt text against the ids recorded in recorded-34-prompts.tsv;
(b) the 12 tokens we generated, as text, against the 12 tokens recorded in recorded-34-results.md.
Environment: P34DIR = the run's directory (default recorded-34-exact)."""
import re, sys, os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")
rows = {}
for line in open(os.path.join(HERE, "recorded-34-prompts.tsv"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    rows[int(p[0])] = (p[2], p[3])

rec = {}
md = open(os.path.join(HERE, "recorded-34-results.md"), encoding="utf-8").read()
# a block runs from its heading to the next one; the output itself may contain code fences, so only the first and the last fence delimit it
for part in re.split(r"\n### (?=\d+\n)", "\n" + md)[1:]:
    num, rest = part.split("\n", 1)
    rest = rest.split("\n---\n")[0]
    start = rest.index("```\n") + 4
    body = rest[start:rest.rindex("\n```")]
    if "\n->" in body:
        k = body.index("\n->")
        rec[int(num)] = (body[:k], body[k + 3:])

ids_ok = out_ok = first_ok = 0
P = os.environ.get("P34DIR", "recorded-34-exact")
for i in sorted(rows):
    log = open("%s/%d.log" % (P, i), encoding="utf-8", errors="replace").read() if os.path.exists("%s/%d.log" % (P, i)) else ""
    m = re.search(r"prompt ids: ([\d ]+)", log)
    ours_ids = ",".join(m.group(1).split()) if m else None
    ids_match = ours_ids == rows[i][0]
    gen = open("%s/%d.out" % (P, i), encoding="utf-8", errors="replace").read() if os.path.exists("%s/%d.out" % (P, i)) else None
    gids = open("%s/%d.ids" % (P, i)).read().strip() if os.path.exists("%s/%d.ids" % (P, i)) else ""
    want = rec.get(i, (None, None))[1]
    # results.md prints "-> " followed by the output without the output's own leading space
    out_match = gen is not None and want is not None and gen.lstrip(" ") == want.lstrip(" ")
    ids_ok += ids_match; out_ok += out_match
    first_ok += bool(gen and want and gen.split()[:1] == want.split()[:1])
    print("%2d ids %s  output %s   %s" % (i, "same" if ids_match else "DIFFERENT", "same" if out_match else "DIFFERENT", rows[i][1][:50]))
    if not ids_match: print("     ours      :", ours_ids, "\n     recorded  :", rows[i][0])
    if not out_match: print("     ours      :", repr(gen), "\n     recorded  :", repr(want), "\n     our ids   :", gids)
print("prompt ids equal to the recorded ids: %d of %d; 12 generated tokens equal to the recorded text: %d of %d" % (ids_ok, len(rows), out_ok, len(rows)))
