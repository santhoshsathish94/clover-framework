#!/usr/bin/env python3
# Driver v2: compares the verdicts of the independent verifier's 3,000-unit run on the old driver (mutants/results-3000.tsv) with the run of the same 122 mutants + control on the new driver
# (out/$SUB/results.tsv, written by v2/analyze2.py). Prints the counts, every mutant whose verdict changed, every regression (caught before, not now) and writes v2/compare.tsv.
# usage (box, ~/rustmig/poolinit/verify): python3 v2/compare.py [SUB]        (SUB default v2mut)
import os, re, sys

V = os.path.expanduser("~/rustmig/poolinit/verify")
SUB = sys.argv[1] if len(sys.argv) > 1 else "v2mut"

def load(path, last=True):
    d = {}
    for l in open(path, errors="replace"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        d[p[0]] = p
    return d

old = load(V + "/mutants/results-3000.tsv")
new = load("%s/out/%s/results.tsv" % (V, SUB))

def nfail(p):
    return int(p[4])

def kind(p):   # what kind of line the first failure is
    t = p[7] if len(p) > 7 else ""
    for key, name in (("the seconds of the summary line", "seconds"), ("serial mode", "serial order"), ("gate ", "gate timeout"), ("distinct threads mapped", "barrier"), ("error_string was called", "error_string count"),
                      ("write() calls", "write count"), ("event recorder cap", "event cap"), ("poolinit:", "refusal text (A)")):
        if key in t:
            return name
    return "-"

names = sorted(set(old) | set(new), key=lambda s: (len(s), s))
rows, newly, lost, same_c, same_n = [], [], [], 0, 0
for n in names:
    o, w = old.get(n), new.get(n)
    if not o or not w:
        rows.append((n, "missing", "missing")); continue
    oc, wc = o[2] == "CAUGHT", w[2] == "CAUGHT"
    note = ""
    if wc and not oc: newly.append(n); note = "NEWLY CAUGHT"
    elif oc and not wc: lost.append(n); note = "REGRESSION"
    elif oc and wc: same_c += 1; note = "caught both"
    else: same_n += 1; note = "not caught either"
    first_scn = w[6]
    rows.append((n, o[2], nfail(o), w[2], nfail(w), note, first_scn, kind(w)))
with open(V + "/v2/compare.tsv", "w") as f:
    f.write("mutant\told verdict\told failing units/3000\tnew verdict\tnew failing units/3060\tchange\tfirst failing unit (new)\tfirst failure kind (new)\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
oc = sum(1 for n in names if n in old and old[n][2] == "CAUGHT" and n != "M00")
wc = sum(1 for n in names if n in new and new[n][2] == "CAUGHT" and n != "M00")
print("mutants (without M00): old caught %d, new caught %d, of %d" % (oc, wc, len([n for n in names if n != 'M00'])))
print("control M00: old %s, new %s" % (old.get("M00", ["", "", "?"])[2], new.get("M00", ["", "", "?"])[2]))
print("newly caught (%d): %s" % (len(newly), " ".join(newly)))
print("regressions (%d): %s" % (len(lost), " ".join(lost) or "none"))
print("caught in both %d, not caught in either %d" % (same_c, same_n))
print("still not caught: %s" % " ".join(n for n in names if n in new and new[n][2] != "CAUGHT" and n != "M00"))
