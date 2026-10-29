#!/usr/bin/env python3
# Per mutant caught by the first pass: failing units there (of 700) against v2 (of 1120); lists the ones with fewer failing units now. Reads mutants/logs/table.md and v2/logs/table-v2.md (read only).
import os, re
V = os.path.expanduser("~/rustmig/supply/verify")
old, new = {}, {}
for l in open(os.path.join(V, "mutants/logs/table.md"), errors="replace"):
    m = re.match(r"\| (M\d\d) \| (.*) \| (?:[0-9a-f]{32}|-) \| (driver FAIL|not caught by driver)[^|]*\| (\d+) / (\d+) \|", l)
    if m: old[m.group(1)] = int(m.group(4))
for l in open(os.path.join(V, "v2/logs/table-v2.md"), errors="replace"):
    m = re.match(r"\| (M\d\d) \| (.*) \| (driver FAIL|not caught by driver) \| (\d+) / (\d+) \|", l)
    if m: new[m.group(1)] = int(m.group(4))
caught = [n for n in old if old[n] > 0]
fewer = [(n, old[n], new[n]) for n in caught if new[n] < old[n]]
more = [n for n in caught if new[n] > old[n]]
same = [n for n in caught if new[n] == old[n]]
print("caught before:", len(caught), " fewer failing units now:", fewer, " equal:", len(same), " more:", len(more))
print("min new count among them:", min((new[n], n) for n in caught))
print("lowest 6 new counts overall:", sorted((new[n], n) for n in new if new[n] > 0)[:6])
