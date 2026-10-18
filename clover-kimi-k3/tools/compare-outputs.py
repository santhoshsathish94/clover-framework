#!/usr/bin/env python3
"""Side-by-side comparison of the outputs of the arithmetic modes.

  compare-outputs.py batch   the 34 recorded prompts (run in work/results): recorded-34-exact, recorded-34-use_4bit_trunk, and the older runs
                             recorded-34-decode_only_4bit_trunk_old and recorded-34-experts_8bit_activations_old; only the prompts present in a directory are compared
  compare-outputs.py single  one prompt run by compare-exact-and-4bit-trunk.sh: <prefix>-<mode>.log and <prefix>-<mode>.ids

For every mode other than exact: same, or the index of the first generated token that differs from the exact run, and the text."""
import os, re, sys

MODES = ["exact", "use_4bit_trunk", "decode_only_4bit_trunk_old", "experts_8bit_activations_old"]
DIRS = {m: "recorded-34-" + m for m in MODES}
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "recorded-34-prompts.tsv")


def read(path):
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else None


def ids_of(path):
    t = read(path)
    return t.strip().split(",") if t and t.strip() else None


def diverge(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def batch():
    prompts = {}
    for line in open(PROMPTS, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        prompts[int(f[0])] = f[3]
    stats = {m: [0, 0, []] for m in MODES if m != "exact"}   # compared, identical, divergence indices
    for n in range(1, 35):
        ex_out, ex_ids = read("%s/%d.out" % (DIRS["exact"], n)), ids_of("%s/%d.ids" % (DIRS["exact"], n))
        if ex_out is None:
            continue
        lines = []
        for m in MODES[1:]:
            out = read("%s/%d.out" % (DIRS[m], n))
            if out is None:
                continue
            same = out == ex_out
            stats[m][0] += 1
            stats[m][1] += same
            d = diverge(ex_ids, ids_of("%s/%d.ids" % (DIRS[m], n)))
            if not same:
                stats[m][2].append(d if d is not None else -1)
                lines.append("      %-8s differs from token %s: %r" % (m, d, out))
        status = " ".join("%s=%s" % (m, "same" if read("%s/%d.out" % (DIRS[m], n)) == ex_out else "DIFF")
                          for m in MODES[1:] if read("%s/%d.out" % (DIRS[m], n)) is not None)
        print("%2d %-48r [%s]" % (n, prompts[n][:46], status))
        print("      exact    : %r" % ex_out)
        for l in lines:
            print(l)
    print("\nsummary (12 generated tokens per prompt, compared with the exact run)")
    for m, (c, same, ds) in stats.items():
        if c:
            early = sum(1 for d in ds if 0 <= d < 5)
            print("  %-30s prompts compared %2d, identical %2d, different %2d (first difference before token 5: %d; mean first-difference index %.1f)"
                  % (m, c, same, c - same, early, (sum(ds) / len(ds)) if ds else 0.0))


def single():
    prefix = sys.argv[2] if len(sys.argv) > 2 else "qc"
    texts, ids = {}, {}
    for m in MODES:
        log = read("%s-%s.log" % (prefix, m))
        if not log:
            continue
        g = re.search(r"generated text \(\d+ tokens\):\n-----\n(.*?)\n-----", log, re.S)
        chat = re.search(r"response:\n-----\n(.*?)\n-----", log, re.S)   # a chat run prints the reply without control tokens
        texts[m] = chat.group(1) if chat else (g.group(1) if g else None)
        ids[m] = ids_of("%s-%s.ids" % (prefix, m))
        arith = re.search(r"ARITHMETIC: (.*)", log)
        dec = re.search(r"DECODE, .*mean ([\d.]+) s per token", log)
        req = re.search(r"REQUEST, .*= ([\d.]+) s in total", log)
        print("%-30s %s | prompt pass %s s | decode %s s per token" % (m, arith.group(1) if arith else "(no ARITHMETIC line: run before the switch existed)",
              req.group(1) if req else "?", dec.group(1) if dec else "?"))
    for m in MODES:
        if m not in texts:
            continue
        d = diverge(ids.get("exact"), ids.get(m)) if m != "exact" else None
        print("\n== %s %s" % (m, "" if m == "exact" else ("(identical to exact)" if texts[m] == texts.get("exact") else "(first different token: %s)" % d)))
        print(texts[m])


if __name__ == "__main__":
    {"batch": batch, "single": single}[sys.argv[1] if len(sys.argv) > 1 else "batch"]()
