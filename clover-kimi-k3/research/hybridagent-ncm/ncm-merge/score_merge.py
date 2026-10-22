"""Facts-retention probe for HybridAgent text -> NCM XSum. Key strings were fixed BEFORE any NCM output on HybridAgent text was read.
Crude: case-insensitive regex match with commas removed; it counts mentions, not correctness.
For each item it reports which keys are present in the SOURCE handed to NCM (a truncated source may not hold them) and which survive in the NCM text."""
import json, re, sys

KEYS = {
    "control": ["Harbourside", "Mill Road", r"7 to 2", r"2\.4", "Priya|Raman", "March", r"14 weeks", "Eliot", r"\b40\b", "Dock Street"],
    "q4": [r"\b24\b", r"\b100\b", r"trailing zero|zeros", r"factorial|100!", r"factors? of 5|multiples? of 5|5s\b"],
    "q5": [r"-12", r"-8\b", r"\b36\b", "circle", "reflect"],
    "q1": [r"\b29\b", r"\b343\b", "sound", r"10 ?(km|kilomet)", "second"],
}


def norm(t):
    return t.replace(",", "").replace("\u00a0", " ")


def hit(k, text):
    return bool(re.search(k, norm(text), re.I))


inputs = {(i["id"], i["variant"]): i["text"] for i in json.load(open(sys.argv[1], encoding="utf-8"))}
res = json.load(open(sys.argv[2], encoding="utf-8"))
tot_src = tot_kept = 0
for r in res:
    ks = KEYS[r["id"]]
    src = inputs[(r["id"], r["variant"])]
    in_src = [hit(k, src) for k in ks]
    kept = [hit(k, r["shipped_text"]) for k in ks]
    kept_of_src = [a and b for a, b in zip(in_src, kept)]
    sm = [[hit(k, s) for k in ks] for s in r["samples"]]
    mean = sum(sum(x) for x in sm) / max(1, len(sm))
    tot_src += sum(in_src); tot_kept += sum(kept_of_src)
    print(f"[{r['id']} {r['variant']:<10}] src {r['src_tokens']:>4} tok{' (truncated)' if r['src_truncated'] else ''}  keys in source {sum(in_src)}/{len(ks)}  "
          f"shipped keeps {sum(kept_of_src)} of those  samples mean {mean:.1f}  gen {r['generate_ms_B1']} ms")
    print("   shipped:", r["shipped_text"].strip())
print(f"\nkeys present in sources: {tot_src}; of those, present in the shipped NCM text: {tot_kept}")
