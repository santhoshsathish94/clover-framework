"""Calibration of the fact-retention yardstick on in-domain news (full XSum test set, 11,334 documents; NCM shipped hypotheses from the earlier 8-card run).
Facts of a document = numbers and capitalised words (not the first word, not in a small stop list) in its REFERENCE summary that also occur in the SOURCE passage.
Reported: share of those facts present in (a) the NCM hypothesis, (b) the first 48 GPT-2 tokens of the source (lead-48, a trivial summariser), (c) the reference itself (=1 by construction, not shown).
Definition fixed before the numbers were computed."""
import glob, pickle, re, sys
from transformers import AutoTokenizer

D = sys.argv[1]
tok = AutoTokenizer.from_pretrained("gpt2")
passages = pickle.load(open(f"{D}/passages.pickle", "rb"))
refs_all = pickle.load(open(f"{D}/reference_summaries.pickle", "rb"))
hyp = {}
for p in sorted(glob.glob(f"{D}/eval-xsum-full/shard_*.pkl")):
    o = pickle.load(open(p, "rb"))
    for i, h in zip(o["idxs"], o["hyps"]):
        hyp[i] = h
print("docs with hypotheses:", len(hyp), "of", len(passages))

STOP = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "We", "This", "That", "These", "Those", "There", "His", "Her", "Their", "Our", "My", "I", "If", "As", "But", "And", "For", "By", "With", "From", "To", "Of", "Is", "Are", "Was", "Were"}
FACT = re.compile(r"\b\d[\d,.]*\b|\b[A-Z][a-z]+\b")


def facts(ref, src):
    words = FACT.findall(ref)
    first = ref.split()[0].strip('"') if ref.split() else ""
    out = []
    for w in words:
        if w == first or w in STOP:
            continue
        if w in src and w not in out:
            out.append(w)
    return out


tot = k_ncm = k_lead = docs = docs_with = 0
for i in sorted(hyp):
    src = tok.decode(passages[i])
    f = facts(refs_all[i], src)
    docs += 1
    if not f:
        continue
    docs_with += 1
    lead = tok.decode(passages[i][:48])
    tot += len(f)
    k_ncm += sum(w in hyp[i] for w in f)
    k_lead += sum(w in lead for w in f)
print(f"documents {docs}, with at least one fact {docs_with}, facts {tot}")
print(f"NCM shipped hypothesis keeps {k_ncm}/{tot} = {100 * k_ncm / tot:.1f}% of the reference facts that are in the source")
print(f"lead-48 tokens of the source keeps {k_lead}/{tot} = {100 * k_lead / tot:.1f}%")
