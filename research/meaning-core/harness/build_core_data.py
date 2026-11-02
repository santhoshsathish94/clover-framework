"""B2: build the meaning-core training data from the raw sources in data/raw (licences in data/raw/MANIFEST.tsv and
CONTEXT.md). Standard library only. Deterministic (seed 1234). Dev splits are held out by hash or by the source's own
validation split; test splits are never read.

Outputs (data/core/{train,dev}/):
  pairs.jsonl     {"a","b","label"(1 same meaning / 0 different),"source","objective"}   -> CoSENT-style ranking loss
  triplets.jsonl  {"anchor","positive","negative"(may be null),"source","objective"}    -> InfoNCE + hard negative
  relations.jsonl {"a","b","relation","source"}  a/b = "lemma: definition" or concept text -> relation classifier
  wic.jsonl       {"s1","span1","s2","span2","label"(1 same sense)}                      -> word sense in context
Objectives follow the human's six: equivalence, distinction, composition, relations, context, grounding.
Usage: python3 build_core_data.py <data dir containing raw/>"""
import collections
import gzip
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

D = sys.argv[1]
RAW, OUT = os.path.join(D, "raw"), os.path.join(D, "core")
rng = random.Random(1234)
W = {s: {k: open(os.path.join(OUT, s, f"{k}.jsonl"), "w", encoding="utf-8") for k in ("pairs", "triplets", "relations", "wic")}
     for s in ("train", "dev") if os.makedirs(os.path.join(OUT, s), exist_ok=True) is None}
COUNT = collections.Counter()


def dev_by_hash(key, pct=5):
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100 < pct


def emit(split, kind, row):
    W[split][kind].write(json.dumps(row, ensure_ascii=False) + "\n")
    COUNT[(split, kind, row.get("source"), row.get("objective") or row.get("relation") or "")] += 1


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# 1. PAWS labeled_final (equivalence / distinction with the same words): train -> train, validation -> dev
for split, fname in (("train", "train"), ("dev", "validation")):
    for r in jsonl(os.path.join(RAW, "paws_wiki", f"{fname}.jsonl")):
        emit(split, "pairs", {"a": r["sentence1"], "b": r["sentence2"], "label": int(r["label"]), "source": "paws",
                              "objective": "equivalence" if r["label"] == 1 else "distinction"})

# 2. SNLI + MNLI: premise -> entailed hypothesis (positive), contradiction (hard negative); train -> train,
#    validation(_matched) -> dev. Neutral is not used (related but not implied).
for name, splits in (("snli", (("train", "train"), ("dev", "validation"))),
                     ("multi_nli", (("train", "train"), ("dev", "validation_matched")))):
    for split, fname in splits:
        groups = collections.defaultdict(lambda: {0: [], 2: []})
        for r in jsonl(os.path.join(RAW, name, f"{fname}.jsonl")):
            if r["label"] in (0, 2):
                groups[r["premise"]][r["label"]].append(r["hypothesis"])
        for prem, g in groups.items():
            for pos in g[0]:
                neg = rng.choice(g[2]) if g[2] else None
                emit(split, "triplets", {"anchor": prem, "positive": pos, "negative": neg, "source": name,
                                         "objective": "equivalence" if neg is None else "equivalence+distinction"})

# 3. CondaQA (negation): the edited sentence of each edited passage vs the original negated sentence.
#    Edit 1 paraphrase -> label 1; edit 2 scope change and edit 3 negation undone -> label 0. train -> train, validation -> dev.
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
for split, fname in (("train", "train"), ("dev", "validation")):
    seen = set()
    by_passage = collections.defaultdict(dict)
    for r in jsonl(os.path.join(RAW, "condaqa", f"{fname}.jsonl")):
        if r["PassageID"] is None:
            continue
        key = (r["PassageID"], int(r["PassageEditID"]))
        if key in seen:
            continue
        seen.add(key)
        by_passage[r["PassageID"]][int(r["PassageEditID"])] = r
    for pid, eds in by_passage.items():
        anyr = next(iter(eds.values()))
        orig_sents = set(SENT.split(anyr["original passage"].strip()))
        trip = {}
        for e in (1, 2, 3):
            if e not in eds:
                continue
            changed = [s for s in SENT.split(eds[e]["sentence1"].strip()) if s not in orig_sents]
            if len(changed) != 1:
                COUNT[(split, "skipped", "condaqa", f"edit{e}_changed_{min(len(changed), 3)}")] += 1
                continue
            wa = set(re.findall(r"\w+", anyr["original sentence"].lower()))
            wb = set(re.findall(r"\w+", changed[0].lower()))
            if not wa or len(wa & wb) / len(wa | wb) < 0.4:  # the changed sentence must be an edit of the negated one
                COUNT[(split, "skipped", "condaqa", f"edit{e}_low_overlap")] += 1
                continue
            label = 1 if e == 1 else 0
            trip[e] = changed[0]
            emit(split, "pairs", {"a": anyr["original sentence"], "b": changed[0], "label": label, "source": "condaqa",
                                  "objective": {1: "equivalence", 2: "distinction:negation_scope",
                                                3: "distinction:negation_undone"}[e]})
        if 1 in trip and (3 in trip or 2 in trip):
            emit(split, "triplets", {"anchor": anyr["original sentence"], "positive": trip[1],
                                     "negative": trip.get(3, trip.get(2)), "source": "condaqa",
                                     "objective": "equivalence+distinction:negation"})

# 4. PropBank frames: ARG0/ARG1 role swap -> same words, different who-did-what (composition, label 0).
#    Only plain verbal clauses: ARG0 right before the predicate (<= 2 tokens between, e.g. an auxiliary), ARG1 right
#    after it (<= 1 token, e.g. a particle), both short plain noun phrases (no preposition, possessive, pronoun,
#    reflexive or trace) with the same crude number (both or neither ending in -s), so the swap stays grammatical
#    and only the roles change. Noun predicates ("the chemist's concentration of the fluid") are excluded.
PRON = {"he", "she", "they", "him", "her", "them", "i", "me", "we", "us", "you", "it", "who", "which", "that", "what",
        "themselves", "himself", "herself", "itself", "myself", "yourself", "yourselves", "ourselves", "oneself",
        "this", "these", "those", "there", "here", "one", "someone", "something", "anyone", "anything", "everyone"}
PREP = {"of", "to", "with", "by", "for", "from", "in", "on", "at", "into", "about", "as", "than", "over", "under",
        "through", "between", "among", "against", "without", "within", "during", "after", "before", "because", "if"}
FUNC_CAP = {"The", "A", "An", "His", "Her", "Their", "Its", "This", "That", "These", "Those", "Some", "Many", "Most",
            "All", "Our", "My", "Your", "Each", "Every", "No", "Several", "Both"}


def plain_np(span):
    low = [t.lower() for t in span]
    return (1 <= len(span) <= 6 and low[0] not in PREP and not any(t in PRON for t in low) and "'s" not in low
            and not any(t.startswith("*") or t.startswith("-") or t in {",", ";", ":", "(", ")", "``", "''"} for t in span))


def detok(tokens):
    s = " ".join(t for t in tokens if not re.fullmatch(r"\*[^ ]*\*(-\d+)?|\*|0|\*-\d+", t))
    for a, b in ((" 's", "'s"), (" n't", "n't"), (" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), ("( ", "("),
                 (" )", ")"), ("`` ", '"'), (" ''", '"'), ("-LRB-", "("), ("-RRB-", ")"), (" %", "%"), ("$ ", "$")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


frames_dir = os.path.join(RAW, "propbank-frames", "frames")
for fn in sorted(os.listdir(frames_dir)):
    if not fn.endswith(".xml"):
        continue
    split = "dev" if dev_by_hash("propbank:" + fn) else "train"
    try:
        root = ET.parse(os.path.join(frames_dir, fn)).getroot()
    except ET.ParseError:
        COUNT[(split, "skipped", "propbank", "xml_error")] += 1
        continue
    for ex in root.iter("example"):
        text_el, pb = ex.find("text"), ex.find("propbank")
        if text_el is None or pb is None or not (text_el.text or "").strip():
            continue
        toks = text_el.text.strip().split(" ")
        rel = pb.find("rel")
        relloc = (rel.get("relloc") or "").split() if rel is not None else []
        if len(relloc) != 1 or not relloc[0].isdigit():
            continue
        rpos = int(relloc[0])
        lemma = fn[:-4].lower()
        if rpos >= len(toks) or not toks[rpos].lower().startswith(lemma[:max(3, len(lemma) - 2)]):
            # the predicate token must be a form of the frame's verb; some examples have shifted indices
            # (commute.xml: rel="her", ARG0="colonial government commuted")
            COUNT[(split, "skipped", "propbank", "rel_not_lemma")] += 1
            continue
        args = {}
        for a in pb.findall("arg"):
            if a.get("type") in ("ARG0", "ARG1") and (a.get("start") or "").isdigit() and (a.get("end") or "").isdigit():
                s_, e_ = int(a.get("start")), int(a.get("end"))
                if (a.text or "").strip() != " ".join(toks[s_:e_ + 1]):  # indices must match the argument's own text
                    COUNT[(("dev" if dev_by_hash("propbank:" + fn) else "train"), "skipped", "propbank", "span_mismatch")] += 1
                    args = {}
                    break
                args[a.get("type")] = (s_, e_)
        if len(args) != 2:
            continue
        (s0, e0), (s1, e1) = args["ARG0"], args["ARG1"]
        if not (e0 < rpos < s1 and rpos - e0 <= 3 and s1 - rpos <= 2 and e1 < len(toks)):
            continue
        span0, span1 = toks[s0:e0 + 1], toks[s1:e1 + 1]
        if not (plain_np(span0) and plain_np(span1)) or (span0[-1].endswith("s") != span1[-1].endswith("s")):
            continue
        first, last = span0, span1
        swapped = toks[:s0] + list(last) + toks[e0 + 1:s1] + list(first) + toks[e1 + 1:]
        if s0 == 0:
            pos_first = len(last) + (s1 - e0 - 1)
            if first[0] in FUNC_CAP:
                swapped[pos_first] = first[0].lower()
            swapped[0] = swapped[0][:1].upper() + swapped[0][1:]
        a, b = detok(toks), detok(swapped)
        if a and b and a != b and len(a.split()) >= 4:
            emit(split, "pairs", {"a": a, "b": b, "label": 0, "source": "propbank_frames",
                                  "objective": "composition:role_swap"})

# 5. Open English WordNet 2025: synsets, sense relations, examples.
lemma_of_sense, synset_info, sense_rel, syn_rel = {}, {}, [], []
lex = collections.defaultdict(list)  # (lemma, pos) -> synsets
with gzip.open(os.path.join(RAW, "english-wordnet-2025.xml.gz")) as f:
    for _, el in ET.iterparse(f, events=("end",)):
        if el.tag == "LexicalEntry":
            lem = el.find("Lemma")
            form, pos = lem.get("writtenForm"), lem.get("partOfSpeech")
            for s in el.findall("Sense"):
                lemma_of_sense[s.get("id")] = (form, s.get("synset"))
                lex[(form.lower(), pos)].append(s.get("synset"))
                for sr in s.findall("SenseRelation"):
                    sense_rel.append((s.get("id"), sr.get("relType"), sr.get("target")))
            el.clear()
        elif el.tag == "Synset":
            d = el.find("Definition")
            synset_info[el.get("id")] = {"pos": el.get("partOfSpeech"), "members": (el.get("members") or "").split(),
                                         "definition": (d.text or "").strip() if d is not None else "",
                                         "examples": [(x.text or "").strip() for x in el.findall("Example") if x.text]}
            for r in el.findall("SynsetRelation"):
                syn_rel.append((el.get("id"), r.get("relType"), r.get("target")))
            el.clear()
first_lemma = {}
for sid, (form, syn) in lemma_of_sense.items():
    first_lemma.setdefault(syn, form)


def gloss(syn):
    i = synset_info[syn]
    return f"{first_lemma.get(syn, syn)}: {i['definition']}"


KEEP_SYN = {"hypernym", "hyponym", "similar", "mero_part", "holo_part", "mero_substance", "holo_substance",
            "mero_member", "holo_member", "entails", "is_entailed_by", "causes", "is_caused_by", "attribute", "also"}
KEEP_SENSE = {"antonym", "derivation", "pertainym"}
rel_rows = collections.defaultdict(list)
for a, t, b in syn_rel:
    if t in KEEP_SYN and a in synset_info and b in synset_info and synset_info[a]["definition"] and synset_info[b]["definition"]:
        rel_rows[t].append((a, b))
for sa, t, sb in sense_rel:
    if t in KEEP_SENSE and sa in lemma_of_sense and sb in lemma_of_sense:
        a, b = lemma_of_sense[sa][1], lemma_of_sense[sb][1]
        if a != b and synset_info[a]["definition"] and synset_info[b]["definition"]:
            rel_rows[t].append((a, b))
all_syn = [s for s in synset_info if synset_info[s]["definition"]]
for t, rows in rel_rows.items():
    rng.shuffle(rows)
    for a, b in rows[:40000]:
        split = "dev" if dev_by_hash("oewn:" + min(a, b)) else "train"
        emit(split, "relations", {"a": gloss(a), "b": gloss(b), "relation": t, "source": "oewn2025"})
for _ in range(40000):  # unrelated pairs
    a, b = rng.sample(all_syn, 2)
    split = "dev" if dev_by_hash("oewn:" + min(a, b)) else "train"
    emit(split, "relations", {"a": gloss(a), "b": gloss(b), "relation": "none", "source": "oewn2025"})

# 5b. Antonym substitution in OEWN example sentences (distinction, label 0): "able to swim" -> "unable to swim"
for sa, t, sb in sense_rel:
    if t != "antonym" or sa not in lemma_of_sense or sb not in lemma_of_sense:
        continue
    (wa, syn_a), (wb, _) = lemma_of_sense[sa], lemma_of_sense[sb]
    if " " in wa or " " in wb or not wa.isalpha() or not wb.isalpha():
        continue
    for ex in synset_info[syn_a]["examples"]:
        m = re.search(rf"\b{re.escape(wa)}\b", ex)
        if m:
            split = "dev" if dev_by_hash("oewn:" + syn_a) else "train"
            new = ex[:m.start()] + wb + ex[m.end():]
            art = re.search(r"\b(a|an|A|An)\s+$", ex[:m.start()])  # keep a/an right for the new word
            if art:
                want = ("an" if wb[0].lower() in "aeiou" else "a")
                want = want.capitalize() if art.group(1)[0].isupper() else want
                new = ex[:art.start(1)] + want + ex[art.end(1):m.start()] + wb + ex[m.end():]
            emit(split, "pairs", {"a": ex, "b": new, "label": 0, "source": "oewn2025",
                                  "objective": "distinction:antonym_substitution"})
            break

# 5c. Word sense in context from OEWN examples (replaces WiC, which is non-commercial).
cands = collections.defaultdict(list)  # word -> [(synset, example, span)]
for (form, pos), syns in lex.items():
    if not form.isalpha() or len(set(syns)) < 2:
        continue
    for syn in set(syns):
        for ex in synset_info[syn]["examples"]:
            m = re.search(rf"\b{re.escape(form)}\b", ex, flags=re.I)
            if m:
                cands[form].append((syn, ex, [m.start(), m.end()]))
for word, items in cands.items():
    by = collections.defaultdict(list)
    for syn, ex, sp in items:
        by[syn].append((ex, sp))
    if len(by) < 2:
        continue
    split = "dev" if dev_by_hash("wic:" + word) else "train"
    syns = sorted(by)
    for syn in syns:  # positive: two examples of the same sense
        if len(by[syn]) >= 2:
            (e1, s1), (e2, s2) = rng.sample(by[syn], 2)
            emit(split, "wic", {"s1": e1, "span1": s1, "s2": e2, "span2": s2, "label": 1, "word": word, "source": "oewn2025"})
    for _ in range(min(3, len(syns) - 1)):  # negatives: different senses of the same word
        a, b = rng.sample(syns, 2)
        (e1, s1), (e2, s2) = rng.choice(by[a]), rng.choice(by[b])
        emit(split, "wic", {"s1": e1, "span1": s1, "s2": e2, "span2": s2, "label": 0, "word": word, "source": "oewn2025"})

# 6. ConceptNet 5.7 English edges (relations + grounding), weight >= 1, capped per relation.
CN_KEEP = {"IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation", "Causes", "HasSubevent", "HasFirstSubevent",
           "HasLastSubevent", "HasPrerequisite", "MotivatedByGoal", "CausesDesire", "Desires", "HasProperty", "MadeOf",
           "ReceivesAction", "Antonym", "DistinctFrom", "Synonym", "SimilarTo", "Entails", "CreatedBy", "LocatedNear",
           "ObstructedBy", "NotDesires", "NotCapableOf", "NotHasProperty", "NotUsedFor", "MannerOf", "DefinedAs"}


def cn_text(uri, surface):
    if surface:
        return surface
    parts = uri.split("/")
    return parts[3].replace("_", " ") if len(parts) > 3 else uri


cn = collections.defaultdict(list)
with gzip.open(os.path.join(RAW, "conceptnet-assertions-5.7.0.csv.gz"), "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue
        _, rel, start, end, meta = parts
        if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
            continue
        r = rel.split("/")[2] if rel.startswith("/r/") else rel
        if r not in CN_KEEP:
            continue
        m = json.loads(meta)
        if m.get("weight", 0) < 1.0:
            continue
        cn[r].append((cn_text(start, m.get("surfaceStart")), cn_text(end, m.get("surfaceEnd")), start, m.get("license", "")))
for r, rows in cn.items():
    rng.shuffle(rows)
    for a, b, start, lic in rows[:30000]:
        split = "dev" if dev_by_hash("cn:" + start) else "train"
        emit(split, "relations", {"a": a, "b": b, "relation": "cn:" + r, "source": "conceptnet57", "license": lic})

for s in W.values():
    for fh in s.values():
        fh.close()
stats = collections.defaultdict(dict)
for (split, kind, src, obj), n in sorted(COUNT.items()):
    stats[f"{split}/{kind}"][f"{src}|{obj}"] = n
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
for k, v in stats.items():
    print(k, sum(v.values()), dict(list(v.items())[:12]))
