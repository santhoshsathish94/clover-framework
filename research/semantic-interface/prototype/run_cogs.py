"""Experiment 6a (E4 external role check): CSI v0.1 propositions against COGS gold logical forms (Kim & Linzen 2020,
MIT). CSI is not trained on COGS; the test asks whether its main proposition (predicate, ARG0/1/2 heads, embedding)
matches the gold agent/theme/recipient of the same event across COGS's structures (test = in-distribution
structures, gen = 21 generalisation cases). Usage: python run_cogs.py <out json> <COGS data dir>"""
import collections
import functools
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, analyze  # noqa: E402

UNARY = re.compile(r"^(\w+) \( x _ (\d+) \)$")
BINARY = re.compile(r"^(\w+) \. (\w+(?: \. \w+)?) \( x _ (\d+) , (?:x _ (\d+)|(\w+)) \)$")
ROLE_TO_ARG = {"agent": "ARG0", "theme": "ARG1", "recipient": "ARG2"}


def parse_lf(lf):
    """-> {event index: {"verb": lemma, "roles": {ARG: head}, "ccomp"/"xcomp": child event}}; main event index."""
    nouns, events, links = {}, {}, []
    conj = [c.strip().lstrip("* ").strip() for part in lf.split(" ; ") for c in part.split(" AND ")]
    for c in conj:
        m = UNARY.match(c)
        if m:
            nouns[int(m.group(2))] = m.group(1)
            continue
        m = BINARY.match(c)
        if not m:
            raise ValueError(f"unparsed conjunct: {c!r}")
        pred, role, e = m.group(1), m.group(2), int(m.group(3))
        arg = int(m.group(4)) if m.group(4) is not None else m.group(5)
        if role.startswith("nmod"):
            continue
        ev = events.setdefault(e, {"verb": pred, "roles": {}})
        if role in ("ccomp", "xcomp"):
            ev[role] = arg
            links.append(arg)
        else:
            ev["roles"][ROLE_TO_ARG[role]] = arg
    for ev in events.values():
        ev["roles"] = {k: (nouns.get(v, f"x{v}") if isinstance(v, int) else v).lower() for k, v in ev["roles"].items()}
    mains = [e for e in events if e not in links]
    return events, (mains[0] if len(mains) == 1 else None)


@functools.lru_cache(maxsize=None)
def csi_prop(text):
    an = analyze(text)
    props = an["readings"][0]["propositions"]
    if not props or an["unsupported"]:
        return None, an["unsupported"]
    p = props[0]
    return {"predicate": p["predicate"]["lemma"], "embedded_under": p["embedded_under"],
            "roles": {k: v["head_lemma"].lower() for k, v in p["roles"].items()}}, []


def gold_for(prop, events, main):
    """Gold event CSI's proposition should describe: the main event, or its xcomp child when CSI embeds (rule 7)."""
    ev = events[main]
    if prop["embedded_under"] and prop["embedded_under"] == ev["verb"] and "xcomp" in ev:
        return events.get(ev["xcomp"]), "xcomp"
    return ev, "main"


def score_split(rows):
    n = covered = pred_ok = exact = 0
    role_tp, role_gold, role_pred = collections.Counter(), collections.Counter(), collections.Counter()
    patterns, errors = collections.Counter(), []
    for text, lf, case in rows:
        n += 1
        try:
            events, main = parse_lf(lf)
        except ValueError as e:
            patterns["gold LF not parsed"] += 1
            errors.append({"text": text, "case": case, "why": str(e)})
            continue
        prop, unsup = csi_prop(text)
        if main is None:
            patterns["gold LF has no single main event"] += 1
            errors.append({"text": text, "case": case, "why": lf})
            continue
        if prop is None:
            patterns["unsupported"] += 1
            errors.append({"text": text, "case": case, "why": unsup})
            continue
        covered += 1
        gold, how = gold_for(prop, events, main)
        if gold is None or prop["predicate"] != gold["verb"]:
            patterns["predicate"] += 1
            errors.append({"text": text, "case": case, "why": "predicate", "csi": prop, "gold": gold})
            for k in ROLE_TO_ARG.values():
                role_pred[k] += k in prop["roles"]
                role_gold[k] += gold is not None and k in gold["roles"]
            continue
        pred_ok += 1
        for k in ROLE_TO_ARG.values():
            g, c = gold["roles"].get(k), prop["roles"].get(k)
            role_gold[k] += g is not None
            role_pred[k] += c is not None
            role_tp[k] += g is not None and g == c
        if gold["roles"] == prop["roles"]:
            exact += 1
        else:
            key = "gold " + ",".join(f"{k}" for k in sorted(gold["roles"])) + " | csi " + ",".join(sorted(prop["roles"]))
            if set(gold["roles"]) == set(prop["roles"]):
                key += " (same roles, other heads)"
            patterns[key] += 1
            errors.append({"text": text, "case": case, "why": "roles", "csi": prop["roles"], "gold": gold["roles"]})
    roles = {}
    for k in ROLE_TO_ARG.values():
        p = role_tp[k] / role_pred[k] if role_pred[k] else 0.0
        r = role_tp[k] / role_gold[k] if role_gold[k] else 0.0
        roles[k] = {"p": round(p, 4), "r": round(r, 4), "n_gold": role_gold[k]}
    return {"n": n, "coverage": round(covered / n, 4), "predicate_match": round(pred_ok / n, 4),
            "exact_match": round(exact / n, 4), "roles": roles,
            "error_patterns": dict(patterns.most_common(12)), "errors": errors}


def main():
    out, d = sys.argv[1], sys.argv[2]
    data = {}
    for split in ("test", "gen"):
        rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(d, f"{split}.tsv"), encoding="utf-8")]
        data[split] = [(r[0], r[1], r[2] if len(r) > 2 else "in_distribution") for r in rows]
    t0 = time.time()
    res = {"experiment": "E6a COGS role check", "csi_version": CSI_VERSION, "splits": {}}
    for split, rows in data.items():
        r = score_split(rows)
        by_case = collections.defaultdict(list)
        for row in rows:
            by_case[row[2]].append(row)
        r["by_case"] = {}
        if split == "gen":
            for case, crow in sorted(by_case.items()):
                s = score_split(crow)
                r["by_case"][case] = {k: s[k] for k in ("n", "coverage", "predicate_match", "exact_match")}
                r["by_case"][case]["top_error"] = next(iter(s["error_patterns"].items()), None)
        r["error_sample_seed7"] = random.Random(7).sample(r["errors"], min(40, len(r["errors"])))
        del r["errors"]
        res["splits"][split] = r
        print(split, json.dumps({k: v for k, v in r.items() if k not in ("error_sample_seed7",)}, indent=1), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
