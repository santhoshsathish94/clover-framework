"""Experiment 3c error analysis: for each failed synonym equivalence, which word's chosen meaning was not shared, and
whether that word's lemma had SemCor training examples (usage_trained). Usage: python analyse_3c.py <v02 json> [set]"""
import collections
import json
import sys


def main():
    res = json.load(open(sys.argv[1]))
    split = sys.argv[2] if len(sys.argv) > 2 else "P"
    by_pair, layer = collections.Counter(), collections.Counter()
    examples = []
    for e in res[split]["errors"]:
        if e["label"] != 1:
            layer["control judged equivalent"] += 1
            continue
        d = e["differences"]
        kinds = {x["type"] for x in d}
        if not kinds <= {"predicate", "role:ARG0", "role:ARG1"} or len(d) != 1:
            layer["other difference (parser or structure)"] += 1
            examples.append({"a": e["a"], "b": e["b"], "differences": d})
            continue
        slot = "predicate" if "predicate" in kinds else d[0]["type"].split(":")[1]
        sa, sb = e["senses"]
        if slot == "predicate":
            (la, ka), (lb, kb) = sa["predicate"], sb["predicate"]
        else:
            (la, ka), (lb, kb) = sa["roles"].get(slot, [None, None]), sb["roles"].get(slot, [None, None])
        by_pair[(la, lb)] += 1
        layer[f"dictionary: meaning not shared ({slot})"] += 1
        examples.append({"a": e["a"], "b": e["b"], "words": [la, lb],
                         "chosen": [ka and ka["synset"], kb and kb["synset"]],
                         "usage_trained": [ka and ka["usage_trained"], kb and kb["usage_trained"]]})
    print(json.dumps({"errors_by_layer": layer, "failed_pairs": [[f"{a}/{b}", n] for (a, b), n in by_pair.most_common()]},
                     indent=1))
    for x in examples[:60]:
        print(json.dumps(x))


if __name__ == "__main__":
    main()
