#!/usr/bin/env python3
"""The full byte composition of the trunk, by slot, from the 93 static manifests.

The per-operator report says "Q int8 projection" costs 43.7% of a token and moves
53.8 GB. That is one operator's row summed over every int8 matrix in the model -
roughly 37 slots x 93 layers. It cannot say which slots hold the bytes, so it
cannot say what would be worth changing.

Each trunk-N/manifest.json lists its records with a slot id, a group file, shape
and byte count. The id is the SLOTN index: records 0-4 are 7168x1 float32
vectors, matching ARN, ARP, MRN, MRP, IN_LN. Summing by id gives the composition
exactly, with no engine run and no estimate.

usage: trunk-composition.py [dataset-dir]
"""
import json
import os
import sys

SLOTN = [
    "ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
    "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
    "QA", "QAN", "QB", "KA", "KAN", "KB",
    "MGATE", "MUP", "MDOWN",
    "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2",
]

GB = float(1 << 30)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/clover-k3/clover-intelegence/dataset"
    by_slot = {}
    by_group = {}
    by_attn = {}
    layers = 0
    shapes = {}

    for n in range(93):
        path = os.path.join(root, "trunk-%d" % n, "manifest.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        layers += 1
        attn = m.get("attention", "?")
        for r in m.get("records", []):
            sid = r.get("id")
            name = SLOTN[sid] if isinstance(sid, int) and sid < len(SLOTN) else "id%s" % sid
            b = r.get("bytes", 0)
            by_slot[name] = by_slot.get(name, 0) + b
            by_group[r.get("group", "?")] = by_group.get(r.get("group", "?"), 0) + b
            by_attn.setdefault(attn, {})
            by_attn[attn][name] = by_attn[attn].get(name, 0) + b
            shapes.setdefault(name, set()).add((r.get("rows"), r.get("columns"), r.get("kind")))

    total = sum(by_slot.values())
    print("%d layers, %.4f GB total" % (layers, total / GB))

    print()
    print("by group file")
    for g, b in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print("  %-20s %8.3f GB  %5.1f%%" % (g, b / GB, 100.0 * b / total))

    print()
    print("by slot, largest first")
    print("  %-8s %10s %7s  %-6s %s" % ("slot", "GB", "share", "kind", "shape(s)"))
    run = 0.0
    for name, b in sorted(by_slot.items(), key=lambda kv: -kv[1]):
        sh = sorted(shapes[name])
        kind = sh[0][2] if sh else "?"
        dims = " ".join("%sx%s" % (r, c) for r, c, _ in sh[:3])
        run += 100.0 * b / total
        print("  %-8s %10.4f %6.2f%% %6s  %s   (cum %.1f%%)"
              % (name, b / GB, 100.0 * b / total, kind, dims, run))

    print()
    print("KDA vs MLA layers")
    for attn, d in sorted(by_attn.items()):
        t = sum(d.values())
        top = sorted(d.items(), key=lambda kv: -kv[1])[:5]
        print("  %-5s %8.3f GB   top: %s" % (attn, t / GB,
              ", ".join("%s %.2fGB" % (k, v / GB) for k, v in top)))


if __name__ == "__main__":
    main()
