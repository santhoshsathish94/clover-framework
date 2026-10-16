#!/usr/bin/env python3
"""Do layers share weight blocks? The manifests carry a sha256 per record.

The trunk is 52.39 GB and at decode it is streamed to do 2 FLOP/byte of work, so
the cost is data movement, not arithmetic. The only ways that gets cheaper are
fewer bytes per weight, skipping weights, or discovering that bytes thought to be
distinct are in fact the same bytes read more than once.

The last one is free to test and has not been tested: every record in every
trunk-N/manifest.json already carries a sha256 of its contents. If layer L and
layer L+12 hold identical blocks, the repeat structure is real and the distinct
working set is far smaller than the file total.

This reports duplication overall, then specifically whether the period is 12.
A negative result is as useful as a positive one and is reported as such.

usage: trunk-dedup.py [dataset-dir]
"""
import collections
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
    recs = []          # (layer, slot, bytes, sha)
    for n in range(93):
        p = os.path.join(root, "trunk-%d" % n, "manifest.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        for r in m.get("records", []):
            sid = r.get("id")
            name = SLOTN[sid] if isinstance(sid, int) and sid < len(SLOTN) else "id%s" % sid
            recs.append((n, name, r.get("bytes", 0), r.get("sha256")))

    total = sum(r[2] for r in recs)
    by_sha = collections.defaultdict(list)
    unhashed = []
    for layer, slot, b, sha in recs:
        if sha:
            by_sha[sha].append((layer, slot, b))
        else:
            unhashed.append((layer, slot, b))

    hashed_total = sum(b for _, _, b in (r for v in by_sha.values() for r in v))
    distinct = sum(v[0][2] for v in by_sha.values())
    dup_groups = {s: v for s, v in by_sha.items() if len(v) > 1}

    print("%d records, %.4f GB total" % (len(recs), total / GB))
    # unhashed records cannot be compared, so they are excluded from the claim entirely
    if unhashed:
        layers = sorted(set(l for l, _, _ in unhashed))
        print("%d records carry no sha256, %.4f GB, layers %s - NOT COMPARED"
              % (len(unhashed), sum(b for _, _, b in unhashed) / GB,
                 layers if len(layers) <= 6 else "%s..%s" % (layers[0], layers[-1])))
    print("%d hashed records, %.4f GB; %d distinct sha256, %.4f GB distinct"
          % (len(recs) - len(unhashed), hashed_total / GB, len(by_sha), distinct / GB))
    print("duplicated blocks: %d groups, %.4f GB redundant (%.2f%% of the hashed bytes)"
          % (len(dup_groups), (hashed_total - distinct) / GB,
             100.0 * (hashed_total - distinct) / hashed_total if hashed_total else 0.0))

    if not dup_groups:
        print()
        print("No two hashed blocks in the trunk are byte-identical.")
        print("Every layer holds its own weights; there is nothing to deduplicate,")
        print("and a mod-12 repeat is not present in the weights themselves.")
        return

    # Where the duplication is big enough to matter
    big = sorted(dup_groups.values(), key=lambda v: -v[0][2] * (len(v) - 1))[:15]
    print()
    print("largest redundancies")
    for v in big:
        layers = sorted(set(l for l, _, _ in v))
        slots = sorted(set(s for _, s, _ in v))
        print("  %-8s %7.3f GB each x%d  saved %7.3f GB  layers %s"
              % (",".join(slots), v[0][2] / GB, len(v),
                 v[0][2] * (len(v) - 1) / GB,
                 layers if len(layers) <= 10 else "%s..%s (%d)" % (layers[0], layers[-1], len(layers))))

    # Is the period 12?
    print()
    print("layer-offset histogram for duplicate pairs of the same slot")
    offs = collections.Counter()
    for v in dup_groups.values():
        ls = sorted(set(l for l, _, _ in v))
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                offs[ls[j] - ls[i]] += 1
    for d, c in sorted(offs.items())[:20]:
        mark = "  <-- 12" if d == 12 else ""
        print("  offset %-4d %6d pairs%s" % (d, c, mark))


if __name__ == "__main__":
    main()
