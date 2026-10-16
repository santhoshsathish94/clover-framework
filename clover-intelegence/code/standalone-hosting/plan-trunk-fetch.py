#!/usr/bin/env python3
"""Exactly how much must be downloaded to requantise the trunk, and from where.

The trunk is the only part worth converting to MXFP4: it streams from RAM at
83% of the memory ceiling, and a measured int8-vs-mxfp4 kernel ratio of 1.585x
puts the gain at roughly 16% of a token. The 896 routed experts are already
mxfp4 and are not in scope.

The BF16 source is gone from this box except layer 0, but
model.safetensors.index.json survives and maps every tensor to its shard. That
is enough to say precisely which shards carry trunk tensors, how many bytes
that is, and whether it can be streamed within 59 GB of free disk rather than
staged whole.

Nothing is fetched here. This only sizes the job.

usage: plan-trunk-fetch.py
"""
import collections
import json
import os
import re

INDEX = "/root/k3model/model.safetensors.index.json"
SHARD_DIR = "/root/k3model"
GB = float(1 << 30)

# routed experts are already mxfp4 and out of scope; everything else is trunk
EXPERT = re.compile(r"\.experts\.\d+\.")


def main():
    idx = json.load(open(INDEX))
    wmap = idx.get("weight_map", {})
    meta = idx.get("metadata", {})
    print("%d tensors mapped, metadata total_size %s"
          % (len(wmap), meta.get("total_size", "?")))

    trunk_shards = collections.Counter()
    expert_shards = collections.Counter()
    for name, shard in wmap.items():
        if EXPERT.search(name):
            expert_shards[shard] += 1
        else:
            trunk_shards[shard] += 1

    print("\n%d tensors are routed experts, %d are trunk"
          % (sum(expert_shards.values()), sum(trunk_shards.values())))
    print("shards containing any trunk tensor: %d of %d"
          % (len(trunk_shards), len(set(wmap.values()))))

    pure = [s for s in trunk_shards if s not in expert_shards]
    mixed = [s for s in trunk_shards if s in expert_shards]
    print("  trunk-only shards: %d" % len(pure))
    print("  shards mixing trunk and experts: %d" % len(mixed))

    present, missing, present_b = [], [], 0
    for s in sorted(trunk_shards):
        p = os.path.join(SHARD_DIR, s)
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            present.append(s)
            present_b += os.path.getsize(p)
        else:
            missing.append(s)
    print("\nshards with trunk tensors already on disk: %d (%.2f GB)"
          % (len(present), present_b / GB))
    print("shards that would need fetching: %d" % len(missing))

    # size the download from the index's own byte accounting where available
    total = meta.get("total_size")
    if total and len(wmap):
        avg = total / float(len(set(wmap.values())))
        print("average shard %.2f GB; estimated fetch %.1f GB if whole shards"
              % (avg / GB, avg * len(missing) / GB))

    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    print("\nfree disk: %.1f GB" % (free / GB))
    print("int8 trunk in place: 52.39 GB; mxfp4 output would be ~27.8 GB")
    print("streaming one shard at a time needs ~(1 shard + 27.8 GB) peak")

    print("\ntrunk tensor name shapes present in the index (first 12 distinct):")
    seen = set()
    for name in wmap:
        if EXPERT.search(name):
            continue
        key = re.sub(r"\.\d+\.", ".N.", name)
        if key not in seen:
            seen.add(key)
            if len(seen) <= 12:
                print("  %s" % key)
    print("  ... %d distinct trunk tensor patterns" % len(seen))


if __name__ == "__main__":
    main()
