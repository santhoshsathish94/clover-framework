#!/usr/bin/env python3
"""Survey the 96 shard headers to size the trunk fetch exactly.

The plan is to download only the trunk tensors - 2,628 of 497,220 - using byte
ranges, since whole shards would be 1.4 TB against 58.7 GB of free disk. Before
moving any weight, this establishes the real total rather than my estimate of
"about 105 GB", and how many separate ranges it takes, because thousands of
small scattered requests and a few large contiguous ones are very different
jobs.

Headers are cached to disk so the bulk fetch reuses them instead of asking
twice. Nothing but headers is downloaded here.

usage: survey-shards.py [first] [last]
"""
import json
import os
import re
import struct
import sys

import requests

REPO = "moonshotai/Kimi-K3"
COMMIT = "f831ab66814297da540d832a5235f8e904f29d06"
BASE = "https://huggingface.co/%s/resolve/%s/" % (REPO, COMMIT)
CACHE = "/tmp/k3headers"
EXPERT = re.compile(r"\.experts\.\d+\.")
GB = float(1 << 30)


def shard_name(i):
    return "model-%05d-of-000096.safetensors" % i


def get_header(session, i):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%05d.json" % i)
    if os.path.exists(path):
        return json.load(open(path))
    url = BASE + shard_name(i)
    r = session.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    r.raise_for_status()
    n = struct.unpack("<Q", r.content[:8])[0]
    r = session.get(url, headers={"Range": "bytes=8-%d" % (8 + n - 1)}, timeout=120)
    r.raise_for_status()
    hdr = json.loads(r.content[:n])
    json.dump(hdr, open(path, "w"))
    return hdr


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    s = requests.Session()

    total_trunk = 0
    total_all = 0
    ranges = 0
    per_layer = {}
    dtypes = {}
    failed = []

    for i in range(first, last + 1):
        try:
            hdr = get_header(s, i)
        except Exception as e:
            failed.append((i, str(e)[:60]))
            continue
        spans = []
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            total_all += b - a
            if EXPERT.search(name):
                continue
            total_trunk += b - a
            dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + (b - a)
            spans.append((a, b))
            m = re.search(r"layers\.(\d+)\.", name)
            key = int(m.group(1)) if m else -1
            per_layer[key] = per_layer.get(key, 0) + (b - a)
        # merge adjacent spans so the count reflects real requests
        spans.sort()
        merged = 0
        end = -1
        for a, b in spans:
            if a != end:
                merged += 1
            end = b
        ranges += merged
        if i % 16 == 0 or i == last:
            print("  shard %02d: trunk so far %.2f GB, %d ranges" % (i, total_trunk / GB, ranges))

    print()
    print("shards surveyed %d..%d, %d failed" % (first, last, len(failed)))
    for i, e in failed[:5]:
        print("  shard %d failed: %s" % (i, e))
    print("all tensors      %.2f GB" % (total_all / GB))
    print("TRUNK to fetch   %.2f GB in %d contiguous ranges" % (total_trunk / GB, ranges))
    print("dtypes: %s" % {k: round(v / GB, 2) for k, v in dtypes.items()})
    if ranges:
        print("average range    %.1f MB" % (total_trunk / ranges / 1e6))
    known = [k for k in sorted(per_layer) if k >= 0]
    if known:
        print("layers %d..%d, mean %.1f MB per layer"
              % (min(known), max(known),
                 sum(per_layer[k] for k in known) / len(known) / 1e6))
    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    print("free disk %.1f GB; per-layer staging needs ~%.1f MB"
          % (free / GB, (sum(per_layer[k] for k in known) / len(known) / 1e6) if known else 0))


if __name__ == "__main__":
    main()
