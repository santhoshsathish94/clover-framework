#!/usr/bin/env python3
"""What the seeded units contain (from the configurations, which are a pure function of (scenario, seed)): classes of the host cache's chunk counts and faults, and of the card pool's slot count
(exactly 64, 63, 62, wrapped counts, refused, ...), readers count, failing-call position. Usage: coverage.py ./pi_driver [H=600,HX=400,...]   (the counts of one run, default as in run.sh)"""
import collections, re, subprocess, sys

driver = sys.argv[1]
counts = {"H": 600, "HX": 400, "P": 600, "PX": 800, "PR": 200, "M": 400}
if len(sys.argv) > 2:
    for kv in sys.argv[2].split(","):
        k, v = kv.split("=")
        counts[k] = int(v)

def i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x

host = collections.Counter()
pool = collections.Counter()
for scn, n in counts.items():
    for seed in range(1, n + 1):
        out = subprocess.run([driver, "cfg", "--scn", scn, "--seed", str(seed)], stdout=subprocess.PIPE, check=True).stdout.decode()
        for part in out.strip().split(" | ")[1:]:
            kv = dict(re.findall(r"(\w+)=(\S+)", part))
            if part.startswith("H "):
                raw, nb = int(kv["raw"]), int(kv["bytes"])
                nch = nb // raw // 32
                key = scn
                host[(key, "steps")] += 1
                host[(key, "chunks=0" if nch == 0 else "chunks=1" if nch == 1 else "chunks=2..3" if nch <= 3 else "chunks=4..8" if nch <= 8 else "chunks=9..24")] += 1
                host[(key, "bytes not a multiple of the chunk")] += (nb % (32 * raw)) != 0
                host[(key, "raw not a multiple of 4096")] += raw % 4096 != 0
                host[(key, "poison (content check)")] += kv["poison"] == "1"
                host[(key, "no poison (mincore check)")] += kv["poison"] == "0"
                host[(key, "src_file NULL")] += kv["file"] == "NULL"
                host[(key, "fake cache stub")] += kv["fake"] == "1"
                host[(key, "fail: set_device")] += int(kv["fail_at"]) == 1
                host[(key, "fail: a register")] += int(kv["fail_at"]) > 1
                host[(key, "fail: an mmap")] += int(kv["mmap_fail"]) > 0
                host[(key, "error_string returns NULL")] += kv["nullerr"] == "1" and (int(kv["fail_at"]) > 0)
            else:
                free, res, raw, rd, fa = int(kv["free"]), int(kv["reserve"]), int(kv["raw"]), int(kv["readers"]), int(kv["fail_at"])
                usable = free - res if free > res else 0
                slots = i32(usable // raw)
                key = scn
                pool[(key, "steps")] += 1
                cls = "slots<0 (wrapped)" if slots < 0 else "slots=0" if slots == 0 else "slots=1..61" if slots < 62 else "slots=62" if slots == 62 else "slots=63" if slots == 63 else "slots=64 (exactly)" if slots == 64 else "slots=65..199" if slots < 200 else "slots>=200"
                pool[(key, cls)] += 1
                pool[(key, "true quotient differs from the int (wrap)")] += (usable // raw) != slots
                pool[(key, "free <= reserve")] += free <= res
                pool[(key, "remainder bytes beyond whole slots")] += (usable % raw) != 0
                pool[(key, "readers<=0")] += rd <= 0
                pool[(key, "readers=1..31")] += 1 <= rd <= 31
                pool[(key, "readers=32")] += rd == 32
                pool[(key, "readers>32 (clamped)")] += rd > 32
                re_ = max(0, min(rd, 32))
                pool[(key, "op failure configured")] += fa > 0
                if slots >= 64:
                    pool[(key, "op failure inside the run (accepted pools)")] += 0 < fa <= 6 + 2 * re_
                pool[(key, "real constants raw=17547264")] += raw == 17547264
                pool[(key, "src_file NULL")] += kv["file"] == "NULL"
                pool[(key, "fake cache stub")] += kv["fake"] == "1"
for title, c in (("host-cache steps", host), ("card-pool steps", pool)):
    print(title)
    for scn in counts:
        keys = sorted(k for k in c if k[0] == scn)
        if not keys:
            continue
        print("  " + scn + ": " + ", ".join("%s %d" % (k[1], c[k]) for k in keys if c[k]))
