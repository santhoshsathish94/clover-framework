#!/usr/bin/env python3
# Summarises a probe event file written through PI_PROBE (lines "<pid> <tag> key=value ..."): per tag the number of lines and distinct processes, per key=value the count of
# true/false for booleans and min / median / max for numbers; for M00c (original with the "which thread maps" probe) the participation of the calling thread
# and the relation between chunk number and address, per process.
# usage: probe_count.py FILE [more files]
import collections, statistics, sys

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def summarize(path):
    tags = collections.OrderedDict()
    procs = collections.defaultdict(list)
    for l in open(path, errors="replace"):
        p = l.split()
        if len(p) < 2:
            continue
        pid, tag = p[0], p[1]
        kv = dict(x.split("=", 1) for x in p[2:] if "=" in x)
        t = tags.setdefault(tag, {"n": 0, "pids": set(), "kv": collections.defaultdict(list)})
        t["n"] += 1; t["pids"].add(pid)
        for k, v in kv.items():
            t["kv"][k].append(v)
        procs[(tag, pid)].append(kv)
    print("== %s" % path)
    for tag, t in tags.items():
        print("%s: lines=%d processes=%d" % (tag, t["n"], len(t["pids"])))
        for k, vs in t["kv"].items():
            if set(vs) <= {"true", "false"}:
                print("    %s: true=%d false=%d" % (k, vs.count("true"), vs.count("false")))
            else:
                ns = [num(v) for v in vs]
                if all(x is not None for x in ns):
                    print("    %s: min=%g median=%g max=%g" % (k, min(ns), statistics.median(ns), max(ns)))
    if "map" in tags:
        per = [(pid, ev) for (tag, pid), ev in procs.items() if tag == "map"]
        by_n = collections.defaultdict(lambda: [0, 0])
        order = collections.Counter()
        for pid, ev in per:
            n = int(ev[0]["n_chunks"])
            caller = any(e["caller"] == "true" for e in ev)
            by_n[n][0] += 1; by_n[n][1] += caller
            if n >= 4:
                ev.sort(key=lambda e: int(e["i"]))
                addr = [int(e["addr"], 16) for e in ev]
                inc = all(a < b for a, b in zip(addr, addr[1:])); dec = all(a > b for a, b in zip(addr, addr[1:]))
                order["address ascending in chunk number" if inc else "address descending in chunk number" if dec else "neither"] += 1
        print("calling thread mapped >= 1 chunk, per n_chunks (processes with n chunks, of them with a caller map):")
        tot = [0, 0]
        for n in sorted(by_n):
            print("    n=%d: processes=%d caller_mapped=%d" % (n, by_n[n][0], by_n[n][1]))
            if n >= 8:
                tot[0] += by_n[n][0]; tot[1] += by_n[n][1]
        print("  n>=8 overall: processes=%d caller_mapped=%d" % tuple(tot))
        print("  relation of the chunk number to the address (processes with n>=4):", dict(order))
    if "reg" in tags:
        order = collections.Counter()
        for (tag, pid), ev in procs.items():
            if tag != "reg" or int(ev[0]["n_chunks"]) < 4:
                continue
            ev.sort(key=lambda e: int(e["k"]))
            addr = [int(e["addr"], 16) for e in ev]
            inc = all(a < b for a, b in zip(addr, addr[1:])); dec = all(a > b for a, b in zip(addr, addr[1:]))
            order["registered addresses ascending" if inc else "registered addresses descending" if dec else "neither"] += 1
        print("  order of the registered chunk addresses in the host_register calls (processes with n>=4):", dict(order))

for f in sys.argv[1:]:
    summarize(f)
