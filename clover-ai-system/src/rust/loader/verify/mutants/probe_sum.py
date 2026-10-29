#!/usr/bin/env python3
# Sums the probe files /dev/shm/ldprobe-<probe>.txt ("pid n tag"): per tag the number of processes that hit it and the hits (largest n per pid; exact up to 50, a lower bound above).
# usage: probe_sum.py PROBE...
import collections, sys
for p in sys.argv[1:]:
    try:
        lines = open("/dev/shm/ldprobe-%s.txt" % p).read().splitlines()
    except FileNotFoundError:
        print("%s: no probe file (the changed path never ran / the condition never held)" % p); continue
    best = collections.defaultdict(int)
    bad = 0
    for l in lines:
        try:
            pid, n, tag = l.split(" ", 2)
            best[(tag, pid)] = max(best[(tag, pid)], int(n))
        except ValueError:
            bad += 1   # two processes appended at the same moment (writeln! is not one write): the line is garbled, skipped
    if bad:
        print("%s: %d garbled lines skipped" % (p, bad))
    tags = collections.defaultdict(lambda: [0, 0])
    for (tag, pid), n in best.items():
        tags[tag][0] += 1; tags[tag][1] += n
    for tag, (np_, hits) in sorted(tags.items()):
        print("%s: %-72s processes=%d hits>=%d" % (p, tag, np_, hits))
