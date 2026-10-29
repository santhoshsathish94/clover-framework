#!/usr/bin/env python3
"""Summarises out/timing.log: per (workload, library) median/min/max of GB/s and process CPU seconds, ratio Rust/C++ of medians, within-library spread."""
import re, statistics as st, sys, collections
rows = collections.defaultdict(list)
bad = 0
for line in open('out/timing.log'):
    m = re.match(r'TIMING (\w+) lib=(\S+) jobs=(\d+) bytes=(\d+) wall_s=([\d.]+) gbps=([\d.]+) cpu_user_s=([\d.]+) cpu_sys_s=([\d.]+) warm_s=([\d.]+) verify_bad=(\d+)', line)
    if not m: continue
    t, lib = m.group(1), m.group(2)
    rows[(t, lib)].append(dict(wall=float(m.group(5)), gbps=float(m.group(6)), cpu=float(m.group(7)) + float(m.group(8)), user=float(m.group(7)), sys=float(m.group(8)), warm=float(m.group(9))))
    bad += int(m.group(10))
print(f'runs with verify_bad>0: {bad}')
def stats(v): return st.median(v), min(v), max(v), (max(v) - min(v)) / st.median(v) * 100
def cell(r, k): s = stats([x[k] for x in r]); return f'{s[0]:.3f} [{s[1]:.3f}..{s[2]:.3f}] spread {s[3]:.1f}%'
keys = sorted({k[0] for k in rows}, key=lambda s: (s.endswith('numa0'), s))
tags = sorted({k for k in rows})
print(f'{"cell":<14}{"lib":<14}{"n":<3}{"GB/s median [min..max]":<44}{"CPU s (user+sys) median [min..max]":<48}{"warm-up s median"}')
for k in sorted(rows):
    r = rows[k]; print(f'{k[0]:<14}{k[1]:<14}{len(r):<3}{cell(r, "gbps"):<44}{cell(r, "cpu"):<48}{st.median([x["warm"] for x in r]):.3f}')
print()
print('ratio Rust/C++ of medians (GB/s: >1 = Rust faster; CPU s: >1 = Rust uses more), and whether the difference exceeds the larger within-library spread:')
seen = set()
for (t, lib) in sorted(rows):
    if lib.startswith('cpp'):
        suffix = lib[3:]; o = ('rust' + suffix)
        if (t, o) not in rows: continue
        a, b = rows[(t, lib)], rows[(t, o)]
        for k, name in (('gbps', 'GB/s'), ('cpu', 'CPU s')):
            sa, sb = stats([x[k] for x in a]), stats([x[k] for x in b])
            ratio = sb[0] / sa[0]; diff = abs(sb[0] - sa[0]) / sa[0] * 100; spread = max(sa[3], sb[3])
            print(f'{t:<8}{suffix or "":<8}{name:<7} cpp {sa[0]:.3f}  rust {sb[0]:.3f}  ratio {ratio:.3f}  |diff| {diff:.1f}% vs larger spread {spread:.1f}% -> {"differs beyond spread" if diff > spread else "within spread"}')
