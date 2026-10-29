import os, collections
L = [l.split() for l in open(os.path.expanduser("~/k3gpu/results/p1d.trace"))]
print("lines", len(L))
u = [len(l) - 2 for l in L]
print("uniq max", max(u), "min", min(u), "mean", sum(u) / len(u))
lay = collections.Counter(int(l[0]) for l in L)
print("layers", min(lay), max(lay), len(lay), "calls/layer", set(lay.values()))
print("ntok", sorted(set(int(l[1]) for l in L))[:20])
print("layer seq", [int(l[0]) for l in L[:4]], [int(l[0]) for l in L[90:96]])
print("sum uniq", sum(u))
print("all tokens have counts", all(":" in t for l in L for t in l[2:]))
print("lines w/o experts", [i for i, l in enumerate(L) if len(l) < 3][:5])
