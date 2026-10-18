#!/usr/bin/env python3
"""Reads the file PF_ROUTES wrote (prompt-pass expert counts per layer, then per decode step and layer the 16 experts chosen and
32 guessed from the layer input) and reports how much of the per-token expert reading a cache or a guess could avoid.
usage: route-analysis.py routes.bin"""
import struct, sys
from collections import OrderedDict

data = open(sys.argv[1], "rb").read()
decode, layers = struct.unpack_from("<II", data, 0)
off = 8
hits = [list(struct.unpack_from("<896Q", data, off + 8 * 896 * l)) for l in range(layers)]
off += 8 * 896 * layers
rec = struct.unpack_from("<%dH" % (decode * layers * 48), data, off)

def chosen(step, layer): b = (step * layers + layer) * 48; return rec[b:b + 16]
def guessed(step, layer): b = (step * layers + layer) * 48; return rec[b + 16:b + 48]

print("%d decode steps, %d layers (layer index 0 = layer 1)" % (decode, layers))
total = decode * layers * 16

prev = sum(len(set(chosen(s, l)) & set(chosen(s - 1, l))) for s in range(1, decode) for l in range(layers))
print("same expert chosen at the previous step: %.1f%% of the 16 per layer" % (100.0 * prev / ((decode - 1) * layers * 16)))
for k in (16, 32):
    g = sum(len(set(chosen(s, l)) & set(guessed(s, l)[:k])) for s in range(decode) for l in range(layers))
    print("real top-16 inside the %d best guessed from the layer input: %.1f%%" % (k, 100.0 * g / total))

print("cache of C experts per layer (17.5 MB each; 92 layers x C x 17.5 MB in RAM): share of the 16 chosen experts already in it")
print("   C   RAM GB   LRU from empty   fixed top-C of the prompt pass   LRU started from that")
for c in (4, 8, 16, 32, 64, 128):
    lru_h = fix_h = mix_h = 0
    for l in range(layers):
        top = sorted(range(896), key=lambda e: -hits[l][e])[:c]
        fixed = set(top)
        lru = OrderedDict()
        mix = OrderedDict((e, 1) for e in top)
        for s in range(decode):
            ch = chosen(s, l)
            for e in ch:
                lru_h += e in lru
                fix_h += e in fixed
                mix_h += e in mix
            for cache in (lru, mix):
                for e in ch:
                    if e in cache: cache.move_to_end(e)
                    else:
                        cache[e] = 1
                        if len(cache) > c: cache.popitem(last=False)
    print("%4d %8.1f %14.1f%% %26.1f%% %22.1f%%" % (c, layers * c * 17.547264 / 1000, 100.0 * lru_h / total, 100.0 * fix_h / total, 100.0 * mix_h / total))

# distinct experts per layer over all steps, and how concentrated the prompt-pass use was
dist = [len(set(e for s in range(decode) for e in chosen(s, l))) for l in range(layers)]
print("distinct experts chosen per layer over the %d decode steps: mean %.1f of 896 (max %d)" % (decode, sum(dist) / layers, max(dist)))
share = []
for l in range(layers):
    h = sorted(hits[l], reverse=True); t = sum(h)
    share.append(sum(h[:90]) / t if t else 0)
print("prompt pass: the 90 most used experts (10%%) took %.1f%% of the choices on average" % (100.0 * sum(share) / layers))
