#!/usr/bin/env python3
"""Rebuild st_model.json, the five non-layer tensors the equation needs.

These five do not live in the trunk, so dump_eqidx.py takes their location
straight from the checkpoint. Nothing generated this file before; it was made
by hand, which meant eqidx.bin could not be rebuilt from the model alone.

Offsets are ABSOLUTE file offsets (8 + header_len + data_offsets[i]), which is
the form dump_eqidx.py writes into the index without further adjustment.

usage: dump_st_model.py <model_dir> <out>
"""
import json, glob, struct, sys, os

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("K3_MODEL")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("K3_STMODEL")
if not MODEL or not OUT:
    sys.exit("usage: dump_st_model.py <model_dir> <out>"
             "  (or set K3_MODEL and K3_STMODEL; see config.env)")

# Insertion order matters only so the output can be compared byte for byte
# against the hand-made original.
WANT = [
    "language_model.model.embed_tokens.weight",
    "language_model.lm_head.weight",
    "language_model.model.norm.weight",
    "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
]

found = {}
for sp in sorted(glob.glob(MODEL + "/*.safetensors")):
    with open(sp, "rb") as f:
        nh = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(nh))
    base = 8 + nh
    for name in WANT:
        m = hdr.get(name)
        if m is None:
            continue
        if name in found:
            sys.exit("tensor %s appears in more than one shard" % name)
        found[name] = [sp,
                       base + m["data_offsets"][0],
                       base + m["data_offsets"][1],
                       m["shape"],
                       m["dtype"]]

missing = [n for n in WANT if n not in found]
if missing:
    sys.exit("not found in %s: %s" % (MODEL, missing))

with open(OUT, "w") as o:
    json.dump({n: found[n] for n in WANT}, o)
print("wrote %s, %d tensors" % (OUT, len(WANT)))
