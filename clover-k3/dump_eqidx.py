#!/usr/bin/env python3
"""Flatten every index the equation needs into one binary file for the C build.

Parsing trunk.json, st_model.json and 96 safetensors headers is not part of
the equation, so it is done once here and handed to C as fixed-width records.

output: $K3_INDEX
"""
import json, struct, glob, sys, os


def need(name):
    v = os.environ.get(name)
    if not v:
        sys.exit("%s is not set - source config.env, or run ./build.sh" % name)
    return v


MODEL = need("K3_MODEL")
TRUNKJSON = need("K3_TRUNKJSON")
STMODEL = need("K3_STMODEL")
OUT = need("K3_INDEX")
PREF = "language_model.model.layers."
NL = 93

SLOTS = [
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.b_proj.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "dt_bias",
    "o_norm.weight",
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "block_sparse_moe.gate.weight",
    "e_score_correction_bias",
    "routed_expert_down_proj.weight",
    "routed_expert_up_proj.weight",
    "routed_expert_norm.weight",
    "shared_experts.gate_proj.weight",
    "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
]
DT = {"F32": 0, "BF16": 1, "I8R": 2}

MODEL_TENSORS = [
    "language_model.model.embed_tokens.weight",
    "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
    "language_model.model.norm.weight",
    "language_model.lm_head.weight",
]

TJ = json.load(open(TRUNKJSON))
SM = json.load(open(STMODEL))

files = []
fid = {}


def file_id(p):
    if p not in fid:
        fid[p] = len(files); files.append(p)
    return fid[p]


# ---- per layer slots
lay_recs = []
missing = 0
for L in range(NL):
    tens = TJ["layers"][L]["tensors"]
    foff = TJ["layers"][L]["file_off"]
    for s in SLOTS:
        hits = [(n, t) for n, t in tens.items() if n.endswith(s)]
        if len(hits) == 0:
            lay_recs.append((0, 0, 0, 0, 0, 0))
            missing += 1
            continue
        if len(hits) > 1:
            sys.exit("ambiguous slot %s at layer %d: %s" % (s, L, [h[0] for h in hits]))
        n, t = hits[0]
        sh = t.get("shape", [0])
        d0 = sh[0] if len(sh) > 0 else 0
        d1 = sh[1] if len(sh) > 1 else 0
        lay_recs.append((1, DT[t["dtype"]], foff + t["off"], t["nbytes"], d0, d1))

# ---- model level
mod_recs = []
for k in MODEL_TENSORS:
    p, a, b, sh, dt = SM[k]
    d0 = sh[0] if len(sh) > 0 else 0
    d1 = sh[1] if len(sh) > 1 else 0
    mod_recs.append((file_id(p), a, b - a, d0, d1, DT[dt]))

# ---- experts, one pass over every shard
WHICH = {"w1": 0, "w3": 1, "w2": 2}
KIND = {"weight_packed": 0, "weight_scale": 1}
exp_recs = []
for sp in sorted(glob.glob(MODEL + "/*.safetensors")):
    with open(sp, "rb") as f:
        nh = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(nh))
    base = 8 + nh
    for nm, m in hdr.items():
        if ".block_sparse_moe.experts." not in nm:
            continue
        rest = nm[len(PREF):]
        L = int(rest[:rest.index(".")])
        parts = nm.split(".")
        e = int(parts[parts.index("experts") + 1])
        wn = parts[parts.index("experts") + 2]
        kd = parts[-1]
        if wn not in WHICH or kd not in KIND:
            continue
        sh = m["shape"]
        exp_recs.append((L, e, WHICH[wn], KIND[kd], file_id(sp),
                         base + m["data_offsets"][0],
                         m["data_offsets"][1] - m["data_offsets"][0],
                         sh[0], sh[1] if len(sh) > 1 else 0))
exp_recs.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

with open(OUT, "wb") as o:
    o.write(b"K3EQ")
    o.write(struct.pack("<ii", NL, len(SLOTS)))
    for rec in lay_recs:
        o.write(struct.pack("<iiqqii", *rec))
    o.write(struct.pack("<i", len(MODEL_TENSORS)))
    for rec in mod_recs:
        o.write(struct.pack("<iqqiii", *rec))
    o.write(struct.pack("<i", len(files)))
    for p in files:
        pb = p.encode()
        o.write(struct.pack("<i", len(pb))); o.write(pb)
    o.write(struct.pack("<i", len(exp_recs)))
    for rec in exp_recs:
        o.write(struct.pack("<iiiiiqqii", *rec))

import os
print("slots per layer      : %d   (%d empty across %d layers)" % (len(SLOTS), missing, NL))
print("model tensors        : %d" % len(mod_recs))
print("safetensors files    : %d" % len(files))
print("expert records       : %d" % len(exp_recs))
print("wrote %s  %.1f MB" % (OUT, os.path.getsize(OUT) / 1e6))
