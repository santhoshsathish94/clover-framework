#!/usr/bin/env python3
"""Emit one row per (position, pod) describing what the attention produced.

Until this cycle the MLA pods recorded nothing between the normalize and the
residual add, so the two kinds of pod could not be compared at all. They can
now: locals 5011-5016 on an MLA pod are the counterpart of 5001-5009 on a KDA
pod. This writes both into the same columns so the stack can be read as one
thing rather than as two instrumented differently.

The two kinds do not have the same stages, and the columns say so:

  pre      the attention output before the gate
             KDA  local 5007, after the per-head normalize
             MLA  local 5015, the softmax-weighted sum, there is no normalize
  raw      KDA local 5006, the state readout before the per-head normalize
           empty on MLA, which has no such quantity
  gain     pre / raw, empty on MLA for the same reason
  gate     KDA 5008, MLA 5016, the same operation on both
  out      the attention output alone, written over the residual
             KDA  stage 9      MLA  stage 6
  input    local 0, the residual arriving at the pod

`out` and `input` are the comparable pair: `out / input` is how large the
attention's contribution is against what it was given.

The file index is the pod number, checked against this capture rather than
assumed: the files carrying MLA internals are exactly {3,7,..,91} and {92},
which is the MLA set, so no shift. 68 files carry KDA internals. Files 0 and
93 carry neither and are not counted here; what they are is not established.

usage: attention-census.py DUMP-PREFIX PROMPT-NAME OUT.tsv
"""
import glob
import os
import re
import struct
import sys

import numpy

MLA_PODS = set(range(3, 92, 4)) | {92}
WANT = {0, 6, 9, 5006, 5007, 5008, 5015, 5016}


def read(path):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            _, local, position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            if local not in WANT:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            yield local, position, numpy.frombuffer(body, dtype="<f4")


def main():
    prefix, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = []
    skipped = []
    for path in sorted(glob.glob(prefix + "*.bin")):
        found = re.search(r"(\d+)\.bin$", os.path.basename(path))
        if not found:
            continue
        pod = int(found.group(1))
        if pod < 0 or pod > 92:
            continue
        mla = pod in MLA_PODS
        seen = {}
        for local, position, values in read(path):
            seen.setdefault(position, {})[local] = float(numpy.linalg.norm(values))
        for position in sorted(seen):
            have = seen[position]
            pre = have.get(5015 if mla else 5007)
            gate = have.get(5016 if mla else 5008)
            attention_out = have.get(6 if mla else 9)
            arriving = have.get(0)
            if pre is None or attention_out is None or arriving is None:
                skipped.append(pod)
                break
            raw = have.get(5006)
            rows.append((prompt, position, pod, "M" if mla else "K",
                         pod // 4, pod % 4, arriving, raw, pre,
                         (pre / raw) if raw else None, gate, attention_out,
                         attention_out / arriving if arriving else None))

    rows.sort(key=lambda row: (row[1], row[2]))
    with open(out, "a") as handle:
        for row in rows:
            handle.write("\t".join("" if cell is None else
                                   ("%.6f" % cell if isinstance(cell, float) else str(cell))
                                   for cell in row) + "\n")
    sys.stderr.write("%s: %d rows, %d pods, skipped %s\n"
                     % (prompt, len(rows), len(set(r[2] for r in rows)), skipped))


if __name__ == "__main__":
    main()
