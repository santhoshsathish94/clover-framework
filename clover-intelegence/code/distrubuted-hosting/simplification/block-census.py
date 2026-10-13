#!/usr/bin/env python3
"""Emit one row per (position, pod) from a capture, for comparison across prompts.

The single-prompt block fold produced four claims about the 12-pod unit. They are all
falsifiable and none of them has been tried on a second prompt. This writes the raw
quantities as a TSV so a separate script can test them without re-reading 1.6 GB.

Only the locals needed are decoded; the rest are seeked past. NumPy because pure Python
over 7168 floats per record is minutes per prompt.

columns: prompt position block offset pod attn out_norm rel_attn rel_fold rel_moe
         rel_input spread ratio

  rel_*     distance moved at that stage, relative to the state before it
  attn      M for MLA, K for KDA
  spread    participation ratio of the 16 router weights, 1 = one expert holds all
  ratio     |routed| / |shared|

usage: block-census.py DUMP-PREFIX PROMPT-NAME OUT.tsv
"""
import glob
import struct
import sys

import numpy

BLOCK = 12
MLA_PODS = set(range(3, 92, 4)) | {92}
WANT = {0, 6, 9, 20, 116, 119, 3000, 5101, 5102}


def read(path):
    """Yield (local, position, numpy array) for the locals in WANT."""
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


def relative(before, after):
    base = float(numpy.linalg.norm(before))
    return float(numpy.linalg.norm(after - before)) / base if base else float("nan")


def spread(weights):
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    share = weights / total
    squares = float((share * share).sum())
    return 1.0 / squares if squares > 0 else float("nan")


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip())
        return 1
    prefix, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}

    with open(out, "a") as sink:
        for pod in range(1, 93):
            if pod not in files:
                continue
            frames = {}
            for local, position, values in read(files[pod]):
                frames.setdefault(position, {})[local] = values
            for position in sorted(frames):
                frame = frames[position]
                if 0 not in frame or 119 not in frame:
                    continue
                incoming = frame[0]
                attention = frame.get(6 if pod in MLA_PODS else 9)
                if attention is None:
                    continue
                # Stage 2 is a memcpy of the input, so the state entering attention is
                # the input itself; measuring from `incoming` keeps the leftover at
                # local 1 out of the chain.
                rel_attn = relative(incoming, attention)
                rel_fold = relative(attention, frame[20]) if 20 in frame else float("nan")
                before_moe = frame.get(20, attention)
                rel_moe = relative(before_moe, frame[116]) if 116 in frame else float("nan")
                sink.write("%s\t%d\t%d\t%d\t%d\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f\t%.6f\n" % (
                    prompt, position, (pod - 1) // BLOCK + 1, (pod - 1) % BLOCK + 1, pod,
                    "M" if pod in MLA_PODS else "K",
                    float(numpy.linalg.norm(frame[119])),
                    rel_attn, rel_fold, rel_moe,
                    float(numpy.linalg.norm(incoming)),
                    spread(frame[3000]) if 3000 in frame else float("nan"),
                    (float(numpy.linalg.norm(frame[5101])) /
                     float(numpy.linalg.norm(frame[5102])))
                    if 5101 in frame and 5102 in frame
                    and float(numpy.linalg.norm(frame[5102])) else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
