#!/usr/bin/env python3
"""Record what happens inside MLA, so an M pod is as visible as a K pod.

Walking a KDA-KDA-KDA-MLA group showed nine internal values in each K and zero in the M:
stage 6 is one call that jumps to stage 20, so between the pod's input and its attention
residual there is nothing to look at. That is a hole in the instrument, not a property of
the model.

The KDA side already records Q, K, V, beta, decay, state magnitude, attention raw,
attention normalized and gate. These are the MLA counterparts, placed at the points where
the named buffers already hold a finished value:

  5011  query latent, after projection 20 and its normalize     1536
  5012  query, after projection 22                             18432   96 heads x 192
  5013  kv latent, after projection 23 and its normalize         576   512 + 64 positional
  5014  expanded, after projection 25                          24576   96 heads x (128+128)
  5015  attention, the softmax-weighted sum over positions     12288
  5016  gate, attention * sigmoid(gate projection)             12288

Not captured here: the per-head softmax weights over positions. They live in a thread
scratch buffer inside the parallel loop and are freed, so recording them needs a new
buffer rather than a write call. One step at a time.

Worth noting while placing these: the KDA path normalizes attention per head before
gating and the MLA path does not. There is no MLA counterpart to local 5007.

`transformer_mla` is inside `#if TRANSFORMER_MLA`, so this text is written into all 92
pods and compiles only on the 24 that are MLA. Behind clover_vectors_active. Idempotent.
"""
import glob
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
MARK = "CLOVER_MLA_INTERNALS"

QUERY_ANCHOR = """    transformer_project(transformer, 22, sequence->mla_query_latent, sequence->mla_query);
    transformer_project(transformer, 23, sequence->normalized, sequence->mla_kv_latent);"""

QUERY_PATCH = """    transformer_project(transformer, 22, sequence->mla_query_latent, sequence->mla_query);
    /* CLOVER_MLA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER)) {
        unsigned long at = (unsigned long)sequence->positions;
        clover_vectors_write(TRANSFORMER_LAYER, 5011, at, sequence->mla_query_latent, 1536);
        clover_vectors_write(TRANSFORMER_LAYER, 5012, at, sequence->mla_query, 18432);
    }
    transformer_project(transformer, 23, sequence->normalized, sequence->mla_kv_latent);"""

EXPAND_ANCHOR = """    transformer_project(transformer, 25, sequence->mla_kv_latent, sequence->mla_expanded);
    float *const cache = sequence->mla_cache;"""

EXPAND_PATCH = """    transformer_project(transformer, 25, sequence->mla_kv_latent, sequence->mla_expanded);
    /* CLOVER_MLA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER)) {
        unsigned long at = (unsigned long)sequence->positions;
        clover_vectors_write(TRANSFORMER_LAYER, 5013, at, sequence->mla_kv_latent, 576);
        clover_vectors_write(TRANSFORMER_LAYER, 5014, at, sequence->mla_expanded, 24576);
    }
    float *const cache = sequence->mla_cache;"""

# free(scratch) appears only in transformer_mla, so this cannot match the KDA path that
# carries the same two gate lines.
GATE_ANCHOR = """    free(scratch);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);"""

GATE_PATCH = """    free(scratch);
    /* CLOVER_MLA_INTERNALS: there is no per-head normalize here. The KDA path runs one
       between the raw attention and the gate; this path goes straight to the gate. */
    if (clover_vectors_active(TRANSFORMER_LAYER))
        clover_vectors_write(TRANSFORMER_LAYER, 5015, (unsigned long)sequence->positions,
            sequence->attention, TRANSFORMER_ROWS);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    /* CLOVER_MLA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER))
        clover_vectors_write(TRANSFORMER_LAYER, 5016, (unsigned long)sequence->positions,
            sequence->gate, TRANSFORMER_ROWS);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);"""


def main():
    patched = skipped = failed = 0
    for path in sorted(glob.glob(str(ROOT / "transformer-*/transformer-*.c"))):
        text = open(path, "r", newline="").read()
        if MARK in text:
            skipped += 1
            continue
        hurt = False
        for name, anchor, patch in (("query", QUERY_ANCHOR, QUERY_PATCH),
                                    ("expand", EXPAND_ANCHOR, EXPAND_PATCH),
                                    ("gate", GATE_ANCHOR, GATE_PATCH)):
            for a, p in ((anchor, patch),
                         (anchor.replace("\n", "\r\n"), patch.replace("\n", "\r\n"))):
                if text.count(a) == 1:
                    text = text.replace(a, p)
                    break
            else:
                print("%s: %s anchor found %d times, expected 1"
                      % (pathlib.Path(path).name, name, text.count(anchor)))
                hurt = True
                break
        if hurt:
            failed += 1
            continue
        with open(path, "w", newline="") as handle:
            handle.write(text)
        patched += 1
    print("patched %d, already patched %d, failed %d" % (patched, skipped, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
