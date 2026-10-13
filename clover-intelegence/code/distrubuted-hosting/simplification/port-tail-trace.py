#!/usr/bin/env python3
"""Trace the tail, owner 93: the final normalisation and the head.

The tail already records stage timing and a residual hash through clover_trace_stage.
A hash shows that state changed and nothing about what it became, which is exactly the
gap that made the pod walk need vectors. So this adds the vectors.

What the tail does, read off the source rather than assumed: nine sources go in, the
eight block snapshots and the running residual. normalization_score_sources scores all
nine, softmax turns the scores into weights, normalization_aggregate folds them into
one vector, normalization_apply_rms scales it, and server_output_project takes the
argmax of that vector against every row of the vocabulary. The same fold shape the
pods run twelve times inside a block, run once over the whole model.

  5301  residual arriving, source 8
  5302  the nine scores
  5303  the nine weights
  5304  result after the fold
  5305  result after RMS, the vector the head reads
  5310..5317  the eight incoming block snapshots, sources 0..7

Owner 93. Behind clover_vectors_active, so nothing happens with CLOVER_VECTORS unset.
Idempotent.
"""
import pathlib
import sys

PATH = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting"
                    "/tansformers/transformer-93/transformer-93.c")
MARK = "CLOVER_TAIL_TRACE"

# transformer_93_step is static and takes the model and the sequence; the snapshots and
# the residual arrive as arguments, so every local below is already in scope.
SOURCES_ANCHOR = """    if (stage == 2) {
        for (unsigned source = 0; source < NORMALIZATION_SNAPSHOTS; source++)
            sequence->sources[source] = snapshots + source * NORMALIZATION_WIDTH;
        sequence->stage++; return 1;
    }
    if (stage == 3) { sequence->sources[NORMALIZATION_SNAPSHOTS] = residual; sequence->stage++; return 1; }
"""

SOURCES_PATCH = """    if (stage == 2) {
        for (unsigned source = 0; source < NORMALIZATION_SNAPSHOTS; source++)
            sequence->sources[source] = snapshots + source * NORMALIZATION_WIDTH;
        /* CLOVER_TAIL_TRACE */
        if (clover_vectors_active(93))
            for (unsigned source = 0; source < NORMALIZATION_SNAPSHOTS; source++)
                clover_vectors_write(93, 5310 + (int)source, (unsigned long)sequence->positions,
                    sequence->sources[source], NORMALIZATION_WIDTH);
        sequence->stage++; return 1;
    }
    if (stage == 3) {
        sequence->sources[NORMALIZATION_SNAPSHOTS] = residual;
        /* CLOVER_TAIL_TRACE */
        if (clover_vectors_active(93))
            clover_vectors_write(93, 5301, (unsigned long)sequence->positions,
                residual, NORMALIZATION_WIDTH);
        sequence->stage++; return 1;
    }
"""

FOLD_ANCHOR = """    if (stage == 4) {
        if (!normalization_score_sources(model->normalization, sequence->sources, sequence->scores)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 5) {
        if (!normalization_softmax(sequence->scores, sequence->weights)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 6) {
        normalization_aggregate(sequence->sources, sequence->weights, sequence->result);
        sequence->stage++; return 1;
    }
"""

FOLD_PATCH = """    if (stage == 4) {
        if (!normalization_score_sources(model->normalization, sequence->sources, sequence->scores)) return 0;
        /* CLOVER_TAIL_TRACE */
        if (clover_vectors_active(93))
            clover_vectors_write(93, 5302, (unsigned long)sequence->positions,
                sequence->scores, NORMALIZATION_SOURCES);
        sequence->stage++; return 1;
    }
    if (stage == 5) {
        if (!normalization_softmax(sequence->scores, sequence->weights)) return 0;
        /* CLOVER_TAIL_TRACE */
        if (clover_vectors_active(93))
            clover_vectors_write(93, 5303, (unsigned long)sequence->positions,
                sequence->weights, NORMALIZATION_SOURCES);
        sequence->stage++; return 1;
    }
    if (stage == 6) {
        normalization_aggregate(sequence->sources, sequence->weights, sequence->result);
        /* CLOVER_TAIL_TRACE */
        if (clover_vectors_active(93))
            clover_vectors_write(93, 5304, (unsigned long)sequence->positions,
                sequence->result, NORMALIZATION_WIDTH);
        sequence->stage++; return 1;
    }
"""

RMS_ANCHOR = """    if (stage == 8) { normalization_apply_rms(model->normalization, sequence->result); sequence->stage++; return 1; }
"""

RMS_PATCH = """    if (stage == 8) {
        normalization_apply_rms(model->normalization, sequence->result);
        /* CLOVER_TAIL_TRACE: the vector the head reads. */
        if (clover_vectors_active(93))
            clover_vectors_write(93, 5305, (unsigned long)sequence->positions,
                sequence->result, NORMALIZATION_WIDTH);
        sequence->stage++; return 1;
    }
"""

OPEN_ANCHOR = """    sequence->stage = 1;
    clover_trace_open();
"""

OPEN_PATCH = """    sequence->stage = 1;
    clover_trace_open();
    clover_vectors_open(93); /* CLOVER_TAIL_TRACE */
"""

text = open(PATH, "r", newline="").read()
if MARK in text:
    print("already patched")
    sys.exit(0)

failures = 0
for name, anchor, patch in (("sources", SOURCES_ANCHOR, SOURCES_PATCH),
                            ("fold", FOLD_ANCHOR, FOLD_PATCH),
                            ("rms", RMS_ANCHOR, RMS_PATCH),
                            ("open", OPEN_ANCHOR, OPEN_PATCH)):
    for a, p in ((anchor, patch), (anchor.replace("\n", "\r\n"), patch.replace("\n", "\r\n"))):
        if text.count(a) == 1:
            text = text.replace(a, p)
            print(f"{name}: patched")
            break
    else:
        print(f"{name}: anchor found {text.count(anchor)} times, expected 1")
        failures += 1

if not failures:
    with open(PATH, "w", newline="") as handle:
        handle.write(text)
sys.exit(1 if failures else 0)
