#!/usr/bin/env python3
"""Trace the two MoE paths separately: the shared full-width path and the routed experts.

Direction from the creator, after I had spent a cycle decomposing layer output by
position: the constant part of a layer is the trunk and the varying part is the selected
routed experts. Position is a lens on the symptom; this is the mechanism. The report
agrees in §2.3 — shared experts keep a full-width path for common transformations while
the routed ones operate per token in a compact latent space.

Stage 116 is `residual += routed + shared_output`, so both are live and separable there.
Recording them individually makes the claim testable instead of plausible.

  5101  sequence->routed         the 16 selected experts, latent 3584 projected to 7168
  5102  sequence->shared_output  the shared full-width path

Every pod has an MoE, MLA and KDA alike, so this patches all 92. Behind
clover_vectors_active, so nothing is computed with CLOVER_VECTORS unset.
Idempotent; reports changed / already / absent.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]] or list(range(1, 93))

MARK = "CLOVER_MOE_SPLIT"

ANCHOR = """    if (stage == 116) {
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->residual[coordinate] +
                (sequence->routed[coordinate] + sequence->shared_output[coordinate]);"""

PATCH = """    if (stage == 116) {
        /* CLOVER_MOE_SPLIT: the two paths, before they are summed and become
           indistinguishable in the residual. */
        if (clover_vectors_active(TRANSFORMER_LAYER)) {
            unsigned long at = (unsigned long)sequence->positions;
            clover_vectors_write(TRANSFORMER_LAYER, 5101, at, sequence->routed, TRANSFORMER_WIDTH);
            clover_vectors_write(TRANSFORMER_LAYER, 5102, at, sequence->shared_output, TRANSFORMER_WIDTH);
        }
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->residual[coordinate] +
                (sequence->routed[coordinate] + sequence->shared_output[coordinate]);"""

failures = 0
for layer in LAYERS:
    path = ROOT / f"transformer-{layer}" / f"transformer-{layer}.c"
    if not path.exists():
        print(f"layer {layer}: ABSENT")
        failures += 1
        continue
    text = open(path, "r", newline="").read()
    if MARK in text:
        print(f"layer {layer}: already patched")
        continue
    for a, p in ((ANCHOR, PATCH), (ANCHOR.replace("\n", "\r\n"), PATCH.replace("\n", "\r\n"))):
        if text.count(a) == 1:
            with open(path, "w", newline="") as handle:
                handle.write(text.replace(a, p))
            print(f"layer {layer}: patched")
            break
    else:
        print(f"layer {layer}: anchor found {text.count(ANCHOR)} times, expected 1")
        failures += 1

sys.exit(1 if failures else 0)
