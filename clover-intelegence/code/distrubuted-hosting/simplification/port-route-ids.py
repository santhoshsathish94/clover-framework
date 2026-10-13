#!/usr/bin/env python3
"""Record which experts each token selects.

The output-variance decomposition missed the point. "Constant" means the trunk applies
the same weights to every token, while the routed experts are *chosen* per token; the
variance of an output vector conflates that with variance of the input. What was never
recorded is the choice itself.

sequence->selected holds the sixteen expert ids after stage 24. Written as float32,
which represents ids up to 895 exactly.

  5103  the 16 selected expert ids, in rank order

Behind clover_vectors_active. Idempotent; reports changed / already / absent.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]] or list(range(1, 93))

MARK = "CLOVER_ROUTE_IDS"

ANCHOR = """    if (stage == 24) { if (!transformer_route(transformer, sequence)) return 0; sequence->stage++; return 1; }"""

PATCH = """    if (stage == 24) {
        if (!transformer_route(transformer, sequence)) return 0;
        /* CLOVER_ROUTE_IDS: which experts this token chose, not what they produced. */
        if (clover_vectors_active(TRANSFORMER_LAYER)) {
            float chosen[16];
            for (unsigned rank = 0; rank < 16; rank++) chosen[rank] = (float)sequence->selected[rank];
            clover_vectors_write(TRANSFORMER_LAYER, 5103, (unsigned long)sequence->positions, chosen, 16);
        }
        sequence->stage++; return 1;
    }"""

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
