#!/usr/bin/env python3
"""Method D: keep only the first K routed experts and renormalise.

The router weights turn out to be close to uniform. Effective expert count is 10.5 to
15.7 of 16 and reaching 90 percent of the weight needs 13 to 15 of them, so cutting K
should be expensive. That is a prediction, and the point of this is to find out rather
than to assert it.

The weights of ranks at or beyond K are zeroed and the remainder renormalised, so the
expert contributes nothing. mix_experts still evaluates all sixteen, which means this
measures fidelity only and claims no saving — the same discipline as the attention
substitutions.

Off unless CLOVER_TOPK is set to a value between 1 and 15.
Idempotent; reports changed / already / absent.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]] or list(range(1, 93))

MARK = "CLOVER_TOPK"

ANCHOR = """    for (unsigned rank = 0; rank < 16; rank++) sequence->weights[rank] = scores[sequence->selected[rank]] * inverse;
    return 1;
}"""

PATCH = """    for (unsigned rank = 0; rank < 16; rank++) sequence->weights[rank] = scores[sequence->selected[rank]] * inverse;
    /* CLOVER_TOPK, experiment only: keep the first K ranks and renormalise the rest to
       zero, so the layer behaves as if it had routed to fewer experts. */
    {
        static int keep;
        if (!keep) {
            const char *value = getenv("CLOVER_TOPK");
            keep = value && *value ? atoi(value) : 16;
            if (keep < 1 || keep > 16) keep = 16;
        }
        if (keep < 16) {
            double kept = 0.0;
            for (int rank = 0; rank < keep; rank++) kept += sequence->weights[rank];
            float rescale = (float)(1.0 / (kept + 1e-20));
            for (int rank = 0; rank < 16; rank++)
                sequence->weights[rank] = rank < keep ? sequence->weights[rank] * rescale : 0.0f;
        }
    }
    return 1;
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
