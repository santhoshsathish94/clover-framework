#!/usr/bin/env python3
"""Method B: let a pod substitute a cached constant for its computed attention output.

Pod 12's attention output, after its per-head normalisation, is close to a fixed
direction: cosine 0.9964 to 0.9998 between positions within a prompt, 0.9949 to 0.9998
across prompts as unlike as biblical English and nonsense, and out to position 15 with
norms clamped between 3.81 and 3.90. If that is really a constant, replacing it with one
cached vector should not move the answer, and then the whole KDA recurrence at that
layer, 96 heads of 128x128 state, is redundant.

Fidelity is tested before any compute is removed: the substitution happens after the
work is done, so a failure here is a failure of the premise rather than of a rewrite.

Off unless CLOVER_CONST_ATTN names a file of TRANSFORMER_ROWS float32.
Idempotent; reports changed / already / absent.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]] or [12]

MARK = "CLOVER_CONST_ATTN"

ANCHOR = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    /* CLOVER_KDA_INTERNALS: the per-head normalised attention, before gating. */"""

PATCH = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    /* CLOVER_CONST_ATTN, experiment only: substitute a cached vector for the attention
       this layer just computed. Loaded once; absent or unreadable means no change. */
    {
        static float constant[TRANSFORMER_ROWS];
        static int constant_state;
        if (!constant_state) {
            const char *path = getenv("CLOVER_CONST_ATTN");
            FILE *file = path && *path ? fopen(path, "rb") : NULL;
            constant_state = -1;
            if (file) {
                if (fread(constant, sizeof(float), TRANSFORMER_ROWS, file) == TRANSFORMER_ROWS)
                    constant_state = 1;
                fclose(file);
            }
        }
        if (constant_state > 0) memcpy(sequence->attention, constant, sizeof constant);
    }
    /* CLOVER_KDA_INTERNALS: the per-head normalised attention, before gating. */"""

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
        print(f"layer {layer}: anchor not found exactly once")
        failures += 1

sys.exit(1 if failures else 0)
