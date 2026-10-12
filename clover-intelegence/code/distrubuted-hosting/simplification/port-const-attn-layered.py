#!/usr/bin/env python3
"""Method C: per-layer constant substitution for the KDA attention output.

Method B replaced one layer's attention with a cached vector and nothing moved. That
layer was chosen because its attention was measured to be near-constant. The single
token prompt then showed the substituted vector could be 46 percent too large and still
change the answer by 0.000001, which suggests the result may not depend on the constancy
that motivated it. So: do it to every KDA layer and watch.

CLOVER_CONST_ATTN names a prefix; pod N loads "<prefix>.N.f32" and substitutes it if the
file exists and holds TRANSFORMER_ROWS float32. A pod with no file leaves itself alone,
so any subset of layers can be enabled by choosing which files to write.

Supersedes the single-file form used for Method B; restore pods from the backup before
running this. Idempotent; reports changed / already / absent.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]]
if not LAYERS:
    sys.exit("give layer numbers")

MARK = "CLOVER_CONST_ATTN"

ANCHOR = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    /* CLOVER_KDA_INTERNALS: the per-head normalised attention, before gating. */"""

PATCH = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    /* CLOVER_CONST_ATTN, experiment only: substitute a cached vector for the attention
       this layer just computed. Per layer, so any subset can be enabled by choosing
       which files exist. Loaded once; absent or short means this layer is untouched. */
    {
        static float constant[TRANSFORMER_ROWS];
        static int constant_state;
        if (!constant_state) {
            const char *prefix = getenv("CLOVER_CONST_ATTN");
            char named[4096];
            FILE *file = NULL;
            constant_state = -1;
            if (prefix && *prefix &&
                snprintf(named, sizeof named, "%s.%d.f32", prefix, TRANSFORMER_LAYER) < (int)sizeof named)
                file = fopen(named, "rb");
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
