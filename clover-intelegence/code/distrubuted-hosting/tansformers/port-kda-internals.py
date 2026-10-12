#!/usr/bin/env python3
"""Add gated tracing of the KDA internals to selected pods.

Pod 12's attention output is close to a fixed direction regardless of position or
prompt, and it is the only one of ninety-two that does this. Nothing in the tech report
or the vllm reference implementation distinguishes it from the other block-write layers,
so the cause has to be in that layer's learned weights. The residual trace cannot show
it: it records what the attention produced, not the quantities of Eq. 1 that produced it.

This records query, key, value, the write strength beta, the channel-wise decay, the
attention output before and after its per-head norm, the gated value, and the per-head
Frobenius norm of the recurrent state. Everything sits behind clover_vectors_active, so
with CLOVER_VECTORS unset not a single float is computed and the arithmetic is untouched.

Applied to a control layer as well as the suspect one, because "layer 12 looks odd" means
nothing without a layer that does not.

Run on the host; idempotent; reports changed / already / absent per layer.
"""
import pathlib
import sys

ROOT = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers")
LAYERS = [int(a) for a in sys.argv[1:]] or [12, 24]

MARK = "CLOVER_KDA_INTERNALS"

QKV_ANCHOR = """    transformer_l2_heads(sequence->qkv[0]);
    transformer_l2_heads(sequence->qkv[1]);
}"""

QKV_PATCH = """    transformer_l2_heads(sequence->qkv[0]);
    transformer_l2_heads(sequence->qkv[1]);
    /* CLOVER_KDA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER)) {
        unsigned long at = (unsigned long)sequence->positions;
        clover_vectors_write(TRANSFORMER_LAYER, 5001, at, sequence->qkv[0], TRANSFORMER_ROWS);
        clover_vectors_write(TRANSFORMER_LAYER, 5002, at, sequence->qkv[1], TRANSFORMER_ROWS);
        clover_vectors_write(TRANSFORMER_LAYER, 5003, at, sequence->qkv[2], TRANSFORMER_ROWS);
    }
}"""

DECAY_ANCHOR = """            sequence->decay[index] = expf(-5.0f * transformer_sigmoid(value));
        }
    }
}"""

DECAY_PATCH = """            sequence->decay[index] = expf(-5.0f * transformer_sigmoid(value));
        }
    }
    /* CLOVER_KDA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER)) {
        unsigned long at = (unsigned long)sequence->positions;
        clover_vectors_write(TRANSFORMER_LAYER, 5004, at, sequence->beta, TRANSFORMER_HEADS);
        clover_vectors_write(TRANSFORMER_LAYER, 5005, at, sequence->decay, TRANSFORMER_ROWS);
    }
}"""

STATE_ANCHOR = """                output[column] = output[column] + state * scaled_query;
            }
        }
    }
}"""

STATE_PATCH = """                output[column] = output[column] + state * scaled_query;
            }
        }
    }
    /* CLOVER_KDA_INTERNALS: one Frobenius norm per head, so the recurrent state can be
       watched without dumping 96 x 128 x 128 floats per position. Stack local, because
       several requests share this library at once. */
    if (clover_vectors_active(TRANSFORMER_LAYER)) {
        float magnitude[TRANSFORMER_HEADS];
        unsigned long at = (unsigned long)sequence->positions;
        for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
            double total = 0.0;
            for (unsigned row = 0; row < TRANSFORMER_HEAD; row++)
                for (unsigned column = 0; column < TRANSFORMER_HEAD; column++)
                    total += (double)sequence->recurrent[head][row][column] *
                        sequence->recurrent[head][row][column];
            magnitude[head] = (float)sqrt(total);
        }
        clover_vectors_write(TRANSFORMER_LAYER, 5009, at, magnitude, TRANSFORMER_HEADS);
        clover_vectors_write(TRANSFORMER_LAYER, 5006, at, sequence->attention, TRANSFORMER_ROWS);
    }
}"""

# The MLA path, compiled out but still present in every pod source, ends with the same
# three lines. Anchor on the per-head normalise above them, which only the KDA path has.
OUT_ANCHOR = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);"""

OUT_PATCH = """        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    /* CLOVER_KDA_INTERNALS: the per-head normalised attention, before gating. */
    if (clover_vectors_active(TRANSFORMER_LAYER))
        clover_vectors_write(TRANSFORMER_LAYER, 5007, (unsigned long)sequence->positions,
            sequence->attention, TRANSFORMER_ROWS);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    /* CLOVER_KDA_INTERNALS */
    if (clover_vectors_active(TRANSFORMER_LAYER))
        clover_vectors_write(TRANSFORMER_LAYER, 5008, (unsigned long)sequence->positions,
            sequence->gate, TRANSFORMER_ROWS);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);"""

PAIRS = [
    ("qkv", QKV_ANCHOR, QKV_PATCH),
    ("decay", DECAY_ANCHOR, DECAY_PATCH),
    ("state", STATE_ANCHOR, STATE_PATCH),
    ("output", OUT_ANCHOR, OUT_PATCH),
]

failures = 0
for layer in LAYERS:
    path = ROOT / f"transformer-{layer}" / f"transformer-{layer}.c"
    if not path.exists():
        print(f"layer {layer}: ABSENT {path}")
        failures += 1
        continue
    text = open(path, "r", newline="").read()
    if MARK in text:
        print(f"layer {layer}: already patched")
        continue
    applied = []
    for name, anchor, patch in PAIRS:
        # transformer-1 is known to carry mixed line endings; try both forms rather
        # than inferring one from the file as a whole.
        for a, p in ((anchor, patch), (anchor.replace("\n", "\r\n"), patch.replace("\n", "\r\n"))):
            if text.count(a) == 1:
                text = text.replace(a, p)
                applied.append(name)
                break
        else:
            print(f"layer {layer}: anchor {name!r} not found exactly once")
            failures += 1
    if len(applied) == len(PAIRS):
        with open(path, "w", newline="") as handle:
            handle.write(text)
        print(f"layer {layer}: patched ({', '.join(applied)})")

sys.exit(1 if failures else 0)
