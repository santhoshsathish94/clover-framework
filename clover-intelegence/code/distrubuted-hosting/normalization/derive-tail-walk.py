#!/usr/bin/env python3
"""Derive normalization/tail-walk.c from the real gate test-reference.c.

The gate already builds the nine real sources (eight snapshots plus the final
residual) at layer 92 and checks the engine against the reference. This adds
observation of the intermediate quantities without altering a single assertion,
so the derived file is still the gate.
"""
import pathlib
import sys

root = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/normalization")
source = (root / "test-reference.c").read_text()

anchor = """        if (differences) fprintf(stderr,"normalization differs at position %u: %zu values\\n",position,differences);
        assert(!differences);
"""
if source.count(anchor) != 1:
    sys.exit("anchor appears %d times, expected 1" % source.count(anchor))

observation = anchor + """
        /* Observation only. Reads the same nine sources the gate just verified and
           reports what each contributes, so the transformation is measured rather
           than reasoned about. No assertion above or below is affected. */
        {
            float scores[9], weights[9];
            assert(normalization_score_sources(model,(const float *const *)sources,scores));
            assert(normalization_softmax(scores,weights));
            double aggregate_norm = 0.0, result_norm = 0.0, dot = 0.0, residual_norm = 0.0;
            for (unsigned coordinate = 0; coordinate < 7168; coordinate++) {
                aggregate_norm += (double)aggregate[coordinate] * aggregate[coordinate];
                result_norm += (double)actual[coordinate] * actual[coordinate];
                dot += (double)aggregate[coordinate] * actual[coordinate];
                residual_norm += (double)resid[position][coordinate] * resid[position][coordinate];
            }
            printf("TAILWALK\\tpos\\t%u\\tresidual\\t%.4f\\taggregate\\t%.4f\\trms\\t%.4f\\tcos\\t%.6f\\n",
                position, sqrt(residual_norm), sqrt(aggregate_norm), sqrt(result_norm),
                dot / (sqrt(aggregate_norm) * sqrt(result_norm)));
            for (unsigned slot = 0; slot < 9; slot++) {
                double norm = 0.0;
                for (unsigned coordinate = 0; coordinate < 7168; coordinate++)
                    norm += (double)sources[slot][coordinate] * sources[slot][coordinate];
                printf("TAILSRC\\t%u\\t%u\\t%s\\t%.4f\\t%.6f\\t%.6f\\n",
                    position, slot, slot == 8 ? "residual" : "snapshot",
                    sqrt(norm), scores[slot], weights[slot]);
            }
            /* The vector the vocabulary head will see, so the last transformation
               can be observed against real values rather than a synthetic vector. */
            {
                static FILE *dump = NULL;
                if (!dump) dump = fopen("/tmp/tail-final.f32","wb");
                if (dump) { assert(fwrite(actual,4,7168,dump) == 7168); fflush(dump); }
            }
            fflush(stdout);
        }
"""

derived = source.replace(anchor, observation)
derived = derived.replace(
    'printf("NORMALIZATION_PASS',
    'printf("NORMALIZATION_PASS')

target = root / "tail-walk.c"
target.write_text(derived)
print("wrote %s, %d lines (was %d)" % (target, derived.count("\n"), source.count("\n")))
