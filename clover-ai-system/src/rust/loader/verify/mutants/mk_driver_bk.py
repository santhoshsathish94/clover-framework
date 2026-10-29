#!/usr/bin/env python3
# Proposed fix for the driver weakness "the callbacks never read the bookkeeping" (M82/M83), written as a patch APPLIED TO A COPY (the real ld_driver.cpp is not touched).
# usage: mk_driver_bk.py   -> verify/mutants/probe/ld_driver_bk.cpp   (build: see run_probes.sh)
# Change 1: cb_trunk (the first callback that gets the handle) checks that *h_L, *h_card, *h_k_in and handles[layer] are already written when the job starts.
# Change 2: scenario open, while the jobs are held (hold=1), the caller's `card` int of every handle is overwritten after all ld_open_layer calls returned and restored after ld_wait;
#           a job that re-reads *h_card at run time (instead of the card captured at the call) then runs set_device / the callbacks for the wrong card.
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "..", "ld_driver.cpp"), newline="").read().replace("\r\n", "\n")
out = os.path.join(here, "probe"); os.makedirs(out, exist_ok=True)
a = "static void cb_trunk(void *ctx, void *h, const char *ds, int L)\n{\n    chk_ctx(ctx); LayerRec *r = (LayerRec *)h;\n"
b = a + "    CK(r->L == (int)r->layer && (r->C == r->card || r->C == (r->card + 1) % 8) && r->K == (int)r->k, \"layer %u: bookkeeping not written when the job started (L=%d card=%d k_in=%d)\", r->layer, r->L, r->C, r->K);\n" \
        "    if (r->layer <= 92) CK(g_handles[r->layer] == r, \"layer %u: handles[] entry not written when the job started\", r->layer);\n"
assert src.count(a) == 1
src = src.replace(a, b)
a2 = "    if (hold) { std::fill(dsbuf.begin(), dsbuf.end(), 'X'); g_hold = false; }\n    timed_wait(\"open: ld_wait did not return\", __LINE__);\n"
b2 = "    if (hold) { std::fill(dsbuf.begin(), dsbuf.end(), 'X'); for (LayerRec *lr : recs) lr->C = (lr->card + 1) % 8; g_hold = false; }\n    timed_wait(\"open: ld_wait did not return\", __LINE__);\n    if (hold) for (LayerRec *lr : recs) lr->C = lr->card;\n"
assert src.count(a2) == 1
src = src.replace(a2, b2)
open(os.path.join(out, "ld_driver_bk.cpp"), "w", newline="").write(src)
print("wrote", os.path.join(out, "ld_driver_bk.cpp"))
