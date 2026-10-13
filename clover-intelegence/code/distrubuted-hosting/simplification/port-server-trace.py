#!/usr/bin/env python3
"""Trace the server, which is layer 0 and the one layer with no routed experts.

The stage walk covered pods 1 to 92, stages 121 to 11160 of 11,280. The creator pointed
out that the chain is not ninety-two layers: text becomes tokens, tokens become vectors,
layer 0 runs, then 1 to 92, then normalisation, then a vector becomes a token again.
The first and last of those were never looked at.

server_process is a flat sequence rather than a stage machine, so the equivalent
granularity is one record after each step. The server is pure trunk — dense_gate and
dense_up are 33792 wide and there is no routing — so this is also the cleanest view of
the trunk path on its own.

  5201  input, the token embedding arriving
  5202  normalized
  5203  residual after attention output
  5204  aggregate, the two-source fold of snapshot and residual
  5205  dense_output, the trunk's contribution
  5206  residual on the way out
  5207  the two aggregate weights

Owner 0. Behind clover_vectors_active, so nothing happens with CLOVER_VECTORS unset.
Idempotent.
"""
import pathlib
import sys

PATH = pathlib.Path("/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/server/server.c")
MARK = "CLOVER_SERVER_TRACE"

INCLUDE_ANCHOR = """#include <sys/mman.h>
#include <sys/stat.h>
"""

INCLUDE_PATCH = """#include <sys/mman.h>
#include <sys/stat.h>
/* CLOVER_SERVER_TRACE: relative include because the server is not built with -I common. */
#include "../common/trace.h"
"""

# server_aggregate computes the two weights into a local; capture them where they exist.
AGG_ANCHOR = """    float maximum = scores[1] > scores[0] ? scores[1] : scores[0];
    double total = 0.0;"""

AGG_PATCH = """    /* CLOVER_SERVER_TRACE */
    if (clover_vectors_active(0))
        clover_vectors_write(0, 5207, (unsigned long)sequence->positions, scores, 2);
    float maximum = scores[1] > scores[0] ? scores[1] : scores[0];
    double total = 0.0;"""

PROC_ANCHOR = """    memcpy(sequence->snapshot, input, sizeof sequence->snapshot);
    server_normalize(sequence->normalized, sequence->snapshot, server->qkv + 88, SERVER_WIDTH);
    server_qkv(server, sequence);
    server_decay(server, sequence);
    server_update_attention(sequence);
    server_attention_output(server, sequence);
    server_aggregate(server, sequence);
    server_dense(server, sequence);"""

PROC_PATCH = """    memcpy(sequence->snapshot, input, sizeof sequence->snapshot);
    /* CLOVER_SERVER_TRACE: one record per step, the same granularity the pods get. */
    clover_vectors_open(0);
    if (clover_vectors_active(0))
        clover_vectors_write(0, 5201, (unsigned long)sequence->positions, input, SERVER_WIDTH);
    server_normalize(sequence->normalized, sequence->snapshot, server->qkv + 88, SERVER_WIDTH);
    if (clover_vectors_active(0))
        clover_vectors_write(0, 5202, (unsigned long)sequence->positions, sequence->normalized, SERVER_WIDTH);
    server_qkv(server, sequence);
    server_decay(server, sequence);
    server_update_attention(sequence);
    server_attention_output(server, sequence);
    if (clover_vectors_active(0))
        clover_vectors_write(0, 5203, (unsigned long)sequence->positions, sequence->residual, SERVER_WIDTH);
    server_aggregate(server, sequence);
    if (clover_vectors_active(0))
        clover_vectors_write(0, 5204, (unsigned long)sequence->positions, sequence->aggregate, SERVER_WIDTH);
    server_dense(server, sequence);
    if (clover_vectors_active(0)) {
        clover_vectors_write(0, 5205, (unsigned long)sequence->positions, sequence->dense_output, SERVER_WIDTH);
        clover_vectors_write(0, 5206, (unsigned long)sequence->positions, sequence->residual, SERVER_WIDTH);
    }"""

text = open(PATH, "r", newline="").read()
if MARK in text:
    print("already patched")
    sys.exit(0)

failures = 0
for name, anchor, patch in (("include", INCLUDE_ANCHOR, INCLUDE_PATCH),
                            ("aggregate", AGG_ANCHOR, AGG_PATCH),
                            ("process", PROC_ANCHOR, PROC_PATCH)):
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
