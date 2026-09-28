# clover-server

The same equation as [`clover-k3`](../clover-k3/), decomposed across processes
and machines. **It is a different program for a different reason, not a
replacement**, and the two are expected to coexist indefinitely.

| | `clover-k3` | `clover-server` |
|---|---|---|
| shape | one process, one machine | a client, a server, and 93 layer processes |
| holds | the whole model at once | one layer each |
| experts | read from the checkpoint | read from SQLite, per layer |
| answers | is the equation right? | does the equation survive being cut up? |
| status | working, gated, bit-exact | nothing built yet |

`clover-k3.c` stays exactly as it is. It is the **single-machine version and
the reference**, and the only thing that can say whether anything here is
correct. Nothing in this folder modifies it.

## Why split at all

The single-machine version already answered the computation question: the
model is an equation and the equation runs. What it left is a **data problem
and a concurrency problem** - 1.45 TB of experts that will not fit in one
machine's memory, and one request at a time.

Splitting per layer is the cut the model itself suggests. Each layer's weights
are a contiguous, gap-free range of the trunk - **measured, not assumed**: the
93 slices sum byte-for-byte to the whole trunk with no gap and no overlap. And
each layer's experts belong to that layer alone; the 494,592 expert records
divide as **5,376 per layer for layers 1-92, and zero for layer 0**.

## The shape being tested now

```
  Windows, local          clover-client.c    head, layer 0, tail
        |  socket
  Hetzner, one box        clover-server.c    client requests, and sequencing
        |                                    which layer runs next
  Hetzner, one box        clover-0.c ... clover-92.c    one layer each
        |
                          SQLite, one store per layer   experts, on disk
```

**This is a single-server experiment. There are no pods yet** - Kubernetes is
a later item, and nothing here should be read as a cluster result.

Two different things are called routing and they are kept apart on purpose:

- `clover-server` routes **between layers**. It takes a client request and
  decides which layer runs next.
- The MoE gate routes **to experts**. That is top-k selection inside a single
  layer, driven by the trunk, and it never leaves the layer process.

## What the client holds

The client is the part that needs **no experts and no trunk slice** beyond its
own tensors: `embed_tokens` (2.35 GB), the three output norms (14,336 B each),
and `lm_head` (2.35 GB). **4.70 GB in total.**

Layer 0 sits with the client because it is the one layer with **no experts at
all** - it is a dense MLP - and because it is the cheapest layer in the model
at 0.050 s against a 0.103 s mean, which is 0.5% of the work.

The head is a genuine lookup, 14,336 B per token. The tail is a full matmul,
2.35 GB and 2.35 GFLOP, run once per prompt on the last position only,
measured at 0.053 s and 44.24 GB/s.

## What crosses a layer boundary

Not token ids. Each layer needs the residual for every position **plus every
snapshot taken so far**, so the payload grows with depth: 57,344 B per
position at layer 1 against 258,048 B at layer 85. At 64 positions that is
**9.81 MB per hop and 902.82 MB over the 92 hops**.

## How correctness is established

Every piece is gated against the reference's own output, so nothing here is
ever judged by whether it looks plausible.

`K3_DUMPLAY` in `clover-k3.c` writes each layer's input. So `clover-N` can be
checked against the reference's dump for layer `N+1`, one layer at a time,
**without modifying the reference**. The end of the chain is checked against
the preserved baseline, and the whole thing against the gate:

```
The capital of France is  ->  17374 (' Paris')
logits md5 23d162dcefb18211a7540ef12948f1eb
```

A layer that disagrees with the reference is wrong. There is no second opinion
and no tolerance to argue about.

## Status

**Nothing in this folder is built yet.** The 93 trunk slices exist and are
verified; the expert stores, the modules and the programs do not. The
measurement record is in
[`work-in-progress/clover-scaling-architecture.md`](../work-in-progress/clover-scaling-architecture.md),
where the negatives are kept with the same care as the positives.

### One thing that is unresolved

Migrating the experts into SQLite and reclaiming the checkpoint as it goes
would leave **no expert bytes for `clover-k3.c` to read**. The two programs
are meant to coexist, so either the checkpoint survives the migration or the
reference stops running - and it is the reference that proves this folder
correct. This is named here rather than discovered later.
