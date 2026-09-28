# clover-server-k3

The same equation as [`clover-k3`](../clover-k3/), moved onto a different data
layer and then cut across processes. **Two changes, deliberately not made at
the same time.**

`clover-k3.c` stays exactly as it is. It is the **single-machine version and
the reference**, and the only thing that can say whether anything here is
correct. Nothing in this folder modifies it.

## The order, and why it is this order

Two things are changing: **where the bytes come from**, and **how many
processes there are**. Making both at once means a broken gate tells you
nothing about which one broke it.

**Version 1 - `clover-server-k3.c`.** One process, exactly like
`clover-k3.c`, reading from the new data layer: **93 independent trunk
slices** instead of one 54.47 GB file, and **SQLite** instead of the
checkpoint for experts. Same structure, same arithmetic, same output. It must
reproduce the gate md5 byte for byte.

**Version 2 - the decomposition.** `clover-client.c`, `clover-0.c` through
`clover-92.c`, and `clover-router.c`, built on the data layer that version 1
has already proven.

Version 1 is the one to build and test first.

## What actually changes in version 1

Found by reading the reference, not estimated. The whole program touches its
two data sources in very few places.

**The trunk - three sites.** Every trunk read is `trunk + sl->off`:

| line | where | what it does |
|---|---|---|
| 758 | `slot_ptr` | the pointer every `Q`/`Qm` receives |
| 770 | `slot_vec` | dequantizes a slot to float32 |
| 168 | slot-name-by-address | `K3_STAGE` instrumentation only |

With 93 slices this becomes `slice[L] + (sl->off - layer_off[L])`, because a
slice is a **contiguous, gap-free copy of the layer's range** - measured, the
93 slices sum byte-for-byte to the whole trunk with no gap and no overlap - so
no offset inside it needs rewriting.

**The experts - one function.** `read_range(PRange *g)` is the only place
expert bytes come off disk, two `pread` calls, reached from three call sites
(the pipelined reader, the blocking path, and the serial fallback).
`res_ptr(fid, off)` then resolves a range to the arena, falling back to
`file_ptr(fid) + off` when it is not there. **SQLite replaces the body of
`read_range` and that fallback; the arena, the addressing and the pipeline
stay as they are.**

**What does not change in version 1.** The five non-layer tensors -
`embed_tokens`, the three output norms and `lm_head` - keep reading from the
checkpoint through `file_ptr` (lines 1441, 2031, 2055). They are 4.70 GB, they
are not experts, and moving them is version 2's job.

## Version 2: the shape being tested

```
  Windows, local          clover-client.c    head, layer 0, tail
        |  socket
  Hetzner, one box        clover-router.c    client requests, and sequencing
        |                                    which layer runs next
  Hetzner, one box        clover-0.c ... clover-92.c    one layer each
        |
                          SQLite, one store per layer   experts, on disk
```

**This is a single-server experiment. There are no pods yet** - Kubernetes is
a later item, and nothing here should be read as a cluster result.

Two different things are called routing and they are kept apart on purpose,
which is why the control plane is `clover-router.c` and not
`clover-server.c`:

- **`clover-router` routes between layers.** It takes a client request and
  decides which layer runs next.
- **The MoE gate routes to experts.** Top-k selection inside a single layer,
  driven by the trunk, and it never leaves the layer process.

## Why the cut is per layer

The single-machine version already answered the computation question: the
model is an equation and the equation runs. What it left is a **data problem
and a concurrency problem** - 1.45 TB of experts that will not fit in one
machine's memory, and one request at a time.

Each layer's weights are a contiguous range of the trunk, and each layer's
experts belong to that layer alone: the 494,592 expert records divide as
**5,376 per layer for layers 1-92, and zero for layer 0**.

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

Version 1 has the easiest possible test: it is the same program with different
plumbing, so it must produce the **same logits, bit for bit**.

Version 2 uses `K3_DUMPLAY`, which makes the reference write each layer's
input. So `clover-N` can be checked against the reference's dump for layer
`N+1`, one layer at a time, **without modifying the reference**. The end of
the chain is checked against the preserved baseline, and the whole thing
against the gate:

```
The capital of France is  ->  17374 (' Paris')
logits md5 23d162dcefb18211a7540ef12948f1eb
```

A layer that disagrees with the reference is wrong. There is no second opinion
and no tolerance to argue about.

## Status

**Nothing in this folder is built yet.** The 93 trunk slices exist and are
verified; the expert stores and the programs do not. The measurement record is
in
[`work-in-progress/clover-scaling-architecture.md`](../work-in-progress/clover-scaling-architecture.md),
where the negatives are kept with the same care as the positives.

### One thing that is unresolved

Migrating the experts into SQLite and reclaiming the checkpoint as it goes
would leave **no expert bytes for `clover-k3.c` to read**. The two programs
are meant to coexist, so either the checkpoint survives the migration or the
reference stops running - and it is the reference that proves this folder
correct. This is named here rather than discovered later.
