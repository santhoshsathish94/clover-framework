# Clover scaling architecture

A record of taking the equation from one program on one machine to a client and
93 pods, measured at every step. Companion to
[k3-equation-solution.md](k3-equation-solution.md), which took the single
program as far as it goes.

Everything here was measured on the same box unless it says otherwise: Ryzen 9
7950X3D, 16 cores / 32 threads, 1 NUMA node, 124 GB RAM, 2x KIOXIA NVMe in
RAID1, RAM read ceiling 42-43 GB/s.

## The shape

```
  client   tokenize, embed                 2.35 GB, a lookup
     |     resid[t] for every position
     v
  pod 0    layer 0    trunk slice 1.17 GB, no experts
  pod 1    layer 1    trunk slice 0.64 GB + 896 experts 15.72 GB
   ...
  pod 92   layer 92   trunk slice 0.42 GB + 896 experts 15.72 GB
     |     resid[T-1] + snapshots
     v
  client   AR, rmsnorm, lm_head            2.35 GB, a matmul
```

The client never sees an expert. The pods never see a token id.

## Step 1 - is there anything to normalize?

The first proposal was a normalized store: each distinct vector once, with
(layer, expert, row) referencing it. That only pays if vectors repeat.

Prior work did not answer this. Solution step 39 measured *symbol* entropy -
3.7544 of 4 bits per nibble, zlib and lzma both worse than that floor. That is
about individual 4-bit codes. It says nothing about whether two whole 3584-code
rows are identical, which is what dedup would catch.

`dedup.c` hashed every packed expert row across all 92 MoE layers, 128-bit
equivalent, treating every hash match as a candidate to be verified byte for
byte rather than as a duplicate.

```
  rows scanned          : 801,898,496
  bytes scanned         : 1361.37 GB
  distinct hashes       : 801,898,496
  hash-equal candidates : 0
  VERIFIED duplicates   : 0
  distinct / total      : 1.000000000
  elapsed               : 223.2 s
```

**Not one vector repeats.** Zero candidates is also a check on the instrument:
at 801M 64-bit hashes the birthday expectation was about a 1.7% chance of one
spurious collision, and there were none.

A `vec`/`ref` split would therefore produce a `vec` table with exactly as many
rows as the `ref` table, plus a foreign key on every one - **6.4 GB of overhead
for zero bytes saved**. With step 39 this closes the question from both sides:
the expert data is irreducible symbol-wise and vector-wise.

223 seconds to avoid building a 1.4 TB database.

## Step 2 - what batching is worth

A server does not run one prompt. Every prompt in flight at layer L needs the
*union* of their experts, not the sum. Computed exactly from the 34-prompt
provenance capture, where each X record is an expert block actually read with
its byte count:

| B | experts | batch GB | GB/prompt | vs B=1 |
|---|---|---|---|---|
| 1 | 8,678 | 152.3 | 152.28 | 1.00x |
| 2 | 14,385 | 252.4 | 126.21 | 1.19x |
| 4 | 24,002 | 421.2 | 105.29 | 1.46x |
| 8 | 36,334 | 637.6 | 79.69 | 1.91x |
| 16 | 50,742 | 890.4 | 55.65 | 2.74x |
| 34 | 65,072 | 1141.8 | 33.58 | **4.55x** |

Still climbing at 34: the union is 65,072 of 82,432 experts, 78.9%. Per layer,
union/sum runs 0.181 to 0.270 - layer 12 shares most, layers 84-90 least, so
deeper layers route more diversely.

**A correction I had to make to my own instrument.** The first run reported
"one prompt alone: 1748.9 GB", which is impossible - a 5-token prompt reads
99.72 GB. I had accumulated `nbytes` into a global dict, so an expert seen in
twenty prompts was counted twenty times. An expert's size is a property of the
expert. After the fix, p01 = 5683 experts x 17,547,264 B = **99.72 GB exactly**,
which matches the recorded figure and validates the instrument.

Also worth stating because it is easy to get wrong: **16 experts is per
position, not per layer.** A 5-token prompt touches 5683 experts across 92
layers = **61.8 distinct experts per layer**.

## Step 3 - SQLite against the current path

Same bytes, same order, three read paths, checksum identical in every run
(`712661`). Order is a fixed shuffle, because routing never delivers experts in
file order.

Bulk, 8.98 GB, cold, 14 threads:

```
  direct     0.65 s   13.79 GB/s      <- what clover-k3 does today
  buffered   1.06 s    8.46 GB/s
  sqlite     1.07 s    8.43 GB/s
```

**SQLite lands exactly on the buffered number.** Its own overhead - blob open,
rowid seek, overflow-page walk - is free. What costs is the page-cache path.

At the real unit of work, 64 experts per stage (the measured per-layer union):

```
  direct      79.8 ms   14.07 GB/s
  sqlite     138.8 ms    8.09 GB/s   cold
  sqlite      45.4 ms   24.76 GB/s   warm
```

**Cross-check against the real program:** 79.7 ms x 92 layers = 7.33 s of
expert I/O, against the actual 5-token run's 79.7% of 8.75 s = 6.97 s. Within
5%, so the harness is measuring the right thing.

### Sharding, and a dead end

Splitting one store into several was proposed to test whether SQLite's ceiling
is per-database or per-path. Cold, 64 experts/stage:

| shards | ms/stage | GB/s |
|---|---|---|
| 1 | 138.8 | 8.09 |
| 2 | 117.3 | 9.58 |
| 4 | 112.0 | 10.03 |
| 8 | 107.2 | 10.47 |

**1.29x from sharding**, saturating by 4. So the ceiling is partly
per-database. The rest is the buffered path, and `PRAGMA mmap_size` at 2, 8 and
16 GB moved cold from 117.3 to 117.6 ms - **nothing**. That knob is closed.

Sharding moved the break-even cache hit rate from 63% to 43%. Warm, it barely
matters: 45.4 ms at one shard, 43.2 at eight.

## Step 4 - one pod

The pod design removes the hit-rate question entirely: a pod owns one layer, so
its universe is 896 experts and it fits in RAM. Warm stops being an edge case
and becomes the steady state. Measured on a real 15.77 GB single-layer store:

```
  COLD START   whole-store warmup 1.96 s, 402 MB peak RSS
               first-pass stages  138.5 ms
  STEADY       45.9 / 45.6 / 46.6 / 46.5 ms per stage   ~24.5 GB/s
  O_DIRECT     79.5 ms                                   14.13 GB/s
```

**A pod is 1.73x faster than O_DIRECT in steady state, for a 1.96 s cold
start.** The advantage grows with batch size: 1.23x at 16 experts/stage, 1.69x
at 64, 1.86x at 224.

Two numbers that decide how pods pack onto machines:

- **Peak is at 4 threads**, not 14: 44.9 ms at 4, 46.8 at 14, 56.9 at 24. A pod
  saturates on 4 cores.
- **Process RSS is 402 MB.** The 15.72 GB lives in the page cache, so private
  memory per pod is negligible.

## Step 5 - what happens when pods share a node

This is the measurement that shapes the cluster. N pods, each with its own
store, each 4 threads, all warm, all at once:

| pods | per-pod latency | per-pod GB/s | node aggregate |
|---|---|---|---|
| 1 | 37.5 ms | 26.20 | 26.20 GB/s |
| 2 | 73.7, 76.5 | 13.34, 12.85 | 26.19 GB/s |
| 3 | 90.5, 97.1, 106.0 | 10.86, 10.12, 9.27 | 30.25 GB/s |
| 4 | 134.3 - 138.0 | 7.32 - 7.12 | 28.90 GB/s |

**Aggregate throughput is flat at 26-30 GB/s however many pods you run.** Pods
divide it; they do not add to it. Per-pod latency scales almost exactly
linearly, and four pods each run 3.6x slower than one.

The control explains it: `membench` reads at 42-43 GB/s on this box. **Memory
bandwidth is the binding constraint** - not cores, not RAM capacity, not disk.
More threads makes it worse: at N=4, 8 threads per pod gives ~170 ms against
~136 ms at 4 threads.

So **the currency is nodes, not pods**:

| layers/node | nodes | stage latency | pipeline throughput |
|---|---|---|---|
| 1 | 93 | 37.5 ms | ~27 batches/s |
| 2 | 47 | ~75 ms | ~13 batches/s |
| 4 | 24 | ~136 ms | ~7 batches/s |

**A failure worth recording.** The first contention run reported "wall 0.0035 s"
for reading 3.9 GB, and I pasted it before noticing that is impossible. Every
process had died on an out-of-range expert index - the filtered shard files hold
experts i, i+4, i+8, which are not contiguous, and the indexing assumed they
were - and my script had sent stderr to `/dev/null`, so it failed silently. Two
faults, both mine: the wrong assumption, and discarding the evidence that would
have shown it.

## Step 6 - the pod specification

Every number here comes from a measurement above.

| | value | why |
|---|---|---|
| CPU | `requests: 4`, `limits: 4` | peak at 4 threads |
| Memory | `requests: 20Gi`, `limits: 20Gi` | worst layer needs 15.63 GiB |
| QoS | Guaranteed | the design assumes residency |
| Layer 0 | 2 CPU, 4Gi | no experts, 1.484 GiB total |
| Pods per node | 1-2, anti-affinity | aggregate is flat at 26-30 GB/s |

Footprint by layer kind:

```
  layer   kind        trunk MB   total GiB
  L0      KDA          1172        1.484     no experts
  L1      KDA+MoE       635       15.626     68 of these
  L3      MLA+MoE       423       15.429     24 of these
```

One layer's experts are 14.643 GiB. **A 16Gi pod leaves 0.374 GiB** - too thin
once anything else is added, which is why the spec says 20Gi.

**Verify early:** the box runs cgroup v2, where **page cache counts against the
pod's memory limit**. That is what keeps the store resident, and it is also what
will OOM-kill a pod that crosses the line. Confirm on pod 1, not pod 93.

## Step 7 - what crosses a pod boundary

Not token ids. Each layer consumes `resid[t]` **plus every snapshot**
`snap[t][0..nsnap-1]` (clover-k3.c lines 1470-1472 and 1725-1727), and
snapshots accumulate where `L % 12 == 0`, so the payload grows with depth:

```
  layer    vectors    bytes/position
  L1        2          57,344
  L25       4         114,688
  L48       5         143,360
  L85       9         258,048
```

| positions | all 92 hops | mean per hop |
|---|---|---|
| 1 | 14.11 MB | 0.15 MB |
| 5 | 70.53 MB | 0.77 MB |
| 16 | 225.71 MB | 2.45 MB |
| 64 | 902.82 MB | 9.81 MB |

At 64 positions that is ~10 MB per hop against a 47 ms stage - nothing over a
socket on the same node, and passable through shared memory with no copy at all.
Across nodes at 10 Gb/s it becomes ~0.7 s per prompt, so **co-scheduling is a
real constraint, not a preference**, and the deepest layers are the expensive
hops.

## Step 8 - the client ends

Both ends need the checkpoint's five non-layer tensors and **neither touches the
trunk or any expert**.

```
  embed_tokens          2,348,810,240   [163840, 7168]   HEAD  lookup
  output_attn_res_norm         14,336   [7168]           TAIL  residual fold
  output_attn_res_proj         14,336   [1, 7168]        TAIL  residual fold
  norm                         14,336   [7168]           TAIL  final rmsnorm
  lm_head               2,348,810,240   [163840, 7168]   TAIL  matmul
                        -------------
  client total                4.70 GB
```

The two ends are not the same job. The **head is a genuine lookup** - 14,336 B
per token out of a 2.35 GB table. The **tail is a full matmul** over the whole
vocabulary: 2.35 GB read and 2.35 GFLOP, once per prompt regardless of length,
because line 2049 uses only `NPOS-1`. Measured as operator `B`: **0.251 s**.

Note the numbering: there are 93 layers, **0 to 92**. Layer 92 is a full
MLA+MoE layer and must be a pod. What sits after it is the tail, which the code
labels `cur_L = 93`.

## Step 9 - the box, rebuilt

The machine could not host the structure: 93 stores x 15.77 GB = 1466 GB against
a 1.8 TB disk holding a 1.5 TB checkpoint. You can have the checkpoint or the
stores. Decision: keep the checkpoint, and make the box a **builder and
validation node** - about 9 stores on disk, 4-6 resident in RAM.

**Preserve first.** About 115 MB existed only on that box, including `eq.c` and
`eq_c_logits.baseline.bin`, which had both been removed from the repository.
Archived to 28.9 MB, sha256 verified on both ends, and confirmed extractable -
377 entries - **before anything was deleted**.

**Reclaim.** 130 GB of unused, regenerable data: the bf16 trunk (102 GB), the
mxfp4 trunk (27 GB), old worktrees and measurement directories. 23 GB -> 153 GB
free. The checkpoint, the int8 trunk and the engine repository were not touched.

**Layout.**

```
  /opt/clover-k3        the repo, cloned from GitHub, canonical
  /srv/k3/model         checkpoint (symlink, untouched)
  /srv/k3/trunk         full.bin, trunk.json, st_model.json, eqidx.bin
  /srv/k3/stores        per-layer stores
  /srv/k3/run           logs and gates
  /srv/k3/config.env    the only file that names a path
```

The index was rebuilt from scratch into the new paths and gated:
**`PASS logits md5 23d162dcefb18211a7540ef12948f1eb`**, token 17374.

The OS was not reinstalled. Nothing was wrong with it, and a remote reinstall
would have risked the checkpoint for no measured benefit.

## Step 10 - the components, and how the code is split

`clover-k3.c` stays untouched as the reference. Everything below is new code
beside it, so the working single-process build is never at risk.

```
  clover-client.c   head, layer 0, and tail, on the caller's machine
  clover-server.c   the control plane: topology, sessions, sequencing
  clover-1.c        layer 1
  ...
  clover-92.c       layer 92
```

### The rule, and its one deliberate exception

**Anything that needs the trunk or the experts stays in a pod.** Only work
that needs neither can sit with the caller.

| | needs trunk | needs experts | where |
|---|---|---|---|
| head, embedding | no | no | client |
| layer 0 | yes, 1.17 GB | **no** | **client, by exception** |
| layers 1-92 | yes | yes | pods |
| tail, norms + lm_head | no | no | client |

Layer 0 breaks the rule and is placed with the client anyway, on measurement:

```
  layer 0            0.050 s      the cheapest layer in the model
  layers 1-92 mean   0.103 s      min 0.080, max 0.150
  sum of all layers  9.51 s
  layer 0 share      0.5%
```

It holds the **largest trunk slice of any layer**, 1.17 GB, and is
nevertheless the **fastest**, because it has no experts and the trunk is
resident. Taking it costs the caller ~50 ms and 1.17 GB, and it removes the
only non-uniform pod in the fleet:

```
  before   client + pod 0 (1.48 GiB, no experts) + 92 MoE pods
  after    client (head, layer 0, tail) + 92 identical MoE pods
```

**The cost accepted, stated rather than glossed:** the client's code roughly
triples. It goes from `AR` + `rmsnorm` + `Bf` to also carrying `Q`, `Qm`, the
whole KDA block - shortconv, the delta rule, l2, SiTU - and the dense MLP. That
is most of a layer's implementation living on the caller's machine, and future
fixes to those kernels ship to clients as well as pods. The judgment was that a
uniform fleet is worth it; the codebase cost is real and is the reason it was
not obvious.

What the client still never needs: `X`, `Xm`, the MoE block, MLA, or any
expert store.

### What the client holds

```
  embed_tokens          2.35 GB    disk only, 14,336 B per token
  layer 0 trunk slice   1.17 GB    read whole, every run
  lm_head + 3 norms     2.35 GB    read whole, every prompt
                        --------
  on disk               5.87 GB
  resident              3.52 GB    lm_head + slice; embed can stay on disk
  with working buffers  ~4 GB
```

Measured alongside: the lm_head projection is **0.053 s**, 2.35 GB at
44.24 GB/s. Client compute is therefore about **0.103 s** total - layer 0 plus
the tail.

### Control through the server, data pod to pod

The payload between layers is **9.81 MB per hop at 64 positions and 903 MB
across all 92 hops** (step 7). Routing that through a central process would
make it move 903 MB per prompt while trying to keep up with 47 ms stages, so
the server would become the bottleneck the pods were split up to avoid.

So the server holds the layer-to-endpoint map, accepts a session, and starts
the chain; the residual and snapshots pass **directly from pod L to pod L+1**
over the dedicated socket, and pod 92 returns the final state to the client.
The server sees kilobytes of control traffic per prompt, not megabytes of
activations.

### What each binary links

A responsible split is not only about tidiness here - it changes what is in
each image. Layer 0 has no experts at all, and the client has no projection
kernel.

| module | contents | linked by |
|---|---|---|
| `k3util` | `now_s`, `die`, `map_file`, `load_ram`, `file_ptr` | everyone |
| `k3index` | `load_index`, `expert_rec`, `slot_ptr`, `slot_vec` | everyone |
| `k3state` | the resid + snapshot wire format | everyone |
| `k3core` | `Q`, `Qm`, `rmsnorm`, `rmsnorm_blocks`, `l2_blocks`, `situ`, `AR` | every layer |
| `k3slice` | trunk slice mmap and slot addressing | layers |
| `k3expert` | `X`, `Xm`, `dq_init`, arena, prefetch pipeline | **MoE layers only** |
| `k3attn_kda` | the KDA attention block | 69 layers |
| `k3attn_mla` | the MLA attention block | 24 layers |
| `k3mlp` | the dense MLP | **layer 0 only** |
| `k3moe` | the MoE block | layers 1-92 |
| `k3tail` | `Bf`, final norms, lm_head | **client only** |

```
  clover-client   util index core slice kda mlp tail
  clover-1..92    util index state core slice expert kda|mla moe
  clover-server   util state only
```

The client carries no expert kernel and no MoE block; the pods carry no
lm_head. That is what the split buys beyond tidiness.

### The oracle that makes this testable one layer at a time

The reference already writes a per-layer checkpoint: `K3_DUMPLAY` dumps every
layer's input `hb[t]` (clover-k3.c line 1486). So **each `clover-N` can be
checked against the reference's own dump for layer N+1**, with no change to
`clover-k3.c` and without waiting for the other 92 to exist.

That sets the order of work: the client first, because two of its three parts
can be gated today - the embedding against the reference's layer-0 dump, and
the lm_head against the preserved baseline, which is `nrm[7168]` followed by
`logits[163840]` and so contains both the input and the output of that matmul.
Then layer 1, verified against the reference's layer-2 dump.

## Step 11 - OPEN: how the experts sit across two devices

A pod has two NVMe devices available. Three structures are candidates and
**none of them is settled**; this is to be decided by measurement, not by
argument:

1. **Two SQLite databases, sharded** - half the layer's experts on each device.
2. **Two SQLite databases, mirrored** - the same data twice, reads balanced.
3. **One database on a mirror** - what the current box already does.

What is already measured and bears on it:

- Sharding *within one filesystem* helps **cold only**: 1/2/4/8 shards give
  8.09 / 9.58 / 10.03 / 10.47 GB/s, saturating by four. **Warm it is worth
  almost nothing**: 45.4 ms at one shard against 43.2 at eight.
- A RAID1 mirror **already reads both members concurrently** - 6.87 + 6.86
  GB/s, timestamp-proven, against a per-device rating near 7 GB/s. So option 3
  is not obviously leaving bandwidth unused.
- In the pod design the store is **RAM-resident in steady state**, so the
  device layout governs **cold start (1.96 s)** and little else.

What cannot be measured on the current box: it has **no independent disk0 and
disk1**. Both NVMes are wholly consumed by RAID1 mirrors and `md2` is the root
filesystem, so testing true two-device independence needs either a machine
with separate filesystems or breaking the mirror, which is destructive.

**So this stays open with a named blocker rather than a guess.** The decisive
experiment is the three structures, same bytes, same routing order, cold and
warm, on hardware where the two devices are genuinely separate - which is the
Kubernetes node, not this box.

## What is measured, and what is not

Measured on one box: the dedup negative, the batching curve, every SQLite and
O_DIRECT number, single-pod cold start and steady state, and the contention
curve up to four pods.

**Not measured, and not to be read as measured:** 93-deep pipeline behavior,
cross-node transfer, Kubernetes scheduling and cgroup accounting, and aggregate
cluster throughput. The pipeline figures in step 5 are arithmetic on a
single-node measurement, not an observed cluster.

## Next

- **Phase 3** - `make-slice.sh`, `make-store.sh`, `pod.sh`. These become the
  container build steps.
- **Phase 4, Tier 1** - all 93 layers as processes, trunk slices from shared
  memory, experts O_DIRECT from the checkpoint. Needs no stores and no extra
  disk. Gate: the chain must reproduce md5 `23d162...`. **This is the test that
  can falsify the decomposition**, and it comes before any performance work.
- **Phase 5, Tier 2** - 4-6 resident pods with real sockets and real payloads.
