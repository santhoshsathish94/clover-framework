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

## Step 6 - the pod, with experts on disk

**The equation already solved the computation problem.** Over 34 prompts,
pure arithmetic is **1.9% of wall time** and expert bytes are **79.7%**. What
is left is a data problem and a concurrent-request problem, and the pod exists
to solve those - not to hold a model in memory.

So the experts **stay on disk**, in a SQLite database sliced per layer, read by
that layer's `clover-N`. The pod holds in memory only its trunk slice and its
working buffers.

**A correction to the earlier version of this section.** It specified 20Gi per
pod on the assumption that a layer's 896 experts would be resident, which made
the warm measurements the steady state. That was my inference, not the design.
With experts on disk the applicable numbers are the **cold** ones:

```
  per 64-expert stage, cold, 14 threads
    O_DIRECT            79.8 ms   14.07 GB/s
    SQLite  8 shards   107.2 ms   10.47 GB/s
    SQLite  1 shard    138.8 ms    8.09 GB/s
```

The 43-45 ms warm figures apply only to whatever fraction the page cache
happens to hold, which is a consequence of how much memory the pod is given -
not something the design guarantees.

### What a pod actually needs, under this design

| | |
|---|---|
| on disk | the layer's SQLite store, **15.77 GB** |
| resident | the trunk slice, **423 MB (MLA) to 635 MB (KDA)** |
| plus | `resid`, `snap`, MoE intermediates at `SI=6144` and `DI=33792` per position, KDA state `St[96][128][128]` at 6.29 MB |
| read per stage | about **1.1 GB** of expert blobs, from disk |

**Memory is therefore in the low single-digit GB, not 20Gi** - and the more
that is granted above the working set, the more page cache holds and the closer
the pod moves toward the warm numbers. That is a dial, not a requirement.

Across the fleet: 92 layers x 15.77 GB = **1451 GB of expert store**, one
slice per node rather than a copy of the model anywhere.

**Still not measured, and the reason the spec stays provisional:** every
figure above for CPU and memory comes from `sqstage`, which reads blobs and
checksums them - no attention, no routing, no MoE arithmetic, no layer state.
`clover-1` running on real inputs is what will replace them. The only line
with a real measurement behind it remains pods-per-node, from the contention
curve.

**Verify early:** the box runs cgroup v2, where **page cache counts against the
pod's memory limit**. With experts on disk that cuts the other way from before -
the limit now decides how much of the store can stay cached. Confirm on pod 1,
not pod 92.

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
  clover-router.c   the control plane: topology, sessions, sequencing
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
  clover-router   util state only
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

## Step 12 - the trunk, split 93 ways

The first artifact the decomposition actually needs. `make_slice.py` run over
every layer, each slice written and then **sha256-compared against a second
read of the source range**, so a slice is proven to be the range it claims and
not merely the right length.

```
  slices                93
  sum of all slices     54,468,222,976 B   54.47 GB
  trunk full.bin        54,468,222,976 B   54.47 GB
  contiguous, no gaps   yes
  covers whole trunk    yes
  distinct sha256       93 of 93
  MLA / KDA / layer 0   24 / 68 / 1
  disk free after       102 GB   (was 150 GB)
```

Sizes are as established earlier and unchanged: **L0 1172 MB**, KDA **635 MB**
x68, MLA **423 MB** x24. Padding is 120 B on layer 0 and every KDA layer, 1784
B on every MLA layer.

Two things this settles rather than assumes:

- The sum of the 93 slices is **byte-for-byte the whole trunk**, and each
  slice's recorded `source_off` equals the running total of its predecessors.
  So the layers tile the trunk exactly - no gap, no overlap, nothing dropped.
- **24 MLA + 68 KDA + layer 0**. Layer 0 is itself KDA, which gives 69 KDA in
  total, matching the structure taken from the checkpoint.

The 93 distinct hashes are worth stating plainly: the KDA slices are all the
same size, so identical hashes would have meant a copy bug. They are all
different.

**A note on what this is for.** On this box the slices are redundant - the
layer processes can map their range straight out of the shared
`/dev/shm/trunk.bin` and the 54.47 GB on disk buys nothing. The slices matter
when a layer becomes a separately shipped image, which is the later item. They
were built now because the split had to be proven correct before anything is
built on top of it, and proving it costs disk rather than reasoning.

## Step 13 - the filesystem can reclaim space from inside a file

The incremental migration depends on a property that had to be checked, not
assumed: whether freeing a byte range inside a file returns the blocks to the
filesystem **without moving anything after it**.

```
  filesystem       ext4
  punch-hole rc    0
  du before        64M
  du after         32M
  file size        67108864   unchanged
```

`fallocate --punch-hole --keep-size` works. **The file size does not change**,
which is the load-bearing part: every offset recorded in the index stays valid
for the records that have not been migrated yet, so migration can proceed one
expert at a time with the index still usable throughout.

## Step 14 - two programs, and two changes taken one at a time

`clover-k3.c` is the **single-machine version**: one process, one box, the
whole model in one address space. The work recorded here is a **client-server
version**. They exist for different reasons and both are kept.

The new code lives in **`clover-server-k3/`**, a separate folder, so that
nothing built here can reach into the reference by accident. `clover-k3.c` is
not modified.

**Two things are changing, and they are not being changed together.** Where
the bytes come from, and how many processes there are. Both at once means a
broken gate says nothing about which one broke it.

- **Version 1, `clover-server-k3.c`** - one process, the same structure as the
  reference, reading 93 independent trunk slices and SQLite experts instead of
  one trunk file and the checkpoint. Must reproduce the gate md5 bit for bit.
- **Version 2** - `clover-client.c`, `clover-0.c` .. `clover-92.c`, and
  `clover-router.c`, on the data layer version 1 has already proven.

The control plane is named **`clover-router.c`**, not `clover-server.c`,
because two different things are called routing: the router sequences
**between layers**, while the MoE gate selects **experts** inside one layer.

**What version 1 has to touch, read off the reference rather than estimated.**
Every trunk read is `trunk + sl->off` and there are **three** of them -
`slot_ptr` (758), `slot_vec` (770), and the `K3_STAGE` slot-name-by-address
lookup (168). Every expert byte comes off disk in **one function**,
`read_range`, two `pread` calls, reached from three call sites, with
`res_ptr` falling back to `file_ptr(fid) + off`. The five non-layer tensors
keep reading from the checkpoint; they are 4.70 GB and they are not experts.

The separation is not tidiness. The single-machine version is the **only
oracle** available: `K3_DUMPLAY` gives each layer's input, so `clover-N` can
be checked against the reference's dump for layer `N+1`, and the end of the
chain against the preserved baseline and the gate md5. A decomposed layer that
disagrees with it is wrong, with no tolerance to argue about.

**Which produces a conflict worth naming now rather than discovering later.**
Migrating the experts into SQLite and reclaiming the checkpoint as it goes
would leave no expert bytes for `clover-k3.c` to read. Either the checkpoint
survives the migration, or the reference stops running - and it is the
reference that proves the decomposition correct. Unresolved.

## Step 15 - the trunk split costs nothing, and is bit-exact

`clover-server-k3.c` built: the reference with one change, the trunk arriving
as **93 independent per-layer mappings** instead of one 54.47 GB file.
Experts still come from the checkpoint. Every kernel, operator and buffer is
untouched.

Four sites changed, which is all the reference has: `slot_ptr`, `slot_vec`,
the `K3_STAGE` slot-name-by-address lookup, and teardown. `cover()` translates
a slice address back to the page it had in the whole trunk so `K3_COVER` stays
comparable.

**`layer_off[]` is derived, not read.** It is the running sum of the slice
file sizes, which is only right if the layers tile the trunk with no gap, so
every present slot is bounds-checked against its own slice at startup before a
single weight is addressed. All 93 layers passed silently.

### The measurement

Same session, three runs each. The reference and the slice build cannot both
be tmpfs-resident at once - 54.47 GB each against 124 GB of RAM - so they ran
**in sequence, not interleaved**, which is a real weakness of this comparison.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| A reference, trunk from tmpfs | 9.52 | 8.97 | **8.78** |
| B slices from disk | 10.76 | 8.76 | **8.74** |
| C slices from tmpfs | 8.73 | 8.73 | **8.73** |

**All nine runs produced md5 `23d162dcefb18211a7540ef12948f1eb` and token
17374.** Peak RSS 56.8 GB.

**The result is that there is no result: 8.78 against 8.74 against 8.73 is a
0.05 s spread, which is noise.** Splitting the trunk 93 ways is free. That is
the outcome this step wanted - a plumbing change that is bit-exact and costs
nothing is what makes the next change safe to attribute.

Two things worth reading off it beyond the headline:

- **Cold start is the only cost, and it is one run.** B's first run is 10.76 s
  against 8.74 warm. Once the page cache holds the 54.47 GB, slices on disk
  are indistinguishable from slices in tmpfs - the box has roughly 70 GB free
  with the trunk copy resident, so they simply stay cached.
- **A was still warming** (9.52, 8.97, 8.78) while C was flat from its first
  run. A ran first and C last, so run order is a confound. All three converge
  on the same 8.7 s, which is the figure already on record.

### The error in this step, and how it was caught

The first version of the harness reported **three PASSes for runs that never
happened.** It set the environment inside a shell function and passed the
per-run variables through `"$@"`; assignments are recognized when the command
is parsed, before expansion, so the shell treated `K3_SLICES=...` as the name
of a program to run. Nothing executed. The md5 check then read
`build/srv.bin` **left behind by an earlier manual run** and matched the
baseline.

The tell was not the md5. It was the **empty wall-time column** - a passing
run with no timing is not a run. The fix is in the harness and is the general
one: **delete the output file before every run**, so a stale artifact cannot
pass a check. Same shape as the earlier control-clobber, where a script was
allowed to overwrite the control before the comparison.

## Step 16 - what the expert bytes already are, and one layer in SQLite

### Context, read before designing anything

`probe_experts.py` against the index, layer 1:

```
  expert records                494,592   92 layers x 896 x 6
  bytes per layer               15.72 GB, identical for every layer
  an expert's six ranges        one contiguous run, 17,547,264 B
  experts whose six are one run 896 of 896
  distinct per-expert sizes     exactly one
  files a layer touches         1
  maximal contiguous runs       1, of 5,376 records
```

**A whole layer is a single contiguous run in a single file**, and an expert
is a single contiguous run inside that. So the schema is not a choice:

```sql
  expert(id INTEGER PRIMARY KEY, data BLOB)   -- 896 rows, one per expert
  meta(k, v)                                  -- sizes, source, sha256, parts
```

The six sub-range offsets inside a blob are recorded in `meta`, so a reader
that wants only `gate.w` need not fetch the whole expert. Building a store is
a sequential read.

### The finding the builder tripped over

The first build **failed on its own assertion**: `expert ids are not in file
order`. That was not a bug in the data, it was an assumption in my builder.
Measured: **894 of 896 expert ids are out of file order**. The layer still
tiles perfectly - the contiguity and total-bytes checks both passed - but the
id sequence is permuted within it.

It matters twice. The builder must walk **file order** to keep the read
sequential while inserting each blob under its **own id**; and the read-back
hash must use the same order, or it compares a permutation against a straight
read and fails for no real reason. Both are now explicit.

### The store

```
  L1   896 experts x 17,547,264 B = 15.72 GB  ->  store 15.74 GB (+0.10%)
       write 22.9 s   verify 22.0 s   sha256 8663636d1e24f7c2   MATCH
```

**+0.10% overhead**, better than the 0.3% seen earlier, which is the 64 KB
page size. MATCH means three hashes agree: the blobs as written, the blobs
read back out of SQLite, and **a second independent read of the source
range**. A store that matches only itself proves nothing.

Disk 102 GB -> 87 GB.

### What this forces about the gate

One layer's store is 15.74 GB, so all 92 would be **1448 GB** and the box has
87 GB. The gate runs all 92 MoE layers, so a SQLite build cannot be gated by
converting everything.

The way through is a **hybrid**: a layer reads from its store if one exists,
and from the checkpoint if not. Then the gate can be run with one layer on
SQLite and 91 on the checkpoint, and coverage grows as far as the disk allows.
It also happens to be what the migration needs anyway.

## Step 17 - a layer served from SQLite, bit-exact, and what it costs

`read_range` is the only place expert bytes come off disk, so that is the only
place that changed. A layer reads from its store if one exists and from the
checkpoint if not, which is what lets the gate run with one layer converted
and 91 not.

Addressing is arithmetic rather than a search, because step 16 established the
shape: a layer is one contiguous run and every expert in it is the same size,
so a file offset gives the position directly. Position is mapped through
`pos2id` because the ids are permuted. Reads use `sqlite3_blob_read` at an
offset inside the blob, so a 5.5 MB range does not materialize a 17.5 MB
expert.

### The result

```
  layer 1 from SQLite, cold    12.30 s   PASS
  layer 1 from SQLite, warm     9.44 / 9.04 / 9.08 s   PASS
  same binary, no store         8.77 / 8.75 s          PASS
```

Every run produced md5 `23d162dcefb18211a7540ef12948f1eb` and token 17374.
**One layer served entirely out of SQLite is bit-exact.** That was the
question this step asked, and the answer is yes.

**It costs about 0.3 s for one layer warm**, against a 0.02 s run-to-run
spread in the control, so the cost is real and not noise. SQLite moved 1.30 GB
in 4.81 to 6.49 thread-seconds, a per-thread 0.20 to 0.27 GB/s.

The 1.30 GB is worth reading twice: it is **74 of 896 experts**, not the whole
store. That is the prompt's working set for layer 1, and it matches the
per-layer maximum measured earlier from the provenance.

### What this does not establish

**Do not multiply 0.3 s by 92.** With one layer converted, the other 91 still
run the O_DIRECT pipeline and the two paths overlap; a run with every layer on
SQLite is a different system, not this one scaled up. The honest statement is
narrow: one layer, warm, costs 0.3 s more than the same layer on O_DIRECT, and
the output is identical.

The rate is also well under what SQLite did standalone - 8 to 10 GB/s cold and
24 to 26 GB/s warm in step 3. So there is something in the integration, not
in SQLite, holding it back. The most likely candidate is that with range
granularity consecutive ranges belong to different experts, so the blob handle
is reopened on nearly every read. **That is a hypothesis and it has not been
measured**, which is the next thing to do rather than the next thing to
assume.

### Two defects found while wiring it

**Thread identity was the wrong key.** The first version indexed a connection
per thread with `omp_get_thread_num()`. The reader threads are **pthreads**,
where that returns 0 for every one of them, so all 14 shared one connection
and one blob handle and the run died at the first expert with `blob open`.
Indexing by thread would not have worked even if the number were right:
`pl_start` creates readers and `pl_finish` joins them **once per layer**, so a
run makes roughly 1288 distinct threads.

The fix is a **claimed pool**: 32 slots, each keeping its connection and blob
handle across claims, taken and released around a read. It does not care which
threading model the caller uses or how often threads churn.

**The error message said nothing.** `die("blob open")` gave no reason. Adding
`sqlite3_errmsg` was worth more than the guess it replaced.

## Step 18 - where the SQLite time goes, and a fix that costs more than it saves

### First, how many experts a layer actually reads

Worth stating plainly because it is neither of the two obvious answers. From
the run's own counters:

```
  draws 7360      92 layers x 5 positions x top-16
  distinct 5683   after dedup within each layer  ->  61.8 per layer
  m=1 4409 (77.6%)  m=2 957  m=3 239  m=4 70  m=5 8   mean m 1.295
```

**Top-16 is per position, not per layer.** A layer reads the *union* of its
positions' selections - 61.8 experts on average, 74 in layer 1 - each read
once and used `m` times. That is why layer 1's SQLite traffic is 1.30 GB and
not 0.28 GB (16 experts) or 15.72 GB (all 896), and it is the mechanism
behind the batching curve in step 2: more positions, more uses per byte.

Layer 1: **444 reads = 74 experts x 6 ranges**, exactly.

### The candidate

4.19 thread-seconds for 1.30 GB is 9.4 ms for an average 2.93 MB read, about
a hundred times slower than copying cached bytes. A 17,547,264 B blob at a 64
KB page size is a chain of roughly 268 overflow pages, and reading at an
offset walks that chain; a blob handle caches the page list per row, so
reopening to another row throws it away. Measured reopen rate: **60 to 64% of
reads**.

`K3_PLGRAN=0` hands one thread all six ranges of an expert instead of
spreading ranges across threads, so it tests this for free.

### The 2x2

| | wall | md5 | reopens |
|---|---|---|---|
| PLGRAN=1, store | 8.99 | PASS | 266 of 444 (60%) |
| PLGRAN=1, no store | 8.76 | PASS | - |
| PLGRAN=0, store | 9.66 | PASS | 72 of 444 (16%) |
| PLGRAN=0, no store | 9.56 | PASS | - |

Reopens fall 3.7x and **the SQLite cost does fall with them, 0.23 s to
0.10 s**. So the mechanism is real and accounts for roughly half the overhead.

**And the fix is rejected anyway.** `PLGRAN=0` costs **0.80 s on the baseline
path** - 9.56 against 8.76 with no store at all - which is step 17 of the
earlier arc reappearing: range granularity was worth 1.56 s to 0.29 s of
first-expert stall for O_DIRECT. Paying 0.80 s to save 0.13 s is a bad trade.
The best combination is the one already in use: **range granularity, store,
8.99 s**.

The honest conclusion is narrow: reopening explains about half the SQLite
overhead, and the only lever currently wired to it is worth less than it
costs. A fix that keeps range granularity would have to remove the offset
walk itself - **one row per range instead of one row per expert**, so every
read starts at offset 0 of its own blob. That is a schema change, it is not
built, and it is not claimed to work.

### A rejected explanation, checked rather than assumed

Before blaming SQLite I checked whether the store was even cached. It is
**39.16% resident**, 6.16 GB of 15.74 GB, with 72 GB available and the trunk
copy holding 50 GB of shared memory. That sounds like the answer but is not:
the run touches 1.30 GB, and 6.16 GB of resident pages is nearly five times
that. Residency is not what is slow here. Recorded because a plausible cause
that turns out not to be the cause is worth as much as the one that is.

## Step 19 - the cost is linear, and it is exactly the bandwidth difference

Stores built for layers 2, 3 and 4, each verified against a second read of
its source range, each **+0.10%** and each with **894 of 896 ids out of file
order** - the permutation is consistent across layers, not an accident of
layer 1. Disk 87 GB -> 43 GB.

The only variable is how many stores are visible, controlled by symlinking a
subset into a selection directory. `n=0` is the same binary with no store,
so nothing but the store count differs. Three runs each, **all 15 PASS**.

| stores | warm wall | delta | per layer | SQLite GB | thread-s |
|---|---|---|---|---|---|
| 0 | 8.89 | - | - | - | - |
| 1 | 9.02 | 0.13 | 0.13 | 1.30 | 4.41 |
| 2 | 9.17 | 0.28 | 0.14 | 2.53 | 7.81 |
| 3 | 9.25 | 0.36 | 0.12 | 3.54 | 10.15 |
| 4 | 9.50 | 0.61 | 0.15 | 4.58 | 13.42 |

**Linear, at 0.148 s per converted layer.** Bytes and thread-seconds are
linear too, about 1.15 GB and 3.3 thread-seconds each, and the per-thread
rate is **flat at 0.29 to 0.35 GB/s** across all four settings - so this is a
fixed per-byte cost, not contention that would worsen with more layers.

### The cross-check that makes it an explanation rather than a curve

1.15 GB per layer at 3.3 thread-seconds is 0.35 GB/s per thread, and with 14
readers that is **4.9 GB/s aggregate against O_DIRECT's 14 GB/s**. So moving
one layer's bytes across should cost

```
  1.15 GB / 4.9 GB/s  -  1.15 GB / 14 GB/s  =  0.235 - 0.082  =  0.153 s
```

against **0.148 s measured**. The wall-clock delta is entirely accounted for
by the delivered bandwidth. Nothing else needs to be invoked.

### What it would mean at 92 layers, and why that is arithmetic

A straight line through these four points gives `8.89 + 0.148n`, so 92 layers
would be about **22.5 s, roughly 2.5x the current run**. Equivalently, the
whole 99.72 GB of expert traffic at 4.9 GB/s instead of 14 GB/s.

**This is arithmetic on a four-point measurement, not an observation**, and
there is a specific reason to distrust it: at n=92 there is no O_DIRECT
expert traffic left at all, so the two paths no longer overlap and the system
is not this one scaled up. It cannot be settled on this box - 92 stores are
1448 GB against 43 GB free.

### Where this leaves the store design

SQLite delivered **8 to 10 GB/s cold and 24 to 26 GB/s warm** in the
standalone benchmark of step 3. Here it delivers 4.9. The difference is how
it is being asked: 444 offset reads into 17.5 MB blobs, 60% of them after a
reopen that discards the cursor's overflow page list.

So the schema named at the end of step 18 is now the obvious candidate rather
than a guess - **one row per range instead of one row per expert**, 5,376
rows a layer, every read starting at offset 0 of its own blob and no chain to
walk. It is still not built and still not claimed to work, but it is now
pointed at a measured 2.9x gap rather than at a hunch.

## Step 20 - a row is a tensor, not an expert: 58% of the overhead gone

The store now holds **one row per expert tensor** - `part(id INTEGER PRIMARY
KEY, expert, which, kind, data)`, 5,376 rows a layer, `id = expert * 6 +
which * 2 + kind` so a reader computes the rowid with the same arithmetic
`expert_rec` already uses. Every read is offset 0 of its own blob.

**Six columns would not have worked** and were rejected before building:
SQLite serializes a row into one record with one overflow chain, so reading
the sixth column still walks to its offset. Separate rows are what gives each
tensor its own chain.

Built for layer 1, verified the same way as before, and the result carries a
useful cross-check: **sha256 `8663636d1e24f7c2`, identical to the
expert-per-row store.** The tensors concatenated in file order are the same
bytes as the experts concatenated in file order, which they must be.

```
  L1   896 experts x 6 = 5376 rows, 15.72 GB  ->  15.77 GB (+0.28%)
```

Overhead rises from +0.10% to +0.28%, which is 5,376 rows of per-row cost
instead of 896.

### The A/B, interleaved, one binary

The binary detects which table a store has, so both schemas run through the
same build. Cycles are interleaved rather than blocked, so drift in machine
state cannot be read as a difference between the schemas.

| | cycle 1 | 2 | 3 | mean | vs control |
|---|---|---|---|---|---|
| no store | 8.78 | 8.77 | 8.71 | 8.753 | - |
| row = expert | 8.98 | 8.99 | 8.98 | 8.983 | **+0.230** |
| row = tensor | 8.83 | 8.85 | 8.87 | 8.850 | **+0.097** |

**All 12 runs PASS.** The overhead falls 0.230 s to 0.097 s, **58% of it
gone**, and the per-thread measurement agrees:

```
  row = expert   1.30 GB   4.61 thread-s   0.28 GB/s    65% reopens
  row = tensor   1.30 GB   2.35 thread-s   0.55 GB/s    97% reopens
```

Thread-seconds halve. Aggregate across 14 readers goes 3.9 -> 7.7 GB/s
against O_DIRECT's 14.

### This corrects step 18

Step 18 named the mechanism as **reopens discarding the cursor's cached
overflow page list**. That was not right. **Reopens went up, 65% to 97%, and
the cost halved.** With one row per tensor every read is a different rowid,
so a reopen happens almost every time and costs almost nothing.

The mechanism is specifically **walking to a non-zero offset**. A reopen only
mattered because it forced that walk to start again. Reading from offset 0
traverses pages as it copies them, which is work that has to happen anyway.

Step 18's rejection of `K3_PLGRAN=0` still stands, and for the same reason it
gave: grouping an expert's ranges onto one thread reduced re-walking, which
is why it helped at all.

### What is still open

7.7 GB/s against O_DIRECT's 14 GB/s, so **roughly half the gap remains** and
is not attributed. The linear curve in step 19 was measured for the
expert-per-row schema; **this schema has been measured at one layer only**, so
the 0.097 s should not yet be treated as a per-layer constant. Rebuilding
layers 2 to 4 in the new schema would settle it, and needs the old stores
deleted first - 29 GB free against 47 GB needed.

## Step 21 - the whole model in SQLite, bit-exact, and 3.8x slower

Everything is now in SQLite: the trunk, the experts, the embedding lookup
table, `lm_head`, the output norms and the tokenizer. Nothing reads a
safetensors file or a slice.

```
  catalog.db    164 KB   every relationship, foreign keys on
  trunk/        93 files    54.47 GB   one row per trunk slot
  expert/       92 files  1446.46 GB   one row per expert tensor, 5,376 a layer
  client/        4 files     4.70 GB   embed, lmhead, head, vocab
  -------------------------------------------------------------
  189 payload files, 1,505.62 GB, every one verified against a second
  independent read of its source before being marked complete
```

`PRAGMA foreign_key_check` reports **0 violations**, 0 files incomplete, and
**189 distinct sha256 for 189 files** - no two payload files are the same
bytes.

### The gate

```
  md5 23d162dcefb18211a7540ef12948f1eb   token 17374   every run
  trunk    93 layers from SQLite, 54.47 GB
  experts  92 of 92 layers, 99.72 GB, 34,098 reads
```

**34,098 is exactly 5,683 x 6** - the distinct experts a 5-token prompt
touches, times their six tensors. The whole model answers from a database and
the logits are bit-identical.

### What it costs

| | slices + checkpoint | all SQLite |
|---|---|---|
| trunk load | 0.00 s, mmap | **32.06 / 32.74 / 33.01 s** |
| lm_head | in `Bf`, mmap | 1.82 / 1.93 / 1.97 s |
| timed wall | **8.89 s** | **33.63 / 35.08 / 35.12 s** |
| process total | ~14.9 s | 68.94 / 71.10 / 71.45 s |
| peak RSS | 56.8 GB | 59.0 GB |

**3.8x slower on the timed region.** Two causes, and they are separable.

**The 32 s trunk load is my implementation, not SQLite.** The old path mmapped
93 slice files and let the kernel fault pages in *underneath the arithmetic*,
so the I/O overlapped the compute and cost 0.00 s of measured startup. This
version reads all 54.47 GB eagerly, single-threaded, before anything begins -
1.70 GB/s. Nothing forces that; it is the obvious first implementation and it
is the wrong one. Naming it as mine rather than as a property of SQLite.

**The expert rate fell from 0.55 to 0.23 GB/s per thread**, and that one is
real. With a single layer converted, its 15.77 GB store was largely in page
cache. With all 92 converted, 99.72 GB is read against a page cache that also
has to hold the 54.47 GB trunk buffer, so most reads genuinely come off the
disk.

### This falsifies step 19's extrapolation, in the direction it warned about

Step 19 fitted `8.89 + 0.148n` and projected **22.5 s at 92 layers**, marked
explicitly as "arithmetic on a four-point measurement, not an observation".

**Measured: 33.63 s** - and with a *faster* schema than the one that was
fitted. The extrapolation was optimistic by about 50% because at four layers
the stores were cached and at ninety-two they are not. The caution was worth
writing down, and the habit of writing it down is what makes the number
correctable instead of embarrassing.

### Three defects the build found

- **`lm_head` cannot be one blob.** 2,348,810,240 B against SQLite's 2^31-1
  ceiling. The first build died with `OverflowError: BLOB longer than INT_MAX
  bytes`. It is now one row per vocab entry, like `embed`. The proposed schema
  had said "lm_head stays one row"; that was wrong.
- **`char tab[8]`** truncated the table name `part_data` to `part_da`.
- **The descriptor limit.** 92 stores x 14 reader connections is 1,288 open
  files against a default of 1,024, which surfaced as `unable to open database
  file`. Raised at startup and the pool bounded to 16.

### The checkpoint is gone

Each layer's expert range was punched out of the safetensors **after** its
store matched a second read of the source. The checkpoint now reads **1.5 TB
apparent, 107 GB actual** - 1.39 TB reclaimed, offsets intact, expert bytes
gone.

**So `clover-k3.c` can no longer run.** This was named as unresolved in step
14 and it is now resolved by being accepted: the single-machine reference is
retired, and what proves this system correct from here is the preserved gate
md5 and the recorded baseline, not a program that can be re-run.

## Step 22 - the trunk is a stream, and holding it was the wrong shape

Step 21 loaded all 54.47 GB of trunk before starting and reported 59 GB peak
RSS. That throws away the property the per-tensor schema was built for. The
unit of consumption is not a layer, it is a **stage** - one operator
invocation against one tensor - and a stage needs one tensor.

### Measured first, from the program's own instrument

`K3_STAGE` writes a row per invocation. 18,209 rows for the gate prompt:

```
  Q stages, trunk               1,159
  X stages, experts            17,049

  distinct (layer, slot) pairs  1,159
  resolutions per pair          min 1   mean 1.00   max 1
  resolved more than once       0 of 1,159
```

**Every trunk tensor is resolved exactly once.** Fetch, use, never needed
again - the trunk is a stream. Which sets the residency floor:

```
  largest single trunk tensor    242.36 MB   (MUP, layer 0)
  largest single layer           1,171.53 MB
  whole trunk                    54.47 GB    <- what step 21 held
```

### The change

Sizes and offsets are computed up front; **bytes arrive when the layer runs
and are freed when it ends**, so the reads happen under the arithmetic rather
than before it.

| | eager | streamed per layer |
|---|---|---|
| peak trunk resident | 54.47 GB | **1,172 MB** |
| peak RSS | 59.0 GB | **5.5 GB** |
| process total | 68.94 / 71.10 / 71.45 | **56.50 / 50.84 / 59.65** |
| expert rate per thread | 0.23 GB/s | **0.33 - 0.43 GB/s** |

All runs `md5 23d162dcefb18211a7540ef12948f1eb`, token 17374.

**Peak trunk residency 1,172 MB is exactly the largest layer the catalog
predicted**, 1,171.53 MB, which is the check that the streaming is doing what
it claims and not quietly retaining something.

**Peak RSS falls 10.7x and trunk residency 46x.** The bytes read are
identical - step 43 established the trunk is read exactly once per run at
page granularity - so this is purely residency, not traffic.

A second-order effect worth naming: **the expert rate improved as well**,
0.23 to 0.33-0.43 GB/s per thread, because giving back 54 GB of buffer left
the page cache room to hold expert pages.

### What is still on the table

Per-layer is not the floor. **Per-stage is 242.36 MB**, another 4.8x, and it
is measurable from the same instrument: a slot is dead the moment the operator
that resolved it returns. Freeing it there needs the call sites to say so, and
that is not built.

Trunk streaming runs at 1.61 to 2.63 GB/s, single-threaded per layer, so 21 to
34 s of the run is still one thread reading. That is the next obvious thing
and it is not done either.

And this is still slower than the file-backed path: process total 51 to 60 s
against roughly 15 s. Bit-exact, far smaller, still slower.

## Step 23 - the per-stage floor, reached, and a correction to step 22

### First, the correction

Step 22 said the per-stage floor was **242.36 MB, another 4.8x**, as though
only the freeing was missing. That was not checked. The call sites **hoist
every weight pointer at block start**, so the real figure for scope-based
freeing was:

| block | hoisted | largest single |
|---|---|---|
| MLA attn | 232.45 MB | - |
| KDA attn | **443.86 MB** | 88.13 MB |
| dense MLP | **726.96 MB** | 242.36 MB |
| MoE | 190.05 MB | - |

Freeing at block boundaries would have given **726.96 MB**, a 1.6x
improvement, not 4.8x. Reaching 242.36 MB needed the two oversized blocks
restructured so each weight is fetched at the operator that uses it and
released when that operator returns.

Two blocks, not four: MLA at 232.45 MB and MoE at 190.05 MB are already under
the 242.36 MB that layer 0's `MUP` sets, so they were left alone - except for
releasing MLA's six at the end of its block, since otherwise they would still
be held when the MoE block runs and 232 + 190 would exceed the floor.

### The result

```
  peak trunk resident   242 MB       predicted 242.36 MB
  md5 23d162dcefb18211a7540ef12948f1eb   token 17374
```

| | eager | per layer | per stage |
|---|---|---|---|
| peak trunk resident | 54.47 GB | 1,172 MB | **242 MB** |
| process total | 68.94 / 71.10 / 71.45 | 56.50 / 50.84 / 59.65 | **37.95 / 44.65 / 45.76** |
| trunk stream rate | - | 1.61 - 2.63 GB/s | **2.90 - 3.66 GB/s** |
| peak RSS | 59.0 GB | 5.5 GB | 5.74 - 5.86 GB |

**225x less trunk resident than step 21**, and the measured peak matching the
predicted 242.36 MB is the check that nothing is silently retained.

Streaming also got *faster*, 1.61-2.63 to 2.90-3.66 GB/s. Smaller allocations
and a shorter path between fetch and use; not investigated further.

**Peak RSS barely moved**, 5.5 to 5.85 GB, and slightly the wrong way. The
trunk was never the bulk of it at this point - what remains is `lm_head` at
2.35 GB, the expert arena at 1.63 GB, and SQLite's per-connection page cache
across 1,288 connections. Reported rather than tidied.

### The connection count is the same mistake again

The descriptor question, measured rather than assumed:

```
  soft limit raised at startup   1024 -> 4096
  peak open descriptors          1,384   = 1,288 SQLite connections + ~96
```

So the raise is doing real work - 1,384 would have failed at 1,024. But
**1,288 connections is the same error as holding the trunk**: a layer is
visited exactly once, so its store is dead the moment the layer ends, and the
connections are simply never closed. Closing them per layer would take the
peak to roughly 14. Named, not fixed.

## Step 24 - profiled per stage: nothing is over-read, and my instrument was wrong first

The goal was to stop reading more than a stage needs. Measured, at stage
granularity, **nothing is over-read at all.**

`K3_NEED` records, for every slot fetched, how many of its bytes the operator
that asked for it actually touched. One row per fetch, emitted at release.

```
  trunk   54.47 GB fetched   54.47 GB used   0.00 GB never read   0.00%
          0 of 37 slot types show fetched != used
  expert  99.72 GB fetched   99.72 GB read by X
  stages  2,455, every one fetched exactly once
```

### The instrument was wrong before the program was

The first run said **0.59 GB never read, 1.09%**, all of it `GATE`, 92 fetches
of 6,426,112 B each, 100% unread. That is not waste. The router reads the gate
**directly** rather than through `Q` or `Qm`, so the hook never fired.

This is the same defect, on the same tensor, with the same number, that step
43 of the earlier arc already recorded: *"cover() was called only from Q, Qm,
X and B - not from slot_vec and not from the router's direct gate read, so all
norm weights and the 591 MB gate counted as untouched while being read every
run."* **591 MB, again.** It was in the notes and I rebuilt it anyway.

The lesson is narrow and worth keeping: **a counter hooked into the operators
misses every consumer that reads a weight in place.** In this program that is
the router, and `slot_vec`. Both are now hooked, and the figure goes to
0.00%.

### What each stage fetches

| slot | stages | bytes each | total MB | % trunk |
|---|---|---|---|---|
| G | 93 | 88,129,536 | 8,196.05 | 15.05% |
| O | 93 | 88,109,056 | 8,194.14 | 15.04% |
| Q | 69 | 88,129,536 | 6,080.94 | 11.16% |
| K | 69 | 88,129,536 | 6,080.94 | 11.16% |
| V | 69 | 88,129,536 | 6,080.94 | 11.16% |
| SH2 / SH1 / SH3 | 92 each | ~44,065,000 | 12,162.26 | 22.32% |
| EUP / EDOWN | 92 each | ~25,711,000 | 4,730.94 | 8.68% |
| QB | 24 | 28,385,280 | 681.25 | 1.25% |
| GATE | 92 | 6,426,112 | 591.20 | 1.09% |
| MGATE / MUP / MDOWN | 1 each | ~242,320,000 | 726.97 | 1.34% |
| 26 smaller slots | 1,614 | 512 - 12,681,216 | 941.54 | 1.73% |
| **total** | **2,455** | | **54,468.17** | **100%** |

Five slot types - `G`, `O`, `Q`, `K`, `V` - are **63.6% of the whole trunk**.

**Stage size spans 0.51 KB to 242.36 MB, a factor of 475,000.** That range is
the argument for the per-tensor schema on its own: a layer-granular store
would make the 512-byte stages pay the same price as the 242 MB ones.

### So what is actually left

Not over-reading. The remaining costs are all about **holding, not fetching**:

- peak residency is **242 MB**, already the largest single stage, so the floor
- **1,288 SQLite connections** are never closed, though a layer is visited once
- trunk streaming is **single-threaded per stage** at 3.62 GB/s

The first is done. The second and third are not.

## Step 25 - letting go of connections, and reading a stage with eight threads

Two changes, both the same idea as step 22: stop holding what is finished
with, and stop doing serially what the hardware will do in parallel.

**Connections.** A layer is visited once, so its store will never be asked
again. `store_close(L)` runs at layer end, which is safe because `pl_finish`
has joined every reader by then.

**Parallel stage read.** A stage is up to 242 MB and one thread was moving it.
Slots at or above 4 MB are now read by **eight threads into disjoint parts of
one buffer**, each owning its own connection, indexed by loop iteration so no
two threads share one. 54.09 GB of the 54.47 GB goes this way; the rest is
small slots below the threshold.

### Measured

| | per stage | + close + parallel |
|---|---|---|
| peak open descriptors | 1,384 | **118** |
| trunk stream rate | 2.90 - 3.66 GB/s | **5.77 - 6.34 GB/s** |
| peak RSS | 5.85 GB | **3.91 GB** |
| process total | 37.95 / 44.65 / 45.76 | **35.01 / 35.32 / 35.80** |

All runs `md5 23d162dcefb18211a7540ef12948f1eb`, token 17374. Trunk residency
stays at its 242 MB floor and fetched-versus-used stays at 0.00%.

**118 descriptors is below the 1,024 default**, so the `setrlimit` from step
21 is no longer load-bearing. It is kept because the peak is a property of the
reader count and could move.

**I predicted "roughly 14" and measured 118.** The difference is the trunk
read's eight connections, the layer database, and the file descriptors the
prefetch path still opens against the checkpoint. The prediction was the part
I had in mind, not the whole process; worth recording because a tidy estimate
that ignores everything it did not think of is how the 1,024 limit was
breached in the first place.

**The RSS drop of 1.94 GB is SQLite page cache** - 1,288 live connections each
carrying their own, now at most a few dozen.

### Where this leaves the whole arc

| | eager | per layer | per stage | + close + parallel |
|---|---|---|---|---|
| peak trunk resident | 54.47 GB | 1,172 MB | 242 MB | 242 MB |
| peak RSS | 59.0 GB | 5.5 GB | 5.85 GB | **3.91 GB** |
| process total | 68.9 - 71.5 s | 50.8 - 59.7 s | 38.0 - 45.8 s | **35.0 - 35.8 s** |

Against the file-backed path at roughly 14.9 s, the gap is now **2.35x**, down
from 4.6x. Bit-exact throughout, and 15x smaller in memory than the version
that started this section.

## Step 26 - the trunk pinned in RAM, per layer and per stage

`K3_PIN=1`. At startup, C code copies every trunk slot out of SQLite into its
own buffer - `pinbuf[L * MAXSLOT + s]`, one per layer per stage, 2,455 of
them - and `slot_ptr` then returns the buffer instead of fetching. Layers are
pinned in parallel: 93 separate databases contend for nothing but the device.

**Experts are deliberately not pinned.** Which 16 experts a position wants is
not predictable, so they keep streaming from SQLite and the page cache keeps
whatever recurs. That is what one row per expert tensor is for.

```
  trunk pinned   54.47 GB in 9.84 / 9.95 / 9.90 s   (5.48 - 5.54 GB/s)
  md5 23d162dcefb18211a7540ef12948f1eb, token 17374, every run
```

### It is a wash on this box, and the reason is worth more than the result

| | pin off | pin on |
|---|---|---|
| request wall | 34.30 / 34.84 | 32.62 / 33.60 / 34.23 |
| trunk read, per request | 8.48 s | **0** |
| expert rate per thread | 0.34 - 0.35 GB/s | **0.24 - 0.25 GB/s** |
| expert thread-seconds | ~288 | **392 / 408 / 417** |
| peak RSS | 3.91 GB | **56.87 GB** |

The 8.5 s of trunk reading is gone completely. The request time barely moves.

**The memory comes out of the expert page cache.** Expert thread-seconds rise
by 104 to 128, which across 14 readers is 7.4 to 9.1 s of wall - almost
exactly the 8.5 s the trunk stopped costing. On a 124 GB box, 54.47 GB held
by the process is 54.47 GB the 1.45 TB expert store no longer gets.

Context measured before this step, which predicted it:

```
  trunk DBs     54.77 GB total   51.76 GB resident   94.5%
  expert DBs  1450.84 GB total   73.64 GB resident    5.08%
```

**The trunk was already 94.5% in the page cache**, so pinning was never going
to save disk reads - there were barely any. What it removes is SQLite's copy
path, and what it costs is cache the experts were using.

### What this does and does not establish

**It does not show pinning is wrong.** This is a single request, so the 9.9 s
startup is paid and never amortized. A server pays it once and every later
request keeps the 8.5 s. What the measurement does show is that **the saving
is not free**: it is 8.5 s of trunk time traded for 7.4 to 9.1 s of expert
time, on this box, at this RAM.

The trade would be a clear win on a machine where the expert working set
still fits after the trunk is pinned - which is to say, it is a **memory
capacity question, not a design question.** Not tested, because there is no
second machine.

## Step 27 - the trunk without SQLite: a stage is a file

Step 24 profiled the run per stage and found the trunk fetch costing 8.813 s
of a 41.30 s request. Splitting that cost by whether the bytes came off the
block device - `read_bytes` in `/proc/self/io`, differenced across each
fetch - gave a number that made no sense:

```
  layer 62 onward:   disk_s 0.0000     mem_s unchanged
```

From layer 62 the trunk is entirely in the page cache and the fetch still
costs the same. A read that touches no disk is not a read, so what was the
time?

### It was never a read. It was a copy.

```c
sqlite3_blob_open(tcon[i], "main", "slot_data", "data", s, 0, &b);  /* walk */
sqlite3_blob_read(b, dst + off, len, off);       /* cache -> sqlite -> dst */
sqlite3_blob_close(b);
```

54.47 GB is copied into `sbuf[s]` every run whether or not the bytes came
off the device, at **6.18 GB/s across eight threads - 0.77 GB/s per thread**,
against a `memcpy` that runs about 10 GB/s on one core of this box. The
missing order of magnitude is SQLite: a `blob_open` per thread per slot
walking the overflow page list, then `blob_read` reassembling 65,536-byte
pages.

It is not page-fault cost. `mallopt(M_MMAP_THRESHOLD, 256 MB)` means slot
buffers come from a reused heap arena and are not re-faulted per layer. That
was checked before the claim was made, because the obvious guess was wrong.

### The shape that removes it

One file per (layer, stage). `make_raw_trunk.py` writes
`trunk/L<LL>/s<SS>.bin`, hashes each stage out of SQLite, hashes it again off
the disk, and refuses to continue on a mismatch.

```
stages   : 2455
bytes    : 54468171880  (54.47 GB)
all stages bit-exact against SQLite
```

2,455 files - exactly the 2,455 fetch stages step 24 counted, exactly the
54.47 GB it measured. A stage now starts at offset 0 of its own file, so
there is no page list to walk to reach it and O_DIRECT alignment is free.
That last part is the whole reason the split is per stage and not per layer.

Three read modes on the same layout, `K3_RAWMODE`: `1` streams the layer's
stages with parallel chunked `pread`, O_DIRECT where offset, length and
destination all align and a buffered fd for the tail; `2` loads all of it
into one anonymous THP arena at startup; `3` mmaps each stage and reads in
place.

### Streaming wins, and residency loses

Three consecutive gated runs per mode. Every run re-derives the logits and
is checked against md5 `23d162dcefb18211a7540ef12948f1eb`, token 17374.

| mode | rep 1 | rep 2 | rep 3 | peak RSS |
|---|---|---|---|---|
| **raw stream** | 33.92 | **18.60** | **18.63** | **3.91 GB** |
| raw, all in RAM | 28.51 | 30.99 | 31.08 | 56.85 GB |
| raw, mmap | 34.05 | 31.13 | 31.18 | 56.85 GB |
| SQLite (what it replaces) | 56.47 | 45.66 | 39.39 | 3.91 GB |

Twelve runs, twelve PASS. Steady state is reproducible to 0.03 s.

```
trunk, streamed  : 54.47 GB in 5.12 s (10.65 GB/s)  peak resident 242 MB
   was           : 54.47 GB in 8.81 s ( 6.18 GB/s)  peak resident 242 MB
```

**The trunk held in RAM is 1.67x slower than streaming it, at 14.5x the
memory.** This is step 26's finding again, now isolated: once the copy is
gone a stage read is cheap enough that residency buys nothing and still
costs the experts their page cache. Step 26 could not separate the two
because it pinned *through* SQLite.

### The cache warm: a clean negative

O_DIRECT DMAs into RAM without passing through any cache, so the stage the
operator is about to read really is cold in L1, L2 and L3 - and 99.9% of
stages (2,452 of 2,455) fit in this box's 96 MB L3. Only three exceed it,
all in layer 0. The idea that the cache should be pre-loaded per stage was
therefore worth testing rather than arguing about.

`K3_CPF`: 1 = `prefetcht2`, 2 = `prefetcht0`, 3 = touch every line.

| K3_CPF | mean of 3 | warm cost | delta vs off |
|---|---|---|---|
| 0 - off | **18.593** | - | - |
| 1 - to L2/L3 | 18.910 | 0.41 s | **+0.317** |
| 2 - to L1 | 18.933 | 0.41 s | **+0.340** |
| 3 - touch | 19.683 | 1.45 s | **+1.090** |

**The delta equals the cost. Not a fraction of it is recovered.** All twelve
PASS, so these are timings of a correct program.

The counters say why. On a gated stream run:

```
  L1-dcache-loads          405,294,609,023
  L1-dcache-load-misses     30,974,047,140    7.64%
  L1-dcache-prefetches      18,364,864,439    hardware, already running
  L2 accesses               29,735,438,841
  L2 hits                   29,262,567,196    98.41%
  fills from anywhere       30,795,098,904
  fills from DRAM            1,636,943,336    5.31%
```

Operators read each weight as one sequential stream, which is the case the
hardware prefetcher owns completely. Software prefetch re-issues work already
done; `_MM_HINT_T0` on an 88 MB stage is worse still, because lines land in a
1 MiB L1 and are evicted before the arithmetic reaches them. Mode 3 is the
honest floor: 1.45 s is what moving 54.47 GB through the hierarchy once
costs, and it buys nothing because the operator was going to move it anyway.

**Cold-in-cache was not costing anything.** There is no second read to save,
and the prefetcher already hides the first.

### What this establishes

Dropping SQLite for the trunk is worth **2.1x** at unchanged memory, and the
two intuitions that came with it - hold it in RAM, pre-load the cache - are
both measured losses. The gain is entirely in removing a copy, not in
placing the bytes anywhere.

The bottleneck has moved off memory:

```
  18.914 s elapsed    117.120 s user    19.685 s sys
  136.8 CPU-seconds / (16 threads x 18.91 s) = 45% utilisation
```

Over half the machine is idle, with trunk fetch and compute still strictly
serialised - 5.12 s of fetch that no arithmetic runs underneath.

## Step 28 - reading the next stage early, and why it took io_uring

Step 27 left the trunk fetch at 5.17 s, fully serialised: the arithmetic
stopped, a stage was read, the arithmetic resumed. Two measurements said that
was avoidable.

```
  read_bytes : 53.93 GB   effective 2.88 GB/s      over the whole run
  trunk      : 54.47 GB in 5.17 s = 10.53 GB/s     during a fetch
```

O_DIRECT bypasses the page cache, so every byte really comes off the device
every run - and the device is idle about three quarters of the time. Reading
harder will not help: `load_ram` tops out at 13.33 GB/s on the same files.
The fetch has to happen *earlier*, not faster.

### The schedule is knowable

`K3_PROV` records every slot resolution. Across 93 layers there are exactly
**3 distinct fetch orders**, one per layer kind, and the order is not
ascending by slot id - it is the equation's order.

| kind | stages | order |
|---|---|---|
| L0, KDA dense | 23 | ARN ARP MRN MRP IN_LN CQ CK CV ALOG DTB ONORM Q K V B FA FB G O POST_LN MGATE MUP MDOWN |
| KDA MoE | 28 | ...same head... POST_LN GATE GBIAS EDOWN EUP SH1 SH3 SH2 ENORM |
| MLA MoE | 22 | ARN ARP MRN MRP IN_LN QA QB KA KB G O QAN KAN POST_LN GATE GBIAS EDOWN EUP SH1 SH3 SH2 ENORM |

So stage n+1 is known before stage n is needed. `layer_order()` derives it
from the same `isMLA` expression the layer body uses, rather than a second
source of truth.

### A reader thread cannot work on this box

`K3_RA=1`: a pthread walks the order, reads each stage into its buffer, and
`slot_ptr` waits only for the stage it wants. It hid 3.31 s of a 5.93 s
fetch - and the run got slower.

| reader threads | wall | trunk rate |
|---|---|---|
| 1 | 33.02 | 2.68 GB/s |
| 2 | 24.49 | 4.77 GB/s |
| 4 | 20.68 | 7.25 GB/s |
| 8 | 19.14 | 9.00 GB/s |
| none | **18.59** | **10.67 GB/s** |

The reason is CPU, not I/O:

```
  no read-ahead   116.45 user  19.21 sys
  RA, 8 threads   167.14 user  34.26 sys      +65.7 CPU-seconds, -0.55 s wall
```

One thread cannot reach the device, eight can, and there are none to spare -
16 compute threads and 14 expert readers already share 16 cores. Every
variant that moved threads from compute to the reader was worse still
(omp12/nt4 21.23 s, omp14/nt4 20.94 s).

A side finding worth keeping: `OMP_WAIT_POLICY=passive` drops user from
116.62 to 64.34 with byte-identical output. **Nearly half of all "used" CPU
in this arc is libgomp spinning at barriers.** It costs 0.35 s of wall to
remove, so the spin is buying latency - but any utilisation figure quoted
from `user` time in earlier steps was overstated by roughly 2x.

### io_uring: the same depth with no threads

Raw syscalls against `linux/io_uring.h`, so the build stays one file against
`-lm -lsqlite3`. The layer body is serial, so the whole thing needs no locks:
submit whole stages at layer start and after each `slot_drop` frees budget,
reap completions only when a stage is actually wanted.

**It passed the gate only on the second attempt, and the first attempt looked
like a triumph.** 11.51 s, a 38% win - and md5 `3542ac0e`, then `397b1ad3` on
the next run. Two different wrong answers. Three bugs:

1. **278 stages are smaller than 4096 bytes** (512x138, 1024x24, 3072x24,
   3584x92). For those the O_DIRECT-aligned length is zero, so no read was
   ever issued - but `sbuf[s]` had already been assigned, and `slot_ptr`
   handed the operator raw `malloc` memory. The run's own output said
   `278 never submitted` and I had read past it.
2. **Short reads were not handled.** `IORING_OP_READ` may return less than
   asked; only `res < 0` was checked.
3. **A ring-full abandon left its SQEs queued**, because `ur.pending` was not
   rolled back.

A faster wrong answer is the most dangerous result there is. The only reason
it was caught is that every run is gated on the logits md5, not eyeballed.

### What it buys

Three runs each, all PASS md5 `23d162dcefb18211a7540ef12948f1eb`.

| mode | wall, 3 reps | mean | user | sys | waited on fetch |
|---|---|---|---|---|---|
| none | 18.59 / 18.60 / 18.63 | 18.607 | 114.97 | 19.73 | 5.15 s serial |
| **io_uring 512 MB** | 17.72 / 17.65 / 17.65 | **17.673** | 115.51 | 20.51 | **0.01** |
| io_uring 2048 MB | 17.54 / 17.88 / 17.60 | **17.673** | 113.59 | 20.51 | 0.02 |
| reader thread | 18.94 / 18.99 / 19.14 | 19.023 | 166.48 | 35.32 | 2.53 |

**-0.934 s, -5.0%, reproducible to 0.12 s, at no CPU cost.** 1,983 of 2,455
stages are ready before the operator asks; the fetch is off the critical path
entirely, 5.15 s of waiting down to 0.01 s.

The honest part: **removing 5.15 s of fetch bought 0.93 s of wall.** The rest
was already being overlapped with expert I/O stalls, and the trunk read-ahead
now competes with the expert readers for the same device. Budget past 512 MB
buys nothing.

`K3_RA` defaults to 0, so the committed default is unchanged.

## Step 29 - routing is a function of the prefix, and that is measurable

Step 28 could read the trunk ahead because every trunk stage is known at
startup. The experts are not: which 16 a position wants comes from a router
that needs that position's hidden state, so the dependency runs

```
experts(L-1) -> h(L) -> attention(L) -> x(L) -> J(L) -> experts(L)
```

and there is no point in it where layer L+1's expert ids exist before layer L
finishes. That is why the expert path could not be prefetched the way the
trunk was.

Unless the routing is knowable some other way.

### The equation

For layer `l`, position `t`, with `h` the residual after that layer's attention:

```
x_t     = RMSNorm(h_t) * w_post_ln
s_t,e   = sigmoid( <g_e, x_t> )         g_e = row e of S_GATE, int8 x fp32 scale
J_t     = top-16 over e of ( s_t,e + b_e )      b = S_GBIAS, selection only
p_t,j   = s_t,J[j] / sum_k s_t,J[k]
```

`x_t` depends on every token at or before `t`, through attention. So the
routing *should* be a pure function of the prefix. That is a claim, and it is
falsifiable.

### Three claims, one control

`K3_PROV` writes one S record per (layer, position, rank): the chosen expert
and its weight. Five runs, comparing identities only.

Prompts 22 and 23 share their first seven tokens and differ at index 7:

```
22  113476,387,276,10484,318,15383,316,28202,...   "...France and Berlin is..."
23  113476,387,276,10484,318,15383,316,31082,...   "...France and Rome is..."
```

```
claim R  - determinism: same prompt, two runs
  p1a vs p1b, every (layer,position)           identical  460 /  460   ALL MATCH

claim P0 - position 0 depends only on token 0   (p1 and p2 both start 1008)
  p1 vs p2, position 0, all layers             identical   92 /   92   ALL MATCH

claim PX - causal prefix: p22/p23 share 0..6
  position  0 .. position  6                   identical   92 /   92   ALL MATCH  (x7)
  position  7 .. position 11                   identical    0 /   92   first differs at (1,7)

control: p1 vs p2 at position 1 (token 1 differs)
  p1 vs p2, position 1, all layers             identical    0 /   92   first differs at (1,1)
```

644 of 644 (layer, position) pairs match across the shared prefix - same 16
experts, same rank order, same weights. Position 7 diverges at layer 1 and
never recovers. The control rules out trivial matching.

**Routing at position t is a pure function of tokens 0..t.** Measured.

### What a known prefix unlocks

One expert is 17,547,264 B = 3 x (5,505,024 + 344,064).

| case | selections known before layer 0 | share of expert bytes |
|---|---|---|
| p1, 5 tokens, first token only | 1,472 / 5,683 | **25.9%** - 25.83 of 99.72 GB |
| p22, 12 tokens, first token only | 1,472 / 8,688 | 16.9% - 25.83 of 152.45 GB |
| p22, **7-token prefix known** | 6,395 / 8,688 | **73.6%** - 112.21 of 152.45 GB |
| p22, whole prompt seen before | 8,688 / 8,688 | 100% |

1,472 is exactly 92 layers x 16 experts: position 0 draws 16 distinct experts
per layer, in every layer.

### The table cannot be precomputed, only cached

To know position 0's routing at layer L you need x_0 at layer L, which means
running layers 0..L-1 for that token, including their expert arithmetic -
about 25.8 GB of reads per first token. Across 163,840 tokens that is ~4.2 PB.

So this is **a cache populated by use, not a table built in advance.** For a
server with repeated system prompts or shared openings, that is the right
shape regardless.

### What is not yet established

Knowing the ids early removes the *dependency*, not the *bytes*. All 99.72 GB
still has to be read. Whether that converts to wall time depends on whether
expert I/O is the binding constraint, and right now I/O and compute sit at
near parity - roughly 7.3 s against 7.0 s. Spreading the reads earlier helps
only if the device is not already saturated. **Not measured. Not claimed.**

## Step 31 - putting the weights back, and what the arc actually cost

Asked to clear the box down to the open weights and run `clover-k3.c`, the
first thing to check was whether that was safe. It was not.

```
/root/k3model     apparent 1.5T   actual 107G     92 of 112 files holed
/srv/k3/db        apparent 1.4T   actual 1.4T
```

Step 13 punched holes in the checkpoint to reclaim space. The shards kept
their lengths but lost 1,446.45 GB of blocks - 92.7%. **The SQLite store was
the only intact copy of the expert weights**, and deleting it would have
destroyed the model. Worse, `clover-k3.c` reads experts from those shards, and
a hole reads as zeros with no error: it would have run and produced a wrong
answer rather than failing.

### The holes are exactly the ranges the index names

Before writing a byte, a read-only probe compared `SEEK_HOLE` against the
expert ranges in `eqidx.bin`:

```
layer 92   expert ranges in index : 15.72 GB     holes in the shard : 15.72 GB
           holes NOT named by the index : 0
```

Zero unnamed holes on every layer checked, so everything missing was
recoverable. Each shard is one layer: 16.99 GB apparent, 15.72 GB of experts
punched, and the 1.27 GB that survived is the trunk - which was never
punched.

### Restore in place, verified, then reclaim

Order, per layer, and not negotiable:

```
restore -> verify every tensor sha256 -> move store aside -> gate -> delete
```

Peak extra disk is one layer, ~15.7 GB, because the store goes as soon as the
shard has the bytes. Layer 92 was done alone first: 5,376 tensors written, all
verified, store moved aside, and `csk3` run with **91 of 92 layers from SQLite
and layer 92 from the restored checkpoint** - md5 `23d162dc`, which proves the
restored bytes are right end to end and not merely equal to the store.

```
restore exit 0
final gate, every layer from the checkpoint:  PASS 23d162dcefb18211a7540ef12948f1eb
stores left: 0
checkpoint: apparent 1561.00 GB   actual 1561.00 GB   still holed: 0 files
```

1,446 GB written back, 494,592 tensors verified byte-for-byte, 11 gates along
the way.

### clover-k3.c on the restored weights

```
prefetch ON  trunk mmap
total wall time       : 14.44 s
emitted token         : 17374   engine emitted 17374   MATCH
PASS  logits md5 23d162dcefb18211a7540ef12948f1eb
```

Cold, caches dropped. Which forces an honest comparison:

| | cold | warm |
|---|---|---|
| **clover-k3.c, checkpoint + mmap** | **14.44** | - |
| clover-server-k3.c, raw trunk + SQLite experts | 32.81 | 17.67 |
| clover-server-k3.c, all SQLite | 56.51 | 39.39 |

**The original was faster than anything the SQLite path reached, cold or
warm.** The 41.30 -> 17.67 s improvement recorded in steps 27 and 28 is real,
but it was recovering ground the move to SQLite had lost, not beating the
starting point. The benchmark should have been `clover-k3.c` from the
beginning; it was the previous SQLite step instead, and that flattered every
result in the arc.

What survives as durable is not a speedup. It is step 29: **routing at
position t is a pure function of tokens 0..t**, measured 644/644 with a
control, which holds for any implementation of this model.

### Final state of the box

```
kept:    /root/k3model      1.5 TB    the weights, whole
         /root/k3trunk_i8    51 GB    trunk.bin, which clover-k3.c reads
         /srv/k3/model               symlink the index resolves through
         /opt/clover-k3      43 MB    clover-k3.c, build.sh, gate.sh, eqidx.bin
removed: /srv/k3/db, /srv/k3/raw, /srv/k3/trunk, /root/k3raw, scratch binaries
free:    157 GB
```

Re-gated after the deletion: 14.51 s, PASS.

## What is measured, and what is not

Measured on one box: the dedup negative, the batching curve, every SQLite and
O_DIRECT number, single-pod cold start and steady state, the contention
curve up to four pods, the 93-way trunk split, ext4 hole punching, the
slice build running bit-exact at the same speed as the reference, the
SQLite expert path bit-exact at a linear 0.148 s per converted layer, and the
whole model - trunk, experts, embedding, lm_head, vocabulary - running from
SQLite bit-exact, with the trunk streamed per stage at 242 MB peak residency
instead of 54.47 GB.

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
