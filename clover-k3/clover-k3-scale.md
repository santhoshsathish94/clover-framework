# Scaling this, and why more GPU is the wrong lever

Kimi K3 is 1.5 TB. The obvious reaction is to reach for accelerators. The
measurements say that is the wrong lever for **this** model's shape, and they
also say what the right one is.

Everything below is measured on one box unless marked otherwise.

---

## 1. What the equation actually bought

It did not remove arithmetic. `clover-k3` reproduces the reference float for
float — 79,742,816 values identical, gate md5
`23d162dcefb18211a7540ef12948f1eb` — so it cannot be skipping work. Both it and
the engine issue the same **5,683 distinct expert requests** and read the same
**99.72 GB** of expert weights.

What it bought is that **the cost became countable**. Because the model is
written as ten named operators rather than an opaque call graph, the program
reports exactly what each one costs, and the total can be derived two
independent ways that agree:

```
Q  int8 projection   45,765,600 outputs x 11,753.0 flops/out =   537.9 GFLOP
X  mxfp4 expert proj 71,598,080 outputs x  6,790.7 flops/out =   486.2 GFLOP
                                                       total = 1,024   GFLOP
```

The engine, by contrast, reports no arithmetic figure at all — only I/O. That
is not a criticism of it; it simply was not built to answer this question.
Nobody could state K3's arithmetic cost without first writing the model down
the way the equation does.

**That number is the whole argument**, and everything in the rest of this
document follows from it.

---

## 2. The number: 10.3 FLOP per byte

One five-token prompt:

```
arithmetic        1,024 GFLOP
bytes off disk       99.86 GB
                 -------------
required          10.3 FLOP per byte
```

Ten floating-point operations per byte of weight is an **extremely low
arithmetic intensity**. It is low because of what an MoE layer is: every
expert weight is fetched, used for one or two dot products, and discarded.
Measured here, 77.6% of expert decodes serve exactly one position, and the
mean is 1.295. There is almost no reuse to amortise the fetch against.

### What that ratio demands of a machine

Intensity on its own says nothing about hardware. Turn it into a rate:

```
disk ceiling, measured              14.6 GB/s
x required intensity                10.3 FLOP/byte
                                  --------------
arithmetic needed to keep up       150 GFLOP/s

this machine actually delivers     139 GFLOP/s   (1,024 GFLOP in 7.36 s)
```

Sixteen ordinary CPU cores are **within about 8% of exactly enough**. The
processor is neither the bottleneck nor wasted — it is matched to the storage
in front of it. That is what "I/O bound at 94% of the device ceiling" looks
like when written as a ratio.

Note this is not the same as dividing the run's own FLOPs by its own bytes;
that would give 10.3 whatever the hardware, because the time cancels. The
comparison above is between the arithmetic rate the storage *demands* and the
rate the processor *supplies*.

---

## 3. Why adding GPU does not help

A large accelerator is a machine with an enormous FLOP-to-bandwidth ratio. It
pays off when a workload reuses each loaded byte hundreds of times. This one
reuses each byte about ten times.

**The model does not fit.** 1.5 TB against 80-192 GB of on-device memory. So
the weights live somewhere else and cross an interconnect per token. Apply the
same arithmetic as above to a link delivering ~60 GB/s:

```
  60 GB/s  x  10.3 FLOP/byte  =  618 GFLOP/s of useful work
```

That is the ceiling on what such a device could be doing, however fast its
arithmetic units are. A large accelerator's capacity is one to three orders of
magnitude beyond 618 GFLOP/s, so it waits.

**Holding it all on-device means ~19 cards** at 80 GB each, purely for
capacity, to run arithmetic that sixteen CPU cores already keep pace with.

**The access is data-dependent.** Which 16 of 896 experts a position needs is
decided by the router at run time, from that position's own activations. You
cannot pre-place a useful subset, because the subset is not known until the
layer runs. The only placement that never causes a fetch is holding the
layer's **entire** 896-expert set — 15.72 GB.

That last point is the one that chooses the architecture.

### One caveat, stated plainly

None of this has been measured on a GPU. The interconnect figure above is a
published specification, not something observed here. What is measured is the
arithmetic intensity, and intensity is a property of the model rather than of
the hardware it runs on.

---

## 4. What the structure suggests instead

A layer is the unit at which the expert set is **bounded and fixed**. That is
the natural cut.

```
  client machine        head, layer 0, tail        5.87 GB, no experts
        |  vector
  router                which layer runs next      no weights
        |  vector
  pod 1 ... pod 92      one layer each             ~16.2 GB per pod
```

| | measured |
|---|---|
| client | 4.70 GB of non-layer tensors + layer 0's 1.172 GB trunk slice |
| pod | 0.423-0.635 GB trunk slice + 15.72 GB of that layer's 896 experts |
| whole expert pool | 1.446 TB, of which one prompt touches 99.72 GB (6.9%) |

**Layer 0 goes with the client because it is the only layer with no experts** —
`isMoE = (L >= 1)`, a dense MLP. It is also the largest trunk slice at
1.172 GB, since it carries the 727 MB dense MLP the MoE layers do not have.
Layer 1 cannot join it; the 494,592 expert records divide as 5,376 per layer
for layers 1 to 92 and zero for layer 0.

**A pod is a commodity machine.** Nothing in it reaches another pod's storage,
and no pod ever reads the 1.5 TB whole. The fetch that dominates the
single-machine run does not happen at request time at all, because the bytes
are already there.

---

## 5. What moves instead

Not token ids. Each layer needs the residual for every position plus every
snapshot taken so far, and snapshots accumulate where `L % 12 == 0`:

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
| 64 | 902.82 MB | 9.81 MB |

The expensive thing stops moving; the cheap thing moves.

### What would sink it

At 64 positions those hops total 902.82 MB against a 47 ms stage. Over shared
memory on one node that is free. **Across nodes at 10 Gb/s it is about 0.7 s
per prompt in network alone** — which would dwarf the compute it was meant to
distribute. Co-scheduling is a constraint, not a preference, and the deepest
layers are the expensive hops because the snapshot stack has grown.

---

## 6. Where concurrency actually sits

Memory is not the obstacle. Measured from a running process:

```
Rss        53,376,876 kB    of which
Pss_File   53,060,951 kB    a read-only mapping of trunk.bin
Pss_Anon      313,520 kB    0.31 GB, the only genuinely private part
```

The trunk is a **file mapping served from page cache**, so a second process on
the same box shares the same physical pages. N prompts cost ~53 GB once plus
about 0.3 GB each, not N x 53 GB.

But the expert path uses O_DIRECT, which bypasses the page cache, so
concurrent prompts share nothing there and each still pulls its own 99.72 GB
from a device already running at 98-99%. **Concurrency on one box is expected
to divide throughput rather than multiply it — and that has not been
measured.** It is the obvious next experiment, and the pod shape exists partly
because it sidesteps the question.

---

## 7. What is established, and what is not

**Measured:** the arithmetic count and its intensity, the device ceiling of
14.6 GB/s and the 94% of it already achieved, all the sizes in section 4, the
hop payloads in section 5, the shared/private memory split in section 6, and
three separate failed attempts to speed up the compute side — io_uring 7-11%
slower, cache preload a wash to 0.65 s worse, all 32 threads 2 s slower than
16.

**Not measured:** anything on a GPU, anything across nodes, concurrent load,
and any cost or energy figure. The topology in section 4 is a shape derived
from sizes, not a running system. An earlier single-process version on a
per-layer data layer did run and did reproduce the gate; it is in git history
at `abdcde7^:clover-server-k3/`.

The step-by-step record, including everything tried and abandoned, is in
[`../k3-analysis/clover-scaling-architecture.md`](../k3-analysis/clover-scaling-architecture.md).
