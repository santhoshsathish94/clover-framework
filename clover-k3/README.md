# clover-k3

A single-file C implementation of the Kimi K3 forward pass, written from the
equation rather than from the engine, and kept bit-exact against a preserved
baseline through every optimization.

It is not a re-implementation of `kimi-k3-in-c`. It is the equation in
`work-in-progress/k3-model-equation.md` executed directly, which is why it can
be read end to end and why every operator can be timed and attributed
separately. The measurement record lives in
[`work-in-progress/k3-equation-solution.md`](../work-in-progress/k3-equation-solution.md).

**This is the single-machine version.** One process, one box, the whole model
in one address space. [`clover-server`](../clover-server/) is the same
equation cut across a client, a server and 93 layer processes; it exists for a
different reason and neither replaces the other. This one answers *is the
equation right*, and because it does, it is also the **reference** - the only
thing that can say whether the decomposed version is correct. It is therefore
kept unchanged.

## What is here, and what is deliberately not

This folder holds only what is ours. The model and the trunk belong to K3 and
are left where they are.

| file | what it is |
|---|---|
| `config.env` | the only file here that names a location |
| `clover-k3.c` | the program, ~2,270 lines, no dependencies beyond libc, libm and OpenMP |
| `dump_st_model.py` | locates the five non-layer tensors in the checkpoint |
| `dump_eqidx.py` | flattens trunk.json, st_model.json and the shard headers into one binary index |
| `prompts.tsv` | the 34 prompts used for the correctness campaign |
| `build.sh` `gate.sh` `ab.sh` | build, correctness gate, and the A/B harness |

**Not here, on purpose:**

- **The checkpoint** (`*.safetensors`, ~1.45 TB). K3's own.
- **The trunk** (`trunk.bin` 54.47 GB, `trunk.json`). Also K3's own - it is built
  by the engine's `tools/pack_trunk.py` and `tools/int8_trunk.py`, and the
  engine runs from it too. Nothing here reproduces it.
- **`eqidx.bin`** (21.9 MB) and **`st_model.json`**. Both are generated, and both
  embed absolute paths to the checkpoint, so they are specific to the machine
  that built them. `build.sh` makes them in `build/`.
- **The baseline logits** (`eq_c_logits.baseline.bin`, 684 KB). The gate is the
  md5, which is a constant in `gate.sh` and `ab.sh`, so the file itself is only
  needed for a float-level diff when something fails. It is preserved on the
  working box at `/root/k3raw/eq_c_logits.baseline.bin`.

## Configuration

Every path lives in `config.env`, and nothing else in the folder names a
location. Each entry yields to an already-set environment variable, so a
one-off run can override any of them without editing the file.

| | |
|---|---|
| `K3_MODEL` | directory of the `*.safetensors` shards |
| `K3_TRUNK` | directory holding `trunk.bin` and `trunk.json` |
| `K3_TRUNKPATH` | which copy of the trunk to read; point it at tmpfs and the load cost disappears |

The program takes `K3_INDEX` and `K3_TRUNKPATH` and **fails if they are not
set** rather than defaulting. A default would let a run silently pick up
another machine's index, which is the kind of thing that produces a confident
wrong number.

## Building

Edit `config.env` to point at your checkpoint and trunk, then:

```sh
./build.sh
./gate.sh
```

`NPOS` is a compile-time constant, so a build is for one prompt length:
`NPOS=64 ./build.sh`. Everything generated lands in `build/`, which is
gitignored; nothing in the repo is modified.

`-ffp-contract=off` is part of the verified build, not a preference. Without
it GCC fuses the multiply and add into an FMA, which rounds once instead of
twice, and the logits move by 2-3 ULP. This cost one bisect to learn (step 33)
after the specification had already said it (step 41).

## The gate

```
The capital of France is  ->  17374 (' Paris')
logits md5 23d162dcefb18211a7540ef12948f1eb
```

Every change in the arc had to reproduce those bytes. A timing result from a
build that has not passed `gate.sh` means nothing, and `ab.sh` refuses to
report a winner without it.

## Running

```sh
. ./config.env
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
K3_PREFETCH=4 K3_TRUNKRAM=0 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
K3_INDEX=build/eqidx.bin K3_IDS=1008,10484,318,15383,387 \
  build/clover-k3
```

Thread pinning is not optional on a dual-CCD part: unpinned runs vary by
seconds, and every measurement taken before it was adopted had to be redone.

### Switches

There are 27. The ones that change what the program does:

| | |
|---|---|
| `K3_IDS` | comma-separated token ids, count must equal `NPOS` |
| `K3_INDEX` | path to `eqidx.bin`; required |
| `K3_TRUNKPATH` | path to `trunk.bin`; required, and from `config.env` |
| `K3_TRUNKRAM` | 1 reads the trunk O_DIRECT into anonymous memory instead of mapping it |
| `K3_PREFETCH` | 0 off, 4 the pipelined O_DIRECT arena (best) |
| `K3_NREADER` | reader threads, 14 |
| `K3_HUGE` | 1 THP, 2 hugetlb |
| `K3_PFXSAVE` `K3_PFXLOAD` `K3_PFXN` | prefix reuse: save or reuse the state for a leading prompt, bit-exact, 2.57x at 64/48 |
| `K3_LOGITS` | where to write the logits |

The rest are instrumentation and are off by default: `K3_PROV` (which trunk
slot and expert block each operator took), `K3_LSTAT` (per-layer value
statistics), `K3_STAGE` (one row per operator invocation), `K3_COVER` (reads
per trunk page), `K3_VALUE`, `K3_DUMPROUTE`, `K3_DUMPSEL`, `K3_DUMPLAY`,
`K3_DUMPRES`, `K3_HSTAT`, `K3_SITUSTAT`. Turning all of it off was measured and
makes no difference to wall time (step 47).

## What it costs

On a Ryzen 9 7950X3D, 124 GB, NVMe RAID1, trunk resident in `/dev/shm`:

| | 5 tokens | 64 tokens |
|---|---|---|
| wall | 8.73 s | 41.54 s |

Where the time goes, over 34 prompts: **experts 79.7%, trunk 16.3%, tables
0.4%, arithmetic 1.9%**. The program is not compute-bound; it is bound by
reading 99.72 GB of expert weights per run, which is 6.9% of the 1446 GB of
routed experts and 100% of the trunk.

## A caution about measuring it

The run-to-run spread at 64 tokens is about 0.1 s within a session and up to
2 s across sessions. Three separate times in this work a single run produced an
exciting number that evaporated under n=4. `ab.sh` interleaves the arms for
that reason. Check the machine is idle before believing anything.
