# Kimi K3 on one CPU: what was actually measured

Every number here was produced on one rented machine between 22 and 23 September 2026,
running Fareed Khan's [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c)
(Apache-2.0) against the released 2.78-trillion-parameter checkpoint. Nothing here
reimplements that engine and no model weights are redistributed.

This document is the summary. The chronological working record, including the predictions
that turned out wrong and the methods that had to be thrown away, is in
[CONTEXT-kimi-k3-benchmark.md](CONTEXT-kimi-k3-benchmark.md). Where the two disagree, the
raw logs decide; every figure below was re-derived from them rather than copied forward.

## The machine, including what is wrong with it

Hetzner AX102: AMD Ryzen 9 7950X3D, 16 physical cores / 32 logical across two CCDs,
124 GiB RAM, two 1.92 TB NVMe drives in RAID1, Ubuntu 24.04.

**One of the two drives negotiates PCIe x2 instead of x4.** The boot log reports 31.506
Gb/s available against 63.012 capable, and under load that drive shows 28.8 ms latency
against the healthy drive's 13.9 ms. This matters for reading the numbers below: a change
that improves read concurrency may be helping partly because this machine's storage is
degraded, and may do less on a healthy one. It is stated here rather than left for a reader
to discover.

Runs cannot overlap. A single run holds about 110 GB of the 124 GiB, so every measurement
below is sequential, which is also why the campaigns take hours rather than minutes.

## 1. What the always-resident weights are stored as

The model's "trunk" — the part used on every token — was packed three ways and each was run
end to end. One run each, so these are indicative rather than replicated.

| trunk format | packed size | decode | prefill | whole run | peak RSS |
|---|---|---|---|---|---|
| bf16 | 108.81 GB | 5.311 s/token | 152.77 s | 487.4 s | 118.93 GB |
| int8 | 54.47 GB | 4.133 s/token | 116.26 s | 376.6 s | 64.57 GB |
| MXFP4 | 28.94 GB | 3.965 s/token | 108.07 s | 357.8 s | 39.13 GB |

Two separate things are worth taking from that table, and they point in opposite directions.

**Halving the bytes bought most of the available speed; halving them again bought almost
none.** bf16 to int8 is 22.2% faster and frees 54 GB. int8 to MXFP4 halves the weights
again and returns 4%. By that point the processor, not the memory path, is the limit, so
buying smaller weights stops buying time. An earlier version of this work concluded
"quantization of the trunk is a dead end for speed"; that was wrong as written, and the
correction is that it is a dead end *past int8, on this CPU*.

**The cost is not free, and it is not visible in a speed table.** Output is identical to
bf16 for the first 21 generated tokens under int8, and for only 3 under MXFP4. That is a
divergence measurement on a single prompt, not a quality measurement — no perplexity and no
benchmark was run, so nothing here says whether the divergent output is worse, only that it
is different.

**Trunk size is not peak RSS.** 108.81 GB is the packed bf16 trunk; 118.93 GB is the peak
resident set of the whole run, which also holds embeddings and lm_head (4.70 GB), recurrent
state, buffers and the KV cache. An earlier draft of this work quoted the second figure as
the first in four places. Same conclusion, wrong number, now corrected.

## 2. Engine changes offered upstream — [PR #67](https://github.com/FareedKhan-dev/kimi-k3-in-c/pull/67)

Concurrent chunked expert reads, a batched bf16 matmul that is bit-identical to the serial
kernel, and fewer reads in the KDA recurrence. Measured against that project's current
`main` at `a2ad8e5`, three runs per arm, interleaved so drift cannot land on one arm, every
run reported.

| run | decode s/token | prefill s | whole run s |
|---|---|---|---|
| upstream r1 / r2 / r3 | 5.6337 / 5.6256 / 5.6310 | 168.42 / 168.53 / 168.46 | 523.4 / 522.9 / 523.3 |
| ours r1 / r2 / r3 | 5.3106 / 5.3094 / 5.3106 | 152.65 / 151.32 / 151.50 | 487.2 / 485.8 / 486.1 |

| metric | upstream | ours | gain |
|---|---|---|---|
| decode | 5.6301 (sd 0.0041) | 5.3102 (sd 0.0007) | **5.68%** |
| prefill | 168.47 (sd 0.056) | 151.82 (sd 0.722) | **9.88%** |
| whole run | 523.20 (sd 0.265) | 486.37 (sd 0.737) | **7.04%** |

Steady-state decode turns out to be one of the most reproducible things this machine does:
a standard deviation of 0.0041 s against a 5.68% effect. The project's own CONTRIBUTING
warns of a 33% run-to-run noise floor, which appears to describe prefill and wall clock
rather than steady-state decode.

What this does not separate: three changes were measured as a bundle, so if one of them
contributes nothing, these runs cannot tell which.

## 3. The default nobody had measured — [PR #68](https://github.com/FareedKhan-dev/kimi-k3-in-c/pull/68)

The engine never selected a thread count, so OpenMP chose: one thread per *logical* CPU. On
this part that is 32 rather than 16, and there was no command-line way to change it. The
project's own ROADMAP had predicted the consequence — "on memory-bound workloads throughput
often *declines* past a point. Unknown here" — and its TUNING doc described the default as
"your core count", which is not what OpenMP does on an SMT machine.

Two arms, three runs each, interleaved, all reported. Binaries differed only by the fix.

| run | decode s/token | prefill s | whole run s | involuntary ctx switches |
|---|---|---|---|---|
| 32 logical r1 / r2 / r3 | 7.3230 / 7.3752 / 7.2859 | 172.74 / 177.63 / 169.40 | 634.1 / 642.3 / 628.4 | 127,172 / 136,467 / 113,619 |
| 16 cores r1 / r2 / r3 | 5.6262 / 5.6352 / 5.6319 | 158.94 / 159.43 / 163.29 | 513.4 / 514.4 / 518.1 | 36,836 / 37,663 / 38,231 |

| metric | 32 logical | 16 cores | change |
|---|---|---|---|
| decode | 7.3280 (sd 0.0449, spread 1.22%) | **5.6311** (sd 0.0046, spread 0.16%) | **−23.16%** |
| involuntary ctx switches | 125,753 | 37,577 | −70.1% |

**Counting cores correctly is worth 23.16% of decode time**, and the mechanism is measured
rather than asserted: 3.35x the involuntary context switches is the scheduler moving threads
across two CCDs and losing locality on a working set far larger than any cache. The fix also
cuts variance — spread falls 1.22% to 0.16% — so the shipped default was less repeatable as
well as slower.

One property worth knowing before anyone benchmarks this: **the 32-thread arm degrades over
a run rather than paying a constant cost.** Its mean over the first 23 decode steps is 5.5%
better than over all 63, while the 16-thread arms are flat to 0.2%. A short benchmark
understates the problem.

## What none of this establishes

- **Nothing on a GPU.** Every figure is CPU-only. CPU behavior does not transfer: the MXFP4
  result in particular is a statement about this memory path, not about the format.
- **One machine, one prompt, one memory budget**, with a known-degraded PCIe link on one of
  its two drives.
- **No quality claim.** Output divergence under quantization was counted; whether the
  divergent output is worse was not measured.
- **No reproduction of anyone else's published figures.** An earlier draft quoted a
  llama.cpp placement number that had never been run here; it was removed rather than
  softened.
- **Acceptance is not validation.** Both pull requests are open and may never be merged.
  That would say something about another project's roadmap and hardware, not about whether
  the measurements are real.

## Where the evidence is

310 files, 17 MB, captured 23 September 2026 and verified file-by-file against a manifest
generated on the machine before transfer: `k3-evidence-full-20260923.tgz`, sha256
`17609521a57e32359565d37a015ef700ace27ffcc8b858f3bb52d01dc75214ef`. It holds every run log
and result JSON, the profiling directories, all 48 harness scripts, and a provenance file
recording the host, the CPU, and the exact commit each binary was built from.

It is **kept outside this repository** deliberately. It is raw measurement data rather than
framework material, and the machine that produced it bills until it is canceled, so the
point of capturing it was to stop that machine being the only copy.
