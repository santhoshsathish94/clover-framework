# Comparison with kimi-k3-in-c, 2026-10-08

Against [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Fareed Khan).

**His figures below are published, not reproduced.** We cannot run his binary here: it reads
the raw expert checkpoint and this host holds the prepared form only. Nothing in this file
is a head-to-head measurement, and the hardware differs materially.

| | this engine | kimi-k3-in-c |
|---|---|---|
| machine | Ryzen 9 7950X3D, **32 cores** (16 used) | **124 cores** |
| RAM | 124 GB | 128 GB+ row |
| storage | 2x KIOXIA KCD8XRUG1T92, PCIe 4.0 x4, raid1 | "fast NVMe drive" |

## His published numbers

From his README, one machine, 124 cores, fast NVMe:

| RAM | time per token |
|---|---:|
| 8 GB | 26.5 s |
| 32 GB | 24.2 s |
| 64 GB | 19.8 s |
| 128 GB+ | **5.6 s** |

Demo captures in the same README, which he notes ran on a slower drive:

```
"The capital of France is" --gen 8    8 tokens in 261.5 s, 32.69 s/token, peak RSS   8.24 GB
"def fibonacci(n):"        --gen 28  28 tokens in 299.3 s, 10.69 s/token, peak RSS 127.92 GB
```

## Our measured numbers

Six prompts, six output tokens each, resident process, `CLOVER_SNAPFOLD=score
CLOVER_SNAPSHOT=layer`, startup excluded (~9.5 s, paid once).

| | seconds |
|---|---:|
| prompt pass, 4-6 tokens | **5.97 - 9.30** |
| each output token | **~3.66**, flat |
| France, 5 in + 6 out | **24.30** |
| "Gravity is", 2 in + 32 out | **119** |

## Side by side, with the caveats attached

| | ours, measured | his, published at 128 GB+ |
|---|---:|---:|
| per output token | 3.66 s | 5.6 s |
| 5-token prompt pass | 6.0 - 9.3 s | ~28 s if charged per token |
| France, 5 in + 6 out | 24.3 s | ~62 s |
| "Gravity is", 2 in + 32 out | 119 s | ~190 s |

Roughly **1.5x on output tokens** and **2.5-4x on prompt processing**, against a published
number on a machine with four times the cores.

The prompt-pass advantage is structural, not tuning: all input positions go through one `Qm`
call at `T=5`, so each weight byte is read once and used five times --
**9.99 FLOP/byte against 2.00 FLOP/byte** for a single position. Measured:

```
prompt pass, 5 positions : Q moved 53.83 GB -> 38.8 GB/s, 388.1 GFLOP/s, 9.99 FLOP/byte
output token, 1 position : Q moved 53.83 GB -> 39.2 GB/s,  78.4 GFLOP/s, 2.00 FLOP/byte
```

Identical weight traffic at the same ~39 GB/s; only the useful work differs.

**"Minutes versus seconds" overstates it.** Our own gravity answer took 119 s. The defensible
claim is 1.5-4x depending on the input/output ratio, unverified, on different hardware.

## Output fidelity: his capture settles a question we had open

His France demo generates:

```
 Paris.",
+            "The Eiffel
```

That is what **our fold with the winner-takes-all override disabled** produces, token for
token (`17374 20829 10 427 414` = ` Paris.",\n+            "`). Three independent sources
agreed, so the override has been **removed from the code entirely**:

1. the stored observations match bit-exactly on all 92 layers (7,360 / 7,360 steps)
2. the per-layer divergence cliff sits at exactly layer 46, where it fired
3. his reference implementation produces the same continuation

It used to be on by default at layer 46 and produced ` Paris. The Eiffel` -- more readable,
and not what the reference produces. Earlier notes called it the correct setting on the
strength of the text reading better; that was wrong. Removal was verified output-neutral
over six prompts x six output tokens, all 36 ids identical.

| | stored-observation match | matches kimi-k3-in-c output |
|---|---|---|
| override off (now the only behaviour) | 100%, all 92 layers | yes |
| override on at 46 (removed) | 49%, diverged at L46 | no |

The "noise" after ` Paris.` is therefore the model, not a defect. Three causes, none a bug:
the prompt is an unfinished sentence rather than a question; `stop_reason` was `"length"` on
every run because we asked for a fixed token count and EOS never fired; and selection is
greedy `argmax`. His README says the same of his own demo -- "this particular batch command
deliberately asks for a raw continuation".

## What is not established

- His numbers are not reproduced here and his hardware is not ours.
- Same checkpoint is likely but unverified: he cites 2.78T parameters from a 1.56 TB
  checkpoint; this dataset is ~1.44 TB of experts plus 53 GB of trunk.
- Our figures are n=1 to n=3 per arm with a ~1 s run-to-run spread.
- No claim is made about his implementation's internals. We read his README, not his code.
