# Integrated model speed and RAM

Direction (2026-09-29): measure the speed and RAM of the final all-item
integrated model runs against the unchanged reference on AX102.

## Protocol

- Frozen final step1 candidate versus reference; no source or gate changes.
- Same France/Japan five-token prefills and prior 16-thread configuration.
- One warm-up per prompt/binary, then three measured paired trials per prompt.
  Alternate first variant across pairs; run processes serially.
- Monotonic end-to-end process time, including initialization and teardown.
- Linux wait4 peak resident set size per process, KiB converted to GiB.
- Preserve CPU time, faults, filesystem block counters, load/memory snapshots,
  internal printed timing, outputs, routes and diagnostics for interpretation.
- Full output/routes must match; every integrated run must pass the unchanged
  stage checker after timing. Measurement never disables correctness controls.
- No cache flush or cold-start claim. The candidate retains original work,
  serial encoding stages and diagnostic output, so this measures that actual
  experimental implementation, not hypothetical deployment savings.

The wrapper is [integrated-performance.py](integrated-performance.py).
First validate the clock/RSS/exit-status wrapper, then warm-ups, then trials.
That protocol is complete; measured results follow.

## Instrumentation and warm-up

The wrapper control passed: a process touching 64 MiB peaked at 75776 KiB
(74 MiB including Python), and a separate failing process returned exit 7.
Four model warm-ups passed full-output/routes checks; both integrated warm-ups
also passed all thirteen representation gates. Each reported zero major faults.
Warm-up times were 9.98-10.43 seconds and peak RSS 54.1830-54.1845 GiB on AX102.
These warm-ups are excluded from the measured summary.

## Measured outcome on AX102

Three measured trials per prompt/variant; median end-to-end seconds and median
per-process peak RSS. All 12 measured runs passed full output/routes comparisons;
all 6 measured integrated runs also passed the unchanged thirteen-item checker.

| Prompt | Reference seconds | Integrated seconds | Integrated elapsed increase | Reference peak RSS GiB | Integrated peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| France | 10.0758 | 10.4375 | 3.5903% | 54.1834 | 54.1840 |
| Japan | 9.9975 | 10.3384 | 3.4096% | 54.1830 | 54.1841 |

Observed timing ranges: France reference 10.0697-10.0846 s, integrated 10.3994-
10.4600 s; Japan reference 9.9750-10.0037 s, integrated 10.3307-10.3480 s.
Median peak RSS differences are 660 KiB (France) and 1184 KiB (Japan), small
compared with the approximately 54.18 GiB footprint. **No RAM reduction observed.**

This as-is integrated implementation took about 3.4-3.6% longer on this AX102
sample. It retains reference computations, encoding/decoding stages and
diagnostic writes. The measurement does not isolate codec cost or predict an
optimized deployment. Three trials per cell do not establish statistical
significance or generalize to other prompts, hardware or decode/generation.

Every measured run reported zero major faults, but that does not mean zero
storage traffic. The first France pair each reported 99.86 GB read; the model's
source includes direct-I/O expert loading. No warm-cache-only claim is made.
Peak RSS includes resident mapped pages; it is not system-wide memory consumption
or the size of the kernel filesystem cache. External timing includes process
startup/exit work omitted from the model's narrower internal timer.

The independent evidence verifier is
[integrated-performance-check.py](integrated-performance-check.py).
Its [verification result](integrated-performance-verification.json) passed all
trial, output, route, source-hash, unit-conversion and summary arithmetic checks.
The [complete measured results](integrated-performance-results.json) retain
every trial, ranges, CPU/fault/I/O counters and host snapshots. The
[local-only raw evidence archive](README.md#local-only-artifacts) includes model
logs, outputs, routes, integration traces and validation reports; it was
hash-verified after transfer. Its SHA256 is
b6ad3fe90c83e6b0abfd62be2802e90743c076b33946687f9fe830f3095a76d9.

AX102 originals remain under
`/opt/clover-k3/integrated-performance-20260929-a`. Installed model source,
binary, gate and index hashes are unchanged. No model code, configuration,
correctness gate, persistent store or checkpoint was modified for this run.