# Representing a 7168-component vector

## Direction

Start with the activation vector, before considering experts or trunk storage.
Preserve its ordered coordinates, the learned weights and all 93 layers.
This is a representation study, not a continuation of normalization reduction.

## What the object is

A vector has 7,168 ordered components. At float32 it occupies 28,672 bytes
(28 KiB). The components can be negative, zero or greater than one; they are
not constrained to the interval [0, 1]. Coordinate order matters to the weights
that consume it.

Every finite binary floating-point component admits an exact dyadic form:

$$x_i = m_i 2^{e_i},$$

where $m_i$ is an integer. For a block of components, choose the smallest
exponent among its nonzero values, $e_b$, and write

$$x_i = M_i 2^{e_b}, \qquad M_i = m_i 2^{e_i-e_b}.$$

This changes the representation without rounding the values. Whether it saves
bytes depends on the integer widths and metadata. It does not automatically
reduce the information or the cost of using it. Signed zero and non-finite
bit patterns need separate treatment; the experimental codec retains such
blocks as raw bytes.

Magnitude plus direction is another coordinate system, not automatically a
compression: a unit direction still has 7,167 degrees of freedom and its
magnitude supplies the remaining one. No lower-dimensional representation of
the model's reachable vectors has been established by this study.

## Measurement

Run on AX102, AMD Ryzen 9 7950X3D, 2026-09-29. The unchanged binary at
`/opt/clover-k3/build/clover-k3` processed token IDs
`1008,10484,318,15383,387` using its existing `K3_DUMPLAY` and `K3_DUMPRES`
capture options. No model source was modified or rebuilt.

- Five layer-0 inputs are the exact BF16 embedding widened to float32.
- 460 later inputs are pre-attention aggregates, before RMSNorm, at layers 1-92.
- Five final residuals are the layer-92 residual outputs, before tail aggregation.
- All 470 vectors have 7,168 finite values. These are not captures of every
  internal vector or of the normalized inputs read by projection kernels.
- The captured run retained the preserved logits checksum
  `23d162dcefb18211a7540ef12948f1eb`.

Source SHA256:
`5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65`.
Binary SHA256:
`4cebb69e5379c4f0c9c03d4071c49840d3d7596ecfd2c5c0d8b83f6679717d35`.

Settings: `OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`,
`K3_PREFETCH=4`, `K3_TRUNKRAM=0`, `K3_NREADER=14`, `K3_XDEC=2`,
`K3_PLGRAN=1`, `K3_HUGE=1`; packed trunk `/root/k3trunk_i8/trunk.bin`,
index `/opt/clover-k3/build/eqidx.bin`.

The [analyser](vector-representation.py) checks every encoded vector by
decoding it and comparing all original bytes. Compression is independent per
vector, with no cross-vector dictionary. Zlib counts include a four-byte
length frame; dyadic counts include the implemented vector and block headers.
The byte-plane variant groups the four byte positions of the float32 words
before applying standard zlib compression. It does not reorder coordinates
in the decoded vector.

## Results

Storage as a percentage of the original float32 payload, summed within each
group. Smaller is better; these are not timing measurements.

| Representation | Embeddings (5) | Later aggregates (460) | Final residuals (5) |
|---|---:|---:|---:|
| Float32 | 100% | 100% | 100% |
| BF16 without changing bits on widening | 50% | Not exact | Not exact |
| Shared exponent, blocks of 32 | 50.72% | 95.45% | 95.55% |
| Shared exponent, blocks of 128 | 55.68% | 97.69% | 98.22% |
| Zlib on original bytes | 49.55% | 92.74% | 92.93% |
| Byte planes then zlib | 39.03% | 88.14% | 88.49% |

Every tested lossless codec reconstructed all 470 vectors byte-for-byte.
BF16 was separately eligible for all five embedding vectors, and for none
of the later vectors or final residuals. Only 145 of the 3,297,280 later
aggregate components were individually BF16-exact.

There were no exact zero components in these captures. Later vectors had a
median of 7,168 distinct bit patterns per vector. Their observed component
range was -26.98854637145996 to 109.28179168701172. Sparsity or widespread
exact repetition is therefore not supported by this sample.

The 32-component dyadic format saves 4.55% across later aggregates, but its
largest vector occupies 28,984 bytes, more than raw float32. The best tested
storage representation there, byte planes plus zlib, saves 11.86% overall.
Neither figure is a compression limit or a result for all prompts.

## Interpretation And Next Question

The embedding has a compact exact representation already: its BF16 row, or
its token ID together with access to the embedding table. A token ID alone
does not contain that table. Later context-dependent vectors do not inherit
the embedding's BF16 precision merely because the weights were quantized.

For storage or transfer, byte-plane compression is a measured candidate.
For computation, the current kernels still require float32. Encoding and
decoding throughput, model integration and end-to-end benefit have not been
measured. Exact reconstruction permits the original computation to be retained;
it does not prove that the representation is useful on the consuming path.

The next question is whether the intended consumer can use a compact exact
form economically, rather than repeatedly expanding it. Keep this separate
from changing numerical precision, removing coordinates or skipping layers.

## Artifacts

- [Per-vector measurements](vector-representation-results.json), including
  capture hashes and representation sizes.
- AX102 directory `/opt/clover-k3/vector-representation-20260929-a` holds
  `analyze.py`, `france-layers.bin`, `france-residual.bin`, `france-logits.bin`,
  the run logs and `france-analysis.json`.
- Reproduce analysis on AX102 with
  `python3 analyze.py france-layers.bin france-residual.bin --positions 5 --output new-analysis.json`.
  Output creation is exclusive to avoid overwriting earlier measurements.

One prompt, five positions, selected vector boundaries only. No conclusion
about experts, trunk representation, other prompts, decode or runtime speed.