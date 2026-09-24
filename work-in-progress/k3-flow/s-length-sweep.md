# s — length sweep

**Probes:** the one experiment the eight variations owed. Across v1–v8, prompt length and
prompt content varied together, so the L91/92 amplification threshold between 12 and 18
positions was suggestive and not established. Here **content is held constant** and only
length moves: the same 225-token passage from v6, truncated at exact token boundaries.

```
ids      strict prefixes of v6's 225 ids, from ./bin/test_tok encode
lengths  5, 12, 18, 40
gen      1        (only the prefill is in question)
flags    --incremental --trunk /root/k3trunk_i8 --trunk-gb 60 --cache-gb 30
extra    K3_TRACE_ROWS=1 — per-position L2 and absolute maximum, new this cycle
```

The per-position tap is new. It closes the gap named in v4: previously the taps hashed
the whole `T × hidden` buffer, so an extreme value could be located to a layer but not to
a token. Verified not to disturb the arithmetic — with it compiled in, the fixture oracle
still reports ENGINE MATCHES THE REFERENCE EXACTLY on all four gates.

## Results

| T | L90 residual | L91 attn out | ratio | L91 attn absmax |
|---:|---:|---:|---:|---:|
| 5 | 108.91 | 13.63 | 0.13× | 0.727 |
| 12 | 165.01 | 24.69 | 0.15× | 1.008 |
| 18 | 195.28 | 32.29 | **0.17×** | 1.296 |
| 40 | 323.12 | 99.27 | 0.31× | 3.266 |
| 225 (v6) | — | — | **7.30×** | 3319.94 |

### The length-threshold reading is falsified

v4 spiked at **19.1×** with 18 positions of code. This passage at **the same 18
positions** gives **0.17×**. Length alone does not produce the amplification, and the
"threshold between 12 and 18 positions" recorded after v6 is wrong.

For this passage the ratio does climb with length — 0.13, 0.15, 0.17, 0.31 — and then
reaches 7.30× at 225. So length matters, but the rate is set by content: code arrives
there by 18 positions, this prose has not by 40.

### What actually happens: one token carries all of it

The code prompt was re-run with the new per-position tap. Absolute maximum at layer 91's
attention output, by position:

```
0.7  0.4  0.9  0.8  1.0  1.1  1.4  1.4  3.4  1.0  1.6  1.4  0.8  1.4  1.5  3457.8  1.4  1.3
                                                                          ^^^^^^^^
```

**Position 15 alone carries 3457.8 against a median of 1.397 — 2474×.** The other
seventeen positions are ordinary. Layers 91 and 92 peak at the same position: row L2
10166.0 and 20491.2, against medians of 56.83 and 82.74.

**The token at position 15 is id 198 = `'\n'`** — a bare newline. The prompt's other line
breaks are merged with the following indentation into different tokens; the one
standalone newline is the one that carries the activation.

This closes the gap named in v4. The hypothesis recorded there — "one token dominates" —
was explicitly marked as *not established*. It is now established, and the token is
identified. It is the documented attention-sink behavior on a delimiter, observed
directly rather than inferred.

The s40 prose outlier is a different, milder thing: position 30 reaches 3.27 against
neighbors near 1.5, and its token is `' current'` — not a delimiter, and 1000× smaller
than the newline spike.

### Causality, confirmed by accident

Per-position values for the first 5 positions are identical across the T=5, T=12, T=18
and T=40 runs, the first 12 are **exactly** equal between T=12 and T=18, and the first 18
between T=18 and T=40 — checked as equalities, not by eye.

That is what causal attention requires: a position's value cannot depend on tokens after
it. The sweep was not built to test it, and it is a stronger check than anything in the
variation set, because it compares four independent processes rather than two sites
inside one.

### Cost, for the record

| T | prefill seconds | expert GB |
|---:|---:|---:|
| 5 | 30.84 | 98.30 |
| 12 | 62.18 | 176.58 |
| 18 | 74.34 | 226.97 |
| 40 | 131.85 | 367.05 |
