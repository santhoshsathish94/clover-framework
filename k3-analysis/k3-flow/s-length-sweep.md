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
| 80 | — | — | 0.27× | 3.266 |
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

### Then the long-prose run corrected the delimiter reading

v6 — the same passage at 225 positions — was re-run with the per-position tap:

| | v4 code, T=18 | v6 prose, T=225 |
|---|---:|---:|
| positions above 100× median | **1 of 18** | **1 of 225** |
| position | 15 (3rd from last) | 216 (9th from last) |
| absolute maximum | 3457.84 | **3319.94** |
| median | 1.397 | 1.036 |
| ratio to median | 2474× | 3204× |
| the token | 198 `'\n'` | 2032 **`' read'`** |

**What generalizes is the mechanism, not the token.** Exactly one position absorbs the
whole thing, in both runs, and nothing else comes close — the next largest position in
v6 is 3.27, a thousandth of the spike. But `' read'` is an ordinary word, so "it happens
on delimiters" is wrong. Recorded here because it was written above before this run
existed.

The part worth noticing is the magnitude. Two completely different inputs, 12× apart in
length, land within 4% of each other: **3457.84 and 3319.94**. That looks like a
saturation level rather than a value driven by the input. Two data points, so it is an
observation and not a finding.

## Where it starts, and what it writes to

Two further readings, the first free from traces already taken.

### It begins at layer 90, identically in both runs

Following the one spiking position through all 93 layers:

| layer | v4 code, position 15 | v6 prose, position 216 |
|---|---:|---:|
| 88 | 1.804 (0.9× median) | 2.025 (1.1×) |
| 89 | 2.289 (1.0×) | 2.182 (1.1×) |
| **90** | **14.837 (5.3×)** | **13.199 (5.5×)** |
| **91** | **3457.840 (2474×)** | **3319.940 (3204×)** |
| 92 | 1809.270 (**707.9×**) | 1426.540 (**704.9×**) |

Nothing at all through layer 89. A precursor at layer 90 — 5.3× and 5.5×, values within
12% of each other. Then layer 91. The layer-92 ratios agree to **0.4%**: 707.9× against
704.9×, from two unrelated prompts of very different lengths.

At layer 90 the *residual* at this position is still 0.3× the median in both runs, so the
attention output is elevated before the residual is. Whatever happens, it happens inside
attention at 90 and completes at 91.

### It writes to one specific channel

The tap now records which of the 7168 channels holds each position's maximum. For the
code run, position 15:

| layer | attention out | residual out |
|---|---|---|
| 88 | channel 3680 | channel 3680 |
| 89 | channel 3680 | channel 3680 |
| **90** | **channel 4590** | channel 6549 |
| **91** | **channel 4590** | **channel 4590** |
| **92** | **channel 4590** | **channel 4590** |

The channel switches to **4590** exactly where the precursor appears, and stays there.
Every other position at layer 91 peaks somewhere else — 6532 for thirteen of them, 5671
for two, 537 for one. Only the spiking position uses 4590.

So this is not a large number appearing somewhere. It is one channel, entering at one
layer, on one position.

Whether prose uses the same channel 4590 is the decisive question and is not yet
answered — v6 was traced before the channel index existed.

## There is no length threshold at all

Extending the sweep to 80 and 160 positions:

| T | ratio | L91 max | at position | channel | above 100× median |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.31× | 3.266 | 30 | 537 | 0 |
| 80 | 0.27× | **3.266** | **30** | 537 | 0 |
| 160 | 0.26× | **3.266** | **30** | 537 | 0 |
| 225 | 7.30× | 3319.94 | 216 | — | **1** |

The maximum is *identical* at 40, 80 and 160 — same value, same position, same channel.
Causality again: position 30 is fixed once positions 0–30 are.

**So the threshold was an illusion.** The v6 spike sits at position 216, and no
truncation shorter than 217 tokens can contain that position. By causality its value at
T = 217 must equal its value at T = 225, so the onset is exactly T = 217 and length is
doing no work at all. What looked like a threshold between 80 and 225 is simply the point
at which the spiking position enters the prompt.

## And it is not the token either

Token 2032 `' read'` occurs three times in the passage — positions 20, 92 and 216. The
layer-91 values there are **1.204, 0.768 and 3319.940**. Same token, same model, same
run; one occurrence in three carries the spike.

The surrounding text is unremarkable:

```
pos 213  ' should'    L90  2.380   L91     1.255
pos 214  ' not'       L90  2.490   L91     1.257
pos 215  ' be'        L90  4.628   L91     1.462
pos 216  ' read'      L90 13.199   L91  3319.940   <--
pos 217  ' as'        L90  2.143   L91     1.372
pos 218  ' a'         L90  2.075   L91     1.359
```

## Four explanations, four falsifications

Each was held briefly and killed by one more run:

1. **"Specific to code."** v6, English prose, spikes at 7.30×.
2. **"A length threshold between 12 and 18 positions."** The same passage at 18 positions
   gives 0.17× where code at 18 gives 19.1×.
3. **"It happens on delimiters."** v6's spike is on `' read'`, an ordinary word.
4. **"It is that particular token."** `' read'` appears three times; one occurrence
   spikes.

## What survives

- Exactly **one position** carries it, in both spiking runs — 1 of 18, 1 of 225.
- It enters at **layer 90** as a 5.3×/5.5× precursor and completes at **layer 91**.
- In the code run it lives in **channel 4590**, switched to exactly at layer 90 and held
  through 92, while other positions peak elsewhere.
- Magnitudes are close across unrelated inputs — **3457.84** and **3319.94** — and the
  layer-92 ratios agree to **0.4%**.
- It never reaches the output, because RMSNorm is scale-invariant.

**What selects the position is not explained by anything measured here.** Not the token,
not a delimiter, not the first or last position, not length.

## Still open

- Whether the prose spike uses channel 4590 as well. v6 predates the channel tap and a
  re-run costs eight minutes; the two non-spiking sweep points cannot answer it.
- Whether ~3400 is a ceiling. Two data points is not enough to call it one.


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
