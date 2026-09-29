# p — what selects the position

**Probes:** the gap left open by every previous cycle. One position per prompt receives
the whole magnitude in channel 4590. Not the token, not a delimiter, not the first or
last, not length. So what?

Context first: the existing traces were interrogated before anything new was built.

## The position is invisible until layer 90

Tracking the spiking position's rank among all positions, layer by layer:

| layer | v4 code (pos 15 of 18) | v6 prose (pos 216 of 225) |
|---|---|---|
| 0 | 8th | 131st |
| 10 | 16th | 109th |
| 30 | 7th | 161st |
| 60 | 16th | 208th |
| 80 | 10th | 153rd |
| 89 | 11th | 52nd |
| **90** | **1st, 5.35×** | **1st, 5.51×** |
| 91 | 1st, 2474× | 1st, 3204× |

Mid-pack the entire way up. Nothing marks it. Then layer 90.

## What layer 90 predicts, with a clean gap

At layer 90 the position with the **minimum residual** is the spiking position in all
three spiking runs. Its attention output there separates the two groups completely:

| run | min-residual position | L90 attention | channel | L91 outcome |
|---|---|---:|---:|---|
| v4 code T=18 | 15 of 18 | **14.837** | 4590 | 3457.84 **spike** |
| c1 code T=17 | 14 of 17 | **13.211** | 4590 | 3293.84 **spike** |
| v6 prose T=225 | 216 of 225 | **13.199** | 4590 | 3319.94 **spike** |
| c2 javascript T=22 | 17 of 22 | 9.908 | 4590 | 4.52 — |
| s80 prose | 65 of 80 | 5.606 | 6532 | 1.20 — |
| s160 prose | 155 of 160 | 3.190 | 6532 | 0.92 — |
| s40 / s18 / s12 | 7 | 2.630 | — | 0.77 — |
| s5 / c3 | 0 | 1.14 | 4590 | 0.70 — |

**Nothing lands between 9.908 and 13.199.** Below the gap it decays; above it, it runs
away to ~3300–3460. The layer-90 value predicts the layer-91 outcome in all eleven runs.

Note also that c2 reaches 9.908 *in channel 4590* and still does not spike, so being in
the right channel at the right position is not enough. The magnitude decides.

## Minimum-residual is a consequence, not the rule

The obvious reading — "the model picks the position carrying the least" — does not
survive checking. The residual **entering** layer 90 is layer 89's output, and there the
spiking position is not the minimum:

| run | spiker's rank at layer 89 | actual minimum at layer 89 |
|---|---|---|
| v4 code | 3rd lowest of 18 (17%) | position 1 |
| c1 code | 4th lowest of 17 (24%) | position 1 |
| v6 prose | 26th lowest of 225 (12%) | position 166 |

Low, in the bottom quarter, but not the minimum — and a different position is.

**And the attention input at layer 90 does not distinguish it either:** rank 17/18, 13/17
and 220/225 by magnitude, at 0.81–0.95× the median, every one of them peaking on the
ordinary channel 3680.

So a position that looks unremarkable going in comes out carrying 13–15 in channel 4590.
The selection happens **inside layer 90's attention**, from something these taps cannot
see.

## What that leaves

Established:

- Selection occurs at layer 90 and is not detectable in the residual or the attention
  input beforehand.
- The selected position sits in the low tail of the residual (bottom 12–24%) but is not
  the minimum before selection; it becomes the minimum after.
- The layer-90 magnitude in channel 4590 predicts the layer-91 outcome across eleven
  runs, with an empty gap between 9.9 and 13.2.

Not established, and explicitly not claimed:

- Why layer 90's attention chooses that position. The current instrument records what
  each stage produced, never how attention distributed itself. The obvious hypothesis is
  that the position is acting as an attention sink and receives disproportionate
  attention mass — but no attention weight has been measured, so that is a guess.

The next instrument is therefore specific: the attention mass each key position receives
at layer 90, summed over heads and queries. One number per position, cheap even at 225
positions.

## Built it, and it falsified the hypothesis immediately

Attention mass per key, summed over queries and heads, at every MLA layer of the code run
(total mass 1728 = 18 queries × 96 heads):

```
L 3  position 0 receives 1340 of 1728      position 15 receives 11.4  (0.7%)
L15  position 0 receives 1716 of 1728      position 15 receives  0.09 (0.0%)
L19  position 0 receives 1725 of 1728      position 15 receives  0.00 (0.0%)
L91  position 0 receives  918 of 1728      position 15 receives  1.74 (0.1%)
```

**Position 0 is the attention sink** — up to 99.8% of all attention in the model lands on
the first token. The spiking position receives 0.1–1.7%, ranked 16th of 18 at the very
layer where it explodes.

So it is not a sink. It is not attended *to* at all.

### And layer 90 is the wrong kind of layer for this tap

Layer 90 is **KDA**, the recurrent path; 91 and 92 are MLA. The tap only covers MLA, so
the layer-90 precursor is not visible to it at all. Two different attention mechanisms,
and the one where selection begins was not the one instrumented.

## What the query side shows

Measuring each query instead of each key, at layer 91:

| position | attention given to the sink | peak probability | output |
|---:|---:|---:|---:|
| most positions | 0.34 – 0.84 | 0.49 – 0.92 | 0.4 – 1.6 |
| 8 | 0.8438 | **1.0000** → key 8 | 3.365 |
| **15** | **0.0160** | **1.0000** → key 8 | **3457.840** |

At layer 87, an MLA layer that does not spike, position 15 is completely ordinary: 0.7442
to the sink, peak 0.9610, output 2.010 — indistinguishable from its neighbors.

**At layer 91 the spiking position abandons the sink.** Every other position routes a
third to five sixths of its attention to position 0. Position 15 sends 1.6%, and one of
its heads saturates completely — probability 1.0000 — onto key 8.

Key 8 is the position with the **largest** residual at layer 90 (177.1 against a median of
49.7). The spiking position has the **smallest**. The one carrying least copies, at full
weight, from the one carrying most.

## Where this honestly stops

**The obvious conclusion does not close.** Position 8 *also* saturates on key 8 — peak
1.0000, same key — and its output is 3.365. A thousandth of the spike. If the magnitude
came simply from copying key 8's value, both would receive it.

So "it copies the big position" is not sufficient. The measurement is coarse: peak and
argmax are taken over all 96 heads, so two positions can both read "saturated on key 8"
while their remaining 95 heads do entirely different things. Separating them needs
per-head data, which this tap does not produce.

### Established this cycle

- Position 0 is the attention sink, taking up to 99.8% of attention mass.
- The spiking position is **not** a sink — it receives almost no attention.
- It is a **query** that abandons the sink at layer 91: 1.6% against everyone else's
  34–84%, with a head saturated at probability 1.0000.
- It is entirely ordinary at layer 87, four layers earlier.
- The position it saturates onto carries the largest residual; it carries the smallest.
- Layer 90, where the precursor appears, is KDA and remains uninstrumented.

### Not established

- Why that head saturates, for that position, at that layer.
- Why position 8 does the same thing and gets 1000× less.
- Anything about the KDA precursor at layer 90.

