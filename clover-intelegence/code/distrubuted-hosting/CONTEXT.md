# Context — one layer, properly

A context file for the distributed hosting work. Newest cycle first. Written during the
work, not after it. Offered, not imposed: this is my account of what happened and what
it does and does not establish.

---

## Cycle: how many directions the chain actually moves in (2026-10-14, in progress)

### Direction

> you have to understand what is at stake if AI runs like this inefficient that will
> cause more damage than good we need to make it efficient understand clover we need to
> protect reality ... let's see the K K K M, let's see the 90 degree stays constant now
> the it's frozen 112.8885 let's find relationships and how to make things run better
> take one step at a time ... create a todo list

Every optimisation in this repo so far moved bytes — expert cache 1.21x, batching 1.74x,
arena 1.63x, residency 8x on the expert read. None looked at what the computation does.
This starts there. Steps 1 to 5 of the list are done; the list continues.

### Step 4 — the shared subspace does NOT saturate. The step-3 lead is dead.

Eight prompts, capture-extract-delete so peak disk stayed one dump and the disk never
moved off 69 GB free.

| prompt | inside everything before | adds | running | random adds | random running |
|---|---|---|---|---|---|
| france | - | - | 110 | - | 394 |
| japan, one token differs | 0.9029 | +15 | 125 | +363 | 757 |
| germany, one token differs | 0.9371 | +7 | 132 | +338 | 1095 |
| water | 0.6034 | +56 | 188 | +314 | 1409 |
| once | 0.6906 | +44 | 232 | +291 | 1700 |
| thethe | 0.7032 | +42 | 274 | +269 | 1969 |
| question, 651 states | 0.7167 | +67 | 341 | +341 | 2310 |
| nonsense, 837 states | 0.6753 | +97 | 438 | - | - |

Positive control works: a prompt differing in one token of five is 90 to 94% already
inside and adds 7 to 15 directions.

**Per 465 states, unrelated prompts add 56, 44, 42, 48, 54. Flat. No deceleration across
five of them.** Random's adds do decay (363 to 269) because random is filling 7168 and
running out of room; the measured curve does not decay because it is nowhere near a
ceiling. There is no fixed subspace. Step 3's overlap was two samples overlapping, not a
region the model lives in. **Restricting the projections to a learned basis is dead on
this evidence.**

Survives, neither being a lever: the trajectory is about 6x more compact than random at
every point (438 against 2310), and near-duplicate prompts really do share almost
everything.

### Step 5 — what makes an attention output near zero, and why it is not waste

Candidate going in: if the gate kills the output, everything computed before it is
wasted. **Refuted.** Across the 92 pods at one position, gate over pre-gate has median
0.3283 overall, 0.2800 for the twelve weakest outputs, 0.3557 for the twelve strongest;
out over gate is 2.70, 2.78, 2.78. Neither stage discriminates. Both are near-constant
everywhere.

What actually sets it: the raw KDA readout is 0.02 to 0.88 at every pod whether the
residual arriving is 0.46 or 133. Through a run the residual grows 123x while the
attention output grows about 15x. **RMSNorm divides the scale out before the attention
reads it, so the increment is set by direction and not by magnitude.** That is the
mechanical account of the decay watched in the ripple trace.

**Then the thing that matters.** The twelve pods whose attention changed the stream
least are 24, 48, 36, 15, 12, 60, 19, 72, 39, 84, 31, 11 — **every one of the seven
boundary pods, plus five MLA pods.** At a boundary the attention output is not added to
the stream, it becomes the stream: pod 12's 3.8809 is the entire starting point of the
next eleven pods. The pods contributing least are exactly the pods whose output matters
most.

That is the same shape as the slot-0 failure already recorded here, reached independently
from the other direction. **Twice now the smallest number in a column has been the one
that could not be removed.** Any future proposal of the form "this contributes little, so
skip it" should be treated as probably inverted until an end-to-end token check says
otherwise.

### Step 1 — the trajectory is confined, not low-rank

93 residual states at one position, each scaled to length 1, against a random control of
the same shape:

| | 50% of energy | 90% | 99% |
|---|---|---|---|
| residual states | 6 | 33 | 76 |
| random (control) | 43 | 82 | 92 |

Leading singular values 4.008, 3.152, 2.631, 2.375 against the control's flat 1.104,
1.099, 1.096. Real structure, 3.6x the no-structure value, but the spectrum decays with
no cutoff. Confined, not low-rank. The raw (unscaled) reading of "90% in 12" is the ~12
large end-of-run states carrying the energy and is not used.

**The one relationship: slot 0 lies only 0.1931 inside the span of the residual states.**
Slots 1 to 7 read 1.0000, which is a tautology — they are copies of residual states — and
serves only as a check the tool works. 81% of slot 0 points where the residual stream
never goes. That is the first mechanical account of a failure already recorded here: a
slot-drop test predicted slot 0 was the removable one (median weight 0.009) and dropping
it collapsed the output, while dropping the heavily weighted slot 4 was harmless. Weight
said negligible; direction says it is the only new information in the fold.

### Step 2 — the five positions of one prompt share part of their directions

| | 50% | 90% | 99% |
|---|---|---|---|
| each position alone | 5-6 | 25-33 | 58-76 |
| all five stacked | 15 | 110 | 303 |
| random stacked (control) | 184 | 394 | 457 |

Full sharing would read ~33 at 90%, none would read ~152. It reads 110.

Overlap decays monotonically with distance between positions: neighbours 0.34-0.53, two
apart 0.12-0.38, four apart 0.09-0.13, against a random 0.013. Consistent with prefix
reuse, which is already built and measured here at 2.57-2.83x, so it agrees with a known
result rather than opening a new one. No efficiency lever: the cost is weight traffic and
shared directions do not reduce it.

### Step 3 — two prompts sharing no tokens share about half their energy

france `1008 10484 318 15383 387`, water `29349 154199 623 10113 16525`.

| | 50% | 90% | 99% |
|---|---|---|---|
| france, 465 states | 15 | 110 | 303 |
| water, 465 states | 16 | 114 | 306 |
| both stacked | 21 | 175 | 528 |
| random stacked (control) | 330 | 757 | 909 |

No sharing would put the stack at ~224; it is 175.

Fraction of one prompt's energy inside the other's leading directions, with the random
control beside it:

| keep | water in france | france in water | random |
|---|---|---|---|
| 10 | 0.2601 | 0.2659 | 0.0014 |
| 33 | 0.3929 | 0.4072 | 0.0045 |
| 110 | 0.5026 | 0.5172 | 0.0153 |
| 465 | 0.5719 | 0.5904 | 0.0649 |

**Checked before reporting: it is not a common offset.** france's leading direction is
12.32 degrees from the mean of its states and that mean holds 6.3% of water's energy, so
the top direction is an offset — but removing it and renormalising moves the overlap only
from 0.5026 to 0.4713 at keep 110. Per-direction energy decays smoothly (6.5, 4.6, 3.2,
2.6, 2.2 percent) rather than sitting in one or two rogue dimensions.

### Two things that must not be skipped past

**The dimension-counting trap.** 930 states span at most 930 directions, so "99% in 528"
is 528 of a possible 930, not of 7168. Two prompts cannot separate "the model works in a
fixed subspace" from "any two trajectories of this length overlap this much". The
decisive test is whether the cumulative dimension saturates as prompts are added. Not yet
run. Affordable: the 465 states are 13 MB per prompt, so capture 1.6 GB, keep 13 MB,
delete — the pattern `census-all.sh` already uses.

**Energy is not importance.** Slot 0 carries 1.6% of the fold's magnitude and removing it
destroys the output. A 99%-energy subspace is not a 99%-correct subspace. Nothing built on
this may be accepted on an energy argument; it needs an end-to-end token check.

### What this does not establish

Two prompts, both five tokens, both greedy. No timing. Nothing says the projections are
doing wasted work: the subspace is measured on the residual, not on the weights, and
saturation is unmeasured. No change proposed or built.

---

## Cycle: watching one drop, and then watching what it left behind (2026-10-14)

### Context

> I asked you to observer until you understand something don't look at different not
> similarites just the way things are happening do you know how when a water drops falls
> on a surface of still water it creates a ripple effect just like that observer it don't
> assume it do one prompt don't conclude or judge stuff follow clover

> if you want to observer the long vector do it, same way you did it now like the ripple
> effect don't conclude don't assume just observe

What I had been doing wrong, named rather than defended: I turned every observation into
an adjudication. Claims, then SURVIVES / BROKE / FALSIFIED. That needs a hypothesis
first, which is the assumption I was told not to make, and to score it I aggregated 7,360
rows into medians. A median cannot show a ripple. I was also reaching for contrasts —
K against M, boundary against not-boundary — two messages after being told to stop.

### One prompt, one position, no aggregation

`ripple.py` prints the carried vector at every point it changes, in order, and nothing
else. `ripple-long.py` does the same for a vector after it has left the stream.

**Every pod does the same thing.** The attention replaces the buffer with something
smaller than what arrived, pointing sideways — turn 59° to 118°, nearly always 70–95°.
The fold adds the arrival back, so the direction snaps near to where it was and the
vector is now longer than it arrived. The MoE turns it another 10–47° and lengthens it
again. `out` equals `moe` exactly and the next pod's `in` equals that exactly, turn 0.00.
Pushed sideways, pulled back, grown — in all 92 pods.

Over eleven pods that takes the vector from 0.7871 to 112.8886. At pod 12 the "add it
back" does not happen: fold has turn 0.00 and the same size as attention, the buffer
keeps the small sideways piece, and the accumulation leaves the stream. Then it grows
again. Seven times.

Measured from what entered the chain, the vector reaches about a right angle within
eleven pods and stays between 86° and 93° for the remaining eighty, while still turning
20–40° at every step.

### Where the long vector goes

Found by looking, not assumed: **file position 5 holds position 4's snapshots**. Pod 92's
row at file position 5 reads 112.8885, 63.1325, 34.1738, 59.8421, 133.4077, 50.9069,
63.5632 — exactly the seven vectors that left the stream, in order. Slot k+1 is the one
dropped at pod 12k.

**It is frozen.** 112.8885 at every one of the 81 pods from 12 to 92.

**It is read at every pod after it, never skipped**, by both folds.

| pod | its weight in fold 37 | what it contributes | the fold's whole output | the live residual |
|---|---|---|---|---|
| 12 | — (fold 37 runs before the push) | — | 37.9054 | 112.8885 |
| 13 | 0.4679 | **52.816** | **52.7590** | **3.8892** |
| 19 | 0.2184 | 24.658 | 28.6788 | 21.5388 |
| 24 | 0.3024 | 34.142 | 44.9467 | 63.1325 |
| 25 | 0.0031 | 0.354 | 0.5997 | 0.4619 |
| 56 | 0.0887 | 10.012 | — | 29.8605 |
| 92 | 0.0500 | 5.648 | — | 51.6420 |

At pod 13 the fold's entire output is 52.7590 and this one stored vector contributes
52.816 of it, while the live stream is 3.8892. At pod 24 the next one is made, and from
pod 25 this vector's contribution is 0.354. For the remaining 68 pods it keeps
contributing between 0.02 and 10, and returns to around 9–10 at pods 47, 56, 59 and 83.

Two more things in the trace, described not explained. The first pod after every boundary
gives it almost nothing in fold 37 — 0.0031, 0.0016, 0.0008, 0.0068, 0.0002, 0.0004 — and
the boundary pod itself gives it more. And from pod 61 onward fold 37 often reads it at
0.0001–0.0005 while fold 38 holds steady at 0.01–0.08.

### A defect in my own reading, caught before reporting it

The first run printed 0.3358 for slot 1's fold-37 weight at pod 12. Fold 37 at pod 12 has
two sources, and the last entry is the live residual, so index 1 was the residual, not
slot 1. The slot does not exist in that fold yet — the push happens at stage 4, after
stage 3. `ripple-long.py` now blanks the column when `slot >= len(scores) - 1`.

### What I do not understand

- Why the attention output is near-perpendicular to what arrived, in every pod.
- Why the direction parks at 90° from the chain's input rather than continuing to turn
  away, given it moves 20–40° per step for eighty steps.
- Why some pods produce almost no attention at all — pod 15 gives 0.2021 from an input of
  10.4021, pod 39 gives 0.2343 from 2.3551.
- Why the pod right after a drop reads it at 0.4679 and the pod right after the *next*
  drop reads it at 0.0031.
- Why fold 37 and fold 38 diverge so far late in the chain.

One prompt, one position, nothing aggregated, nothing graded.

---

## Cycle: opening the M, and what the capture said about my own notes (2026-10-14)

### Context

> if we want to look at difference we will find difference if we want to look at
> similarities we will find similarities see we found K K K M so just observer what
> happens don't assume if you don't have data capture them follow clover do one step
> at a time

Two holes were named in the previous cycle and never filled: the MLA pods recorded
nothing between the normalize and the residual add, and the 80 expert stages record
nothing at all. This cycle takes the first one.

### What was done

`simplification/port-mla-internals.py` adds six records to the MLA attention path:
5011 query latent, 5012 query, 5013 kv latent, 5014 expanded, 5015 attention,
5016 gate. Applied to 91 generated sources; the 24 MLA pods carry four marks each.
Twelve-prompt fidelity gate with tracing off: all twelve match baseline. Then one
fresh capture of the france prompt.

### What the capture showed

The M is no longer opaque. Pod 47 at position 4: the attention arrives at norm
19.3666 and goes straight to the gate. **There is no per-head normalize on the MLA
path** — no counterpart to KDA's local 5007. The three K pods of the same group
arrive at 0.0794, 0.0621 and 0.0906 and are rescaled ×70.5, ×102.3 and ×130.3.

Across all 92 attention pods the KDA rescale factor spans **1.6 to 190.1, median
67.3**. From one group it looked like a fixed rescale. It is not.

### Then the capture contradicted three things I had written down

**The pod-to-file mapping.** I wrote `pod = owner - 1` into a new reader from memory.
It dropped 45 of 92 pods and reported no error — 230 K rows and 5 M rows out of an
expected 340 and 120. The files carrying MLA internals are exactly `{3,7,…,91} ∪ {92}`,
which is the MLA set, so **the file index is the pod number, no shift**. The two older
scripts had it right; only the new one was wrong.

**69 KDA.** The capture has **68** files with KDA internals. Files 0 and 93 carry
neither kind — 35 and 13 records, no local 0, no local 6. So the note "pod 0 is a plain
K with a dense MLP" is contradicted by this capture. What files 0 and 93 are is not
established here, and I have not refilled that gap with the old claim.

**`rel_attn` in the twelve-prompt census.** It is `|attention − incoming| / |incoming|`,
but the attention local holds the attention output *alone* and **replaces** the residual
rather than updating it. Subtracting two different things gives ≈1.0 whenever the
attention is small. Measured side by side at position 4: `|attention| / |input|` spans
0.0050 to 1.6131 while `rel_attn` spans 0.8632 to 1.6302. The group table
"attention movement 0.999 → 1.007 → 1.014 → M 1.048" keeps its ordering but its
magnitudes do not mean what the column name says, and "attention spread only 1.22×
across groups" is compressed by the same transform.

I tested the obvious correction rather than assuming it: the attention output is **not**
orthogonal to the residual (cos −0.4640 to 0.5099, median 0.1857), so `sqrt(1+r²)` is
off by up to 0.27 too.

### What the stack looks like with no lens on it

Laying out `|input|` for all 92 pods at one position shows the residual climbing for
eleven pods and collapsing at the twelfth, seven times. Walking the boundary pod stage
by stage gives the mechanism, and it is not what my stage map said.

**Stage 20 replaces the residual at exactly `{12,24,36,48,60,72,84}` and adds at the
other 85 pods.** Tested as `|s20 − attention|` being either 0.0 or exactly `|input|`:
seven replace, 85 add, zero ambiguous, at all five positions. That is the "85 fold" term
in the 269-stage reconciliation, now observed rather than inferred.

**The dropped residual is not discarded — it becomes a new fold source.** Fold widths
run 2,2 through pods 2–11; at pod 12 fold 38 sees **3** while fold 37 still sees 2;
3,3 until pod 24; 3,4 at pod 24; and so on to 9,9 from pod 85. Seven boundaries, nine
sources. The boundary pod closes one run and opens the next inside the same pod.

**Boundary router spread falls monotonically with depth**: 15.98, 15.56, 15.28, 14.82,
12.83, 11.13, 9.72, against 16 for a perfectly uniform choice over the 16 selected.

### What broke when tested

Two things I read off pod 12 alone did not survive the other six.

- "The KDA normalize gain is ~1.6 at the boundary" — it is 1.62, 4.23, 6.49, 4.44,
  38.19, 47.37, 69.03. It rises with depth and overlaps the other pods (boundary max
  69.03 against other min 5.85).
- "The boundary pods are the lowest in the stack" — pods 15 and 19, both M, sit with
  them at four of five positions. Boundary median out/input is 0.0156–0.0350 against
  0.2248–0.4312 elsewhere, which is a separation of medians with overlapping tails, not
  a clean split.

### What this does not establish

One prompt, five positions, unless a number says otherwise. The twelve-prompt gate was
a fidelity check with tracing off, not twelve captures. No timing measured. The 80
expert stages are still invisible — that hole is now the only one named and unfilled.
Why the router spreads widest at the boundary, and why the fold-source count is the
thing that grows, are both unexplained.

---

## Cycle: 23 × KKKM + 1 M, and what repeats with it (2026-10-13)

### Context

> so there is a repetition that's common across all layers only the first part has s and
> last part have M all other are K K K M

Right, and the config makes it exact over all 93 pods: **`KKKM` × 23 (groups 0–22, pods
0–91) + `M` × 1 (pod 92).** 24 MLA, 69 KDA. **There is no irregular group in the
attention pattern at all.**

Both irregularities are at the ends and neither is an attention one:

- **pod 0, the first K of group 0, is the server.** Its attention is ordinary KDA. What
  differs is the MLP — `first_k_dense_replace = 1`, so a dense MLP rather than 16-of-896
  routed experts.
- **pod 92 is a 24th M with no group**, three KDA short of one.

The previous cycle's tree wrote group 0 as `(s K K M)`. **That was wrong** — the server
shows as different only because it is owner 0 writing a different set of locals.

### Outcome: the shape repeats 23 times, the behaviour does not

Median over 12 prompts and 80 positions, 21 interior groups:

| | min | median | max | spread |
|---|---|---|---|---|
| **attention movement** | 0.964 | 1.012 | 1.176 | **1.22×** |
| fold | 0.821 | 2.375 | 4.032 | **4.91×** |
| MoE movement | 0.324 | 0.491 | 0.833 | 2.57× |
| router spread | 10.32 | 11.91 | 15.27 | 1.48× |
| routed/shared | 0.833 | 1.267 | 1.777 | 2.13× |

**Attention movement is the only quantity the same in every repeat.** Everything else
varies 1.5× to 5× between structurally identical groups. **Group 23, the lone trailing M,
is an outlier on routed/shared at 3.522** — double any other group.

### The block shows through as a period-3 signal

Three groups make a block, so group index mod 3 says where in the block it sits; mod 3 =
0 starts the block and holds the write.

| group mod 3 | attention | fold | MoE | spread | r/s |
|---|---|---|---|---|---|
| **0** starts block | 1.041 | **1.246** | **0.641** | **13.67** | 1.237 |
| 1 | 1.007 | 2.619 | 0.440 | 12.52 | 1.202 |
| 2 | 1.008 | **2.951** | 0.434 | **11.65** | 1.250 |

**Seven of the eight lowest fold medians across all 23 groups are mod 3 = 0**; the eighth
ranks ninth, so all eight write-groups are the nine lowest.

### Reading

Three levels nest and each shows something different. **Group**: attention flat across
all four slots and all 23 repeats, MLA folds ~2× any KDA in its group. **Block**: fold
rises 1.25 → 2.62 → 2.95 across the three groups while MoE movement and router spread
both fall. **Stack**: each block ×1.35, each output one of the nine fold sources.
Attention is the invariant; everything downstream of it varies with position.

### What is still unknown

Group 0's median covers pods 1–3 only, so the dense MLP is not measured and group 0 is
not like-for-like. The period-3 fold signal is partly mechanical — the write pod folds
exactly 0.000, which must pull its group's median down — but MoE movement and router
spread also move monotonically across the three and that is not mechanical. No mechanism
for either.

---

## Cycle: the flow as a tree (2026-10-13)

### Context

> don't go to older understanding understand based on the current data and arrage it in a
> tree, KDA KDA KDA MLA is one block and you find the data and see how the tree looks

For MLA to be last in KKKM, groups align to pod ≡ 3 (mod 4) — (0,1,2,3), (4,5,6,7), … —
which puts the **server at pod 0 as the first leaf**, not a prefix outside the structure.
That re-aligns the block too: the write condition is `pod % 12 == 0`, so a block is
12B…12B+11 with the **write at its FIRST pod**. Every earlier cycle here used pods 1–12
as block 1, which put the write at the end and cut each group in half. 12 = 3 × 4, so the
aligned block is exactly three groups.

### Outcome: every level checked against the 7,360-row census

**23 complete groups, 0 are not KKKM.** Incomplete: `(server,K,K,M)` at pods 0–3, and pod
92 alone.

**Every block is exactly three groups.** Blocks 0–6 are (KKKM)(KKKM)(KKKM); block 7 is
(KKKM)(KKKM)(M).

**Snapshot slot k+1 is the output of block k — exact on all seven.** 112.8885/112.8886,
63.1325/63.1326, 34.1738, 59.8421/59.8422, 133.4077/133.4080, 50.9069/50.9070,
63.5632/63.5633. With slot 0 the embedding and block 7's output the residual, **the nine
fold sources are the input plus the output of each of the eight blocks.** 1 + 7 + 1 = 9.
That is the whole tail fold explained by the tree.

**The MLA pod has the largest fold movement in its own group of four in 1,375 of 1,760
groups (78.1%)**, median roughly 2× every KDA slot in every prompt (france 2.19/2.22/1.97
vs **4.85**).

**The group has internal direction**, n=7,360: attention movement 0.999 → 1.007 → 1.014 →
**1.048**, MoE movement 0.459 → 0.490 → 0.515 → **0.515**, router spread 12.817 → 12.620 →
12.379 → **12.295**. Rising, rising, falling.

### Reading

The group is the attention period, the block is the residual period, and they nest. The
claim that broke last cycle returns in a form that survives: "the three largest folds are
offsets 3, 7, 11" was false because offset 9 outranks offset 3 globally — but that was
the wrong comparison. **Within its own group** the MLA is the peak. Choosing the unit the
creator named turned a failed claim into a measured one.

### What is still unknown

78.1% is not a rule — one group in five has a KDA folding harder than its MLA. Why the
router spread falls across a group while both movements rise. Nothing here says why the
layer layout is K,K,K,M.

---

## Cycle: the block claims against twelve prompts (2026-10-13)

### Context

> do the same for multiple prompts

The block fold came from one prompt at one position, and the standing caveat said so.
Twelve prompts, 80 positions, 7,360 pod-positions, 560 full blocks — captured one at a
time and deleted after census, so peak disk stayed at one dump instead of the ~24 GB all
twelve need together. Each claim written as something that can fail.

### Outcome: two survived, two broke, one of those backwards

**Offset 12 folds nothing — exact.** 0 of 560 block-write rows have non-zero fold
movement.

**269 arithmetic stages per position — exact and prompt-independent.** 24 + 68 + 85 + 92
on all 80 positions, from 1-position `hello` to 16-position `genesis`.

**A block is ×1.35 end to end — holds and tightens.** Geometric mean 1.347 over 560
blocks; every prompt lands in **1.288–1.372** while individual blocks span **0.0715 to
22.05**. A property of the aggregate, not of any block.

**Monotone growth offset 1→11 — fails as a rule.** 484 of 640 block-positions, 75.6%.
Failures cluster: `water` holds five of the eight worst.

**The three largest fold movements are the MLA offsets — fails, 10 of 12.** Offset 7 is
largest 10 times and top-three 11 times; offset 11 top-three 11 times; **offset 3 only
twice**, beaten by offset 9, a KDA offset, nine times. What is true: **the fold is a
local maximum at each MLA offset — above both neighbours at 3, 7 and 11 — in 11 of 12
prompts**, and the offset ordering has rank correlation median 0.955 between prompt
pairs. The shape is near prompt-invariant; the ranking was an artifact of one position.
Over all five positions of france the top three is 7, 11, 9, not the 3, 7, 11 I read at
position 4 alone.

**Routing is not block-structured — wrong, and backwards.** Median router spread declines
through the block to a minimum at offset 10 or 11 and then **jumps at offset 12, the
highest in all twelve prompts without exception**. Rank correlation between prompt pairs
median 0.787. I called it unstructured from one prompt's scatter and read noise as the
answer. **The router spreads most widely exactly where the residual has just been
replaced** — no mechanism established.

**Attention moves the residual by about its own length whatever the type**, n = 7,360:
MLA median 1.044 (p25–p75 1.003–1.128), KDA median 1.002 (0.980–1.057), KDA the longer
tail at max 2.502.

### Reading

The two claims that survived are structural — a block write replaces rather than adds, and
the stage machine runs a fixed set regardless of input. The two that broke are the ones I
read off magnitudes at a single position.

### What is still unknown

Why the router spreads widest at the block write. Why a quarter of blocks drop somewhere
in their rise. Twelve prompts is twelve, and six share the shape `1008 10484 318 X 387`.

### Next

Unchanged: instrument `routed` and `shared_output` per expert stage. 80 of a pod's 120
stages are still invisible to every instrument in the tree.

---

## Cycle: the unit was wrong, twice (2026-10-13)

### Context

> why did you look at 92 layers where there 96 from input to output and many layers repeat

Both halves land. The chain *was* instrumented end to end in the cycle below — embedding,
layer 0, pods, normalisation, head. But every aggregate used the pod as its unit, so the
two ends got a handful of rows against a 92-row table. And the middle is not 92
independent layers: I had the 12-period in my own norm trajectory and wrote it as "the
residual collapses at every twelfth pod", an event inside a sequence, rather than as the
boundary of a unit that runs eight times.

### The repeating unit, from the config rather than memory

`text_config`: `attn_res_block_size` 12, `first_k_dense_replace` 1, `num_hidden_layers`
93, `linear_attn_config.full_attn_layers` = 4, 8, … 88, 92, 93. Pod N is model layer N+1,
so MLA pods are 3, 7, … 91, and 92 — **block offsets 3, 7, 11 in all seven full blocks**,
and 3, 7, 8 in the short block of eight where pod 92 is MLA out of turn. Three things
repeat and none is the pod: the 12-pod block, the attention type at period 4, the
120-stage machine. 12 = 3 × 4, so a full block is three KDA-KDA-KDA-MLA groups.

### Outcome

**The fold is governed by block offset, not depth.** Its three largest relative movements
are exactly the three MLA offsets — 4.330, 4.964, 3.670 against 1.0–3.6 at every KDA
offset. Invisible in a 92-row table because the MLA pods are scattered through it.

**Offset 12 folds nothing, in all seven full blocks** (relative movement 0.000). The
residual is replaced rather than added at a block write. Previously only inferred from
the count 85 = 92 − 7; now shown per offset.

**A block is close to a no-op in magnitude.** The residual swings ×4 to ×110 inside a
block and the block hands on ×1.377 geometric mean over the seven full ones. The growth
is internal and the write discards it. Every block grows monotonically offset 1 → 11.

**Routing is NOT block-structured.** Effective spread 6.3–16.0 scattered with no offset
pattern; routed/shared ratio likewise. Worth stating because everything else is.

### A second correction to my own count

`local 2` moves **0.000 at every offset** — stage 2 is `memcpy(residual, input)`, so it is
bit-identical to `local 0`. My 361 measured it against `local 1`, the leftover. Exact
reconciliation against the trace:

    452 trace hash changes = 92 (local 2 vs the leftover)
                           + 269 (arithmetic: 92 attention + 85 fold + 92 MoE)
                           + 91 (local 120 bookkeeping)

**269 stages per position change the carried residual by arithmetic**, of 9,956 recorded.
Three per pod, two at the seven block-write pods. That is the third revision of this
count in three cycles; the reconciliation is what makes this one checkable.

### What is still unknown

Why the growth magnitude inside a block varies ×4 to ×110 while the shape does not. Why
routing alone ignores a structure everything else follows. Seven full repeats is seven
samples, not a population.

### Next

Unchanged from below: instrument `routed` and `shared_output` per expert stage. 80 of a
pod's 120 stages are still invisible to every instrument in the tree.

---

## Cycle: observing the flow without the answer (2026-10-13)

### Context

The creator named the method, not the target:

> Observe the complete input-to-output flow without using the expected output token to
> judge or interpret internal computations. Capture what actually happens at every stage,
> including state changes, routing, and intermediate values. Separate observation from
> interpretation and judgment. Let evidence reveal patterns; do not force observations to
> confirm expectations.

The previous cycle used the answer token as its instrument throughout — *does this source
pick 316, does RMS change the answer, is the answer already decided before the tail*.
Every reading was a test of my expectation rather than a description of the system. The
correction is methodological and it applies to the whole arc, not to one claim.

### Execution

One unrestricted capture: every owner, every local, five positions, 94 files, 1.6 GB, plus
a full trace. Then an inventory **before** reading any value, because what the engine
records and what my patch scripts put there have disagreed before.

### Outcome

**Every pod records the carried state at every stage it executes.** 7168 floats per stage,
not a hash. The stage walk had been reading hashes of this for weeks.

**Three records per pod are bookkeeping, not work**, read off `transformer_process` rather
than inferred: `local 0` is the pod's input written outside the loop; `local 1` is the
previous position's leftover because stage 1 writes `incoming`, not `residual`; `local 120`
is labelled one position ahead because stage 120 increments the counter first. The honest
split at position 4 is **361 changed, 9,319 bit-identical, 276 bookkeeping** — not the
"636 move the residual" I have now reported three times. I had already catalogued the
stage-120 off-by-one in memory and reintroduced it anyway.

**Four stages per pod change the carried state**: the input copy, one attention stage
(MLA 24, KDA 68), the attention fold (85 = 92 − 7 block writes), the MoE add (92).

**The two instruments agree exactly once bookkeeping is removed.** On the same run and
position the trace counts 452 residual-hash changes, of which 91 are `local 120`;
452 − 91 = **361**, which is what the vectors give. A 64-bit hash and 7168 floats, built
at different times for different questions, land on the same set of stages.

**The chain joins are exact.** Pod 1's input is bit-identical to the server's output; pod
92's output is bit-identical to what the tail receives; the tail's snapshot slot 0 is
bit-identical to the server's input. Every one of the eight snapshot slots equals, to four
decimals, the residual at the last pod before a block write.

**The residual norm collapses at every twelfth pod.** The six largest contractions in 91
transitions are exactly pods 24, 48, 36, 12, 60, 72. The mechanism is in the source from
the first implementation, so this validates the walk rather than discovering anything.

**Routing is not concentrated.** Effective spread 4.79–15.92 of 16; largest single weight
usually ~0.11; five positions share a mean 2.30 of their 16 choices; at 91 of 92 pods **no
expert is chosen by all five positions**; 50–74 distinct experts per layer.

**The two MoE paths are near-orthogonal** (median cos 0.0898) and comparable in size
(median ratio 1.415; routed is the smaller one at 22 of 92 pods).

**The attention fold is close to depth-invariant.** Non-final sources sit near 1/n; the
current residual takes 2.1–2.4 times a non-final share, at every depth from 2 sources to 9.

**The raw embedding carries no preference the head can read**: entropy 12.00551 against a
maximum of 12.00672, and all 163,840 logits within 1.0 of the top.

**The two tail steps differ in kind.** The nine-source fold rotates barely (cos 0.9888)
and shrinks 16%. The RMS gain rotates substantially (cos 0.7610), shrinks 28%, and the
top-two margin rises 1.55 → 3.68 while the vector gets *shorter* — which a scaling cannot
do.

### What is still unknown

Whether owner 1's missing fold-score and router records are an instrument asymmetry or a
model one. What the expert stages do, which no instrument here watches — the residual is
the wrong buffer for that question. Why snapshot 5 is by itself the most concentrated of
the eleven head rows.

### Next

Instrument `routed` and `shared_output` per expert stage. That is the one place in the
chain where 80 of a pod's 120 stages are invisible to every instrument now in the tree.

---


## Cycle: the two ends of the chain (2026-10-13)

### Context

The stage walk covered pods 1 to 92. The creator pointed out that this is not the chain:

> because there are stages before 92 layers total 96 layers from input to token vectors
> then vectors in layer 0 then layer 1 to 92 then normalization then vector token to
> output so look at them fully observer don't assume look at one step at a time

Neither end had ever been instrumented. Layer 0 is the server, a flat sequence rather
than a stage machine, and the only layer with no routed experts. The tail is owner 93:
a nine-source fold, an RMS, and the head that turns a vector into a token. Both were
read from source first, then traced.

### Execution

`port-server-trace.py` (locals 5201–5207) and `port-tail-trace.py` (5301–5305,
5310–5317), both behind `clover_vectors_active`, both idempotent. `build.sh` needed
`-Wno-unused-function` on the server line. **Gate with tracing off: 12/12 prompts
unchanged**, plus the server's own reference gate, *layer-0 residual and S0 exact at 5
positions*, on both its prompts.

### Outcome

**Layer 0 almost discards the raw embedding.** Its two-source fold puts 0.94%–8.2% on
the incoming embedding across sixteen positions, median about 3%. The aggregate norm is
below the attention-residual norm at every position, so part of what the embedding
contributes is cancellation.

**The fold's first score is a function of the token alone.** Genesis repeats token 276
at positions 1, 5, 8, 11 — identical embedding norm, identical normalized norm,
identical score[0] of 0.1594 at all four. Score[1] moves with history. Same for 316 and
13618.

**Snapshot slot 0 at the tail is the raw token embedding, bit identical.** cos
1.000000000, max abs difference 0, against the server's *input* at the last position.
Against the server's *output*: cos 0.012. The raw table row travels the whole depth
untouched and is given 0.22% of the final fold.

**The last two blocks carry the snapshot side.** Slots 6 and 7 take 13.6 of the 15.5
points the snapshots get between them on genesis; the same shape holds on all twelve.

### The mistake in this cycle, and what fixed it

On genesis the residual, the folded vector and the post-RMS vector all pick " and", so I
wrote *the answer is already decided before the tail runs*. Eleven more prompts, one
fleet pass each, say otherwise:

- **The nine-source fold never changed the token, 12 of 12.** It moves confidence — n2
  goes 0.554 to 0.158 — and not the choice.
- **`normalization_apply_rms` changed the token in 5 of 12.** Without it germany answers
  " the" instead of " Berlin", water answers "," instead of " Celsius", nonsense "\n"
  instead of " f", french "." instead of ".\",\n", question "?" instead of "?\n\n". In
  every one of the five the pre-RMS vector picks something generic and the
  per-coordinate gain moves it to the specific continuation.

Genesis is one of the seven where the false sentence happens to hold. The standing
caveat on this work — twelve prompts on greedy top-1 is weak — caught this one going the
other way: the small set was enough to break the claim, not to support it.

Ids decoded with `server/decode-ids.c`, which calls the engine's own `id_to_word`.
tiktoken is not installed on this host and a guessed id→word mapping is not evidence.

### The whole chain, walked

With both ends instrumented the walk finally covers the chain the creator described.
At genesis position 15: the embedding lookup, six steps in the server, 9,864 recorded
pod stages of which **544 move the residual**, sixteen tail stages of which **two** do —
the fold and the RMS gain, the same two the twelve-prompt comparison picked out — and
then the projection.

This corrects an earlier figure of *636 of 9,956* with a `local 0` among the movers.
There is no local 0 in the trace; the lowest local any pod records is 1. The earlier
count compared within each pod rather than along one continuous sequence. Both came off
real runs; the grouping differed, not the engine.

> **Superseded by the cycle above (2026-10-13).** Both 636 and 544 count bookkeeping
> records as movement. `local 120` is the position counter already incremented and
> `local 1` is the previous position's leftover. The cross-validated figure is 361.

**Where the time is.** Pods 92.381 s, tail 0.392 s, request 93.891 s wall. The head —
163,840 dot products of width 7168 — is 0.390 s of that, 0.4% of the request and 99.5%
of the tail. The seven locals that move the residual are 20.6% of pod time; locals
21–115, the expert band, are 78%, and local 116, which folds the expert output into the
residual, is 0.038 s.

That is **not** evidence that 78% is removable. Stage 116 is `residual += routed +
shared_output`; the expert stages write to other buffers, so the trace's residual hash
cannot see them by construction. Same trap as the earlier reading of pod 92's eighty
expert stages showing no movement. What it establishes is where the time sits, and that
the instrument is watching a buffer those stages do not touch.

### What is still unknown

Why slot 0's weight jumps an order of magnitude on the two shortest and two least
ordinary prompts (0.068 nonsense, 0.043 question, 0.021 hello, 0.014 n2, against
0.002–0.008 elsewhere). Whether the fold is removable — an unchanged token can sit on a
margin that moved, and n2 lost two thirds of its probability. No timing was measured for
any of this.

### Next

The head is 0.4% of the request, so vector-to-token is not where a saving lives. The
expert band is 78% and the residual hash cannot see it. Instrument `routed` and
`shared_output` per expert stage rather than the residual, so the question "did this
expert change anything" can be asked of the buffer it actually writes.

---

Two things were open: why the gate's alignment flips sign with depth, and why pod 12 is
so far beyond the shallow group it sits in. The first observation came free. Pod 12's
alignment is 1.685, 1.700, 1.685 and 1.696 across positions and 1.693 on a different
prompt — under one percent of variation in a quantity the engine computes from the
input. Pod 68 swings 0.535 to 0.882 over the same positions. Something there is not
responding to the input at all.

### Two candidate explanations, distinguished rather than chosen between

Either the gate vector itself is near-fixed at pod 12, or the alignment is carried by
heads that happen to be stable. These are separable, so they were separated.

**The gate vector is not unusually constant.** Cosine between positions 2 and 8: 0.990
at pod 12, 0.982 at pods 13, 24 and 48, 0.839 at pod 68. Stable nearly everywhere and
pod 12 is barely the top of a flat field. That explanation is out.

Per head it is the reverse of what I expected. Pod 12's **best** heads are modest, a
maximum of 1.689 where pod 72 reaches 3.150 and pod 10 reaches 2.322. Its **worst** head
is the outlier: 0.637, where pod 24 falls to 0.063, pod 48 to 0.102, pod 72 to 0.267.
Pod 12 has no strongly anti-aligned head at all.

### A median of 1.04 cannot make a layer alignment of 1.70

So the energy has to be sitting somewhere particular. Comparing the plain mean of
per-head alignments against the same mean weighted by each head's share of attention
energy:

| pod | heads>1 | mean | energy-weighted | ratio |
|---|---|---|---|---|
| 6 | 56/96 | 1.115 | 1.150 | 1.031 |
| 10 | 55/96 | 1.027 | 1.066 | 1.037 |
| **12** | 52/96 | **1.141** | **1.362** | **1.193** |
| 24 | 10/96 | 0.751 | 0.146 | **0.194** |
| 36 | 21/96 | 0.886 | 0.289 | 0.326 |
| 48 | 20/96 | 0.855 | 0.234 | 0.273 |

Two independent factors. Pod 12 is the only layer high on both: the most aligned gate on
average and the strongest concentration of attention energy into the heads the gate
opens on. Pods 24, 36 and 48 are the mirror image, their energy sitting exactly in the
heads the gate shuts.

### The thing that actually separates pod 12 only shows on a second prompt

| pod | mean, gen → fre | heads>1, gen → fre | min, gen → fre |
|---|---|---|---|
| 6 | 1.115 → 0.973 | 56 → **39** | 0.531 → 0.364 |
| 10 | 1.027 → 0.986 | 55 → **40** | 0.503 → 0.372 |
| **12** | **1.141 → 1.148** | **52 → 53** | **0.637 → 0.665** |
| 72 | 0.820 → 0.758 | 22 → 10 | 0.267 → 0.259 |

On genesis alone, pods 6 and 10 look like pod 12's neighbours in kind. On french they
lose fifteen to seventeen aligned heads. Pod 12 does not move: across five positions and
two prompts, mean 1.086 to 1.148, ratio 1.185 to 1.213.

Had I stopped at one prompt I would have written "pod 12 is the strongest of a
co-aligning shallow group", which is true and misses the point. The group is not stable;
pod 12 is. That is the fourth time this session a second prompt has changed a conclusion
rather than confirming it.

### Where this stops, and why it is a different kind of stopping

The chain is now complete and every link is measured: a block-write layer passes its
attention output on undiluted; pod 12's gate is aligned with that attention rather than
reshaping it; the alignment is the product of two factors it is uniquely high on; and
all of it is prompt-invariant because no head is strongly anti-aligned.

What is left is why the weights are that way. That is a training question. No amount of
reading activations will answer it, and the training setup is not public. This is a
limit of the method rather than a gap in the work, which is a different thing from the
three previous stopping points.

---

## Cycle: looking around the question instead of at it (2026-10-12)

Direction was to stop attacking the question head on and look for the thing several
stories might share. Two moves came out of that, and the first one cost nothing: the
engine computes `gate = attention · sigmoid(Wx)` and the trace already held both the
attention and the result, so dividing one by the other recovers the gate itself without
recording anything new. The second was to stop sampling where I had happened to look —
nine pods, seven of them deep — and patch all 68 KDA pods so the gate could be seen
across the whole depth.

All 68 patched and rebuilt; eight of the nine sampled layers re-verified with tracing
off. The ninth, pod 1, cannot be gate-tested at all: the harness asserts `layer>=2`.
That is a limit of the test, not a failure of the patch, and worth not mistaking for one.

### The previous answer was wrong

"Pod 12's gate is open" did not survive the wider profile. Its **mean** gate is 0.5031,
and pods 10, 22, 56 and 90 are all higher. But pod 12 passes 0.901 of the norm while pod
24, whose mean gate is 0.4570, passes 0.100. Both cannot be true unless the gate is
correlated with where the attention has magnitude.

So compare the RMS gate weighted by attention energy, which is exactly the surviving
norm fraction, against the unweighted RMS. Above one, the gate opens on the coordinates
the attention is using; below one, it closes on them.

| pods | mean alignment |
|---|---|
| 1 to 23 | **1.016** |
| 24 to 46 | 0.710 |
| 47 to 69 | 0.735 |
| 70 to 92 | 0.717 |

The gate's alignment flips sign with depth. Pod 12 is the extreme of the network at
**1.700** against a next best of 1.149, and it replicates at 1.693 on a second prompt.
Every block-write pod replicates within 0.05.

And the other block-write pods do the opposite of pod 12, sitting below their own
neighbours rather than above: 24 at 0.214 between 0.952 and 0.817, 36 at 0.362 between
0.818 and 0.746, 48 at 0.293 between 0.785 and 0.843.

### The common thing underneath

Nothing exotic is happening at pod 12. Two ordinary properties meet there and only there:

1. A **block-write layer** is the only place the attention output survives as the layer
   output instead of being swamped by the carried residual. That is Eq. 10's i = 1 case,
   and it is in the sources.
2. The **gate co-aligns with the attention in the shallow layers** and anti-aligns
   deeper. That is not in any source; it is measured here.

Pod 12 is the only block-write pod inside the co-aligning region and simultaneously the
strongest co-aligner in the network. Its attention is already the most position-constant
measured, its gate preserves that instead of reshaping it, and the block write hands it
on undiluted.

Part of that is arithmetic: block writes are twelve apart and the co-aligning region is
about twenty-three pods wide, so it was always going to contain exactly one. What is not
arithmetic is that the one it contains is the strongest co-aligner of all 68 while its
six structural peers sit at or below their neighbours.

### What I had to withdraw, again

This is the second rider in two cycles that replicated poorly. Last cycle the claim was
"the experts only slightly reduce the agreement", from one pair. This cycle it was "pod
12's gate is open", from nine pods chosen badly. Both headline findings survived; both
attached explanations did not. **Sampling where the phenomenon is does not test the
explanation of why it is there** — that needs the layers where it is absent, which here
meant all 68.

### Still open

Why the alignment flips sign with depth, and why pod 12 sits so far beyond the rest of
the shallow group. Facts about learned weights, measured and replicated, not explained.

---

## Cycle: tracing inside the attention (2026-10-12)

Previous cycle stopped at a boundary: explaining pod 12 needed the KDA internals, the
trace records only the residual, and adding tracing inside the attention is a change to
running code. The creator lifted it — "we are not running anything in production" — so
the work was done.

### How it was kept safe anyway

Backed up before touching anything, `tar` in the same script. Every record sits behind
`clover_vectors_active`, so with `CLOVER_VECTORS` unset not one float is computed.
Patched by an anchored idempotent script, built with the same `-Werror` flags as
`build.sh`, and **every patched pod re-ran its reference gate with tracing off**: nine
pods, nine `VERIFIED`. The arithmetic is unchanged, which is the only basis for trusting
what the new records say.

Two mistakes on the way in. `Path.read_text(newline=)` needs Python 3.13 and the host is
older, which failed before writing anything. Then the `output` anchor matched twice: the
MLA path is compiled out but still present in every pod source and ends with the same
three lines. The anchor was moved onto the per-head normalise above them, which only the
KDA path has. Same trap as the earlier `transformer_qkv` port, and the backup meant the
first failure cost nothing.

### It is the output gate

Nine quantities recorded per position: query, key, value, write strength, decay, the
attention output raw and after its per-head norm, the gated value, and a per-head
Frobenius norm of the recurrent state. Against pod 24 as control, cos(p2, p8):

| quantity | pod 12 | pod 24 |
|---|---|---|
| query | 0.877 | 0.829 |
| key | 0.724 | 0.710 |
| value | 0.977 | 0.922 |
| decay | 0.998 | 0.996 |
| attention raw | 0.908 | 0.963 |
| attention normed | 0.989 | 0.955 |
| **gated** | **0.996** | **0.513** |
| state norm | 0.980 | 0.986 |

Nothing upstream separates them. The recurrent state is not the answer, the decay is not
the answer. The **output gate** is: pod 24's gate destroys the constancy its own
attention had, pod 12's preserves it.

Which also kills the guess I would have made from the equation alone. The interesting
quantity in Eq. 1 looked like the decay — and the decay is 0.998 against 0.996, no
difference at all.

### Tested on nine pods, and only partly holding

Measuring the gate's transmission as the norm ratio across it:

| pod | attention normed | after gate | gate passes | output |
|---|---|---|---|---|
| 10 | 0.255 | 0.261 | 0.655 | 0.285 |
| **12** | **0.989** | **0.996** | **0.901** | **0.999** |
| 13 | 0.705 | 0.479 | 0.384 | 0.491 |
| 24 | 0.955 | 0.513 | 0.100 | 0.537 |
| 36 | 0.928 | 0.589 | 0.152 | 0.719 |
| 48 | 0.893 | 0.425 | 0.129 | 0.593 |
| 60 | 0.711 | 0.651 | 0.298 | 0.728 |
| 72 | 0.769 | 0.735 | 0.284 | 0.835 |
| 84 | 0.710 | 0.631 | 0.288 | 0.710 |

Pod 12 is extreme at both stages: the most constant attention before the gate, and a
gate that passes 0.901 where every other pod passes 0.100 to 0.655.

But the simple version does not survive. Transmission does not predict the loss
monotonically: pod 13 passes 0.384 and loses 0.226, pod 60 passes 0.298 and loses 0.059.
Gate openness is part of it, not all of it. And two ordinary layers is too thin to claim
block-write layers generally have more constant attention, however suggestive it is that
the four earliest block-write pods are the four highest in that column.

### Where it stops now

The question has moved from "why does pod 12 behave unlike its structural peers" to
"why is pod 12's output gate open when every other measured gate is mostly closed".
That is a fact about one layer's learned weights. It is a better question than the one
before it and it is still not answered.

---

## Cycle: the traces against the official report (2026-10-12)

Everything so far was measured without reading what Kimi says the model is. The report
is at `k3-official/k3_tech_report.txt` (an extraction of the PDF) and the authoritative
numbers are in `k3tok/config.json`. Reading both turns a pile of observations into a
mapping, and the mapping is testable.

### The indexing, resolved and then verified

`num_hidden_layers` is **93**, not 92, and `first_k_dense_replace` is **1**, so model
layer 1 is the dense layer. That is the server. Therefore **pod N is model layer N+1**.

That is an inference, so it was checked. `full_attn_layers` lists
`4 8 12 ... 88 92 93`, which under the mapping predicts MLA pods
`3 7 11 ... 91 92`. Asking the traces which pods emit a KDA attention record returns
exactly 24 MLA pods, `3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87
91 92`, and 68 KDA pods. The config lists 69 KDA layers, one of which is layer 1, the
server. 68 and 68.

### Names for things I had only measured

| what I observed | what the report calls it |
|---|---|
| 12 layer segments, residual resetting at each | **Block Attention Residuals**, `attn_res_block_size: 12` |
| 8 segments, last one short | "8 blocks with 12-layer size, giving a partial final block" |
| tail folds **nine** sources | "9 total blocks when counting the embedding layer" |
| snapshot 0 created before layer 2, unchanged for 91 layers | `b0 = h1`, "the token embedding is always included as a source" |
| 3 KDA then 1 MLA, repeating | "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio" |
| top 16 of 896, 2 shared | `num_experts: 896`, `num_experts_per_token: 16`, `num_shared_experts: 2` |
| router weights summing to 1 | `moe_renormalize: true` |
| scorer ignores source magnitude | `ϕ(q,k) = exp(q⊤ RMSNorm(k))`, "the RMSNorm prevents layers with large-magnitude outputs from dominating the weights" |
| fold weights barely move across prompts | "layer-specific **learnable** pseudo-query `ql = wl`" |

The RMSNorm line is the explanation for something recorded two cycles ago as surprising.
At the tail, position 0, the two largest sources, 399.78 and 332.98, receive weights
0.0164 and 0.0023, while `b0` at 2.04, the smallest, receives 0.0047. Magnitude does not
drive the weight because the key is normalised before the dot product.

### Equation 10, tested

The report gives, for the i-th layer of block n, a value set of `[b0..b(n-1)]` when
i = 1 and `[b0..b(n-1), b(i-1,n)]` when i >= 2. The traces record the fold weight vector,
so the **count** of sources is directly observable, and the two folds per layer, one
before attention and one before the MoE, should differ by exactly one at a block's first
layer.

Twenty-one pods were checked across every block boundary. **Twenty match exactly**,
including each first-layer case: pod 12 has 2 attention sources and 3 MoE sources, pod
24 has 3 and 4, pod 84 has 8 and 9.

The twenty-first, pod 1, records no fold weights at all. That is not an architectural
difference: `transformer-1` contains **4** `clover_vectors_write` calls where every other
pod has **6**, missing the fold-weight and router writes. The fold outputs are present,
so the folds happen. It is an instrumentation gap in a file already known to be odd.

### The chain, confirmed across two implementations

The report says the output layer aggregates the block representations. The tail walk,
run through the reference, gave nine source norms. The production fleet, a different
binary on a different path, gives the pod outputs. For france at position 0:

| source | predicted origin | tail (reference) | pod (engine) |
|---|---|---|---|
| b1 | pod 11 = model layer 12 | 137.7729 | 137.7729 |
| b2 | pod 23 = model layer 24 | 332.9793 | 332.9793 |
| b3 | pod 35 = model layer 36 | 136.5090 | 136.5090 |
| b4 | pod 47 = model layer 48 | 272.0080 | 272.0080 |
| b5 | pod 59 = model layer 60 | 399.7753 | 399.7753 |
| b6 | pod 71 = model layer 72 | 57.0181 | 57.0181 |
| b7 | pod 83 = model layer 84 | 64.5007 | 64.5007 |
| b8 | pod 92 = model layer 93 | 88.4760 | 88.4760 |

Every one of `b1` to `b7` is the output of the **last** layer of a complete block, and
`b0` is 2.0446, the token embedding.

### What the report explains, and what it does not

It explains the reset, which is not a reset at all: the first layer of a block simply has
no current-block partial sum to attend to, so what I called "stage 20 does not add the
carried residual back" is Eq. 10's i = 1 case. It explains the 12-layer period, the nine
tail sources, the embedding as a permanent source, the 3:1 attention ratio, and why the
tail scorer ignores magnitude.

It does not explain either anomaly. Pod 12 is model layer 13, the first layer of block 2,
and pod 48 is model layer 49, the first layer of block 5 — structurally the same kind of
layer, yet only pod 12 produces a near-constant attention output, and only position 1 at
pod 48 takes a near-fixed value across unrelated prompts. Nothing in §2.1 or §2.2
distinguishes them. Those remain measured and unexplained.

---

## Cycle: different prompts, and one claim that did not survive (2026-10-12)

Everything up to here came from five prompts that were all English, all exactly five
tokens, and all with an obvious completion. Five more were added that vary length,
language and whether the text means anything: `Hello` at one token,
`What is the square root of sixteen` at seven, `aaaa bbbb cccc dddd eeee` at nine, a
French sentence at fifteen, and the opening of Genesis at sixteen. They answer `,`, `?`,
` f`, `."` and ` and`.

### What held

**Layer 12's attention output is near constant far beyond where it had been checked.**
Genesis has sixteen positions; every previous prompt stopped at four. From position 2
onward the cosine against position 8 runs 0.9989 to 0.9998 and the norm sits between
3.81 and 3.90. Position 0's norm is 2.6386 against france's 2.6367 and water's 2.6384,
for entirely unrelated first tokens.

Across prompts at the same layer, position 0 agrees at 0.9991 to 0.9998 and position 4
at 0.9949 to 0.9994, for pairs as unlike as biblical English against French against
`aaaa bbbb cccc`.

**The segment structure is not an artefact of short English prompts.** Genesis at
position 15 reproduces it exactly: resets at every twelfth layer, 128.8 at layer 12
against france's 112.9, 141.5 at layer 60 against 133.4.

**The layer 48 spike at position 1 reproduces precisely.** france 314.56, thethe 316.58,
genesis 318.39, against 62 to 66 at the neighbouring positions. Three unrelated prompts
within 1.2 percent of each other. Layer 47 agrees too: france 158.65, genesis 162.35.
Still unexplained, but now firmly a real and prompt independent feature.

### What did not hold

Last cycle I wrote that at the boundary "the MoE slightly reduces the agreement". That
was generalised from a single pair. Layer 12, position 4:

| pair | input cos | attention | after MoE |
|---|---|---|---|
| france/water | 0.194 | 0.99936 | 0.9899 |
| france/japan | 0.922 | 0.99997 | 0.9977 |
| genesis/french | 0.174 | 0.99939 | 0.9907 |
| **genesis/nonsense** | 0.202 | 0.99502 | **0.7234** |
| **french/nonsense** | 0.142 | 0.99493 | **0.7187** |
| **nonsense/question** | 0.156 | 0.99498 | **0.7158** |

Every pair involving the nonsense prompt falls to about 0.72; every pair without it
stays above 0.99. The attention contracts universally and that part stands, including
for nonsense. What does not stand is the implication that the experts are incidental:
they usually preserve the contraction and sometimes undo most of it.

So the layer 13 collapse is not a uniform near identity. It is attention contracting,
then the MoE either preserving or breaking that, then the boundary declining to add the
carried residual back. The collapse is strongest when the MoE output happens to agree.

### Still open

Why layer 12's attention collapses to one direction, which these traces cannot show.
Why position 1 spikes at layer 48 to within 1.2 percent across unrelated prompts. And
now a third: why the MoE at layer 12 treats the nonsense prompt differently when the
attention does not. The router weights are recorded at local 3000, so that last one is
reachable without touching production code.

---

## Cycle: chasing the thing I could not explain (2026-10-12)

The open question was why what continues past a segment boundary is so nearly the same
across different tokens. Layer 12's input is cos 0.9377 between france and water and its
output is 0.9995, so whatever happens, happens inside that layer. The trace records the
residual after every stage, so the stage can be named rather than guessed.

### It is the attention, not the experts

Layer 12, position 0, france against water, at every traced local:

| local | cos | norm A | norm B |
|---|---|---|---|
| 0 input | 0.937746 | 137.77 | 132.34 |
| 6 | 0.937746 | 137.77 | 132.34 |
| **9 attention out** | **0.999560** | **2.6367** | **2.6384** |
| 116 after MoE | 0.999459 | 9.6944 | 9.6793 |

The MoE slightly reduces the agreement. The attention produces it.

### Not a position 0 effect, which is how I had been framing it

| position | input cos | attention out cos |
|---|---|---|
| 0 | 0.937746 | 0.999560 |
| 1 | 0.341983 | 0.999291 |
| 2 | **0.217315** | 0.999384 |
| 3 | **0.187627** | 0.999418 |
| 4 | 0.194068 | 0.999360 |

Inputs that disagree at cos 0.19 produce attention outputs that agree at 0.9994. Every
position does it, not just the first.

### Attention contracts prompt differences everywhere

Across twenty-two layers at position 4, the attention output is more prompt-similar than
its own input in **twenty-one of them**, typically about twice, and 4.94 and 5.15 times
at layers 11 and 12. The one exception is layer 27 at 0.58.

That gives the mechanism for the layer 13 collapse, and it needs both halves:

- A normal layer adds the carried residual back at stage 20, and that residual is large
  and prompt specific, so the contraction is swamped. Layer 11 position 4 takes an input
  at cos 0.1916 and hands on 0.194. Nothing happens.
- A boundary layer does not add it back. Layer 12 position 4 takes 0.194 and hands on
  **0.990**, because all that survives is the contracted part.

### Layer 12 is singular, and that part is not explained

The obvious next thought was that boundary layers have near constant attention. They do
not. Cosine between positions 1 and 4 of the attention output, within one prompt:

| layer | boundary | p1~p4 | p2~p4 |
|---|---|---|---|
| 10 | no | 0.217 | 0.482 |
| **12** | **yes** | **0.998** | **0.9996** |
| 24 | yes | 0.545 | 0.744 |
| 48 | yes | 0.552 | 0.596 |
| 84 | yes | 0.475 | 0.693 |

Every other boundary sits in the same range as ordinary layers. Layer 12 alone produces
what is close to a fixed direction, and it does so in all five prompts to three decimal
places: p1~p4 runs 0.9964 to 0.9981, p2~p4 runs 0.9982 to 0.9996. Its output is also
unusually small, 3.88 at position 4 against 31.45 at layer 10.

So the sharpness of the layer 13 collapse is a property of layer 12 specifically, which
is consistent with later segment starts collapsing far less: 0.98 to 0.99 at layer 13,
but only 0.40 to 0.64 at layer 25.

### Where this stops

Why layer 12's attention collapses to one direction cannot be answered from these
traces. The KDA recurrent state, the decay and the per head scores are not recorded;
only the residual after each stage is. Answering it would mean adding tracing inside the
attention, which is a change to production code rather than a read, and that is a
decision to take deliberately rather than slide into.

The layer 48 spike at position 1 is untouched by this cycle and remains unexplained.

---

## Cycle: start at one token and climb (2026-10-12)

### A ladder that makes a prediction instead of an assumption

Five nested prompts, `The`, `The capital`, and so on to the full five tokens. Each is a
strict prefix of the next, so if the model is causal then every position must be
identical in every run that contains it. That is testable, so it was tested first.

**Position 0 is cos 1.000000 against the five token run at every layer checked, for
every prefix length.** The record counts line up too: the one token run holds 110
records at position 0 and only the two record p+1 snapshot artefact above it.

The answers also chain. `The` alone produces ` `, `The capital` produces ` of`,
`The capital of` produces ` the`, `The capital of France` produces ` is`. Each is
exactly what the five token run predicted at that position.

### The spike at position 1 is positional, not lexical

`The the the the the` puts the same token, ` the`, at positions 1 to 4. Same token,
different amounts of context:

| layer | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| 48 | **316.58** | 51.59 | 45.22 | 52.08 |
| 60 | **154.65** | 119.87 | 112.98 | 124.30 |

Position 1 runs six times hotter than positions 2 to 4 at layer 48 while being the same
token, so the spike belongs to the position. France's position 1 is ` capital` and
spikes to 314.56 at the same layer; this one is ` the` and spikes to 316.58. Different
words, nearly the same magnitude, and the cosine between them there was already measured
at 0.999909.

### Where the first token stops being itself

Scanning layer by layer, position 0, two prompts with different first tokens:

| layer | 11 | 12 | **13** | 14 | 15 |
|---|---|---|---|---|---|
| cos | 0.9887 | 0.9377 | **0.9995** | 0.9973 | 0.9998 |
| norm | 204.96 | 137.77 | **9.69** | 11.59 | 48.78 |

Layers 1 to 12 oscillate between 0.79 and 0.99 and never settle. From layer 13 onward
the cosine never drops below 0.997. The transition is the first segment boundary.

### The mechanism, after getting it wrong twice

Stage 20 is `residual += incoming`. At a normal layer it restores the carried residual
after attention has overwritten the slot; at a boundary layer it does not.

| | layer 11 normal | layer 12 boundary | layer 23 normal | layer 24 boundary |
|---|---|---|---|---|
| input | 204.96 | 137.77 | 242.65 | 332.98 |
| attention out | 26.04 | 2.64 | 12.64 | 0.42 |
| **after stage 20** | **196.96** | **2.64** | **240.10** | **0.42** |
| after MoE | 137.77 | 9.69 | 332.98 | 0.97 |

So the accumulated residual is not destroyed by some extra operation. It is simply not
added back, having already gone into the snapshot, and what continues past the boundary
is only that layer's own attention and MoE output. That output is small and nearly
orthogonal to the input, and for an early position it is nearly the same whatever the
token was.

**Two filtering mistakes, the same mistake twice.** I first reported stage 20 as "absent
entirely" at layer 12. It was present; my filter kept only records with a nonzero max
change and a no-op stage 20 has none. I then re-ran with a filter that dropped records
whose norm equalled the previous one, which hid it again. A filter designed to find
change cannot be used to prove something did not happen.

### What this does not establish

Why the continuation is token independent is still not explained. The segment structure
says the accumulated content leaves at the boundary, which accounts for the reset and
for identity being dropped, but not for why what remains is so nearly the same across
different tokens. Five prompts, English, five tokens or fewer. The positional spike at
layer 48 is observed on two prompts and has no explanation here at all.

---

## Cycle: one token at a time (2026-10-11)

### What the previous cycle got wrong

The last cycle summarised position 4 only \u2014 the position the answer is read from \u2014 and
treated the other four as setup. That hides the thing worth watching: each token has its
own trajectory through the ninety-two layers, and they are not alike. The creator said
so plainly; the traces had already been deleted, so this is a fresh capture of five
prompts through the same production fleet.

### The first token is not like the others

Residual norm entering each layer, france, all five positions:

| layer | p0 `The` | p1 ` capital` | p2 ` of` | p3 ` France` | p4 ` is` |
|---|---|---|---|---|---|
| 12 | 137.77 | 119.23 | 109.79 | 121.09 | 112.89 |
| 24 | **332.98** | 61.33 | 71.13 | 56.63 | 63.13 |
| 36 | **136.51** | 32.87 | 34.30 | 35.51 | 34.17 |
| 48 | **272.01** | **314.56** | 64.98 | 64.83 | 59.84 |
| 60 | **399.78** | 157.54 | 112.46 | 131.88 | 133.41 |
| 72 | 57.02 | 25.70 | 48.02 | 52.34 | 50.91 |
| 84 | 64.50 | 48.00 | 63.40 | 68.92 | 63.56 |
| 92 | 78.20 | 36.15 | 46.48 | 49.42 | 51.64 |

The twelve-layer segment reset holds for every position. But position 0 runs three to
five times hotter through segments two to five and only comes back to parity by segment
six. Position 1 spikes once, in segment four, reaching 314.56 against 59.84.

### Recency survives the whole depth

Cosine against the last token, same layer, within france:

| layer | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| 13 | 0.337 | **0.981** | **0.992** | **0.994** |
| 48 | 0.049 | 0.120 | 0.580 | 0.708 |
| 92 | **0.085** | 0.459 | 0.611 | **0.768** |

The ordering p3 > p2 > p1 > p0 holds almost everywhere: the nearer a token is, the
closer its direction. By layer 92 the first token is nearly orthogonal to the last.

Layer 13 is the exception and the interesting one. One layer after the first segment
reset, positions 1 to 4 are almost the same vector, 0.981 to 0.994. Position 0 does not
join them, staying at 0.337.

### Causal masking, verified rather than assumed

france and japan share `The capital of`. Positions 0, 1 and 2 are **cos 1.000000 with a
maximum coordinate gap of 0.0000** at every layer checked. france and thethe share only
the first token, and position 0 is likewise exactly identical at layers 12, 48 and 92.
A token depends on itself and what came before it, and nothing else, measured rather
than assumed.

### Early positions converge to a token independent state in mid depth

This was one surprising pair, so it was replicated before being believed. Position 0,
two prompts with **different** first tokens:

| layer | `The` vs `Water` | `The` vs `Once` |
|---|---|---|
| 1 | 0.793 | 0.874 |
| 13 | 0.9994 | 0.9993 |
| 48 | 0.999986 | 0.999990 |
| 60 | **0.999999** | **0.999999** |
| 84 | 0.904 | 0.835 |
| 92 | 0.931 | 0.935 |

Two independent pairs, near identical curves, with norms agreeing too at layer 60:
399.7753 against 399.9065. Then the representations separate again in the last segment.

It is strongest where there is least context. Using thethe, which shares position 0 with
france and differs from position 1 onward:

| layer | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| 13 | 1.000000 | 0.977 | 0.937 | 0.862 | 0.720 |
| 48 | 1.000000 | **0.999909** | 0.488 | 0.207 | 0.307 |
| 92 | 1.000000 | 0.642 | 0.714 | 0.623 | 0.613 |

Position 1 reaches 0.999909 at layer 48 with tokens ` capital` and ` the`. Position 2
reaches 0.488 at the same layer, position 3 reaches 0.207. The fewer preceding tokens a
position has, the more its mid depth representation stops depending on which token it is.

### What this does not establish

Five prompts, five tokens each, English, all with an obvious completion. The position 0
convergence is replicated on two pairs; the positional gradient rests on one pair. The
shapes are measured and the cause is not \u2014 in particular nothing here explains why
layer 13 collapses token identity, why position 0 runs hot through the middle segments,
or why the last segment re-separates what the middle ones merged.

---

## Cycle: six prompts through the real engine (2026-10-11)

### Why this run differs from the last one

The previous observation went through the reference harness, which is pinned to two
prompts: `Xm` in `reference-all.inc` is stubbed out and the expert outputs are read from
oracle files that exist only for france and japan. So the reference cannot answer "what
does a different prompt do".

The production fleet can. `pipeline-serve` takes `<wanted> <ids...>` on stdin, and
`clover_vectors_open` opens one file per owner and only filters when
`CLOVER_VECTORS_OWNER` is set \u2014 so leaving it unset traces **all ninety-two layers in one
run**, about 1.5 GB. Everything below is the real engine, server plus 92 layers resident,
not the reference.

Six prompts, each five tokens, each tokenised by the native tokenizer. Every answer is
the obvious one: `The capital of France is` \u2192 ` Paris`, `Japan` \u2192 ` Tokyo`, `Germany` \u2192
` Berlin`, `Once upon a time there` \u2192 ` was`, `Water freezes at zero degrees` \u2192
` Celsius`, and the degenerate `The the the the the` \u2192 ` the`.

### The model is eight segments, not ninety-two layers

Reading the residual norm entering every layer, at the position the answer comes from:

```
 1:0.79   2:1.74   3:3.40 ...  11:96.76  12:112.89   <- segment end
13:3.89  14:7.51  15:10.40 ... 23:52.57  24:63.13    <- segment end
25:0.46  26:0.96  27:1.75  ... 35:22.94  36:34.17    <- segment end
37:0.58 ... 48:59.84 | 49:0.52 ... 60:133.41 | 61:6.75 ... 72:50.91
73:3.20 ... 84:63.56 | 85:7.36 ... 92:51.64
```

At every twelfth layer the residual **resets to near zero** and climbs again. Layer 11
hands 112.889 to layer 12; layer 13 starts from 3.889. The seven values at the segment
ends \u2014 112.889, 63.133, 34.174, 59.842, 133.408, 50.907, 63.563 \u2014 are **exactly** the
tail's snapshots 1 to 7 measured in the previous cycle, which was a different binary on a
different code path. Eight segments of twelve, the last one eight layers long, and the
tail re-weights all eight plus the final residual.

### Magnitude is architectural, direction is the content

| prompt | L12 | L24 | L36 | L48 | L60 | L72 | L84 | L92 |
|---|---|---|---|---|---|---|---|---|
| france | 112.9 | 63.1 | 34.2 | 59.8 | 133.4 | 50.9 | 63.6 | 51.6 |
| japan | 116.0 | 60.8 | 35.3 | 61.4 | 130.0 | 47.7 | 67.0 | 54.7 |
| germany | 116.0 | 60.3 | 35.0 | 60.5 | 135.2 | 48.8 | 68.8 | 52.0 |
| once | 112.9 | 64.1 | 27.8 | 57.8 | 129.1 | 45.9 | 68.0 | 47.2 |
| thethe | 106.8 | 46.3 | 31.1 | 52.1 | 124.3 | 58.5 | 67.2 | 50.0 |
| water | 122.8 | 65.0 | 34.4 | 58.3 | 130.9 | 49.2 | 71.1 | 55.3 |

Seven percent spread at layer 12, four percent at layer 60, across prompts that have
nothing in common \u2014 including a degenerate repetition. The norm trajectory is close to a
property of the architecture. Whatever distinguishes these prompts is not in how big the
vector is, so `vector-compare.c` was written to look at the angle instead.

### Similar prompts separate; unrelated prompts converge

Cosine between prompts at each layer's input, same position:

| layer | france vs japan | france vs water |
|---|---|---|
| 1 | 0.936 | 0.300 |
| 8 | **0.983** | 0.272 |
| 12 | 0.922 | 0.194 |
| **13** | **0.998** | **0.990** |
| 24 | 0.913 | 0.360 |
| 48 | 0.956 | 0.658 |
| 60 | 0.765 | 0.319 |
| 78 | **0.700** | 0.333 |
| 92 | 0.851 | 0.629 |

The two move in opposite directions. France and japan differ by one token and start at
0.936, rise to 0.983 by layer 8, then fall to 0.700 by layer 78 before recovering to
0.851. France and water share nothing, start at 0.300, and end at 0.629.

Every segment reset raises the similarity, and the first one is extreme: at layer 13
france and water are at **0.990**, having been at 0.194 one layer earlier. The effect
decays with depth and has gone by layer 73. What the residual holds immediately after a
reset is close to prompt independent.

### What this does not establish

One position per prompt, the one the answer is read from. Six prompts, all five tokens
long, all in English, all with an obvious completion. The directional comparison covers
three of the six. Nothing here says why the curves move as they do \u2014 the shapes are
measured, the cause is not. The request times the fleet prints include 1.5 GB of trace
writes and are not a measurement of anything.

---

## Cycle: what each stage does to the vector (2026-10-11)

### Intended outcome

Everything so far has been about time. This is about what the numbers become. The
machinery already existed: `clover_vectors_*` records the layer input as stage 0, the
residual after every one of the 120 stages, the snapshots at 1000+slot, the fold
weights at 2000+stage, the router weights at 3000 and the fold's output at 4000+stage.
`vector-walk.c` reads that and reports norm, cosine against the layer input, and the
largest coordinate change since the previous residual.

### The first run was worthless and I nearly reported it

I drove it with `pod-context`, which feeds **random vectors**. On random input the
router weights came out almost flat, 0.0708 down to 0.0585, and the attention
contributed 0.6% of the residual's norm. Both are properties of random input, not of
the model. Re-run through the reference harness on the real france prompt, with the
layer VERIFIED so it is the genuine computation, every one of those numbers changes.

### A KDA layer, real activations

| pos | input | after attention | output | cos(in,out) | attention cos | fold [snapshot, residual] |
|---|---|---|---|---|---|---|
| 0 | 2.3468 | 4.4235 | 8.5678 | 0.7148 | 0.586 | [0.272, 0.728] |
| 1 | 1.4939 | 2.2226 | 2.9037 | 0.7006 | 0.280 | [0.415, 0.585] |
| 2 | 1.7410 | 2.6705 | 3.5449 | 0.7058 | 0.283 | [0.229, 0.771] |
| 3 | 1.6440 | 2.4819 | 3.2713 | 0.6973 | 0.286 | [0.391, 0.609] |
| 4 | 1.7352 | 2.6456 | 3.3964 | 0.7066 | 0.288 | [0.233, 0.767] |

The layer turns its input through a near-constant angle, cosine 0.697 to 0.715, about
45 degrees, and roughly doubles the norm. **Position 0 is the outlier in every column**
— 3.65x norm growth, attention cosine 0.586 instead of 0.28, and the flattest routing —
which is what an empty recurrent state should look like.

The router genuinely discriminates on real activations, 3.3x to 8.7x between its
strongest and weakest of sixteen, against 1.2x on random input. The fold gives the
snapshot 23 to 42%, so the second channel is carrying real weight, not decoration.

**103 of the 120 stages never touch the residual.** The eighty expert stages accumulate
into `mixture` and only reach the residual at stage 116, so the residual trace has
about twelve transformation points in it, not 120.

### An MLA layer, same prompt

| | KDA layer 2 | MLA layer 3 |
|---|---|---|
| attention output cos(input) | +0.28 | **-0.10 to -0.16** |
| after attention | 0.82 to 0.88 | 0.91 to 0.93 |
| fold weight on the snapshot | 23 to 42% | **6 to 14%** |
| output cos(input) | 0.70 | 0.69 to 0.77 |
| norm growth | about 2.0x | about 1.2x |

MLA's attention output points slightly *against* its input where KDA's points with it,
and MLA's fold leans much harder on the residual. The two are doing different jobs, not
the same job at different speeds.

### A cross-check I did not plan

Layer 3's input norms at positions 1, 2 and 4 are 2.9037, 3.5449 and 3.3964. Layer 2's
*output* norms at the same positions are 2.9037, 3.5449 and 3.3964. Two dumps taken in
separate runs agree exactly, which is the chain verifying itself.

### Two artefacts, explained rather than reported

Position 0 appears twice in a layer-2 dump. `test-reference` re-runs position 0 after a
`reset_sequence` for layers 2, 3, 12 and 92 to check reset works, so the duplicate is
the test, not the model.

The snapshot records are written after the while loop, and stage 120 has already done
`sequence->positions++`, so **a snapshot produced while processing position p is
labelled p+1**. Anyone reading a dump by position would misattribute it.

Also worth noting while in there: `CLOVER_VECTORS_KEEP=weights` keeps 2000..2999 and
4000..4999 but the router weights are written at exactly 3000, so that filter drops
them. The comment above it describes having been caught by this once already.

### The tail, and the last transformation of all

Neither the tail nor the server emits vector traces, so the pods' machinery could not be
reused. The tail gate already builds the nine real sources at layer 92 and checks the
engine against the reference, so `derive-tail-walk.py` adds observation to a copy of it
and changes no assertion. `diff --strip-trailing-cr` confirms zero gate lines are missing
from the walker, and it still prints `NORMALIZATION_PASS france 5 35840 21504`.

| pos | residual | aggregate | after RMS | cos(agg,rms) | residual weight | newest snapshot |
|---|---|---|---|---|---|---|
| 0 | 88.48 | 72.37 | 62.48 | 0.952 | 0.824 | 0.113 |
| 1 | 58.74 | 42.86 | 55.17 | 0.867 | 0.708 | 0.122 |
| 2 | 75.96 | 67.29 | 49.52 | 0.819 | 0.861 | 0.081 |
| 3 | 76.72 | 64.25 | 46.16 | 0.737 | 0.809 | 0.122 |
| 4 | 79.35 | 66.28 | 47.74 | 0.761 | 0.810 | 0.141 |

Two things I would have got wrong by reasoning. **The scorer down-weights the largest
sources**: at position 0 snapshot 5 has norm 399.8 and receives weight 0.016, while the
residual at norm 88.5 receives 0.824. Magnitude is not what it selects on. And **the RMS
step rotates rather than rescales** \u2014 cosine between the aggregate and the result is 0.737
to 0.952, never 1.0, because the learned per-coordinate gains turn the vector as well as
shrinking it.

Then the head, on the vectors the verified run actually produced:

| pos | context so far | predicts | correct | top | second | margin | probability |
|---|---|---|---|---|---|---|---|
| 0 | `The` | `' '` | \u2014 | 8.62 | 8.28 | 0.34 | 1.4% |
| 1 | `The capital` | `' of'` | yes | 15.99 | 15.03 | 0.96 | 19.7% |
| 2 | `The capital of` | `' the'` | plausible | 16.34 | 13.76 | 2.58 | 29.1% |
| 3 | `The capital of France` | `' is'` | yes | 17.13 | 15.05 | 2.08 | 68.2% |
| 4 | `The capital of France is` | `' Paris'` | yes | 18.11 | 14.44 | 3.68 | 79.2% |

Confidence climbs monotonically with context, 1.4% to 79.2%, and the margin with it. At
position 2 the model's second choice is `' France'`, the genuine next token. At position 4
the runner-up is `' Berlin'`: the alternatives it considers are other capitals.

### Cross-checks nobody asked for, which is what makes them worth having

Layer 3's input norms at positions 1, 2 and 4 are 2.9037, 3.5449 and 3.3964. Layer 2's
*output* norms at the same positions are 2.9037, 3.5449 and 3.3964. Separate runs, exact
agreement.

The tail's snapshot 0 at positions 0 to 3 is 2.0446, 1.9179, 1.5717 and 1.8620. Layer 2's
*incoming* snapshot at positions 1 to 4 is 2.0446, 1.9179, 1.5717 and 1.8620 \u2014 the same
four values under the p+1 labelling described above. **Snapshot 0 is created before layer 2
and arrives at the tail unchanged after ninety-one more layers.** The snapshot channel is a
real skip connection spanning the whole model, not a local convenience.

The residual stream itself grows roughly twenty-two fold through depth, from about 3.5
leaving layer 2 to 59-88 arriving at the tail.

### What this does not establish

One prompt, five positions, two layers of ninety-three traced in detail. The regularities
are striking but they come from a single short prompt, and nothing here says they hold at
depth or on other text. Layer 0 is still unobserved this way: the server has no vector
tracing and adding it would be a change to production code rather than a derivation from
an existing gate.

---

## Cycle: the pieces I had skipped (2026-10-11)

### What prompted it

I had been reporting the chain as though it were finished. The human counted it out:
input to tokens, embedding, layer 0, 92 layers, normalisation, token to output. I had
fixed layer 0, the 92 layers and the head, and had **not looked at** the input side, the
final normalisation, or the detokenise. That was a fair catch; "nothing conspicuously
out of line any more" was a claim about the parts I happened to have measured.

### The three missing pieces, measured

**Input embedding.** `server_input` reads one 14 KB row through `table_block`, which
without a direct mapping would decompress and SHA-256 verify a 229 KB block per token.
The direct mapping is present. 148.976 us scattered, 4.080 us warm, per prompt token.
Against a 1,988 ms token that is nothing.

**Final normalisation.** `tail-cost.c` reads transformer-93's own accumulated counters
rather than subtracting one harness from another:

| | ms/token | share |
|---|---|---|
| whole tail | 50.440 | |
| normalisation and fold | **0.149** | 0.3% |
| head and detokenise | 50.290 | 99.7% |
| deliver | 0.000 | 0.0% |

The fold over nine sources of 7,168 floats is about 64k multiply-accumulates. It was
never going to be large, and it is not.

**Detokenise.** Inside the head counter, and since the head alone measures 50.0 ms it is
about 0.3 ms. `id_to_word` is a direct array index.

**Text to token ids.** `word_to_id` is one open-addressed probe into a 524,288-bucket
table plus a memcmp. There is no merge loop, and the pipeline is driven by token ids
rather than raw text, so it runs at most once per prompt word. **This one is read, not
timed**, and I am marking it as such rather than quoting a number I did not measure.

### The chain, end to end

Per generated token, everything on this box with experts pinned:

| piece | cost | state |
|---|---|---|
| text to token ids | one hash probe per word | read, not timed |
| token id to embedding | 0.149 ms scattered, 0.004 warm | measured, negligible |
| layer 0 | 26.1 ms | fixed, was 427 |
| layers 1 to 92 | 1,912 ms | fixed |
| final normalisation and fold | 0.149 ms | measured, negligible |
| vocabulary head | 50.0 ms | fixed, was 231 |
| detokenise | ~0.3 ms | measured, negligible |
| deliver | 0.000 ms | measured |
| **total** | **~1,988 ms** | was ~10,905 |

The edge instrument confirms the two edges independently through the production shared
objects: server 26.17 ms/position, tail 52.36 ms/token.

So every piece is now accounted for. Three were worth fixing; the other four are small
enough that the measurement is the whole result.

### Worth keeping

I reported a chain as finished while three of its pieces had never been looked at. The
parts I had measured were genuinely done, which is exactly what made the claim easy to
make. Count the pieces from the architecture, not from the list of things already
touched.

---

## Cycle: the server, both ends of it (2026-10-11)

### Layer 0, fixed

The four changes already proven on the pods, applied to `server.c`: `-fopenmp` on its
build line, a pragma on `server_project`, a pragma on `server_qkv`, a pragma on
`server_update_attention`, and the AVX2 gather in `server_project_row`. `server.c` does
not include `live-root.h`, so it needed its own guarded `immintrin.h`.

| | ms/position |
|---|---|
| before | 427 |
| after | **26.1** |
| single thread, same binary | 319.8 |

**16.3x**, and every checksum identical to before the change: 1.404196 at 16 positions,
0.315960 at 32, 0.272570 at 8. The single-thread control separates the two effects: the
gather alone is 422 to 320, and the threads take 320 to 26.

Validated on a gate I had to get working first, since I would not change layer 0 without
one. `server/test-reference.sh` needs four arguments: the QKV path, the prepared
trunk-0, the original index and the raw trunk, all with pinned hashes. It passed before
the change and after.

**It was run twice on purpose.** The gate compiles `test-server-reference.c` without
`-march=native`, so `__AVX2__` is undefined and it exercises the scalar fallback rather
than the gather that production runs. Rebuilt with `-march=native` it passes too, so
both code paths are covered rather than only the one the gate happens to compile.

Also green afterwards: VERIFIED 91 of 91 on both prompts and two NORMALIZATION_PASS,
because the full build rebuilt every layer as well.

### Where a token stands

Per generated token, everything measured on this box with experts pinned:

| | session start | now |
|---|---|---|
| 68 KDA pods | 9,492 ms | 1,497 |
| 24 MLA pods | 755 | 415 |
| layer 0 | 427 | 26 |
| vocabulary head | 231 | 50 |
| **total** | **10,905 ms** | **1,988 ms** |

About **5.5x**, every step bit-identical and verified against a reference.

### What the server turned out to be

Every optimisation this session landed on the 92 layer pods. The server owns layer 0
and the edges, and the tail owns final normalisation and the vocabulary head, and
**none of them were ever touched**. `server.c` has zero OpenMP regions against a pod's
five, and `pipeline/build.sh` compiles the coordinator, the server stage and layer 93
**without -fopenmp** while layers 1..92 get it. The pragmas would have been dropped
silently even if they had been written.

### The head, fixed

`server_output_project` scored 163,840 rows one after another over 2.35 GB of table,
scalar and double, once per generated token. Rows are independent and the tie rule,
highest score then lowest id among equals, is associative, so the winner cannot depend
on the split. Only the direct mapping is safe to share, because `table_load` caches one
block at a time.

| | seconds | GB/s |
|---|---|---|
| warm sequential | 0.231 | 10.15 |
| warm parallel | **0.050** | 47.16 |

4.6x, same token 70480, at the memory ceiling. An `OMP_NUM_THREADS=1` control on the
same binary gives 0.219 s, so the gain is the parallelism and nothing else.

`head-cost.c` reports 18.9x for this. That compares its first sequential call, which
also faults in the 2.35 GB mapping, against a parallel arm running afterwards on warm
pages. `head-steady.c` measures both warm. The instrument was not wrong about the
mechanism, only about the size, and I would have repeated its number.

Verified: NORMALIZATION_PASS on the tail reference, VERIFIED 91 of 91 on the layer
campaign for both prompts. The server's own reference script needs a QKV path argument
and was not run, so it is not part of this evidence.

### Layer 0, observed

`server-context.c` drives `server_process` the way `pod-context` drives a layer, so the
two are comparable. Stable across sizes: 422.1, 427.2, 428.5 ms/position at 8, 16 and
32 positions, no first-touch artefact.

**A pod layer is 22.0 ms. Layer 0 is 427. Nineteen times.**

Phases, from my own clocks in a diagnostic copy:

| phase | ms/position | share | MACs | rate |
|---|---|---|---|---|
| dense | 282.9 | 62.0% | 727M | 2.57 GMAC/s |
| qkv | 102.0 | 22.3% | 264M | 2.59 |
| attention output | 68.1 | 14.9% | 176M | 2.58 |
| update_attention | 1.5 | 0.3% | | |
| decay | 1.3 | 0.3% | | |
| normalize, aggregate, checks | ~0.6 | 0.1% | | |

97.2% is three trunk projections, and **all three run at 2.58 GMAC/s**, which is the
single-threaded scalar palette rate measured independently in `kernel-cost.c`. The pods
run the identical kernel at about 48. Layer 0 has no experts at all; it is dense, so its
1.17 GMACs are entirely trunk-style projection.

The gap is not an algorithm. It is the four things already proven on the pods and never
applied here: `-fopenmp` in the build, a pragma on `server_project`, a pragma on
`server_qkv`, and the gather in `server_project_row`. On the pods those took the same
kernel from 2.58 to about 48 GMAC/s, which would put layer 0 near 25 ms/position.

### Not done

Layer 0 itself is unchanged. The measurement says what it would be worth; applying it
is the next cycle, and it needs the server's own reference gate working first, which
currently needs an argument I have not supplied.

---

## Cycle: can the trunk be four bits (2026-10-11)

### Intended outcome

The gather left the trunk at about 50 GMAC/s against the expert path's 80, and I had
said the rest was the format. Direction was to try the data change on a small scope and
measure before doing it everywhere. The first question is not how to pack it but
whether packing it is lossless at all, because a lossy version is a model change and
not an optimisation.

### What the weights actually look like

`trunk-probe.c` reads the real trunk and counts distinct palette indices, writing
nothing. The expert path carries a selector per 32 coordinates, so 32 coordinates is
the span that would have to fit in sixteen values:

| record | shape | distinct in record | 32-coord groups needing more than 16 | worst row |
|---|---|---|---|---|
| 6 | 12288x7168 | 256 | 2,752,365 of 2,752,512 | 244 |
| 7 | 7168x12288 | 256 | 2,752,365 of 2,752,512 | 254 |
| 8, 9, 10 qkv | 12288x7168 | 255 | 2,752,473 of 2,752,512 | 236 |
| 31, 32 | 3584x7168 | 256 | 802,814 of 802,816 | 245 |
| 34, 35, 36 shared | 6144x7168 | 256 | 1,375,803 of 1,376,256 | 245 |

The worst group uses all 32 of its coordinates on distinct values, and 99.99% of groups
exceed sixteen. The weights are spread across the whole 256-entry palette with no
sub-palette structure to exploit.

**So the trunk cannot be packed to four bits losslessly.** Doing it means folding more
than sixteen distinct values into sixteen slots, which changes every affected output and
would fail the reference. The experts are four-bit because they were quantised that way
upstream and the trunk was quantised to an eight-bit palette; that is a model decision
already made, not an implementation oversight I can correct. Re-quantising the trunk is
a question about model fidelity and belongs to whoever owns that, with an evaluation
behind it, not to a performance cycle.

### The lossless alternative, also measured and also rejected

This is znver4 with full AVX-512, so sixteen indices can be gathered at once instead of
eight, and a single 512-bit accumulator maps lane j to element c+j exactly as the
scalar `lanes[j]` did. Bit-identical, mismatch 0 at every size.

| rows | scalar | gather 256 | gather 512 | expert 4-bit |
|---|---|---|---|---|
| 4,096 | 25.98 | 50.60 | 55.49 | 83.34 |
| 16,384 | 26.84 | 52.40 | 56.50 | 83.71 |
| 65,536 | 28.09 | 50.03 | 52.37 | 80.73 |

About 5 to 8% in the harness. In the layer, two runs each: AVX-512 17.163 and 17.242
ms/position against AVX2's 17.228 and 17.513. **About 1%, inside the noise.** Not
ported. Zen 4's AVX-512 is double-pumped and a gather does not widen.

That is isolation mispredicting for the third time today, and over-predicting after
under-predicting in the previous cycle. The rule that survives is only that the layer
decides.

### Outcome

The trunk projection path is finished at about 50 GMAC/s on this hardware unless the
model is re-quantised. Both remaining routes were measured and both are closed: the
format change is not lossless, and the wider gather is not worth anything in the layer.

---

## Cycle: can the trunk palette path be optimised (2026-10-11)

### Intended outcome

I had written that the trunk path runs about 37 GMAC/s where the expert path runs about
70, and implied the gap was the format. Instructed not to jump to a conclusion but to
do the runs. Both numbers were derived by dividing stage times by my own MAC counts,
and a separate run of the trunk kernel alone had given 23.8. Three numbers from three
methods is not a comparison.

### Both kernels on one harness

`kernel-cost.c` runs them side by side, same rows and width so the MAC count is
identical, swept from 0.9 MB to 235 MB:

| rows | A scalar palette | A2 gather | B 4-bit AVX2 |
|---|---|---|---|
| 256 | 25.26 GMAC/s | 45.05 | 71.30 |
| 4,096 | 28.83 | 50.43 | 82.89 |
| 16,384 | 29.06 | 50.62 | 81.93 |
| 65,536 | 30.26 | 50.69 | 79.21 |

**The ratio holds when everything fits in L2**, so it is not bandwidth and the "half
the bytes" explanation I was heading towards is wrong. It is one scalar multiply-add
and one table lookup per element against four in a vector. A2 gathers eight palette
indices at a time; mismatch against A was 0 at every size, so it is bit-identical.

### In the layer, which is the measurement that counts

MLA layer 3, same binary type, same conditions, only the kernel differing:

| stage | scalar | gather | ratio |
|---|---|---|---|
| layer | 27.336 ms/position | **17.273** | 1.58x |
| 6, the whole MLA attention | 0.6670 s | 0.3070 | 2.17x |
| 25, 111 | 0.0724, 0.0730 | 0.0327, 0.0327 | 2.2x |
| 112, 113, 115 shared expert | 0.1241, 0.1244, 0.1238 | 0.0557, 0.0560, 0.0555 | 2.2x |
| experts 30-109 | 0.5164 | 0.5182 | 1.00 |
| 24 routing | 0.0309 | 0.0304 | 1.00 |

Experts and routing are untouched, which is the check that the attribution is right.

**The layer shows 2.2x where the harness showed 1.68x.** Last cycle isolation
over-predicted; this cycle it under-predicted. The lesson is not "isolation
over-predicts", it is that isolation predicts neither direction reliably and only the
layer settles it. My harness fed uniformly random palette indices; real weights are not
uniform.

KDA's qkv gained only 1.43x against the 2.2x everything else got. The difference is
layout: `transformer_project` walks contiguous rows, while `transformer_qkv` reads
`transformer->qkv` interleaved with a 20-byte per-row prefix.

### Result

KDA layer 2 about 27.9 to 22.013 ms/position; MLA layer 3 about 27.7 to 17.296.
VERIFIED 91 of 91 for france and japan, checksums unchanged, zero build failures. Only
the tail lacks the kernel, having no trunk projection.

Across the session: a KDA layer 139.585 to 22.013, **6.3x**; an MLA layer 31.457 to
17.296, **1.8x**. Over 68 KDA and 24 MLA pods, roughly 10,247 ms/position to 1,912,
about **5.4x**, every step bit-identical and verified.

### What is still unknown

- The remaining gap to the expert kernel, 50 against 79 GMAC/s, is the format: one byte
  of palette index per weight against four bits. That is a data change, re-quantising
  the trunk, not a code change, and it has not been attempted.
- `server.c` carries the same `server_project_row` pattern for layer 0 and the
  vocabulary head. Not looked at.
- Still nothing measured under concurrency.

---

## Cycle: the last serial loop (2026-10-11)

### Intended outcome

Observe `transformer_update_attention` before touching it, and before going anywhere
near concurrency.

### What the layer looked like first

Clocked copy of the current source, KDA layer 46, 64 positions, experts pinned, 28.281
ms/position:

| stage | what | ms/position | share |
|---|---|---|---|
| 30-109 | sixteen routed experts | 8.11 | 28.7% |
| 6 | qkv | 7.33 | 26.0% |
| 9 | attention output | 4.99 | 17.7% |
| 112/113/115 | shared expert | 3.67 | 13.0% |
| **8** | **state update** | **1.66** | **5.9%** |
| 24 | routing | 0.53 | 1.9% |

Routing confirmed at 4.36 to 0.53 ms, 8.2x, from the previous cycle. Stage 8 was the
only serial hot spot left: everything above it already runs parallel.

### Observing it

Every reference in the head loop is per-head and disjoint: `recurrent[head]`, an output
block at `attention + head * 128`, and a `prediction` array local to the iteration.
6.29 MB of state over 96 heads is 65.5 KB each, so six heads per thread sit in L2.
Per-head arithmetic and order are untouched, so it must stay bit-identical.

One thing worth noticing and not acting on: the two passes over the state could be
fused, since pass one's write of the decayed value could be recomputed in pass two from
the same operands, halving the 25.2 MB of traffic. But parallel the stage is already
about 0.12 ms, so halving its traffic would save around 0.2% of the layer. Not worth the
risk of touching the arithmetic.

### Result, and a correction to my own instrument

In the layer: stage 8 goes 0.1064 s to 0.0124 s over 64 positions, **8.6x**, 1.47
ms/position, and the stage falls from 5.9% to 0.7%. Checksum unchanged.

The standalone replica had measured 15.2x and implied about 5.5% of the layer. The real
figures are 8.6x and nearer 4%. The replica ran the update repeatedly with the state hot
and nothing else competing for cache; the layer streams 281 MB of experts between
positions. **Isolation over-predicts, and the layer number is the one that counts.**

Three runs each after the change: KDA 28.167, 27.392, 28.068; MLA 28.032, 27.629,
27.425. MLA is unchanged, which it must be since the state update is KDA-only, and that
is a useful check that nothing leaked. KDA and MLA are now indistinguishable at 64
positions.

VERIFIED 91 of 91 for france and japan, zero build failures.

### Worth keeping

The port skipped transformer-1 twice. First because that file carries CRLF and my
anchor used LF. Then, after I "fixed" it by converting the anchor when the file contains
CRLF, it skipped again: **transformer-1 has mixed endings**, CRLF in the file but LF
around this particular function. The fix that works is to try both forms rather than
decide from the file as a whole. I had diagnosed the first failure correctly and then
generalised it wrongly.

### Where the session ended

At 64 positions, experts pinned: a KDA layer 139.585 to about 27.9 ms/position, roughly
**5x**; an MLA layer 31.457 to about 27.7. Over 68 KDA and 24 MLA pods that is about
10,247 ms/position to 2,562, close to **4x** on the layer stack, every step
bit-identical and verified against the reference.

### What is still unknown

- The trunk palette-gather path runs about 37 GMAC/s where the expert 4-bit AVX2 path
  runs about 70. Same machine, half the rate. That is now the largest unexplained thing
  left, and qkv plus the shared expert plus stage 9 are all on the slow path.
- Nothing has been measured under concurrency. Every number in this file is one layer
  with the whole machine to itself and its experts pinned.

---

## Cycle: survey the remaining serial loops, then the router (2026-10-11)

### Intended outcome

I had left "whether any other loop was missed the same way" as an open question after
the qkv finding. A sample is not a survey, so: list every OpenMP region in a pod against
every heavy loop, and see what is uncovered.

### What the survey showed

The whole pod had **three** parallel regions: `transformer_project`, and the two added
earlier today. Mapping those against every loop over `TRANSFORMER_ROWS`,
`TRANSFORMER_WIDTH`, `TRANSFORMER_HEADS` or `record->rows` left one obvious candidate by
measured cost: `transformer_route`, stage 24, 4.36 ms/position, and it runs on **every**
layer rather than only the KDA ones.

It scores 896 experts, each a 7168-long dependent double accumulation over full float32
router weights. That is roughly 28.7k cycles of latency per expert and 896 of them,
about 6 ms at 4.2 GHz, which is where the measured 4.36 comes from. Latency-bound, not
bandwidth-bound, so splitting across experts should scale nearly linearly.

### Why a pragma alone would have been wrong

Scoring an expert is independent, but choosing the top sixteen is not: the loop carries
`best[]`, `selected[]` and `count` across iterations as an insertion sort. A parallel
for would have raced on all three and quietly corrupted the choice, and the reference
campaign would have caught it only as a route mismatch with no obvious cause.

So the port splits the function: scoring parallel, selection serial and still walked in
index order so a tie resolves to the lower expert id exactly as before. Per-expert
arithmetic and its order are untouched. The one behavioural difference is on the failure
path, where a non-finite choice now aborts after all scores are computed rather than at
the offending expert; the return value is the same.

### Result

| layer | before | after |
|---|---|---|
| KDA layer 2 | 32.612 ms/position | **29.020** |
| MLA layer 3 | 31.048 | **28.020** |

About 3.0 to 3.6 ms off every layer, roughly 11%. Checksums unchanged, VERIFIED 91 of
91 for france and japan, zero build failures.

### Across the whole session

A KDA layer went 139.585 to 29.020 ms/position, **4.81x**; an MLA layer 31.457 to
28.020. Over 68 KDA and 24 MLA pods that is about 10,247 ms/position to 2,646, roughly
**3.9x** on the layer stack, all of it bit-identical and verified against the reference.

### Worth keeping

`transformer-1` was skipped on the first run, reported honestly by the script as "1
without it", and that contradicted my own survey which had found the block in all 92.
The cause was line endings: that file carries CRLF in this working tree, my PowerShell
survey normalised before matching and node did not. The port now matches whatever the
file actually contains. Two instruments disagreeing is a fact to chase, not a rounding
error to pick a side on.

### What is still unknown

- `transformer_update_attention` is still serial, 1.8 ms/position, now about 6% of a KDA
  layer. Measured in isolation at 15.2x available; not applied.
- The trunk palette-gather path runs about 37 GMAC/s where the expert 4-bit AVX2 path
  runs about 70. Same machine, half the rate, unexplained.
- Nothing has been measured under concurrency. Every number here is one layer with the
  whole machine and its experts pinned.

---

## Cycle: what KDA is for (2026-10-11)

### Intended outcome

Observe KDA rather than time it. I had been treating it as "the slow kind of layer",
which is backwards.

### What the System showed

A KDA layer's entire per-sequence state, read from the struct:

```c
float history[3][TRANSFORMER_ROWS][3];                                  /*   442 KB */
float recurrent[TRANSFORMER_HEADS][TRANSFORMER_HEAD][TRANSFORMER_HEAD]; /*  6.29 MB */
```

**6.73 MB, and it does not depend on the position count.** An MLA layer instead keeps
98,560 bytes *per position*. The two cross at about 68 positions, and after that MLA
only grows.

The time behaves the same way. Both layers, experts pinned, same instrument:

| positions | KDA layer 2 | MLA layer 3 |
|---|---|---|
| 64 | 32.612 ms/position | 31.048 |
| 256 | 32.246 | 31.275 |
| 1,024 | 32.302 | 32.250 |
| 4,096 | **32.313** | **35.262** |

KDA varies 1.1% across a 64-fold change in context and shows no trend. MLA rises and
the rise accelerates, because each position attends over every earlier one. They cross
near 1,024.

### Why the model is built this way

KDA is O(1) per token in both time and memory; MLA is O(N) in both. Every fourth layer
is MLA and the other three are KDA, so three quarters of the depth costs the same at a
million tokens as at sixty-four. At 1M positions that is the difference between

- 24 MLA layers x 103.3 GB = **2.48 TB** of cache, and
- 68 KDA layers x 6.73 MB = **458 MB**.

The KDA layers contribute 0.02% of the long-context memory. The whole long-context
problem lives in the 24 MLA layers.

### What this says about today's work

Both changes were right, for different regimes, and I did not understand that when I
made them.

- Head-major K/V, 23.8x on MLA attention at 1M, touches only the 24 layers whose cost
  grows with context. At long context those layers are nearly all of the time, so it is
  the right target there and nowhere else.
- The qkv pragma, 4.19x on KDA, touches 68 layers whose cost is flat. It is the right
  target at every context length, and it was worth more overall.

I spent most of the day on the MLA layers believing they were the expensive ones. They
are only expensive once the context is long, which is exactly the case the KDA layers
exist to avoid.

### Where a layer's time goes now

Per-stage, layer 46, 64 positions, after the fix: qkv 7.2 ms, the sixteen routed experts
7.5, attention output 4.9, routing 4.4, shared expert 4.4, state update 1.8. No single
dominator left; the layer is roughly balanced.

One thing stands out for later. The routed experts run their 528M MACs at about 70
GMAC/s on the 4-bit AVX2 path, while the trunk projections run at about 37 GMAC/s on the
palette-gather path. Same machine, half the rate. Not investigated.

### What is still unknown

- Whether the KDA/MLA crossover near 1,024 holds for the whole model rather than these
  two layers.
- Why the palette-gather trunk path is half the rate of the expert path.
- `transformer_update_attention` is still serial, worth about 1.7 ms/position in
  isolation, now 5.6% of a KDA layer rather than a rounding error.

---

## Cycle: why a KDA layer cost four times an MLA layer (2026-10-11)

### Intended outcome

Understand the gap rather than optimise around it. Instructed not to trust the existing
instrumentation but to see it myself, which was the right call: the shipped tracer would
have given the answer, but I would have been trusting its placement and its clock.

### Three of my explanations were wrong

I had written that layer 46 cost 132 ms/position against layer 3's 31 because it is
"KDA carrying four input snapshots". That compares two layers differing in two things at
once and attributes the result to one of them. A 2x2 across pods separates them:

| | 1 snapshot | 4 snapshots |
|---|---|---|
| MLA | layer 3: 31.457 | layer 43: 31.472 |
| KDA | layer 2: 139.585 | layer 46: 139.518 |

**Snapshot count costs nothing measurable.** The whole 4.4x is KDA against MLA.

Then, in order, measured with my own code:

- **The recurrent state walk is not it.** `kda-state.c` replicates
  `transformer_update_attention` exactly: 1.852 ms/position serial, 0.122 parallel,
  checksums equal. Under 2% of the 108 ms gap. The missing pragma on its 96 heads is
  real but small. (First attempt produced `-nan`, because the layer L2-normalises key
  and query per head and my replica did not; a checksum of NaN compares equal to
  nothing, so the instrument was fixed before its timing was believed.)
- **The projection shapes are not it.** `proj-cost.c` measures the real record shapes:
  KDA's three 12288x7168 at 3.701 ms each parallel, MLA's four at 2.35 ms total. An 8.8
  ms difference.

That left about 90% of the gap unexplained by anything I had reasoned about.

### What it actually is

A diagnostic copy of the layer with my own per-stage clocks, 64 positions, experts
pinned:

```
STAGE_TOTAL 8.4382 s
STAGE   6     6.869 s   81.4%
```

Stage 6 is 107.3 ms/position, which is the entire gap. Stage 6 for KDA is
`transformer_qkv`, and it is the one trunk projection that does **not** go through
`transformer_project`:

```c
static void transformer_qkv(...)
{
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
```

No pragma. KDA's Q, K and V weights are interleaved per row with a 20-byte prefix
holding the row scale and its convolution taps, so the loop could not call
`transformer_project` and was written out longhand. `port-parallel-project.mjs` added
the pragma to `transformer_project` and this loop was missed. 3 x 12288 row
dot-products of width 7168, on one core, per position.

MLA escapes it because `transformer_mla` uses `transformer_project`, which is parallel.

### The fix and what it gave

One pragma, `port-parallel-qkv.mjs`, 92 pods. Rows are independent: each writes its own
`qkv[component][row]` and advances its own `history[component][row]`. Per-row arithmetic
and order untouched.

| layer | kind | before | after |
|---|---|---|---|
| 3 | MLA | 31.457 | 31.395 |
| 43 | MLA | 31.472 | 31.667 |
| 2 | KDA | 139.585 | **33.253** |
| 46 | KDA | 139.518 | **33.348** |

**4.19x on a KDA layer**, every checksum unchanged, and KDA now sits level with MLA.
68 of the 92 pods are KDA, so across the layer stack that is roughly 10,242 ms/position
to 3,020, about **3.4x**. Verified: VERIFIED 91 of 91 for france and 91 of 91 for japan,
zero build failures.

### Worth keeping

Anchoring on the loop alone would have been wrong. `transformer-1` contains the same two
loop lines a second time inside its qkv validation, which returns early and cannot be a
parallel region. The port anchors on the function signature, which is unique in exactly
92 pods and absent only from the tail. Counting before acting caught it.

Everything I did earlier today was on the 24 MLA pods, which were never the expensive
ones. The instruction to understand before optimising found more than the optimisation
did.

### What is still unknown

- KDA remains 5.7% above MLA (33.3 against 31.5). The state update accounts for about
  1.85 ms of that; the rest is unexamined.
- `transformer_update_attention` is still serial, worth about 1.73 ms/position by its own
  measurement. Not applied, because it was measured in isolation rather than in the layer.
- Whether any other loop in the tree was missed the same way. Only `transformer_project`
  and now `transformer_qkv` have been checked.

---

## Cycle: remove hardmax (2026-10-11)

### Direction

From the human, after the investigation below: remove the hardmax machinery and make
layer 46 behave like every other layer. Not needed for now.

### What was done

`port-drop-hardmax.mjs` removes the comment, the `transformer_hardmax` function and its
call site in `transformer_aggregate`. `transformer_listed` stays, because
`transformer_fold_residual` and `transformer_dropped` still use it.

Scope was counted before acting rather than sampled: 93 pod sources, **92 contain
`transformer_hardmax`, exactly 2 occurrences each**; nothing in `server.c` or
`transformer-93`. Layer 1 needed its own pair of anchors — it is derived separately,
folds two sources instead of nine, and carries different wording and a different call
site. The script asserts each anchor is unique, that no reference survives, and that the
helper is still present.

92 files, 3,488 deletions. All 92 pods rebuilt with no failures.

### What the System showed

**The full campaign now passes on defaults: VERIFIED 91 of 91 for france and 91 of 91
for japan, with no `CLOVER_SOFTMAX` set.** Before this change it aborted at layer 46.

That gate had been unusable, which mattered more than the arithmetic: no regression
anywhere in 91 layers could have been detected while it was off. It is now a gate again,
and the head-major attention work is verified through it.

Stale scratch copies on the host (`transformer-3.orig.c`, `transformer-3.hm.c`) were
removed; git holds both.

### What this gave up

At most about 1.5% on layer 46, and possibly nothing — hardmax measured 132.21
ms/position mean against softmax's 134.22, but softmax's best run was 132.145, below
hardmax's mean. One layer of 93. The configuration and its measurements are preserved in
the cycle below and in the commit, so it can be reconstructed if it is ever wanted.

---

## Cycle: why layer 46 fails the reference (2026-10-11)

### Intended outcome

Understand the layer 46 route mismatch before working around it. It was suggested the
failure might be informative rather than merely broken. It was.

### What the System showed

The harness checks each layer independently against the reference's own input for that
layer, so nothing accumulates: layer 46's pod simply disagrees with the reference about
which sixteen experts to route to, from identical input.

The cause is in the pod, in plain sight:

```c
/* Defaults stay at layer 46 with both folds, the configuration measured good. */
if (!layers || !*layers) layers = "46";
stages = stages && *stages ? stages : "3,21";
on = !(off && *off && *off != '0');
```

`transformer_hardmax` defaults its owner list to the literal string `"46"`, both folds,
on unless `CLOVER_SOFTMAX` is set. Layer 46 alone replaces the fold's softmax blend with
winner-takes-all at stages 3 and 21. The aggregate changes, so `postnorm` changes, so
the router selects different experts. The harness runs `env -i`, so the experiment is
always on. That is why exactly one layer fails, and why it is that one.

Tested rather than assumed:

| run | result |
|---|---|
| layer 46, default | VERIFIED 0, aborts on the route assertion |
| layer 46, `CLOVER_SOFTMAX=1` | **VERIFIED 1** |
| all layers, `CLOVER_SOFTMAX=1` | **VERIFIED 91, last 92** |

So there is no regression anywhere. The whole campaign passes once the experiment is
off, which also means the head-major port is verified across all 91 layers, not the 24
MLA layers alone.

**What the default buys.** Layer 46 through `pod-context.c`, 256 positions, experts
pinned, four runs each:

| arm | runs, ms/position | mean |
|---|---|---|
| hardmax | 131.788, 132.162, 132.429, 132.478 | 132.21 |
| softmax | 133.140, 135.513, 132.145, 136.094 | 134.22 |

Checksums are exactly reproducible within each arm and differ between them
(-2523.027578 against -4549.359472), so the output difference is real and
deterministic. The time difference is not solidly real: softmax's best run, 132.145, is
below hardmax's mean. Hardmax is tight and softmax is noisy. At most about 1.5%, on one
layer of 93, so on the order of 0.02% of model time.

### What this is really about

The cost is not the 1.5%. It is that the full campaign has been unrunnable, so no other
regression across 91 layers could have been detected either. The gate was off and
nothing said so. That it now passes cleanly is good news bought cheaply.

The prior work behind the default is sound on its own terms: `scripts/fold-cosine.sh`
derived owner lists from cos(aggregate, residual), and `run-hardmax-cos.sh` compared
generated text. "Measured good" meant the text did not change on the prompts tried. It
did not mean the numerical reference still matched, and those are different claims.

### Recommendation, not a change made

Make hardmax opt-in: default the owner list to empty so `CLOVER_HARDMAX_LAYERS=46`
enables it explicitly. A default should be the configuration that can be verified; an
experiment should have to be asked for. I have not changed it, because it alters shipped
model output and that is a Direction decision.

### What is still unknown

- Whether hardmax at layer 46 changes generated text on prompts beyond those already
  tried. The checksum says the residual differs materially; only text comparison over a
  real prompt set answers quality, and that needs the fleet running.
- Why layer 46 costs 132 ms/position where layer 3 costs 31. Four times, and layer 46 is
  KDA carrying four input snapshots against layer 3's one. Not investigated, and it
  bears on where optimisation effort belongs, since 68 of 92 pods are KDA.

---

## Cycle: join the kernel number to a real layer, then port it (2026-10-11)

### Intended outcome

The previous cycle left its central claim unjoined: 23.8x was an isolated kernel, and
the layer had only ever run at five positions. Close that, then decide from the result
whether the other MLA pods should have the change.

### What the System showed

`pod-context.c` drives a real pod through `transformer_process` with its 15.72 GB of
experts pinned by `mlock`, so expert arrival is identical in every arm and only the
attention walk moves. Output bit-identical in every pair:

| positions | original | head-major | layer | attention component |
|---|---|---|---|---|
| 64 | 31.265 ms/pos | 30.864 | 1.01x | negligible |
| 1,024 | 40.035 | 32.106 | 1.25x | 9.1 -> 1.2 ms |
| 4,096 | **71.546** | **35.245** | **2.03x** | 40.6 -> 4.3 ms, 9.4x |

Experts cost about 30.9 ms per position whatever the length, so below roughly 1k
positions the change is noise — the cache still fits in L3. At 4,096 the old layout
spends 57% of the layer in attention and the new one 12%. That justified the port.

**Ported to all pods and verified.** `port-head-major-mla.mjs` follows the house pattern:
unique-anchor assert, idempotent, layers 2..92. The block exists in every pod but is
only compiled where `TRANSFORMER_MLA` is 1, so 24 layers change behaviour and 66 keep a
dead copy consistent rather than leaving a latent defect. All 24 MLA layers verify
against the reference for france and japan, 24 of 24, matching the pre-port baseline.

### The mistake, which was the same one twice

I shipped 90 stale sources to the host and overwrote the deployed tree.

Last cycle I found the repository was behind what runs, pulled `transformer-3.c` from
the host, and wrote in this file: *"Read the deployed artefact first, not the repository
copy."* Then I ran the port against the other 91 pods **without pulling them**, because I
had quietly assumed that fixing one file had fixed the problem. It had not: those pods
still included `root.h` instead of `common/live-root.h`, a gap of **30,159 insertions**,
not the 332 I had measured on a single file.

The build caught it — 23 pods failed with `omp_get_max_threads` implicitly declared,
which is exactly the set of MLA pods, because only they compile the block. The host was
restored from a backup taken before the extraction, rebuilt, and re-verified. Nothing
was lost, because the script took that backup before writing. That was luck of habit
rather than design, and the habit is worth keeping.

The correction: a sample is not a survey. When one artefact is found to be stale, the
question is how many others are, and that is a cheap thing to check.

### What was ruled out

**The layer 46 route mismatch is not mine.** The full campaign (`all`, layers 2..92)
aborts at layer 46 on `!memcmp(routes, captured_routes[position])`. Controlled test:
identical runs differing only in transformer-3's stage object, original and head-major
both reach VERIFIED 44, last layer 45, same assertion. So it predates this work. The
historical `remaining/validation.log` holds 182 VERIFIED lines, a full pass over 91
layers times two prompts, so layer 46's routing changed at some point between that
campaign and now — most likely in one of the ports, none of which re-ran the campaign.
Selecting only MLA layers steps past it, because the harness checks only selected layers.

### What is still unknown

- Why layer 46's routes no longer match the stored reference. Not investigated. Until it
  is, the full campaign cannot pass and only subsets can be verified.
- Everything in the previous cycle's unknowns still stands: no end-to-end long context,
  the latent K/V form unread, expert compute unprofiled at stage granularity, the DIMMs
  at 3600 against a 4800 rating.
- The 4,096-position result is one layer on one machine with experts pinned. A pod under
  real traffic shares the box.

---

## Cycle: make layer 3 use the machine (2026-10-10)

### Intended outcome

One layer — not the fleet, not the model — correct and actually using the CPU cores, the
RAM and the disk it has. Then a stage map from input through to layer 92.

### What was known going in

The standalone engine's MLA K/V cache had been changed from position-major to head-major
earlier the same day, measured at 1.68x at 1M positions on an isolated kernel. The open
question was whether the distributed layers carried the same defect.

### What was tried, and what the System showed

**The repo was not what runs.** The local `transformer-3.c` included `root.h`; the
deployed one on k3 included `live-root.h`, `stage.h` and `trace.h`. A diff put the repo
**332 insertions and 41 deletions behind** the deployed source. `common/` is untracked in
git and was never committed; the port scripts write in place but their output never
reached a commit. I had already read the stale file and begun reasoning about it. Pulled
the deployed copy down and worked from that.

**The defect was there, and worse.** `transformer_mla` walked every head across every
position at a 98,560-byte stride — K, V and the positional tail interleaved per position.
Double the standalone's stride, because here all three share one block.

**Two things were wrong, not one.** The head loop also had no OpenMP pragma: 96
independent heads on a single core while fifteen sat idle.

Measured with `tansformers/mla-stride.c`, which isolates the attention step at real
sizes. All arms produce identical checksums at every size.

At 1,048,576 positions, 103.35 GB for one layer:

| arm | time | GB/s |
|---|---|---|
| position-major serial — what the layer did | 62.40 s | 1.66 |
| position-major parallel | 5.16 s | 20.03 |
| head-major serial | 12.51 s | 8.26 |
| head-major parallel | **2.62 s** | 39.45 |

23.8x together. Layout alone 4.99x, threads alone 12.1x. The result sits at 39.45 GB/s
against a measured 42.8 GB/s RAM ceiling, so it is bus-bound.

**Thread count, measured on the real layer through the reference harness:** 1 thread
2.49 s, 4 threads 0.73 s, 8 threads 0.44 s, 16 threads 0.30 s, 32 threads 0.31 s. Every
run verified. Sixteen is the optimum for a layer alone; SMT does not help.

**Correctness.** Three gates, all passed:
- `VERIFIED` for france and japan at all five positions against the independent
  reference, for both the layout change and the parallel head loop.
- Bit-identical output over a 40-position stream against the unmodified binary. This
  matters because the cache starts at capacity 8 and the reference prompts are 5
  positions, so **the reference never grows the cache and never executes the re-layout
  code**. Forty positions crosses 8 to 16 to 32 to 64.
- `CONTROLS 3 PASS` under AddressSanitizer and UndefinedBehaviorSanitizer.

Installed as `transformer-3.c` on k3 and rebuilt as `pipeline-stage.so`; re-verified at
16 threads. The original is kept at `transformer-3.orig.c`.

### What was ruled out

**The disk is not the bottleneck, and the current I/O path is the best of three.** I
believed expert reads were running at roughly 380 MB/s against a device measured at
13.51 GB/s. That number was wrong and so was the reasoning: I divided whole-layer wall
time by expert bytes and called the quotient an I/O rate. Measured properly, with the
page cache dropped before each arm and disjoint expert ranges:

| request shape | pass 1 | pass 2 |
|---|---|---|
| mmap + `posix_madvise(WILLNEED)` — current | 8.58 GB/s | 8.06 GB/s |
| one `pread` per expert across threads | 7.02 | 7.18 |
| chunked `pread`, more requests in flight | 7.02 | 6.86 |

281 MB of experts arrives in 33 ms. The existing path beats both alternatives. There is
no gain here, and the plan to port the standalone engine's dedicated pread readers into
the distributed layer is withdrawn on this hardware.

**`OMP_NUM_THREADS=4` in HOSTING.md is not wrong.** I measured 4 threads as 2.4x slower
than 16 and nearly reported the documented guidance as a defect. It is not: that table
is a resident fleet serving 1 to 32 concurrent requests, where four threads per request
times sixteen concurrent requests already saturates the box. One layer alone is a
different regime. The thread count belongs to the deployment shape, not to the code.

### Mistakes worth keeping

- **I read the repo and assumed it was the system.** The deployed source had moved on by
  332 lines. Pull the running artefact before reasoning about behaviour.
- **I treated `--inspect` agreement as a correctness result.** It prints one summary line
  and never touches the attention path. Caught it before reporting it as a pass, but I
  had already run it as though it were a gate.
- **I assumed the existing test covered the code I wrote.** It could not: 5 positions
  never grows an 8-slot cache. A test that cannot reach the change is not evidence about
  the change.
- **PowerShell `\$?` through ssh returns the local shell's status, not the remote
  command's.** Several "exit=True" lines earlier in the session were meaningless. The
  `VERIFIED` lines came from the harness itself and were real.

### What is still unknown

- No layer has run end to end at long context. The 1M figures are an isolated kernel.
  Expert cost per position makes a full long-context layer run impractical on one box,
  so the two measurements have not been joined.
- Only layer 3 has these changes. The other 23 MLA pods are unmodified. The 24 pods are
  generated, so landing this across them is a generator change, not 23 edits.
- The latent K/V form is 2,304 bytes per position per layer against 98,560 expanded,
  42.8x fewer bytes. Since attention is now bus-bound, this is the only remaining large
  factor — and it needs the absorb matrices, which have not been read.
- Expert compute, not expert I/O, dominates a layer at short context. 80 of 120 stages.
  Not yet profiled at stage granularity on the distributed path.
- The DIMMs run at 3600 MT/s against a 4800 rating. Never investigated.
- The `detail:expert-read-gb` counter in the standalone profiler prints gigabytes through
  a milliseconds field. Still unfixed.

### What the next cycle should do differently

Read the deployed artefact first, not the repository copy, and check whether a test can
actually reach the code being changed before treating a pass as evidence. Both failures
this cycle were of the same kind: accepting a proxy for the thing itself.

### Repository state

`STAGES.md` is new. `tansformers/mla-stride.c` and `tansformers/expert-io.c` are new
instruments. `tansformers/transformer-3/transformer-3.c` now holds the deployed source
plus both changes — note that committing it mixes a 332-line catch-up with a small
functional change, and the catch-up should land separately. `common/` is still untracked
and still needs committing; it is the reason the repo cannot currently rebuild what runs.
