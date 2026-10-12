# Method A: drop the older block snapshots

**Status: inconclusive. The instrument did not do what I assumed, and the control caught
it. No conclusion about snapshots should be drawn from this run.**

## What motivated it

At the tail, the nine sources are weighted by softmax and the weights are lopsided: the
final residual takes 71 to 86 percent, the newest block snapshot 8 to 14 percent, and
the other seven share about 5 percent between them. At position 0 the two largest
sources by norm, 399.78 and 332.98, receive weights 0.0164 and 0.0023.

If seven of nine sources contribute 5 percent of the tail's aggregate, perhaps they can
go.

## What was run

`common/trace.h` already carries an ablation hook written for this kind of question:

> Experiment only, off unless `CLOVER_FREEZE` names a capture prefix: replace this
> layer's incoming snapshots, from slot `CLOVER_FREEZE_FROM` upward, with the ones a
> different prompt produced. If the output does not move, those slots were not carrying
> anything the answer depended on.

Twelve prompts, baseline against `CLOVER_FREEZE=/tmp/kda-genesis/t`,
`CLOVER_FREEZE_FROM=0`, `CLOVER_FREEZE_TO=6`. Genesis was included deliberately as a
no-op control: it is the source of the frozen snapshots, so its own answer must not move.

## Result

| prompt | baseline | method A | cos at pod 92 |
|---|---|---|---|
| france | ` Paris` | ` the` | 0.189 |
| japan | ` Tokyo` | ` the` | 0.192 |
| germany | ` Berlin` | ` the` | 0.200 |
| water | ` Celsius` | ` the` | 0.170 |
| **genesis (control)** | ` and` | **` the`** | **0.187** |
| hello | `,` | ` the` | 0.922 |

Every prompt collapsed to the same token. One of twelve matched, and that one was
`thethe`, whose baseline answer was already ` the`.

## Why this is not a finding about snapshots

**The control failed.** Genesis is the freeze source; freezing it with its own
snapshots must be a no-op, and it was not. That says the instrument is not doing what
the comment describes, so nothing else in the table can be interpreted.

Reading `clover_freeze_load` rather than its comment:

```c
static float clover_frozen[8][7168];
...
if (header[1] >= 1000 && header[1] < 1008 && header[3] == 7168 && !clover_frozen_have[header[1] - 1000]) {
```

It keeps **one vector per slot**, taken from the first matching record in the file, and
`clover_freeze_apply` writes that same vector at **every position**. The real snapshots
differ per position. So the hook does not swap one prompt's snapshots for another's; it
replaces every position's snapshot with a single fixed vector, destroying positional
information across the whole network.

Collapsing every position onto one snapshot and watching the model collapse to one
answer is not evidence that snapshots carry the answer. It is evidence that the
experiment was wrong.

The hook is exact only for single-position prompts. The comment is not wrong about its
intent, but it does not mention the restriction, and I did not check the code before
relying on it.

## What would make this a real test

Either extend the hook to store per-position snapshots and index them by position, or
run it only on one-token prompts where the distinction vanishes. Neither was done here.

## What it cost, and what it bought

One nine-minute run. It bought the control: without genesis in the set I would have read
the collapse as "the older snapshots carry the answer after all, despite 5 percent of the
tail weight", written it up as a surprising finding, and been wrong. The 5 percent
reasoning may still be wrong for a different reason — the snapshots feed every layer's
AttnRes fold, not only the tail's — but this run says nothing either way.
