# The block structure, and what it explains

A correction to how the tolerance results were analysed. Not a method.

## The correction

Every result in Methods B, C and D was grouped by absolute layer number — depth bands,
shallowest-first orderings, decades. The creator stopped that:

> it's not just 92 layers, every %12 layers repeats from layer 0, take that into account

The residual does not run continuously through ninety-two layers. At every twelfth layer
the running prefix sum is replaced rather than accumulated, so each block begins again
near zero. Analysing by absolute depth measures across a boundary that resets.

The coordinates are: pod N is model layer N+1, block n = ⌈(N+1)/12⌉, and within-block
index i = (N+1) − 12(n−1). MLA sits at i = 4, 8 and 12.

## Regrouping the data already collected

No new runs. The per-layer tolerance sweep, laid out as block against position:

| block | i=1 | 2 | 3 | 5 | 6 | 7 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | . | 0.9933 | 0.9947 | 0.9927 | 0.9935 | 0.9823 | 0.9943 | 0.9925 | 0.9947 |
| 2 | 0.9991 | 0.9969 | 0.9973 | 0.9981 | 0.9992 | 0.9959 | 0.9978 | 0.9970 | 0.9958 |
| 3 | 0.9987 | 0.9979 | 0.9991 | 0.9987 | 0.9981 | 0.9975 | 0.9966 | 0.9944 | 0.9954 |
| 4 | 0.9967 | 0.9968 | 0.9978 | 0.9945 | 0.9968 | 0.9971 | 0.9960 | 0.9963 | 0.9965 |
| 5 | 0.9958 | 0.9952 | 0.9969 | 0.9953 | 0.9963 | 0.9911 | 0.9890 | 0.9910 | 0.9890 |
| 6 | 0.9858 | 0.9938 | 0.9923 | 0.9962 | 0.9974 | 0.9947 | 0.9982 | 0.9956 | 0.9956 |
| 7 | 0.9920 | 0.9921 | 0.9915 | 0.9958 | 0.9946 | 0.9940 | 0.9956 | 0.9954 | 0.9966 |
| 8 | 0.9689 | 0.9936 | 0.9911 | 0.9607 | 0.9641 | 0.9524 | . | . | . |

Separating the two effects:

| | range | spread |
|---|---|---|
| by block | 0.97178 to 0.99746 | **0.02568** |
| by position within block | 0.98811 to 0.99537 | 0.00726 |

**The block effect is three and a half times the position effect.** Which block a layer
belongs to matters; where it sits inside that block barely does.

## And the block effect has a structural cause

Each block's output `b_n` is one of the nine sources the tail folds, and those weights
were measured two cycles earlier. Sorting blocks by the weight their output receives:

| block | tail weight of its output | mean tolerance |
|---|---|---|
| 4 | 0.001419 | 0.99650 |
| 2 | 0.002288 | **0.99746** |
| 3 | 0.002434 | 0.99739 |
| 6 | 0.015247 | 0.99440 |
| 5 | 0.016354 | 0.99328 |
| 1 | 0.020787 | 0.99225 |
| 7 | 0.112503 | 0.99415 |
| 8 | **0.824227** | **0.97178** |

Almost monotonic. The three blocks whose outputs the tail nearly ignores are the three
most tolerant of damage; the block whose output carries 82 percent of the tail's weight
is by far the least. Block 7 is the one inversion, sitting higher than its 0.1125 weight
would suggest.

Block 8 is also the partial block, six KDA layers rather than nine, and its output is
the final residual. Damage there reaches the answer essentially undiluted.

## What this fixes about the earlier work

Three measures had failed to predict tolerance: how constant the substituted value was,
the size of the injected error, and the share of router weight. A fourth — absolute
depth — looked weakly predictive and was an artefact of averaging across block
boundaries.

The block's tail weight predicts it. That is a property of the architecture, not of the
layer, and it was measurable from data already on disk.

It also explains, after the fact, why the tolerance-ordered cut in Method C selected
pods 12 to 30: those are blocks 2 and 3, the two most tolerant. The ordering found the
structure empirically without knowing why.

## What it does not fix

The cut still failed on French at both sixteen and thirty-two layers. Knowing which
blocks tolerate damage does not make the damage additive, and nothing here measures what
happens when several layers in a tolerant block are substituted together. Block 7
remains unexplained.

Tail weights were measured at one position of one prompt. They vary — the residual's
share ran 0.708 to 0.861 across five positions — so the ordering above is not fixed
either.
