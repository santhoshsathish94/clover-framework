# Observing the stage machine, the two paths, and each token

A Clover context file for the cycle that stopped grouping.

## Intended outcome

Stop building predictors and look at what the traces hold. Direction was explicit: look
by stages rather than by layers, look at the trunk, look at the routed experts, look at
the tokens, each individually and without grouping.

## What was known going in

Four candidate predictors of substitution tolerance had been measured and all four
failed: how constant the substituted value was, the size of the injected error, the
share of router weight, and absolute depth. A fifth, the tail weight of a layer's block,
tracked it well. Every one of those was an aggregate over something.

## What was tried

`stage-walk.py` emits one line per recorded stage, in engine order, with the residual
norm and the distance moved since the previous stage. Global stage for pod L local s is
120 + (L-1)*120 + s. One token, 9,956 recorded stages, no grouping.

Then the two MoE paths per pod, and then each of genesis's sixteen tokens separately at
four pods.

## What the system showed

### The residual is a thin spine

**636 of 9,956 recorded stages move the residual. Six point four percent.**

| local | pods where it moves |
|---|---|
| 0, the handoff | 91 |
| 1 | 92 |
| 2 | 92 |
| 6 (MLA) | 24 |
| 9 (KDA) | 68 |
| **20** | **85** |
| 116 | 92 |
| 120 | 92 |

Six events per layer. Pod 92 raw: stages 11061 to 11115, the whole eighty-stage expert
computation, every one reading `moved 0.0000  cos 1.000000`, and then stage 116 delivers
all of it in a single step.

Local 20 moving at 85 of 92 pods is 92 minus 7, exactly the block-write pods. That was
established separately and falls out of the stage walk without being looked for.

### The two paths, per pod

Routed over shared runs from **0.52 to 3.84**. The routed path usually dominates, but at
pods 74, 78, 90 and 32 the trunk is larger. At pod 92, the last layer before the tail,
it is 3.78.

At every block-write pod **both paths nearly vanish**:

| pod | routed | shared | neighbour pod | routed | shared |
|---|---|---|---|---|---|
| 12 | 0.345 | 0.480 | 11 | 25.28 | 29.97 |
| 24 | 0.147 | 0.242 | 23 | 15.15 | 7.52 |
| 36 | 0.256 | 0.330 | 35 | 13.10 | 13.29 |
| 48 | 0.245 | 0.133 | 47 | 23.60 | 20.42 |

Two orders of magnitude, at both paths at once.

### Each token, and the first two are not like the others

Sixteen tokens of genesis, separately:

| pod | token 0 | token 1 | tokens 2 to 15 |
|---|---|---|---|
| 12 | routed **8.79**, R/S 7.89 | routed 0.695 | routed 0.320 to 0.451, R/S 0.47 to 0.72 |
| 47 | shared **124.3**, R/S 0.20 | shared **174.4**, R/S 0.11 | shared 17.6 to 27.2, R/S 0.75 to 1.41 |
| 59 | shared **177.9**, R/S 0.12 | shared 57.3, R/S 0.40 | shared 30.8 to 45.8, R/S 0.88 to 1.47 |
| 92 | R/S **7.95** | R/S 2.82 | R/S 2.56 to 4.87 |

At pod 12 token 0's routed path is twenty-five times any other token's. At pods 47 and
59 it is the **shared** path that is inflated at tokens 0 and 1, five to ten times.
So the first tokens are not merely larger; which of the two paths carries them differs
from layer to layer.

Tokens 2 through 15 are close to uniform everywhere.

## What this rules out

That the early-position behaviour recorded in earlier cycles is a single phenomenon. It
had been described as "early positions converge to position-determined states", measured
on residual norms. Seen per path it is not one thing: at some layers it is the trunk
that inflates for tokens 0 and 1, at others the routed experts.

## What is still unknown

Why both MoE paths collapse together at a block-write layer. The input to the MoE is the
postnorm of a residual that has just been replaced by a small attention output, so a
small input is expected; whether that fully accounts for two orders of magnitude is not
established.

Why the inflated path at tokens 0 and 1 is the trunk at pods 47 and 59 but the routed
experts at pods 12 and 92.

Whether tokens 2 to 15 being uniform holds on prompts that are not sixteen tokens of
biblical English. Only genesis was walked this way.

## What the next cycle should do differently

Walk the stages of the server and the tail as well. Only pods 1 to 92 were covered here,
which is stages 121 to 11160 of 11,280; the first 120 and the last 120 were not looked
at and the tail is where the answer is produced.
