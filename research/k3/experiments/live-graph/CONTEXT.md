# Composed Live-Vector Graph

## Direction

The human authorized reducing the composed function after treating prepared
trunk and expert operators as fixed inputs, without privileging layer scheduling
boundaries. The ongoing prepared-trunk campaign must remain unchanged.

## Latest Direction: Wait for Trunk Completion

On 2026-10-01 the human asked to record the next investigation and wait for the
prepared-trunk campaign to complete. Do not start another composition experiment
while waiting. Leave the campaign running unchanged; its completion has not been
rechecked by this note.

After confirming all 92 prepared trunks have passed their durable model gates,
investigate composing the unique transformations along the critical path into
one equivalent, potentially cheaper input-dependent transformation. Expand and
substitute intermediate expressions rather than treating current IR nodes or
layer boundaries as indivisible.

The current 3416-node chain stayed unchanged because CSE removed duplicate work,
not the unique sequential transformations. It is a property of this IR, not a
proven minimum. Distinguish eliminating vector materialization from eliminating
the arithmetic that produces those vectors. Naming the composition F_Theta does
not itself reduce either. The result need not be one matrix; a specialized
nonlinear computation is a valid target.

Preserve intermediate rounding effects or prove their removal valid for the
specific transformation. For example, real-linear composition B(Ax)=(BA)x does
not automatically establish bitwise equality of R(B R(Ax)) and R((BA)x).
Retain all original verification outputs and route effects. The next cycle starts
with the completed prepared data and critical-path expressions, not a claim of
already-reduced depth or a new full-model executor.

## Local Hypothesis and First Check

Snapshot scoring against a fixed future direction depends on the captured
snapshot, not on the intervening transitions or their live residuals. A global
single-assignment graph can hoist those scores and merge identical rounded
subexpressions across sites, while keeping arithmetic operand order and parameter
identity. Snapshot capture aliases an immutable version of the incoming residual;
the prior transition label does not define a distinct numerical value.

Added an operation-level graph for all93transitions, five positions, KDA/MLA,
dense/shared/routed FFN and tail. Prepared parameters are symbolic external inputs;
this does not claim every dataset is already converted. Numerical primitives
retain their exact semantics, some internally opaque. A first unit check compares
all output expression fingerprints before/after CSE, validates dependencies and
checks that T92's S0 scores are available from S0 without T91's residual.

## Observed Outcome

Eight graph tests pass, including exact before/after fingerprints for all 32213
named outputs, ordered parameter-aware CSE, source aliases, causal coverage,
global topological order and audit-versus-logit liveness. Actual France/Japan
route plans and model source hashes bind the IR. All 32210 audit keys/shapes
are bound per prompt; independent trace counts and eight capture boundaries match.

67119 nodes before CSE, 59028 after. Of 8091 shared nodes, 8021 correspond to
previous inverse/scoring caches; only 70 are newly identified cross-capture reuse:
35 inverses and 35 vectors, or250880rounded coordinates per prompt. The source is
the same incoming residual before pre-attention and after snapshot capture.

Future snapshot scores have only inverse/scoring-vector/dot dependencies on their
source. S0's T92 scores precede T0's residual in the ready schedule. The extracted
score-bank C test passed 16332 scores and117067776product/accumulator steps over
both actual prompts and normal/reverse request orders. All recomputed inverses
and scoring vectors match earlier model evidence. Populated score storage16332B
per prompt; dense prototype29920B. Existing products are not reused by this new
test bank; it tests scheduling/identity, not stacked multiplication savings.

Output-only liveness marks64T92expert-position evaluations (positions0-3) unneeded
by final logits. They remain in the full audit graph and running model. Initial
numeric-only traversal accidentally lost four route-check effects; the unchanged
test caught it. Fixed IR by retaining route effects before experts and final
logits. No audit assertion or model fixture changed. A later trace parser read
snapshot count instead of capture flag; checked emitter and corrected field6,
not the expected capture layers, then reran the complete focused checks.

The longest dependency depth stays3416IRoperations before/after. Opaque primitives
mean this is not scalar/hardware depth. Less duplication does not establish a
shorter critical path, and layer independence does not establish global minimality.

## Boundaries and Next Gate

This is a source-authored IR plus a numerically executed scoring subgraph, NOT
a full-model executor or proof of globally minimal depth. Recorded vector hashes
are identity bindings, not independently computed full-model outputs. Symbolic
prepared bindings do not claim all trunk conversions have finished. The graph
models fresh five-token next-logit inference; reusable generation state needs
its own explicit output contract.

No SSH, campaign signals, server model executions or campaign-code changes in
this cycle. Prepared-trunk progress was not checked and must not be inferred.
Next: separate global executor with stable source lifetimes and score-ready work,
followed by full-model/audit comparison. Output-only pruning remains a separate
contract/test, not permission to remove the existing validation outputs.