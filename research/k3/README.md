# K3 analysis

**Role: investigation index.** Start with the maintained references below, then follow
the journals when you need the derivation or original experiment. Historical status
statements apply to their campaign, not automatically to the current implementation.

## Current reading path

1. [Run the single-machine reference](../../clover-k3/README.md).
2. [Read the model specification](model/k3-model-equation.md) and the separate
	[implementation equation](../../clover-k3/docs/reference/implementation-equation.md).
3. [Inspect evidence methods and limits](../../clover-k3/docs/evidence/evidence-methods.md).
4. [Read the proposed scaling design](../../clover-k3/docs/scaling/README.md).

## Complete investigation map

Folder guides: [model](model/README.md), [experiments](experiments/README.md),
[architecture](architecture/README.md), [context](context/README.md),
[upstream](upstream/README.md), [measurements](measurements/README.md),
[flow](measurements/flow/README.md), [prompt cases](measurements/prompts/README.md).

| Area | Documents | Role |
|---|---|---|
| Model understanding | [Overview](model/understanding-k3.md), [stage walk](model/k3-stages.md), [initial equations](model/k3-maths-equation.md), [model specification](model/k3-model-equation.md) | Derivation history and maintained specification; not interchangeable |
| Computation and data | [Reduction](experiments/k3-equation-reduction.md), [optimization journal](experiments/k3-equation-solution.md), [redundancy](experiments/k3-redundancy.md), [data problem](experiments/k3-data-problem.md) | Experiments and corrections, with campaign-specific limits |
| Exact representations | [Vector equations](experiments/vector-equations.md), [vector storage](experiments/vector-representation.md), [remaining candidates](experiments/remaining-equation-candidates.md) | Scoped consumer passes, failed approaches and untested research directions |
| Cumulative integration | [Reverse-order campaign](experiments/reverse-integration.md), [snapshot guide](experiments/reverse-integration/README.md), [speed and RAM](experiments/integrated-performance.md) | Separately gated additions and measured as-is implementation costs |
| Actual source data | [Source datasets](experiments/source-datasets.md), [data roles and formats](experiments/model-data-inventory.md), [input value census](experiments/input-table-values/README.md) | Server tensor ownership, physical data boundaries and complete input-table scan |
| Seed input | [Dataset](experiments/seed/README.md), [integration evidence](experiments/seed-integrated.md), [tested model guide](experiments/seed-integrated/README.md) | Lossless input-table storage and guarded model consumption |
| Fruit output | [Dataset](experiments/fruit/README.md), [seed and fruit model guide](experiments/fruit-integrated/README.md) | Lossless output-head storage, complete reconstruction and combined model tests |
| Placement and scaling | [Scaling journal](architecture/clover-scaling-architecture.md), [earlier topology](architecture/k3-client-server-architecture.md), [heterogeneous inference](architecture/heterogeneous-inference.md), [residency](architecture/residency-not-speed.md) | Measurements and design evolution; use the current scaling entry point for the chosen boundary |
| Upstream engine | [Explanation](upstream/fareed-khan-kimi-k3-in-c-explanation.md), [measurements](upstream/kimi-k3-measurements.md) | Separate engine and its hardware experiments |
| Kimi K3 itself | [Official repo and tech report](upstream/moonshot-kimi-k3-official.md), [vLLM execution](upstream/vllm-kimi-k3-execution.md) | The model as its authors and serving engines define it, distinct from the C reimplementation we measure |
| Context and handoffs | [Model](context/CONTEXT-clover-model.md), [reduction](context/CONTEXT-k3-equation-reduction.md), [observation](context/CONTEXT-k3-observation.md), [benchmark](context/CONTEXT-kimi-k3-benchmark.md) | Historical working memory; preserve corrections and dates |
| Preparation and composition handoffs | [Prepared trunks](experiments/prepared-trunks/CONTEXT.md), [composed live graph](experiments/live-graph/CONTEXT.md) | Recorded outcomes and deferred work; wait for trunk completion before further composition experiments |
| Flow measurements | [Ceiling](measurements/flow/c-ceiling-and-channel.md), [cost](measurements/flow/cost-notes.md), [blocks](measurements/flow/i-inside-the-blocks.md), [context-free map](measurements/flow/m-context-free-map.md), [position](measurements/flow/p-position-selection.md), [reverse replication](measurements/flow/r-reverse-replication.md), [length sweep](measurements/flow/s-length-sweep.md) | Observations, not general capability claims |
| Prompt cases | [Short factual](measurements/prompts/v1-factual-short.md), [multi-factual](measurements/prompts/v2-factual-multi.md), [repetitive](measurements/prompts/v3-repetitive.md), [code](measurements/prompts/v4-code.md), [French](measurements/prompts/v5-french.md), [long context](measurements/prompts/v6-long-context.md), [nonsense](measurements/prompts/v7-nonsense.md), [control](measurements/prompts/v8-control.md) | Individual experiment records |
| Flow synthesis | [France walkthrough](measurements/flow/w-walkthrough-france.md), [routing predictability](measurements/flow/x-routing-predictability.md) | Explanation and tested routing-cache behavior |

Broader [Clover AI direction](../clover-ai/README.md) is separate from this model investigation.

The investigation into running Kimi K3 — a 2.78-trillion-parameter model — on
ordinary hardware, and what it took to establish that the model's equation was
understood rather than merely reproduced.

This is exploratory material, not framework doctrine. Documents here describe
work that was measured, and they record what was ruled out and what was got
wrong as carefully as what worked. Nothing in this folder should be cited as a
Clover practice.

Built work can have a maintained implementation reference or case study while its
research record stays here. Evidence does not automatically make a model-specific
technique a framework principle.

The working implementation this analysis produced lives in
[`../clover-k3/`](../../clover-k3/).

## Documents

| Document | Subject | State |
|---|---|---|
| [clover-ai.md](../clover-ai/README.md) | Implementation direction for Clover AI — bounded tools, enforced permissions, deterministic verification | Direction, not implemented |
| [heterogeneous-inference.md](architecture/heterogeneous-inference.md) | Three experiments on running a very large model on machines you can actually get, and where they point next | Direction, measured on one CPU, nothing on a GPU |
| [clover-scaling-architecture.md](architecture/clover-scaling-architecture.md) | The measurement record, step by step, including every wrong turn | Measured, one box |
| [k3-client-server-architecture.md](architecture/k3-client-server-architecture.md) | The client-server shape: what a client holds, what a pod holds, what crosses between them | Sizes measured, topology not built |
| [k3-model-equation.md](model/k3-model-equation.md) | The model stated as one composed expression | Executed and verified |

## [Kimi K3 storage-streamed inference](upstream/fareed-khan-kimi-k3-in-c-explanation.md)

Six files covering one line of work: whether a 2.78-trillion-parameter model running on
one CPU changes what AI infrastructure has to be. **The engine is Fareed Khan's
[`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), Apache-2.0. Nothing here
reimplements it and no model weights are redistributed.**

| File | What it is |
|---|---|
| [fareed-khan-kimi-k3-in-c-explanation.md](upstream/fareed-khan-kimi-k3-in-c-explanation.md) | How the engine works, what it measured, and what it implies for the Clover infrastructure direction |
| [residency-not-speed.md](architecture/residency-not-speed.md) | Why one machine cannot run a 1.56 TB model and many small ones can — the constraint is residency, not speed |
| [kimi-k3-measurements.md](upstream/kimi-k3-measurements.md) | Every measured figure in one place, each with what it does not establish, re-derived from the raw logs |
| [kimi-k3-local-evidence.json](upstream/kimi-k3-local-evidence.json) | What the experiment established on local hardware, and what it is still waiting on |
| [kimi-k3-bench-run.sh](upstream/kimi-k3-bench-run.sh) | The measurement campaign for rented hardware. Gated and shellcheck-clean |
| [CONTEXT-kimi-k3-benchmark.md](context/CONTEXT-kimi-k3-benchmark.md) | Handoff record: what is settled, what was ruled out and why, corrections made, abort criteria, and what remains unknown |
| [heterogeneous-inference.md](architecture/heterogeneous-inference.md) | Where this line of work goes next, and the measurements behind that choice |

Reproduced with no checkpoint and no GPU: the weightless gate ladder, the released
configuration, the byte-exact tokenizer round-trip, the published 100,096-request
expert-cache table, and a kernel compute baseline. The checkpoint's 96 shards were
confirmed to total 1,560,936,091,448 bytes without downloading them.

**Since run on rented hardware, against the real model.** A 2.78-trillion-parameter model
generated text on one CPU machine at about 5.3 seconds per token. Shrinking the always-used
part to 8-bit brought that to 4.1 and nearly halved the memory needed; shrinking it again to
4-bit gave almost nothing more, because by then the processor rather than the memory was the
limit. The story in order is in
[heterogeneous-inference.md](architecture/heterogeneous-inference.md); the full figures, including the
predictions that turned out wrong, are in
[CONTEXT-kimi-k3-benchmark.md](context/CONTEXT-kimi-k3-benchmark.md).

For this upstream-engine campaign, still not established: anything measured on a GPU,
any quality claim beyond comparing output on a single prompt, and any reproduction of
someone else's published speed figures. The later Clover first-token campaign below
is separate evidence, not a retroactive expansion of this campaign's coverage.

**Contributed back upstream, two pull requests.**
[#67](https://github.com/FareedKhan-dev/kimi-k3-in-c/pull/67) — concurrent chunked expert
reads, a batched bf16 matmul that is bit-identical to the serial kernel, and fewer reads in
the KDA recurrence. Three runs per arm against that project's current `main`, every run
reported: decode +5.7%, prefill +9.9%, whole run +7.0%.
[#68](https://github.com/FareedKhan-dev/kimi-k3-in-c/pull/68) — the engine never chose a
thread count, so OpenMP took one thread per *logical* CPU, and no flag existed to change it.
Counting physical cores instead is worth **23.2% of decode time**, with 3.35x fewer
involuntary context switches showing why. That PR also fixes `--trunk-gb auto`, which could
not start on any machine large enough to reach the configuration it exists to produce.

The measurement is what validates this, not whether it is accepted. Both pull requests are
open and may never be merged, and that would say something about another project's roadmap
and hardware rather than about whether the changes work here. What does bound the evidence
is stated in the pull requests: bundle measurement for three changes in #67 so one may
contribute nothing, a degraded PCIe link on the test machine that may flatter the storage
change, and one machine, one prompt and one memory budget throughout.

Held back deliberately: the int8 and MXFP4 trunk work is out of scope upstream — that
project's `ROADMAP.md` lists a precision dial for the trunk as explicitly not planned, and
its author had already measured the same accuracy wall independently. Thread *binding* was
dropped from #68 for the same kind of reason: it measured a few percent here, but that is
one CPU topology, and a recommendation in someone else's documentation lands on every
machine their users own. Neither is a failure; a change can be right for our reality and
wrong for someone else's.

## What this became

The analysis above produced a working implementation: [`../clover-k3/`](../../clover-k3/),
a single C file written from the equation rather than from any engine, and kept
bit-exact against a preserved baseline through every optimisation.

| | |
|---|---|
| [`../clover-k3/clover-k3-equation.md`](../../clover-k3/docs/reference/implementation-equation.md) | the equation as that program actually evaluates it, and where it differs from [k3-model-equation.md](model/k3-model-equation.md) |
| [`../clover-k3/clover-k3-proof.md`](../../clover-k3/docs/evidence/results.md) | all 34 prompts, the text the model wrote for each, and the comparison: 34/34 identical answers, 3.40x |
| [Scaling design](../../clover-k3/docs/scaling/README.md) | Proposed distributed architecture, resource sizing, evidence limits, and validation plan |
| [clover-scaling-architecture.md](architecture/clover-scaling-architecture.md) | the step-by-step measurement record behind it |
