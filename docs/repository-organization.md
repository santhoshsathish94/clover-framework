# Repository organization

## Reader paths

| Need | Maintained entry point |
|---|---|
| Understand Clover | [Framework](framework/04-framework.md), [philosophy](framework/02-philosophy.md), [principles](framework/03-principles.md) |
| Apply it | [Quickstart](../QUICKSTART.md), [guidance index](README.md), [templates](templates/) |
| Operate an agent | [AGENTS.md](../AGENTS.md), the self-contained operating specification |
| Inspect real outcomes | [Case studies](case-studies/README.md) |
| Read an illustrative scenario | [Examples](examples/) |
| Inspect executable controls | [Runtime enforcement reference](runtime-enforcement/README.md) |
| Run the K3 engine | [Set up, tune and run](../clover-kimi-k3/README.md) |
| Inspect K3 measurements | [Status log](runtime-context/engine-and-architecture/clover-kimi-k3-cpu-context.md) |
| Follow investigations | [Research](../research/README.md) |
| Maintain public presentation | [Website](../site/README.md) |

## Document ownership

The framework chapter defines the cycle; philosophy explains its rationale; principles
state constraints; governance describes controls. Summarize and link instead of copying
definitions between chapters. Keep the agent specification self-contained, but reconcile
meaning across it and the explanatory documents when the underlying direction changes.

Label proposed designs, maintained references, historical experiments and speculative
arguments. Put hardware, configuration, evidence scope and unknowns beside results.
Do not imply that a proposal is implemented or that a one-machine observation is universal.

The model equation and implementation equation deliberately remain independent references.
Research journals preserve the path, including wrong turns; current summaries explain what
still holds. Case studies describe observed work; examples may be illustrative.

## Migration map

Applied locally on 2026-09-29. The initial compatibility pages preserved the old clutter;
the author's follow-up requested a physical reorganization. Those pages are now removed.
Internal links and same-repository website URLs point to maintained locations. Old external
GitHub file URLs are not redirected. Chapter numbers remain, including the intentional missing `06`.

```text
docs/
	framework/              conceptual core (01-problem .. 07-outcome)
	guides/                 adoption and practical work (09-adoption, 10-roadmap, field-practices)
	governance/evidence/    controls, responsibility and sourced dossier
	case-studies/           observed production migrations and incident remediations
	examples/               illustrative end-to-end walkthroughs
	experiments/            meaning formation, grounded execution, and developmental growth pilot
	templates/              reusable orchestration brief template
	runtime-enforcement/    executable container write-guard verification policy
	runtime-context/        engine architecture, 11 foundational paper studies, and hardware data-movement ledgers
	reference/              glossary, clover-origin, release-v4, and repository organization
research/
	ai-fire/                               dependency and concentration argument
	ai-future/                             explicit hypothesis
	foundational-papers-1943-to-2017/      11 classic paper replications and mathematical verifiers
	hardware-data-movement/                red-blue pebbling, eyeriss v1/v2, flashattention, and specmd studies
	object-oriented-reduction/             398 -> 88 -> 24 -> 12 -> 5 complete-English OOP reduction
	observe/                               reciprocal observation experiment
	k3/                                    model derivation, experiments, measurements, and upstream records
benchmarks/
	swebench-pro-v2/        multi-domain HARD-51 software engineering verification suite (Problems 1..5)
	verified-runs/          70-task standardized AI benchmark runs (baseline, 8x RTX 3090 GPU, Ryzen 7950X3D CPU)
clover-kimi-k3/
	src/                    code, rust host-path crates, scripts, tools, and prompts; datasets and work are host only
	vscode-extension/       standalone VS Code language model provider
site/
	index.html, ai/, ...    public website (cloverframework.com)
assets/
	branding/               project identity and previews
	evidence/               canonical evidence diagrams
	k3/                     scaling artwork
	ai-future/              hypothesis illustrations
```

| Earlier location | Maintained location |
|---|---|
| `why-clover-is-important.md` | [AI-Fire research](../research/ai-fire/README.md) |
| `hypothesis/ai-future.md` | [AI future hypothesis](../research/ai-future/README.md) |
| `k3-analysis/clover-ai.md` | [Meaning development experiment](experiments/meaning-development/experiment-design.md) |
| Other `k3-analysis/**/*.md` | Purpose-based folders under [research/k3](../research/k3/README.md) |
| `clover-k3/clover-k3-equation.md` | [Implementation equation](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/reference/implementation-equation.md) |
| `clover-k3/clover-k3-proof.md` | [Results](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/evidence/results.md) |
| `clover-k3/clover-k3-scaling.md` | [Scaling entry point](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/scaling/README.md), with the image at the end and no forwarding-heading list |

The legacy benchmark script and JSON evidence live with the [upstream investigation](../research/k3/upstream/README.md).
The script was moved without content changes and was not executed. The historical
`clover-k3/proof-campaign.sh` is in git history; its scope is described in
[evidence methods](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/evidence/evidence-methods.md).

The runnable K3 engine is `clover-kimi-k3/`. The first version `clover-k3/` and the distributed experiment `clover-intelegence/` were removed from the working tree on 2026-10-19; both are in git history at commit 013cbc8, and links to them point there.
Case studies, examples and templates already have small, coherent scopes; they do not
need one extra folder per file. Website route directories and executable reference
projects retain their working structure.

## Validation

From the repository root:

```sh
python -m unittest discover -s scripts/tests
python scripts/check-links.py
python scripts/check-site.py
```

The checks cover local documentation and site integrity, not external source truth,
benchmark reproduction, deployment or model quality. Hardware gates remain separate.
`assets/evidence/` owns shared evidence diagrams; `site/assets/` contains identical deployment
copies because the Pages artifact includes only `site/`.