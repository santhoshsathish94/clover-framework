# Repository organization

## Reader paths

| Need | Maintained entry point |
|---|---|
| Understand Clover | [Framework](framework/04-framework.md), [philosophy](framework/02-philosophy.md), [principles](framework/03-principles.md) |
| Apply it | [Quickstart](../QUICKSTART.md), [guidance index](README.md), [templates](../templates/) |
| Operate an agent | [AGENTS.md](../AGENTS.md), the self-contained operating specification |
| Inspect real outcomes | [Case studies](../case-studies/README.md) |
| Read an illustrative scenario | [Examples](../examples/) |
| Inspect executable controls | [Runtime enforcement reference](../reference/runtime-enforcement/README.md) |
| Run Clover-K3 | [Build, configuration and gate](../clover-k3/README.md) |
| Inspect K3 design and evidence | [Implementation documents](../clover-k3/docs/README.md) |
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
	framework/              conceptual core
	guides/                 adoption and practical work
	governance/evidence/    controls, responsibility and sourced dossier
	reference/              glossary and name/mark
research/
	ai-fire/                dependency and concentration argument
	ai-future/              explicit hypothesis
	clover-ai/              broader implementation direction
	k3/
		model/                specification and derivations
		experiments/          computation and data investigations
		architecture/         placement proposals and measurement journals
		context/              historical handoffs
		upstream/             independent engine, evidence and campaign script
		measurements/
			flow/               instrumented flow observations
			prompts/            individual prompt cases
clover-k3/docs/
	reference/              implementation equation
	evidence/               results and methods
	scaling/                overview, architecture, resources, validation
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
| `k3-analysis/clover-ai.md` | [Clover AI direction](../research/clover-ai/README.md) |
| Other `k3-analysis/**/*.md` | Purpose-based folders under [research/k3](../research/k3/README.md) |
| `clover-k3/clover-k3-equation.md` | [Implementation equation](../clover-k3/docs/reference/implementation-equation.md) |
| `clover-k3/clover-k3-proof.md` | [Results](../clover-k3/docs/evidence/results.md) |
| `clover-k3/clover-k3-scaling.md` | [Scaling entry point](../clover-k3/docs/scaling/README.md), with the image at the end and no forwarding-heading list |

The legacy benchmark script and JSON evidence live with the [upstream investigation](../research/k3/upstream/README.md).
The script was moved without content changes and was not executed. The historical
`clover-k3/proof-campaign.sh` remains at its original path; its scope is described in
[evidence methods](../clover-k3/docs/evidence/evidence-methods.md).

Runnable Clover-K3 entry points remain beside their configuration and single-file source.
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