# Repository organization

## Reader paths

| Need | Maintained entry point |
|---|---|
| Understand Clover | [Framework](framework/04-framework.md), [philosophy](framework/02-philosophy.md), [principles](framework/03-principles.md) |
| Apply it | [Quickstart](../QUICKSTART.md), [guidance index](README.md), [templates](templates/) |
| Operate an agent | [AGENTS.md](../AGENTS.md), the self-contained operating specification |
| Verify HG4 & all 6 AI Myths | [AI Mythbuster](../site/hg-format/index.html), [REPRODUCE_AGENTS.md](../site/hg-format/REPRODUCE_AGENTS.md), [public ledgers](../benchmarks/public-runs-20261126/README.md) |
| Inspect v6.0 release & next milestone | [Clover v6.0 release](reference/release-v6.md), [5-Senses + 1-Brain roadmap](runtime-context/one-brain-five-senses-research-context.md) |
| Inspect real outcomes | [Case studies](case-studies/README.md) |
| Read an illustrative scenario | [Examples](examples/) |
| Inspect executable controls | [Runtime enforcement reference](runtime-enforcement/README.md) |
| Run the K3 engine | [Set up, tune and run](../clover-ai-system/README.md) |
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
	runtime-context/        85 engineering context documents across 6 domain folders
	reference/              glossary, clover-origin, release-v4, release-v6, and repository organization
research/
	ai-fire/                               dependency and concentration argument
	ai-future/                             explicit hypothesis
	benchmarks/                            v6.0 public-review A/B, Lean 4, Myth 6 steering, and reality probe runners
	collective-learning/                   privacy-preserving collective problem-solving learning (CPSL) experiments
	consciousness-and-subjective-experience/ 17 consciousness & interoception paper evaluations
	curiosity-and-developmental-learning/  13 intrinsic motivation & curiosity paper evaluations
	foundational-papers-1943-to-2017/      11 classic paper replications and mathematical verifiers
	hardware-data-movement/                red-blue pebbling, eyeriss v1/v2, flashattention, and specmd studies
	object-oriented-reduction/             398 -> 88 -> 24 -> 12 -> 5 complete-English OOP reduction
	observe/                               reciprocal observation experiment
	k3/                                    model derivation, experiments, measurements, and upstream records
benchmarks/
	public-runs-20261126/   v6.0 live cluster ledgers (tsock mesh, HumanEval-164, Lean 4 MiniF2F, Myth 6, reality probes)
	riemann-lean4/          sealed 4-tier Lean 4 + SciCode 288-step + RTX 4090 HG4 benchmark ledgers
	swebench-pro-v2/        multi-domain HARD-51 software engineering verification suite (Problems 1..5)
	verified-runs/          70-task standardized AI benchmark runs (baseline, 8x RTX 3090 GPU, Ryzen 7950X3D CPU)
clover-ai-system/
	src/                    HG4/K31H C11 & CUDA engines, HG4 Fusion Gateway, MCP 2.0 mesh, and training scripts
	studio/                 self-hosted Clover AI Studio v6.0 server, 4-Part Coordinator, and WIP desktop packaging
	mobile/                 shared semantics/memory core and WIP Android Studio project
	vscode-extension/       standalone VS Code language model provider
site/
	index.html, hg-format/, ai-system/, ...  public website (cloverframework.com)
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

The runnable v6.0 AI system lives in `clover-ai-system/`. The first version `clover-k3/`, `clover-kimi-k3/` (renamed into `clover-ai-system/`), and the distributed experiment `clover-intelegence/` were retired from the root working tree; early K3 derivation artifacts remain in git history at commit `013cbc8`, and links to them point there.
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