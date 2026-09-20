# Work in progress

Exploratory material that is not part of the framework doctrine.

Documents here describe work that is being designed or trialed. They are not
guidance, they have not been validated in practice, and they may be wrong,
abandoned, or rewritten. Nothing in this folder should be cited as a Clover
practice.

When something here is built and evidenced, it moves into `docs/`,
`case-studies/`, or `examples/` with its evidence stated plainly. Until then it
stays here.

## Documents

| Document | Subject | State |
|---|---|---|
| [clover-ai.md](clover-ai.md) | Implementation direction for Clover AI — bounded tools, enforced permissions, deterministic verification | Direction, not implemented |
| [self-hosting-slms.md](self-hosting-slms.md) | Running small language models on infrastructure an organization controls, wired into VS Code | Designed, not built |
| [self-observation-loop.md](self-observation-loop.md) | Whether persistent external state produces traceable adaptation across runs | Experimental, running |

`self-observation-state.json` holds the run state for the self-observation loop.

## [Kimi K3 storage-streamed inference](fareed-khan-kimi-k3-in-c-explanation.md)

Four files covering one line of work: whether a 2.78-trillion-parameter model running on
one CPU changes what AI infrastructure has to be. **The engine is Fareed Khan's
[`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), Apache-2.0. Nothing here
reimplements it and no model weights are redistributed.**

| File | What it is |
|---|---|
| [fareed-khan-kimi-k3-in-c-explanation.md](fareed-khan-kimi-k3-in-c-explanation.md) | How the engine works, what it measured, and what it implies for the Clover infrastructure direction |
| [kimi-k3-local-evidence.json](kimi-k3-local-evidence.json) | What the experiment established on local hardware, and what it is still waiting on |
| [kimi-k3-bench-run.sh](kimi-k3-bench-run.sh) | The measurement campaign for rented hardware. Gated, shellcheck-clean, and **never yet run against weights** |
| [CONTEXT-kimi-k3-benchmark.md](CONTEXT-kimi-k3-benchmark.md) | Handoff record: what is settled, what was ruled out and why, corrections made, abort criteria, and what remains unknown |

Reproduced locally, with no checkpoint and no GPU: the weightless gate ladder, the released
configuration, the byte-exact tokenizer round-trip, the published 100,096-request
expert-cache table, and a kernel compute baseline. The checkpoint's 96 shards were
confirmed to total 1,560,936,091,448 bytes without downloading them.

Not reproduced: any full-model measurement. That needs about 1.7 TB of storage and more
memory than the machine here has, which is what the rented box is for.

## [AI Manipulation](ai-manipulation/)

A model-free developmental engine, a persistent supervisor with replaceable
workers, and the documents describing both. It moved here from the repository
root because it is an experiment rather than doctrine, and sitting at the root
implied otherwise.

### What has been demonstrated

Each of these was observed in a bounded toy domain and nowhere else.

| | Evidence |
|---|---|
| Persistent state survives process termination | The engines reload and continue from committed JSON |
| A capability can be acquired and kept | Five tasks on the ladder in `task_ladder.py`, each promoted only after passing anchor tests the engine cannot write |
| Memory reduces the cost of later work | `control_comparison.py` solved the same ladder with and without memory: 2,397 evaluations against 1,869, a 22% saving, though transfer fired on only one of five tasks |
| A promotion that breaks earlier work is reverted | Verified by corrupting an anchor deliberately and watching the rollback fire |
| Self-certified claims are refused | `supervisor.py` routes every claim through the evaluator; only claims carrying independent evidence reach `validated_knowledge` |
| Continuity survives replacing the worker | `model_swap_test.py` against two local language models from different families, `llama3.2:1b` then `qwen2.5:0.5b`; recorded in `model_swap_evidence.json` |

### What has not been demonstrated

| | Why not |
|---|---|
| That any of this holds outside a toy domain | The task ladder, the representations and the evaluator are all supplied by the experimenter |
| That the process can leave the space designed for it | It searches parameters. It cannot invent a representation, and says so by pausing when a task is inexpressible |
| That a language model continues the trajectory *usefully* | The handover carried, but neither model produced anything that passed the evaluator. `qwen2.5:0.5b` declines every cycle, so the swap only succeeds with the abler model first. The hosted worker has never completed a cycle: it reached the API and stopped at `429 insufficient_quota` |
| Open-ended development, general capability, subjective experience | Untested, and the first two are not reachable from here |

### Running it

```bash
python control_comparison.py           # control vs developmental, same ladder and budget
python growth_report.py                # what the recorded state actually shows
python model_swap_test.py              # does continuity survive replacing the worker
python unified_development_engine.py   # one cycle
```

The anchor tests live in `task_ladder.py`. The engines read them and never write them.
