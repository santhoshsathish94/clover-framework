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

`kimi-k3-local-evidence.json` records what the storage-streamed inference
experiment established on local hardware, and what it is still waiting on.

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
