# Clover Development Loop v0.1

This folder contains the first experimental controller for a persistent AI developmental loop.

## Purpose

The controller repeatedly invokes an AI model, gives it persistent developmental state, lets it select the next unresolved investigation, records evidence and learning, and automatically continues until the model pauses or stops.

The model is treated as replaceable. Developmental state lives outside the model.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai
export OPENAI_API_KEY="YOUR_API_KEY"
python clover_loop.py
```

Optional:

```bash
export CLOVER_MODEL="gpt-6-astra"
export CLOVER_MAX_CYCLES="20"
```

## Experimental principle

No human supplies the next question. Each cycle must identify the highest-value unresolved uncertainty and attempt to investigate it.

The investigator must distinguish:

- observation
- inference
- hypothesis
- experiment
- result
- evidence
- learning
- uncertainty

A claimed learning change should not be accepted without evidence.

## Important limitation

This is an experiment harness, not proof of autonomous learning or subjective experience.

For a stronger experiment, provide a bounded environment and real tools, and use an independent evaluator so the investigator cannot be the sole judge of its own claimed improvement.

## Documents

Nothing else in the repository links these, so this is their only index.

| Document | Subject |
|---|---|
| [what-is-still-missing.md](what-is-still-missing.md) | The criteria table: what is demonstrated, what is untested, and the limits of the toy domain |
| [development-engine.md](development-engine.md) | The unified loop, the four enforced invariants, and how growth is measured |
| [capability-growth.md](capability-growth.md) | What it means to acquire a capability rather than a better answer |
| [meta-development.md](meta-development.md) | Changing the search strategy itself, and evaluating the change |
| [model-free-baseline.md](model-free-baseline.md) | Whether a loop can improve without any model inference |
| [program-development.md](program-development.md) | Searching for executable procedures rather than numbers |
| [open-ended-development.md](open-ended-development.md) | Whether improvement continues once the obvious gains are exhausted |
| [self-model-revision.md](self-model-revision.md) | Testing the system's assumptions about itself against evidence |
| [growing-the-investigator.md](growing-the-investigator.md) | Applying the loop to the process running it |
| [runtime-boundary-investigation.md](runtime-boundary-investigation.md) | What the runtime actually permits, probed rather than assumed |
| [ai-monoculture-systemic-risk.md](ai-monoculture-systemic-risk.md) | Shared dependencies across the AI stack as correlated failure |
| [chatgpt-codex-no-api-credit-path.md](chatgpt-codex-no-api-credit-path.md) | Reaching a model without API billing, and why not from a public repository |
| [supervisor/README.md](supervisor/README.md) | The persistent supervisor and its replaceable backends |
| [supervisor/local-model.md](supervisor/local-model.md) | Local endpoints: the Responses and chat-completions shapes are not interchangeable |
| [supervisor/openai-backend.md](supervisor/openai-backend.md) | The hosted backend, which has never completed a cycle |

## Evaluation

The comparison this folder was built around is now implemented rather than suggested:

```bash
python control_comparison.py   # control vs developmental, same ladder and budget
python growth_report.py     # what the recorded state actually shows
python model_swap_test.py   # does continuity survive replacing the worker
```

**Control:** each task solved cold, with no memory of what previously worked.

**Developmental:** the same tasks, carrying the representation that last succeeded.

Both arms get the same shuffled search order and the same budget, so memory is
the only difference between them. Identical totals would mean persistence
contributed nothing, and that outcome is reachable — which is the point.

The anchor tests live in `task_ladder.py`. The engines read them and never write them.
