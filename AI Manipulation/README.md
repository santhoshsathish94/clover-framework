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

## Suggested evaluation

Compare:

**Control:** repeated model calls without persistent developmental state.

**Developmental:** repeated calls with persistent state, self-selected investigations, experiments, and verification.

Measure capability, retention, transfer, novel discoveries, experiment quality, learning efficiency, and whether the learning strategy itself improves.
