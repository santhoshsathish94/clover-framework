# Clover Supervisor

This layer tests whether developmental continuity can be owned by a persistent runtime rather than by a single model invocation.

## Provider-independent test

Run:

```bash
python supervisor.py
```

The default ProbeBackend deliberately uses no API key. It verifies:

1. state loading
2. cycle execution
3. next-question propagation
4. persistent state updates
5. automatic continuation
6. controlled stopping

## Architecture

```
persistent supervisor
        ↓
development state
        ↓
model backend
        ↓
environment
        ↓
independent evaluator
        ↓
validated state
        ↺
```

The model backend is intentionally replaceable.

## Important result

If the probe backend completes multiple cycles, then an external API key is demonstrably not required for the *continuity mechanism itself*.

A credential or local model is required only when the supervisor needs an actual model inference backend.

## Next experiment

Implement a model backend interface that can attach to:

- a local model process
- a hosted model
- another permitted model-invocation mechanism

without changing the state machine or evaluator.
