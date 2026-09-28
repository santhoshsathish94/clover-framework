# Model-Free Developmental Baseline

This experiment intentionally removes the language model.

The system has only:

- persistent state
- candidate generation
- mutation
- execution
- objective evaluation
- selection
- retained history

It asks a narrow question:

> Can a computational process improve its behavior across cycles without another model inference?

This is not intended to demonstrate general intelligence. The objective is deliberately simple and known to the evaluator.

## Why this experiment matters

A model-based loop can improve because the model contributes reasoning.

This baseline gives us a control condition.

If the model-free system improves on a measurable task, then computation + feedback + memory can produce developmental change without model inference.

If it does not generalize, that also gives evidence about what additional capability the model contributes.

## Run

```bash
python model_free_baseline.py
```

Then inspect `model_free_state.json`.

## Next experiment

Replace the hand-designed mutation strategy with increasingly general program synthesis and search while keeping the evaluator independent.

The long-term question is whether useful developmental structure can emerge from the substrate itself.
