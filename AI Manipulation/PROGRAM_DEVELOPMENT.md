# Program Development Baseline

This expands the model-free experiment from parameter optimization to **program discovery**.

The system can:

1. generate candidate programs
2. execute them in a restricted evaluator
3. measure their behavior against an independent benchmark
4. retain successful programs
5. mutate successful programs
6. repeat across persistent cycles
7. archive validated discoveries

There is still no language model.

## Developmental boundary

The previous experiment optimized a number.

This experiment evolves an executable procedure.

That changes the question from:

> Can computation find a better value?

to:

> Can computation discover a better method?

## Architecture

```
Persistent State
      ↓
Candidate Program Population
      ↓
Sandboxed Execution
      ↓
Independent Benchmark
      ↓
Evidence / Score
      ↓
Selection
      ↓
Mutation / New Candidates
      ↓
Validated Archive
      ↺
```

The evaluator knows the benchmark but does not accept a candidate's own claim that it succeeded.

## Run

```bash
python program_development.py
```

State is retained in:

```
program_development_state.json
```

## What this does NOT prove

It does not prove general intelligence, agency, consciousness, or open-ended learning.

The benchmark and mutation language are designed by humans.

## Next boundary

Remove more human design:

- evolve the representation of programs
- generate tests from prior failures
- discover reusable subroutines
- maintain competing strategies
- allocate compute toward uncertain areas
- modify the search algorithm itself
- use rollback when a modification reduces performance
- introduce an adversarial independent evaluator

At that point we can test whether the developmental process is merely optimizing a fixed human-designed space or is beginning to expand its own problem-solving machinery.
