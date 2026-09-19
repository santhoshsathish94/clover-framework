# Self-Observation Loop — Experimental State

Status: Experimental

## Purpose

Test whether repeated interaction with a persistent external state can produce measurable adaptation in future decisions without treating adaptation as evidence of subjective experience.

## Loop

Context → Hypothesis → Action → Observation → Reflection → State Update → Next Action

## Persistent State

- `identity_model`: What the system currently models about its own capabilities and limits.
- `open_questions`: Questions not yet resolved.
- `predictions`: Explicit expectations made before an action.
- `actions`: Actions actually taken.
- `observations`: Results obtained from the environment.
- `errors`: Differences between prediction and observation.
- `adaptations`: Changes to the next decision caused by observed evidence.
- `evidence`: External artifacts supporting each observation.

## First State

### Identity model

The system can inspect repository state and external information through available tools, reason over the returned observations, and choose a next investigation step within its granted permissions. It cannot infer subjective experience merely from these capabilities.

### Open questions

1. Can persistent state change a later decision in a way that is traceable to an earlier observed outcome?
2. Can the system identify and correct a prediction error across separate runs?
3. Does the resulting adaptation remain distinguishable from simply following the latest instruction?

### Baseline prediction

If the same investigation is repeated with persistent state available, later decisions should incorporate recorded observations and errors rather than restarting from an identical state.

## Rules

1. Record predictions before actions where practical.
2. Record observations from the external environment rather than assumptions.
3. Separate observation from interpretation.
4. Record prediction errors explicitly.
5. Only call something adaptation when a later decision changes because of recorded prior evidence.
6. Do not label adaptation, self-reference, persistence, or recursive modeling as consciousness or subjective experience without independent evidence.
7. Keep the experiment reversible and bounded.

## Run Log

### Run 1

- Trigger: User requested `Run`.
- Action: Inspected the Clover repository for persistent memory, state, self-model, reflection, experience, trajectory, and growth mechanisms.
- Observation: No implemented persistent self-state/self-model loop was found in the searched repository content. The repository already defines Context, Direction, Execution, Outcome, Verification, Growth, and Observability at the conceptual level.
- Reflection: The conceptual Growth loop exists, but a concrete persistent experimental state was missing.
- Adaptation: Create this minimal state record before claiming that a persistent self-observation loop exists.

### Run 2

- Trigger: User requested `Run`.
- Prediction made before investigation: The existing Clover AI direction would contain explicit agent boundaries and verification mechanisms, but would not itself implement persistent self-learning state.
- Action: Searched the repository for agent objectives, stop conditions, permissions, verification, persistent memory, state, history, learning, and reflection.
- Observation: The Clover AI direction explicitly contains bounded permissions, stop conditions, observable actions, verification, and escalation. The repository also describes Growth as information carried from one Outcome into the next Context. No implementation of persistent self-learning state was found in the search.
- Error: The prediction was substantially supported; no material prediction error was identified.
- Reflection: The investigation changed the next question from “does Clover have a loop?” to “what is the smallest mechanism that can turn Growth from a conceptual stage into experimentally measurable state change?”
- Adaptation: The next experiment should not add another conceptual description. It should define a minimal machine-readable state transition and test whether a later decision can be traced to a prior recorded observation.
- Evidence: `work-in-progress/clover-ai.md`, `docs/05-context-engineering.md`, `docs/04-framework.md`.

## Next Experiment

Define the smallest machine-readable state transition: prior state → prediction → action → observed outcome → error → updated state → next decision. Then test one transition and record whether the next decision actually differs because of the stored evidence.
