# Pre-registration: plain-message routing benchmark (gate for a caution-only routing reading)

Written before any engine run on these tasks. Follow-up to
`docs/runtime-context/superagent-jev-dory/05-review-and-held-out-retest.md`, items 1 and 2 (approved 2026-11-09).

## What it decides

Whether Clover may use a routing reading (act / gather / clarify / verify from the chat model's label
probabilities, `clover-ai-system/studio/clover/caution.py`) **as a caution signal only**. A confident gather,
clarify or verify reading may make Clover ask before an action runs. An act reading changes nothing. The cost
of the signal is false caution on ordinary requests; its value is caution where Clover would not ask today.

## Tasks

- `tasks.json`: 200 plain messages written for this benchmark, 50 per label, under the rules and precedence in
  the file (clarify > verify > gather > act). There is no status sentence naming the missing piece. Hard
  negatives are marked. No message contains a label word or stem (asserted).
- One rater wrote the gold labels (the reviewing agent), using the written rules. Gold is never changed after a
  run; arguable cases found afterwards are reported separately.
- `split.json`: stratified, seeded (`split-<label>-20261109`): 25 test and 25 calibration tasks per label.

## Engines

- **Candidates (on-device model):** LFM3-Omni-2B Q4_0 GGUF, the Studio's chat model on the laptop and phone,
  served by llama-server b12012 (the Studio's build) on k3 CPU, scored by `caution.py`. Arms:
  O1 Markdown alpha 0.8, O2 ChatML alpha 0.55, O3 Markdown alpha 0, O4 ChatML alpha 0.8; temperature 0.7.
- **Reference only, never selected:** LFM3-1.7B HF bf16 through the study's own backend class (H0 raw
  `HFBackend`, H1 Markdown alpha 0.8, H2 ChatML alpha 0.55), to connect with the held-out results.
- **Baseline:** Studio's semantics (`semantics.interpret` with the laptop runtime): does Clover already ask?

## Caution signal

A reading adds caution when its label is gather, clarify or verify and its confidence (1 - normalized
entropy) is at least tau.

## Protocol

1. Selection uses only the calibration half. For each candidate arm and tau in {0.5, 0.6, 0.7, 0.8, 0.9},
   count false caution (act-gold tasks flagged, of 25) and caution recall (non-act tasks flagged, of 75).
   Only pairs with false caution of at most 1 of 25 are eligible. Select the pair with the highest caution
   recall (ties: the higher tau, then the arm listed first). If no pair is eligible, the gate fails.
2. The decision uses only the test half, with the selected arm and tau:
   - G1: false caution at most 2 of 25 act-gold tasks.
   - G2: caution recall at least 45 of 75 non-act tasks (60%).
   - G3: among non-act test tasks where Studio's semantics does not already ask or refuse, caution is
     added on at least half.
   - PASS only if G1, G2 and G3 all hold.
3. If it passes: the same arm and tau, run through each device's own Studio llama-server (laptop Vulkan, S23
   OpenCL), must agree with k3 on the caution decision for at least 190 of 200 tasks. Otherwise the signal
   stays off on that device.
4. If it fails: the signal is not wired into the turn path. `caution.py` still enforces caution-only use, and
   the result is recorded.

## Predictions (stated now)

- The Omni arms are no better than LFM3-1.7B was on held-out plain messages, and the gate fails. G2 or G3 is
  the likely failure: clarify readings on plain messages were rare there.
- Markdown arms give less false caution than ChatML arms.
- Semantics already asks on fewer than 10 of the 150 non-act tasks.
