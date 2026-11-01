# Core observation on k3 — working context

## Direction (human, 2026-11-01, in their words)
- "before any phone or laptop version we need to get the underlaying version solid because the device specific
  changes are different from the core layers, C1 to 3 S1 to 3 inputs and outputs all are common weather it's laptop or
  phone or anything else"
- "the sematics is the real core the meaning engine which makes everything works"
- Two semantics: "one the universal one that is known and that can grow and the next one is the version the user way
  of using it ... mapping their way of words to meaning is the seperate version, so 2 things can grow but the user
  sematics should not corrupt the core sematics". Recorded only; designed after the observation (human's choice).
- "setup the system in k3 machine observer each layers how it interacts with each other understand it don't judge it
  let's first document how things are working before deciding what to fix and what to improve"
- Choices made: framework runtime and laptop stack side by side; write only to a new isolated folder on k3, own port on
  127.0.0.1, low priority; push to GitHub then clone on k3 (the push also deploys the website; the human chose to
  include it). GPU server (8x3090) offered; not needed while both stacks run on CPU; ask if a full-precision C1
  comparison is needed.

## Boundaries on k3
- Write only under /opt/clover-k3/core-observe. Bind 127.0.0.1 on ports other than 8765. nice 19.
- Untouched: the live K3 chat server (127.0.0.1:8765 behind caddy 80/443); its checkout /opt/clover-k3/clover-framework
  (178da556 plus other sessions' uncommitted edits); /opt/clover-k3/s23-omni2b (another session's staging: model
  files used read-only, .app_secret and vault/ never read); /opt/clover-k3/models; /root/ncm-cpu/venv (used, no
  installs).
- Observing must not change behaviour: entry points are wrapped from outside the runtime code; no edits for tracing.

## Context (2026-11-01)
- k3: Ubuntu, 32 threads, 124 GB RAM (97 GB available), load 0.00, 731 GB free on /.
- Model files: our trained Omni-2B S3 (/opt/clover-k3/models/LFM3-Omni-2B-HybridSFT-S3) exported to the framework
  runtime's format in s23-omni2b (24 pruned trunks with 4-bit FFN; ncm/: C2 HybridLink adapter, C3 NCM core, VQ-VAE;
  tokenizer.json; liblfm3_cpu_engine.so). Base Omni GGUF Q4_0: /opt/clover-k3/memory-research/gguf/.
- Code: no device-free core exists yet. Framework: C1-C3, S1-S3 inside mobile/s23_on_device_clover.py; semantics in
  mobile/semantics.py (common since 2d68166). Laptop: scratch-agents/lfm3-hybridagent/laptop/clover (Windows; C1 via
  llama-server; its own S1/S2/S3 modules; semantics.py is a byte copy).
- clover-ai-system/CLOVER_FRAMEWORK_ARCHITECTURE.md describes the intended design; the observation records what runs.
- Noticed, not acted on: redis-server listens on 0.0.0.0:6379 and [::]:6379 on k3.

## Plan
1. Clone the pushed framework into the isolated folder; copy the laptop clover/ tree beside it.
2. Map each stack's entry points for input, semantics, C1, C2, C3, S1, S2, S3 and output.
3. Run both on k3, with the same model weights where the formats allow.
4. A harness wraps those entry points from outside and records each turn's per-layer inputs, outputs and timings.
5. The same inputs through both: greeting, question, tool request, file, photo, follow-up, try again, memory.
6. Document what each layer does and how the layers hand off: observations with evidence, no fixes.

## Progress
- 2026-11-01: four commits made (2d68166 semantics; ba43256, 77b75b3, db2486d other sessions' work). Direction recorded.
- 2026-11-01: pushed (75f8c5d; the push also deployed the website, as the human chose). Both stacks set up on k3 in
  /opt/clover-k3/core-observe as sandboxed services; 15 messages run through each; observation written in README.md
  with traces/ as evidence. One device part was substituted for the laptop stack on Linux: its Python runner
  (harness/linux_device). Services left running for further observation; stop with harness/k3.sh stop.
- Next (human's call): decide from the observation what the device-free core should be. The core and user semantics
  design waits for that.
