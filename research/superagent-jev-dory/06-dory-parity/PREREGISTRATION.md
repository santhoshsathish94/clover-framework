# Pre-registration: Dory standalone engine against an independent implementation

Written before any comparison run. Part of the follow-up to
`docs/runtime-context/superagent-jev-dory/05-review-and-held-out-retest.md`, item 3 (approved 2026-11-09).

## Question

Does the study's standalone PyTorch engine (`StandaloneDoryModel` in
`research/superagent-jev-dory/03-nvidia-dory/run_dory_live_experiments.py`) compute the same function as an
independent implementation of NVIDIA Dory 1.0? Every loop-count finding in records 03 and 04 rests on it.

## Reference

- llama.cpp pull request #30306 (`brianchmielNV:dory-support`), head `1c3cdd23787750814aa21a070dc4d6dc451af871`
  (2026-10-29), built for CPU on k3. Written in C++/ggml by NVIDIA's author, separately from the study's engine.
  The pull request is a draft, not merged; its author reports comparisons against vLLM PR #60149.
- Checkpoint `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (the study's copy on k3), converted with the
  pull request's converter (transformers 4.57.6, as the pull request pins) to BF16 and F32 GGUF.
- Logits come from a small tool (`dory_logits.cpp`) on the pull request's `llama.h`: each prompt is decoded
  as one batch from an empty cache, with `dory.num_loops` and `dory.recurrent_kv_cache_mode` set by KV override.
  One batch per prompt keeps prefill chunking out of the comparison (chunking changes Dory logits, as the
  pull request discussion reports).

## Runs

- Prompts: every prompt in `prompts.json`, frozen (SHA-256 recorded) before the first model run: the study's
  8 mixed-workload prompts, its decision prompts, its two generation prompts, and generic texts, two of them
  longer than the 1024-token sliding window. Both engines receive identical token ids.
- Loops K = 1, 2, 4; cache mode `per_loop`; one-pass prefill; final-position logits over the full vocabulary.
- PRIMARY: standalone engine in fp32 (`model.float()`) against the F32 GGUF. Both run in full precision, so
  differences should come from the implementations rather than from rounding.
- SECONDARY: standalone engine in bf16 (as the study ran it) against the BF16 GGUF.
- CONTROL: standalone bf16 against standalone fp32 (what bf16 rounding alone does inside the engine).
- LIVENESS: two deliberate defects in the standalone engine, fp32, K = 2, every prompt: M1 drops the input
  re-injection before each loop; M2 swaps the RoPE theta between sliding and full attention layers. The
  comparison must flag both; if it cannot, it cannot establish parity either. The un-mutated copy of the
  forward pass used for M1 must equal the engine's own forward exactly (max abs difference 0).

## Metrics per prompt and K

Top-1 agreement; overlap of the top-5 sets; KL(reference || standalone) over the full vocabulary; logit cosine;
difference in log-probability of the reference's top token.

## Decision rule (PRIMARY, applied to each K)

- PASS: top-1 agreement on all prompts but at most one, median KL <= 0.01, and 90th-percentile KL <= 0.1.
- FAIL: top-1 agreement misses on more than 4 prompts, or median KL > 0.1.
- Otherwise INCONCLUSIVE.
- The engine is called consistent with the reference only if all three K pass and both mutants FAIL.

## Also reported, no decision attached

- The study's single-case claims, re-run on the reference: the `w7` verification routing probability at
  K = 1, 2, 4 and the chain-of-thought case.
- Whether SECONDARY and CONTROL differences are of the same size (bf16 rounding versus implementation).

## Predictions (stated now)

- PRIMARY passes at all K: the study's output was coherent ("216", solved CoT), which a structurally wrong
  engine rarely gives, and its scales follow the published config. Confidence moderate: a subtle per-layer
  detail (for example the RoPE profile per layer) could still differ.
- Both mutants fail.
- SECONDARY shows larger differences than PRIMARY (bf16 rounding is amplified by Dory's learned scales).
