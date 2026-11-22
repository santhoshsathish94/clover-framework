# HG4 Fusion Gateway (Phase 1a — `p1a-hg4-v1-fusion-gateway`)

This is an OpenAI-compatible `/v1` server with **Copilot Fusion** and **Elephant–Goldfish (EGM)** virtual models. It exposes the native C11 `HG4-16R` engines to **OpenCode**, **opencode-egm** and **VS Code Copilot custom endpoints**.

It needs only the Python stdlib, `numpy`, `ctypes` and `tokenizers`. If `tokenizers` is missing, it falls back to the pure-Python `studio/clover/omni_c2_c3.py` tokenizer.

```
python gateway.py                 # http://127.0.0.1:8790/v1 — native engines, mock fallback per model
python gateway.py --mode real     # native engines only (unavailable models are omitted)
python gateway.py --mode mock     # deterministic offline mock engines
python test_hg4_fusion_gateway.py # full test-suite (mock + real Omni DLL smoke test when present)
```

## Files

| File | Purpose |
|---|---|
| `tool_grammar.py` | Renders tool prompts (Hermes JSON `<tool_call>` or `<function=..><parameter=..>` XML). Parses every common dialect (Hermes/Qwen, XML tags, Llama `<\|python_tag\|>`, Mistral `[TOOL_CALLS]`, fenced JSON). Repairs JSON deterministically, coerces types against the JSON Schema and validates required parameters. |
| `backends.py` | ctypes adapters for the 4 engines. Holds the global `ENGINE_LOCK` (`RLock`), the LCP prefix cache, exact-token assistant splicing, `MockEngine`/`MockTokenizer`, and the Kimi proxy/native fallback. |
| `fusion_router.py` | Direct and virtual models, plus Cascade escalation, the Critique auditor, Goldfish enforcement and `x_clover_telemetry`. |
| `gateway.py` | `ThreadingHTTPServer` serving `/health`, `/v1/models`, `/v1/chat/completions` (JSON + SSE) and `/v1/telemetry`. |
| `configs/opencode.json` | Ready-to-use OpenCode / opencode-egm config. |
| `configs/vscode-settings.json` | VS Code Copilot `customOAIModels` entries. |

## Engines and ctypes contracts

| Model id | D / V | HG4 dir (env override) | Library | Cache API used |
|---|---|---|---|---|
| `lfm3-omni-2b` | 2048 / 128000 | `CLOVER_OMNI_HG4_DIR`, `%USERPROFILE%\CloverAI\data\omni\hg4`, `/opt/clover-k3/lfm3-omni-hg4` | `studio/clover/fused/liblfm3_hg4_engine.dll` / `.so` | `lfm3_hg4_init`, `lfm3_hg4_cache_init`, `lfm3_hg4_cache_truncate`, `lfm3_hg4_cache_get_len`, `lfm3_hg4_cache_feed` |
| `clover-35b-a3b` | 2048 / 248320 | `CLOVER35B_HG4_DIR`, `/opt/clover-k3/clover35b-full-hg4` | `libclover35b_hg4_engine.so` | `clover35b_hg4_init`, `clover35b_hg4_cache_reset`, `clover35b_hg4_cache_feed` (ChatML, eos `<\|im_end\|>`=248046) |
| `nemotron-3.5-super-vl` | 4096 / 131072 | `CLOVER_NEMO_HG4_DIR`, `/opt/clover-k3/nemotron-full-hg4` | `libnemotron_hg4_engine.so` | `nemo_engine_init`, `nemo_cache_reset`, `nemo_cache_feed`. Optional `nemo_generate_speculative` (2L-MTP) with `CLOVER_NEMO_SPECULATIVE=1` |
| `kimi-k3` (`kimi-k3.1-pro`) | 7168 / 163840 | `CLOVER_KIMI_URL` (default `http://127.0.0.1:8765`), `CLOVER_KIMI_HG4_DIR` | `/api/chat` proxy → `libkimi_k31_hg4_engine.so` | `kimi_k31_hg4_init`, `kimi_k31_generate_speculative` |

Extra library search directories can be listed in `CLOVER_NATIVE_LIB_DIR` (separated by `os.pathsep`). Context sizes are set with `CLOVER_OMNI_CTX`, `CLOVER35B_CTX` and `CLOVER_NEMO_CTX`.

### Longest-Common-Prefix (LCP) state reuse

Each stateful backend tracks `last_cached_ids`, an exact mirror of the native cache.

* **Omni:** calls `lfm3_hg4_cache_truncate(lcp)` and then feeds only `prompt_ids[lcp:]`.
* **Clover-35B / Nemotron:** these models keep Gated-DeltaNet / Mamba recurrent state, which cannot be rolled back. When the new prompt strictly extends `last_cached_ids` (the next OpenCode turn, or an appended tool result), only the suffix is fed through `*_cache_feed`. The 10K-token system/repo prefix is not prefilled again. If the prefix diverges, the gateway calls `*_cache_reset()` and does a full prefill.
* **Exact-token splicing:** prompts are encoded one segment at a time, and every segment starts at a special token, so concatenating them is lossless. When a client sends back an assistant turn that the gateway generated, that turn is replaced with the token ids the model actually produced. The next request is then a strict token-level extension of the cache.
* **Serialization:** all engine work runs under one process-wide `RLock`. Concurrent subagents (for example 1 Elephant + 2 Goldfish) run one after another instead of competing for the 16-core OpenMP pool and DDR5 bandwidth.

## Models

| Model | Behaviour |
|---|---|
| `lfm3-omni-2b`, `clover-35b-a3b`, `nemotron-3.5-super-vl`, `kimi-k3` | Direct. An invalid tool call gets one corrective retry. |
| `clover-fusion-standard` | Trivial triage/formatting (short, no code markers) → Omni-2B. Coding, tool and tool-result turns → Clover-35B. |
| `clover-fusion-high` | **Cascade.** Starts on Clover-35B. Escalates to Nemotron-120B **only** on objective signals: (a) the trailing `tool` output matches compiler/Lean/pytest markers (`error:`, `unsolved goals`, `Traceback`, `FAILED`, `N failed`, `error TS…`, `panicked at`, non-zero exit codes), or (b) the tool call is still schema-invalid after one retry. Self-reported confidence never triggers escalation. |
| `clover-fusion-max` | **Critique.** Drafts on Nemotron. Runs one stateless auditor pass on Clover-35B (≤ 350 tokens, non-veto, lines starting `FLAW:`). Revises on Nemotron only when the auditor flags a concrete flaw. Tool-call drafts skip the audit. |
| `clover-egm-elephant` | Nemotron-120B with an Elephant system role (specs, theorem decomposition). |
| `clover-egm-goldfish` | Clover-35B with a hard `max_tokens ≤ 400` cap. Output is forced into `{"findings":[...],"verdict":"non-veto"}` JSON. |

Every response, and the final SSE chunk, includes `x_clover_telemetry`. It reports per-step tokens, `cached_tokens`, `prefill_tokens`, `cache_hit_pct`, `prefill_ms`, `decode_ms`, `escalations`, `escalation_reason`/`marker`, tool-call repairs and retries, audit findings, `wall_ms` and `decode_tokens_per_sec`. `usage.prompt_tokens_details.cached_tokens` is filled in as well. `GET /v1/telemetry` returns cumulative cache-hit rate, tokens/sec, and Cascade escalation counts and reasons.

## API notes

* `POST /v1/chat/completions` supports `stream` (SSE `delta.content`, OpenAI-spec `delta.tool_calls` with `index`/`id`/`function.name` followed by argument fragments, then `finish_reason` and `[DONE]`), `stream_options.include_usage`, `temperature` (defaults to greedy when omitted), `top_p`, `seed`, `max_tokens`/`max_completion_tokens`, `stop`, `tools`, and `tool_choice` (`auto`, `none`, `required`, or a named function).
* While streaming with tools, plain content is streamed live and any tool-call markup is held back. The calls are emitted as validated `tool_calls` deltas, so clients never receive malformed `arguments` JSON.
* Optional auth: set `--api-key` or `CLOVER_GATEWAY_API_KEY` and send `Authorization: Bearer <key>`. `/health` stays open.

## OpenCode / opencode-egm / VS Code

1. **OpenCode:** copy `configs/opencode.json` to your project root, or merge it into `~/.config/opencode/opencode.json`. The default CPU primary model is `clover/clover-35b-a3b` (with `TokenEconomizer`, `OutputBudgeter`, `SKILL.md` index, and `REA` MCP enabled by default); `clover/clover-fusion-high`, `elephant` (primary), and `goldfish` (read-only subagent) remain available on demand.
2. **opencode-egm** (`google/elephant-goldfish-model/adapters/opencode`): point its Elephant model at `clover/clover-egm-elephant` and its Goldfish model at `clover/clover-egm-goldfish`, using the same `clover` provider block. The gateway already enforces the Goldfish token cap and the structured non-veto findings.
3. **VS Code Copilot:** merge `configs/vscode-settings.json` into your user `settings.json`, then pick the models from the Copilot model picker. This uses the custom OpenAI-compatible endpoint support.
