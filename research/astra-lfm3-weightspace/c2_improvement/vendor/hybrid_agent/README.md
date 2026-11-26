# HybridAgent (model architecture, generation engine, and tools)

The core package shared by HybridSFT, HybridRL and inference.

| Module | Contents |
|---|---|
| `hybridlink.py` | `HybridLink`, the residual adapter from last-layer hidden states to the input-embedding space |
| `model.py` | `HybridAgentModel` (backbone + HybridLink), anchor-token registration, checkpoint save/load, `load_hybrid_agent(path_or_repo)` |
| `engine.py` | `HybridGenerationEngine`: multi-turn hybrid generation over the working memory with concurrent tool calls |
| `sampling.py` | KV-cache fan-out, lockstep forward, sampling under the mode-dependent token masks |
| `readout.py` | nearest-token readout of latent inputs (Interpretable Latent Mode) |
| `prompts.py` | the system prompt defining the hybrid trajectory format |
| `tools/` | Python interpreter tool, Tavily search tool, tool executor |

## Generation engine

<p align="center"><img src="../assets/working_memory.png" width="100%" alt="Multi-turn hybrid generation over working memory"></p>

`HybridGenerationEngine` generates B rollouts of one prompt in lockstep over a shared working
memory: the prompt is prefilled once and its KV cache is fanned out to the rows; each row then
alternates between text mode (sampled tokens) and latent mode (HybridLink outputs fed back as
inputs until the exit-anchor readout closes the span), pauses on a closing tool tag, and resumes
from its own memory once the pending tool calls have executed concurrently.

```python
import torch
from hybrid_agent import HybridGenerationEngine, SYSTEM_PROMPT, build_tool_executor, load_hybrid_agent

model, tokenizer = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", device="cuda")
engine = HybridGenerationEngine(
    model, tokenizer, torch.device("cuda"), build_tool_executor(), SYSTEM_PROMPT,
    "How many trailing zeros does 100! have?", batch_size=4, record_latent_readout=True,
)
engine.run(torch.Generator(device="cuda").manual_seed(0))
for row in engine.rows:
    result = engine.row_result(row)
    print(result["output_rendered"], result["latent_steps"], result["end_reason"])
```

Constructor parameters: `max_response_tokens` (text tokens + latent steps per row; `count_tool_tokens=True` also counts observations), `max_tool_calls` / `max_search_calls` / `max_python_calls` and `terminate_on_tool_cap`, `max_turns`, `temperature` / `top_p` / `top_k` / `repetition_penalty`, `min_latent_steps` / `max_latent_steps`, `think_guards` (keep `<think>` tags well nested in text mode), `reject_fabricated_result` (end a row that writes `<result>` itself), `record_latent_readout` (attach the pseudo-text of every latent span), `tool_workers`.

`row_result(row)` returns the decoded output, the turn and tool-call counts, the generation and latent step counts, the number of spans closed on the exit anchor or at the step cap, and the end reason.

## Tools

- **Python interpreter** (`<python>…</python>`): runs the code in a separate subprocess with a timeout, an address-space limit and an output cap; the last bare expression is printed. Generated code runs as the current user.
- **Search** (`<search>…</search>`): Tavily web search returning the top results as `Page i: snippet` lines (the number of results and the snippet length are configurable). Needs `TAVILY_API_KEY`. Set `HYBRID_SEARCH_CACHE_DIR` to cache raw responses on disk across runs.

`build_tool_executor(python_timeout, search_max_results, search_result_length)` registers both; any object with `trigger_tag` and `execute(content)` can be registered as an additional tool.

## Checkpoints

`load_hybrid_agent` accepts a local directory or a Hugging Face repo id and returns the model and tokenizer with the anchor tokens registered. A checkpoint directory contains the backbone (`model.safetensors`, `config.json`), the tokenizer files, and the HybridLink weights `adapter.pt` with `adapter_config.json`.
