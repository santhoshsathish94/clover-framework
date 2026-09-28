# Local Model Adapter

The supervisor can use a local OpenAI-compatible endpoint. Two shapes exist and
they are not interchangeable — pointing one backend at the other's path returns
404.

## Responses shape (LM Studio)

```bash
export CLOVER_LOCAL_MODEL_URL=http://localhost:1234/v1/responses
export CLOVER_LOCAL_MODEL=openai/gpt-oss-20b
python -c "from supervisor.supervisor import run; from supervisor.local_backend import LocalResponsesBackend; run(LocalResponsesBackend())"
```

## Chat-completions shape (Ollama, llama.cpp server, vLLM)

Ollama listens on 11434 and serves `/v1/chat/completions`. It does **not** serve
`/v1/responses`, so `local_backend.py` cannot talk to it.

```bash
export CLOVER_LOCAL_CHAT_URL=http://localhost:11434/v1/chat/completions
python -c "from supervisor.supervisor import run; from supervisor.local_chat_backend import ChatCompletionsBackend; run(ChatCompletionsBackend('qwen2.5:0.5b'))"
```

No API key is required if the local server does not demand one.

This separates:

1. Supervisor — owns continuity.
2. Inference runtime — owns model computation.
3. Hosted provider — optional.

The local model still needs to be installed and running somewhere. This repository does not assume that model weights or GPU infrastructure are available automatically.
