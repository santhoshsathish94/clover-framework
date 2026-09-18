# Local Model Adapter

The supervisor can use a local OpenAI-compatible Responses endpoint.

Example:

```bash
export CLOVER_LOCAL_MODEL_URL=http://localhost:11434/v1/responses
export CLOVER_LOCAL_MODEL=gpt-oss
python -c "from supervisor.supervisor import run; from supervisor.local_backend import LocalResponsesBackend; run(LocalResponsesBackend())"
```

No OpenAI API key is required if the local inference server does not require authentication.

This separates:

1. Supervisor — owns continuity.
2. Inference runtime — owns model computation.
3. Hosted provider — optional.

OpenAI documents local OSS inference through an OpenAI-compatible localhost Responses endpoint in its Codex agent-loop documentation.

The local model still needs to be installed and running somewhere. This repository does not assume that model weights or GPU infrastructure are available automatically.
