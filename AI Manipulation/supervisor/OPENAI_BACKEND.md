# OpenAI Model Backend

This backend attaches a hosted OpenAI model to the provider-independent Clover supervisor.

## Credential boundary

The supervisor does not require an API key.

This backend does, because hosted model inference requires authenticated API access.

Set:

```bash
export OPENAI_API_KEY="..."
export CLOVER_MODEL="gpt-5.6-luna"
```

Then instantiate `OpenAIBackend` from the supervisor.

## Why this matters

The model is now an interchangeable worker.

The supervisor owns:

- continuity
- state
- scheduling
- environment
- evaluation
- stopping

The model owns:

- reasoning for the current cycle

That lets the experiment test whether developmental continuity belongs to the surrounding process rather than to the model session.

Current OpenAI documentation lists GPT-5.6 Luna as available through the Responses API and client SDKs.
