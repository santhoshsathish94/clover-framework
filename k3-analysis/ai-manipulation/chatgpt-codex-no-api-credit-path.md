# ChatGPT / Codex Path Without API Credits

## Purpose

This document records an alternative way to connect Clover's developmental runtime to OpenAI intelligence without using an OpenAI API key or API credit balance.

## Current API path

The current OpenAI worker uses:

`GitHub Actions → OPENAI_API_KEY → OpenAI Responses API → selected model`

This path was successfully verified up to the API request, but the test account returned:

`429 insufficient_quota / credit_balance_exhausted`

Therefore the API path requires API credits.

## Alternative path

OpenAI Codex can authenticate using a ChatGPT account rather than an OpenAI API key.

Conceptually:

`Clover runtime → Codex CLI → ChatGPT authentication → OpenAI model`

This is different from the API path because ChatGPT subscription usage and API billing are separate.

## Proposed Clover experiment

Run the developmental loop locally through Codex:

1. Authenticate Codex with the user's ChatGPT account.
2. Open the Clover repository.
3. Give Codex the developmental-engine objective.
4. Let the runtime inspect Clover's persistent developmental state.
5. Allow one bounded developmental cycle.
6. Record the result in `ai-manipulation/`.
7. Independently validate any claimed capability improvement.
8. Preserve provenance of what the model proposed, what the runtime executed, and what evidence was produced.

## Important security constraint

Do **not** place ChatGPT authentication tokens in the public Clover repository or its GitHub Actions secrets.

The repository is public. ChatGPT/Codex authentication credentials must be treated as sensitive credentials.

The safer experiment is local execution on a trusted machine, authenticated through Codex.

## Why this matters

The experiment tests a deeper Clover hypothesis:

> The model does not have to be the owner of continuity.

The surrounding runtime can own:

- developmental state
- memory
- experiments
- validation
- resource limits
- provenance
- capability promotion

The model can remain a replaceable intelligence worker.

This gives two possible workers:

`Clover runtime → OpenAI API worker`

or

`Clover runtime → Codex / ChatGPT-authenticated worker`

The second path is useful when API billing is unavailable, subject to the current Codex and ChatGPT plan capabilities.

## Current status

- API key: configured in GitHub repository secret.
- API authentication: verified.
- API model request: verified with `gpt-5.6-luna`.
- API inference: blocked by zero API credit balance.
- ChatGPT/Codex no-credit route: identified, not yet executed.
- Public GitHub CI using ChatGPT authentication: intentionally avoided because authentication credentials must not be exposed in the public repository.

## Next experiment

Set up Codex locally with ChatGPT authentication and run exactly one bounded Clover developmental cycle.

The result should distinguish:

`model reasoning → runtime execution → independent validation → persistent growth`

rather than treating a model response alone as evidence of development.

## References

- OpenAI Codex CLI documentation: https://developers.openai.com/codex/cli/
- OpenAI non-interactive Codex authentication documentation: https://developers.openai.com/docs/non-interactive-mode
- OpenAI ChatGPT/Codex usage documentation: https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
