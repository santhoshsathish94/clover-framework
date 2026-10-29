"""Tokens and tags of the hybrid trajectory format.

`<latent>` and `</latent>` are the entry and exit anchors of latent mode; tool calls,
tool observations and the final answer are written in text mode.
"""
from __future__ import annotations

ENTRY_ANCHOR = "<latent>"
EXIT_ANCHOR = "</latent>"
ANCHOR_TOKENS = (ENTRY_ANCHOR, EXIT_ANCHOR)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
RESULT_OPEN = "<result>"
RESULT_CLOSE = "</result>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

# A closing tool tag ends the current turn and triggers the tool call.
TOOL_TAGS = ("search", "python")
TOOL_STOP_TAGS = tuple(f"</{tag}>" for tag in TOOL_TAGS)


def special_token_id(tokenizer, token: str) -> int:
    """Id of a special token; fails loudly if the tokenizer does not know it."""
    tok_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = tokenizer.unk_token_id
    if tok_id is None or (unk_id is not None and tok_id == unk_id):
        raise ValueError(
            f"Tokenizer does not know the special token {token!r}; "
            "load a HybridAgent checkpoint or add the anchor tokens first."
        )
    return int(tok_id)
