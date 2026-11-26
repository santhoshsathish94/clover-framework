"""Final-answer helpers for hybrid trajectories: extract the \\boxed{} answer of the
<answer> block and strip latent spans / tool observations from an output.
"""
from __future__ import annotations

import re

from .constants import ANSWER_CLOSE, ANSWER_OPEN


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return ""
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return string[idx:]
    return string[idx : right_brace_idx + 1]


def _unbox(boxed: str) -> str:
    if boxed and boxed.startswith("\\boxed{") and boxed.endswith("}"):
        boxed = boxed[7:-1]
        if boxed and boxed.startswith("\\text{") and boxed.endswith("}"):
            boxed = boxed[6:-1]
    return boxed


def extract_answer(full_text: str) -> str:
    """Content of the last \\boxed{} inside the final <answer> block (fallback: whole text)."""
    text = full_text
    last_answer_end = text.rfind(ANSWER_CLOSE)
    temp_answer = None
    if last_answer_end != -1:
        last_answer_start = text[:last_answer_end].rfind(ANSWER_OPEN)
        if last_answer_start != -1:
            temp_answer = text[last_answer_start + len(ANSWER_OPEN) : last_answer_end]
    if temp_answer:
        boxed_answer = _unbox(last_boxed_only_string(temp_answer.strip()))
        return boxed_answer if boxed_answer else temp_answer
    return _unbox(last_boxed_only_string(text.strip()))


def strip_latent_spans(text: str) -> str:
    """Visible text only: latent spans and <result> observations removed."""
    if not text:
        return ""
    out = re.sub(r"<latent>.*?</latent>", "", text, flags=re.DOTALL)
    out = out.replace("<latent>", "").replace("</latent>", "")
    out = re.sub(r"<result>.*?</result>", "", out, flags=re.DOTALL)
    return out.strip()
