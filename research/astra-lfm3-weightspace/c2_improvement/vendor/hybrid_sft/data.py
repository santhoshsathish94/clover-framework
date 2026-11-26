"""HybridSFT data: converts hybrid trajectories into token ids with a
text-loss mask (backbone CE targets, exit anchor up-weighted) and a latent mask
(positions whose HybridLink output is regressed onto the next input embedding).
Loads a local JSONL file or a file from a Hugging Face dataset repo.

Record format, one JSON object per line:
  {"id": ..., "system": "<system prompt>", "question": "...",
   "rounds": [{"assistant": "<think>\\n...<latent>...</latent>...\\n</think><python>...</python>",
               "tool": "python" | "search" | null,
               "tool_result": "<result>...</result>" | null}, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR
from hybrid_agent.prompts import SYSTEM_PROMPT

EXIT_ANCHOR_CE_WEIGHT = 2.0

THINK_PREFILL = "<think>\n"
_LATENT_SPAN_RE = re.compile(re.escape(ENTRY_ANCHOR) + r"(.*?)" + re.escape(EXIT_ANCHOR), re.DOTALL)


def resolve_data_file(data: str, filename: str) -> Path:
    """Local file as is; otherwise `filename` from the Hugging Face dataset repo `data`."""
    path = Path(data)
    if path.is_file():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=data, filename=filename, repo_type="dataset"))


def _build_chat_messages(system: str, question: str) -> list[dict]:
    messages: list[dict] = []
    if (system or "").strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def build_prompt(tokenizer: PreTrainedTokenizerBase, system: str, question: str) -> str | None:
    """Generation prompt from the chat template; None if the template is unusable."""
    try:
        prompt = tokenizer.apply_chat_template(
            _build_chat_messages(system, question), add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        return None
    if not isinstance(prompt, str):
        return None
    # The template either prefills "<think>\n" or ends at the assistant header.
    if prompt.endswith(THINK_PREFILL) or prompt.endswith("assistant\n"):
        return prompt
    return None


def _first_round_ok(rounds: list[dict]) -> bool:
    """Round 0 must open with the think prefill (a bare answer is only legal after a tool round)."""
    if not rounds:
        return False
    return (rounds[0].get("assistant") or "").startswith(THINK_PREFILL)


def _tokenize_assistant_text(
    tokenizer: PreTrainedTokenizerBase, chunk: str,
) -> tuple[list[int], list[float], list[int]] | None:
    """Ids and masks of one assistant chunk: text -> loss 1 / latent 0; `<latent>` -> 1 / 1;
    latent content -> 0 / 1; `</latent>` -> EXIT_ANCHOR_CE_WEIGHT / 0."""
    ids: list[int] = []
    loss: list[float] = []
    lat: list[int] = []
    pos = 0
    while pos < len(chunk):
        m = _LATENT_SPAN_RE.search(chunk, pos)
        plain_end = m.start() if m else len(chunk)
        if plain_end > pos:
            piece_ids = tokenizer(chunk[pos:plain_end], add_special_tokens=False)["input_ids"]
            ids.extend(piece_ids)
            loss.extend([1] * len(piece_ids))
            lat.extend([0] * len(piece_ids))
        if m is None:
            break
        open_ids = tokenizer(ENTRY_ANCHOR, add_special_tokens=False)["input_ids"]
        content_ids = tokenizer(m.group(1), add_special_tokens=False)["input_ids"]
        close_ids = tokenizer(EXIT_ANCHOR, add_special_tokens=False)["input_ids"]
        if len(open_ids) != 1 or len(close_ids) != 1 or not content_ids:
            return None
        ids.extend(open_ids + content_ids + close_ids)
        loss.extend([1] + [0] * len(content_ids) + [EXIT_ANCHOR_CE_WEIGHT])
        lat.extend([1] + [1] * len(content_ids) + [0])
        pos = m.end()
    return ids, loss, lat


def tokenize_trajectory(
    tokenizer: PreTrainedTokenizerBase,
    system: str,
    question: str,
    rounds: list[dict],
    max_length: int,
    text_only: bool = False,
) -> dict[str, Any] | None:
    """Tokenize one trajectory segment by segment (prompt, assistant chunks, tool results)
    so token boundaries match multi-turn generation. Tool results are context only (no
    loss). With `text_only`, every assistant token is a CE target and the latent mask is
    empty (format activation). Over-length trajectories are dropped, not truncated."""
    prompt = build_prompt(tokenizer, system, question)
    if prompt is None or not _first_round_ok(rounds):
        return None
    prompt_has_prefill = prompt.endswith(THINK_PREFILL)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids: list[int] = list(prompt_ids)
    loss_mask: list[float] = [0] * len(prompt_ids)
    latent_mask: list[int] = [0] * len(prompt_ids)
    prompt_len = len(prompt_ids)

    for i, rd in enumerate(rounds):
        a = rd.get("assistant") or ""
        chunk = a[len(THINK_PREFILL):] if (i == 0 and prompt_has_prefill) else a
        out = _tokenize_assistant_text(tokenizer, chunk)
        if out is None:
            return None
        ids, loss, lat = out
        if text_only:
            loss = [1] * len(ids)
            lat = [0] * len(ids)
        input_ids.extend(ids)
        loss_mask.extend(loss)
        latent_mask.extend(lat)

        tr = rd.get("tool_result")
        if tr:
            r_ids = tokenizer(tr, add_special_tokens=False)["input_ids"]
            input_ids.extend(r_ids)
            loss_mask.extend([0] * len(r_ids))
            latent_mask.extend([0] * len(r_ids))

    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        loss_mask.append(1)
        latent_mask.append(0)

    if len(input_ids) > max_length:
        return None

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "latent_mask": latent_mask,
        "prompt_len": prompt_len,
        "n_tokens": len(input_ids),
    }


class HybridSFTDataset(Dataset):
    """Tokenized hybrid trajectories from one JSONL file."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        path: str | Path,
        max_length: int,
        text_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_only = text_only
        self.samples: list[dict[str, Any]] = []
        n_skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        for line in tqdm(lines, desc="Tokenizing", unit=" examples"):
            rec = json.loads(line)
            rounds = rec.get("rounds")
            if not isinstance(rounds, list) or not rounds:
                n_skipped += 1
                continue
            out = tokenize_trajectory(
                tokenizer,
                rec.get("system") or SYSTEM_PROMPT,
                rec.get("question", ""),
                rounds,
                self.max_length,
                text_only=self.text_only,
            )
            if out is None:
                n_skipped += 1
                continue
            self.samples.append(out)
        if n_skipped:
            print(f"[data] kept {len(self.samples)} records, skipped {n_skipped} "
                  f"(over-length > {max_length} tokens or malformed)", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx].copy()


def collate_batch(
    tokenizer: PreTrainedTokenizerBase,
    features: list[dict],
    pad_to_multiple: int = 8,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(f["n_tokens"] for f in features)
    if pad_to_multiple > 1:
        max_len = (max_len + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple

    input_ids = []
    attention_mask = []
    loss_mask = []
    latent_mask = []
    for f in features:
        ids = f["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        loss_mask.append(f["loss_mask"] + [0] * pad_len)
        latent_mask.append(f["latent_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        "latent_mask": torch.tensor(latent_mask, dtype=torch.float32),
    }
