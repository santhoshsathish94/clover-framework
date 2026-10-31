#!/usr/bin/env python3
"""
Builds the Nemotron 3.5 Super VL (`d=4096`, `V=131072` Tekken tokenizer) 3-Stage HybridSFT
dataset (`train-2400-nemotron.jsonl` and `eval-32-nemotron.jsonl`) from HybridSFT-v2 (`data/train.jsonl`).

Special Token Mapping inside Nemotron's 131,072 Tekken Vocabulary:
  - `<think>`   -> ID 12 (native Tekken special token)
  - `</think>`  -> ID 13 (native Tekken special token)
  - `<latent>`  -> ID 30 (`<SPECIAL_30>`, mapped to `<latent>`)
  - `</latent>` -> ID 31 (`<SPECIAL_31>`, mapped to `</latent>`)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

THINK_START_ID = 12
THINK_END_ID = 13
LATENT_START_ID = 30
LATENT_END_ID = 31

SPAN_SPLIT_RE = re.compile(r"(<think>|</think>|<latent>|</latent>)")


def encode_text_with_nemotron_specials(tokenizer: Any, text: str) -> list[int]:
    """Encodes text with exact single-token IDs for `<think>` (12), `</think>` (13), `<latent>` (30), `</latent>` (31)."""
    parts = SPAN_SPLIT_RE.split(text)
    ids: list[int] = []
    for p in parts:
        if not p:
            continue
        if p == "<think>":
            ids.append(THINK_START_ID)
        elif p == "</think>":
            ids.append(THINK_END_ID)
        elif p == "<latent>":
            ids.append(LATENT_START_ID)
        elif p == "</latent>":
            ids.append(LATENT_END_ID)
        else:
            ids.extend(tokenizer.encode(p, add_special_tokens=False))
    return ids


def tokenize_trajectory_for_nemotron(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_seq_len: int = 512,
    max_latent_steps: int = 8,
) -> dict[str, Any] | None:
    """
    Tokenizes a multi-turn HybridSFT trajectory into `input_ids`, `labels`, and `latent_mask`
    using Nemotron's `<|im_start|>role\\n...<|im_end|>\\n` chat format and Tekken special token IDs.
    """
    im_start = 10  # `<|im_start|>`
    im_end = 11    # `<|im_end|>`
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    input_ids: list[int] = [1]  # `<s>` BOS
    labels: list[int] = [-100]

    for msg in messages:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "") or "")
        header_ids = [im_start] + tokenizer.encode(f"{role}\n", add_special_tokens=False)
        body_ids = encode_text_with_nemotron_specials(tokenizer, content)
        footer_ids = [im_end] + nl_ids

        # Compress any long interior `<latent> ... </latent>` span to at most `max_latent_steps` latent tokens
        compressed_body: list[int] = []
        i = 0
        while i < len(body_ids):
            if body_ids[i] == LATENT_START_ID:
                j = i + 1
                while j < len(body_ids) and body_ids[j] != LATENT_END_ID:
                    j += 1
                if j < len(body_ids) and body_ids[j] == LATENT_END_ID:
                    interior = body_ids[i + 1 : j]
                    if len(interior) > max_latent_steps:
                        # Subsample interior tokens evenly to preserve semantic trajectory
                        step = len(interior) / float(max_latent_steps)
                        interior = [interior[int(k * step)] for k in range(max_latent_steps)]
                    elif len(interior) == 0:
                        interior = tokenizer.encode("plan verify synthesize", add_special_tokens=False)[:3]
                    compressed_body.append(LATENT_START_ID)
                    compressed_body.extend(interior)
                    compressed_body.append(LATENT_END_ID)
                    i = j + 1
                    continue
            compressed_body.append(body_ids[i])
            i += 1

        turn_ids = header_ids + compressed_body + footer_ids
        if role == "assistant":
            turn_labels = [-100] * len(header_ids) + compressed_body + [im_end] + [-100] * len(nl_ids)
        else:
            turn_labels = [-100] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    # Build `latent_mask` and mask interior latent tokens in `labels` (keeping `<latent>`=30 and `</latent>`=31 supervised!)
    latent_mask = [0] * len(input_ids)
    in_latent = False
    has_complete_span = False
    for idx, tok_id in enumerate(input_ids):
        if tok_id == LATENT_START_ID:
            in_latent = True
            latent_mask[idx] = 1
            # `<latent>` itself is predicted by the LM head so the model learns when to enter latent mode!
        elif tok_id == LATENT_END_ID and in_latent:
            in_latent = False
            has_complete_span = True
            # `</latent>` is predicted by the LM head so the model learns when to exit latent mode!
        elif in_latent:
            latent_mask[idx] = 1
            labels[idx] = -100  # Interior latent steps are supervised via HybridLink cosine alignment, not token CE

    supervised_tokens = sum(1 for y in labels if y != -100)
    latent_steps = sum(latent_mask)
    if not has_complete_span or supervised_tokens < 8 or latent_steps < 2:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "latent_mask": latent_mask,
    }


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    msgs: list[dict[str, str]] = []
    sys_txt = str(row.get("system", "") or "").strip()
    if sys_txt:
        msgs.append({"role": "system", "content": sys_txt})
    q_txt = str(row.get("question", "") or "").strip()
    if q_txt:
        msgs.append({"role": "user", "content": q_txt})
    for rd in row.get("rounds", []):
        asst = str(rd.get("assistant", "") or "").strip()
        if asst:
            msgs.append({"role": "assistant", "content": asst})
        obs = str(rd.get("tool_result") or rd.get("observation") or "").strip()
        if obs:
            msgs.append({"role": "user", "content": obs})
    return msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-source-jsonl",
        type=str,
        default="/dev/shm/clover-train/lfm3-hybridagent/data/splits/train-2400-omni2b.jsonl",
    )
    parser.add_argument(
        "--eval-source-jsonl",
        type=str,
        default="/dev/shm/clover-train/lfm3-hybridagent/data/splits/eval-32-omni2b.jsonl",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/model_meta",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/data",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_nemotron_hybridsft] Loading Nemotron Tekken tokenizer from {args.tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    train_records: list[dict[str, Any]] = []
    with open(args.train_source_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row_to_messages(row)
            item = tokenize_trajectory_for_nemotron(tokenizer, msgs, max_seq_len=args.max_seq_len)
            if item is not None:
                item["id"] = str(row.get("id", len(train_records)))
                train_records.append(item)

    eval_records: list[dict[str, Any]] = []
    with open(args.eval_source_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row_to_messages(row)
            item = tokenize_trajectory_for_nemotron(tokenizer, msgs, max_seq_len=args.max_seq_len)
            if item is not None:
                item["id"] = str(row.get("id", len(eval_records)))
                eval_records.append(item)

    train_path = out_dir / "train-2400-nemotron.jsonl"
    eval_path = out_dir / "eval-32-nemotron.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    avg_len = sum(len(r["input_ids"]) for r in train_records) / max(1, len(train_records))
    avg_lat = sum(sum(r["latent_mask"]) for r in train_records) / max(1, len(train_records))
    print(
        f"[build_nemotron_hybridsft] Wrote {len(train_records)} train -> {train_path} | "
        f"{len(eval_records)} eval -> {eval_path} | avg_len={avg_len:.1f} | avg_latent_tokens={avg_lat:.1f}"
    )


if __name__ == "__main__":
    main()
