"""Hybrid generation engine: multi-turn generation that switches between
text mode and latent mode over one persistent KV cache per trajectory, the agent's
working memory. Text mode samples tokens until the entry anchor `<latent>` is emitted;
latent mode feeds HybridLink outputs back as the next inputs until the exit-anchor
readout fires; tool calls execute concurrently at a barrier and their observations are
appended to the working memory as text. B rollouts of one prompt advance in lockstep.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from .constants import (
    ENTRY_ANCHOR,
    EXIT_ANCHOR,
    RESULT_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_STOP_TAGS,
    special_token_id,
)
from .readout import LatentSpanReadout, nearest_token_ids, normalized_embeddings, pseudo_text, render_with_readout
from .sampling import expand_cache_batch, forward_step, sample_rows, token_log_probs
from .tools import ToolExecutor

DECODE, LATENT, FEED, IDLE, DONE = "DECODE", "LATENT", "FEED", "IDLE", "DONE"


@dataclass
class RowState:
    """Per-trajectory state of one row of the lockstep batch."""
    idx: int
    mode: str = DECODE
    in_think: bool = False
    latent_steps: int = 0                       # steps of the open latent span
    generated_ids: list[int] = field(default_factory=list)
    text_ids_since_resume: list[int] = field(default_factory=list)
    kv_ledger: list[int | None] = field(default_factory=list)   # token per cache slot, None = latent
    generation_steps: int = 0                   # text tokens + latent steps
    latent_steps_total: int = 0
    latent_spans_opened: int = 0
    anchor_exits: int = 0
    forced_exits: int = 0
    turn_count: int = 1
    python_calls: int = 0
    search_calls: int = 0
    tool_requests: int = 0
    search_query_history: set[str] = field(default_factory=set)
    full_output: str = ""
    _last_decoded_len: int = 0
    pending_tool: tuple[str, str] | None = None
    feed_queue: list[int] = field(default_factory=list)
    readout_ids: list[int] = field(default_factory=list)        # readout of the open span
    latent_spans: list[LatentSpanReadout] = field(default_factory=list)
    end_reason: str = ""                        # "eos" | "budget" | "tool_cap" | "max_turns"
    extra: dict = field(default_factory=dict)

    @property
    def true_len(self) -> int:
        return len(self.kv_ledger)


class HybridGenerationEngine:
    """B rollouts of one prompt, generated in lockstep with text/latent mode transitions,
    a shared working memory per row, and concurrent tool execution."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        tool_executor: ToolExecutor,
        system_prompt: str,
        question: str,
        *,
        batch_size: int,
        max_response_tokens: int = 4096,
        count_tool_tokens: bool = False,
        max_tool_calls: int | None = None,
        max_search_calls: int | None = None,
        max_python_calls: int | None = None,
        terminate_on_tool_cap: bool = False,
        max_turns: int = 30,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        min_latent_steps: int = 4,
        max_latent_steps: int = 200,
        think_guards: bool = True,
        reject_fabricated_result: bool = True,
        record_latent_readout: bool = False,
        tool_workers: int = 8,
    ):
        self.model = model
        self.backbone = model.backbone
        self.tokenizer = tokenizer
        self.device = device
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.question = question
        self.B = int(batch_size)

        # Budget per trajectory: text tokens + latent steps, or every slot after the
        # prompt (tool observations included) when count_tool_tokens is set.
        self.max_response_tokens = int(max_response_tokens)
        self.count_tool_tokens = bool(count_tool_tokens)
        self.max_tool_calls = max_tool_calls
        self.max_search_calls = max_search_calls
        self.max_python_calls = max_python_calls
        self.terminate_on_tool_cap = bool(terminate_on_tool_cap)
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = float(repetition_penalty)
        self.min_latent_steps = min_latent_steps
        self.max_latent_steps = max_latent_steps
        self.think_guards = bool(think_guards)
        self.reject_fabricated_result = bool(reject_fabricated_result)
        self.record_latent_readout = bool(record_latent_readout)
        self.tool_stop_tags = tuple(TOOL_STOP_TAGS)
        self.tool_workers = int(tool_workers)

        self.embed_layer = self.backbone.get_input_embeddings()
        self.latent_open_id = special_token_id(tokenizer, ENTRY_ANCHOR)
        self.latent_close_id = special_token_id(tokenizer, EXIT_ANCHOR)
        self.think_open_id = special_token_id(tokenizer, THINK_OPEN)
        self.think_close_id = special_token_id(tokenizer, THINK_CLOSE)
        self.eos_ids = self._collect_eos_ids()
        pad_id = tokenizer.pad_token_id
        self.pad_id = int(pad_id if pad_id is not None else tokenizer.eos_token_id)

        # Text-mode token masks: `</latent>` is only produced by the exit readout and
        # `<latent>` only inside <think>; think guards keep the think tags well nested.
        in_think = {self.latent_close_id}
        out_think = {self.latent_close_id, self.latent_open_id}
        if self.think_guards:
            in_think.add(self.think_open_id)
            out_think.add(self.think_close_id)
        self.blocked_token_sets = {"in": frozenset(in_think), "out": frozenset(out_think)}
        vocab = self.embed_layer.num_embeddings
        self._blocked_masks = {
            key: self._bool_mask(ids, vocab) for key, ids in self.blocked_token_sets.items()
        }
        self._readout_bank = (
            normalized_embeddings(self.embed_layer.weight) if self.record_latent_readout else None
        )

        # lockstep state
        self.rows = [RowState(idx=i) for i in range(self.B)]
        self.past_kv = None
        self.next_logits = None      # [B, V]
        self.last_hidden = None      # [B, D]
        self.attn_mask = None        # [B, L_total]
        self.positions = None        # [B] next RoPE position per row
        self.global_len = 0          # lockstep cache length (all rows equal)
        self.initial_prompt = ""
        self.prompt_len = 0
        self.prompt_ends_in_think = False

        self.llm_time = 0.0
        self.tool_time = 0.0
        self.total_time: float | None = None

    # ------------------------------------------------------------------
    def _bool_mask(self, ids, vocab) -> torch.Tensor:
        mask = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        if ids:
            mask[torch.tensor(sorted(ids), device=self.device, dtype=torch.long)] = True
        return mask

    def _collect_eos_ids(self) -> set[int]:
        config = getattr(self.backbone, "config", None)
        candidates = [
            self.tokenizer.eos_token_id,
            getattr(config, "eos_token_id", None),
            getattr(getattr(config, "text_config", None), "eos_token_id", None),
            getattr(getattr(self.backbone, "generation_config", None), "eos_token_id", None),
        ]
        eos: set[int] = set()
        for cand in candidates:
            if cand is None:
                continue
            for tok in cand if isinstance(cand, (list, tuple)) else [cand]:
                if tok is not None:
                    eos.add(int(tok))
        try:
            vocab = self.tokenizer.get_vocab()
        except (AttributeError, TypeError):
            vocab = {}
        if "<|endoftext|>" in vocab:
            eos.add(int(vocab["<|endoftext|>"]))
        return eos

    # ------------------------------------------------------------------
    # Prefill: shared prompt at batch 1, then fan the working memory out to B rows
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        self.initial_prompt = prompt
        return prompt

    @torch.no_grad()
    def _init_generation(self) -> bool:
        prompt = self._build_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        self.prompt_len = len(prompt_ids)
        self.prompt_ends_in_think = prompt.rstrip("\n").endswith(THINK_OPEN)
        if not self._prompt_length_ok(len(prompt_ids)):
            return False

        prefill_kwargs = dict(use_cache=True, output_hidden_states=True, return_dict=True)
        try:
            outputs = self.backbone(**inputs, logits_to_keep=1, **prefill_kwargs)
        except TypeError:
            outputs = self.backbone(**inputs, **prefill_kwargs)
        self.past_kv = outputs.past_key_values
        expand_cache_batch(self.past_kv, self.B)
        self.next_logits = outputs.logits[:, -1, :].repeat(self.B, 1)
        self.last_hidden = outputs.hidden_states[-1][:, -1, :].repeat(self.B, 1)
        self.global_len = len(prompt_ids)
        self.attn_mask = torch.ones((self.B, self.global_len), dtype=torch.long, device=self.device)
        self.positions = torch.full((self.B,), self.global_len, dtype=torch.long, device=self.device)
        for row in self.rows:
            row.kv_ledger = list(prompt_ids)
            row.in_think = self.prompt_ends_in_think
            row.mode = DECODE
        return True

    def _prompt_length_ok(self, n: int) -> bool:  # HybridRL overrides
        return True

    # ------------------------------------------------------------------
    # Lockstep stepping
    # ------------------------------------------------------------------

    def _row_budget_left(self, row: RowState) -> int:
        if self.count_tool_tokens:
            return self.max_response_tokens - (row.true_len - self.prompt_len)
        return self.max_response_tokens - row.generation_steps

    @torch.no_grad()
    def _lockstep_forward(self, inputs_embeds: torch.Tensor, real_rows: list[int]):
        """One batched forward; only `real_rows` slots carry real content, the other
        slots are masked out of attention forever and do not advance their row's position."""
        new_mask_col = torch.zeros((self.B, 1), dtype=torch.long, device=self.device)
        if real_rows:
            new_mask_col[torch.tensor(real_rows, device=self.device), 0] = 1
        self.attn_mask = torch.cat([self.attn_mask, new_mask_col], dim=1)
        position_ids = self.positions.unsqueeze(1)
        cache_position = torch.tensor([self.global_len], device=self.device)
        t0 = time.time()
        self.past_kv, self.next_logits, self.last_hidden = forward_step(
            self.backbone,
            input_embeds=inputs_embeds,
            past_key_values=self.past_kv,
            attention_mask=self.attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        self.llm_time += time.time() - t0
        self.global_len += 1
        if real_rows:
            idx = torch.tensor(real_rows, device=self.device)
            self.positions[idx] += 1

    # -- per-row hooks HybridRL overrides to record the trajectory ---------

    def _record_text_action(self, row: RowState, token_id: int,
                            log_probs_row: torch.Tensor, blocked_key: str) -> None:
        pass

    def _record_latent_vec(self, row: RowState, injected_row: torch.Tensor) -> None:
        pass

    def _on_env_token_fed(self, row: RowState) -> None:
        pass

    def _on_latent_span_opened(self, row: RowState) -> None:
        pass

    def _on_latent_span_closed(self, row: RowState) -> None:
        pass

    def _sequence_cap_ok(self, row: RowState) -> bool:
        return True

    def _env_tokens_fit(self, row: RowState, n_new: int) -> bool:
        return True

    def _row_invalidate(self, row: RowState, reason: str) -> None:
        row.end_reason = row.end_reason or f"invalid:{reason}"
        row.mode = DONE

    # ------------------------------------------------------------------

    def _flush_new_model_text(self, row: RowState) -> None:
        decoded = self.tokenizer.decode(row.generated_ids, skip_special_tokens=False)
        new_text = decoded[row._last_decoded_len:]
        row._last_decoded_len = len(decoded)
        if new_text:
            row.full_output += new_text

    def _row_tail(self, row: RowState) -> str:
        if len(row.text_ids_since_resume) < 2:
            return ""
        return self.tokenizer.decode(row.text_ids_since_resume[-50:], skip_special_tokens=False)

    def _check_tool_tags(self, tail: str) -> str | None:
        for stop in self.tool_stop_tags:
            if stop in tail:
                return stop
        return None

    def _latent_exit(self, row: RowState, readout_id: int) -> str | None:
        """Exit-anchor readout: the span closes when the readout of the
        current last-layer state is `</latent>` (after min_latent_steps), or at
        max_latent_steps."""
        if self.max_latent_steps > 0 and row.latent_steps >= self.max_latent_steps:
            return "max_steps"
        if row.latent_steps >= self.min_latent_steps and readout_id == self.latent_close_id:
            return "anchor"
        return None

    def _close_span_readout(self, row: RowState, exit_reason: str) -> None:
        if not self.record_latent_readout:
            return
        row.latent_spans.append(LatentSpanReadout(
            steps=len(row.readout_ids), exit=exit_reason,
            token_ids=list(row.readout_ids), pseudo_text=pseudo_text(row.readout_ids, self.tokenizer),
        ))
        row.readout_ids = []

    # ------------------------------------------------------------------
    # One global step over all rows
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_step(self, rng: torch.Generator | None) -> None:
        """Advance every DECODE/LATENT/FEED row by one slot; IDLE/DONE rows get PAD."""
        embed_w = self.embed_layer.weight
        inputs = torch.empty((self.B, 1, embed_w.shape[1]), dtype=embed_w.dtype, device=self.device)
        pad_embed = embed_w[self.pad_id]

        sample_rows_list: list[int] = []
        blocked_keys: dict[int, str] = {}
        real_rows: list[int] = []
        forced_tokens: dict[int, int] = {}
        latent_vec_rows: dict[int, torch.Tensor] = {}

        # Latent rows: next inputs R(h) from the current last-layer states and
        # the token readout of those states for the exit decision.
        latent_rows = [r.idx for r in self.rows if r.mode == LATENT]
        injected_all = None
        readout_ids = None
        span_readout = None
        if latent_rows:
            idx = torch.tensor(latent_rows, device=self.device)
            injected_all = self.model.run_hybridlink(self.last_hidden[idx], output_dtype=embed_w.dtype)
            readout_ids = self.next_logits.argmax(dim=-1)  # [B]
            if self.record_latent_readout:
                span_readout = nearest_token_ids(injected_all, self._readout_bank).tolist()

        for row in self.rows:
            i = row.idx
            if row.mode in (IDLE, DONE):
                inputs[i, 0] = pad_embed
                continue

            # budget gate for generating rows (observations are fed in full)
            if row.mode != FEED:
                if self._row_budget_left(row) <= 0:
                    row.extra["budget_truncated"] = True
                    row.end_reason = row.end_reason or "budget"
                    row.mode = DONE
                    inputs[i, 0] = pad_embed
                    continue
                if not self._sequence_cap_ok(row):
                    inputs[i, 0] = pad_embed
                    continue

            if row.mode == LATENT:
                row.generation_steps += 1
                pos_in_batch = latent_rows.index(i)
                injected = injected_all[pos_in_batch]
                exit_reason = self._latent_exit(row, int(readout_ids[i].item()))
                if exit_reason is not None:
                    row.generated_ids.append(self.latent_close_id)
                    row.kv_ledger.append(self.latent_close_id)
                    forced_tokens[i] = self.latent_close_id
                    real_rows.append(i)
                    inputs[i, 0] = embed_w[self.latent_close_id]
                    if exit_reason == "max_steps":
                        row.forced_exits += 1
                    else:
                        row.anchor_exits += 1
                    self._close_span_readout(row, exit_reason)
                    self._on_latent_span_closed(row)
                    row.mode = DECODE
                    row.latent_steps = 0
                else:
                    inputs[i, 0] = injected
                    real_rows.append(i)
                    latent_vec_rows[i] = injected
                    row.kv_ledger.append(None)
                    row.latent_steps += 1
                    row.latent_steps_total += 1
                    if span_readout is not None:
                        row.readout_ids.append(int(span_readout[pos_in_batch]))
                continue

            if row.mode == FEED:
                tok = row.feed_queue.pop(0)
                inputs[i, 0] = embed_w[tok]
                row.kv_ledger.append(tok)
                self._on_env_token_fed(row)
                forced_tokens[i] = tok
                real_rows.append(i)
                if not row.feed_queue:
                    # observation fed: next turn starts with the model's own <think>
                    row.in_think = False
                    row.text_ids_since_resume = []
                    if row.turn_count > self.max_turns:
                        row.end_reason = "max_turns"
                        row.mode = DONE
                    else:
                        row.mode = DECODE
                continue

            # DECODE: sample from this row's current logits
            row.generation_steps += 1
            sample_rows_list.append(i)
            blocked_keys[i] = "in" if row.in_think else "out"

        if sample_rows_list:
            blocked_matrix = torch.zeros_like(self.next_logits, dtype=torch.bool)
            for i in sample_rows_list:
                blocked_matrix[i] = self._blocked_masks[blocked_keys[i]]
            sampling_logits = self.next_logits
            if self.repetition_penalty != 1.0:
                # penalize every token already in the row's context (prompt, text, observations)
                sampling_logits = self.next_logits.clone()
                for i in sample_rows_list:
                    seen = {t for t in self.rows[i].kv_ledger if t is not None}
                    if not seen:
                        continue
                    idx = torch.tensor(sorted(seen), dtype=torch.long, device=sampling_logits.device)
                    vals = sampling_logits[i, idx]
                    sampling_logits[i, idx] = torch.where(
                        vals > 0, vals / self.repetition_penalty, vals * self.repetition_penalty,
                    )
            log_probs = token_log_probs(
                sampling_logits, self.temperature, self.top_p, self.top_k, blocked_matrix,
            )
            sampled = sample_rows(log_probs, sample_rows_list, rng)
            for i, tok in sampled.items():
                row = self.rows[i]
                row.generated_ids.append(tok)
                row.text_ids_since_resume.append(tok)
                row.kv_ledger.append(tok)
                self._record_text_action(row, tok, log_probs[i], blocked_keys[i])
                inputs[i, 0] = embed_w[tok]
                forced_tokens[i] = tok
                real_rows.append(i)

        for i, vec in latent_vec_rows.items():
            self._record_latent_vec(self.rows[i], vec)

        self._lockstep_forward(inputs, sorted(set(real_rows)))

        # events for rows that sampled a text token this step
        for i in sample_rows_list:
            row = self.rows[i]
            tok = row.generated_ids[-1]

            if tok in self.eos_ids:
                row.end_reason = "eos"
                row.mode = DONE
                continue
            if tok == self.think_open_id:
                row.in_think = True
            elif tok == self.think_close_id:
                row.in_think = False
            elif tok == self.latent_open_id:
                row.mode = LATENT
                row.latent_steps = 0
                row.latent_spans_opened += 1
                self._on_latent_span_opened(row)
                continue

            tail = self._row_tail(row)
            if self.reject_fabricated_result and RESULT_OPEN in tail:
                self._row_invalidate(row, "fabricated_result")
                continue

            stop = self._check_tool_tags(tail)
            if stop:
                tag = stop[2:-1]
                decoded = self.tokenizer.decode(row.generated_ids, skip_special_tokens=False)
                content = ToolExecutor.extract_content(decoded, tag)
                self._on_tool_request(row, tag, content)

    # ------------------------------------------------------------------
    # Tool barrier
    # ------------------------------------------------------------------

    def _on_tool_request(self, row: RowState, tag: str, content: str) -> None:
        """A row emitted a tool call: apply the caps, then pause it until the barrier."""
        if self.terminate_on_tool_cap:
            cap = None
            if self.max_tool_calls is not None and row.tool_requests >= self.max_tool_calls:
                cap = "shared"
            elif tag == "python" and self.max_python_calls is not None and row.python_calls >= self.max_python_calls:
                cap = "python"
            elif tag == "search" and self.max_search_calls is not None and row.search_calls >= self.max_search_calls:
                cap = "search"
            if cap is not None:
                row.end_reason = "tool_cap"
                row.mode = DONE
                return
            row.tool_requests += 1
        row.pending_tool = (tag, content)
        row.mode = IDLE

    def _execute_tool(self, row: RowState) -> str:
        """Run one row's pending tool call; cap notices are returned as observations."""
        tag, content = row.pending_tool
        if (self.max_tool_calls is not None
                and (row.python_calls + row.search_calls) >= self.max_tool_calls):
            return ("<result>The maximum tool call limit is exceeded. "
                    "You are not allowed to call any more tools.</result>")
        if tag == "python":
            if self.max_python_calls is None or row.python_calls < self.max_python_calls:
                row.python_calls += 1
                result = self.tool_executor.execute("python", content)
            else:
                result = ("The maximum python call limit is exceeded. "
                          "You are not allowed to use python.")
        elif tag == "search":
            if self.max_search_calls is None or row.search_calls < self.max_search_calls:
                if content not in row.search_query_history:
                    row.search_query_history.add(content)
                    row.search_calls += 1
                    result = self.tool_executor.execute("search", content)
                else:
                    result = ("You have searched this query. "
                              "Please refer to previous results.")
            else:
                result = ("The maximum search limit is exceeded. "
                          "You are not allowed to search.")
        else:
            result = ""
        return f"<result>{result}</result>" if result else "<result></result>"

    @torch.no_grad()
    def _tool_barrier(self) -> None:
        """Execute all pending tool calls concurrently; queue their observations."""
        pending = [r for r in self.rows if r.mode == IDLE and r.pending_tool]
        if not pending:
            return
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, self.tool_workers)) as pool:
            results = list(pool.map(self._execute_tool, pending))
        self.tool_time += time.time() - t0
        for row, result_text in zip(pending, results):
            row.pending_tool = None
            self._flush_new_model_text(row)
            row.full_output += result_text
            token_ids = self.tokenizer(result_text, add_special_tokens=False)["input_ids"]
            if not self._env_tokens_fit(row, len(token_ids)):
                continue
            row.feed_queue = list(token_ids)
            row.turn_count += 1
            if token_ids:
                row.mode = FEED
            else:
                row.in_think = False
                row.text_ids_since_resume = []
                if row.turn_count > self.max_turns:
                    row.end_reason = "max_turns"
                    row.mode = DONE
                else:
                    row.mode = DECODE

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, rng: torch.Generator | None = None) -> None:
        start = time.time()
        if not self._init_generation():
            self.total_time = time.time() - start
            return
        if self.max_turns <= 0:
            for row in self.rows:
                row.end_reason = "max_turns"
                row.mode = DONE
        while True:
            active = [r for r in self.rows if r.mode in (DECODE, LATENT, FEED)]
            if not active:
                pending = [r for r in self.rows if r.mode == IDLE]
                if not pending:
                    break
                self._tool_barrier()
                continue
            self._decode_step(rng)
        for row in self.rows:
            self._flush_new_model_text(row)
            if row.readout_ids:
                self._close_span_readout(row, row.end_reason or "budget")
        self.total_time = time.time() - start

    def row_result(self, row: RowState) -> dict[str, Any]:
        out = {
            "output": row.full_output,
            "turns": row.turn_count,
            "python_calls": row.python_calls,
            "search_calls": row.search_calls,
            "tool_requests": row.tool_requests,
            "generation_steps": row.generation_steps,
            "latent_steps": row.latent_steps_total,
            "latent_spans": row.latent_spans_opened,
            "anchor_exits": row.anchor_exits,
            "forced_exits": row.forced_exits,
            "end_reason": row.end_reason,
        }
        if self.record_latent_readout:
            out["latent_readouts"] = [span.to_dict() for span in row.latent_spans]
            out["output_rendered"] = render_with_readout(row.full_output, row.latent_spans)
        return out
