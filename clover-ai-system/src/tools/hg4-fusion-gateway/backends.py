"""Thread-safe serialized CPU runners for the four native C11 `HG4-16R` engines (+ deterministic mock mode).

Every stateful backend keeps `last_cached_ids` mirroring the native KV / recurrent cache and applies
Longest-Common-Prefix (LCP) state reuse:

* `lfm3-omni-2b` (`liblfm3_hg4_engine`): `lfm3_hg4_cache_truncate(lcp)` then feeds only `prompt_ids[lcp:]`.
* `clover-35b-a3b` / `nemotron-3.5-super-vl`: their Gated-DeltaNet / Mamba recurrent state cannot be rolled back,
  so the suffix is fed via `*_cache_feed` only when the new prompt strictly extends `last_cached_ids`
  (next OpenCode turn, appended tool result); otherwise `*_cache_reset()` + full prefill.

Prompts are encoded segment-by-segment (each segment starts at a special token, so concatenation is lossless) and
assistant turns produced by the gateway are spliced back with the exact token ids the model generated, so the
next request is a strict token-level extension of the cache => zero re-prefill of the system/repo prefix.

A single process-wide `ENGINE_LOCK` (`threading.RLock`) serializes all engine work so concurrent subagents
(Elephant + Goldfish) never thrash the 16-core OpenMP pool / DDR5 bandwidth.
"""
from __future__ import annotations

import collections
import ctypes
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

import tool_grammar as tg

HERE = Path(__file__).resolve().parent
SYSTEM_ROOT = HERE.parents[2]  # clover-ai-system/
STUDIO_CLOVER = SYSTEM_ROOT / "studio" / "clover"

ENGINE_LOCK = threading.RLock()

_IS_WIN = platform.system() == "Windows"
_SO = ".dll" if _IS_WIN else (".dylib" if platform.system() == "Darwin" else ".so")


# --------------------------------------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class GenParams:
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int = 0


@dataclass
class GenResult:
    model: str
    text: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefill_tokens: int = 0
    completion_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    backend_kind: str = "native"
    gen_ids: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_hit_pct(self) -> float:
        return 100.0 * self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


# --------------------------------------------------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------------------------------------------------

class HFTokenizer:
    """Rust `tokenizers` (fast) with fallback to the pure-Python `omni_c2_c3.Lfm3PureBpeTokenizer`."""

    def __init__(self, tok_json: Path, vocab_size: int | None = None):
        self.path = Path(tok_json)
        self._fast = None
        self._pure = None
        try:
            from tokenizers import Tokenizer
            self._fast = Tokenizer.from_file(str(self.path))
        except Exception:
            if str(STUDIO_CLOVER) not in sys.path:
                sys.path.append(str(STUDIO_CLOVER))
            import omni_c2_c3  # type: ignore[import-not-found]
            self._pure = omni_c2_c3.Lfm3PureBpeTokenizer(self.path, vocab_size=vocab_size)

    def token_to_id(self, s: str) -> int | None:
        if self._fast is not None:
            return self._fast.token_to_id(s)
        return self._pure.vocab.get(s)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._fast is not None:
            return list(self._fast.encode(text, add_special_tokens=False).ids)
        return self._pure.encode(text, allow_special=True)

    def decode(self, ids: list[int]) -> str:
        if self._fast is not None:
            return self._fast.decode(list(ids), skip_special_tokens=False)
        return self._pure.decode(list(ids), skip_special_tokens=False)


class TiktokenFallbackTokenizer:
    """Kimi `tiktoken.model` via `omni_c2_c3.PureTiktokenTokenizer`."""

    def __init__(self, model_path: Path):
        if str(STUDIO_CLOVER) not in sys.path:
            sys.path.append(str(STUDIO_CLOVER))
        import omni_c2_c3  # type: ignore[import-not-found]
        self._t = omni_c2_c3.PureTiktokenTokenizer(model_path)

    def token_to_id(self, s: str) -> int | None:
        return None

    def encode(self, text: str) -> list[int]:
        return self._t.encode(text, allow_special=True) if text else []

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(list(ids), skip_special_tokens=False)


MOCK_SPECIALS = (
    "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
)


class MockTokenizer:
    """Deterministic char-level tokenizer with atomic special tokens (prefix-stable => exact LCP semantics)."""

    vocab_size = 70000
    _OFF = 64

    def __init__(self):
        self.specials = {s: i + 1 for i, s in enumerate(MOCK_SPECIALS)}
        self.inv = {v: k for k, v in self.specials.items()}
        self._re = re.compile("(" + "|".join(re.escape(s) for s in sorted(MOCK_SPECIALS, key=len, reverse=True)) + ")")
        self.unk = self._OFF + ord("?")

    def token_to_id(self, s: str) -> int | None:
        return self.specials.get(s)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in self._re.split(text or ""):
            if not part:
                continue
            if part in self.specials:
                out.append(self.specials[part])
            else:
                for ch in part:
                    o = ord(ch) + self._OFF
                    out.append(o if o < self.vocab_size else self.unk)
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] if i in self.inv else chr(i - self._OFF) for i in ids if i >= self._OFF or i in self.inv)


# --------------------------------------------------------------------------------------------------------------------
# Chat templates (segment-based)
# --------------------------------------------------------------------------------------------------------------------

def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") in ("text", "input_text"):
                    parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or "")
    return str(content)


def _tool_call_args(tc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    name = str(fn.get("name") or "")
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = tg.repair_json(a) if a.strip() else {}
        except tg.JsonRepairError:
            a = {}
    return name, a if isinstance(a, dict) else {}


def assistant_signature(content: Any, tool_calls: Any) -> str:
    """Canonical key of an assistant message (ignores tool-call ids / whitespace) for exact-token splicing."""
    calls = []
    for tc in tool_calls or ():
        if isinstance(tc, dict):
            calls.append(list(_tool_call_args(tc)))
    blob = json.dumps({"c": _content_text(content).strip(), "t": calls}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _render_arg_val(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


@dataclass
class ChatTemplate:
    name: str
    bos: str
    sys_open: str
    user_open: str
    asst_open: str  # generation prompt (also prefix of every assistant history turn)
    tool_open: str
    end: str
    tool_style: str = "json"
    wrap_tool_response: bool = True
    stop_markers: tuple[str, ...] = ()

    def render_tool_calls(self, tool_calls: Any) -> str:
        out = []
        for tc in tool_calls or ():
            if not isinstance(tc, dict):
                continue
            name, args = _tool_call_args(tc)
            if self.tool_style == "xml":
                body = "".join(f"<parameter={k}>\n{_render_arg_val(v)}\n</parameter>\n" for k, v in args.items())
                out.append(f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>")
            else:
                out.append("<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                           + "\n</tool_call>")
        return "\n".join(out)

    def segments(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tool_choice: Any,
    ) -> list[tuple[str, str | None]]:
        """Return `[(text, splice_key_or_None), ...]` for the full prompt incl. the generation prompt."""
        msgs = [m for m in (messages or []) if isinstance(m, dict)]
        sys_parts: list[str] = []
        while msgs and msgs[0].get("role") in ("system", "developer"):
            t = _content_text(msgs[0].get("content"))
            if t:
                sys_parts.append(t)
            msgs = msgs[1:]
        tool_prompt = tg.render_tool_system_prompt(tools, tool_choice, style=self.tool_style)
        if tool_prompt:
            sys_parts.append(tool_prompt)
        segs: list[tuple[str, str | None]] = []
        head = self.bos
        if sys_parts:
            head += self.sys_open + "\n\n".join(sys_parts) + self.end
        if head:
            segs.append((head, None))
        for m in msgs:
            role = str(m.get("role") or "user")
            if role == "assistant":
                body = _content_text(m.get("content"))
                if "</think>" in body:
                    body = body.split("</think>")[-1].lstrip()
                calls = self.render_tool_calls(m.get("tool_calls"))
                if calls:
                    body = (body.rstrip() + "\n" if body.strip() else "") + calls
                segs.append((self.asst_open + body + self.end,
                             assistant_signature(m.get("content"), m.get("tool_calls"))))
            elif role in ("tool", "function"):
                c = _content_text(m.get("content"))
                if self.wrap_tool_response and "<tool_response>" not in c:
                    c = f"<tool_response>\n{c}\n</tool_response>"
                segs.append((self.tool_open + c + self.end, None))
            elif role in ("system", "developer"):
                segs.append((self.sys_open + _content_text(m.get("content")) + self.end, None))
            else:
                segs.append((self.user_open + _content_text(m.get("content")) + self.end, None))
        segs.append((self.asst_open, None))
        return segs


TEMPLATES: dict[str, ChatTemplate] = {
    "lfm3": ChatTemplate(
        name="lfm3", bos="<|startoftext|>", sys_open="<|im_start|>system\n", user_open="<|im_start|>user\n",
        asst_open="<|im_start|>assistant\n<think></think>", tool_open="<|im_start|>tool\n", end="<|im_end|>\n",
        tool_style="xml", stop_markers=("<|im_end|>", "<|endoftext|>", "<tool_response>"),
    ),
    "qwen-chatml": ChatTemplate(
        name="qwen-chatml", bos="", sys_open="<|im_start|>system\n", user_open="<|im_start|>user\n",
        asst_open="<|im_start|>assistant\n<think>\n\n</think>\n\n", tool_open="<|im_start|>user\n",
        end="<|im_end|>\n", tool_style="xml", stop_markers=("<|im_end|>", "<|endoftext|>", "<tool_response>"),
    ),
    "nemotron-chatml": ChatTemplate(
        name="nemotron-chatml", bos="", sys_open="<|im_start|>system\n", user_open="<|im_start|>user\n",
        asst_open="<|im_start|>assistant\n<think></think>", tool_open="<|im_start|>user\n", end="<|im_end|>\n",
        tool_style="json", stop_markers=("<|im_end|>", "<|endoftext|>", "<tool_response>"),
    ),
    "kimi": ChatTemplate(
        name="kimi", bos="", sys_open="<|im_system|>system<|im_middle|>", user_open="<|im_user|>user<|im_middle|>",
        asst_open="<|im_assistant|>assistant<|im_middle|>", tool_open="<|im_system|>tool<|im_middle|>",
        end="<|im_end|>", tool_style="json", stop_markers=("<|im_end|>", "[EOS]"),
    ),
}


# --------------------------------------------------------------------------------------------------------------------
# Native engine adapters
# --------------------------------------------------------------------------------------------------------------------

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_flt_p = ctypes.POINTER(ctypes.c_float)
_c_dbl_p = ctypes.POINTER(ctypes.c_double)


def _load_cdll(path: Path) -> ctypes.CDLL:
    if _IS_WIN and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
    return ctypes.CDLL(str(path))


class EngineAdapter:
    """Minimal stateful engine contract: `reset`, optional `truncate`, `feed(ids) -> last-token logits`."""

    name = "engine"
    vocab = 0
    hidden = 0
    max_seq = 16384
    supports_truncate = False

    def reset(self) -> None:
        raise NotImplementedError

    def truncate(self, n: int) -> None:
        raise NotImplementedError

    def feed(self, ids: list[int]) -> np.ndarray:
        raise NotImplementedError


class _FeedEngine(EngineAdapter):
    def __init__(self, vocab: int, hidden: int):
        self.vocab = vocab
        self.hidden = hidden
        self._logits = np.zeros(vocab, dtype=np.float32)
        self._hidden = np.zeros(hidden, dtype=np.float32)

    def _call_feed(self, fn: Any, ids: list[int]) -> np.ndarray:
        arr = (ctypes.c_int * len(ids))(*ids)
        rc = fn(arr, len(ids), self._hidden.ctypes.data_as(_c_flt_p), self._logits.ctypes.data_as(_c_flt_p))
        if rc < 0:
            raise RuntimeError(f"{self.name} cache_feed failed rc={rc}")
        return self._logits.copy()


class Lfm3Engine(_FeedEngine):
    name = "liblfm3_hg4_engine"
    supports_truncate = True

    def __init__(self, lib_path: Path, hg4_dir: Path, max_seq: int = 16384, threads: int | None = None):
        super().__init__(128000, 2048)
        lib = _load_cdll(lib_path)
        lib.lfm3_hg4_init.argtypes = [ctypes.c_char_p]
        lib.lfm3_hg4_init.restype = ctypes.c_int
        lib.lfm3_hg4_cache_init.argtypes = [ctypes.c_int]
        lib.lfm3_hg4_cache_init.restype = ctypes.c_int
        lib.lfm3_hg4_cache_truncate.argtypes = [ctypes.c_int]
        lib.lfm3_hg4_cache_truncate.restype = None
        lib.lfm3_hg4_cache_get_len.argtypes = []
        lib.lfm3_hg4_cache_get_len.restype = ctypes.c_int
        lib.lfm3_hg4_cache_feed.argtypes = [_c_int_p, ctypes.c_int, _c_flt_p, _c_flt_p]
        lib.lfm3_hg4_cache_feed.restype = ctypes.c_int
        if hasattr(lib, "lfm3_hg4_set_num_threads"):
            lib.lfm3_hg4_set_num_threads.argtypes = [ctypes.c_int]
            lib.lfm3_hg4_set_num_threads.restype = None
            lib.lfm3_hg4_set_num_threads(int(threads or max(2, min(16, os.cpu_count() or 8))))
        rc = lib.lfm3_hg4_init(str(hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"lfm3_hg4_init({hg4_dir}) rc={rc}")
        rc = lib.lfm3_hg4_cache_init(int(max_seq))
        if rc != 0:
            raise RuntimeError(f"lfm3_hg4_cache_init({max_seq}) rc={rc}")
        self.lib = lib
        self.max_seq = int(max_seq)

    def reset(self) -> None:
        self.lib.lfm3_hg4_cache_truncate(0)

    def truncate(self, n: int) -> None:
        self.lib.lfm3_hg4_cache_truncate(int(n))

    def cache_len(self) -> int:
        return int(self.lib.lfm3_hg4_cache_get_len())

    def feed(self, ids: list[int]) -> np.ndarray:
        return self._call_feed(self.lib.lfm3_hg4_cache_feed, ids)


class Clover35bEngine(_FeedEngine):
    name = "libclover35b_hg4_engine"

    def __init__(self, lib_path: Path, hg4_dir: Path, max_seq: int = 32768):
        super().__init__(248320, 2048)
        lib = _load_cdll(lib_path)
        lib.clover35b_hg4_init.argtypes = [ctypes.c_char_p]
        lib.clover35b_hg4_init.restype = ctypes.c_int
        lib.clover35b_hg4_cache_reset.argtypes = []
        lib.clover35b_hg4_cache_reset.restype = None
        lib.clover35b_hg4_cache_feed.argtypes = [_c_int_p, ctypes.c_int, _c_flt_p, _c_flt_p]
        lib.clover35b_hg4_cache_feed.restype = ctypes.c_int
        lib.clover35b_hg4_generate.argtypes = [_c_int_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               _c_int_p, _c_dbl_p, _c_dbl_p]
        lib.clover35b_hg4_generate.restype = ctypes.c_int
        rc = lib.clover35b_hg4_init(str(hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"clover35b_hg4_init({hg4_dir}) rc={rc}")
        self.lib = lib
        self.max_seq = int(max_seq)

    def reset(self) -> None:
        self.lib.clover35b_hg4_cache_reset()

    def feed(self, ids: list[int]) -> np.ndarray:
        return self._call_feed(self.lib.clover35b_hg4_cache_feed, ids)


class NemotronEngine(_FeedEngine):
    name = "libnemotron_hg4_engine"

    def __init__(self, lib_path: Path, hg4_dir: Path, max_seq: int = 32768):
        super().__init__(131072, 4096)
        lib = _load_cdll(lib_path)
        lib.nemo_engine_init.argtypes = [ctypes.c_char_p]
        lib.nemo_engine_init.restype = ctypes.c_int
        lib.nemo_cache_reset.argtypes = []
        lib.nemo_cache_reset.restype = None
        lib.nemo_cache_feed.argtypes = [_c_int_p, ctypes.c_int, _c_flt_p, _c_flt_p]
        lib.nemo_cache_feed.restype = ctypes.c_int
        lib.nemo_generate_speculative.argtypes = [
            _c_int_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _c_int_p, _c_dbl_p, _c_dbl_p, _c_int_p, _c_int_p,
        ]
        lib.nemo_generate_speculative.restype = ctypes.c_int
        rc = lib.nemo_engine_init(str(hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"nemo_engine_init({hg4_dir}) rc={rc}")
        self.lib = lib
        self.max_seq = int(max_seq)

    def reset(self) -> None:
        self.lib.nemo_cache_reset()

    def feed(self, ids: list[int]) -> np.ndarray:
        return self._call_feed(self.lib.nemo_cache_feed, ids)

    def generate_speculative(self, prompt_ids: list[int], max_new: int, eos_id: int, draft_k: int = 2):
        """Stateless 2L-MTP lossless speculative greedy decode (resets the cache)."""
        arr = (ctypes.c_int * len(prompt_ids))(*prompt_ids)
        out = (ctypes.c_int * (max_new + 8))()
        pf, dc = ctypes.c_double(0), ctypes.c_double(0)
        acc, tot = ctypes.c_int(0), ctypes.c_int(0)
        n = self.lib.nemo_generate_speculative(arr, len(prompt_ids), int(max_new), int(eos_id), int(draft_k), out,
                                               ctypes.byref(pf), ctypes.byref(dc), ctypes.byref(acc), ctypes.byref(tot))
        if n < 0:
            raise RuntimeError(f"nemo_generate_speculative rc={n}")
        return list(out[:n]), pf.value, dc.value, acc.value, tot.value


# --------------------------------------------------------------------------------------------------------------------
# Mock engine (deterministic offline testing)
# --------------------------------------------------------------------------------------------------------------------

Responder = Callable[[str, str], str]


def _last_block(prompt_text: str) -> tuple[str, str]:
    """Return (role, text) of the last message before the generation prompt in a ChatML-like prompt."""
    body = prompt_text
    idx = body.rfind("<|im_start|>assistant")
    if idx >= 0:
        body = body[:idx]
    blocks = body.split("<|im_start|>")
    for blk in reversed(blocks):
        blk = blk.replace("<|im_end|>", "").strip()
        if not blk:
            continue
        role, _, text = blk.partition("\n")
        return role.strip(), text.strip()
    return "", ""


def default_mock_responder(model_id: str, prompt_text: str) -> str:
    """Deterministic scripted behaviour used by the offline test-suite and `--mock` mode."""
    role, text = _last_block(prompt_text)
    sys_text = prompt_text.split("<|im_end|>")[0]
    if "CLOVER_AUDITOR" in sys_text:
        return "FLAW: off-by-one in the loop bound (range(n) skips n)." if "BUG" in prompt_text else "NO_ISSUES"
    if "CLOVER_REVISE" in prompt_text and role == "user":
        return f"[{model_id}] revised answer with the flagged flaw fixed."
    if "MOCK_TOOL_RAW:" in text:
        return text.split("MOCK_TOOL_RAW:", 1)[1].strip()
    if "MOCK_LONG" in text:
        return " ".join(f"w{i}" for i in range(1500))
    if "Your previous tool call was invalid" in text:
        if "MOCK_BAD_TOOL_STUBBORN" in prompt_text and not model_id.startswith("nemotron"):
            return '<tool_call>\n{"name": "bash", "arguments": {"timeout": 5}}\n</tool_call>'
        return '<tool_call>\n{"name": "bash", "arguments": {"command": "lake build"}}\n</tool_call>'
    if "MOCK_BAD_TOOL" in text:
        if model_id.startswith("nemotron"):
            return '<tool_call>\n{"name": "bash", "arguments": {"command": "lake build"}}\n</tool_call>'
        return '<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call>'
    if "<tool_response>" in text:
        snippet = text.replace("<tool_response>", "").replace("</tool_response>", "").strip()[:60]
        return f"[{model_id}] observed tool result: {snippet}"
    if role == "user" and "MOCK_CALL_BASH" in text:
        return '<tool_call>\n{"name": "bash", "arguments": {"command": "lake build"}}\n</tool_call>'
    return f"[{model_id}] {text[:80]}"


class MockEngine(EngineAdapter):
    """Char-level scripted engine with a real cache model (counts every fed token)."""

    def __init__(self, model_id: str, tokenizer: MockTokenizer, template: ChatTemplate, eos_id: int,
                 responder: Responder | None = None, supports_truncate: bool = False, max_seq: int = 1 << 20):
        self.name = f"mock:{model_id}"
        self.model_id = model_id
        self.tok = tokenizer
        self.template = template
        self.eos_id = eos_id
        self.responder = responder or default_mock_responder
        self.supports_truncate = supports_truncate
        self.vocab = tokenizer.vocab_size
        self.max_seq = max_seq
        self.state: list[int] = []
        self.total_fed = 0
        self.feed_calls = 0
        self.resets = 0
        self._plans: dict[str, list[int]] = {}

    def reset(self) -> None:
        self.state = []
        self.resets += 1

    def truncate(self, n: int) -> None:
        del self.state[n:]

    def feed(self, ids: list[int]) -> np.ndarray:
        self.state.extend(ids)
        self.total_fed += len(ids)
        self.feed_calls += 1
        marker = self.tok.encode(self.template.asst_open)
        m = len(marker)
        pos = -1
        for k in range(len(self.state) - m, -1, -1):
            if self.state[k:k + m] == marker:
                pos = k
                break
        logits = np.full(self.vocab, -10.0, dtype=np.float32)
        if pos < 0:
            logits[self.eos_id] = 10.0
            return logits
        prompt_ids = self.state[:pos + m]
        key = hashlib.sha1(np.asarray(prompt_ids, dtype=np.int64).tobytes()).hexdigest()
        plan = self._plans.get(key)
        if plan is None:
            plan = self.tok.encode(self.responder(self.model_id, self.tok.decode(prompt_ids))) + [self.eos_id]
            self._plans[key] = plan
        k = len(self.state) - (pos + m)
        nxt = plan[k] if k < len(plan) else self.eos_id
        logits[nxt] = 10.0
        return logits


# --------------------------------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------------------------------

def _sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    lg = logits.astype(np.float64) / max(temperature, 1e-4)
    lg -= np.max(lg)
    p = np.exp(lg)
    p /= p.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


class BackendBase:
    model_id = ""
    kind = "native"
    family = ""

    def __init__(self) -> None:
        self.stats = collections.Counter()

    def describe(self) -> dict[str, Any]:
        return {"id": self.model_id, "kind": self.kind, "family": self.family}

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tool_choice: Any,
               params: GenParams) -> Iterator[tuple[str, Any]]:
        raise NotImplementedError

    def generate(self, messages, tools, tool_choice, params: GenParams) -> GenResult:
        res: GenResult | None = None
        for ev, payload in self.stream(messages, tools, tool_choice, params):
            if ev == "done":
                res = payload
        assert res is not None
        return res

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int]) -> None:
        pass

    def _account(self, r: GenResult) -> None:
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += r.prompt_tokens
        self.stats["cached_tokens"] += r.cached_tokens
        self.stats["prefill_tokens"] += r.prefill_tokens
        self.stats["completion_tokens"] += r.completion_tokens
        self.stats["prefill_ms"] += r.prefill_ms
        self.stats["decode_ms"] += r.decode_ms


class StatefulBackend(BackendBase):
    """LCP prefix-cached incremental decoder over an `EngineAdapter`."""

    def __init__(self, model_id: str, engine: EngineAdapter, tokenizer: Any, template: ChatTemplate,
                 eos_ids: set[int], family: str, kind: str = "native", splice_capacity: int = 256):
        super().__init__()
        self.model_id = model_id
        self.engine = engine
        self.tok = tokenizer
        self.template = template
        self.eos_ids = {int(e) for e in eos_ids if e is not None and int(e) >= 0}
        self.family = family
        self.kind = kind
        self.last_cached_ids: list[int] = []
        self._last_logits: np.ndarray | None = None
        self._seg_cache: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
        self._splice: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
        self._splice_cap = splice_capacity

    # -- prompt construction ---------------------------------------------------------------------------------------
    def _encode_seg(self, text: str) -> list[int]:
        ids = self._seg_cache.get(text)
        if ids is None:
            ids = self.tok.encode(text)
            self._seg_cache[text] = ids
            if len(self._seg_cache) > 4096:
                self._seg_cache.popitem(last=False)
        else:
            self._seg_cache.move_to_end(text)
        return ids

    def build_prompt_ids(self, messages, tools, tool_choice) -> list[int]:
        ids: list[int] = []
        asst_ids = self._encode_seg(self.template.asst_open)
        end_ids = self._encode_seg(self.template.end)
        for text, key in self.template.segments(messages, tools, tool_choice):
            if key is not None and key in self._splice:
                ids.extend(asst_ids)
                ids.extend(self._splice[key])
                ids.extend(end_ids)
            else:
                ids.extend(self._encode_seg(text))
        return ids

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int]) -> None:
        if not gen_ids:
            return
        key = assistant_signature(content, tool_calls)
        with ENGINE_LOCK:
            self._splice[key] = list(gen_ids)
            self._splice.move_to_end(key)
            while len(self._splice) > self._splice_cap:
                self._splice.popitem(last=False)

    def _fit_context(self, messages, tools, tool_choice, max_new: int) -> list[int]:
        msgs = list(messages or [])
        while True:
            ids = self.build_prompt_ids(msgs, tools, tool_choice)
            budget = self.engine.max_seq - min(max_new, 1024) - 8
            if len(ids) <= budget:
                return ids
            n_sys = 0
            while n_sys < len(msgs) and msgs[n_sys].get("role") in ("system", "developer"):
                n_sys += 1
            if len(msgs) - n_sys <= 1:
                return ids[-budget:]
            del msgs[n_sys]

    # -- LCP prefill ------------------------------------------------------------------------------------------------
    def reset_cache(self) -> None:
        with ENGINE_LOCK:
            self.engine.reset()
            self.last_cached_ids = []
            self._last_logits = None

    def _prefill(self, ids: list[int]) -> tuple[np.ndarray, int, int]:
        cached = self.last_cached_ids
        if cached and cached == ids and self._last_logits is not None:
            return self._last_logits.copy(), len(ids), 0
        lcp = 0
        lim = min(len(cached), len(ids) - 1)
        while lcp < lim and cached[lcp] == ids[lcp]:
            lcp += 1
        if self.engine.supports_truncate:
            if lcp < len(cached):
                self.engine.truncate(lcp)
                del cached[lcp:]
        elif lcp < len(cached) or not cached:
            # recurrent GDN / Mamba state cannot roll back: only strict extensions reuse the cache
            self.engine.reset()
            cached.clear()
            lcp = 0
        suffix = ids[lcp:]
        logits = self.engine.feed(suffix)
        cached.extend(suffix)
        self._last_logits = logits
        return logits, lcp, len(suffix)

    def stream(self, messages, tools, tool_choice, params: GenParams) -> Iterator[tuple[str, Any]]:
        with ENGINE_LOCK:
            ids = self._fit_context(messages, tools, tool_choice, params.max_tokens)
            max_new = max(1, min(int(params.max_tokens), self.engine.max_seq - len(ids) - 1))
            t0 = time.perf_counter()
            try:
                logits, lcp, n_fed = self._prefill(ids)
            except Exception:
                self.last_cached_ids = []
                self._last_logits = None
                raise
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            res = GenResult(model=self.model_id, prompt_tokens=len(ids), cached_tokens=lcp, prefill_tokens=n_fed,
                            prefill_ms=prefill_ms, backend_kind=self.kind)
            rng = np.random.default_rng(int(params.seed or 0))
            stops = [s for s in (params.stop or []) if s]
            markers = list(self.template.stop_markers)
            hold = max([len(s) for s in stops + markers] + [1]) - 1
            gen: list[int] = []
            text = ""
            emitted = 0
            finish = "length"
            splice_ok = True
            t1 = time.perf_counter()
            for step in range(max_new):
                tok = _sample(logits, params.temperature, params.top_p, rng)
                if tok in self.eos_ids:
                    finish = "stop"
                    break
                gen.append(tok)
                text = self.tok.decode(gen)
                cut = -1
                for s in markers + stops:
                    k = text.find(s, max(0, emitted - len(s)))
                    if k >= 0 and (cut < 0 or k < cut):
                        cut = k
                if cut >= 0:
                    if cut > emitted:
                        yield "delta", text[emitted:cut]
                    text = text[:cut]
                    emitted = cut
                    finish = "stop"
                    splice_ok = False
                    break
                safe = len(text) - hold
                if text.endswith("\ufffd"):
                    safe = min(safe, len(text) - 1)
                if safe > emitted:
                    yield "delta", text[emitted:safe]
                    emitted = safe
                if step + 1 < max_new:
                    logits = self.engine.feed([tok])
                    self.last_cached_ids.append(tok)
                    self._last_logits = logits
            if len(text) > emitted:
                tail = text[emitted:].replace("\ufffd", "")
                if tail:
                    yield "delta", tail
            res.text = text
            res.finish_reason = finish
            res.completion_tokens = len(gen)
            res.decode_ms = (time.perf_counter() - t1) * 1000.0
            res.gen_ids = list(gen) if splice_ok else []
            self._account(res)
            yield "done", res


class NemotronBackend(StatefulBackend):
    """Nemotron with optional stateless 2L-MTP speculative path (`CLOVER_NEMO_SPECULATIVE=1`, greedy, no stops)."""

    def stream(self, messages, tools, tool_choice, params: GenParams):
        eng = self.engine
        use_spec = (
            isinstance(eng, NemotronEngine) and os.environ.get("CLOVER_NEMO_SPECULATIVE") == "1"
            and params.temperature <= 0.0 and not params.stop and self.eos_ids
        )
        if not use_spec:
            yield from super().stream(messages, tools, tool_choice, params)
            return
        with ENGINE_LOCK:
            ids = self._fit_context(messages, tools, tool_choice, params.max_tokens)
            eos = min(self.eos_ids)
            draft_k = 2
            spec_max_new = max(1, min(int(params.max_tokens), eng.max_seq - len(ids) - draft_k - 3))
            gen, pf, dc, acc, tot = eng.generate_speculative(ids, spec_max_new, eos, draft_k)
            self.last_cached_ids = []  # speculative path resets the native cache
            self._last_logits = None
            gen = [g for g in gen if g not in self.eos_ids]
            text = self.tok.decode(gen)
            for mk in self.template.stop_markers:
                if mk in text:
                    text = text.split(mk)[0]
            res = GenResult(model=self.model_id, text=text, finish_reason="stop", prompt_tokens=len(ids),
                            prefill_tokens=len(ids), completion_tokens=len(gen), prefill_ms=pf, decode_ms=dc,
                            backend_kind=self.kind, extra={"mtp_accepted": acc, "mtp_drafted": tot})
            if text:
                yield "delta", text
            self._account(res)
            yield "done", res


class KimiBackend(BackendBase):
    """Kimi K3 / K3.1 Pro: proxies to `clover-k3-chat.service` (`/api/chat`) or the native `libkimi_k31_hg4_engine`."""

    family = "kimi"

    def __init__(self, model_id: str, url: str, hg4_dir: Path | None = None, lib_path: Path | None = None,
                 timeout_s: float = 900.0):
        super().__init__()
        self.model_id = model_id
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.hg4_dir = hg4_dir
        self.lib_path = lib_path
        self._lib = None
        self._tok = None
        self._n_layers = 0
        self.kind = "proxy"
        self.template = TEMPLATES["kimi"]

    @staticmethod
    def reachable(url: str, timeout: float = 0.5) -> bool:
        base = url.rstrip("/")
        for path in ("/api/health", "/health", "/"):
            try:
                with urllib.request.urlopen(base + path, timeout=timeout) as r:
                    if r.status < 500:
                        return True
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return True
            except Exception:
                continue
        return False

    def native_available(self) -> bool:
        return bool(self.lib_path and self.lib_path.is_file() and self.hg4_dir and (self.hg4_dir / "meta.bin").is_file())

    def _transcript(self, messages, tools, tool_choice) -> tuple[str, list[dict[str, str]]]:
        segs = self.template.segments(messages, tools, tool_choice)
        flat = "".join(t for t, _ in segs)
        hist = []
        for m in messages or []:
            if m.get("role") in ("user", "assistant"):
                hist.append({"role": m["role"], "content": _content_text(m.get("content"))})
        return flat, hist

    def _proxy(self, messages, tools, tool_choice, params: GenParams) -> str:
        sys_prompt = tg.render_tool_system_prompt(tools, tool_choice, style="json")
        last_user = ""
        for m in reversed(messages or []):
            if m.get("role") in ("user", "tool"):
                last_user = _content_text(m.get("content"))
                break
        history = []
        turns = list(messages or [])
        pending_user = None
        for m in turns[:-1]:
            r = m.get("role")
            if r == "user":
                pending_user = _content_text(m.get("content"))
            elif r == "assistant" and pending_user is not None:
                history.append({"user": pending_user, "reply": _content_text(m.get("content"))})
                pending_user = None
        system = "\n\n".join(x for x in [*(_content_text(m.get("content")) for m in turns
                                           if m.get("role") in ("system", "developer")), sys_prompt] if x)
        body = {
            "session_id": f"hg4-fusion-{uuid.uuid4().hex[:12]}", "prompt": last_user, "history": history,
            "system": system, "messages": messages, "max_tokens": params.max_tokens,
            "temperature": params.temperature, "remember": False, "stream": False,
        }
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        for k in ("reply", "response", "text", "content", "answer"):
            if isinstance(data.get(k), str):
                return data[k]
        ch = data.get("choices")
        if isinstance(ch, list) and ch:
            msg = ch[0].get("message") or {}
            return str(msg.get("content") or ch[0].get("text") or "")
        return ""

    def _ensure_native(self) -> None:
        if self._lib is not None:
            return
        assert self.lib_path is not None and self.hg4_dir is not None
        lib = _load_cdll(self.lib_path)
        lib.kimi_k31_hg4_init.argtypes = [ctypes.c_char_p]
        lib.kimi_k31_hg4_init.restype = ctypes.c_int
        lib.kimi_k31_hg4_num_layers.argtypes = []
        lib.kimi_k31_hg4_num_layers.restype = ctypes.c_int
        lib.kimi_k31_generate_speculative.argtypes = [
            _c_int_p, ctypes.c_int, _c_int_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _c_int_p, _c_dbl_p, _c_dbl_p, _c_int_p, _c_int_p,
        ]
        lib.kimi_k31_generate_speculative.restype = ctypes.c_int
        rc = lib.kimi_k31_hg4_init(str(self.hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"kimi_k31_hg4_init rc={rc}")
        self._n_layers = int(lib.kimi_k31_hg4_num_layers())
        tj = self.hg4_dir / "tokenizer.json"
        self._tok = HFTokenizer(tj) if tj.is_file() else TiktokenFallbackTokenizer(self.hg4_dir / "tiktoken.model")
        self._lib = lib

    def _native(self, messages, tools, tool_choice, params: GenParams) -> tuple[str, GenResult]:
        self._ensure_native()
        flat, _ = self._transcript(messages, tools, tool_choice)
        ids = self._tok.encode(flat)
        eos = self._tok.token_to_id("<|im_end|>") or 163586
        layers = (ctypes.c_int * self._n_layers)(*range(self._n_layers))
        arr = (ctypes.c_int * len(ids))(*ids)
        out = (ctypes.c_int * (params.max_tokens + 8))()
        pf, dc = ctypes.c_double(0), ctypes.c_double(0)
        acc, tot = ctypes.c_int(0), ctypes.c_int(0)
        n = self._lib.kimi_k31_generate_speculative(arr, len(ids), layers, self._n_layers, params.max_tokens, eos, 2,
                                                    out, ctypes.byref(pf), ctypes.byref(dc), ctypes.byref(acc),
                                                    ctypes.byref(tot))
        if n < 0:
            raise RuntimeError(f"kimi_k31_generate_speculative rc={n}")
        gen = [g for g in out[:n] if g != eos]
        text = self._tok.decode(gen)
        for mk in self.template.stop_markers:
            text = text.split(mk)[0]
        return text, GenResult(model=self.model_id, prompt_tokens=len(ids), prefill_tokens=len(ids),
                               completion_tokens=len(gen), prefill_ms=pf.value, decode_ms=dc.value,
                               backend_kind="native", extra={"dspark_accepted": acc.value, "dspark_drafted": tot.value})

    def stream(self, messages, tools, tool_choice, params: GenParams):
        with ENGINE_LOCK:
            t0 = time.perf_counter()
            res: GenResult
            if self.reachable(self.url):
                text = self._proxy(messages, tools, tool_choice, params)
                res = GenResult(model=self.model_id, backend_kind="proxy",
                                prompt_tokens=sum(len(_content_text(m.get("content"))) // 4 for m in messages or []),
                                completion_tokens=max(1, len(text) // 4) if text else 0,
                                decode_ms=(time.perf_counter() - t0) * 1000.0)
            elif self.native_available():
                text, res = self._native(messages, tools, tool_choice, params)
            else:
                raise RuntimeError(f"kimi backend unavailable: {self.url} unreachable and no native engine")
            for s in params.stop or []:
                if s and s in text:
                    text = text.split(s)[0]
            res.text = text
            res.finish_reason = "stop"
            if text:
                yield "delta", text
            self._account(res)
            yield "done", res


# --------------------------------------------------------------------------------------------------------------------
# Registry / discovery
# --------------------------------------------------------------------------------------------------------------------

DIRECT_MODELS = ("lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3")
MODEL_ALIASES = {"kimi-k3.1-pro": "kimi-k3", "lfm3-omni": "lfm3-omni-2b", "clover-35b": "clover-35b-a3b",
                 "nemotron-120b": "nemotron-3.5-super-vl", "nemotron": "nemotron-3.5-super-vl"}

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "lfm3-omni-2b": {"template": "lfm3", "family": "lfm3", "truncate": True, "eos": ("<|im_end|>", "<|endoftext|>")},
    "clover-35b-a3b": {"template": "qwen-chatml", "family": "clover35b", "truncate": False,
                       "eos": ("<|im_end|>", "<|endoftext|>"), "eos_fallback": 248046},
    "nemotron-3.5-super-vl": {"template": "nemotron-chatml", "family": "nemotron", "truncate": False,
                              "eos": ("<|im_end|>", "</s>", "<|endoftext|>")},
    "kimi-k3": {"template": "nemotron-chatml", "family": "kimi", "truncate": False, "eos": ("<|im_end|>",)},
}


def _first_dir(cands: list[Path], need: str = "meta.bin") -> Path | None:
    for c in cands:
        if c and (c / need).is_file():
            return c
    return None


def _first_file(cands: list[Path]) -> Path | None:
    for c in cands:
        if c and c.is_file():
            return c
    return None


def _env_path(name: str) -> list[Path]:
    v = (os.environ.get(name) or "").strip()
    return [Path(v)] if v else []


def discover_paths() -> dict[str, dict[str, Path | None]]:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    lib_dirs = [Path(d) for d in (os.environ.get("CLOVER_NATIVE_LIB_DIR") or "").split(os.pathsep) if d.strip()]

    def lib(name: str, dirs: list[Path]) -> Path | None:
        fn = f"lib{name}{_SO}"
        return _first_file([d / fn for d in lib_dirs + dirs])

    omni_dir = _first_dir(_env_path("CLOVER_OMNI_HG4_DIR") + [
        home / "CloverAI" / "data" / "omni" / "hg4", Path("/opt/clover-k3/lfm3-omni-hg4"),
        Path("/opt/clover-k3/web-studio/data/omni/hg4")])
    c35_dir = _first_dir(_env_path("CLOVER35B_HG4_DIR") + [Path("/opt/clover-k3/clover35b-full-hg4")])
    nemo_dir = _first_dir(_env_path("CLOVER_NEMO_HG4_DIR") + [Path("/opt/clover-k3/nemotron-full-hg4")])
    kimi_dir = _first_dir(_env_path("CLOVER_KIMI_HG4_DIR") + [Path("/opt/clover-k3/kimi-k31-hg4")])
    return {
        "lfm3-omni-2b": {"dir": omni_dir, "lib": lib("lfm3_hg4_engine", [
            STUDIO_CLOVER / "fused", *( [omni_dir] if omni_dir else [] ), Path("/opt/clover-k3/lfm3-omni-hg4")])},
        "clover-35b-a3b": {"dir": c35_dir, "lib": lib("clover35b_hg4_engine", [
            *([c35_dir] if c35_dir else []), Path("/opt/clover-k3/clover35b-full-hg4"),
            SYSTEM_ROOT / "src" / "models" / "clover-35b-a3b"])},
        "nemotron-3.5-super-vl": {"dir": nemo_dir, "lib": lib("nemotron_hg4_engine", [
            *([nemo_dir] if nemo_dir else []), Path("/opt/clover-k3/nemotron-full-hg4"),
            SYSTEM_ROOT / "src" / "models" / "nemotron-3.5-super-vl"])},
        "kimi-k3": {"dir": kimi_dir, "lib": lib("kimi_k31_hg4_engine", [
            *([kimi_dir] if kimi_dir else []), Path("/opt/clover-k3/kimi-k31-hg4")])},
    }


def _eos_ids(tok: Any, names: tuple[str, ...], fallback: int | None = None) -> set[int]:
    out = set()
    for n in names:
        i = tok.token_to_id(n)
        if i is not None:
            out.add(int(i))
    if not out and fallback is not None:
        out.add(int(fallback))
    return out


def make_mock_backend(model_id: str, responder: Responder | None = None) -> StatefulBackend:
    spec = MODEL_SPECS[model_id]
    tpl = TEMPLATES[spec["template"]]
    tok = MockTokenizer()
    eos = tok.specials["<|im_end|>"]
    eng = MockEngine(model_id, tok, tpl, eos, responder=responder, supports_truncate=spec["truncate"])
    return StatefulBackend(model_id, eng, tok, tpl, {eos, tok.specials["<|endoftext|>"]}, spec["family"], kind="mock")


def make_native_backend(model_id: str, paths: dict[str, Path | None]) -> BackendBase:
    d, lp = paths.get("dir"), paths.get("lib")
    if model_id == "kimi-k3":
        return KimiBackend(model_id, os.environ.get("CLOVER_KIMI_URL", "http://127.0.0.1:8765"), d, lp)
    if d is None or lp is None:
        raise FileNotFoundError(f"{model_id}: hg4 dir or native library not found")
    spec = MODEL_SPECS[model_id]
    tpl = TEMPLATES[spec["template"]]
    if model_id == "lfm3-omni-2b":
        ctx = int(os.environ.get("CLOVER_OMNI_CTX", "16384"))
        eng: EngineAdapter = Lfm3Engine(lp, d, max_seq=ctx)
        tok = HFTokenizer(d / "tokenizer.json", vocab_size=128000)
        return StatefulBackend(model_id, eng, tok, tpl, _eos_ids(tok, spec["eos"], 7), spec["family"])
    if model_id == "clover-35b-a3b":
        eng = Clover35bEngine(lp, d, max_seq=int(os.environ.get("CLOVER35B_CTX", "32768")))
        tok = HFTokenizer(d / "tokenizer.json")
        return StatefulBackend(model_id, eng, tok, tpl, _eos_ids(tok, spec["eos"], 248046), spec["family"])
    if model_id == "nemotron-3.5-super-vl":
        eng = NemotronEngine(lp, d, max_seq=int(os.environ.get("CLOVER_NEMO_CTX", "32768")))
        tok = HFTokenizer(d / "tokenizer.json")
        return NemotronBackend(model_id, eng, tok, tpl, _eos_ids(tok, spec["eos"], 11), spec["family"])
    raise KeyError(model_id)


def build_backends(mode: str = "auto", responders: dict[str, Responder] | None = None,
                   log: Callable[[str], None] | None = None) -> dict[str, BackendBase]:
    """`mode`: `mock` (all mock), `real` (native only; unavailable models omitted), `auto` (native else mock)."""
    log = log or (lambda s: None)
    responders = responders or {}
    out: dict[str, BackendBase] = {}
    paths = discover_paths() if mode != "mock" else {}
    for mid in DIRECT_MODELS:
        if mode == "mock":
            out[mid] = make_mock_backend(mid, responders.get(mid))
            continue
        try:
            be = make_native_backend(mid, paths.get(mid, {}))
            if isinstance(be, KimiBackend) and not (KimiBackend.reachable(be.url) or be.native_available()):
                raise FileNotFoundError(f"kimi: {be.url} unreachable and no native engine")
            out[mid] = be
            log(f"[backends] {mid}: {be.kind} ready")
        except Exception as exc:
            if mode == "auto":
                out[mid] = make_mock_backend(mid, responders.get(mid))
                log(f"[backends] {mid}: native unavailable ({exc}); using mock")
            else:
                log(f"[backends] {mid}: unavailable ({exc})")
    return out
