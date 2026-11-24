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


_KIMI_K3_PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


class TiktokenFallbackTokenizer:
    """Kimi K3.1 Pro `tiktoken.model` via official `tiktoken` (when installed) or `PureTiktokenTokenizer`."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._enc = None
        self._t = None
        self.special_tokens: dict[str, int] = {}
        cfg_path = self.model_path.parent / "tokenizer_config.json"
        special_map: dict[int, str] = {}
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for k_id, info in (cfg.get("added_tokens_decoder") or {}).items():
                    c_str = str(info.get("content") or "")
                    if c_str:
                        special_map[int(k_id)] = c_str
            except Exception:
                pass
        try:
            import tiktoken  # type: ignore[import-not-found]
            from tiktoken.load import load_tiktoken_bpe  # type: ignore[import-not-found]
            mergeable = load_tiktoken_bpe(str(self.model_path))
            num_base = len(mergeable)
            self.special_tokens = {
                special_map.get(i, f"<|reserved_token_{i}|>"): i
                for i in range(num_base, num_base + 256)
            }
            self._enc = tiktoken.Encoding(
                name="kimi_k31", pat_str=_KIMI_K3_PAT_STR,
                mergeable_ranks=mergeable, special_tokens=self.special_tokens,
            )
        except Exception:
            if str(STUDIO_CLOVER) not in sys.path:
                sys.path.append(str(STUDIO_CLOVER))
            import omni_c2_c3  # type: ignore[import-not-found]
            self._t = omni_c2_c3.PureTiktokenTokenizer(self.model_path)
            self.special_tokens = dict(self._t.added_tokens_map)
        self._sp_re = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            if self.special_tokens else None
        )

    def token_to_id(self, s: str) -> int | None:
        return self.special_tokens.get(s)

    def encode_piece(self, text: str, allow_special: bool = True) -> list[int]:
        if not text:
            return []
        if self._enc is not None:
            if allow_special:
                return list(self._enc.encode(text, allowed_special="all"))
            return list(self._enc.encode(text, disallowed_special=()))
        assert self._t is not None
        if not allow_special or self._sp_re is None:
            return self._t.encode(text, allow_special=False)
        out: list[int] = []
        for part in self._sp_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self._t.encode(part, allow_special=False))
        return out

    def encode(self, text: str) -> list[int]:
        return self.encode_piece(text, allow_special=True)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        if self._enc is not None:
            return self._enc.decode(list(ids))
        assert self._t is not None
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


def assistant_signature(content: Any, tool_calls: Any, prefix_messages: list[dict[str, Any]] | None = None) -> str:
    """Conversation-prefix-scoped canonical key of an assistant message for exact-token splicing."""
    calls = []
    for tc in tool_calls or ():
        if isinstance(tc, dict):
            calls.append(list(_tool_call_args(tc)))
    pfx: list[tuple[str, str]] = []
    for m in prefix_messages or ():
        if isinstance(m, dict):
            role = str(m.get("role") or "user")
            if role in ("system", "developer"):
                continue
            pfx.append((role, _content_text(m.get("content")).strip()))
    blob = json.dumps({"p": pfx, "c": _content_text(content).strip(), "t": calls}, sort_keys=True, ensure_ascii=False)
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
        seen_msgs: list[dict[str, Any]] = []
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
                             assistant_signature(m.get("content"), m.get("tool_calls"), seen_msgs)
                             + "|" + assistant_signature(m.get("content"), m.get("tool_calls"), None)))
            elif role in ("tool", "function"):
                c = _content_text(m.get("content"))
                if self.wrap_tool_response and "<tool_response>" not in c:
                    c = f"<tool_response>\n{c}\n</tool_response>"
                segs.append((self.tool_open + c + self.end, None))
            elif role in ("system", "developer"):
                segs.append((self.sys_open + _content_text(m.get("content")) + self.end, None))
            else:
                segs.append((self.user_open + _content_text(m.get("content")) + self.end, None))
            seen_msgs.append(m)
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
    if "[Native C11 x-cruncher verified exact values" in text:
        return f"[{model_id}] " + text.split("[Native C11 x-cruncher verified exact values", 1)[1]
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

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int],
                           prefix_messages: list[dict[str, Any]] | None = None) -> None:
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
            cand = None
            if key is not None:
                k_scoped, _, k_plain = key.partition("|")
                cand = self._splice.get(k_scoped)
                if cand is None and k_plain:
                    cand = self._splice.get(k_plain)
            if cand is not None:
                ids.extend(asst_ids)
                ids.extend(cand)
                ids.extend(end_ids)
            else:
                ids.extend(self._encode_seg(text))
        return ids

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int],
                           prefix_messages: list[dict[str, Any]] | None = None) -> None:
        if not gen_ids:
            return
        key = assistant_signature(content, tool_calls, prefix_messages)
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
            self._splice.clear()

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


_XTML_CALL_RE = re.compile(
    r'<\|open\|>call\s+tool="([^"]+)"(?:\s+index="[^"]*")?<\|sep\|>(.*?)(?:<\|close\|>call<\|sep\|>|\Z)',
    re.DOTALL,
)
_XTML_ARG_RE = re.compile(
    r'<\|open\|>argument\s+key="([^"]+)"\s+type="([^"]+)"<\|sep\|>(.*?)(?:<\|close\|>argument<\|sep\|>|\Z)',
    re.DOTALL,
)
_XTML_JSON_RE = re.compile(
    r'<\|open\|>json(?:\s+type="[^"]*")?<\|sep\|>(.*?)(?:<\|close\|>json<\|sep\|>|\Z)',
    re.DOTALL,
)


def parse_kimi_xtml_output(raw: str) -> str:
    """Convert Kimi K3.1 Pro XTML assistant output (`<think>`, `<response>`, `<tools>`) into clean text + `<tool_call>`."""
    s = raw or ""
    # 1. Strip `<|open|>think<|sep|>...<|close|>think<|sep|>`; if truncated inside `<think>`, never leak chain-of-thought
    if "<|close|>think<|sep|>" in s:
        s = s.split("<|close|>think<|sep|>", 1)[1]
    elif s.startswith("<|open|>think<|sep|>") and "<|open|>response<|sep|>" not in s and "<|open|>tools<|sep|>" not in s:
        return ""
    elif "<|open|>response<|sep|>" in s:
        s = s[s.index("<|open|>response<|sep|>"):]

    # 2. Extract `<response>` content and `<tools>` block
    tools_part = ""
    if "<|open|>tools<|sep|>" in s:
        s, tools_part = s.split("<|open|>tools<|sep|>", 1)

    if "<|open|>response<|sep|>" in s:
        s = s.split("<|open|>response<|sep|>", 1)[1]
    for stop_tag in ("<|close|>response<|sep|>", "<|close|>message<|sep|>", "<|end_of_msg|>", "[EOS]"):
        if stop_tag in s:
            s = s.split(stop_tag, 1)[0]
    content = s.strip()

    tc_blocks: list[str] = []
    if tools_part:
        if "<|close|>tools<|sep|>" in tools_part:
            tools_part = tools_part.split("<|close|>tools<|sep|>", 1)[0]
        for m in _XTML_CALL_RE.finditer(tools_part):
            fn_name = m.group(1).replace("&quot;", '"').replace("&amp;", "&")
            body = m.group(2)
            jm = _XTML_JSON_RE.search(body)
            args: dict[str, Any] = {}
            if jm:
                try:
                    parsed = tg.repair_json(jm.group(1))
                    if isinstance(parsed, dict):
                        args = parsed
                except tg.JsonRepairError:
                    args = {}
            else:
                for am in _XTML_ARG_RE.finditer(body):
                    akey = am.group(1).replace("&quot;", '"').replace("&amp;", "&")
                    atype = am.group(2).strip().lower()
                    aval = am.group(3)
                    if atype == "string":
                        args[akey] = aval
                    elif atype == "boolean":
                        args[akey] = aval.strip().lower() == "true"
                    elif atype == "null":
                        args[akey] = None
                    elif atype == "number":
                        try:
                            args[akey] = int(aval.strip()) if re.fullmatch(r"-?\d+", aval.strip()) else float(aval.strip())
                        except ValueError:
                            args[akey] = aval
                    else:
                        try:
                            args[akey] = json.loads(aval)
                        except ValueError:
                            try:
                                args[akey] = tg.repair_json(aval)
                            except tg.JsonRepairError:
                                args[akey] = aval
            tc_blocks.append("<tool_call>\n" + json.dumps({"name": fn_name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>")

    if tc_blocks:
        return (content + "\n" if content else "") + "\n".join(tc_blocks)
    return content


class KimiBackend(BackendBase):
    """Kimi K3 / K3.1 Pro: proxies to `clover-k3-chat.service` (`/api/chat`) or the native `libkimi_k31_hg4_engine`."""

    family = "kimi"

    def __init__(self, model_id: str, url: str, hg4_dir: Path | None = None, lib_path: Path | None = None,
                 timeout_s: float = 900.0, splice_capacity: int = 256):
        super().__init__()
        self.model_id = model_id
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.hg4_dir = hg4_dir
        self.lib_path = lib_path
        self._lib = None
        self._tok: Any = None
        self._enc_k3: Any = None
        self._n_layers = 0
        self.kind = "proxy"
        self.template = TEMPLATES["kimi"]
        self._splice: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
        self._splice_cap = splice_capacity

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

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int],
                           prefix_messages: list[dict[str, Any]] | None = None) -> None:
        if not gen_ids:
            return
        key = assistant_signature(content, tool_calls, prefix_messages)
        with ENGINE_LOCK:
            self._splice[key] = list(gen_ids)
            self._splice.move_to_end(key)
            while len(self._splice) > self._splice_cap:
                self._splice.popitem(last=False)

    def _transcript(self, messages, tools, tool_choice) -> tuple[str, list[dict[str, str]]]:
        segs = self.template.segments(messages, tools, tool_choice)
        flat = "".join(t for t, _ in segs)
        hist = []
        for m in messages or []:
            if m.get("role") in ("user", "assistant"):
                hist.append({"role": m["role"], "content": _content_text(m.get("content"))})
        return flat, hist

    def _proxy(self, messages, tools, tool_choice, params: GenParams) -> str:
        body: dict[str, Any] = {
            "model": "kimi-k3",
            "cands": 1,
            "session_id": f"hg4-fusion-{uuid.uuid4().hex[:12]}",
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        ch = data.get("choices")
        if isinstance(ch, list) and ch:
            msg = ch[0].get("message") or {}
            tcs = msg.get("tool_calls") or []
            content = str(msg.get("content") or ch[0].get("text") or "")
            if tcs:
                tc_blocks = []
                for tc in tcs:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments") or "{}"
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    tc_blocks.append("<tool_call>\n" + json.dumps({"name": fn.get("name", ""), "arguments": args}) + "\n</tool_call>")
                return (content + "\n" if content else "") + "\n".join(tc_blocks)
            return content
        for k in ("reply", "response", "text", "content", "answer"):
            if isinstance(data.get(k), str):
                return data[k]
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
        if hasattr(lib, "kimi_k31_cache_reset"):
            lib.kimi_k31_cache_reset.argtypes = []
            lib.kimi_k31_cache_reset.restype = None
        if hasattr(lib, "kimi_k31_last_lcp_tokens"):
            lib.kimi_k31_last_lcp_tokens.argtypes = []
            lib.kimi_k31_last_lcp_tokens.restype = ctypes.c_int
        rc = lib.kimi_k31_hg4_init(str(self.hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"kimi_k31_hg4_init rc={rc}")
        self._n_layers = int(lib.kimi_k31_hg4_num_layers())
        tj = self.hg4_dir / "tokenizer.json"
        self._tok = HFTokenizer(tj) if tj.is_file() else TiktokenFallbackTokenizer(self.hg4_dir / "tiktoken.model")
        ek3 = self.hg4_dir / "encoding_k3.py"
        if ek3.is_file():
            if str(self.hg4_dir) not in sys.path:
                sys.path.insert(0, str(self.hg4_dir))
            import encoding_k3  # type: ignore[import-not-found]
            self._enc_k3 = encoding_k3
        self._lib = lib

    def reset_cache(self) -> None:
        with ENGINE_LOCK:
            self._splice.clear()
            if self._lib is not None and hasattr(self._lib, "kimi_k31_cache_reset"):
                self._lib.kimi_k31_cache_reset()

    def _encode_xtml_segments(self, segs: list[Any]) -> list[int]:
        out: list[int] = []
        has_piece = hasattr(self._tok, "encode_piece")
        for seg in segs:
            if has_piece:
                out.extend(self._tok.encode_piece(seg.text, allow_special=bool(seg.allow_special)))
            else:
                out.extend(self._tok.encode(seg.text))
        return out

    def _normalize_k31_messages(self, messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        norm: list[dict[str, Any]] = []
        last_tool_names: dict[str, str] = {}
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            if role == "developer":
                role = "system"
            elif role == "function":
                role = "tool"
            cm: dict[str, Any] = {"role": role, "content": _content_text(m.get("content"))}
            if m.get("name"):
                cm["name"] = str(m["name"])
            if role == "assistant":
                last_tool_names = {}
                if m.get("reasoning_content"):
                    cm["reasoning_content"] = str(m["reasoning_content"])
                if m.get("tool_calls"):
                    cm["tool_calls"] = m["tool_calls"]
                    for tc in m["tool_calls"]:
                        if isinstance(tc, dict) and tc.get("id"):
                            fn = tc.get("function") or tc
                            last_tool_names[str(tc["id"])] = str(fn.get("name") or "tool")
            elif role == "tool":
                cid = str(m.get("tool_call_id") or m.get("id") or "")
                if cid:
                    cm["tool_call_id"] = cid
                tname = m.get("tool") or m.get("name") or last_tool_names.get(cid) or "tool"
                cm["name"] = str(tname)
                cm["tool"] = str(tname)
            norm.append(cm)
        return norm

    def _build_k31_prompt_ids(self, messages, tools, tool_choice, think_mode: str) -> list[int]:
        if self._enc_k3 is None:
            flat, _ = self._transcript(messages, tools, tool_choice)
            return self._tok.encode(flat)

        norm_msgs = self._normalize_k31_messages(messages)
        use_thinking = think_mode != "off"
        kw: dict[str, Any] = {}
        if use_thinking:
            if think_mode == "bounded":
                kw["thinking_effort"] = "low"
            elif think_mode in ("low", "medium", "high", "max"):
                kw["thinking_effort"] = think_mode
        if tool_choice in ("required", "none"):
            kw["tool_choice"] = tool_choice

        # Check if the latest prior assistant message has exact prefix-scoped spliced token IDs ending in `<|end_of_msg|>`
        eos_tok = self._tok.token_to_id("<|end_of_msg|>") or 163586
        last_asst_idx = -1
        spliced_gen: list[int] | None = None
        for idx in range(len(norm_msgs) - 1, -1, -1):
            if norm_msgs[idx]["role"] == "assistant":
                sig = assistant_signature(
                    norm_msgs[idx].get("content"), norm_msgs[idx].get("tool_calls"), norm_msgs[:idx]
                )
                cand = self._splice.get(sig)
                if cand and cand[-1] == eos_tok:
                    last_asst_idx = idx
                    spliced_gen = cand
                    break

        if last_asst_idx >= 0 and spliced_gen is not None:
            # `prefix_segs` omits `tool_choice` only because `encoding_k3` appends `tool-choice` at the end of the prompt
            prefix_kw = {k: v for k, v in kw.items() if k != "tool_choice"}
            prefix_segs = self._enc_k3.build_chat_segments(
                norm_msgs[:last_asst_idx], tools=tools, add_generation_prompt=True, thinking=use_thinking, **prefix_kw
            )
            non_gen_through_asst = self._enc_k3.build_chat_segments(
                norm_msgs[:last_asst_idx + 1], tools=tools, add_generation_prompt=False, thinking=use_thinking, **prefix_kw
            )
            full_segs = self._enc_k3.build_chat_segments(
                norm_msgs, tools=tools, add_generation_prompt=True, thinking=use_thinking, **kw
            )
            cut = len(non_gen_through_asst)
            if cut <= len(full_segs) and full_segs[:cut] == non_gen_through_asst:
                tail_segs = full_segs[cut:]
                return self._encode_xtml_segments(prefix_segs) + list(spliced_gen) + self._encode_xtml_segments(tail_segs)

        segs = self._enc_k3.build_chat_segments(
            norm_msgs, tools=tools, add_generation_prompt=True, thinking=use_thinking, **kw
        )
        return self._encode_xtml_segments(segs)

    def _native(self, messages, tools, tool_choice, params: GenParams) -> tuple[str, GenResult]:
        self._ensure_native()
        think_mode = os.environ.get("CLOVER_KIMI_THINKING", "bounded").strip().lower()
        ids = self._build_k31_prompt_ids(messages, tools, tool_choice, think_mode)
        eos = self._tok.token_to_id("<|end_of_msg|>") or self._tok.token_to_id("<|im_end|>") or 163586
        close_tok = self._tok.token_to_id("<|close|>") or 163588
        layers = (ctypes.c_int * self._n_layers)(*range(self._n_layers))
        draft_k = int(os.environ.get("CLOVER_KIMI_DRAFT_K", "4"))

        if self._enc_k3 is not None and think_mode == "bounded":
            think_cap = max(8, int(os.environ.get("CLOVER_KIMI_THINK_CAP", "48")))
            arr1 = (ctypes.c_int * len(ids))(*ids)
            out1 = (ctypes.c_int * (think_cap + 8))()
            pf1, dc1 = ctypes.c_double(0), ctypes.c_double(0)
            acc1, tot1 = ctypes.c_int(0), ctypes.c_int(0)
            n1 = self._lib.kimi_k31_generate_speculative(
                arr1, len(ids), layers, self._n_layers, think_cap, close_tok,
                draft_k, out1, ctypes.byref(pf1), ctypes.byref(dc1),
                ctypes.byref(acc1), ctypes.byref(tot1),
            )
            if n1 < 0:
                raise RuntimeError(f"kimi_k31_generate_speculative (think) rc={n1}")
            lcp1 = int(self._lib.kimi_k31_last_lcp_tokens()) if hasattr(self._lib, "kimi_k31_last_lcp_tokens") else 0
            raw_think = list(out1[:n1])
            if not raw_think or raw_think[-1] != close_tok:
                raw_think.append(close_tok)

            bridge = self._encode_xtml_segments([
                self._enc_k3.EncodeSegment("think", allow_special=False),
                self._enc_k3.EncodeSegment("<|sep|>", allow_special=True),
                self._enc_k3.EncodeSegment("<|open|>", allow_special=True),
                self._enc_k3.EncodeSegment("response", allow_special=False),
                self._enc_k3.EncodeSegment("<|sep|>", allow_special=True),
            ])
            ids2 = ids + raw_think + bridge
            resp_stop = eos if tools else close_tok
            resp_max = max(16, int(params.max_tokens))
            arr2 = (ctypes.c_int * len(ids2))(*ids2)
            out2 = (ctypes.c_int * (resp_max + 8))()
            pf2, dc2 = ctypes.c_double(0), ctypes.c_double(0)
            acc2, tot2 = ctypes.c_int(0), ctypes.c_int(0)
            n2 = self._lib.kimi_k31_generate_speculative(
                arr2, len(ids2), layers, self._n_layers, resp_max, resp_stop,
                draft_k, out2, ctypes.byref(pf2), ctypes.byref(dc2),
                ctypes.byref(acc2), ctypes.byref(tot2),
            )
            if n2 < 0:
                raise RuntimeError(f"kimi_k31_generate_speculative (response) rc={n2}")
            raw_resp = list(out2[:n2])

            full_gen = raw_think + bridge + raw_resp
            if not tools and raw_resp and raw_resp[-1] == close_tok:
                close_msg = self._encode_xtml_segments([
                    self._enc_k3.EncodeSegment("response", allow_special=False),
                    self._enc_k3.EncodeSegment("<|sep|>", allow_special=True),
                    self._enc_k3.EncodeSegment("<|close|>", allow_special=True),
                    self._enc_k3.EncodeSegment("message", allow_special=False),
                    self._enc_k3.EncodeSegment("<|sep|>", allow_special=True),
                    self._enc_k3.EncodeSegment("<|end_of_msg|>", allow_special=True),
                ])
                full_gen = full_gen + close_msg

            resp_only = [g for g in raw_resp if g != eos and (tools or g != close_tok)]
            raw_text = "<|open|>response<|sep|>" + self._tok.decode(resp_only)
            text = parse_kimi_xtml_output(raw_text)
            tot_comp = len(raw_think) + len(raw_resp)
            hit_stop = bool(raw_resp and raw_resp[-1] == resp_stop)
            return text, GenResult(
                model=self.model_id, finish_reason="stop" if hit_stop else "length",
                prompt_tokens=len(ids), cached_tokens=lcp1,
                prefill_tokens=max(0, len(ids) - lcp1), completion_tokens=tot_comp,
                prefill_ms=pf1.value, decode_ms=dc1.value + pf2.value + dc2.value,
                backend_kind="native", gen_ids=full_gen if hit_stop else [],
                extra={"dspark_accepted": acc1.value + acc2.value, "dspark_drafted": tot1.value + tot2.value},
            )

        think_budget = int(os.environ.get("CLOVER_KIMI_THINK_BUDGET", "128")) if think_mode != "off" else 0
        eng_max_tokens = int(params.max_tokens) + think_budget
        arr = (ctypes.c_int * len(ids))(*ids)
        out = (ctypes.c_int * (eng_max_tokens + 8))()
        pf, dc = ctypes.c_double(0), ctypes.c_double(0)
        acc, tot = ctypes.c_int(0), ctypes.c_int(0)
        n = self._lib.kimi_k31_generate_speculative(arr, len(ids), layers, self._n_layers, eng_max_tokens, eos,
                                                    draft_k, out, ctypes.byref(pf), ctypes.byref(dc),
                                                    ctypes.byref(acc), ctypes.byref(tot))
        if n < 0:
            raise RuntimeError(f"kimi_k31_generate_speculative rc={n}")
        raw_gen = list(out[:n])
        lcp = int(self._lib.kimi_k31_last_lcp_tokens()) if hasattr(self._lib, "kimi_k31_last_lcp_tokens") else 0
        gen = [g for g in raw_gen if g != eos]
        raw_text = self._tok.decode(gen)
        if self._enc_k3 is not None:
            text = parse_kimi_xtml_output(raw_text)
        else:
            text = raw_text
            for mk in self.template.stop_markers:
                text = text.split(mk)[0]
        hit_stop = bool(raw_gen and raw_gen[-1] == eos)
        return text, GenResult(
            model=self.model_id, finish_reason="stop" if hit_stop else "length",
            prompt_tokens=len(ids), cached_tokens=lcp, prefill_tokens=max(0, len(ids) - lcp),
            completion_tokens=len(gen), prefill_ms=pf.value, decode_ms=dc.value,
            backend_kind="native", gen_ids=raw_gen if hit_stop else [],
            extra={"dspark_accepted": acc.value, "dspark_drafted": tot.value},
        )

    def stream(self, messages, tools, tool_choice, params: GenParams):
        with ENGINE_LOCK:
            t0 = time.perf_counter()
            res: GenResult
            prefer_native = os.environ.get("CLOVER_KIMI_PREFER_NATIVE") == "1"
            if prefer_native and self.native_available():
                self.kind = "native"
                text, res = self._native(messages, tools, tool_choice, params)
            elif self.reachable(self.url):
                self.kind = "proxy"
                text = self._proxy(messages, tools, tool_choice, params)
                res = GenResult(model=self.model_id, finish_reason="stop", backend_kind="proxy",
                                prompt_tokens=sum(len(_content_text(m.get("content"))) // 4 for m in messages or []),
                                completion_tokens=max(1, len(text) // 4) if text else 0,
                                decode_ms=(time.perf_counter() - t0) * 1000.0)
            elif self.native_available():
                self.kind = "native"
                text, res = self._native(messages, tools, tool_choice, params)
            else:
                raise RuntimeError(f"kimi backend unavailable: {self.url} unreachable and no native engine")
            stop_cut = False
            for s in params.stop or []:
                if s and s in text:
                    text = text.split(s)[0]
                    stop_cut = True
            if stop_cut:
                res.gen_ids = []
                res.finish_reason = "stop"
            res.text = text
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
    active_env = (os.environ.get("CLOVER_ACTIVE_MODELS") or "").strip()
    allowed = {m.strip() for m in active_env.split(",") if m.strip()} if active_env else set(DIRECT_MODELS)
    for mid in DIRECT_MODELS:
        if mid not in allowed:
            continue
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
