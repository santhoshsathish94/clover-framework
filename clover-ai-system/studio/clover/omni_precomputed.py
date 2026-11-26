"""Precomputed C1 Runtime for Official `LiquidAI/LFM3-Omni-2B` (`omni_precomputed.py`).

Executes `LFM3-Omni-2B`'s static 24-layer language trunks (`trunk-0..23/operator.4bit.bin`, `common.bin`, `operator.bin`)
and static dense SwiGLU experts (`trunk-0..23/ffn.4bit.bin` / `experts/expert-0..23`) plus Two-Stage 4-bit + Top-64 BF16
`global-0/lm_head.4bit.bin` directly on Clover's native C/OpenMP engine (`liblfm3_cpu_engine.dll` / `.so` with
`AVX2`/`AVX-VNNI` on x86-64 and `ARMv9 NEON` on Android) with `C2` (`HybridAgent`) and `C3` (`11-Level NCM`) disabled.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EOS_IDS = {124900, 124895, 125014}  # <|im_end|>, <|endoftext|>, <|text_end|>
THINK_OPEN_ID = 124901
THINK_CLOSE_ID = 124902
MAX_NATIVE_PROMPT_TOKENS = 1280


def precomputed_omni_dir(base_dir: Path | str) -> Path | None:
    """Return a complete precomputed 24-layer LFM3-Omni-2B `SoC-Q4` directory under `base_dir`, or None."""
    base = Path(base_dir)
    candidates = [
        base / "data" / "omni" / "precomputed" / "trunks-omni",
        base / "data" / "omni" / "precomputed" / "trunks",
    ]
    for cand in candidates:
        if _is_complete_trunks_dir(cand):
            return cand
    return None


def _is_complete_trunks_dir(cand: Path) -> bool:
    if not cand.is_dir():
        return False
    g0 = cand / "global-0"
    if not ((g0 / "embed_tokens.bf16.bin").is_file() and (g0 / "embedding_norm.bf16.bin").is_file()
            and (g0 / "rope_inv_freq.bin").is_file() and (cand / "tokenizer.json").is_file()):
        return False
    for idx in range(24):
        td = cand / f"trunk-{idx}"
        if not ((td / "index.bin").is_file() and (td / "common.bin").is_file()
                and (td / "operator.bin").is_file() and (td / "ffn.4bit.bin").is_file()):
            return False
    return True


def find_engine_library() -> Path | None:
    """Locate `liblfm3_cpu_engine.dll` (Windows) or `liblfm3_cpu_engine.so` (Android / Linux)."""
    if os.name == "nt":
        for p in (
            HERE / "fused" / "liblfm3_cpu_engine.dll",
            HERE.parent.parent / "src" / "models" / "lfm3-1.7b" / "liblfm3_cpu_engine.dll",
        ):
            if p.is_file():
                return p
        return None
    # Android Chaquopy nativeLibraryDir
    try:
        from com.chaquo.python import Python  # type: ignore
        ctx = Python.getPlatform().getApplication()
        nat = Path(str(ctx.getApplicationInfo().nativeLibraryDir)) / "liblfm3_cpu_engine.so"
        if nat.is_file():
            return nat
    except Exception:
        pass
    for p in (
        HERE / "fused" / "liblfm3_cpu_engine.so",
        Path("/opt/clover-k3/liblfm3_cpu_engine_omni.so"),
        HERE.parent.parent / "mobile" / "liblfm3_cpu_engine.so",
        HERE.parent.parent / "src" / "models" / "lfm3-1.7b" / "liblfm3_cpu_engine.so",
    ):
        if p.is_file():
            return p
    return None


def precomputed_available(base_dir: Path | str) -> tuple[bool, str]:
    lib_p = find_engine_library()
    if lib_p is None:
        return False, "missing native C1 engine library (liblfm3_cpu_engine)"
    ds_p = precomputed_omni_dir(base_dir)
    if ds_p is None:
        return False, "missing precomputed LFM3-Omni-2B trunks (data/omni/precomputed/trunks-omni)"
    return True, ""


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class _PurePythonBPE:
    """Zero-dependency fallback reader for Hugging Face `tokenizer.json` (used on Android when Rust `tokenizers` is absent)."""

    def __init__(self, tok_json_path: Path):
        raw = json.loads(tok_json_path.read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = dict(raw["model"]["vocab"])
        self.id_to_tok: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.special_ids: set[int] = set()
        added_literals: list[str] = []
        for at in raw.get("added_tokens") or []:
            c = at["content"]
            tid = int(at["id"])
            self.vocab[c] = tid
            self.id_to_tok[tid] = c
            if at.get("special"):
                self.special_ids.add(tid)
            added_literals.append(c)
        added_literals.sort(key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(x) for x in added_literals) + ")") if added_literals else None
        merges = raw["model"].get("merges") or []
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            pair = tuple(m.split(" ", 1)) if isinstance(m, str) else (m[0], m[1])
            self.bpe_ranks[pair] = rank
        self._b2u = _gpt2_bytes_to_unicode()
        self._u2b = {v: k for k, v in self._b2u.items()}
        self._word_re = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10**12))
            if best not in self.bpe_ranks:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                    new_word.append(best[0] + best[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        self._cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        parts = self._special_re.split(text) if self._special_re else [text]
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.vocab and (self._special_re is not None and self._special_re.fullmatch(part)):
                ids.append(self.vocab[part])
                continue
            for piece in self._word_re.findall(part):
                mapped = "".join(self._b2u[b] for b in piece.encode("utf-8"))
                for sub in self._bpe(mapped):
                    tid = self.vocab.get(sub)
                    if tid is not None:
                        ids.append(tid)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for tid in ids:
            if skip_special_tokens and tid in self.special_ids:
                continue
            tok = self.id_to_tok.get(int(tid), "")
            if tok in self.vocab and tid in self.special_ids:
                pieces.append(tok)
            else:
                pieces.append(tok)
        joined = "".join(pieces)
        raw_bytes = bytearray()
        for ch in joined:
            b = self._u2b.get(ch)
            if b is not None:
                raw_bytes.append(b)
            else:
                raw_bytes.extend(ch.encode("utf-8", errors="replace"))
        return raw_bytes.decode("utf-8", errors="replace")


class OmniTokenizer:
    def __init__(self, tok_json_path: Path):
        self._fast = None
        self._py = None
        try:
            from tokenizers import Tokenizer  # type: ignore
            self._fast = Tokenizer.from_file(str(tok_json_path))
        except Exception:
            self._py = _PurePythonBPE(tok_json_path)

    def encode(self, text: str) -> list[int]:
        if self._fast is not None:
            return list(self._fast.encode(text, add_special_tokens=False).ids)
        return self._py.encode(text)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if not ids:
            return ""
        if self._fast is not None:
            return self._fast.decode([int(x) for x in ids], skip_special_tokens=skip_special_tokens)
        return self._py.decode(ids, skip_special_tokens=skip_special_tokens)


def has_multimodal_parts(msgs: list[dict]) -> bool:
    for m in msgs or ():
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_audio", "input_video"):
                    return True
    return False


def format_omni_prompt(msgs: list[dict], schemas: list[dict] | None = None) -> str:
    """Render OpenAI-style `msgs` + optional tool `schemas` into Official `LiquidAI/LFM3-Omni-2B` ChatML prompt."""
    out = ["<|startoftext|>"]
    wrote_sys = False
    tool_defs = [s.get("function", s) for s in (schemas or []) if isinstance(s, dict)]

    def _text_of(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content or "")

    for m in msgs or ():
        role = m.get("role") or "user"
        txt = _text_of(m.get("content"))
        if role == "system":
            if not wrote_sys and tool_defs:
                txt = txt.rstrip() + "\nList of tools: <|tool_list_start|>" + json.dumps(tool_defs) + "<|tool_list_end|>"
            wrote_sys = True
            out.append(f"<|im_start|>system\n{txt}<|im_end|>\n")
        elif role == "user":
            if not wrote_sys:
                sys_txt = "Respond with text."
                if tool_defs:
                    sys_txt += "\nList of tools: <|tool_list_start|>" + json.dumps(tool_defs) + "<|tool_list_end|>"
                out.append(f"<|im_start|>system\n{sys_txt}<|im_end|>\n")
                wrote_sys = True
            out.append(f"<|im_start|>user\n{txt}<|im_end|>\n")
        elif role == "assistant":
            tcalls = m.get("tool_calls") or []
            if tcalls:
                rendered_calls = []
                for tc in tcalls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name") or ""
                    raw_args = fn.get("arguments", tc.get("arguments", {}))
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except ValueError:
                            raw_args = {}
                    if isinstance(raw_args, dict):
                        kvs = ", ".join(f"{k}={json.dumps(v)}" for k, v in raw_args.items())
                    else:
                        kvs = ""
                    rendered_calls.append(f"{name}({kvs})")
                tc_str = "<|tool_call_start|>[" + ", ".join(rendered_calls) + "]<|tool_call_end|>"
                body = (tc_str + (txt or "")) if txt else tc_str
                out.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>assistant\n{txt}<|text_end|><|im_end|>\n")
        elif role == "tool":
            out.append(f"<|im_start|>tool\n<|tool_response_start|>{txt}<|tool_response_end|><|im_end|>\n")

    out.append("<|im_start|>assistant\n")
    return "".join(out)


def parse_lfm3_bracket_tool_calls(text: str, by_name: dict) -> tuple[str, list[dict]]:
    """Parse LFM3 `<|tool_call_start|>[fn(k="v")]<|tool_call_end|>` calls if emitted by the native C1 engine."""
    import ast
    import uuid
    m = re.search(r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>", text, re.S)
    if not m:
        return text, []
    expr_src = "[" + m.group(1) + "]"
    calls = []
    try:
        tree = ast.parse(expr_src, mode="eval")
        if isinstance(tree.body, ast.List):
            for elt in tree.body.elts:
                if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) and elt.func.id in by_name:
                    args = {}
                    for kw in elt.keywords:
                        if kw.arg:
                            args[kw.arg] = ast.literal_eval(kw.value)
                    calls.append({"id": "call_" + uuid.uuid4().hex[:16], "name": elt.func.id, "arguments": args})
    except Exception:
        return text, []
    cleaned = (text[: m.start()] + text[m.end() :]).strip()
    return cleaned, calls


# Process-global singleton because `liblfm3_cpu_engine` maintains `mmap`ped global trunks (`g_lfm3`).
_RUNTIME_LOCK = threading.Lock()
_SINGLETON: PrecomputedOmniRuntime | None = None


class PrecomputedOmniRuntime:
    """In-process ctypes wrapper around `liblfm3_cpu_engine` executing precomputed 4-bit LFM3-Omni-2B trunks + experts."""

    def __init__(self, ds_dir: Path, lib_path: Path, threads: int | None = None):
        t0 = time.perf_counter()
        self.ds_dir = Path(ds_dir)
        self.lib_path = Path(lib_path)
        if threads and int(threads) > 0:
            os.environ["OMP_NUM_THREADS"] = str(int(threads))
        self.tokenizer = OmniTokenizer(self.ds_dir / "tokenizer.json")
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
        self.lib.lfm3_engine_init.restype = ctypes.c_int
        self.lib.lfm3_set_use_4bit_ffn.argtypes = [ctypes.c_int]
        self.lib.lfm3_set_use_4bit_operators.argtypes = [ctypes.c_int]
        self.lib.lfm3_set_use_4bit_lm_head.argtypes = [ctypes.c_int]
        self.lib.lfm3_prefill_cached.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.lfm3_prefill_cached.restype = ctypes.c_float
        self.lib.lfm3_step_token_cached.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.lfm3_step_token_cached.restype = ctypes.c_float
        self.lib.lfm3_lm_head_logits.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_lm_head_logits.restype = ctypes.c_int

        rc = self.lib.lfm3_engine_init(str(self.ds_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"liblfm3_cpu_engine failed to initialize {self.ds_dir} (rc={rc})")
        has_op4 = (self.ds_dir / "trunk-0" / "operator.4bit.bin").is_file()
        has_lm4 = (self.ds_dir / "global-0" / "lm_head.4bit.bin").is_file()
        self.lib.lfm3_set_use_4bit_ffn(1)
        self.lib.lfm3_set_use_4bit_operators(1 if has_op4 else 0)
        self.lib.lfm3_set_use_4bit_lm_head(1 if has_lm4 else 0)

        self._exec_lock = threading.Lock()
        self._h_buf = (ctypes.c_float * 2048)()
        self._logits_buf = (ctypes.c_float * 128000)()
        self._max_l = ctypes.c_float()
        self._cached_seq: list[int] = []
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def stream_chat(self, prompt_ids: list[int], max_new: int, *, on_delta=None, cancel=None, by_name=None,
                    looping_fn=None, trim_loop_fn=None, rambling_fn=None) -> tuple[str, list[dict], dict]:
        with self._exec_lock:
            n_prompt = len(prompt_ids)
            lcp = 0
            limit_lcp = min(n_prompt - 1, len(self._cached_seq))
            while lcp < limit_lcp and prompt_ids[lcp] == self._cached_seq[lcp]:
                lcp += 1

            arr = (ctypes.c_int * n_prompt)(*prompt_ids)
            t_pre0 = time.perf_counter()
            self.lib.lfm3_prefill_cached(arr, n_prompt, self._h_buf)
            nxt = int(self.lib.lfm3_lm_head_logits(self._h_buf, self._logits_buf, ctypes.byref(self._max_l)))
            prefill_s = time.perf_counter() - t_pre0
            self._cached_seq = list(prompt_ids)

            max_steps = max(1, min(int(max_new), 1536 - n_prompt - 1))
            gen_ids: list[int] = []
            visible_ids: list[int] = []
            emitted_text = ""
            in_empty_think = False
            finish = "stop"
            looped = False
            cut = None
            checked = 0

            t_dec0 = time.perf_counter()
            for step in range(max_steps):
                if cancel is not None and cancel.is_set():
                    finish = "cancelled"
                    break
                if nxt in EOS_IDS:
                    finish = "stop"
                    break
                gen_ids.append(nxt)

                # Strip leading `<think></think>` wrapper emitted by LFM3-Omni-2B
                if len(gen_ids) == 1 and nxt == THINK_OPEN_ID:
                    in_empty_think = True
                elif in_empty_think and nxt == THINK_CLOSE_ID:
                    in_empty_think = False
                else:
                    in_empty_think = False
                    visible_ids.append(nxt)
                    full_now = self.tokenizer.decode(visible_ids, skip_special_tokens=False)
                    # Hide incomplete special tags at tail until completed
                    clean_now = full_now.replace("<|text_end|>", "")
                    if "<|tool_call_end|>" in clean_now or "</tool_call>" in clean_now:
                        emitted_text = clean_now
                        finish = "tool_calls"
                        break
                    if "<|tool_call_start|>" not in clean_now and "<tool_call>" not in clean_now:
                        delta = clean_now[len(emitted_text) :]
                        if delta:
                            emitted_text = clean_now
                            if on_delta:
                                on_delta(delta, emitted_text)
                    if looping_fn and len(emitted_text) - checked >= 200:
                        checked = len(emitted_text)
                        looped = bool(looping_fn(emitted_text))
                        cut = rambling_fn(emitted_text) if rambling_fn else None
                        if looped:
                            emitted_text = trim_loop_fn(emitted_text) if trim_loop_fn else emitted_text
                            finish = "stop"
                            break
                        if cut is not None:
                            emitted_text = emitted_text[:cut]
                            finish = "stop"
                            break

                if step + 1 >= max_steps:
                    finish = "length"
                    break
                self.lib.lfm3_step_token_cached(int(nxt), self._h_buf)
                self._cached_seq.append(int(nxt))
                nxt = int(self.lib.lfm3_lm_head_logits(self._h_buf, self._logits_buf, ctypes.byref(self._max_l)))

            decode_s = time.perf_counter() - t_dec0
            raw_decoded = self.tokenizer.decode(visible_ids, skip_special_tokens=False).replace("<|text_end|>", "")
            if looped or cut is not None:
                final_text = emitted_text
            else:
                final_text = raw_decoded

            calls = []
            if by_name:
                final_text, calls = parse_lfm3_bracket_tool_calls(final_text, by_name)

            n_new = len(gen_ids)
            stats = {
                "prompt_tokens": n_prompt,
                "reused_tokens": lcp,
                "new_tokens": n_new,
                "prefill_s": round(prefill_s, 3),
                "decode_s": round(decode_s, 3),
                "decode_tok_s": round(n_new / decode_s, 2) if decode_s > 0 and n_new else None,
                "finish": "stop" if finish == "tool_calls" else finish,
                "backend": "liblfm3_cpu_engine (precomputed 4-bit trunks + experts)",
                **({"looped": True} if looped else {}),
                **({"rambling": True} if cut is not None else {}),
            }
            return final_text, calls, stats


def get_runtime(base_dir: Path | str, threads: int | None = None) -> PrecomputedOmniRuntime:
    global _SINGLETON
    ds_dir = precomputed_omni_dir(base_dir)
    lib_path = find_engine_library()
    if ds_dir is None or lib_path is None:
        ok, why = precomputed_available(base_dir)
        raise FileNotFoundError(why or "precomputed LFM3-Omni-2B trunks not available")
    with _RUNTIME_LOCK:
        if _SINGLETON is None or _SINGLETON.ds_dir != ds_dir:
            _SINGLETON = PrecomputedOmniRuntime(ds_dir, lib_path, threads=threads)
        return _SINGLETON
