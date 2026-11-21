"""Native C11 `HG4-128` Runtime Backend (`liblfm3_hg4_engine`) for `LFM3-Omni-2B` in Clover AI Studio.

Provides `Hg4Server`, a drop-in in-process replacement for `llamacpp_engine.LlamaServer` backed by the `853.88 MiB`
`HG4-128` weight arrangement (`Column-Balanced FWHT-128` + `1-Cycle SIMD LUT16` + `Unified 128.91 MiB Tied Embedding/LM-Head`
+ `4-Row Tile Register-Fused SwiGLU`):
  - Full OpenAI/LlamaServer-compatible `chat_stream(body, cancel)` with Longest-Common-Prefix (`LCP`) BF16 KV + ShortConv
    cache reuse across multi-turn chats and multi-round Studio tool loops (`<tool_call><function=...>`).
  - Direct in-process 24-layer `hidden_states(ids)` for `C1 -> C2 HybridAgent -> C3 11-Level NCM` (`zero` extra server process).
  - Automatic cross-platform library loading (`Windows .dll`, `Linux .so`, `Android arm64-v8a .so`) and tokenizer fallback
    (`Rust tokenizers` -> pure-Python `Lfm3PureBpeTokenizer`).
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
CSRC_CANDIDATES = (
    HERE / "models" / "lfm3-1.7b" / "lfm3_hg4_engine.c",
    HERE.parent.parent / "mobile" / "lfm3_hg4_engine.c",
    HERE.parent.parent / "src" / "models" / "lfm3-1.7b" / "lfm3_hg4_engine.c",
)

V_OMNI = 128000
D_OMNI = 2048

_LIB_LOCK = threading.Lock()
_LIB_CACHE: ctypes.CDLL | None = None


def hg4_dir_for_base(base_dir: str | Path) -> Path:
    b = Path(base_dir)
    candidates = [
        b / "data" / "omni" / "hg4",
        Path("/opt/clover-k3/web-studio/data/omni/hg4"),
        Path("/opt/clover-k3/lfm3-omni-hg4"),
    ]
    env_d = (os.environ.get("CLOVER_OMNI_HG4_DIR") or "").strip()
    if env_d:
        candidates.insert(0, Path(env_d))
    for c in candidates:
        if (c / "meta.bin").is_file() and (c / "embed_lm_head.hg4").is_file() and (c / "layer_23.hg4").is_file():
            return c
    return b / "data" / "omni" / "hg4"


def hg4_available(base_dir: str | Path) -> tuple[bool, str]:
    d = hg4_dir_for_base(base_dir)
    need = ["meta.bin", "embed_lm_head.hg4", "layer_00.hg4", "layer_23.hg4"]
    missing = [n for n in need if not (d / n).is_file()]
    if missing:
        return False, "missing data/omni/hg4 (" + ", ".join(missing) + ")"
    lib = get_hg4_lib(base_dir)
    if lib is None:
        return False, "missing liblfm3_hg4_engine native library"
    return True, ""


def _bind_hg4_cdll(path_or_name: str | Path) -> ctypes.CDLL:
    if isinstance(path_or_name, Path):
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path_or_name.parent))
        lib = ctypes.CDLL(str(path_or_name))
    else:
        lib = ctypes.CDLL(path_or_name)

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_flt_p = ctypes.POINTER(ctypes.c_float)
    c_dbl_p = ctypes.POINTER(ctypes.c_double)

    lib.lfm3_hg4_isa_name.argtypes = []
    lib.lfm3_hg4_isa_name.restype = ctypes.c_char_p
    lib.lfm3_hg4_init.argtypes = [ctypes.c_char_p]
    lib.lfm3_hg4_init.restype = ctypes.c_int
    lib.lfm3_hg4_set_num_threads.argtypes = [ctypes.c_int]
    lib.lfm3_hg4_set_num_threads.restype = None
    lib.lfm3_hg4_get_num_threads.argtypes = []
    lib.lfm3_hg4_get_num_threads.restype = ctypes.c_int
    lib.lfm3_hg4_cache_init.argtypes = [ctypes.c_int]
    lib.lfm3_hg4_cache_init.restype = ctypes.c_int
    lib.lfm3_hg4_cache_truncate.argtypes = [ctypes.c_int]
    lib.lfm3_hg4_cache_truncate.restype = None
    lib.lfm3_hg4_cache_get_len.argtypes = []
    lib.lfm3_hg4_cache_get_len.restype = ctypes.c_int
    lib.lfm3_hg4_cache_feed.argtypes = [c_int_p, ctypes.c_int, c_flt_p, c_flt_p]
    lib.lfm3_hg4_cache_feed.restype = ctypes.c_int
    lib.lfm3_hg4_cache_feed_embeddings.argtypes = [c_flt_p, ctypes.c_int, c_flt_p, c_flt_p]
    lib.lfm3_hg4_cache_feed_embeddings.restype = ctypes.c_int
    lib.lfm3_hg4_hidden_states.argtypes = [c_int_p, ctypes.c_int, ctypes.c_int, c_flt_p]
    lib.lfm3_hg4_hidden_states.restype = ctypes.c_int
    lib.lfm3_hg4_generate.argtypes = [
        c_int_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_int_p, c_flt_p, c_dbl_p, c_dbl_p,
    ]
    lib.lfm3_hg4_generate.restype = ctypes.c_int
    return lib


def get_hg4_lib(base_dir: str | Path | None = None) -> ctypes.CDLL | None:
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    with _LIB_LOCK:
        if _LIB_CACHE is not None:
            return _LIB_CACHE
        sys_name = platform.system()
        lib_name = (
            "liblfm3_hg4_engine.dll" if sys_name == "Windows"
            else ("liblfm3_hg4_engine.dylib" if sys_name == "Darwin" else "liblfm3_hg4_engine.so")
        )
        search_paths: list[Path] = [HERE / "fused" / lib_name]
        if base_dir is not None:
            b = Path(base_dir)
            search_paths.extend([
                b / "engine" / lib_name,
                b / "data" / "omni" / "hg4" / lib_name,
            ])
        search_paths.append(Path("/opt/clover-k3/lfm3-omni-hg4") / lib_name)
        for env_k in ("CLOVER_NATIVE_LIB_DIR", "LD_LIBRARY_PATH"):
            for d_str in (os.environ.get(env_k) or "").split(os.pathsep):
                if d_str.strip():
                    search_paths.append(Path(d_str.strip()) / lib_name)
        for cand in search_paths:
            if cand.is_file():
                try:
                    _LIB_CACHE = _bind_hg4_cdll(cand)
                    return _LIB_CACHE
                except OSError:
                    continue
        if sys_name != "Windows":
            try:
                _LIB_CACHE = _bind_hg4_cdll(lib_name)
                return _LIB_CACHE
            except OSError:
                pass
        # Auto-compile fallback if C compiler and source are present on dev/server host
        csrc = next((p for p in CSRC_CANDIDATES if p.is_file()), None)
        if csrc is not None:
            cache_root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            out_dir = cache_root / "clover-hg4" / f"{sys_name.lower()}-{platform.machine().lower()}"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_so = out_dir / lib_name
                if not out_so.is_file():
                    cc = "gcc"
                    cmd = [cc, "-O3", "-std=c11", "-fopenmp", "-shared"]
                    if sys_name != "Windows":
                        cmd.append("-fPIC")
                    cmd += [str(csrc), "-lm", "-o", str(out_so)]
                    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                _LIB_CACHE = _bind_hg4_cdll(out_so)
                return _LIB_CACHE
            except Exception:
                pass
        return None


class _TokenizerAdapter:
    """Wraps either HuggingFace Rust `tokenizers.Tokenizer` (fast C++) or `omni_c2_c3.Lfm3PureBpeTokenizer` (Android/pure Python)."""

    def __init__(self, tok_json: Path):
        self.path = Path(tok_json)
        self._fast = None
        self._pure = None
        try:
            from tokenizers import Tokenizer
            self._fast = Tokenizer.from_file(str(self.path))
            self.im_end_id = self._fast.token_to_id("<|im_end|>") or 7
            self.eos_id = self._fast.token_to_id("<|endoftext|>") or 2
            self.tool_resp_id = self._fast.token_to_id("<tool_response>") or -1
        except Exception:
            import omni_c2_c3
            self._pure = omni_c2_c3.Lfm3PureBpeTokenizer(self.path, vocab_size=V_OMNI)
            self.im_end_id = self._pure.added_tokens_map.get("<|im_end|>", 7)
            self.eos_id = self._pure.added_tokens_map.get("<|endoftext|>", 2)
            self.tool_resp_id = self._pure.added_tokens_map.get("<tool_response>", -1)

    def encode(self, text: str) -> list[int]:
        if self._fast is not None:
            return self._fast.encode(text, add_special_tokens=False).ids
        return self._pure.encode(text, allow_special=True)

    def decode(self, ids: list[int]) -> str:
        if self._fast is not None:
            return self._fast.decode(ids, skip_special_tokens=False)
        return self._pure.decode(ids, skip_special_tokens=False)


def _find_tokenizer_json(base_dir: Path, hg4_dir: Path) -> Path:
    candidates = [
        hg4_dir / "tokenizer.json",
        base_dir / "data" / "omni" / "precomputed" / "trunks" / "tokenizer.json",
        base_dir / "data" / "lfm3" / "hf" / "tokenizer.json",
        Path("/opt/clover-k3/web-studio/data/omni/precomputed/trunks/tokenizer.json"),
        Path("/opt/clover-k3/lfm3-omni-2b-hf/tokenizer.json"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("missing LFM3-Omni-2B tokenizer.json")


def _render_arg_val(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _parse_msg_content(content: Any, *, include_media_tags: bool = False) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("type")
        if t in ("image", "image_url"):
            return "<|vision_start|><|image|><|vision_end|>" if include_media_tags else ""
        if t in ("video", "input_video"):
            return "<|vision_start|><|video|><|vision_end|>" if include_media_tags else ""
        if t in ("audio", "audio_url", "input_audio"):
            return "<|vision_start|><|audio|><|vision_end|>" if include_media_tags else ""
        if t == "text":
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, (list, tuple)):
        return "".join(_parse_msg_content(item, include_media_tags=include_media_tags) for item in content)
    return str(content)


def _tool_to_json(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    clean = {k: v for k, v in fn.items() if k != "defer_loading"}
    return json.dumps(clean, ensure_ascii=False)


def render_lfm3_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Exact Python implementation of `LiquidAI/LFM3-Omni-2B`'s `chat_template.jinja` (`enable_thinking=False`)."""
    msgs = list(messages or ())
    sys_prompt = ""
    if msgs and msgs[0].get("role") == "system":
        if msgs[0].get("content"):
            sys_prompt = _parse_msg_content(msgs[0]["content"])
        msgs = msgs[1:]

    if tools:
        tool_lines = []
        for t in tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if fn.get("defer_loading") or t.get("defer_loading"):
                continue
            tool_lines.append(_tool_to_json(fn))
        if tool_lines:
            sys_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + "<tools>\n" + "\n".join(tool_lines) + "\n</tools>"

    out = ["<|startoftext|>"]
    if sys_prompt:
        out.append(f"<|im_start|>system\n{sys_prompt}<|im_end|>\n")

    for m in msgs:
        role = str(m.get("role") or "user")
        out.append(f"<|im_start|>{role}\n")
        if role == "assistant":
            content = _parse_msg_content(m.get("content")) if m.get("content") else ""
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            out.append("<think></think>" + content)
            tcs = m.get("tool_calls") or ()
            for idx, tc in enumerate(tcs):
                if idx > 0:
                    out.append("\n\n")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                fn_name = str(fn.get("name") or "")
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args_map = json.loads(raw_args) if raw_args.strip() else {}
                    except ValueError:
                        args_map = {}
                elif isinstance(raw_args, dict):
                    args_map = raw_args
                else:
                    args_map = {}
                out.append(f"<tool_call>\n<function={fn_name}>\n")
                if isinstance(args_map, dict):
                    for ak, av in args_map.items():
                        out.append(f"<parameter={ak}>\n{_render_arg_val(av)}\n</parameter>\n")
                out.append("</function>\n</tool_call>")
            out.append("<|im_end|>\n")
        elif role == "tool":
            c_raw = m.get("content") or ""
            if isinstance(c_raw, str) and "<tool_response>" in c_raw:
                out.append(c_raw)
            else:
                out.append(f"<tool_response>\n{_parse_msg_content(c_raw)}\n</tool_response>")
            out.append("<|im_end|>\n")
        else:
            if m.get("content"):
                out.append(_parse_msg_content(m["content"]))
            out.append("<|im_end|>\n")

    if add_generation_prompt:
        out.append("<|im_start|>assistant\n<think></think>")
    return "".join(out)


def _sample_hg4_token(
    logits: np.ndarray,
    history_ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    rng: np.random.Generator,
) -> int:
    if repeat_penalty == 1.0 and temperature <= 0.0:
        return int(np.argmax(logits))
    lg = logits.astype(np.float64, copy=True)
    if repeat_penalty != 1.0 and history_ids:
        idx = np.fromiter({i for i in history_ids if 0 <= i < V_OMNI}, dtype=np.int64)
        if idx.size > 0:
            vals = lg[idx]
            lg[idx] = np.where(vals > 0, vals / repeat_penalty, vals * repeat_penalty)
    if temperature <= 0.0:
        return int(np.argmax(lg))
    lg /= temperature
    if 0 < top_k < V_OMNI:
        kth = np.partition(lg, -top_k)[-top_k]
        lg[lg < kth] = -np.inf
    lg -= np.max(lg)
    p = np.exp(lg)
    p /= np.sum(p)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p /= np.sum(p)
    return int(rng.choice(V_OMNI, p=p))


def _has_raw_media_part(messages: list[dict[str, Any]]) -> bool:
    for m in messages or ():
        c = m.get("content")
        if isinstance(c, (list, tuple)):
            for p in c:
                if isinstance(p, dict) and p.get("type") in (
                    "image_url", "image", "input_audio", "audio", "audio_url", "input_video", "video"
                ):
                    return True
    return False


class Hg4Server:
    """In-process C11 `HG4-128` backend implementing the `LlamaServer` contract used by `OmniNative` & `server.py`."""

    proc = None
    kind = "hg4"

    def __init__(
        self,
        base_dir: str | Path,
        max_tokens: int = 8192,
        threads: int | None = None,
        mmproj_paths: dict[str, Path] | None = None,
        media_dir: Path | None = None,
        voice_threads: int = 8,
    ):
        self.base_dir = Path(base_dir)
        self.hg4_dir = hg4_dir_for_base(self.base_dir)
        ok, why = hg4_available(self.base_dir)
        if not ok:
            raise FileNotFoundError(why)
        self.lib = get_hg4_lib(self.base_dir)
        assert self.lib is not None
        self._lock = threading.Lock()

        eff_ctx = max(2048, min(int(max_tokens or 8192), 16384))
        self.ctx = eff_ctx
        if threads:
            n_thr = int(threads)
        else:
            try:
                import hardware
                n_thr = int(hardware.cpu().get("cores") or 0) or max(2, min(16, (os.cpu_count() or 8)))
            except Exception:
                n_thr = max(2, min(16, (os.cpu_count() or 8)))
        n_thr = max(2, min(16, n_thr))
        self.lib.lfm3_hg4_set_num_threads(n_thr)
        rc = self.lib.lfm3_hg4_init(str(self.hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"lfm3_hg4_init({self.hg4_dir}) failed rc={rc}")
        rc_c = self.lib.lfm3_hg4_cache_init(eff_ctx)
        if rc_c != 0:
            raise RuntimeError(f"lfm3_hg4_cache_init({eff_ctx}) failed rc={rc_c}")

        self.isa_name = self.lib.lfm3_hg4_isa_name().decode("utf-8", errors="replace")
        self.threads = self.lib.lfm3_hg4_get_num_threads()
        self.device = f"CPU · {self.isa_name.split('(')[0].strip()} ({self.threads} threads)"
        self.model = self.hg4_dir

        tok_p = _find_tokenizer_json(self.base_dir, self.hg4_dir)
        self.tok = _TokenizerAdapter(tok_p)
        self.stop_ids = {i for i in (self.tok.im_end_id, self.tok.eos_id, self.tok.tool_resp_id) if i >= 0}

        self._cached_ids: list[int] = []
        self._logits_buf = np.zeros(V_OMNI, dtype=np.float32)
        self._hidden_buf = np.zeros(D_OMNI, dtype=np.float32)

        # Optional lazy GGUF+mmproj helper ONLY for raw audio ASR (`transcribe`) or `/tts` speech synthesis when installed
        self._mmproj_paths = mmproj_paths if (
            mmproj_paths and mmproj_paths.get("model") and Path(mmproj_paths["model"]).is_file()
            and mmproj_paths.get("mmproj") and Path(mmproj_paths["mmproj"]).is_file()
        ) else None
        self._media_dir = media_dir or (self.base_dir / "media")
        self._voice_threads = int(voice_threads or 8)
        self._aux_srv = None
        self._aux_lock = threading.Lock()

    def props(self) -> dict[str, Any]:
        # S3 sensory inspection handles vision/audio/video across all platforms; raw mmproj adds direct ASR/TTS too
        return {"modalities": {"vision": True, "audio": True}}

    def apply_template(self, messages: list[dict[str, Any]]) -> str:
        return render_lfm3_chat(messages, tools=None, add_generation_prompt=True)

    def tokenize(self, text: str) -> list[int]:
        return self.tok.encode(str(text or ""))

    def hidden_states(self, ids: list[int]) -> np.ndarray:
        """Compute exact 24-layer post-final_norm `[len(ids), 2048]` hidden states in-process with prefix reuse."""
        seq = [int(x) for x in ids]
        n = len(seq)
        if n == 0:
            return np.zeros((0, D_OMNI), dtype=np.float32)
        with self._lock:
            if n > self.ctx:
                seq = seq[-self.ctx:]
                n = len(seq)
            out = np.empty((n, D_OMNI), dtype=np.float32)
            c_arr = (ctypes.c_int * n)(*seq)
            # Invalidate chat token cache since hidden_states writes positions [0..n)
            self._cached_ids.clear()
            rc = self.lib.lfm3_hg4_hidden_states(
                c_arr, n, 0, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            if rc < 0:
                raise RuntimeError(f"lfm3_hg4_hidden_states failed rc={rc}")
            self._cached_ids = list(seq)
            return out

    def _get_aux_llama(self):
        if self._mmproj_paths is None:
            return None
        with self._aux_lock:
            if self._aux_srv is not None and self._aux_srv.proc.poll() is not None:
                self._aux_srv = None
            if self._aux_srv is None:
                import llamacpp_engine as L
                extra = ["--reasoning", "off", "--media-path", str(self._media_dir), "-t", str(self._voice_threads)]
                self._aux_srv = L.LlamaServer(
                    self.base_dir, "cpu", 8192,
                    model=self._mmproj_paths["model"], mmproj=self._mmproj_paths["mmproj"],
                    name="omni-hg4-aux", timeout_s=120, extra_args=extra,
                )
            return self._aux_srv

    def tts(self, text: str, voice: str = "female", lang: str = "en"):
        aux = self._get_aux_llama()
        if aux is None:
            raise RuntimeError("Speech synthesis requires the optional LFM3-Omni GGUF speech projector.")
        return aux.tts(text, voice=voice, lang=lang)

    def chat_stream(self, body: dict[str, Any], cancel: threading.Event | None = None):
        """OpenAI `/v1/chat/completions`-shaped streaming generator backed by `liblfm3_hg4_engine`."""
        msgs = list(body.get("messages") or ())
        # Multimodal turns with raw image/audio/video attachments use the SigLIP2/Conformer mmproj helper when installed
        if _has_raw_media_part(msgs):
            aux = self._get_aux_llama()
            if aux is not None:
                yield from aux.chat_stream(body, cancel)
                return

        tools = body.get("tools") or None
        tool_choice = body.get("tool_choice")
        max_new = max(1, int(body.get("max_tokens") or 1024))
        temp = float(body.get("temperature") or 0.0)
        top_k = int(body.get("top_k") or 0)
        top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        rep_pen = float(body.get("repeat_penalty") if body.get("repeat_penalty") is not None else 1.0)
        seed = int(body.get("seed") or 0)
        cache_prompt = bool(body.get("cache_prompt", True))

        active_fns = []
        if tools:
            for t in tools:
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                if not (fn.get("defer_loading") or t.get("defer_loading")) and fn.get("name"):
                    active_fns.append(fn)

        forced_prefix = ""
        if tool_choice == "required" and active_fns:
            if len(active_fns) == 1:
                forced_prefix = f"<tool_call>\n<function={active_fns[0]['name']}>\n"
            else:
                forced_prefix = "<tool_call>\n<function="

        # Trim oldest conversation turns if prompt exceeds context window
        work_msgs = list(msgs)
        while True:
            prompt_str = render_lfm3_chat(work_msgs, tools=tools, add_generation_prompt=True) + forced_prefix
            ids = self.tok.encode(prompt_str)
            if len(ids) + min(max_new, 512) < self.ctx or len(work_msgs) <= 2:
                break
            del work_msgs[1]
        if len(ids) >= self.ctx - 16:
            ids = ids[-(self.ctx - max(64, min(max_new, 256)) - 1):]
        eff_max_new = min(max_new, max(1, self.ctx - len(ids) - 1))

        c_flt_p = ctypes.POINTER(ctypes.c_float)
        with self._lock:
            if cancel is not None and cancel.is_set():
                yield {"cancelled": True}
                return

            t_pf0 = time.perf_counter()
            lcp = 0
            if cache_prompt and self._cached_ids:
                lim = min(len(self._cached_ids), len(ids) - 1)
                while lcp < lim and self._cached_ids[lcp] == ids[lcp]:
                    lcp += 1
            self.lib.lfm3_hg4_cache_truncate(lcp)
            del self._cached_ids[lcp:]

            feed_slice = ids[lcp:]
            c_feed = (ctypes.c_int * len(feed_slice))(*feed_slice)
            rc = self.lib.lfm3_hg4_cache_feed(
                c_feed, len(feed_slice),
                self._hidden_buf.ctypes.data_as(c_flt_p),
                self._logits_buf.ctypes.data_as(c_flt_p),
            )
            if rc < 0:
                raise RuntimeError(f"lfm3_hg4_cache_feed failed rc={rc}")
            self._cached_ids.extend(feed_slice)
            prompt_ms = (time.perf_counter() - t_pf0) * 1000.0

            if forced_prefix:
                yield {"choices": [{"index": 0, "delta": {"content": forced_prefix}, "finish_reason": None}]}

            rng = np.random.default_rng(seed)
            history_ids = list(ids)
            gen_ids: list[int] = []
            emitted_len = 0
            finish_reason = "length"
            t_dc0 = time.perf_counter()

            for step in range(eff_max_new):
                if cancel is not None and cancel.is_set():
                    yield {"cancelled": True}
                    return

                nxt = _sample_hg4_token(
                    self._logits_buf, history_ids, temp, top_k, top_p, rep_pen, rng
                )
                if nxt in self.stop_ids:
                    finish_reason = "stop"
                    break

                gen_ids.append(nxt)
                history_ids.append(nxt)
                decoded = self.tok.decode(gen_ids)
                full_now = forced_prefix + decoded

                if "<|im_end|>" in decoded or "<|endoftext|>" in decoded:
                    cut_i = min(
                        (decoded.find(tag) for tag in ("<|im_end|>", "<|endoftext|>") if tag in decoded),
                        default=len(decoded),
                    )
                    clean_dec = decoded[:cut_i]
                    if len(clean_dec) > emitted_len:
                        yield {"choices": [{"index": 0, "delta": {"content": clean_dec[emitted_len:]}, "finish_reason": None}]}
                    finish_reason = "stop"
                    break

                if "<tool_response>" in full_now:
                    cut_tr = decoded.find("<tool_response>")
                    if cut_tr >= 0 and cut_tr > emitted_len:
                        yield {"choices": [{"index": 0, "delta": {"content": decoded[emitted_len:cut_tr]}, "finish_reason": None}]}
                    finish_reason = "stop"
                    break

                if not decoded.endswith("\ufffd") and len(decoded) > emitted_len:
                    delta = decoded[emitted_len:]
                    emitted_len = len(decoded)
                    ev_out: dict[str, Any] = {"choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    if "</tool_call>" in full_now:
                        ev_out["timings"] = {
                            "prompt_n": len(feed_slice),
                            "cache_n": lcp,
                            "prompt_ms": round(prompt_ms, 2),
                            "predicted_n": len(gen_ids),
                            "predicted_ms": round((time.perf_counter() - t_dc0) * 1000.0, 2),
                        }
                    yield ev_out

                if "</tool_call>" in full_now:
                    finish_reason = "tool_calls"
                    c_one = (ctypes.c_int * 1)(nxt)
                    if len(self._cached_ids) + 1 <= self.ctx:
                        self.lib.lfm3_hg4_cache_feed(
                            c_one, 1,
                            self._hidden_buf.ctypes.data_as(c_flt_p),
                            self._logits_buf.ctypes.data_as(c_flt_p),
                        )
                        self._cached_ids.append(nxt)
                    break

                if step + 1 < eff_max_new:
                    c_one = (ctypes.c_int * 1)(nxt)
                    rc_s = self.lib.lfm3_hg4_cache_feed(
                        c_one, 1,
                        self._hidden_buf.ctypes.data_as(c_flt_p),
                        self._logits_buf.ctypes.data_as(c_flt_p),
                    )
                    if rc_s < 0:
                        finish_reason = "length"
                        break
                    self._cached_ids.append(nxt)

            pred_ms = (time.perf_counter() - t_dc0) * 1000.0
            yield {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "timings": {
                    "prompt_n": len(feed_slice),
                    "cache_n": lcp,
                    "prompt_ms": round(prompt_ms, 2),
                    "predicted_n": len(gen_ids),
                    "predicted_ms": round(pred_ms, 2),
                },
            }

    def close(self) -> None:
        with self._aux_lock:
            if self._aux_srv is not None:
                self._aux_srv.close()
                self._aux_srv = None
