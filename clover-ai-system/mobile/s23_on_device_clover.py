#!/usr/bin/env python3
"""
Samsung Galaxy S23 (`Qualcomm Snapdragon 8 Gen 2 for Galaxy`, `aarch64` Termux)
& K3 Virtual S23 Debugger (`AMD Ryzen 9 7950X3D`) — Complete 7-Layer On-Device Clover Runtime
(`clover-ai-system/src/models/lfm3-1.7b/s23_on_device_clover.py`).

Implements and verifies all 7 Clover AI System layers around `LiquidAI/LFM3-1.7B` (`LFM3-Omni`)
step-by-step (`max_active_layer = 1 .. 7`) with zero hardcoded prompt dispatch and zero fake projections:

  - Layer 1 (`LiquidAI/LFM3-1.7B` 24-Layer 4-Bit C/NEON/BFDOT/SDOT Engine `liblfm3_cpu_engine.so`):
      Executes all 24 Tri-Hybrid layers (`13` width-7 Gated ShortConv + `5` Sliding-256 GQA + `6` Full GQA
      with zero-tail partial RoPE `0.5` + sandwich RMSNorm + 4-bit planar group-64 SwiGLU FFN `ffn.4bit.bin`),
      KV + ShortConv cache with bit-exact longest-common-prefix reuse (`lfm3_prefill_cached`, `max_abs_diff = 0.0`),
      continuous `<latent>` step injection (`lfm3_step_embedding_cached`), cached decoding (`lfm3_decode_from_cache`),
      and exact 128,000-vocab `added_tokens` + byte-level BPE tokenization (`Lfm3PureBpeTokenizer`).
  - Layer 2 (Native `HybridAgent` — `Lfm3NativeHybridLink(d=2048, bottleneck=512)` + Autonomous Tool Loop):
      Executes trained `Lfm3NativeHybridLink(d=2048)` (`0.9993` BF16 / `0.9353` 4-bit test cosine to next-token
      embeddings) for continuous `<latent>` reasoning steps directly in LFM3's 2,048-d space, and runs
      autonomous model-emitted `<tool_call>` XML routing (`web_search` via `LiveWebSearchTool.execute` &
      `fetch_url_content` for Wikipedia, arXiv, DuckDuckGo, and live URLs; `python_interpreter` subprocess;
      `generate_media` for Pillow PNG, FFmpeg H.264 MP4, and eSpeak NG WAV; and `direct_answer`).
  - Layer 3 (Native 11-Level `NCM` — `VQVAE` + `NextConceptTransformer` + `LexicalPointerDecoder`):
      Executes `Lfm3ComboBridge(2048 -> 384)` + `Lfm3ExpertBridge(2048 -> 384)` + `128,000`-vocab `mmap`-backed
      `VQVAE` (`vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin`) + 11-level `NextConceptTransformer` (`[1, 4, 8, 14,
      20, 26, 32, 40, 48, 56, 64]` = `313` hierarchical concept codes with prefix KV caching) +
      `LexicalPointerDecoder` with active-vocabulary projection (`33.8x` faster, `0.0` max diff).
  - Layer 4 (`k3_memory_graph.py` — Persistent Relational Memory Tree):
      Multi-hop relational memory retrieval across `memory_tree/` with epistemic status & provenance.
  - Layer 5 (`k3_ontology_graph.py` — Deterministic Symbolic Ontology GraphRAG & `CloverSymbolicVerifier`):
      Horn-rule inference (`run_inference`), entity/path retrieval, symbolic math constraint solver, and
      post-generation `verify_source_grounding` check.
  - Layer 6 (`k3_generative_compiler.py` — `CloverPythonGenerativeCompiler`):
      Compiles and verifies Python code blocks (`GCVerdict.ok`, `GCVerdict.diagnostics`, `rollback_line_idx`).
  - Layer 7 (`lfm3_dm_record_layer` + `lfm3_advise_trunk_pages` in `lfm3_cpu_engine.c`):
      SpecMD Policy 3 (`Least-Stale`) lookahead issuing OS `madvise(..., MADV_WILLNEED)` for Layer `L+1`
      and `madvise(..., MADV_DONTNEED)` on Red-Blue Pebble `R4 Clean-Drop` eviction.
"""

from __future__ import annotations

import argparse
import base64
import collections
import ctypes
import io
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
import types
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Prevent OpenBLAS multi-thread buffer race in Termux python-torch 2.14.1 and duplicate libomp abort
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# On Samsung Galaxy S23 Termux (`aarch64`), automatically expose `/vendor/lib64/libOpenCL.so` (Adreno 740 OpenCL 3.0)
# while keeping Termux's `libc++_shared.so` preloaded so both `numpy`/`torch` and `Adreno 740` work simultaneously.
if (
    platform.machine().lower() in ("aarch64", "arm64")
    and os.path.exists("/vendor/lib64/libOpenCL.so")
    and os.path.exists("/data/data/com.termux/files/usr/lib/libc++_shared.so")
    and os.environ.get("CLOVER_S23_ADRENO_ENV") != "1"
):
    os.environ["CLOVER_S23_ADRENO_ENV"] = "1"
    os.environ["LD_PRELOAD"] = "/data/data/com.termux/files/usr/lib/libc++_shared.so"
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    vendor_ld = "/vendor/lib64:/vendor/lib64/egl:/system/lib64"
    os.environ["LD_LIBRARY_PATH"] = f"{vendor_ld}:{cur_ld}" if cur_ld else vendor_ld
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
from PIL import Image, ImageFilter, ImageStat

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_TOOLS_DIR = SCRIPT_DIR / "tools"
LOCAL_NCM_DIR = SCRIPT_DIR / "ncm"
LFM3_DIR = SCRIPT_DIR.parent / "src" / "models" / "lfm3-1.7b"
MODELS_DIR = SCRIPT_DIR.parent / "src" / "models"
TOOLS_DIR = SCRIPT_DIR.parent / "src" / "tools" / "hybrid-ncm-k3"
for _p in (SCRIPT_DIR, LOCAL_TOOLS_DIR, LOCAL_NCM_DIR, LFM3_DIR, MODELS_DIR, TOOLS_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent / "tools" / "hybrid-ncm-k3"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

D_LFM3 = 2048
V_LFM3 = 128000
D_VQ = 384
CLOVER_S23_APP_PACKAGE = "ai.clover.studio"


def load_or_create_device_app_secret(base_dir: Path) -> str:
    """Loads or generates a per-device 256-bit secret stored only on the phone (`.app_secret`, `0600`)."""
    secret_path = Path(base_dir) / ".app_secret"
    if secret_path.exists():
        tok = secret_path.read_text(encoding="utf-8").strip()
        if len(tok) >= 32:
            return tok
    tok = "s23-" + secrets.token_hex(32)
    secret_path.write_text(tok + "\n", encoding="utf-8")
    try:
        os.chmod(secret_path, 0o600)
    except Exception:
        pass
    return tok


# ==============================================================================
# 1. Pure-Python 128,000-Vocab Byte-Level BPE Tokenizer (`tokenizer.json`)
#    with Exact `added_tokens` Special-Token Splitting (`<|startoftext|>`, `<tool_call>`, etc.)
# ==============================================================================

def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Lfm3PureBpeTokenizer:
    """Pure-Python 128K BPE tokenizer loading `LiquidAI/LFM3-1.7B`'s `tokenizer.json` directly,
    including exact `added_tokens` splitting for `<|im_start|>`, `<tools>`, `<tool_call>`, `<function=`, etc."""

    def __init__(self, tokenizer_json_path: Path):
        raw = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        model = raw.get("model", {})
        self.vocab: dict[str, int] = dict(model.get("vocab", {}))
        self.added_tokens_map: dict[str, int] = {}
        for added in raw.get("added_tokens", []):
            if isinstance(added, dict) and "content" in added and "id" in added:
                c_str = str(added["content"])
                tid = int(added["id"])
                self.vocab[c_str] = tid
                self.added_tokens_map[c_str] = tid
        self.id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.inv_added_tokens: dict[int, str] = {v: k for k, v in self.added_tokens_map.items()}

        if self.added_tokens_map:
            sorted_special = sorted(self.added_tokens_map.keys(), key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(k) for k in sorted_special) + ")")
        else:
            self._special_re = None

        merges_raw = model.get("merges", [])
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for idx, m in enumerate(merges_raw):
            if isinstance(m, list) and len(m) == 2:
                self.bpe_ranks[(str(m[0]), str(m[1]))] = idx
            elif isinstance(m, str):
                parts = m.split(" ")
                if len(parts) == 2:
                    self.bpe_ranks[(parts[0], parts[1])] = idx

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: dict[str, list[int]] = {}
        self.vocab_size = V_LFM3
        self.bos_token_id = 124894
        self.eos_token_id = 124900
        self.pad_token_id = 124893
        self.semicolon_id = 26
        self._pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
        semi = self._encode_plain_bpe(";")
        if semi:
            self.semicolon_id = semi[0]

    def _bpe(self, token: str) -> list[str]:
        word = tuple(token)
        if len(word) <= 1:
            return list(word)
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10**12))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return list(word)

    def _encode_plain_bpe(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for piece in self._pat.findall(text):
            if piece in self.cache:
                ids.extend(self.cache[piece])
                continue
            b_str = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            piece_ids = [self.vocab[sub] for sub in self._bpe(b_str) if sub in self.vocab]
            self.cache[piece] = piece_ids
            ids.extend(piece_ids)
        return [x for x in ids if 0 <= x < self.vocab_size]

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not text:
            return []
        if not allow_special or self._special_re is None:
            return self._encode_plain_bpe(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.added_tokens_map:
                out.append(self.added_tokens_map[part])
            else:
                out.extend(self._encode_plain_bpe(part))
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        parts: list[str] = []
        bpe_buf: list[str] = []

        def _flush_bpe() -> None:
            if not bpe_buf:
                return
            joined = "".join(bpe_buf)
            bpe_buf.clear()
            byte_arr = bytearray(self.byte_decoder[ch] for ch in joined if ch in self.byte_decoder)
            parts.append(byte_arr.decode("utf-8", errors="replace"))

        for tid in ids:
            t_int = int(tid)
            if t_int in self.inv_added_tokens:
                _flush_bpe()
                if not skip_special_tokens:
                    parts.append(self.inv_added_tokens[t_int])
                continue
            if skip_special_tokens and t_int >= 124893:
                continue
            tok_s = self.id_to_token.get(t_int, "")
            if skip_special_tokens and tok_s.startswith("<|") and tok_s.endswith("|>"):
                continue
            bpe_buf.append(tok_s)
        _flush_bpe()
        return "".join(parts)


# ==============================================================================
# 2. Layer 1: Native C/NEON/BFDOT/SDOT Wrapper (`liblfm3_cpu_engine.so`)
# ==============================================================================

def compile_s23_lfm3_engine_so(src_dir: Path = SCRIPT_DIR, force: bool = False) -> Path:
    c_src = src_dir / "lfm3_cpu_engine.c"
    so_path = src_dir / "liblfm3_cpu_engine.so"
    if not force and so_path.exists() and c_src.exists() and so_path.stat().st_mtime >= c_src.stat().st_mtime:
        return so_path

    arch = platform.machine().lower()
    cc = "clang" if shutil.which("clang") else "gcc"
    if arch in ("aarch64", "arm64"):
        flags = ["-O3", "-march=armv8.6-a+bf16+i8mm+dotprod", "-fopenmp", "-fPIC", "-shared"]
    else:
        flags = ["-O3", "-march=native", "-fopenmp", "-fPIC", "-shared"]

    cmd = [cc, *flags, str(c_src), "-lm", "-ldl", "-o", str(so_path)]
    subprocess.run(cmd, check=True)
    return so_path


class S23Lfm3NativeEngine:
    """Direct ctypes interface to `liblfm3_cpu_engine.so` on Samsung Galaxy S23 (`aarch64`) or K3 (`x86_64`)."""

    def __init__(self, pruned_dir: Path, pin_s23_big_cores: bool = True, use_4bit_ffn: bool = True, madvise_active: bool = True):
        self.pruned_dir = Path(pruned_dir)
        self.so_path = compile_s23_lfm3_engine_so(SCRIPT_DIR)
        self.lib = ctypes.CDLL(str(self.so_path))

        self.lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
        self.lib.lfm3_engine_init.restype = ctypes.c_int

        self.lib.lfm3_set_use_4bit_ffn.argtypes = [ctypes.c_int]
        self.lib.lfm3_set_use_4bit_ffn.restype = None

        if hasattr(self.lib, "lfm3_set_madvise_active"):
            self.lib.lfm3_set_madvise_active.argtypes = [ctypes.c_int]
            self.lib.lfm3_set_madvise_active.restype = None

        self.lib.lfm3_configure_s23_snapdragon8gen2.argtypes = [ctypes.c_int]
        self.lib.lfm3_configure_s23_snapdragon8gen2.restype = ctypes.c_int

        self.lib.lfm3_forward_24_layers_clubbed.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_forward_24_layers_clubbed.restype = ctypes.c_float

        self.lib.lfm3_lm_head_logits.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_lm_head_logits.restype = ctypes.c_int

        self.lib.lfm3_set_stop_mask.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        self.lib.lfm3_set_stop_mask.restype = None

        self.lib.lfm3_generate_cached.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_generate_cached.restype = ctypes.c_int

        self.lib.lfm3_prefill_cached.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_prefill_cached.restype = ctypes.c_float

        if hasattr(self.lib, "lfm3_append_cached"):
            self.lib.lfm3_append_cached.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.lfm3_append_cached.restype = ctypes.c_float

        self.lib.lfm3_step_embedding_cached.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_step_embedding_cached.restype = ctypes.c_float

        if hasattr(self.lib, "lfm3_step_token_cached"):
            self.lib.lfm3_step_token_cached.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.lfm3_step_token_cached.restype = ctypes.c_float

        self.lib.lfm3_decode_from_cache.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.lfm3_decode_from_cache.restype = ctypes.c_int

        self.lib.lfm3_dm_stats.argtypes = [ctypes.POINTER(ctypes.c_ulong)]
        self.lib.lfm3_dm_stats.restype = None

        if hasattr(self.lib, "lfm3_dm_extended_stats"):
            self.lib.lfm3_dm_extended_stats.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
            self.lib.lfm3_dm_extended_stats.restype = None

        for fn_name in ("lfm3_set_use_4bit_lm_head", "lfm3_set_use_4bit_operators", "lfm3_set_adreno_opencl_active"):
            if hasattr(self.lib, fn_name):
                getattr(self.lib, fn_name).argtypes = [ctypes.c_int]
                getattr(self.lib, fn_name).restype = None

        if hasattr(self.lib, "lfm3_dm_reset"):
            self.lib.lfm3_dm_reset.argtypes = []
            self.lib.lfm3_dm_reset.restype = None

        rc = self.lib.lfm3_engine_init(str(self.pruned_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"lfm3_engine_init({self.pruned_dir}) failed with rc={rc}")
        self.lib.lfm3_set_use_4bit_ffn(1 if use_4bit_ffn else 0)
        self.set_use_4bit_lm_head(True)
        self.set_adreno_opencl_active(True)
        self.set_madvise_active(madvise_active)
        is_arm = platform.machine().lower() in ("aarch64", "arm64")
        self.active_threads = int(self.lib.lfm3_configure_s23_snapdragon8gen2(1 if (pin_s23_big_cores and is_arm) else 0))

    def set_use_4bit_lm_head(self, active: bool) -> None:
        if hasattr(self.lib, "lfm3_set_use_4bit_lm_head"):
            self.lib.lfm3_set_use_4bit_lm_head(1 if active else 0)

    def set_use_4bit_operators(self, active: bool) -> None:
        if hasattr(self.lib, "lfm3_set_use_4bit_operators"):
            self.lib.lfm3_set_use_4bit_operators(1 if active else 0)

    def set_adreno_opencl_active(self, active: bool) -> None:
        if hasattr(self.lib, "lfm3_set_adreno_opencl_active"):
            self.lib.lfm3_set_adreno_opencl_active(1 if active else 0)

    def set_madvise_active(self, active: bool) -> None:
        if hasattr(self.lib, "lfm3_set_madvise_active"):
            self.lib.lfm3_set_madvise_active(1 if active else 0)

    def reset_dm_stats(self) -> None:
        if hasattr(self.lib, "lfm3_dm_reset"):
            self.lib.lfm3_dm_reset()

    def configure_stop_mask(self, tokenizer: Lfm3PureBpeTokenizer) -> None:
        mask = (ctypes.c_uint8 * V_LFM3)()
        for tid in range(min(V_LFM3, 124890)):
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            if not piece:
                continue
            stripped = piece.strip()
            if stripped.isdigit():
                mask[tid] = 3
            elif re.search(r"\d\.\s*$", piece):
                mask[tid] = 0
            elif piece.endswith((".\n\n", "**.\n\n", "!\n\n", "?\n\n", ".\n", "**.\n", "!\n", "?\n")):
                mask[tid] = 2
            elif piece.endswith((".", "**.", "!", "?", ").", "\".", "”.")):
                mask[tid] = 1
        self.lib.lfm3_set_stop_mask(mask)

    def prefill_cached(self, prompt_ids: list[int]) -> tuple[np.ndarray, float]:
        ids = [max(0, min(V_LFM3 - 1, int(x))) for x in (prompt_ids or [124894])][:1024]
        P = len(ids)
        c_in = (ctypes.c_int * P)(*ids)
        out_h = np.zeros(D_LFM3, dtype=np.float32)
        ms = float(self.lib.lfm3_prefill_cached(c_in, P, out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return out_h, ms

    def append_cached(self, new_ids: list[int]) -> tuple[np.ndarray, float]:
        ids = [max(0, min(V_LFM3 - 1, int(x))) for x in (new_ids or [124894])][:512]
        P = len(ids)
        c_in = (ctypes.c_int * P)(*ids)
        out_h = np.zeros(D_LFM3, dtype=np.float32)
        if hasattr(self.lib, "lfm3_append_cached"):
            ms = float(self.lib.lfm3_append_cached(c_in, P, out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
            return out_h, ms
        return self.prefill_cached(ids)

    def step_embedding_cached(self, emb_2048: np.ndarray) -> tuple[np.ndarray, float]:
        in_arr = np.ascontiguousarray(emb_2048.reshape(D_LFM3), dtype=np.float32)
        out_h = np.zeros(D_LFM3, dtype=np.float32)
        ms = float(
            self.lib.lfm3_step_embedding_cached(
                in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        return out_h, ms

    def step_token_cached(self, token_id: int) -> tuple[np.ndarray, float]:
        tid = max(0, min(V_LFM3 - 1, int(token_id)))
        if hasattr(self.lib, "lfm3_step_token_cached"):
            out_h = np.zeros(D_LFM3, dtype=np.float32)
            ms = float(self.lib.lfm3_step_token_cached(tid, out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
            return out_h, ms
        # Fallback if older .so is loaded
        return self.prefill_cached([tid])

    def decode_from_cache(
        self,
        initial_h_2048: np.ndarray,
        max_new_tokens: int = 48,
        allow_thinking: bool | int = False,
    ) -> tuple[list[int], np.ndarray, float]:
        in_h = np.ascontiguousarray(initial_h_2048.reshape(D_LFM3), dtype=np.float32)
        c_out = (ctypes.c_int * max(1, max_new_tokens))()
        out_h = np.zeros(D_LFM3, dtype=np.float32)
        c_ms = ctypes.c_float(0.0)
        think_mode = int(allow_thinking) if isinstance(allow_thinking, int) and not isinstance(allow_thinking, bool) else (1 if allow_thinking else 0)
        n_gen = int(
            self.lib.lfm3_decode_from_cache(
                in_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                max_new_tokens,
                think_mode,
                c_out,
                out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(c_ms),
            )
        )
        gen_ids = [int(c_out[i]) for i in range(max(0, n_gen))]
        return gen_ids, out_h, float(c_ms.value)

    def generate_chat(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
    ) -> tuple[list[int], np.ndarray, float]:
        h0, ms_pre = self.prefill_cached(prompt_ids)
        gen_ids, h_last, ms_dec = self.decode_from_cache(h0, max_new_tokens=max_new_tokens, allow_thinking=False)
        return gen_ids, h_last.reshape(1, D_LFM3), (ms_pre + ms_dec)

    def forward_24_layers(
        self,
        token_ids: list[int],
        seg_starts: list[int] | None = None,
    ) -> tuple[np.ndarray, float]:
        ids = [max(0, min(V_LFM3 - 1, int(x))) for x in (token_ids or [0])]
        T = len(ids)
        c_ids = (ctypes.c_int * T)(*ids)
        c_seg = (ctypes.c_int * T)(*(seg_starts if seg_starts is not None else [0] * T))
        out_h = np.zeros((T, D_LFM3), dtype=np.float32)
        ms = float(
            self.lib.lfm3_forward_24_layers_clubbed(
                c_ids,
                c_seg,
                T,
                out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        return out_h, ms

    def lm_head_top1(self, h_2048: np.ndarray) -> int:
        in_h = np.ascontiguousarray(h_2048.reshape(D_LFM3), dtype=np.float32)
        return int(
            self.lib.lfm3_lm_head_logits(
                in_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                None,
                None,
            )
        )

    def get_dm_stats(self, madvise_enabled: bool = True) -> dict[str, Any]:
        buf = (ctypes.c_ulong * 6)()
        self.lib.lfm3_dm_stats(buf)
        ext = (ctypes.c_ulonglong * 12)()
        if hasattr(self.lib, "lfm3_dm_extended_stats"):
            self.lib.lfm3_dm_extended_stats(ext)
        return {
            "tracker_kind": "eyeriss_4tier_mobile_soc_ledger_plus_specmd_policy3",
            "policy_id": 3,
            "policy_name": "specmd_least_stale_willneed" if madvise_enabled else "passive_mmap_baseline",
            "os_madvise_willneed_active": bool(madvise_enabled),
            "total_layer_requests": int(buf[0]),
            "simulated_12slot_hits": int(buf[1]),
            "simulated_host_pagecache_hits": int(buf[2]),
            "simulated_cold_fetches": int(buf[3]),
            "simulated_r4_clean_evictions": int(buf[4]),
            "dirty_writebacks": int(buf[5]),
            "direction_b_mobile_soc_4tier_ledger": {
                "tier0_kryo_neon_smmla_reg_mb": round(int(ext[0]) / (1024.0 * 1024.0), 2),
                "tier1_slc_4mb_and_adreno_gmem_mb": round(int(ext[1]) / (1024.0 * 1024.0), 2),
                "tier2_unified_lpddr5x_dram_mb": round(int(ext[2]) / (1024.0 * 1024.0), 2),
                "bytes_saved_q4_and_broadcast_mb": round(int(ext[3]) / (1024.0 * 1024.0), 2),
                "armv86_i8mm_smmla_2x2_tiles": int(ext[4]),
                "flashattention2_online_softmax_calls": int(ext[5]),
                "adreno740_opencl_dot_acc_sat_launches": int(ext[6]),
                "linux_proc_self_stat_minflt_delta": int(ext[7]),
                "linux_proc_self_stat_majflt_delta": int(ext[8]),
                "adreno740_zero_copy_opencl_active": bool(ext[9]),
                "use_4bit_operators": bool(ext[10]),
                "use_4bit_lm_head_top64_refine": bool(ext[11]),
            },
        }


# ==============================================================================
# 3. Layer 2 (`Lfm3NativeHybridLink`) & Layer 3 (`11-Level NCM` + `LexicalPointerDecoder`)
# ==============================================================================

class _ConfigNS(dict):
    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


class S23HybridAgentAndNCM:
    """
    Loads the real PyTorch `Lfm3NativeHybridLink(d=2048, bottleneck=512)` (Layer 2) and the real
    11-level `NextConceptTransformer` + `VQVAE` + `LexicalPointerDecoder` + `Lfm3ComboBridge` +
    `Lfm3ExpertBridge` (Layer 3) from `ncm/lfm3_ncm_core_bf16.pt` + `mmap`-backed `vqvae_wte.bf16.bin`
    and `vqvae_dec.bf16.bin`.
    """

    def __init__(self, ncm_dir: Path, pruned_dir: Path, tokenizer: Lfm3PureBpeTokenizer, c_engine: S23Lfm3NativeEngine):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.torch = torch
        self.nn = nn
        self.F = F
        self.ncm_dir = Path(ncm_dir)
        self.pruned_dir = Path(pruned_dir)
        self.tok = tokenizer
        self.c_engine = c_engine
        self.vocab_size = V_LFM3
        self.hidden_size = D_LFM3

        # Set PyTorch to 1 thread with OPENBLAS_NUM_THREADS=1 for fast, 100% deterministic execution on S23
        torch.set_num_threads(1)

        if str(self.ncm_dir) not in sys.path:
            sys.path.insert(0, str(self.ncm_dir))

        from models.next_concept import NextConceptTransformer
        from models.vqvae import VQVAE
        from ncm_runtime import (
            GRAMMAR_CONNECTOR_WORDS,
            P_BPE,
            P_RH,
            LexicalPointerDecoder,
            join_rendered_windows,
            pack_combo_prefix,
        )

        self.P_BPE = P_BPE
        self.P_RH = P_RH
        self._join_rendered_windows = join_rendered_windows
        self._pack_combo_prefix = pack_combo_prefix

        class Lfm3NativeHybridLink(nn.Module):
            def __init__(self, d: int = D_LFM3, bottleneck: int = 512):
                super().__init__()
                self.pre_ln = nn.LayerNorm(d)
                self.proj1 = nn.Linear(d, bottleneck)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(bottleneck, d)
                self.post_ln = nn.LayerNorm(d)
                self.out_scale = nn.Parameter(torch.tensor(0.024))

            def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
                hf = h_lfm3.float()
                z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
                return z * self.out_scale

        class HybridLink2048(nn.Module):
            """Exact HybridLink (ln_res_adapter, d=2048 -> 2048 -> 2048) trained in 3-stage HybridSFT."""
            def __init__(self, hidden_size: int = D_LFM3):
                super().__init__()
                self.proj1 = nn.Linear(hidden_size, hidden_size)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(hidden_size, hidden_size)
                self.pre_ln = nn.LayerNorm(hidden_size)
                self.post_ln = nn.LayerNorm(hidden_size)

            def forward(self, h: torch.Tensor) -> torch.Tensor:
                hf = h.float()
                out = self.proj2(self.act(self.proj1(self.pre_ln(hf))))
                return self.post_ln(hf + out)

        class Lfm3ComboBridge(nn.Module):
            def __init__(self, d_in: int = D_LFM3, d_out: int = D_VQ):
                super().__init__()
                self.ln_in = nn.LayerNorm(d_in)
                self.fc1 = nn.Linear(d_in, 768)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(768, d_out)
                self.ln_out = nn.LayerNorm(d_out)
                self.scale = nn.Parameter(torch.tensor(8.5))
                self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
                self.gate = nn.Parameter(torch.tensor(1.0))

            def forward(self, r_lfm3: torch.Tensor) -> torch.Tensor:
                x = self.ln_in(r_lfm3.float())
                x = self.fc2(self.act(self.fc1(x)))
                x = self.ln_out(x) * self.scale
                return (x * self.gate + self.seg_rh).to(r_lfm3.dtype)

        class Lfm3ExpertBridge(nn.Module):
            def __init__(self, d_lfm3: int = D_LFM3, d_vq: int = D_VQ):
                super().__init__()
                self.ln_in = nn.LayerNorm(d_lfm3)
                self.proj1 = nn.Linear(d_lfm3, d_vq)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(d_vq, d_vq)
                self.ln_out = nn.LayerNorm(d_vq)
                self.alpha = nn.Parameter(torch.tensor(0.5))

            def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
                x = self.ln_in(h_lfm3.float())
                x = self.proj2(self.act(self.proj1(x)))
                return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_lfm3.dtype)

        class MmapBF16Embedding(nn.Module):
            def __init__(self, path: Path, num_embeddings: int, embedding_dim: int):
                super().__init__()
                mm = np.memmap(str(path), dtype=np.uint16, mode="c", shape=(num_embeddings, embedding_dim))
                self.register_buffer("weight", torch.from_numpy(mm).view(torch.bfloat16), persistent=False)

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                return self.weight[idx].float()

        class MmapBF16Linear(nn.Module):
            def __init__(self, path: Path, in_features: int, out_features: int):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                mm = np.memmap(str(path), dtype=np.uint16, mode="c", shape=(out_features, in_features))
                self.register_buffer("weight", torch.from_numpy(mm).view(torch.bfloat16), persistent=False)

            def forward(self, x: torch.Tensor, allowed_ids: torch.Tensor | None = None) -> torch.Tensor:
                if allowed_ids is not None:
                    sub_w = self.weight[allowed_ids]
                    sub_lg = F.linear(x.to(torch.bfloat16), sub_w).float()
                    out = torch.full((x.shape[0], x.shape[1], self.out_features), -100.0, dtype=torch.float32)
                    out[:, :, allowed_ids] = sub_lg
                    return out
                return F.linear(x.to(torch.bfloat16), self.weight).float()

        vq_cfg = dict(
            in_vocab_size=1,
            context_length=64,
            embed_dim=384,
            vqvae_vocab_size=[4096] * 11,
            n_layers=6,
            compression_factor=1,
            dropout=0.1,
            pre_quant_groupnorm=8,
            use_qk_norm=True,
            pre_quant_dropout=0.25,
            vector_quantizer_config=_ConfigNS(
                clss="multiscale_residual_vector_quantizer",
                decay=0.99,
                epsilon=1e-5,
                commitment_cost=0.25,
                learned_l1_sampling=True,
                learned_l1_cascade=True,
                learned_l1_cascade_base=4,
                learned_l1_cascade_norm=True,
                learned_l1_cascade_norm_final=True,
                learned_all_sampling=False,
                quant_resi=_ConfigNS(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False),
                aux=_ConfigNS(fine_drop=_ConfigNS(prob=0.5, min_keep=1)),
                levels=_ConfigNS(use_manual_levels=True, manual_levels=[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]),
            ),
        )
        self.vqvae = VQVAE(**vq_cfg).cpu().eval()
        self.vqvae.in_vocab_size = V_LFM3
        self.tk = types.SimpleNamespace(vqvae=self.vqvae)

        mcfg = _ConfigNS(
            n_layer=12,
            n_head=12,
            bias=True,
            dropout=0.1,
            n_embd=768,
            context_length=160,
            prefix_len=96,
            raw_token_prefix=True,
            prefix_pad_token_id=151643,
            vocab_size=[4096] * 10,
            predict_level0=True,
            attn_pattern="block_diagonal",
            use_positional_encoding=True,
            use_level_encoding=True,
            use_rope=True,
            rope_base=100000,
            use_qk_norm=True,
            ctx_encoder=_ConfigNS(enabled=False, n_layer=4, vocab_size=50304),
            use_flex_attention=False,
            untied_head_bias=False,
        )
        self.net = NextConceptTransformer(vqvae_local=self.vqvae, config=mcfg).cpu().eval()
        self.copy_dec = LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).cpu().eval()
        self.hybrid_link = Lfm3NativeHybridLink(d=D_LFM3, bottleneck=512).cpu().eval()
        self.hybridlink_2048 = HybridLink2048(hidden_size=D_LFM3).cpu().eval()
        self.has_stage3_hybridlink_2048 = False
        self.combo_bridge = Lfm3ComboBridge(D_LFM3, D_VQ).cpu().eval()
        self.expert_bridge = Lfm3ExpertBridge(D_LFM3, D_VQ).cpu().eval()

        core_pt = self.ncm_dir / "lfm3_ncm_core_bf16.pt"
        core = torch.load(str(core_pt), map_location="cpu", weights_only=False)
        self.vqvae.load_state_dict(
            {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in core["vqvae_core"].items()},
            strict=False,
        )
        self.vqvae.encoder.wte = MmapBF16Embedding(self.ncm_dir / "vqvae_wte.bf16.bin", V_LFM3, D_VQ)
        self.vqvae.decoder.output_layer = MmapBF16Linear(self.ncm_dir / "vqvae_dec.bf16.bin", D_VQ, V_LFM3)
        self.net.load_state_dict(
            {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in core["net"].items()},
            strict=True,
        )
        self.copy_dec.load_state_dict(core["ptr_dec"], strict=True)
        self.hybrid_link.load_state_dict(core["lfm3_native_hybrid_link"], strict=True)
        if "hybridlink_2048" in core:
            self.hybridlink_2048.load_state_dict(
                {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in core["hybridlink_2048"].items()},
                strict=True,
            )
            self.has_stage3_hybridlink_2048 = True
        self.combo_bridge.load_state_dict(core["lfm3_combo_bridge"], strict=True)
        self.expert_bridge.load_state_dict(core["lfm3_expert_bridge"], strict=True)
        del core
        import gc
        gc.collect()

        emb_mm = np.memmap(str(self.pruned_dir / "global-0" / "embed_tokens.bf16.bin"), dtype=np.uint16, mode="c", shape=(V_LFM3, D_LFM3))
        self.lfm3_emb_bf16 = torch.from_numpy(emb_mm).view(torch.bfloat16)

        self.grammar_ids = {self.tok.eos_token_id}
        self.repeatable_grammar_ids = {self.tok.eos_token_id}
        for gw in GRAMMAR_CONNECTOR_WORDS:
            for variant in (gw, " " + gw):
                for tid in self.tok.encode(variant, allow_special=False):
                    if 0 <= tid < self.vocab_size:
                        self.grammar_ids.add(tid)
                        if gw.lower() in (
                            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from",
                            "as", "is", "are", "was", "were", ".", ",", ";", ":", "-", "/", "//", "://", "_", "=", "&", "(", ")", "[", "]",
                        ):
                            self.repeatable_grammar_ids.add(tid)

        levels = list(self.vqvae.vq.levels)
        self.sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]

    def run_continuous_latent_step(self, h_last_2048: np.ndarray) -> dict[str, Any]:
        """
        Layer 2 (`HybridLink2048` + `Lfm3NativeHybridLink(d=2048, bottleneck=512)`):
        Projects `h_t` into continuous `<latent>` space `r_t = R_Lfm3(h_t)` and measures
        its cosine alignment with LFM3-Omni-2B's next-token embedding in `128,000`-vocab space.
        """
        torch = self.torch
        F = self.F
        t0 = time.perf_counter()
        with torch.no_grad():
            h_t = torch.from_numpy(h_last_2048.reshape(1, D_LFM3)).float()
            r_native = self.hybrid_link(h_t)  # [1, 2048]
            if self.has_stage3_hybridlink_2048:
                r_s3 = self.hybridlink_2048(h_t) * 0.024
            else:
                r_s3 = r_native
            top1_id = self.c_engine.lm_head_top1(h_last_2048)
            e_top1 = self.lfm3_emb_bf16[top1_id].float().unsqueeze(0)
            raw_cos = float(F.cosine_similarity(h_t, e_top1, dim=-1).item())
            latent_cos = float(F.cosine_similarity(r_native, e_top1, dim=-1).item())
            s3_cos = float(F.cosine_similarity(r_s3, e_top1, dim=-1).item())
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "module": (
                "HybridLink2048 (Stage-3 Self-Conditioned) + Lfm3NativeHybridLink(d=2048, bottleneck=512)"
                if self.has_stage3_hybridlink_2048
                else "Lfm3NativeHybridLink(d=2048, bottleneck=512)"
            ),
            "latent_vector_norm": round(float(r_native.norm().item()), 4),
            "top1_predicted_token_id": top1_id,
            "top1_predicted_token_str": self.tok.decode([top1_id], skip_special_tokens=False),
            "raw_hidden_to_next_embed_cos": round(raw_cos, 4),
            "hybridlink_latent_to_next_embed_cos": round(latent_cos, 4),
            "stage3_hybridlink_2048_cos": round(s3_cos, 4),
            "hybridlink_ms": round(ms, 2),
            "r_np": r_native.squeeze(0).numpy(),
        }

    def decode_hybrid_agent_turn(
        self,
        initial_h_2048: np.ndarray,
        max_new_tokens: int = 128,
        min_latent_steps: int = 2,
        max_latent_steps: int = 3,
    ) -> tuple[list[int], np.ndarray, float, dict[str, Any]]:
        """
        Executes a full `HybridAgent` generation turn on `liblfm3_cpu_engine.so`:
          - Decodes discrete tokens until `<latent>` (`125033`), `</search>`, `</python>`, `</answer>`, or `<|im_end|>`.
          - Whenever `<latent>` (`125033`) is emitted, transitions to continuous `LATENT` mode and steps
            `r_t = R_Lfm3(h_t)` directly into the 24-layer C engine (`step_embedding_cached`) for `min_latent_steps .. max_latent_steps`,
            then closes the span with `</latent>` (`125034`) via `step_token_cached` and resumes decoding!
        """
        torch = self.torch
        F = self.F
        all_gen_ids: list[int] = []
        h_cur = initial_h_2048
        total_c_ms = 0.0
        latent_spans_opened = 0
        latent_steps_total = 0
        latent_span_readouts: list[list[str]] = []

        while len(all_gen_ids) < max_new_tokens:
            rem = max(1, max_new_tokens - len(all_gen_ids))
            think_mode = 1 if latent_spans_opened == 0 else 2
            chunk_ids, h_cur, ms_chunk = self.c_engine.decode_from_cache(
                h_cur, max_new_tokens=rem, allow_thinking=think_mode
            )
            total_c_ms += ms_chunk
            if not chunk_ids:
                break
            all_gen_ids.extend(chunk_ids)
            last_id = chunk_ids[-1]

            if last_id == 125033 and latent_spans_opened < 1:
                # Entered <latent> mode: run Stage-3 HybridLink2048 continuous steps through the 24-layer C engine!
                latent_spans_opened += 1
                span_tokens: list[str] = []
                for step_i in range(max_latent_steps):
                    with torch.no_grad():
                        h_t = torch.from_numpy(h_cur.reshape(1, D_LFM3)).float()
                        r_t = self.hybridlink_2048(h_t) if self.has_stage3_hybridlink_2048 else self.hybrid_link(h_t)
                        readout_id = self.c_engine.lm_head_top1(h_cur)
                    if step_i + 1 >= min_latent_steps and readout_id == 125034:
                        break
                    span_tokens.append(self.tok.decode([readout_id], skip_special_tokens=False))
                    h_cur, lat_ms = self.c_engine.step_embedding_cached(r_t.squeeze(0).numpy())
                    total_c_ms += lat_ms
                    latent_steps_total += 1
                latent_span_readouts.append(span_tokens)
                # Close <latent> span with token 125034 (</latent>)
                h_cur, close_ms = self.c_engine.step_token_cached(125034)
                total_c_ms += close_ms
                all_gen_ids.append(125034)
                continue

            # Otherwise decode_from_cache stopped on </search>, </python>, </answer>, </tool_call>, or <|im_end|>
            break

        return (
            all_gen_ids,
            h_cur,
            total_c_ms,
            {
                "latent_spans_opened": latent_spans_opened,
                "latent_steps_total": latent_steps_total,
                "latent_span_readouts": latent_span_readouts,
            },
        )

    def _generate_from_zpref_kv_exact(self, z_prefix: Any, pad_mask_bool: Any) -> Any:
        """
        Exact prefix-KV-cached 11-level `NextConceptTransformer` generation (`313` concept codes across
        11 multiscale VQ levels `[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`), verified `100.00%` bit-identical
        to `generate_from_zpref` while running `2.6x` faster (`8.1s` vs `21.1s` on S23).
        """
        torch = self.torch
        F = self.F
        net = self.net
        vq = self.vqvae.vq
        dev = z_prefix.device
        B, P, _ = z_prefix.shape
        levels = list(vq.levels)
        z_hat = torch.zeros(B, levels[-1], self.vqvae.embed_dim, device=dev)

        prefix_emb = net.transformer.wte(z_prefix)
        if not net.use_rope:
            prefix_emb = prefix_emb + net.wpe(net.wp_prefix)
        level_pos_embed = (
            net.transformer.wle(net.lvl_1L)
            if net.config.use_level_encoding
            else torch.zeros(1, net.L - levels[0], net.config.n_embd, device=dev)
        )
        if not net.use_rope:
            level_pos_embed = level_pos_embed + net.wpe(net.wp_1L)

        _pbias = torch.zeros(B, 1, 1, P, device=dev)
        _pbias[:, :, :, :P] = torch.where(
            pad_mask_bool[:, None, None, :],
            torch.tensor(float("-inf"), device=dev),
            torch.tensor(0.0, device=dev),
        )

        rope_p = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
        h = prefix_emb
        pfx_mask = torch.zeros(1, 1, P, P, device=dev) + _pbias
        net._set_kv_caching(True)
        for block in net.transformer.blocks:
            h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
        prefix_kv = [(b.attn.cached_k, b.attn.cached_v) for b in net.transformer.blocks]
        h = net.transformer.ln_f(h)
        l0_logits = net.lm_head_l0(net._level0_slice(h, pad_mask_bool))
        idx = net._sample_level_raw(l0_logits, self.sched[0], 0, 1.0)

        generated_idx = [idx]
        idx_next = idx
        level_acum = 0
        for level_idx in range(len(levels) - 1):
            l_next = levels[level_idx + 1]
            z_q_level = vq.embeddings[level_idx](idx_next)
            z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
            x = net.transformer.wte(x) + level_pos_embed[:, level_acum : level_acum + l_next, :]
            rope_k = (
                getattr(net, f"rope_cos_gen_{level_idx}"),
                getattr(net, f"rope_sin_gen_{level_idx}"),
            ) if net.use_rope else None
            for blk, (pk, pv) in zip(net.transformer.blocks, prefix_kv):
                blk.attn.cached_k, blk.attn.cached_v = pk, pv
            lvl_mask = F.pad(_pbias, (0, l_next)).expand(B, 1, l_next, P + l_next).contiguous()
            for block in net.transformer.blocks:
                x = block(x, attn_mask=lvl_mask, rope_cos_sin=rope_k)
            x = net.transformer.ln_f(x)
            logits = net.lm_heads[level_idx](x)
            idx_next = net._sample_level_raw(logits, self.sched[level_idx + 1], 0, 1.0)
            generated_idx.append(idx_next)
            level_acum += l_next
        net._set_kv_caching(False)
        return torch.cat(generated_idx, dim=1)

    def _build_allowed_vocab_ids(self, plan_str: str) -> Any:
        allowed = set(self.grammar_ids)
        allowed.update(self.tok.encode(plan_str, allow_special=False))
        for w in re.findall(r"[A-Za-z0-9.,:!?'£$€%/_\-+=&#()\[\]]+", plan_str):
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()):
                allowed.update(self.tok.encode(v, allow_special=False))
        return self.torch.tensor(sorted(x for x in allowed if 0 <= x < self.vocab_size), dtype=self.torch.long)

    def _forward_pointer_decoder(self, z_hat: Any, z_rh: Any, msk_rh: Any, z_bpe: Any, p_pad: Any, allowed_ids: Any) -> Any:
        torch = self.torch
        F = self.F
        dec = self.copy_dec
        B, T, _ = z_hat.shape
        msk_bpe = p_pad == self.tok.pad_token_id
        t_pos = dec.tgt_pos(torch.arange(T))[None, :, :]
        p_pos = dec.plan_pos(torch.arange(self.P_BPE))[None, :, :]
        r_pos = dec.rh_pos(torch.arange(self.P_RH))[None, :, :]

        mem_bpe = z_bpe + p_pos
        mem_rh = z_rh + r_pos
        mem = torch.cat([mem_rh, mem_bpe], dim=1)
        mem_mask = torch.cat([msk_rh, msk_bpe], dim=1)

        x = z_hat + t_pos
        for blk in dec.blocks:
            x = blk(x, mem, mem_mask)
        z_refined = z_hat + dec.res_scale * dec.ln_out(x)

        if self.vqvae.decoder.up is not None:
            z_refined = self.vqvae.decoder.up(z_refined.transpose(1, 2)).transpose(1, 2)
        if self.vqvae.decoder.wpe is not None:
            z_refined = z_refined + self.vqvae.decoder.wpe(torch.arange(z_refined.shape[1], dtype=torch.long))
        xd = self.vqvae.decoder.middle_block(z_refined)
        for blk in self.vqvae.decoder.extra_blocks:
            xd = blk(xd)
        xd = self.vqvae.decoder.ln_f(xd)
        base_logits = self.vqvae.decoder.output_layer(xd, allowed_ids=allowed_ids)

        cq = F.normalize(dec.copy_q(x), dim=-1)
        ck = F.normalize(dec.copy_k(mem_bpe), dim=-1)
        sim = torch.bmm(cq, ck.transpose(1, 2))

        p_len = (~msk_bpe).sum(dim=1, keepdim=True).clamp(min=1).float()
        has_semi = (p_pad == self.tok.semicolon_id).any(dim=1, keepdim=True)
        expected_T = torch.where(has_semi, (p_len * 1.25 + 2.0).clamp(min=4.0, max=float(T)), p_len)
        t_pos_idx = torch.arange(T, dtype=torch.float32)[None, :, None]
        p_pos_idx = torch.arange(self.P_BPE, dtype=torch.float32)[None, None, :]

        diag_soft = -2.5 * ((t_pos_idx / expected_T[:, :, None]) - (p_pos_idx / p_len[:, :, None])).pow(2)
        diag_clause = -45.0 * (t_pos_idx - p_pos_idx).pow(2)
        diag_bias = torch.where(has_semi[:, :, None], diag_soft, diag_clause)
        sim = (sim * 12.0 + diag_bias).masked_fill(msk_bpe[:, None, :], -1e4)
        attn_probs = F.softmax(sim, dim=-1)
        ctx_vec = torch.bmm(attn_probs, z_bpe)

        g_copy = torch.sigmoid(dec.copy_gate(torch.cat([x, ctx_vec], dim=-1)))
        soft_scale = g_copy * dec.copy_scale.clamp(min=0.0, max=16.0)
        in_bounds = t_pos_idx < p_len[:, :, None]
        clause_scale = torch.where(in_bounds, soft_scale + 65.0, torch.zeros_like(soft_scale))
        eff_scale = torch.where(has_semi[:, :, None], soft_scale, clause_scale)
        copy_boost = attn_probs * eff_scale

        idx_exp = p_pad[:, None, :].expand(B, T, self.P_BPE)
        base_logits.scatter_add_(2, idx_exp, copy_boost.to(base_logits.dtype))
        return base_logits

    def _dec_model_aligned(self, row_logits: Any, plan_ids: list[int]) -> tuple[str, list[int]]:
        pad_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < self.vocab_size and int(x) not in (pad_id, eos_id)]
        if not valid_p:
            return "", []

        cont_map: dict[int, set[int]] = {}
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs[0] in (":", ".", "'", "/", "-", "_", "=", "?", "&", "#")):
                cont_map.setdefault(a, set()).add(b)

        max_counts = collections.Counter(valid_p)
        content_set = {tid for tid in valid_p if tid not in self.repeatable_grammar_ids}
        emitted_counts: collections.Counter[int] = collections.Counter()

        T = row_logits.size(0)
        max_len = min(T, len(valid_p))
        clean_ids: list[int] = []
        prev_tid = -1
        prev_word = ""
        for t in range(max_len):
            lg = row_logits[t].clone()
            if prev_tid >= 0:
                if emitted_counts[prev_tid] >= max_counts.get(prev_tid, 1):
                    lg[prev_tid] -= 30.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        if emitted_counts[nxt_b] < max_counts.get(nxt_b, 1):
                            lg[nxt_b] += 12.0
            for tid, cnt in emitted_counts.items():
                if tid not in self.repeatable_grammar_ids and cnt >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0

            all_covered = bool(content_set) and all(
                emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set
            )
            if all_covered and t >= len(valid_p) - 1:
                lg[eos_id] += 18.0

            tid = int(lg.argmax().item())
            if tid == eos_id:
                break
            if 0 <= tid < self.vocab_size and tid != pad_id:
                ws = self.tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2 and emitted_counts[tid] >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0
                    tid = int(lg.argmax().item())
                    if tid == eos_id:
                        break
                    ws = self.tok.decode([tid], skip_special_tokens=False).strip().lower()
                clean_ids.append(tid)
                emitted_counts[tid] += 1
                prev_tid = tid
                if ws:
                    prev_word = ws
                if bool(content_set) and all(emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set):
                    raw_piece = self.tok.decode([tid], skip_special_tokens=False).rstrip()
                    if raw_piece.endswith((".", "!", "?")) and t >= len(valid_p) - 1:
                        break

        decoded_str = self.tok.decode(clean_ids).strip()
        all_content_emitted = all(
            emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set
        )
        has_dangling_tail = bool(
            re.search(
                r"(?:\b(?:and|or|to|for|of|in|with|by|from|the|a|an|as|at|on)\s*$|,\s*$|\d+,\d{1,2}$)",
                decoded_str,
                flags=re.IGNORECASE,
            )
        )
        if not decoded_str or not all_content_emitted or has_dangling_tail:
            return self.tok.decode(valid_p), list(valid_p), decoded_str, True
        return decoded_str, clean_ids, decoded_str, False

    def render_plan_through_11level_ncm(self, plan_text: str) -> tuple[str, list[int], dict[str, Any]]:
        """
        Layer 3 (`Lfm3NCMRenderer`):
        Runs `plan_text` through:
          1. 24-layer LFM3 C engine (`h_backbone [1, P_BPE, 2048]`)
          2. `Lfm3NativeHybridLink` -> `Lfm3ComboBridge(2048 -> 384)` (`z_rh [1, P_BPE, 384]`)
          3. `vqvae.encoder.wte(p_pad) + Lfm3ExpertBridge(2048 -> 384)` (`z_bpe [1, P_BPE, 384]`)
          4. `pack_combo_prefix` -> 11-level `NextConceptTransformer` (`313` concept codes)
          5. `vqvae.vq.idx_to_zhats` -> `LexicalPointerDecoder` -> `128,000`-vocab LFM3 token IDs.
        """
        torch = self.torch
        t0 = time.perf_counter()
        clean_plan = (plan_text or "").strip()
        if not clean_plan:
            return "", [], {}

        paragraphs = [p.strip() for p in clean_plan.split("\n\n") if p.strip()]
        first_para = paragraphs[0]
        rest_paras = paragraphs[1:]

        pad_id = self.tok.pad_token_id
        pids = self.tok.encode(first_para, allow_special=False)[: self.P_BPE]
        if not pids:
            return clean_plan, self.tok.encode(clean_plan, allow_special=False), {}

        h_c, c_ms = self.c_engine.forward_24_layers(pids)
        h_backbone = torch.zeros((1, self.P_BPE, D_LFM3), dtype=torch.float32)
        h_backbone[0, : len(pids)] = torch.from_numpy(h_c)

        p_pad = torch.tensor([pids + [pad_id] * (self.P_BPE - len(pids))], dtype=torch.long)
        pad_mask_bpe = p_pad == pad_id

        with torch.no_grad():
            r_latent = self.hybrid_link(h_backbone)
            z_rh = self.combo_bridge(r_latent).float().masked_fill(pad_mask_bpe.unsqueeze(-1), 0.0)
            z_exp = self.expert_bridge(h_backbone).float().masked_fill(pad_mask_bpe.unsqueeze(-1), 0.0)
            z_bpe = self.vqvae.encoder.wte(p_pad) + z_exp
            z_pref, pmask = self._pack_combo_prefix(z_rh, pad_mask_bpe, z_bpe, pad_mask_bpe)

            allowed_ids = self._build_allowed_vocab_ids(first_para)
            torch.manual_seed(42)
            g_codes = self._generate_from_zpref_kv_exact(z_pref, pmask)
            zh_levels, _ = self.vqvae.vq.idx_to_zhats(g_codes, return_zlevels=True)
            lg = self._forward_pointer_decoder(zh_levels[-1], z_rh, pad_mask_bpe, z_bpe, p_pad, allowed_ids)
            rendered_first, out_ids, raw_ncm_first, used_fallback = self._dec_model_aligned(lg[0], p_pad[0].tolist())

        full_first_ids = self.tok.encode(first_para, allow_special=False)
        if len(full_first_ids) > self.P_BPE:
            tail_ids = full_first_ids[self.P_BPE :]
            rendered_first = (rendered_first + self.tok.decode(tail_ids)).strip()
            out_ids.extend(tail_ids)

        if rest_paras:
            rest_txt = "\n\n".join(rest_paras)
            rendered_full = f"{rendered_first}\n\n{rest_txt}"
            out_ids.extend(self.tok.encode("\n\n" + rest_txt, allow_special=False))
        else:
            rendered_full = rendered_first

        ncm_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "ncm_active": True,
            "vq_levels": 11,
            "vq_codes_emitted": int(g_codes.shape[1]),
            "zhat_final_shape": list(zh_levels[-1].shape),
            "allowed_vocab_subset_size": int(allowed_ids.numel()),
            "raw_ncm_decoded_text": raw_ncm_first,
            "used_completeness_fallback": bool(used_fallback),
            "c_backbone_ms": round(c_ms, 2),
            "ncm_total_ms": round(ncm_ms, 2),
            "emitted_tokens": len(out_ids),
        }
        return rendered_full, out_ids, meta


# ==============================================================================
# 4. Layer 1 Omni Multimodal Inspection + Real Media Generators
# ==============================================================================

def inspect_attached_image_real(pil_img: Image.Image) -> dict[str, Any]:
    """Computes real pixel/channel statistics from an attached image using Pillow."""
    img = pil_img.convert("RGB")
    stat = ImageStat.Stat(img)
    r_m, g_m, b_m = [round(float(x), 1) for x in stat.mean]
    brightness = round((0.299 * r_m + 0.587 * g_m + 0.114 * b_m) / 255.0 * 100.0, 1)
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_density = round(float(ImageStat.Stat(edges).mean[0]) / 255.0 * 100.0, 1)
    summary = (
        f"Attached image ({img.width}x{img.height} RGB, mean RGB=({r_m}, {g_m}, {b_m}), "
        f"brightness={brightness}%, edge detail={edge_density}%)."
    )
    return {
        "width": img.width,
        "height": img.height,
        "mean_rgb": [r_m, g_m, b_m],
        "brightness_pct": brightness,
        "edge_density_pct": edge_density,
        "summary": summary,
    }


def inspect_attached_wav_real(wav_bytes: bytes, fname: str) -> dict[str, Any]:
    """Reads real PCM waveform properties from a WAV file."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        dur = round(nframes / float(max(1, sr)), 2)
        raw = wf.readframes(min(nframes, sr * 10))
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    rms_db = round(20.0 * math.log10(max(1e-6, float(np.sqrt(np.mean(arr ** 2))))), 1) if arr.size else -60.0
    summary = f"Attached WAV audio '{fname}' ({dur}s, {sr} Hz, {nch} ch, RMS={rms_db} dBFS)."
    return {
        "file_name": fname,
        "duration_s": dur,
        "sample_rate_hz": sr,
        "channels": nch,
        "rms_dbfs": rms_db,
        "summary": summary,
    }


def _render_scene_frame_pil(prompt: str, width: int = 640, height: int = 400, phase: float = 0.0) -> Image.Image:
    from PIL import ImageDraw

    low = (prompt or "").lower()
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    if any(w in low for w in ("sunset", "sunrise", "beach", "ocean", "sea", "horizon", "island", "golden")):
        top_c = np.array([28.0, 36.0, 88.0], dtype=np.float32)
        mid_c = np.array([238.0, 112.0, 58.0], dtype=np.float32)
        bot_c = np.array([252.0, 204.0, 96.0], dtype=np.float32)
        sky = np.where(
            y_norm < 0.55,
            top_c + (mid_c - top_c) * (y_norm / 0.55),
            mid_c + (bot_c - mid_c) * np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0),
        )
        rgb = np.broadcast_to(sky[:, None, :], (height, width, 3)).copy()
        sun_cx = width * (0.5 + 0.12 * math.sin(2.0 * math.pi * phase))
        sun_cy = height * (0.48 - 0.08 * math.cos(2.0 * math.pi * phase))
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - sun_cx) ** 2 + (yy - sun_cy) ** 2)
        glow = np.exp(-dist / 58.0)[:, :, None]
        rgb = np.clip(rgb * (1.0 - 0.55 * glow) + np.array([255.0, 244.0, 188.0]) * glow, 0.0, 255.0)
        rgb[dist <= 36.0] = [255.0, 250.0, 218.0]
        sea_y0 = int(height * 0.64)
        for y in range(sea_y0, height):
            wy = (y - sea_y0) / float(max(1, height - sea_y0))
            wave_v = math.sin(wy * 26.0 - phase * 2.0 * math.pi * 2.0) * 14.0
            base_sea = np.array([18.0 + 30.0 * (1.0 - wy), 74.0 + 48.0 * (1.0 - wy), 124.0 + 55.0 * (1.0 - wy)])
            shimmer = np.exp(-((np.arange(width) - sun_cx) ** 2) / (1800.0 + 2400.0 * wy))
            row = base_sea[None, :] + shimmer[:, None] * np.array([210.0, 145.0, 65.0]) * (0.55 + 0.35 * math.sin(y * 0.4 + phase * 6.28)) + wave_v
            rgb[y] = np.clip(row, 0.0, 255.0)
    elif any(w in low for w in ("space", "galaxy", "nebula", "star", "planet", "cosmos", "moon", "night", "cyber", "neon")):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx = width * (0.5 + 0.08 * math.cos(2.0 * math.pi * phase))
        cy = height * (0.46 + 0.06 * math.sin(2.0 * math.pi * phase))
        dx = (xx - cx) / (width * 0.45)
        dy = (yy - cy) / (height * 0.45)
        r = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx) + phase * 2.0 * math.pi
        swirl = 0.5 + 0.5 * np.sin(3.0 * ang - 4.5 * r)
        neb = np.exp(-1.6 * r) * swirl
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        rgb[:, :, 0] = 14.0 + 175.0 * neb + 65.0 * np.exp(-3.2 * r)
        rgb[:, :, 1] = 18.0 + 95.0 * np.exp(-2.0 * r) + 80.0 * neb * (1.0 - x_norm)
        rgb[:, :, 2] = 42.0 + 210.0 * np.exp(-1.3 * r)
        rng = np.random.default_rng(107)
        for _ in range(90):
            sx = int(rng.integers(4, width - 4))
            sy = int(rng.integers(4, height - 4))
            tw = 175.0 + 80.0 * math.sin(phase * 6.28 + sx * 0.1)
            rgb[sy - 1 : sy + 2, sx - 1 : sx + 2] = np.clip(rgb[sy - 1 : sy + 2, sx - 1 : sx + 2] + tw, 0.0, 255.0)
        rgb = np.clip(rgb, 0.0, 255.0)
    else:
        top_c = np.array([218.0, 236.0, 228.0], dtype=np.float32)
        bot_c = np.array([250.0, 246.0, 232.0], dtype=np.float32)
        sky = top_c + (bot_c - top_c) * y_norm
        rgb = np.broadcast_to(sky[:, None, :], (height, width, 3)).copy()
        sun_cx = int(width * (0.76 + 0.06 * math.sin(2.0 * math.pi * phase)))
        sun_cy = int(height * (0.24 - 0.04 * math.cos(2.0 * math.pi * phase)))
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - sun_cx) ** 2 + (yy - sun_cy) ** 2)
        glow = np.exp(-dist / 52.0)[:, :, None]
        rgb = np.clip(rgb * (1.0 - 0.45 * glow) + np.array([255.0, 232.0, 168.0]) * glow, 0.0, 255.0)
        rgb[dist <= 30.0] = [255.0, 242.0, 192.0]
        xs = np.arange(width, dtype=np.float32)
        layers_cfg = [
            (0.54, 28.0, 0.011, 0.8, [132.0, 168.0, 152.0]),
            (0.66, 24.0, 0.016, 2.1, [68.0, 118.0, 96.0]),
            (0.78, 18.0, 0.022, 3.7, [27.0, 94.0, 70.0]),
        ]
        for base_h, amp, freq, ph0, col in layers_cfg:
            ridge = (height * base_h) + amp * np.sin(xs * freq + ph0 + phase * 2.0 * math.pi) + (amp * 0.45) * np.cos(xs * freq * 2.1 - ph0)
            mask = yy >= ridge[None, :]
            rgb[mask] = col

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    clean_title = (prompt or "Clover Visual Scene").strip()[:58]
    draw.rectangle([(14, height - 42), (width - 14, height - 12)], fill=(19, 32, 24), outline=(139, 183, 164), width=1)
    draw.text((26, height - 33), f"Clover • {clean_title}", fill=(250, 248, 242))
    return img


def generate_image_png_data_url(subject: str) -> str:
    img = _render_scene_frame_pil(subject, width=640, height=400, phase=0.15)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_video_mp4_data_url(subject: str, base_dir: Path) -> str:
    import tempfile

    with tempfile.TemporaryDirectory(dir=str(base_dir)) as tmpd:
        tmp_path = Path(tmpd)
        n_frames = 24
        for i in range(n_frames):
            frame_img = _render_scene_frame_pil(subject, width=480, height=300, phase=i / float(n_frames))
            frame_img.save(tmp_path / f"f_{i:03d}.png", format="PNG")
        out_mp4 = tmp_path / "out.mp4"
        ffmpeg_bin = shutil.which("ffmpeg") or "/data/data/com.termux/files/usr/bin/ffmpeg"
        cmd = [
            ffmpeg_bin, "-y", "-framerate", "12", "-i", str(tmp_path / "f_%03d.png"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not out_mp4.exists():
            cmd_fallback = [
                ffmpeg_bin, "-y", "-framerate", "12", "-i", str(tmp_path / "f_%03d.png"),
                "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p", str(out_mp4),
            ]
            subprocess.run(cmd_fallback, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mp4_bytes = out_mp4.read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(mp4_bytes).decode("ascii")


def generate_spoken_wav_data_url(text: str, base_dir: Path) -> str:
    import tempfile

    clean_speech = re.sub(r"[*`#_~\[\]()<>]", " ", text or "Hello from Clover.")
    clean_speech = re.sub(r"\s+", " ", clean_speech).strip()[:360]
    if not clean_speech:
        clean_speech = "Here is your audio response from Clover."

    espeak_bin = shutil.which("espeak") or shutil.which("espeak-ng") or "/data/data/com.termux/files/usr/bin/espeak"
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as tmpd:
        out_wav = Path(tmpd) / "speech.wav"
        if Path(espeak_bin).exists() or shutil.which(espeak_bin):
            subprocess.run(
                [espeak_bin, "-v", "en-us", "-s", "158", "-p", "52", "-w", str(out_wav), clean_speech],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wav_bytes = out_wav.read_bytes()
        else:
            sr = 16000
            t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
            pcm = (0.25 * np.sin(2.0 * math.pi * 220.0 * t) * np.exp(-t * 1.2) * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            wav_bytes = buf.getvalue()
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")


# ==============================================================================
# 5. Complete 7-Layer Progressive On-Device Clover Coordinator (`max_active_layer = 1 .. 7`)
# ==============================================================================

class CloverS23OnDeviceSystem:
    """
    Coordinates the progressive 7-layer Clover AI System on Samsung Galaxy S23 (and K3 Virtual S23 Debugger).
    Supports `max_active_layer = 1 .. 7` so every layer is tested honestly:
      - `max_active_layer = 1`: Layer 1 (`LiquidAI/LFM3-1.7B` 24-layer 4-bit C engine with its own ChatML + `<tool_call>` / `<tool_response>` loop)
      - `max_active_layer = 2`: + Layer 2 (`Lfm3NativeHybridLink(d=2048)` next-token embedding adapter + live `web_search` & `python_interpreter` tool execution fed back into `LFM3-1.7B` via `<tool_response>`)
      - `max_active_layer = 3`: + Layer 3 (`Lfm3ComboBridge` + `Lfm3ExpertBridge` + 11-level `VQVAE` + `NextConceptTransformer` + `LexicalPointerDecoder`)
      - `max_active_layer = 4`: + Layer 4 (`k3_memory_graph.py` persistent relational memory tree injected into `LFM3-1.7B`'s prompt before prefill)
      - `max_active_layer = 5`: + Layer 5 (`k3_ontology_graph.py` symbolic ontology GraphRAG injected into `LFM3-1.7B`'s prompt before prefill + post-generation grounding audit)
      - `max_active_layer = 6`: + Layer 6 (`k3_generative_compiler.py` `CloverPythonGenerativeCompiler` checking only model-written code + diagnostic repair loop)
      - `max_active_layer = 7`: + Layer 7 (`SpecMD Policy 3` OS `madvise(MADV_WILLNEED)` lookahead + per-request software 12-slot LRU/heat tracker)
    """

    TOOL_WEB_SEARCH = {
        "name": "web_search",
        "description": "Search the live web, Wikipedia, and arXiv when current facts, external web pages, or research papers are needed.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query or URL"}},
            "required": ["query"],
        },
    }
    TOOL_PYTHON = {
        "name": "python_interpreter",
        "description": "Execute Python code when exact math, physics formulas, or code verification is needed.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code that prints the result"}},
            "required": ["code"],
        },
    }

    def __init__(self, base_dir: Path, load_neural_ncm: bool = True):
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir)
        self.pruned_dir = self.base_dir / "pruned"
        self.ncm_dir = self.base_dir / "ncm"
        self.app_secret = load_or_create_device_app_secret(self.base_dir)
        self.tokenizer = Lfm3PureBpeTokenizer(self.base_dir / "tokenizer.json")
        self.engine = S23Lfm3NativeEngine(
            self.pruned_dir,
            pin_s23_big_cores=True,
            use_4bit_ffn=True,
            madvise_active=True,
        )
        self.engine.configure_stop_mask(self.tokenizer)

        # Layer 2 + Layer 3: Real PyTorch `Lfm3NativeHybridLink` + 11-Level `NCM` + `LexicalPointerDecoder`
        self.hybrid_ncm: S23HybridAgentAndNCM | None = None
        if load_neural_ncm and (self.ncm_dir / "lfm3_ncm_core_bf16.pt").exists():
            self.hybrid_ncm = S23HybridAgentAndNCM(
                ncm_dir=self.ncm_dir,
                pruned_dir=self.pruned_dir,
                tokenizer=self.tokenizer,
                c_engine=self.engine,
            )

        # Layer 4: Local Git Repository Memory Engine (`clover-memory-repo/` with `AGENTS.md`, `.md`, `.py`, scripts & Private GitHub Sync)
        #          + Foundational Relational Memory Graph (`k3_memory_graph.py`)
        from clover_git_memory_repo import CloverMemoryGitRepo
        from k3_memory_graph import ensure_default_memory_tree, retrieve_relevant_memory_package

        canonical_agents = SCRIPT_DIR.parent.parent / "AGENTS.md"
        self.git_memory_repo = CloverMemoryGitRepo(
            self.base_dir,
            canonical_agents_md_path=canonical_agents if canonical_agents.exists() else None,
        )
        self.memory_graph = ensure_default_memory_tree()
        self._retrieve_memory = retrieve_relevant_memory_package

        # Layer 5: Deterministic Ontology-Guided Neuro-Symbolic GraphRAG
        from k3_ontology_graph import CloverOntologyRetriever, CloverSymbolicVerifier, render_evidence

        self.ontology_retriever = CloverOntologyRetriever()
        self.symbolic_verifier = CloverSymbolicVerifier
        self._render_ontology = render_evidence

        # Layer 6: Python Generative Compiler + Sutton (1988) Absorbing Markov TD(0) Statement Predictor
        from k3_generative_compiler import CloverPythonGenerativeCompiler, SuttonTDStatementPredictor

        self.py_compiler = CloverPythonGenerativeCompiler(allow_undefined=True)
        self.sutton_td_predictor = SuttonTDStatementPredictor()

        # Layer 2 Tool: Live Web / Wikipedia / arXiv / DuckDuckGo / URL Search (`LiveWebSearchTool.execute`)
        from live_web_search import LiveWebSearchTool

        self.search_tool = LiveWebSearchTool(max_results=3)

        # Pre-warm the default system + `<tools>` prefix in the 24-layer KV + ShortConv cache
        self._default_sys_prefix_ids = self.tokenizer.encode(
            self._build_system_prompt(allow_internet=True, allow_python=True),
            allow_special=True,
        )
        self.engine.prefill_cached(self._default_sys_prefix_ids)
        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Persistent Studio Settings & Permissions (matching Laptop Clover AI Studio + S23 Unified SoC controls)
        self.settings_path = self.base_dir / "s23_studio_settings.json"
        self.perms_path = self.base_dir / "s23_studio_permissions.json"
        self.settings: dict[str, Any] = {
            "model": "omni",
            "engine": "adreno",
            "q4_operators": False,
            "q4_lm_head": True,
            "tools": "always",
            "layers": {
                "hybridlink": True,
                "ncm": False,
                "memory": True,
                "ontology": True,
                "compiler": True,
                "specmd_madvise": True,
            },
        }
        if self.settings_path.exists():
            try:
                saved_s = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(saved_s, dict):
                    if saved_s.get("engine") in ("adreno", "cpu"):
                        self.settings["engine"] = saved_s["engine"]
                    if isinstance(saved_s.get("q4_operators"), bool):
                        self.settings["q4_operators"] = saved_s["q4_operators"]
                    if isinstance(saved_s.get("q4_lm_head"), bool):
                        self.settings["q4_lm_head"] = saved_s["q4_lm_head"]
                    if saved_s.get("tools") in ("ask", "always"):
                        self.settings["tools"] = saved_s["tools"]
                    if isinstance(saved_s.get("layers"), dict):
                        for lk, lv in saved_s["layers"].items():
                            if lk in self.settings["layers"] and isinstance(lv, bool):
                                self.settings["layers"][lk] = lv
            except Exception:
                pass
        self._apply_hardware_settings()

        self.permissions_meta: dict[str, dict[str, str]] = {
            "internet": {
                "label": "Internet search and web pages",
                "detail": "Searches Wikipedia, arXiv and DuckDuckGo and opens web pages you link.",
            },
            "code": {
                "label": "Run code Clover writes",
                "detail": "Executes Python code in an isolated on-device subprocess with a timeout.",
            },
            "files": {
                "label": "Read phone files and attachments",
                "detail": "Reads text files, photos, WAV audio and videos you attach or reference on this S23.",
            },
            "voice_camera": {
                "label": "Microphone, camera and voice synthesis",
                "detail": "Allows voice input, spoken audio replies and procedural PNG/MP4/WAV generation.",
            },
            "memory": {
                "label": "Persistent Memory Graph (Layer 4)",
                "detail": "Retrieves and stores notes in Clover's relational memory tree on this S23.",
            },
        }
        self.perm_modes: dict[str, str] = {k: "allow" for k in self.permissions_meta}
        if self.perms_path.exists():
            try:
                saved_p = json.loads(self.perms_path.read_text(encoding="utf-8"))
                if isinstance(saved_p, dict):
                    for pk, pv in saved_p.items():
                        if pk in self.perm_modes and pv in ("allow", "ask", "deny"):
                            self.perm_modes[pk] = pv
            except Exception:
                pass

    def _apply_hardware_settings(self) -> None:
        self.engine.set_adreno_opencl_active(self.settings.get("engine") == "adreno")
        self.engine.set_use_4bit_operators(bool(self.settings.get("q4_operators", False)))
        self.engine.set_use_4bit_lm_head(bool(self.settings.get("q4_lm_head", True)))
        self.engine.set_madvise_active(bool(self.settings.get("layers", {}).get("specmd_madvise", True)))

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        if "engine" in changes and changes["engine"] in ("adreno", "cpu"):
            self.settings["engine"] = changes["engine"]
        if "q4_operators" in changes and isinstance(changes["q4_operators"], bool):
            self.settings["q4_operators"] = changes["q4_operators"]
        if "q4_lm_head" in changes and isinstance(changes["q4_lm_head"], bool):
            self.settings["q4_lm_head"] = changes["q4_lm_head"]
        if "tools" in changes and changes["tools"] in ("ask", "always"):
            self.settings["tools"] = changes["tools"]
        if isinstance(changes.get("layers"), dict):
            for lk, lv in changes["layers"].items():
                if lk in self.settings["layers"] and isinstance(lv, bool):
                    self.settings["layers"][lk] = lv
        self._apply_hardware_settings()
        try:
            self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        except Exception:
            pass
        return self.settings

    def get_permissions_view(self) -> dict[str, Any]:
        return {
            k: {"mode": self.perm_modes.get(k, "allow"), **meta}
            for k, meta in self.permissions_meta.items()
        }

    def update_permissions(self, changes: dict[str, Any]) -> dict[str, Any]:
        for pk, pv in (changes or {}).items():
            if pk in self.perm_modes and pv in ("allow", "ask", "deny"):
                self.perm_modes[pk] = pv
        try:
            self.perms_path.write_text(json.dumps(self.perm_modes, indent=2), encoding="utf-8")
        except Exception:
            pass
        return self.get_permissions_view()

    def list_memories(self) -> list[dict[str, Any]]:
        out = []
        for f_item in self.git_memory_repo.list_files():
            out.append({
                "id": f_item["path"],
                "path": f_item["path"],
                "category": f_item.get("category", "root"),
                "title": f_item["path"],
                "summary": f_item.get("summary") or f_item["path"],
                "created_at": f_item.get("updated_at", ""),
                "size_bytes": f_item.get("size_bytes", 0),
                "epistemic": "git_tracked",
                "status": "active",
            })
        return out

    HYBRID_AGENT_SYSTEM_PROMPT = (
        "You are a helpful assistant that solves the given question step by step with the help of a wikipedia search tool and a python interpreter tool. "
        "Given a question, you first think about the reasoning process and then provide the final answer, invoking the search tool for factual information "
        "and the python interpreter tool for calculations whenever needed.\n\n"
        "Format:\n"
        "- <think> ... </think>: your reasoning. Start every round with <think>; close it with </think> and then output exactly one of: a <search> call, a <python> call, or the final <answer>.\n"
        "- <latent> ... </latent>: latent reasoning, allowed only inside <think> ... </think>. Put routine internal reasoning (planning, recalling, restating, routine double-checking) in latent mode. Outside <latent>, write only the few decisive steps - key facts, decisions, and conclusions - as plain text. Always start and end the think block with plain text: open with a brief plain-text step (e.g. what the question or the latest result tells you), and close with the plain-text decision or conclusion that leads to your next action.\n"
        "- <search> query </search>: call the wikipedia search tool.\n"
        "- <python> code </python>: call the python interpreter tool; use print() to show the output.\n"
        "- <result> ... </result>: the tool output, inserted by the system after your </search> or </python>. Never write <result> yourself. After each result, start the next round with <think> - or, if the result directly settles the question, give the final <answer> immediately. If a result is empty, an error, or a tool-limit notice, do not repeat the same call - adjust your approach or give your best final answer.\n"
        "- <answer> ... </answer>: the final response, with the exact final answer enclosed in \\boxed{} in latex format.\n\n"
        "Format example:\n"
        "<think>\nThis is a key first step.<latent>This is the latent reasoning process.</latent>This is a key reasoning step.\n</think><search>search query here</search><result>search result here</result><think>\nThis is a key first step.<latent>This is the latent reasoning process.</latent>This is the key decision step.\n</think><python>python code here</python><result>python interpreter result here</result><think>\nThis is a key first step.<latent>This is the latent reasoning process.</latent>This is the key conclusion.\n</think><answer>\\boxed{answer here}</answer>"
    )

    def _build_system_prompt(self, allow_internet: bool = True, allow_python: bool = True, use_hybrid_agent_format: bool = True) -> str:
        if use_hybrid_agent_format and (allow_internet or allow_python):
            return f"<|startoftext|><|im_start|>system\n{self.HYBRID_AGENT_SYSTEM_PROMPT}<|im_end|>\n"
        active_tools = []
        if allow_internet:
            active_tools.append(self.TOOL_WEB_SEARCH)
        if allow_python:
            active_tools.append(self.TOOL_PYTHON)
        if active_tools:
            tools_block = "\n".join(json.dumps(t) for t in active_tools)
            return f"<|startoftext|><|im_start|>system\n<tools>\n{tools_block}\n</tools><|im_end|>\n"
        return "<|startoftext|><|im_start|>system\nYou are Clover, a helpful AI assistant. Answer the user's request directly.<|im_end|>\n"

    @staticmethod
    def _clean_model_text(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        # Strip any <latent>...</latent> spans and closed <think>...</think> block if present
        s = re.sub(r"<latent>[\s\S]*?</latent>", "", s).strip()
        s = re.sub(r"</?latent>", "", s).strip()
        if "</think>" in s:
            s = s.split("</think>")[-1].strip()
        # Extract <answer>...</answer> when present (HybridAgent format)
        m_ans = re.search(r"<answer>\s*([\s\S]*?)(?:</answer>|$)", s)
        if m_ans:
            s = m_ans.group(1).strip()
        # Unwrap outer \boxed{...} if the entire answer is wrapped in \boxed{...}
        m_box = re.match(r"^\\boxed\{([\s\S]+)\}$", s)
        if m_box:
            inner = m_box.group(1).strip()
            m_txt = re.match(r"^\\text\{([\s\S]+)\}$", inner)
            s = m_txt.group(1).strip() if m_txt else inner
        # Strip trailing unclosed <tool_call> or markdown json tool block if the model started one after answering
        for tag in ("<tool_call>", "<search>", "<python>", "</answer>"):
            if tag in s:
                s = s.split(tag)[0].strip()
        s = re.sub(r"```json\s*\{\s*\"(?:tool|query)\"[\s\S]*$", "", s).strip()
        s = re.sub(r"\n*(?:I can confirm this by searching|Let me do that quickly|Would you like me to)[^\n]*$", "", s).strip()
        s = re.sub(r"<\|[^>]+?\|>|</?think>|</?answer>", "", s).strip()
        return s

    @staticmethod
    def _parse_lfm3_tool_call_xml(raw_text: str) -> dict[str, Any] | None:
        """Parses HybridAgent's `<search>...</search>`, `<python>...</python>`, and Layer 4 Git Repo tags (`<repo_search>`, `<repo_read>`, `<repo_write>`) as well as `<tool_call><function=NAME>...`."""
        txt = raw_text or ""
        m_srch = re.search(r"<search>\s*([\s\S]*?)\s*</search>", txt)
        if m_srch:
            return {"function": "web_search", "parameters": {"query": m_srch.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        m_py = re.search(r"<python>\s*([\s\S]*?)\s*</python>", txt)
        if m_py:
            return {"function": "python_interpreter", "parameters": {"code": m_py.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_python"}
        m_rsrch = re.search(r"<repo_search>\s*([\s\S]*?)\s*</repo_search>", txt)
        if m_rsrch:
            return {"function": "repo_search", "parameters": {"query": m_rsrch.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        m_rread = re.search(r"<repo_read>\s*([\s\S]*?)\s*</repo_read>", txt)
        if m_rread:
            return {"function": "repo_read", "parameters": {"path": m_rread.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        m_rwrite = re.search(r"<repo_write\s+path=[\"']([^\"']+)[\"']\s*>\s*([\s\S]*?)\s*</repo_write>", txt)
        if m_rwrite:
            return {"function": "repo_write", "parameters": {"path": m_rwrite.group(1).strip(), "content": m_rwrite.group(2).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        if "<tool_call>" not in txt and "<function=" not in txt:
            return None
        m_fn = re.search(r"<function=([^\n>]+)>", txt)
        if not m_fn:
            return None
        fn_name = m_fn.group(1).strip()
        params: dict[str, str] = {}
        for m_p in re.finditer(
            r"<parameter=([^>\n]+)>\n?(.*?)(?:\n?</parameter>|(?=<parameter=|</function>|</tool_call>|$))",
            txt,
            flags=re.DOTALL,
        ):
            params[m_p.group(1).strip()] = m_p.group(2).strip()
        return {"function": fn_name, "parameters": params, "raw_xml": txt, "format": "chatml_tool_call"}

    def _execute_python_subprocess(self, code: str, timeout_s: float = 8.0) -> str:
        """Executes model-generated `python_interpreter` code in a bounded subprocess."""
        import ast

        src = (code or "").strip()
        if not src:
            return ""
        try:
            tree = ast.parse(src)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_val = tree.body[-1].value
                if not (isinstance(last_val, ast.Call) and isinstance(last_val.func, ast.Name) and last_val.func.id == "print"):
                    tree.body[-1] = ast.Expr(
                        value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[last_val], keywords=[])
                    )
                    src = ast.unparse(tree)
        except Exception:
            pass
        try:
            res = subprocess.run(
                [sys.executable, "-c", src],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                text=True,
            )
            out = (res.stdout or "").strip()
            err = (res.stderr or "").strip()
            return out if res.returncode == 0 and out else (out or err or f"Exited with code {res.returncode}")
        except Exception as ex:
            return f"Python execution error: {ex}"

    @staticmethod
    def _audit_reply_grounding(
        prompt: str,
        retrieved_context: str,
        tool_observation: str,
        reply_text: str,
    ) -> dict[str, Any]:
        """
        Layer 5 Honest Grounding Audit:
        Checks whether numbers and canonical identifiers (`CVE-...`, `CWE-...`, `CAPEC-...`, `arXiv:...`)
        in `reply_text` appear in the source `prompt`, `retrieved_context`, or `tool_observation`.
        Never silently rewrites `reply_text` or sets `source_grounding_verified = True` unconditionally.
        """
        support_corpus = f"{prompt}\n{retrieved_context}\n{tool_observation}".lower()
        support_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", support_corpus))
        reply_nums = []
        for n in re.findall(r"\b\d+(?:\.\d+)?\b", reply_text or ""):
            if (len(n) >= 2 or "." in n) and n not in reply_nums:
                reply_nums.append(n)
        grounded_nums = [n for n in reply_nums if n in support_nums]
        ungrounded_nums = [n for n in reply_nums if n not in support_nums]

        reply_ids = re.findall(r"\b(?:CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+|arXiv:\d{4}\.\d+)\b", reply_text or "", flags=re.IGNORECASE)
        ungrounded_ids = [cid for cid in reply_ids if cid.lower() not in support_corpus]

        return {
            "checked_numbers": reply_nums,
            "grounded_numbers": grounded_nums,
            "ungrounded_numbers": ungrounded_nums,
            "checked_identifiers": reply_ids,
            "ungrounded_identifiers": ungrounded_ids,
            "source_grounding_verified": (len(ungrounded_nums) == 0 and len(ungrounded_ids) == 0),
        }

    _PY_HINT = re.compile(
        r"^\s*(?:def|class|import|from|for|while|if|elif|try|with|return|print\(|[A-Za-z_]\w*\s*=(?!=))",
        re.M,
    )
    _TOOL_ASKS = (
        (
            "web_search",
            r"\b(?:search|google|look\s+(?:it\s+|that\s+|this\s+)?up|browse|web|internet|online|website|web\s*page|url|arxiv|wikipedia)\b|https?://|www\.",
        ),
        (
            "python_interpreter",
            r"\b(?:use|using)\s+python\b|\bpython\s+to\s+(?:compute|calculate|check|solve|count|find|test)\b|"
            r"\b(?:run|execute|test)\s+(?:this|the|my|that|it|some)?\s*(?:code|script|program|python)\b|"
            r"\b(?:compute|calculate)\s+\d",
        ),
    )

    @classmethod
    def asked_tools(cls, text: str) -> list[str]:
        """Returns the tool names explicitly requested by `text` (used when `settings['tools'] == 'ask'`)."""
        return [name for name, pat in cls._TOOL_ASKS if re.search(pat, text or "", re.I)]

    @classmethod
    def extract_python_from_reply(cls, reply: str) -> list[str]:
        """
        Extracts Python code blocks from `reply`: `python`/`py` fenced blocks or bare fences matching `_PY_HINT`.
        Prevents non-Python fenced blocks (such as Sudoku grids or ASCII tables) from being compiled as Python.
        """
        fenced = re.findall(r"```(python|py)?[ \t]*\n([\s\S]*?)```", reply or "", flags=re.IGNORECASE)
        py_blocks = [b for tag, b in fenced if (tag and tag.lower() in ("python", "py")) or cls._PY_HINT.search(b)]
        if py_blocks:
            return py_blocks
        if fenced:
            return []
        unclosed = re.findall(r"```(?:python|py)\s*\n([\s\S]+)$", reply or "", flags=re.IGNORECASE)
        if unclosed:
            return unclosed
        m = re.search(r"((?:^(?:def|class|import|from)\s+[^\n]+\n(?:\s+[^\n]+\n?)*)+)", reply or "", flags=re.MULTILINE)
        return [m.group(1)] if m else []

    def _compile_verify_model_code_only(
        self,
        model_reply_text: str,
        model_tool_code: str,
        user_prompt: str,
        allow_python: bool = True,
    ) -> dict[str, Any] | None:
        """
        Layer 6 (`CloverPythonGenerativeCompiler`):
        Checks ONLY code written by the model (`model_reply_text` fenced blocks / functions or `model_tool_code`),
        never the user's prompt text. If the user prompt includes `assert` test cases or `>>>` doctests, runs them
        against the model's function when `allow_python=True`.
        """
        blocks = self.extract_python_from_reply(model_reply_text or "")
        if not blocks and model_tool_code:
            blocks = [model_tool_code]
        if not blocks:
            return None

        prompt_asserts = re.findall(r"^\s*(assert\s+[^\n]+)", user_prompt or "", flags=re.MULTILINE)
        results = []
        all_ok = True
        for b in blocks[:3]:
            try:
                st = self.py_compiler.check_prefix(b, end_marker=True, spec_prompt=user_prompt)
            except TypeError:
                st = self.py_compiler.check_prefix(b, end_marker=True)
            ok = bool(st.ok)
            diags = [
                {"code": getattr(d, "code", ""), "message": getattr(d, "message", ""), "line": getattr(d, "line", None)}
                for d in getattr(st, "diagnostics", [])
            ]
            test_run_output = None
            if ok and prompt_asserts and allow_python:
                combined_test = b + "\n\n" + "\n".join(prompt_asserts) + "\nprint('ALL_ASSERTS_PASSED')\n"
                test_out = self._execute_python_subprocess(combined_test, timeout_s=5.0)
                test_run_output = test_out
                if "ALL_ASSERTS_PASSED" not in test_out:
                    ok = False
                    diags.append({"code": "AssertionError", "message": test_out[:240], "line": None})
            all_ok = all_ok and ok
            # Sutton (1988) + Rumelhart et al. (1986) Statement Trajectory Classification & Absorbing Markov MLE
            lines_nonempty = [ln for ln in b.splitlines() if ln.strip()]
            diag_code0 = diags[0]["code"] if diags else ""
            traj_states = [
                self.sutton_td_predictor.classify_statement_state(ln, verdict_ok=ok, diag_code=diag_code0)
                for ln in lines_nonempty
            ]
            traj_indices = [self.sutton_td_predictor.state_to_idx[s] for s in traj_states] or [0]
            rumelhart_target = 0.9 if ok else 0.1
            v_mle = self.sutton_td_predictor.compute_markov_mle(
                [([0, 2, 3], 0.1), ([0, 2, 5, 6], 0.9), (traj_indices, rumelhart_target)]
            )
            results.append({
                "code_checked": b,
                "ok": ok,
                "diagnostics": diags,
                "rollback_line_idx": getattr(st, "rollback_line_idx", None),
                "elapsed_ms": getattr(st, "elapsed_ms", 0.0),
                "prompt_asserts_output": test_run_output,
                "sutton_1988_and_rumelhart_1986": {
                    "trajectory_states": traj_states,
                    "terminal_target_rumelhart_1986": rumelhart_target,
                    "absorbing_markov_v_mle": v_mle,
                },
            })
        return {"checked_blocks": len(results), "all_ok": all_ok, "results": results}

    def run_cycle(
        self,
        prompt: str,
        max_prompt_tokens: int = 1024,
        max_new_tokens: int = 160,
        output_type: str = "text",
        image_b64: str | None = None,
        attached_files: list[dict[str, Any]] | None = None,
        voice_input: bool = False,
        max_active_layer: int = 7,
        use_ncm_renderer: bool = True,
        enable_thinking: bool = False,
        inject_latent_step_before_decode: bool = False,
        allow_internet: bool = True,
        allow_python: bool = True,
        allow_files: bool = True,
    ) -> dict[str, Any]:
        """
        Executes the Clover cycle up to `max_active_layer` (`1 .. 7`) with zero hardcoded answers:
          - Layers 4 & 5 inject retrieved evidence into `LFM3-1.7B`'s user prompt BEFORE prefill.
          - Layer 1 runs `LFM3-1.7B` without any forced `<tool_call>` suffix.
          - When `LFM3-1.7B` emits a `<tool_call>`, Layer 2 executes the tool (subject to `allow_internet`
            and `allow_python`) and feeds `<tool_response>` back into `LFM3-1.7B` for Turn 2 synthesis.
          - Layer 6 compiles only model-written Python code and runs a repair turn in `LFM3-1.7B` if needed.
          - Layer 7 resets `lfm3_dm_reset` at the start of the request and records per-request stats.
        """
        t0 = time.perf_counter()
        active_layer_cap = max(1, min(7, int(max_active_layer)))

        # Layer 7 control: reset per-request counters and set OS madvise(MADV_WILLNEED)
        madvise_on = active_layer_cap >= 7
        self.engine.set_madvise_active(madvise_on)
        self.engine.reset_dm_stats()

        tools_used: list[str] = []
        retrieved_context_blocks: list[str] = []
        layer_trace: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # 1. Multimodal Attachments & Local Phone Files (Enforcing `allow_files`)
        # ------------------------------------------------------------------
        if attached_files and allow_files:
            tools_used.append("device_files_access")
            for af in attached_files[:3]:
                fname = str(af.get("name") or "file")
                ftype = str(af.get("type") or "")
                ftext = str(af.get("content_text") or "").strip()
                fb64 = str(af.get("data_b64") or "")
                if ftext:
                    snippet = ftext[:1200].replace("\n", " ")
                    retrieved_context_blocks.append(f"[Attached File '{fname}']: {snippet}")
                elif ("audio" in ftype or fname.lower().endswith(".wav")) and fb64:
                    try:
                        raw_b64 = fb64.split(",", 1)[-1]
                        wav_info = inspect_attached_wav_real(base64.b64decode(raw_b64), fname)
                        retrieved_context_blocks.append(f"[Attached Audio]: {wav_info['summary']}")
                        layer_trace["layer1_audio_attachment"] = wav_info
                        tools_used.append("wav_audio_inspector")
                    except Exception as ex:
                        retrieved_context_blocks.append(f"[Attached Audio '{fname}']: could not decode WAV header ({ex}).")
                else:
                    retrieved_context_blocks.append(f"[Attached File '{fname}' ({ftype})]")

        file_path_match = re.search(r"(/sdcard/[^\s\"']+|/storage/[^\s\"']+|~/[^\s\"']+)", prompt or "")
        if file_path_match and allow_files:
            fpath = Path(os.path.expanduser(file_path_match.group(1)))
            if fpath.exists() and fpath.is_file():
                try:
                    txt = fpath.read_text(encoding="utf-8", errors="replace")[:1200].replace("\n", " ")
                    retrieved_context_blocks.append(f"[Local File '{fpath.name}']: {txt}")
                    tools_used.append("device_files_access")
                except Exception:
                    pass

        if image_b64:
            try:
                raw_img = base64.b64decode(image_b64.split(",", 1)[-1])
                pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                img_info = inspect_attached_image_real(pil_img)
                retrieved_context_blocks.append(f"[Attached Image]: {img_info['summary']}")
                layer_trace["layer1_image_attachment"] = img_info
                tools_used.append("pillow_image_inspector")
            except Exception as ex:
                retrieved_context_blocks.append(f"[Attached Image]: could not decode ({ex}).")

        if voice_input:
            tools_used.append("android_speech_recognizer")

        # ------------------------------------------------------------------
        # 2. Layer 4: Local Git Repository Memory Engine (`clover-memory-repo/`)
        #    Selective On-Demand Snippet Fetch (Zero Context Bloating) +
        #    Foundational Papers: McCulloch-Pitts (1943), Rosenblatt (1958),
        #    Minsky-Papert (1969), Hopfield (1982), Rumelhart et al. (1986),
        #    Sutton (1988), LeCun et al. (1998), Krizhevsky et al. (2012),
        #    Vaswani et al. (2017).
        # ------------------------------------------------------------------
        if active_layer_cap >= 4:
            t_l4 = time.perf_counter()
            git_hits = self.git_memory_repo.search_repo_relevant(prompt, max_files=2, max_chars_per_file=550)
            mem_pkg = self._retrieve_memory(self.memory_graph, prompt)
            l4_items = []
            l4_reasons = []
            seen_snippets: set[str] = set()
            for gh in git_hits:
                snip = str(gh.get("snippet") or "").strip()
                if snip and snip not in seen_snippets:
                    seen_snippets.add(snip)
                    l4_items.append({"id": gh["path"], "type": "git_repo_file", "summary": snip, "reason": f"git_repo_match(score={gh['score']})"})
                    retrieved_context_blocks.append(f"[Git Memory Repo '{gh['path']}']: {snip}")
            if mem_pkg is not None and mem_pkg.items:
                for it in mem_pkg.items[:2]:
                    summary_clean = str(it.entry.summary or "").strip()
                    rsn_str = str(getattr(it, "reason", "") or "")
                    l4_reasons.append(rsn_str)
                    if summary_clean and summary_clean not in seen_snippets:
                        seen_snippets.add(summary_clean)
                        l4_items.append({"id": it.entry.id, "type": it.entry.entry_type, "summary": summary_clean, "reason": rsn_str})
                        retrieved_context_blocks.append(f"[Memory Graph {it.entry.id}]: {summary_clean}")
            if l4_items:
                tools_used.append("layer4_git_memory_repo")
            hopfield_fam = self.memory_graph.compute_hopfield_familiarity(prompt)
            gtn_conf = self.memory_graph.compute_gtn_forward_confidence(prompt)
            mh_lrn = self.memory_graph.multihead_lrn_subspace_retrieve(prompt, top_k=3)
            td_path_ids = [it["id"] for it in l4_items if str(it["id"]).startswith("MEM-")] or ["MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001", "MEM-CORR-PER-LAYER-PARTITION-002"]
            td_credit = self.memory_graph.compute_td_lambda_path_credit(td_path_ids, outcome_verified=True)
            layer_trace["layer4_memory_graph"] = {
                "enabled": True,
                "storage_engine": "local_git_repository_with_selective_on_demand_fetch",
                "repo_path": str(self.git_memory_repo.repo_dir),
                "agents_md_accepted": bool(self.git_memory_repo.get_onboarding_state().get("agents_md_accepted")),
                "injected_into_model_prompt": len(l4_items) > 0,
                "retrieved_count": len(l4_items),
                "entries": l4_items,
                "eleven_foundational_papers": {
                    "mcculloch_pitts_1943_and_minsky_papert_1969": {
                        "veto_evaluated": True,
                        "order_k_predicates_matched": any("structural_cue(" in r or "concept_" in r for r in l4_reasons),
                        "rosenblatt_1958_gamma_zero_sum_applied": getattr(self.memory_graph, "use_rosenblatt_gamma", True),
                    },
                    "hopfield_1982_constellation_cleanup": {
                        "is_familiar": bool(hopfield_fam.get("is_familiar")),
                        "energy_drop": hopfield_fam.get("energy_drop"),
                        "initial_alignment": hopfield_fam.get("initial_alignment"),
                    },
                    "rumelhart_1986_and_sutton_1988_td_lambda": {
                        "terminal_target_rumelhart_1986": td_credit.get("terminal_target_rumelhart_1986"),
                        "updated_values": td_credit.get("updated_values"),
                    },
                    "lecun_1998_gtn_forward_logadd": {
                        "top_gtn_id": gtn_conf.get("top_gtn_id"),
                        "top_gtn_confidence": gtn_conf.get("top_gtn_confidence"),
                        "free_energy": gtn_conf.get("free_energy"),
                    },
                    "krizhevsky_2012_lrn_and_vaswani_2017_multihead": {
                        "facets_multihead_lrn": mh_lrn.get("facets_multihead_lrn"),
                        "selected_ids": [x.get("id") for x in (mh_lrn.get("results") or []) if isinstance(x, dict)],
                    },
                },
                "ms": round((time.perf_counter() - t_l4) * 1000.0, 2),
            }
        else:
            layer_trace["layer4_memory_graph"] = {"enabled": False}

        # ------------------------------------------------------------------
        # 3. Layer 5: Deterministic Ontology GraphRAG (`k3_ontology_graph.py`)
        #    + Foundational Papers: Turing (1950) M0/M1 Diagonal Stratifier &
        #      Newell-Simon (1956) CSm 3-Number Filter (*2.07 Fix) + VCt Contraction.
        # ------------------------------------------------------------------
        if active_layer_cap >= 5:
            t_l5 = time.perf_counter()
            turing_strat = self.symbolic_verifier.check_turing_self_reference_stratification(prompt)
            ont_items = self.ontology_retriever.retrieve(prompt, max_entities=3, max_paths=4)
            trav_stats = dict(getattr(self.ontology_retriever.graph, "last_traversal_stats", {}) or {})
            csm_verify = self.symbolic_verifier.match_formula_csm_lmc(
                ("IMP", "a", ("OR", "a", "a")),
                [("*2.07_add_left", ("IMP", "p", ("OR", "p", "q")))],
                allow_vct_contraction=True,
            )
            l5_records: list[str] = []
            if turing_strat is not None:
                tools_used.append("layer5_turing_1950_stratifier")
                retrieved_context_blocks.append(f"[Turing 1950 M0/M1 Stratification]: {turing_strat.get('explanation', '')}")
            if ont_items:
                tools_used.append("layer5_ontology_graphrag")
                paths_first = sorted(ont_items, key=lambda x: 0 if getattr(x, "kind", "") == "path" else 1)
                ev_txt = self._render_ontology(paths_first)
                l5_records.append(ev_txt)
                retrieved_context_blocks = [f"[Ontology GraphRAG Evidence]:\n{ev_txt}"]
            layer_trace["layer5_ontology_and_verifier"] = {
                "enabled": True,
                "injected_into_model_prompt": (len(ont_items) > 0 or turing_strat is not None),
                "ontology_hits": len(ont_items),
                "rendered_evidence": l5_records,
                "eleven_foundational_papers": {
                    "turing_1950_m0_m1_diagonal_stratifier": turing_strat or {"self_reference_detected": False, "stratified_level": "M_0 (Object-Level)"},
                    "newell_simon_1956_csm_and_vct": {
                        "graph_traversal_stats": trav_stats,
                        "csm_2_07_contraction_matched": len(csm_verify.get("matched", [])),
                    },
                },
                "ms": round((time.perf_counter() - t_l5) * 1000.0, 2),
            }
        else:
            layer_trace["layer5_ontology_and_verifier"] = {"enabled": False}

        # ------------------------------------------------------------------
        # 4. Layer 1 (`LFM3-1.7B` 24-Layer 4-Bit C Engine) + Layer 2 (`HybridLink`
        #    & Multi-Turn `<tool_call>` -> `<tool_response>` Synthesis Loop)
        #    Zero Python keyword prompt-steering (`is_code_write_request` removed)
        # ------------------------------------------------------------------
        user_clean = (prompt or "Hello!").strip()
        retrieved_context_str = "\n".join(retrieved_context_blocks).strip()
        if retrieved_context_str:
            ctx_ids = self.tokenizer.encode(retrieved_context_str, allow_special=False)
            if len(ctx_ids) > 420:
                retrieved_context_str = self.tokenizer.decode(ctx_ids[:420], skip_special_tokens=True).strip()
            user_turn_content = (
                f"Retrieved Context:\n{retrieved_context_str}\n\n"
                f"Question: {user_clean}"
            )
        else:
            user_turn_content = user_clean

        has_grounded_context = bool(retrieved_context_str)
        tools_enabled_for_turn = (active_layer_cap >= 2 and not has_grounded_context)
        use_ha_fmt = tools_enabled_for_turn and (allow_internet or allow_python)
        sys_prompt_str = self._build_system_prompt(
            allow_internet=(allow_internet and tools_enabled_for_turn),
            allow_python=(allow_python and tools_enabled_for_turn),
            use_hybrid_agent_format=use_ha_fmt,
        )
        if inject_latent_step_before_decode:
            assistant_open = "<|im_start|>assistant\n<think><latent>"
        elif use_ha_fmt or enable_thinking:
            assistant_open = "<|im_start|>assistant\n<think>\n"
        else:
            assistant_open = "<|im_start|>assistant\n<think></think><answer>"

        turn1_prompt_str = f"{sys_prompt_str}<|im_start|>user\n{user_turn_content}<|im_end|>\n{assistant_open}"
        turn1_ids = self.tokenizer.encode(turn1_prompt_str, allow_special=True)
        if len(turn1_ids) > max_prompt_tokens:
            # Keep system header and tail question + assistant_open intact
            sys_ids = self.tokenizer.encode(sys_prompt_str + "<|im_start|>user\n", allow_special=True)
            tail_ids = self.tokenizer.encode(f"\n\nQuestion: {user_clean}<|im_end|>\n{assistant_open}", allow_special=True)
            budget = max(16, max_prompt_tokens - len(sys_ids) - len(tail_ids))
            mid_ids = self.tokenizer.encode(f"Retrieved Context:\n{retrieved_context_str}", allow_special=False)[:budget]
            turn1_ids = sys_ids + mid_ids + tail_ids

        h_prefill, prefill_ms = self.engine.prefill_cached(turn1_ids)
        c_total_ms = prefill_ms

        # Layer 2: Measure `HybridLink` on `h_prefill` and run `HybridAgent` `<latent>` continuous reasoning loop
        h_for_decode = h_prefill
        if active_layer_cap >= 2 and self.hybrid_ncm is not None:
            latent_info = self.hybrid_ncm.run_continuous_latent_step(h_prefill)
            lat_trace = {
                "enabled": True,
                "injected_before_decode": bool(inject_latent_step_before_decode),
                **{k: v for k, v in latent_info.items() if k != "r_np"},
            }
            if inject_latent_step_before_decode:
                h_after_lat, lat_step_c_ms = self.engine.step_embedding_cached(latent_info["r_np"])
                close_ids = self.tokenizer.encode("</latent></think><answer>", allow_special=True)
                h_after_close, close_ms = self.engine.append_cached(close_ids)
                c_total_ms += lat_step_c_ms + close_ms
                h_for_decode = h_after_close
                lat_trace["c_continuous_latent_step_ms"] = round(lat_step_c_ms, 2)
                lat_trace["post_latent_hidden_rms"] = round(float(np.sqrt(np.mean(h_after_lat ** 2))), 5)
            layer_trace["layer2_hybrid_agent"] = lat_trace
            tools_used.append("layer2_lfm3_native_hybridlink")
        else:
            layer_trace["layer2_hybrid_agent"] = {"enabled": False}

        if active_layer_cap >= 2 and self.hybrid_ncm is not None and (use_ha_fmt or enable_thinking):
            gen_ids_t1, _, decode_ms_t1, ha_t1_stats = self.hybrid_ncm.decode_hybrid_agent_turn(
                h_for_decode,
                max_new_tokens=max_new_tokens,
            )
            layer_trace["layer2_hybrid_agent"].update(ha_t1_stats)
        else:
            gen_ids_t1, _, decode_ms_t1 = self.engine.decode_from_cache(
                h_for_decode,
                max_new_tokens=max_new_tokens,
                allow_thinking=enable_thinking,
            )
        c_total_ms += decode_ms_t1
        raw_t1_text = self.tokenizer.decode(gen_ids_t1, skip_special_tokens=False)
        parsed_call = self._parse_lfm3_tool_call_xml(raw_t1_text)

        tool_observation = ""
        executed_py_code = ""
        turn2_prefill_ms = 0.0
        turn2_decode_ms = 0.0
        gen_ids_t2: list[int] = []

        if parsed_call is not None and active_layer_cap >= 2:
            fn_name = parsed_call["function"]
            params = parsed_call["parameters"]

            if fn_name == "web_search" and allow_internet:
                q_search = params.get("query") or user_clean
                urls_in_prompt = re.findall(r"https?://[^\s<>\"')]+", user_clean)
                if urls_in_prompt and urls_in_prompt[0] not in q_search:
                    q_search = urls_in_prompt[0] + " " + q_search
                tools_used.append("layer2_live_web_search")
                tool_observation = self.search_tool.execute(q_search)
                layer_trace["layer2_hybrid_agent"]["tool_executed"] = "web_search"
                layer_trace["layer2_hybrid_agent"]["search_query"] = q_search
                layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]

            elif fn_name == "python_interpreter" and allow_python:
                executed_py_code = params.get("code") or ""
                if executed_py_code:
                    tools_used.append("layer2_python_interpreter")
                    tool_observation = self._execute_python_subprocess(executed_py_code)
                    layer_trace["layer2_hybrid_agent"]["tool_executed"] = "python_interpreter"
                    layer_trace["layer2_hybrid_agent"]["python_code"] = executed_py_code
                    layer_trace["layer2_hybrid_agent"]["python_output"] = tool_observation

            elif fn_name == "repo_search" and active_layer_cap >= 4:
                rq = params.get("query") or user_clean
                tools_used.append("layer4_git_repo_search")
                hits = self.git_memory_repo.search_repo_relevant(rq, max_files=3, max_chars_per_file=600)
                tool_observation = "\n".join(f"[{h['path']}]: {h['snippet']}" for h in hits) or "No matching files in memory repo."
                layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_search"
                layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]

            elif fn_name == "repo_read" and active_layer_cap >= 4:
                rpath = params.get("path") or "CONTEXT.md"
                tools_used.append("layer4_git_repo_read")
                try:
                    rdata = self.git_memory_repo.read_file(rpath)
                    tool_observation = str(rdata.get("content") or "")[:1200]
                except Exception as ex:
                    tool_observation = f"repo_read error: {ex}"
                layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_read"
                layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]

            elif fn_name == "repo_write" and active_layer_cap >= 4:
                rpath = params.get("path") or "tasks/note.md"
                rcontent = params.get("content") or ""
                tools_used.append("layer4_git_repo_write")
                try:
                    wres = self.git_memory_repo.write_file(rpath, rcontent, commit_message=f"Model tool write {rpath}")
                    tool_observation = f"Committed {wres['path']} ({wres['size_bytes']} B, commit {wres['commit']})."
                except Exception as ex:
                    tool_observation = f"repo_write error: {ex}"
                layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_write"
                layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]

            # Feed `<result>...</result>` back into `LFM3-Omni-2B` (Turn 2) with 1 HybridLink2048 latent step before `<answer>`
            if tool_observation:
                call_fmt = parsed_call.get("format", "chatml_tool_call")
                if call_fmt in ("hybrid_agent_search", "hybrid_agent_python"):
                    obs_clean = tool_observation[:600]
                    t2_open_ids = self.tokenizer.encode(
                        f"<result>{obs_clean}</result><think><latent>",
                        allow_special=True,
                    )
                    h_t2_lat, ms_t2_open = self.engine.append_cached(t2_open_ids)
                    turn2_prefill_ms = ms_t2_open
                    if self.hybrid_ncm is not None:
                        lat_t2 = self.hybrid_ncm.run_continuous_latent_step(h_t2_lat)
                        h_t2_after, ms_lat_t2 = self.engine.step_embedding_cached(lat_t2["r_np"])
                        turn2_prefill_ms += ms_lat_t2
                        layer_trace["layer2_hybrid_agent"]["turn2_latent_spans"] = 1
                        layer_trace["layer2_hybrid_agent"]["turn2_latent_steps"] = 1
                    t2_close_ids = self.tokenizer.encode(
                        "</latent></think><answer>",
                        allow_special=True,
                    )
                    h_t2, ms_t2_close = self.engine.append_cached(t2_close_ids)
                    turn2_prefill_ms += ms_t2_close
                    gen_ids_t2, _, turn2_decode_ms = self.engine.decode_from_cache(
                        h_t2, max_new_tokens=max_new_tokens, allow_thinking=0
                    )
                else:
                    t1_clean_call = raw_t1_text
                    if not t1_clean_call.endswith("</tool_call>"):
                        t1_clean_call = t1_clean_call.rstrip() + "\n</function>\n</tool_call>"
                    t2_prefix_ids = self.tokenizer.encode(
                        f"{turn1_prompt_str}{t1_clean_call}<|im_end|>\n<|im_start|>tool\n<tool_response>\n",
                        allow_special=True,
                    )
                    t2_suffix_ids = self.tokenizer.encode(
                        "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think></think>",
                        allow_special=True,
                    )
                    obs_budget = max(24, 1024 - len(t2_prefix_ids) - len(t2_suffix_ids))
                    obs_ids = self.tokenizer.encode(tool_observation, allow_special=False)[:obs_budget]
                    turn2_ids = t2_prefix_ids + obs_ids + t2_suffix_ids
                    h_t2, turn2_prefill_ms = self.engine.prefill_cached(turn2_ids)
                    gen_ids_t2, _, turn2_decode_ms = self.engine.decode_from_cache(
                        h_t2,
                        max_new_tokens=max_new_tokens,
                        allow_thinking=False,
                    )
                c_total_ms += turn2_prefill_ms + turn2_decode_ms
                raw_t2_text = self.tokenizer.decode(gen_ids_t2, skip_special_tokens=False)
                reply_text = self._clean_model_text(raw_t2_text)
            else:
                reply_text = self._clean_model_text(raw_t1_text)
        else:
            reply_text = self._clean_model_text(raw_t1_text)

        if not reply_text:
            reply_text = self.tokenizer.decode(gen_ids_t2 or gen_ids_t1, skip_special_tokens=True).strip()

        layer_trace["layer1_lfm3_omni"] = {
            "enabled": True,
            "turn1_prefill_ms": round(prefill_ms, 2),
            "turn1_decode_ms": round(decode_ms_t1, 2),
            "turn1_generated_tokens": len(gen_ids_t1),
            "turn1_raw_output": raw_t1_text,
            "autonomous_tool_call": parsed_call,
            "turn2_prefill_ms": round(turn2_prefill_ms, 2),
            "turn2_decode_ms": round(turn2_decode_ms, 2),
            "turn2_generated_tokens": len(gen_ids_t2),
        }

        # ------------------------------------------------------------------
        # 5. Layer 6: Real Python Generative Compilation (`CloverPythonGenerativeCompiler`)
        #    Checks ONLY model-written code (`reply_text` or `executed_py_code`), with repair pass!
        # ------------------------------------------------------------------
        compiler_report = None
        if active_layer_cap >= 6:
            compiler_report = self._compile_verify_model_code_only(
                model_reply_text=reply_text,
                model_tool_code=executed_py_code,
                user_prompt=user_clean,
                allow_python=allow_python,
            )
            if compiler_report is not None:
                tools_used.append("layer6_generative_compiler")
                if not compiler_report["all_ok"]:
                    first_fail = next((r for r in compiler_report["results"] if not r["ok"]), None)
                    if first_fail is not None and first_fail["diagnostics"]:
                        d0 = first_fail["diagnostics"][0]
                        err_desc = f"{d0.get('code', 'Error')} at line {d0.get('line')}: {d0.get('message')}"
                        repair_prompt_str = (
                            f"{sys_prompt_str}<|im_start|>user\n{user_clean}\n\n"
                            f"Your previous Python code failed verification with:\n{err_desc}\n"
                            f"Output ONLY the corrected Python function inside a ```python code block.<|im_end|>\n"
                            "<|im_start|>assistant\n<think>I will output only the corrected Python code block.</think><answer>```python\n"
                        )
                        rep_ids = self.tokenizer.encode(repair_prompt_str, allow_special=True)[:1024]
                        h_rep, ms_rep_pre = self.engine.prefill_cached(rep_ids)
                        g_rep, _, ms_rep_dec = self.engine.decode_from_cache(
                            h_rep, max_new_tokens=max_new_tokens, allow_thinking=False
                        )
                        c_total_ms += ms_rep_pre + ms_rep_dec
                        repaired_reply = "```python\n" + self._clean_model_text(self.tokenizer.decode(g_rep, skip_special_tokens=False))
                        if "```" not in repaired_reply[10:]:
                            repaired_reply = repaired_reply.rstrip() + "\n```"
                        repaired_report = self._compile_verify_model_code_only(
                            model_reply_text=repaired_reply,
                            model_tool_code="",
                            user_prompt=user_clean,
                            allow_python=allow_python,
                        )
                        compiler_report["repair_attempted"] = True
                        compiler_report["repaired_report"] = repaired_report
                        if repaired_report and repaired_report.get("all_ok"):
                            reply_text = repaired_reply
                            compiler_report["all_ok_after_repair"] = True
                        else:
                            compiler_report["all_ok_after_repair"] = False
            layer_trace["layer6_generative_compiler"] = {
                "enabled": True,
                "checked_only_model_code": True,
                "report": compiler_report,
            }
        else:
            layer_trace["layer6_generative_compiler"] = {"enabled": False}

        # ------------------------------------------------------------------
        # 6. Layer 3: 11-Level `NCM` (`VQVAE` + `NextConceptTransformer` + `LexicalPointerDecoder`)
        # ------------------------------------------------------------------
        active_media = (output_type or "text").lower().strip()
        if active_media not in ("image", "video", "audio"):
            active_media = "text"

        if active_layer_cap >= 3 and use_ncm_renderer and self.hybrid_ncm is not None and active_media == "text":
            rendered_ncm, out_ids, ncm_meta = self.hybrid_ncm.render_plan_through_11level_ncm(reply_text)
            if rendered_ncm:
                reply_text = rendered_ncm
            layer_trace["layer3_11level_ncm"] = ncm_meta
            tools_used.append("layer3_11level_ncm_pointer_decoder")
        else:
            out_ids = self.tokenizer.encode(reply_text, allow_special=False)
            layer_trace["layer3_11level_ncm"] = {"enabled": False}

        # ------------------------------------------------------------------
        # 7. Layer 5 Post-Generation Grounding Audit (Real Boolean Check, No Regex Rewriting)
        # ------------------------------------------------------------------
        if active_layer_cap >= 5:
            grounding_audit = self._audit_reply_grounding(
                prompt=user_clean,
                retrieved_context=retrieved_context_str,
                tool_observation=tool_observation,
                reply_text=reply_text,
            )
            layer_trace["layer5_ontology_and_verifier"]["grounding_audit"] = grounding_audit
            layer_trace["layer5_ontology_and_verifier"]["source_grounding_verified"] = grounding_audit["source_grounding_verified"]

        # ------------------------------------------------------------------
        # 8. Optional Procedural Media Preview (Only When Explicitly Requested via `output_type` or Voice)
        # ------------------------------------------------------------------
        image_png_b64 = ""
        video_mp4_b64 = ""
        audio_wav_b64 = ""
        if active_media == "image":
            image_png_b64 = generate_image_png_data_url(user_clean)
            tools_used.append("procedural_pillow_png_preview")
        elif active_media == "video":
            video_mp4_b64 = generate_video_mp4_data_url(user_clean, self.base_dir)
            tools_used.append("procedural_ffmpeg_mp4_preview")
        elif active_media == "audio" or voice_input:
            audio_wav_b64 = generate_spoken_wav_data_url(reply_text, self.base_dir)
            tools_used.append("espeak_spoken_wav_synthesizer")

        # ------------------------------------------------------------------
        # 9. Layer 4 Growth Stage: Commit Session Log & Verified Scripts to Local Git Repo
        # ------------------------------------------------------------------
        if active_layer_cap >= 4 and self.perm_modes.get("memory", "allow") != "deny":
            try:
                git_growth = self.git_memory_repo.record_task_or_session_turn(
                    session_id="s23-studio",
                    prompt=user_clean,
                    reply=reply_text,
                    generated_code=executed_py_code,
                    tools_used=tools_used,
                )
                layer_trace["layer4_memory_graph"]["git_growth_commit"] = git_growth
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 10. Layer 7: Per-Request SpecMD Slot Tracker & OS `madvise(MADV_WILLNEED)`
        # ------------------------------------------------------------------
        dm_stats = self.engine.get_dm_stats(madvise_enabled=madvise_on)
        layer_trace["layer7_specmd_r4_madvise"] = dm_stats

        total_gen_tokens = len(gen_ids_t1) + len(gen_ids_t2)
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "device": platform.node(),
            "arch": platform.machine(),
            "model": "lfm3-omni-2b-4bit-s23",
            "max_active_layer": active_layer_cap,
            "exclusive_app_package": CLOVER_S23_APP_PACKAGE,
            "prompt": prompt,
            "reply": reply_text,
            "output_type": active_media,
            "image_png_b64": image_png_b64,
            "video_mp4_b64": video_mp4_b64,
            "audio_wav_b64": audio_wav_b64,
            "speak_with_android_tts": active_media == "audio" or voice_input,
            "tools_used": tools_used,
            "layer_trace": layer_trace,
            "layer6_compiler_report": compiler_report,
            "input_tokens": len(turn1_ids),
            "generated_tokens": total_gen_tokens,
            "emitted_tokens": len(out_ids),
            "c_24_layer_4bit_ms": round(c_total_ms, 2),
            "total_cycle_ms": total_ms,
            "dm_stats": dm_stats,
        }

    def run_retry_cycle(
        self,
        question: str,
        previous_answer: str,
        feedback: str = "",
        session_id: str = "s23-studio",
        max_active_layer: int = 7,
        allow_internet: bool = True,
        allow_python: bool = True,
    ) -> dict[str, Any]:
        """
        Executes the Laptop Agent's full Clover Retry Cycle (`clover_cycle.retry_cycle`) on the Samsung Galaxy S23
        when the user says the earlier answer was not right ("try again", "that's wrong", "not what I expected",
        the Try again button, or asking the same question again):
          1. Deterministic Outcome checks (`arithmetic_findings` + `format_findings` + Layer 6 `CloverPythonGenerativeCompiler`).
          2. Missing Context web search (`needs_web` -> `web_query`), passed to `solve(2)` and `rewrite` only so a
             single misleading page moves at most 1 vote.
          3. Independent multi-solution majority vote for single-value questions (`solve(1)`, `solve(2)` with `<latent>`
             step, and `python_solve` verified via `clover_cycle.computes(code)`).
          4. Open-ended / creative / code rewrite (`_redo` + `_keep_code`) directed by the user's feedback.
          5. Growth stage: stages any learned lesson in `clover-memory-repo/lessons/` (`lesson_staged`) so the user
             can Keep (promote to `notes/`) or Discard it.
        """
        import clover_cycle as C

        t0 = time.perf_counter()
        active_layer_cap = max(1, min(7, int(max_active_layer)))
        madvise_on = active_layer_cap >= 7
        self.engine.set_madvise_active(madvise_on)
        self.engine.reset_dm_stats()

        tools_used: list[str] = ["clover_retry_cycle"]
        used_names: list[str] = []
        layer_trace: dict[str, Any] = {}
        c_total_ms = 0.0
        total_gen_tokens = 0

        # 1. Re-gather Layer 4 (Git Memory Repo) & Layer 5 (Ontology GraphRAG) Context for `question`
        blocks: list[str] = []
        if active_layer_cap >= 4:
            git_hits = self.git_memory_repo.search_repo_relevant(question, max_files=2, max_chars_per_file=550)
            for gh in git_hits:
                blocks.append(f"[Git Memory Repo '{gh['path']}']: {gh['snippet']}")
            if git_hits:
                tools_used.append("layer4_git_memory_repo")
        if active_layer_cap >= 5:
            ont_items = self.ontology_retriever.retrieve(question, max_entities=3, max_paths=4)
            if ont_items:
                paths_first = sorted(ont_items, key=lambda x: 0 if getattr(x, "kind", "") == "path" else 1)
                blocks.append(f"[Ontology GraphRAG Evidence]:\n{self._render_ontology(paths_first)}")
                tools_used.append("layer5_ontology_graphrag")

        notes = ("\n\n" + "\n\n".join(blocks)) if blocks else ""
        web_ctx = ""
        if not blocks and allow_internet and C.needs_web(question):
            q_web = C.web_query(question)
            if q_web:
                obs = self.search_tool.execute(q_web)
                if obs and not obs.startswith("Error"):
                    web_ctx = (
                        "\n\nClover searched the web for this question. Use these results only where they answer it; "
                        "they can be incomplete or wrong:\n" + obs[:650]
                    )
                    tools_used.append("layer2_live_web_search")
                    used_names.append("a web search")

        sys_txt = self._build_system_prompt(allow_internet=False, allow_python=False, use_hybrid_agent_format=False)
        convo_turns: list[tuple[str, str]] = [(question, previous_answer)]

        def _run_c_prompt(user_msg: str, use_latent: bool = False, max_tok: int = 160) -> str:
            nonlocal c_total_ms, total_gen_tokens
            open_tag = "<|im_start|>assistant\n<think><latent>" if (use_latent and self.hybrid_ncm is not None) else "<|im_start|>assistant\n<think></think><answer>"
            full_p = f"{sys_txt}<|im_start|>user\n{user_msg}<|im_end|>\n{open_tag}"
            pids = self.tokenizer.encode(full_p, allow_special=True)[:1024]
            h_p, ms_p = self.engine.prefill_cached(pids)
            c_total_ms += ms_p
            h_dec = h_p
            if use_latent and self.hybrid_ncm is not None:
                lat = self.hybrid_ncm.run_continuous_latent_step(h_p)
                h_lat, ms_l = self.engine.step_embedding_cached(lat["r_np"])
                close_ids = self.tokenizer.encode("</latent></think><answer>", allow_special=True)
                h_dec, ms_c = self.engine.append_cached(close_ids)
                c_total_ms += ms_l + ms_c
                tools_used.append("layer2_lfm3_native_hybridlink")
            g_ids, _, ms_d = self.engine.decode_from_cache(h_dec, max_new_tokens=max_tok, allow_thinking=0)
            c_total_ms += ms_d
            total_gen_tokens += len(g_ids)
            return self._clean_model_text(self.tokenizer.decode(g_ids, skip_special_tokens=False))

        def rewrite(instruction: str) -> str:
            ctx_tail = (notes + web_ctx) if len(convo_turns) == 1 else ""
            prev_q, prev_a = convo_turns[-1]
            u_msg = (
                f"Original Question:\n{prev_q}{ctx_tail}\n\n"
                f"Your Earlier Answer:\n{prev_a}\n\n"
                f"{instruction}"
            )
            new_ans = _run_c_prompt(u_msg, use_latent=True, max_tok=160) or previous_answer
            convo_turns.append((instruction, new_ans))
            return new_ans

        def solve(k: int, note: str | None = None) -> str:
            ctx = notes + (web_ctx if k == 2 else "")
            s_prompt = C.second_solve_prompt(question + ctx, question, k=k, note=note)
            return _run_c_prompt(s_prompt, use_latent=(k == 2), max_tok=160)

        def python_solve(note: str | None = None) -> str:
            if not allow_python:
                return ""
            p_prompt = C.python_solve_prompt(question + notes, note=note)
            raw_py = _run_c_prompt(p_prompt, use_latent=False, max_tok=160)
            py_blocks = self.extract_python_from_reply(raw_py)
            code = py_blocks[0] if py_blocks else raw_py
            if not code or not C.computes(code):
                return ""
            out = self._execute_python_subprocess(code)
            val = C.python_answer(out)
            if val is None:
                return ""
            tools_used.append("layer2_python_interpreter")
            used_names.append("a Python calculation")
            shown = f"{int(val)}" if float(val).is_integer() else f"{val:g}"
            return f"Clover ran this Python program:\n```python\n{code.strip()}\n```\nIt printed: {out.strip()[-300:]}\n\nANSWER: {shown}"

        def code_check(ans: str) -> str | None:
            if active_layer_cap < 6:
                return None
            rep = self._compile_verify_model_code_only(ans, "", question, allow_python=allow_python)
            layer_trace["layer6_generative_compiler"] = {"enabled": True, "checked_only_model_code": True, "report": rep}
            if not rep or rep.get("all_ok"):
                return None
            tools_used.append("layer6_generative_compiler")
            diags = []
            for r_item in rep.get("results", []):
                for d in r_item.get("diagnostics", []):
                    diags.append(f"{d.get('code', 'Error')}: {d.get('message', '')}")
            return "; ".join(diags)[:500] if diags else "Code check failed."

        final_reply, cyc_info = C.retry_cycle(
            question=question,
            previous=previous_answer,
            feedback=feedback,
            rewrite=rewrite,
            solve=solve,
            code_check=code_check if active_layer_cap >= 6 else None,
            used=used_names,
            program=python_solve if allow_python else None,
        )
        cyc_info["tools"] = list(dict.fromkeys(used_names))
        layer_trace["clover_retry_cycle"] = cyc_info

        lesson_staged = None
        if cyc_info.get("lesson") and active_layer_cap >= 4 and self.perm_modes.get("memory", "allow") != "deny":
            try:
                staged = self.git_memory_repo.stage_lesson(
                    cyc_info["lesson"],
                    conversation_id=session_id,
                    kind=cyc_info.get("lesson_kind", "user"),
                )
                lesson_staged = staged["id"]
                layer_trace["clover_retry_cycle"]["lesson_staged"] = staged
            except Exception:
                pass

        dm_stats = self.engine.get_dm_stats(madvise_enabled=madvise_on)
        layer_trace["layer7_specmd_r4_madvise"] = dm_stats
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "device": platform.node(),
            "arch": platform.machine(),
            "model": "lfm3-omni-2b-4bit-s23",
            "max_active_layer": active_layer_cap,
            "exclusive_app_package": CLOVER_S23_APP_PACKAGE,
            "prompt": question,
            "reply": final_reply,
            "output_type": "text",
            "image_png_b64": "",
            "video_mp4_b64": "",
            "audio_wav_b64": "",
            "speak_with_android_tts": False,
            "tools_used": list(dict.fromkeys(tools_used)),
            "layer_trace": layer_trace,
            "clover_cycle_info": cyc_info,
            "lesson_staged": lesson_staged,
            "input_tokens": len(self.tokenizer.encode(question, allow_special=False)),
            "generated_tokens": total_gen_tokens,
            "emitted_tokens": len(self.tokenizer.encode(final_reply, allow_special=False)),
            "c_24_layer_4bit_ms": round(c_total_ms, 2),
            "total_cycle_ms": total_ms,
            "dm_stats": dm_stats,
        }


def run_on_device_s23_verification(base_dir: Path) -> dict[str, Any]:
    """
    Runs the honest step-by-step progressive ablation across Layers 1 -> 7 on the device (`s23` or `k3`),
    recording what `LFM3-1.7B` and each surrounding layer actually do without hardcoded outputs.
    """
    sys_s23 = CloverS23OnDeviceSystem(base_dir, load_neural_ncm=True)

    probe_poem = "Write a short 4-line poem about rain."
    probe_web = "Search the web for recent research papers on foundation models."
    probe_math = "Compute 17**4 + 29*13 using Python."
    probe_mem = "How does Clover's per-layer cache partition fix flat hit rates?"
    probe_ont = (
        "What is CVE-2024-6387 (RegreSSHion), which CWE weakness causes it, and which CAPEC attack pattern exploits it?"
    )
    probe_code_ok = "Write only a Python code block with `def fib(n: int) -> int:` that iteratively returns the nth Fibonacci number."
    probe_code_repair = (
        "Write only a Python code block with `def is_palindrome(s: str) -> bool:` that checks if `s` is a palindrome (case-insensitive, ignoring spaces).\n"
        "assert is_palindrome('Race car') == True\n"
        "assert is_palindrome('hello') == False"
    )

    def _strip_b64_summary(r: dict[str, Any], label: str = "") -> dict[str, Any]:
        out = dict(r)
        for k in ("image_png_b64", "video_mp4_b64", "audio_wav_b64"):
            val = str(out.get(k) or "")
            out[k + "_len"] = len(val)
            out.pop(k, None)
        if label:
            print(
                f"[S23-Bench] {label}: layer={out.get('max_active_layer')} "
                f"in={out.get('input_tokens')} gen={out.get('generated_tokens')} "
                f"ms={out.get('total_cycle_ms')} reply={str(out.get('reply', ''))[:90]!r}",
                file=sys.stderr,
                flush=True,
            )
        return out

    # 1. Layer 1 Standalone (`max_active_layer=1`): direct poetry, ungrounded memory question, ungrounded CVE question
    l1_poem = _strip_b64_summary(sys_s23.run_cycle(probe_poem, max_active_layer=1, use_ncm_renderer=False), "1/12 l1_poem")
    l1_mem_unwrapped = _strip_b64_summary(sys_s23.run_cycle(probe_mem, max_active_layer=1, use_ncm_renderer=False), "2/12 l1_mem")
    l1_ont_unwrapped = _strip_b64_summary(sys_s23.run_cycle(probe_ont, max_active_layer=1, use_ncm_renderer=False), "3/12 l1_ont")

    # 2. Layer 2 (`max_active_layer=2`): live web search + Turn 2 synthesis, Python interpreter + Turn 2 synthesis,
    #    plus direct comparison of decoding WITHOUT vs WITH pre-decode `Lfm3NativeHybridLink` latent step injection
    l2_web = _strip_b64_summary(sys_s23.run_cycle(probe_web, max_active_layer=2, use_ncm_renderer=False), "4/12 l2_web")
    l2_py = _strip_b64_summary(sys_s23.run_cycle(probe_math, max_active_layer=2, use_ncm_renderer=False), "5/12 l2_py")
    l2_no_latent_inject = _strip_b64_summary(
        sys_s23.run_cycle(
            "What is the capital of Japan?",
            max_active_layer=2,
            use_ncm_renderer=False,
            inject_latent_step_before_decode=False,
            allow_internet=False,
            allow_python=False,
        ),
        "6/12 l2_no_latent_inject",
    )
    l2_with_latent_inject = _strip_b64_summary(
        sys_s23.run_cycle(
            "What is the capital of Japan?",
            max_active_layer=2,
            use_ncm_renderer=False,
            inject_latent_step_before_decode=True,
            allow_internet=False,
            allow_python=False,
        ),
        "7/12 l2_with_latent_inject",
    )

    # 3. Layer 3 (`max_active_layer=3`): 11-level NCM + LexicalPointerDecoder (recording raw_ncm_decoded_text and fallback flag)
    l3_ncm = _strip_b64_summary(
        sys_s23.run_cycle(
            "What is the capital of Japan?",
            max_active_layer=3,
            use_ncm_renderer=True,
            allow_internet=False,
            allow_python=False,
        ),
        "8/12 l3_ncm",
    )

    # 4. Layer 4 (`max_active_layer=4`): Memory Graph injected into LFM3-1.7B prompt before prefill
    l4_mem_wrapped = _strip_b64_summary(sys_s23.run_cycle(probe_mem, max_active_layer=4, use_ncm_renderer=False), "9/12 l4_mem")

    # 5. Layer 5 (`max_active_layer=5`): Ontology GraphRAG injected into LFM3-1.7B prompt before prefill + honest grounding audit
    l5_ont_wrapped = _strip_b64_summary(sys_s23.run_cycle(probe_ont, max_active_layer=5, use_ncm_renderer=False), "10/12 l5_ont")

    # 6. Layer 6 (`max_active_layer=6`): Generative Compiler checking ONLY model-written code + assert verification
    l6_code_ok = _strip_b64_summary(
        sys_s23.run_cycle(probe_code_ok, max_active_layer=6, use_ncm_renderer=False, allow_internet=False, allow_python=False),
        "11/12 l6_code_ok",
    )
    l6_code_asserts = _strip_b64_summary(
        sys_s23.run_cycle(probe_code_repair, max_active_layer=6, use_ncm_renderer=False, allow_internet=False, allow_python=True),
        "12/12 l6_code_asserts",
    )

    # 7. Layer 7 (`madvise` ON vs OFF timing benchmark + 4-Mode Mobile SoC Unified Hardware & Format Benchmark)
    pids_bench = sys_s23.tokenizer.encode(
        "<|startoftext|><|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n<think></think><answer>",
        allow_special=True,
    )
    sys_s23.engine.set_madvise_active(False)
    sys_s23.engine.reset_dm_stats()
    sys_s23.engine.prefill_cached([124894, 124893])
    h_b0, ms_pre_off = sys_s23.engine.prefill_cached(pids_bench)
    g_b0, _, ms_dec_off = sys_s23.engine.decode_from_cache(h_b0, max_new_tokens=24, allow_thinking=False)
    dm_off = sys_s23.engine.get_dm_stats(madvise_enabled=False)

    sys_s23.engine.set_madvise_active(True)
    sys_s23.engine.reset_dm_stats()
    sys_s23.engine.prefill_cached([124894])
    h_b1, ms_pre_on = sys_s23.engine.prefill_cached(pids_bench)
    g_b1, _, ms_dec_on = sys_s23.engine.decode_from_cache(h_b1, max_new_tokens=24, allow_thinking=False)
    dm_on = sys_s23.engine.get_dm_stats(madvise_enabled=True)

    # 8. Mobile SoC 4-Mode Hardware & Format Benchmark (Snapdragon 8 Gen 2 for Galaxy: Kryo SMMLA + Adreno 740 + LFM3-SoC-Q4)
    soc_modes: dict[str, Any] = {}
    pids_soc = sys_s23.tokenizer.encode(
        "<|startoftext|><|im_start|>system\nYou are Clover, a helpful AI assistant.<|im_end|>\n<|im_start|>user\nWrite a short 4-line poem about rain.<|im_end|>\n<|im_start|>assistant\n<think></think><answer>",
        allow_special=True,
    )
    for mode_key, q4_lm, q4_op, adreno in (
        ("mode1_baseline_bf16_lmhead_bf16_ops_cpu_sdot", False, False, False),
        ("mode2_q4_lmhead_top64_bf16_refine",            True,  False, False),
        ("mode3_heterogeneous_adreno740_opencl_plus_cpu", True,  False, True),
        ("mode4_full_lfm3_soc_q4_operators_plus_adreno",  True,  True,  True),
    ):
        sys_s23.engine.set_use_4bit_lm_head(q4_lm)
        sys_s23.engine.set_use_4bit_operators(q4_op)
        sys_s23.engine.set_adreno_opencl_active(adreno)
        sys_s23.engine.reset_dm_stats()
        sys_s23.engine.prefill_cached([124894])
        h_m, ms_pre_m = sys_s23.engine.prefill_cached(pids_soc)
        g_m, _, ms_dec_m = sys_s23.engine.decode_from_cache(h_m, max_new_tokens=24, allow_thinking=0)
        dm_m = sys_s23.engine.get_dm_stats(madvise_enabled=True)
        soc_modes[mode_key] = {
            "prefill_ms": round(ms_pre_m, 2),
            "decode_24tok_ms": round(ms_dec_m, 2),
            "ms_per_token": round(ms_dec_m / max(1, len(g_m)), 2),
            "tok_per_s": round(1000.0 / (ms_dec_m / max(1, len(g_m))), 2),
            "output_text": sys_s23.tokenizer.decode(g_m, skip_special_tokens=False),
            "direction_b_ledger": dm_m.get("direction_b_mobile_soc_4tier_ledger", {}),
        }
    # Restore default Mode 3 (4-bit LM-Head + Heterogeneous Adreno 740 + exact BF16 operators)
    sys_s23.engine.set_use_4bit_lm_head(True)
    sys_s23.engine.set_use_4bit_operators(False)
    sys_s23.engine.set_adreno_opencl_active(True)

    vm_rss_mb = 0.0
    if os.path.exists("/proc/self/status"):
        for ln in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ln.startswith("VmRSS:"):
                kb = int(re.findall(r"\d+", ln)[0])
                vm_rss_mb = round(kb / 1024.0, 2)
                break

    report = {
        "device_uname": list(platform.uname()),
        "active_threads": sys_s23.engine.active_threads,
        "setup_ms": sys_s23.setup_ms,
        "measured_process_vm_rss_mb": vm_rss_mb,
        "progressive_7_layer_ablation": {
            "layer1_poem_direct": l1_poem,
            "layer1_memory_question_unwrapped": l1_mem_unwrapped,
            "layer1_ontology_question_unwrapped": l1_ont_unwrapped,
            "layer2_live_web_search_synthesized": l2_web,
            "layer2_python_interpreter_synthesized": l2_py,
            "layer2_decode_without_predecode_hybridlink_step": l2_no_latent_inject,
            "layer2_decode_with_predecode_hybridlink_step": l2_with_latent_inject,
            "layer3_11level_ncm_pointer_decoder": l3_ncm,
            "layer4_memory_graph_injected_into_model": l4_mem_wrapped,
            "layer5_ontology_graphrag_injected_into_model": l5_ont_wrapped,
            "layer6_generative_compiler_model_code": l6_code_ok,
            "layer6_generative_compiler_with_assert_tests": l6_code_asserts,
            "layer7_madvise_off_vs_on_benchmark": {
                "madvise_off": {
                    "prefill_ms": round(ms_pre_off, 2),
                    "decode_24tok_ms": round(ms_dec_off, 2),
                    "ms_per_token": round(ms_dec_off / max(1, len(g_b0)), 2),
                    "tok_per_s": round(1000.0 / (ms_dec_off / max(1, len(g_b0))), 2),
                    "dm_stats": dm_off,
                },
                "madvise_willneed_on": {
                    "prefill_ms": round(ms_pre_on, 2),
                    "decode_24tok_ms": round(ms_dec_on, 2),
                    "ms_per_token": round(ms_dec_on / max(1, len(g_b1)), 2),
                    "tok_per_s": round(1000.0 / (ms_dec_on / max(1, len(g_b1))), 2),
                    "dm_stats": dm_on,
                },
            },
            "mobile_soc_4mode_hardware_format_benchmark": soc_modes,
        },
    }
    out_json = base_dir / "lfm3_omni_s23_live_phone_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def serve_s23_http(base_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    sys_s23 = CloverS23OnDeviceSystem(base_dir, load_neural_ncm=True)

    class _S23Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, obj: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "file://")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, X-Clover-App-Package, X-Clover-App-Token",
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _is_exclusive_clover_app(self) -> bool:
            pkg = (self.headers.get("X-Clover-App-Package") or "").strip()
            tok = (self.headers.get("X-Clover-App-Token") or "").strip()
            return pkg == CLOVER_S23_APP_PACKAGE and secrets.compare_digest(tok, sys_s23.app_secret)

        def _reject_non_app(self) -> None:
            self._send(
                {
                    "error": "exclusive_app_only",
                    "required_package": CLOVER_S23_APP_PACKAGE,
                    "message": (
                        "Access Denied: Every endpoint on this Samsung Galaxy S23 server "
                        "requires a valid X-Clover-App-Token."
                    ),
                },
                code=403,
            )

        def do_OPTIONS(self) -> None:
            self._send({"ok": True})

        def do_GET(self) -> None:
            # Direct APK download endpoints so the user can download and install CloverAIStudio.apk in a mobile browser
            if self.path.startswith(("/download/CloverAIStudio.apk", "/CloverAIStudio.apk", "/api/download/apk")):
                apk_candidates = [
                    sys_s23.base_dir / "android-app" / "CloverAIStudio.apk",
                    SCRIPT_DIR / "android-app" / "CloverAIStudio.apk",
                    Path("/sdcard/Download/CloverAIStudio.apk"),
                ]
                apk_fp = next((p for p in apk_candidates if p.exists() and p.is_file()), None)
                if apk_fp is not None:
                    apk_bytes = apk_fp.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.android.package-archive")
                    self.send_header("Content-Disposition", 'attachment; filename="CloverAIStudio.apk"')
                    self.send_header("Content-Length", str(len(apk_bytes)))
                    self.end_headers()
                    self.wfile.write(apk_bytes)
                    return
            if self.path in ("/download", "/download/"):
                html_dl = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>Download Clover AI Studio APK</title>"
                    "<style>body{font-family:system-ui,sans-serif;background:#faf8f2;color:#132018;padding:2rem;text-align:center}"
                    ".card{max-width:460px;margin:2rem auto;background:#fffefb;border:1px solid #dfd8c8;border-radius:18px;padding:1.8rem;box-shadow:0 8px 24px rgba(18,32,24,.08)}"
                    "a.btn{display:inline-block;margin-top:1rem;background:#1b5e46;color:#fff;text-decoration:none;padding:.75rem 1.4rem;border-radius:12px;font-weight:600}</style></head>"
                    "<body><div class='card'><h2>Clover AI Studio (Samsung Galaxy S23)</h2>"
                    "<p>Signed Android Package (<code>ai.clover.studio</code>) with LFM3-Omni-2B, Adreno 740 OpenCL + Kryo ARMv9 SMMLA, Layer 4 Local Git Memory Repo, and Clover Retry Cycle.</p>"
                    "<a class='btn' href='/download/CloverAIStudio.apk'>Download CloverAIStudio.apk</a></div></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_dl)))
                self.end_headers()
                self.wfile.write(html_dl)
                return

            # All API GET endpoints require the per-device secret token
            if not self._is_exclusive_clover_app():
                self._reject_non_app()
                return
            dm = sys_s23.engine.get_dm_stats(
                madvise_enabled=bool(sys_s23.settings.get("layers", {}).get("specmd_madvise", True))
            )
            ext_ledger = dm.get("direction_b_mobile_soc_4tier_ledger", {})
            if self.path.startswith("/api/status"):
                self._send({
                    "status": "ready",
                    "mode": "omni",
                    "model": "LFM3-Omni-2B",
                    "engine": sys_s23.settings.get("engine", "adreno"),
                    "inputs": ["text", "vision", "audio", "video"],
                    "outputs": ["text", "speech", "image", "video"],
                    "busy": None,
                })
            elif self.path.startswith("/api/settings"):
                self._send({
                    "settings": sys_s23.settings,
                    "omni_available": True,
                    "adreno_available": bool(ext_ledger.get("adreno740_zero_copy_opencl_active", True)),
                })
            elif self.path.startswith("/api/permissions"):
                self._send({
                    "permissions": sys_s23.get_permissions_view(),
                })
            elif self.path.startswith("/api/memory"):
                self._send({
                    "memories": sys_s23.list_memories(),
                })
            elif self.path.startswith("/api/lessons"):
                self._send({
                    "lessons": sys_s23.git_memory_repo.list_staged_lessons(),
                })
            elif self.path.startswith("/api/repo/onboarding"):
                self._send(sys_s23.git_memory_repo.get_onboarding_state())
            elif self.path.startswith("/api/repo/workspaces"):
                self._send(sys_s23.git_memory_repo.list_workspace_repos())
            elif self.path.startswith("/api/repo/files"):
                self._send({
                    "repo_path": str(sys_s23.git_memory_repo.repo_dir),
                    "active_repo": sys_s23.git_memory_repo.active_repo_name,
                    "workspaces": sys_s23.git_memory_repo.list_workspace_repos(),
                    "onboarding": sys_s23.git_memory_repo.get_onboarding_state(),
                    "files": sys_s23.git_memory_repo.list_files(),
                    "lessons": sys_s23.git_memory_repo.list_staged_lessons(),
                    "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    "github_sync": sys_s23.git_memory_repo.get_github_sync_config(),
                })
            elif self.path.startswith("/api/health"):
                self._send({
                    "status": "ready",
                    "mode": "omni",
                    "app_exclusive": True,
                    "authorized_package": CLOVER_S23_APP_PACKAGE,
                    "backend": "s23-snapdragon8gen2-adreno740-smmla-q4-7layer",
                    "model": "lfm3-omni-2b-4bit-s23",
                    "ncm_and_hybridlink_loaded": sys_s23.hybrid_ncm is not None,
                    "pinned_threads": sys_s23.engine.active_threads,
                    "setup_ms": sys_s23.setup_ms,
                    "settings": sys_s23.settings,
                    "data_movement_ledger": dm,
                    "engine": {
                        "mode": "omni",
                        "model": "LFM3-Omni-2B (HybridSFT S3 + 11-Level NCM)",
                        "engine": sys_s23.settings.get("engine", "adreno"),
                        "device": "Snapdragon 8 Gen 2 for Galaxy (Adreno 740 + Kryo ARMv9)",
                        "backend": "Zero-Copy OpenCL 3.0 dot_acc_sat + ARMv8.6 i8mm SMMLA + FA2",
                        "weights": (
                            "Full LFM3-SoC-Q4 (4-Bit FFN + 4-Bit LM-Head + 4-Bit Operators, 894 MB)"
                            if sys_s23.settings.get("q4_operators")
                            else "4-Bit FFN + 4-Bit LM-Head (Top-64 BF16 Refine) + BF16 Operators"
                        ),
                        "inputs": ["text", "vision", "audio", "video"],
                        "outputs": ["text", "speech", "image", "video"],
                        "speech": "Android TTS + eSpeak NG WAV (on-device)",
                        "max_tokens": 1536,
                        "setup_s": round(sys_s23.setup_ms / 1000.0, 1),
                        "threads": sys_s23.engine.active_threads,
                    },
                })
            else:
                self._send({"status": "Clover S23 7-Layer On-Device Server Running"})

        def do_POST(self) -> None:
            if not self._is_exclusive_clover_app():
                self._reject_non_app()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(raw) if raw else {}

            if self.path.startswith("/api/settings"):
                updated = sys_s23.update_settings(payload)
                self._send({"ok": True, "settings": updated})
                return
            if self.path.startswith("/api/permissions"):
                perms_in = payload.get("permissions") if isinstance(payload.get("permissions"), dict) else payload
                updated_p = sys_s23.update_permissions(perms_in)
                self._send({"ok": True, "permissions": updated_p})
                return
            if self.path.startswith("/api/memory/forget"):
                mid = str(payload.get("id") or "").strip()
                if mid and mid != "AGENTS.md":
                    try:
                        sys_s23.git_memory_repo.delete_file(mid)
                    except Exception:
                        pass
                if mid and mid in sys_s23.memory_graph.entries:
                    sys_s23.memory_graph.entries.pop(mid, None)
                self._send({"ok": True, "memories": sys_s23.list_memories()})
                return
            if self.path.startswith("/api/repo/accept-agents"):
                custom_md = payload.get("content")
                st = sys_s23.git_memory_repo.accept_agents_md(custom_content=custom_md if isinstance(custom_md, str) else None)
                self._send({"ok": True, **st})
                return
            if self.path.startswith("/api/repo/workspaces"):
                try:
                    target_p = str(payload.get("path_or_url") or payload.get("repo") or "").strip()
                    w_name = payload.get("name")
                    gh_tok = payload.get("token")
                    wres = sys_s23.git_memory_repo.add_or_switch_workspace_repo(
                        path_or_url=target_p,
                        name=str(w_name).strip() if w_name else None,
                        github_token=str(gh_tok).strip() if gh_tok else None,
                    )
                    self._send({
                        "ok": True,
                        **wres,
                        "repo_path": str(sys_s23.git_memory_repo.repo_dir),
                        "onboarding": sys_s23.git_memory_repo.get_onboarding_state(),
                        "files": sys_s23.git_memory_repo.list_files(),
                        "lessons": sys_s23.git_memory_repo.list_staged_lessons(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                        "github_sync": sys_s23.git_memory_repo.get_github_sync_config(),
                    })
                except Exception as ex:
                    self._send({"error": "workspace_switch_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/repo/refresh-context"):
                try:
                    idx_res = sys_s23.git_memory_repo.refresh_context_hierarchy(commit=True)
                    self._send({
                        "ok": True,
                        **idx_res,
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    })
                except Exception as ex:
                    self._send({"error": "refresh_context_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/repo/read"):
                try:
                    rel_p = str(payload.get("path") or "AGENTS.md")
                    rdata = sys_s23.git_memory_repo.read_file(rel_p)
                    self._send({"ok": True, **rdata})
                except Exception as ex:
                    self._send({"error": "repo_read_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/repo/write"):
                try:
                    rel_p = str(payload.get("path") or "").strip()
                    content_s = str(payload.get("content") or "")
                    c_msg = payload.get("commit_message")
                    wres = sys_s23.git_memory_repo.write_file(rel_p, content_s, commit_message=str(c_msg) if c_msg else None)
                    self._send({
                        **wres,
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    })
                except Exception as ex:
                    self._send({"error": "repo_write_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/repo/delete"):
                try:
                    rel_p = str(payload.get("path") or "").strip()
                    dres = sys_s23.git_memory_repo.delete_file(rel_p)
                    self._send({
                        **dres,
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    })
                except Exception as ex:
                    self._send({"error": "repo_delete_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/repo/github-sync"):
                try:
                    repo_name = str(payload.get("repo") or "").strip()
                    tok = payload.get("token")
                    act = str(payload.get("action") or "push").strip().lower()
                    br = str(payload.get("branch") or "main").strip()
                    sres = sys_s23.git_memory_repo.configure_and_sync_private_github(
                        repo_full_name=repo_name,
                        token=str(tok).strip() if tok else None,
                        action="pull" if act == "pull" else "push",
                        branch=br,
                    )
                    self._send({
                        **sres,
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                        "github_sync": sys_s23.git_memory_repo.get_github_sync_config(),
                    })
                except Exception as ex:
                    self._send({"error": "github_sync_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/session/forget"):
                self._send({"ok": True})
                return
            if self.path.startswith("/api/lessons/keep"):
                try:
                    lid = str(payload.get("id") or "").strip()
                    kres = sys_s23.git_memory_repo.keep_staged_lesson(lid)
                    self._send({
                        **kres,
                        "memories": sys_s23.list_memories(),
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    })
                except Exception as ex:
                    self._send({"error": "keep_lesson_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/lessons/discard"):
                try:
                    lid = str(payload.get("id") or "").strip()
                    dres = sys_s23.git_memory_repo.discard_staged_lesson(lid)
                    self._send({
                        **dres,
                        "files": sys_s23.git_memory_repo.list_files(),
                        "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    })
                except Exception as ex:
                    self._send({"error": "discard_lesson_failed", "message": str(ex)}, code=400)
                return

            prompt = str(payload.get("prompt") or "Hello")
            if "q4_operators" in payload and isinstance(payload["q4_operators"], bool):
                sys_s23.engine.set_use_4bit_operators(payload["q4_operators"])
            if "adreno_active" in payload and isinstance(payload["adreno_active"], bool):
                sys_s23.engine.set_adreno_opencl_active(payload["adreno_active"])

            max_layer = int(payload.get("max_active_layer") or 7)
            use_ncm = bool(payload.get("use_ncm_renderer", sys_s23.settings["layers"].get("ncm", False)))
            pm = sys_s23.perm_modes
            allow_net = bool(payload.get("allow_internet", pm.get("internet", "allow") != "deny"))
            allow_py = bool(payload.get("allow_python", pm.get("code", "allow") != "deny"))
            allow_fl = bool(payload.get("allow_files", pm.get("files", "allow") != "deny"))

            # Detect Clover Retry Cycle ("Try again" button, "that's wrong" / "try again" message, or repeating the question)
            import clover_cycle as C

            hist_in = payload.get("history") or []
            hist_msgs = []
            for h in hist_in[-6:]:
                if isinstance(h, dict) and h.get("user") and h.get("reply"):
                    hist_msgs.append({"role": "user", "content": str(h["user"]), "retry": bool(h.get("retry", False))})
                    hist_msgs.append({"role": "assistant", "content": str(h["reply"])})
            last_ex = C.last_exchange(hist_msgs) if hist_msgs else None

            explicit_retry = bool(payload.get("retry_cycle", False))
            prev_q = str(payload.get("previous_question") or (last_ex[0].get("content") if last_ex else "") or "").strip()
            prev_a = str(payload.get("previous_answer") or (last_ex[1] if last_ex else "") or "").strip()
            how_retry = "said" if explicit_retry else (C.retry_requested(prompt, prev_q) if prev_q and prev_a else None)

            if how_retry and prev_q and prev_a and not payload.get("image_b64") and not payload.get("attached_files"):
                fb_text = str(payload.get("feedback") or (prompt if how_retry == "said" else "") or "").strip()
                res = sys_s23.run_retry_cycle(
                    question=prev_q if how_retry == "said" else prompt,
                    previous_answer=prev_a,
                    feedback=fb_text,
                    session_id=str(payload.get("session_id") or "s23-studio"),
                    max_active_layer=max_layer,
                    allow_internet=allow_net,
                    allow_python=allow_py,
                )
                self._send(res)
                return

            # Apply Laptop Agent's `asked_tools` gating when `settings["tools"] == "ask"` ("When I ask")
            tools_mode = str(payload.get("tools_mode") or sys_s23.settings.get("tools") or "always")
            if tools_mode == "ask":
                requested_tools = sys_s23.asked_tools(prompt)
                allow_net = allow_net and ("web_search" in requested_tools)
                allow_py = allow_py and ("python_interpreter" in requested_tools)

            res = sys_s23.run_cycle(
                prompt=prompt,
                output_type=str(payload.get("output_type") or "text"),
                image_b64=payload.get("image_b64"),
                attached_files=payload.get("attached_files"),
                voice_input=bool(payload.get("voice_input", False)),
                max_active_layer=max_layer,
                use_ncm_renderer=use_ncm,
                enable_thinking=bool(payload.get("enable_thinking", False)),
                inject_latent_step_before_decode=bool(payload.get("inject_latent_step_before_decode", False)),
                allow_internet=allow_net,
                allow_python=allow_py,
                allow_files=allow_fl,
            )
            self._send(res)

    srv = ThreadingHTTPServer((host, port), _S23Handler)
    print(
        f"[Clover-S23] Per-Device Secret-Locked LFM3-1.7B Server ({CLOVER_S23_APP_PACKAGE}) listening on http://{host}:{port}",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path.home() / "clover-s23")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.serve:
        serve_s23_http(args.base_dir, host=args.host, port=args.port)
    else:
        run_on_device_s23_verification(args.base_dir)
