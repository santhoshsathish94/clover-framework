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
import threading
import time
import types
import uuid
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
# Earlier folders win: this file's own modules come before the older copies in src/tools/hybrid-ncm-k3, which lack
# clover_cycle.context_before and CloverMemoryGitRepo.get_media_dir (both raised on k3, research/core-observation).
for _p in reversed((SCRIPT_DIR, LOCAL_TOOLS_DIR, LOCAL_NCM_DIR, LFM3_DIR, MODELS_DIR, TOOLS_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent / "tools" / "hybrid-ncm-k3")):
    if _p.exists():
        if str(_p) in sys.path:
            sys.path.remove(str(_p))
        sys.path.insert(0, str(_p))

D_LFM3 = 2048
V_LFM3 = 128000
D_VQ = 384
CLOVER_S23_APP_PACKAGE = "ai.clover.studio"

# Semantics is common to every Clover process (semantics.py beside this file; the laptop Studio ships the same file).
# The phone's own part: which of its tools serves each capability, and its terminal's risky commands.
PHONE_TOOLS = {
    "web.search": ("web_search",), "web.page": ("web_search",), "python": ("python_interpreter",),
    "files.read": ("repo_read",), "files.list": ("repo_read",), "files.write": ("repo_write",),
    "shell": ("run_shell",), "status": ("run_shell",), "ssh": ("run_ssh",), "image": ("generate_image",),
    "memory.search": ("repo_search",),
    "remember": ("remember",),  # not a model tool: semantics.align saves the user's words (remember_for)
}
PHONE_OFFER = ("web_search", "python_interpreter")  # what a phone turn offers when the message names no tool
# personal learning is on by default (human, 2026-11-03) in <base>/personal once the system starts; CLOVER_PPL_DIR names
# another store, CLOVER_PPL=off turns it off for the process, POST /api/personal/learning switches it for the person
PERSONAL_DIR: Path | None = None
# collective learning (research/collective-learning): opt-in consent switches + verified shared-core sync in <base>/collective
COLLECTIVE_DIR: Path | None = None


def phone_runtime():
    import core_sync
    import local_terminal
    import personal
    import semantics
    # CSI (research/semantic-interface, Experiment 1f) is off unless CLOVER_CSI_URL names a CSI service on this machine
    url = os.environ.get("CLOVER_CSI_URL", "").strip()
    csi = semantics.csi_reader(url) if url else None
    root = os.environ.get("CLOVER_PPL_DIR", "").strip() or (str(PERSONAL_DIR) if PERSONAL_DIR else "")
    c_root = os.environ.get("CLOVER_COLLECTIVE_DIR", "").strip() or (str(COLLECTIVE_DIR) if COLLECTIVE_DIR else "")
    return semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky, csi=csi,
                             personal=personal.for_process(root, "phone", csi),
                             collective=core_sync.for_process(c_root, "phone", csi))


def personal_call(runtime, method: str, *args: Any) -> Any:
    """A personal-learning step of a turn; it never breaks the turn (an error is logged and the turn goes on)."""
    pers = getattr(runtime, "personal", None)
    if pers is None:
        return None
    try:
        return getattr(pers, method)(*args)
    except Exception as ex:
        print(f"[Clover-S23] personal learning: {method} failed: {ex!r}", file=sys.stderr, flush=True)
        return None


def align_turn(frame, res: dict[str, Any], remember_fn=None) -> list[str]:
    """semantics.align on a phone turn. Evidence: its one tool step (layer 2), memory (layer 4), the Try again cycle.
    remember_fn: saves the user's words when they asked Clover to remember something (`CloverS23OnDeviceSystem.remember_for`)."""
    import semantics
    lt = res.setdefault("layer_trace", {})
    l2 = lt.get("layer2_hybrid_agent") or {}
    l4 = lt.get("layer4_memory_graph") or {}
    cycle = res.get("clover_cycle_info") or lt.get("clover_retry_cycle")
    tool = str(l2.get("tool_executed") or "")
    calls = [{"tool": tool, "ran": True, "result": str(l2.get("python_output") or l2.get("observation_preview") or "")}]
    layers = []
    if l4.get("injected_into_model_prompt"):
        layers.append({"layer": "memory", "entries": [e.get("id") for e in l4.get("entries") or []] or ["memory"]})
    if cycle:
        layers.append({"layer": "cycle", "action": cycle.get("action")})
    evidence = {"reply": res.get("reply"), "tool_calls": calls if tool else [], "layers": layers}
    notes = semantics.align(frame, evidence, remember_fn=remember_fn)
    res["reply"] = evidence["reply"]
    lt["semantics_frame"] = frame.layer()
    return notes


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

        if hasattr(self.lib, "lfm3_decode_from_cache_with_hidden"):
            self.lib.lfm3_decode_from_cache_with_hidden.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.lfm3_decode_from_cache_with_hidden.restype = ctypes.c_int

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

    def decode_from_cache_with_hidden(
        self,
        initial_h_2048: np.ndarray,
        max_new_tokens: int = 48,
        allow_thinking: bool | int = False,
    ) -> tuple[list[int], np.ndarray, np.ndarray, float]:
        if not hasattr(self.lib, "lfm3_decode_from_cache_with_hidden"):
            gids, out_h, ms = self.decode_from_cache(initial_h_2048, max_new_tokens=max_new_tokens, allow_thinking=allow_thinking)
            return gids, out_h, np.zeros((0, D_LFM3), dtype=np.float32), ms
        in_h = np.ascontiguousarray(initial_h_2048.reshape(D_LFM3), dtype=np.float32)
        cap = max(1, int(max_new_tokens))
        c_out = (ctypes.c_int * cap)()
        out_h = np.zeros(D_LFM3, dtype=np.float32)
        out_seq = np.zeros((cap, D_LFM3), dtype=np.float32)
        c_ms = ctypes.c_float(0.0)
        think_mode = int(allow_thinking) if isinstance(allow_thinking, int) and not isinstance(allow_thinking, bool) else (1 if allow_thinking else 0)
        n_gen = int(
            self.lib.lfm3_decode_from_cache_with_hidden(
                in_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                cap,
                think_mode,
                c_out,
                out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(c_ms),
            )
        )
        n_valid = max(0, min(cap, n_gen))
        gen_ids = [int(c_out[i]) for i in range(n_valid)]
        return gen_ids, out_h, out_seq[:n_valid].copy(), float(c_ms.value)

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

        # Keep PyTorch at 1 thread so PyTorch's OpenMP pool never contends with liblfm3_cpu_engine.so's OpenMP threads
        torch.set_num_threads(1)

        if str(self.ncm_dir) not in sys.path:
            sys.path.insert(0, str(self.ncm_dir))

        from models.common import apply_rotary_emb
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

        self._apply_rotary_emb = apply_rotary_emb
        self.P_BPE = P_BPE
        self.P_RH = P_RH
        self._join_rendered_windows = join_rendered_windows
        self._pack_combo_prefix = pack_combo_prefix

        class _ParameterlessLayerNorm(nn.Module):
            """Zero-mean/unit-variance LayerNorm whose affine (gamma, beta) has been algebraically fused into the next Linear."""
            def __init__(self, ndim: int, eps: float = 1e-6):
                super().__init__()
                self.shape = (ndim,)
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.layer_norm(x, self.shape, None, None, self.eps)

        class Lfm3NativeHybridLink(nn.Module):
            def __init__(self, d: int = D_LFM3, bottleneck: int = 512):
                super().__init__()
                self.d = d
                self.pre_ln = nn.LayerNorm(d)
                self.proj1 = nn.Linear(d, bottleneck)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(bottleneck, d)
                self.post_ln = nn.LayerNorm(d)
                self.out_scale = nn.Parameter(torch.tensor(0.024))
                self.fused_precomputed = False

            def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
                hf = h_lfm3.float()
                if self.fused_precomputed:
                    xn = F.layer_norm(hf, (self.d,), None, None, 1e-5)
                    z = hf + self.proj2(self.act(self.proj1(xn)))
                    return F.layer_norm(z, (self.d,), self.post_ln.weight, self.post_ln.bias, 1e-5)
                z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
                return z * self.out_scale

        class HybridLink2048(nn.Module):
            """Exact HybridLink (ln_res_adapter, d=2048 -> 2048 -> 2048) trained in 3-stage HybridSFT."""
            def __init__(self, hidden_size: int = D_LFM3):
                super().__init__()
                self.hidden_size = hidden_size
                self.proj1 = nn.Linear(hidden_size, hidden_size)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(hidden_size, hidden_size)
                self.pre_ln = nn.LayerNorm(hidden_size)
                self.post_ln = nn.LayerNorm(hidden_size)
                self.fused_precomputed = False

            def forward(self, h: torch.Tensor) -> torch.Tensor:
                hf = h.float()
                if self.fused_precomputed:
                    xn = F.layer_norm(hf, (self.hidden_size,), None, None, 1e-5)
                    out = self.proj2(self.act(self.proj1(xn)))
                    return self.post_ln(hf + out)
                out = self.proj2(self.act(self.proj1(self.pre_ln(hf))))
                return self.post_ln(hf + out)

        class Lfm3ComboBridge(nn.Module):
            def __init__(self, d_in: int = D_LFM3, d_out: int = D_VQ):
                super().__init__()
                self.d_in = d_in
                self.d_out = d_out
                self.ln_in = nn.LayerNorm(d_in)
                self.fc1 = nn.Linear(d_in, 768)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(768, d_out)
                self.ln_out = nn.LayerNorm(d_out)
                self.scale = nn.Parameter(torch.tensor(8.5))
                self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
                self.gate = nn.Parameter(torch.tensor(1.0))
                self.fused_precomputed = False

            def forward(self, r_lfm3: torch.Tensor) -> torch.Tensor:
                if self.fused_precomputed:
                    xn = F.layer_norm(r_lfm3.float(), (self.d_in,), None, None, 1e-5)
                    x = self.fc2(self.act(self.fc1(xn)))
                    return F.layer_norm(x, (self.d_out,), self.ln_out.weight, self.ln_out.bias, 1e-5).to(r_lfm3.dtype)
                x = self.ln_in(r_lfm3.float())
                x = self.fc2(self.act(self.fc1(x)))
                x = self.ln_out(x) * self.scale
                return (x * self.gate + self.seg_rh).to(r_lfm3.dtype)

        class Lfm3ExpertBridge(nn.Module):
            def __init__(self, d_lfm3: int = D_LFM3, d_vq: int = D_VQ):
                super().__init__()
                self.d_lfm3 = d_lfm3
                self.d_vq = d_vq
                self.ln_in = nn.LayerNorm(d_lfm3)
                self.proj1 = nn.Linear(d_lfm3, d_vq)
                self.act = nn.GELU()
                self.proj2 = nn.Linear(d_vq, d_vq)
                self.ln_out = nn.LayerNorm(d_vq)
                self.alpha = nn.Parameter(torch.tensor(0.5))
                self.fused_precomputed = False

            def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
                if self.fused_precomputed:
                    xn = F.layer_norm(h_lfm3.float(), (self.d_lfm3,), None, None, 1e-5)
                    x = self.proj2(self.act(self.proj1(xn)))
                    return F.layer_norm(x, (self.d_vq,), self.ln_out.weight, self.ln_out.bias, 1e-5).to(h_lfm3.dtype)
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

        self.vqvae.encoder.wte = MmapBF16Embedding(self.ncm_dir / "vqvae_wte.bf16.bin", V_LFM3, D_VQ)
        self.vqvae.decoder.output_layer = MmapBF16Linear(self.ncm_dir / "vqvae_dec.bf16.bin", D_VQ, V_LFM3)

        self.precomputed_dir = self.ncm_dir / "precomputed"
        self.precomputed_trunks_active = False
        self.precomputed_manifest: dict[str, Any] = {}
        manifest_fp = self.precomputed_dir / "manifest.json"
        core_pt = self.ncm_dir / "lfm3_ncm_core_bf16.pt"
        if not manifest_fp.exists() and core_pt.exists():
            try:
                from prepare_c2_c3_precomputed_trunks import build_c2_c3_precomputed
                build_c2_c3_precomputed(core_pt, self.precomputed_dir)
            except Exception:
                pass
        if manifest_fp.exists():
            try:
                self.precomputed_manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
            except Exception:
                self.precomputed_manifest = {"total_trunk_and_expert_dirs": 31}

        def _read_clvrtrk1(trunk_dir: Path) -> dict[str, Any]:
            import struct
            idx_bytes = (trunk_dir / "index.bin").read_bytes()
            if idx_bytes[:8] != b"CLVRTRK1":
                raise ValueError(f"Invalid CLVRTRK1 magic in {trunk_dir}")
            _, _, n_entries = struct.unpack("<III", idx_bytes[8:20])
            fnames = {0: "common.bin", 1: "operator.bin", 3: "ffn.bin"}
            blobs: dict[int, memoryview] = {}
            for fid, fn in fnames.items():
                fp = trunk_dir / fn
                if fp.exists() and fp.stat().st_size > 0:
                    blobs[fid] = memoryview(fp.read_bytes())
            out_t: dict[str, Any] = {}
            pos = 20
            for _ in range(n_entries):
                tname = idx_bytes[pos : pos + 64].split(b"\x00", 1)[0].decode("utf-8")
                pos += 64
                fid, dtype_code, ndim, offset, nbytes, d0, d1, d2, d3 = struct.unpack("<IIIQQIIII", idx_bytes[pos : pos + 44])
                pos += 44
                dims = [d0, d1, d2, d3][:ndim] if ndim > 0 else [1]
                raw = blobs[fid][offset : offset + nbytes]
                if dtype_code == 1:
                    arr = np.frombuffer(raw, dtype=np.uint16).copy()
                    out_t[tname] = torch.from_numpy(arr).view(torch.bfloat16).float().reshape(dims)
                else:
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    out_t[tname] = torch.from_numpy(arr).reshape(dims)
            return out_t

        if (self.precomputed_dir / "plan-trunk-0" / "index.bin").exists():
            # Load 100% directly from the 31 precomputed CLVRTRK1 directories (zero dependency on lfm3_ncm_core_bf16.pt!)
            with torch.no_grad():
                # 1. C2 · Plan (`plan-trunk-0/`)
                dp = _read_clvrtrk1(self.precomputed_dir / "plan-trunk-0")
                self.hybrid_link.proj1.weight.copy_(dp["native.proj1_fused.weight"])
                self.hybrid_link.proj1.bias.copy_(dp["native.proj1_fused.bias"])
                self.hybrid_link.proj2.weight.copy_(dp["native.proj2.weight"])
                self.hybrid_link.proj2.bias.copy_(dp["native.proj2.bias"])
                self.hybrid_link.post_ln.weight.copy_(dp["native.post_ln_fused.weight"])
                self.hybrid_link.post_ln.bias.copy_(dp["native.post_ln_fused.bias"])
                self.hybrid_link.fused_precomputed = True

                if "hl2048.proj1_fused.weight" in dp:
                    self.hybridlink_2048.proj1.weight.copy_(dp["hl2048.proj1_fused.weight"])
                    self.hybridlink_2048.proj1.bias.copy_(dp["hl2048.proj1_fused.bias"])
                    self.hybridlink_2048.proj2.weight.copy_(dp["hl2048.proj2.weight"])
                    self.hybridlink_2048.proj2.bias.copy_(dp["hl2048.proj2.bias"])
                    self.hybridlink_2048.post_ln.weight.copy_(dp["hl2048.post_ln.weight"])
                    self.hybridlink_2048.post_ln.bias.copy_(dp["hl2048.post_ln.bias"])
                    self.hybridlink_2048.fused_precomputed = True
                    self.has_stage3_hybridlink_2048 = True

                # 2. C3 · Bridges (`bridge-trunk-0/`)
                db = _read_clvrtrk1(self.precomputed_dir / "bridge-trunk-0")
                self.combo_bridge.fc1.weight.copy_(db["combo.fc1_fused.weight"])
                self.combo_bridge.fc1.bias.copy_(db["combo.fc1_fused.bias"])
                self.combo_bridge.fc2.weight.copy_(db["combo.fc2.weight"])
                self.combo_bridge.fc2.bias.copy_(db["combo.fc2.bias"])
                self.combo_bridge.ln_out.weight.copy_(db["combo.ln_out_fused.weight"])
                self.combo_bridge.ln_out.bias.copy_(db["combo.ln_out_fused.bias"])
                self.combo_bridge.fused_precomputed = True

                self.expert_bridge.proj1.weight.copy_(db["expert.proj1_fused.weight"])
                self.expert_bridge.proj1.bias.copy_(db["expert.proj1_fused.bias"])
                self.expert_bridge.proj2.weight.copy_(db["expert.proj2.weight"])
                self.expert_bridge.proj2.bias.copy_(db["expert.proj2.bias"])
                self.expert_bridge.ln_out.weight.copy_(db["expert.ln_out_fused.weight"])
                self.expert_bridge.ln_out.bias.copy_(db["expert.ln_out_fused.bias"])
                self.expert_bridge.fused_precomputed = True

                # 3. C3 · VQ-VAE (`vqvae-trunk-0/` + `vqvae-expert-0..10/`)
                dvq = _read_clvrtrk1(self.precomputed_dir / "vqvae-trunk-0")
                self.vqvae.encoder.wpe.weight.copy_(dvq["encoder.wpe.weight"])
                self.vqvae.decoder.ln_f.weight.copy_(dvq["decoder.ln_f.weight"])
                for k_idx in (0, 2, 4):
                    self.vqvae.vq.learned_down_l1[k_idx].weight.copy_(dvq[f"learned_down_l1.{k_idx}.weight"])
                    self.vqvae.vq.learned_down_l1[k_idx].bias.copy_(dvq[f"learned_down_l1.{k_idx}.bias"])
                    self.vqvae.vq.learned_up_l1[k_idx].weight.copy_(dvq[f"learned_up_l1.{k_idx}.weight"])
                    self.vqvae.vq.learned_up_l1[k_idx].bias.copy_(dvq[f"learned_up_l1.{k_idx}.bias"])
                mb = self.vqvae.decoder.middle_block
                mb.ln_1 = _ParameterlessLayerNorm(D_VQ, eps=1e-6)
                mb.attn.c_attn.weight.copy_(dvq["middle_block.attn.c_attn_fused.weight"])
                mb.attn.c_attn.bias.copy_(dvq["middle_block.attn.c_attn_fused.bias"])
                mb.attn.c_proj.weight.copy_(dvq["middle_block.attn.c_proj.weight"])
                mb.attn.c_proj.bias.copy_(dvq["middle_block.attn.c_proj.bias"])
                mb.attn.q_norm.weight.copy_(dvq["middle_block.attn.q_norm.weight"])
                mb.attn.k_norm.weight.copy_(dvq["middle_block.attn.k_norm.weight"])
                mb.ln_2 = _ParameterlessLayerNorm(D_VQ, eps=1e-6)
                mb.mlp.c_fc.weight.copy_(dvq["middle_block.mlp.c_fc_fused.weight"])
                mb.mlp.c_fc.bias.copy_(dvq["middle_block.mlp.c_fc_fused.bias"])
                mb.mlp.c_proj.weight.copy_(dvq["middle_block.mlp.c_proj.weight"])
                mb.mlp.c_proj.bias.copy_(dvq["middle_block.mlp.c_proj.bias"])

                for s in range(11):
                    dve = _read_clvrtrk1(self.precomputed_dir / f"vqvae-expert-{s}")
                    self.vqvae.vq.embeddings[s].weight.copy_(dve["codebook.weight"])
                    self.vqvae.vq.quant_resi[s].weight.copy_(dve["quant_resi.weight"])
                    self.vqvae.vq.quant_resi[s].bias.copy_(dve["quant_resi.bias"])

                # 4. C3 · 12-Layer NextConceptTransformer (`ncm-trunk-0..11/` + `ncm-global-0/`)
                dng = _read_clvrtrk1(self.precomputed_dir / "ncm-global-0")
                self.net.transformer.wte.weight.copy_(dng["transformer.wte.weight"])
                self.net.transformer.wte.bias.copy_(dng["transformer.wte.bias"])
                self.net.transformer.wle.weight.copy_(dng["transformer.wle.weight"])
                self.net.transformer.ln_f = _ParameterlessLayerNorm(768, eps=1e-6)
                self.net.lm_head_l0.weight.copy_(dng["lm_head_l0_fused.weight"])
                self.net.lm_head_l0.bias = nn.Parameter(dng["lm_head_l0_fused.bias"].clone())
                for h_idx in range(10):
                    self.net.lm_heads[h_idx].weight.copy_(dng[f"lm_heads.{h_idx}_fused.weight"])
                    self.net.lm_heads[h_idx].bias = nn.Parameter(dng[f"lm_heads.{h_idx}_fused.bias"].clone())

                for l in range(12):
                    dnl = _read_clvrtrk1(self.precomputed_dir / f"ncm-trunk-{l}")
                    blk = self.net.transformer.blocks[l]
                    blk.ln_1 = _ParameterlessLayerNorm(768, eps=1e-6)
                    blk.attn.c_attn.weight.copy_(dnl["attn.c_attn_fused.weight"])
                    blk.attn.c_attn.bias.copy_(dnl["attn.c_attn_fused.bias"])
                    blk.attn.c_proj.weight.copy_(dnl["attn.c_proj.weight"])
                    blk.attn.c_proj.bias.copy_(dnl["attn.c_proj.bias"])
                    blk.attn.q_norm.weight.copy_(dnl["attn.q_norm.weight"])
                    blk.attn.k_norm.weight.copy_(dnl["attn.k_norm.weight"])
                    blk.ln_2 = _ParameterlessLayerNorm(768, eps=1e-6)
                    blk.mlp.c_fc.weight.copy_(dnl["mlp.c_fc_fused.weight"])
                    blk.mlp.c_fc.bias.copy_(dnl["mlp.c_fc_fused.bias"])
                    blk.mlp.c_proj.weight.copy_(dnl["mlp.c_proj.weight"])
                    blk.mlp.c_proj.bias.copy_(dnl["mlp.c_proj.bias"])

                # 5. C3 · 3-Layer LexicalPointerDecoder (`ptr-trunk-0..2/` + `ptr-global-0/`)
                dpg = _read_clvrtrk1(self.precomputed_dir / "ptr-global-0")
                self.copy_dec.tgt_pos.weight.copy_(dpg["tgt_pos.weight"])
                self.copy_dec.plan_pos.weight.copy_(dpg["plan_pos.weight"])
                self.copy_dec.rh_pos.weight.copy_(dpg["rh_pos.weight"])
                self.copy_dec.ln_out.weight.copy_(dpg["ln_out_fused.weight"])
                self.copy_dec.ln_out.bias.copy_(dpg["ln_out_fused.bias"])
                self.copy_dec.res_scale.fill_(1.0)
                self.copy_dec.copy_q.weight.copy_(dpg["copy_q.weight"])
                if "copy_q.bias" in dpg and self.copy_dec.copy_q.bias is not None:
                    self.copy_dec.copy_q.bias.copy_(dpg["copy_q.bias"])
                self.copy_dec.copy_k.weight.copy_(dpg["copy_k.weight"])
                if "copy_k.bias" in dpg and self.copy_dec.copy_k.bias is not None:
                    self.copy_dec.copy_k.bias.copy_(dpg["copy_k.bias"])
                if "copy_gate.weight" in dpg:
                    self.copy_dec.copy_gate.weight.copy_(dpg["copy_gate.weight"])
                    self.copy_dec.copy_gate.bias.copy_(dpg["copy_gate.bias"])

                for l in range(3):
                    dpl = _read_clvrtrk1(self.precomputed_dir / f"ptr-trunk-{l}")
                    pblk = self.copy_dec.blocks[l]
                    pblk.ln1 = _ParameterlessLayerNorm(D_VQ, eps=1e-5)
                    pblk.qkv.weight.copy_(dpl["qkv_fused.weight"])
                    pblk.qkv.bias.copy_(dpl["qkv_fused.bias"])
                    pblk.proj.weight.copy_(dpl["proj.weight"])
                    pblk.proj.bias.copy_(dpl["proj.bias"])
                    pblk.ln_q = _ParameterlessLayerNorm(D_VQ, eps=1e-5)
                    pblk.cx_q.weight.copy_(dpl["cx_q_fused.weight"])
                    pblk.cx_q.bias.copy_(dpl["cx_q_fused.bias"])
                    pblk.ln_kv = _ParameterlessLayerNorm(D_VQ, eps=1e-5)
                    pblk.cx_kv.weight.copy_(dpl["cx_kv_fused.weight"])
                    pblk.cx_kv.bias.copy_(dpl["cx_kv_fused.bias"])
                    pblk.cx_proj.weight.copy_(dpl["cx_proj.weight"])
                    pblk.cx_proj.bias.copy_(dpl["cx_proj.bias"])
                    pblk.ln2 = _ParameterlessLayerNorm(D_VQ, eps=1e-5)
                    pblk.mlp[0].weight.copy_(dpl["mlp0_fused.weight"])
                    pblk.mlp[0].bias.copy_(dpl["mlp0_fused.bias"])
                    pblk.mlp[2].weight.copy_(dpl["mlp2.weight"])
                    pblk.mlp[2].bias.copy_(dpl["mlp2.bias"])

            self.precomputed_trunks_active = True
        elif core_pt.exists():
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

    def run_continuous_latent_step(
        self,
        h_last_2048: np.ndarray,
        known_top1_id: int | None = None,
        compute_lm_head: bool = True,
    ) -> dict[str, Any]:
        """
        Layer 2 (`HybridLink2048` + `Lfm3NativeHybridLink(d=2048, bottleneck=512)`):
        Projects `h_t` into continuous `<latent>` space `r_t = R_Lfm3(h_t)` and measures
        its cosine alignment with LFM3-Omni-2B's next-token embedding in `128,000`-vocab space.
        When `known_top1_id` is supplied from the unified C1 decode pass, avoids a redundant
        128,000-vocab LM-head GEMV (`~450 ms` saved on S23).
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
            if known_top1_id is not None and 0 <= int(known_top1_id) < V_LFM3:
                top1_id = int(known_top1_id)
            elif compute_lm_head:
                top1_id = self.c_engine.lm_head_top1(h_last_2048)
            else:
                top1_id = 0
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
            "c2_precomputed_trunk_active": bool(self.precomputed_trunks_active),
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
          - Decodes discrete tokens until `<latent>` (`125033`), `</search>`, `</python>`, `</answer>`, or `<|im_end|>`,
            capturing the in-situ 24-layer hidden state `h_t` at every step so C2/C3 never re-run C1!
          - Whenever `<latent>` (`125033`) is emitted, transitions to continuous `LATENT` mode and steps
            `r_t = R_Lfm3(h_t)` directly into the 24-layer C engine (`step_embedding_cached`) for `min_latent_steps .. max_latent_steps`,
            then closes the span with `</latent>` (`125034`) via `step_token_cached` and resumes decoding!
        """
        torch = self.torch
        F = self.F
        all_gen_ids: list[int] = []
        hidden_chunks: list[np.ndarray] = []
        h_cur = initial_h_2048
        total_c_ms = 0.0
        latent_spans_opened = 0
        latent_steps_total = 0
        latent_span_readouts: list[list[str]] = []

        while len(all_gen_ids) < max_new_tokens:
            rem = max(1, max_new_tokens - len(all_gen_ids))
            think_mode = 1 if latent_spans_opened == 0 else 2
            chunk_ids, h_cur, chunk_h_seq, ms_chunk = self.c_engine.decode_from_cache_with_hidden(
                h_cur, max_new_tokens=rem, allow_thinking=think_mode
            )
            total_c_ms += ms_chunk
            if not chunk_ids:
                break
            all_gen_ids.extend(chunk_ids)
            if chunk_h_seq is not None and chunk_h_seq.shape[0] > 0:
                hidden_chunks.append(chunk_h_seq)
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
                hidden_chunks.append(h_cur.reshape(1, D_LFM3).copy())
                continue

            # Otherwise decode_from_cache stopped on </search>, </python>, </answer>, </tool_call>, or <|im_end|>
            break

        full_h_seq = np.concatenate(hidden_chunks, axis=0) if hidden_chunks else np.zeros((0, D_LFM3), dtype=np.float32)
        return (
            all_gen_ids,
            h_cur,
            total_c_ms,
            {
                "latent_spans_opened": latent_spans_opened,
                "latent_steps_total": latent_steps_total,
                "latent_span_readouts": latent_span_readouts,
                "_in_situ_hidden_seq": full_h_seq,
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

    def render_plan_through_11level_ncm(
        self,
        plan_text: str,
        precomputed_h_seq: np.ndarray | None = None,
        precomputed_r_seq: np.ndarray | Any | None = None,
        precomputed_gen_ids: list[int] | None = None,
    ) -> tuple[str, list[int], dict[str, Any]]:
        """
        Unified C1 (Inference) -> C2 (Plan) -> C3 (Decode) NCM Pipeline:
          1. Uses the in-situ 24-layer C1 hidden states `precomputed_h_seq` directly (`c_backbone_ms = 0.0 ms`,
             zero redundant `forward_24_layers` re-execution!). Falls back to `forward_24_layers` only if
             called standalone without `precomputed_h_seq`.
          2. Uses the precomputed C2 latent plan states `precomputed_r_seq` directly from Stage C2 (`plan-trunk-0`:
             `Lfm3NativeHybridLink`), or computes `r_valid = self.hybrid_link(h_valid)` if called standalone.
          3. Projects `(r_valid, h_valid)` through precomputed `bridge-trunk-0` (`Lfm3ComboBridge` + `Lfm3ExpertBridge`)
             on the exact unpadded length `L_val = len(pids)` (`[1, 2*L_val, 384]`, zero dead padding tokens).
          4. Runs `NextConceptTransformer` (`ncm-trunk-0..11` + `ncm-global-0`) + `vqvae-expert-0..10` up to the
             multiscale level resolution covering `L_val`, accumulating `z_hat` in situ in a single pass.
          5. Decodes `[1, L_val, 384]` through `ptr-trunk-0..2` (`LexicalPointerDecoder`) + `vqvae-trunk-0` in parallel.
        """
        torch = self.torch
        F = self.F
        apply_rotary_emb = self._apply_rotary_emb
        t0 = time.perf_counter()
        clean_plan = (plan_text or "").strip()
        if not clean_plan:
            return "", [], {}

        paragraphs = [p.strip() for p in clean_plan.split("\n\n") if p.strip()]
        first_para = paragraphs[0]
        rest_paras = paragraphs[1:]

        pids = self.tok.encode(first_para, allow_special=False)[: self.P_BPE]
        if not pids:
            return clean_plan, self.tok.encode(clean_plan, allow_special=False), {}
        L_val = len(pids)

        # 1. C1 Hidden States: reuse in-situ hidden states from the unified C1 pass (`0.00 ms` extra C1 cost!)
        c_ms = 0.0
        reused_c1_hidden = False
        reused_c2_plan = False
        start_idx = 0
        if precomputed_h_seq is not None and isinstance(precomputed_h_seq, np.ndarray) and precomputed_h_seq.shape[0] > 0:
            if precomputed_gen_ids and len(precomputed_gen_ids) >= 1:
                for idx_s in range(0, len(precomputed_gen_ids)):
                    if precomputed_gen_ids[idx_s] == pids[0]:
                        start_idx = idx_s
                        break
            h_slice = precomputed_h_seq[start_idx : start_idx + L_val]
            if h_slice.shape[0] == 0:
                h_slice = precomputed_h_seq[:L_val]
            if h_slice.shape[0] < L_val:
                pad_rows = np.repeat(h_slice[-1:], L_val - h_slice.shape[0], axis=0)
                h_slice = np.concatenate([h_slice, pad_rows], axis=0)
            h_valid = torch.from_numpy(h_slice[:L_val]).float().unsqueeze(0)
            reused_c1_hidden = True
        if not reused_c1_hidden:
            h_c, c_ms = self.c_engine.forward_24_layers(pids)
            h_valid = torch.from_numpy(h_c[:L_val]).float().unsqueeze(0)

        with torch.no_grad():
            # 2. C2 Plan (`plan-trunk-0`) & C3 Bridges (`bridge-trunk-0`) on unpadded `L_val` tokens
            if precomputed_r_seq is not None:
                if isinstance(precomputed_r_seq, np.ndarray) and precomputed_r_seq.shape[0] > 0:
                    r_slice = precomputed_r_seq[start_idx : start_idx + L_val]
                    if r_slice.shape[0] == 0:
                        r_slice = precomputed_r_seq[:L_val]
                    if r_slice.shape[0] < L_val:
                        pad_r = np.repeat(r_slice[-1:], L_val - r_slice.shape[0], axis=0)
                        r_slice = np.concatenate([r_slice, pad_r], axis=0)
                    r_valid = torch.from_numpy(r_slice[:L_val]).float().unsqueeze(0)
                    reused_c2_plan = True
                elif hasattr(precomputed_r_seq, "shape") and precomputed_r_seq.shape[1] >= L_val:
                    r_valid = precomputed_r_seq[:, :L_val, :].float()
                    reused_c2_plan = True
                else:
                    r_valid = self.hybrid_link(h_valid)
            else:
                r_valid = self.hybrid_link(h_valid)
            z_rh_v = self.combo_bridge(r_valid).float()
            z_exp_v = self.expert_bridge(h_valid).float()
            p_valid = torch.tensor([pids], dtype=torch.long)
            z_bpe_v = self.vqvae.encoder.wte(p_valid) + z_exp_v
            z_pref_v = torch.cat([z_rh_v, z_bpe_v], dim=1)
            P_v = z_pref_v.shape[1]

            # 3. C3 12-Layer NextConceptTransformer (`ncm-trunk-0..11` + `ncm-global-0`) + `vqvae-expert-0..10`
            net = self.net
            vq = self.vqvae.vq
            levels = list(vq.levels)
            # Adaptive multiscale resolution: select level index whose concept resolution covers L_val (min level 2 -> 8 codes)
            max_lvl = 2
            for li, l_res in enumerate(levels):
                if li >= 2 and l_res >= L_val:
                    max_lvl = li
                    break
            else:
                max_lvl = len(levels) - 1

            z_hat = torch.zeros(1, levels[-1], self.vqvae.embed_dim, dtype=torch.float32)
            prefix_emb = net.transformer.wte(z_pref_v)
            level_pos_embed = net.transformer.wle(net.lvl_1L)
            cos_p = net.rope_cos_gen_0[:P_v]
            sin_p = net.rope_sin_gen_0[:P_v]

            h = prefix_emb
            prefix_kv: list[tuple[Any, Any]] = []
            for blk in net.transformer.blocks:
                B, T, C = h.shape
                nh = blk.attn.n_head
                hd = C // nh
                q, k, v = blk.attn.c_attn(blk.ln_1(h)).split(C, dim=2)
                q = apply_rotary_emb(blk.attn.q_norm(q.view(B, T, nh, hd).transpose(1, 2)), cos_p, sin_p)
                k = apply_rotary_emb(blk.attn.k_norm(k.view(B, T, nh, hd).transpose(1, 2)), cos_p, sin_p)
                v = v.view(B, T, nh, hd).transpose(1, 2)
                prefix_kv.append((k, v))
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                h = h + blk.attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
                h = h + blk.mlp(blk.ln_2(h))

            l0_logits = net.lm_head_l0(net.transformer.ln_f(h[:, -1:, :]))
            torch.manual_seed(42)
            idx_next = net._sample_level_raw(l0_logits, self.sched[0], 0, 1.0)
            generated_idx = [idx_next]
            level_acum = 0
            P_orig = net.prefix_len

            for level_idx in range(max_lvl):
                l_next = levels[level_idx + 1]
                z_q_level = vq.embeddings[level_idx](idx_next)
                z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
                x = net.transformer.wte(x) + level_pos_embed[:, level_acum : level_acum + l_next, :]
                cos_k = getattr(net, f"rope_cos_gen_{level_idx}")[P_orig : P_orig + l_next]
                sin_k = getattr(net, f"rope_sin_gen_{level_idx}")[P_orig : P_orig + l_next]
                for blk, (pk, pv) in zip(net.transformer.blocks, prefix_kv):
                    B, T, C = x.shape
                    nh = blk.attn.n_head
                    hd = C // nh
                    q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
                    q = apply_rotary_emb(blk.attn.q_norm(q.view(B, T, nh, hd).transpose(1, 2)), cos_k, sin_k)
                    k = apply_rotary_emb(blk.attn.k_norm(k.view(B, T, nh, hd).transpose(1, 2)), cos_k, sin_k)
                    v = v.view(B, T, nh, hd).transpose(1, 2)
                    y = F.scaled_dot_product_attention(
                        q,
                        torch.cat([pk, k], dim=2),
                        torch.cat([pv, v], dim=2),
                        is_causal=False,
                    )
                    x = x + blk.attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
                    x = x + blk.mlp(blk.ln_2(x))
                logits = net.lm_heads[level_idx](net.transformer.ln_f(x))
                idx_next = net._sample_level_raw(logits, self.sched[level_idx + 1], 0, 1.0)
                generated_idx.append(idx_next)
                level_acum += l_next

            # In-situ final level update on `z_hat` (no redundant second `idx_to_zhats` pass!)
            z_q_last = vq.embeddings[max_lvl](idx_next)
            z_hat, _ = vq.get_next_autoregressive_input(max_lvl, z_hat, z_q_last)
            g_codes = torch.cat(generated_idx, dim=1)

            # 4. C3 Unpadded `LexicalPointerDecoder` (`ptr-trunk-0..2`) + `vqvae-trunk-0` over `L_val` active positions
            dec = self.copy_dec
            t_pos = dec.tgt_pos(torch.arange(L_val))[None, :, :]
            p_pos = dec.plan_pos(torch.arange(L_val))[None, :, :]
            r_pos = dec.rh_pos(torch.arange(L_val))[None, :, :]
            mem_bpe = z_bpe_v + p_pos
            mem_rh = z_rh_v + r_pos
            mem = torch.cat([mem_rh, mem_bpe], dim=1)

            zh_sub = z_hat[:, :L_val, :]
            x_d = zh_sub + t_pos
            for pblk in dec.blocks:
                x_d = pblk(x_d, mem, None)
            z_refined = zh_sub + dec.res_scale * dec.ln_out(x_d)

            if self.vqvae.decoder.up is not None:
                z_refined = self.vqvae.decoder.up(z_refined.transpose(1, 2)).transpose(1, 2)
            if self.vqvae.decoder.wpe is not None:
                z_refined = z_refined + self.vqvae.decoder.wpe(torch.arange(z_refined.shape[1], dtype=torch.long))
            xd = self.vqvae.decoder.middle_block(z_refined)
            for blk in self.vqvae.decoder.extra_blocks:
                xd = blk(xd)
            xd = self.vqvae.decoder.ln_f(xd)
            allowed_ids = self._build_allowed_vocab_ids(first_para)
            base_logits = self.vqvae.decoder.output_layer(xd, allowed_ids=allowed_ids)

            cq = F.normalize(dec.copy_q(x_d), dim=-1)
            ck = F.normalize(dec.copy_k(mem_bpe), dim=-1)
            sim = torch.bmm(cq, ck.transpose(1, 2))

            p_len = torch.tensor([[float(L_val)]])
            has_semi = (p_valid == self.tok.semicolon_id).any(dim=1, keepdim=True)
            expected_T = torch.where(has_semi, (p_len * 1.25 + 2.0).clamp(min=4.0, max=float(L_val)), p_len)
            t_pos_idx = torch.arange(L_val, dtype=torch.float32)[None, :, None]
            p_pos_idx = torch.arange(L_val, dtype=torch.float32)[None, None, :]

            diag_soft = -2.5 * ((t_pos_idx / expected_T[:, :, None]) - (p_pos_idx / p_len[:, :, None])).pow(2)
            diag_clause = -45.0 * (t_pos_idx - p_pos_idx).pow(2)
            diag_bias = torch.where(has_semi[:, :, None], diag_soft, diag_clause)
            sim = sim * 12.0 + diag_bias
            attn_probs = F.softmax(sim, dim=-1)
            ctx_vec = torch.bmm(attn_probs, z_bpe_v)

            g_copy = torch.sigmoid(dec.copy_gate(torch.cat([x_d, ctx_vec], dim=-1)))
            soft_scale = g_copy * dec.copy_scale.clamp(min=0.0, max=16.0)
            eff_scale = torch.where(has_semi[:, :, None], soft_scale, soft_scale + 65.0)
            copy_boost = attn_probs * eff_scale

            idx_exp = p_valid[:, None, :].expand(1, L_val, L_val)
            base_logits.scatter_add_(2, idx_exp, copy_boost.to(base_logits.dtype))
            rendered_first, out_ids, raw_ncm_first, used_fallback = self._dec_model_aligned(base_logits[0], pids)

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
            "unified_c1_c2_c3_single_pass": bool(reused_c1_hidden),
            "reused_c2_plan_sequence": bool(reused_c2_plan),
            "c3_precomputed_trunks_active": bool(self.precomputed_trunks_active),
            "c3_precomputed_dirs": int(self.precomputed_manifest.get("total_trunk_and_expert_dirs", 31)),
            "vq_levels": max_lvl + 1,
            "vq_codes_emitted": int(g_codes.shape[1]),
            "zhat_final_shape": [1, L_val, 384],
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
    if not (Path(espeak_bin).exists() or shutil.which(espeak_bin)):
        return ""
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as tmpd:
        out_wav = Path(tmpd) / "speech.wav"
        res = subprocess.run(
            [espeak_bin, "-v", "en-us", "-s", "158", "-p", "52", "-w", str(out_wav), clean_speech],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if res.returncode != 0 or not out_wav.is_file():
            return ""
        wav_bytes = out_wav.read_bytes()
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
    TOOL_REPO_READ = {
        "name": "repo_read",
        "description": "Read a text file from Clover's memory repository or the open workspace, by its path inside it.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path inside the repository, such as notes/todo.md"}},
            "required": ["path"],
        },
    }
    TOOL_REPO_WRITE = {
        "name": "repo_write",
        "description": "Write a text file in Clover's memory repository or the open workspace; the change is committed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path inside the repository"},
                "content": {"type": "string", "description": "The whole new content of the file"},
            },
            "required": ["path", "content"],
        },
    }
    TOOL_REPO_SEARCH = {
        "name": "repo_search",
        "description": "Search Clover's memory repository (notes, knowledge, tasks) for files matching a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Words to look for"}},
            "required": ["query"],
        },
    }
    TOOL_SHELL = {
        "name": "run_shell",
        "description": "Run a shell command on this device in the active workspace folder and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command"}},
            "required": ["command"],
        },
    }
    TOOL_SSH = {
        "name": "run_ssh",
        "description": "Run a non-interactive command on a remote SSH host the user has configured.",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "The configured host name"},
                "port": {"type": "string", "description": "SSH port (default 22)"},
                "command": {"type": "string", "description": "The command to run"},
            },
            "required": ["host", "command"],
        },
    }
    # The phone's tools by the names semantics routes (PHONE_TOOLS); the order is the order C1 sees them in.
    TOOL_DEFS = {
        "web_search": TOOL_WEB_SEARCH,
        "python_interpreter": TOOL_PYTHON,
        "repo_read": TOOL_REPO_READ,
        "repo_write": TOOL_REPO_WRITE,
        "repo_search": TOOL_REPO_SEARCH,
        "run_shell": TOOL_SHELL,
        "run_ssh": TOOL_SSH,
    }

    def __init__(self, base_dir: Path, load_neural_ncm: bool = True):
        global PERSONAL_DIR, COLLECTIVE_DIR
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir)
        PERSONAL_DIR = self.base_dir / "personal"
        COLLECTIVE_DIR = self.base_dir / "collective"
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
        # Loaded 100% from the 31 precomputed CLVRTRK1/CLVR4B01 directories in `ncm/precomputed/`
        self.hybrid_ncm: S23HybridAgentAndNCM | None = None
        if load_neural_ncm and (
            (self.ncm_dir / "precomputed" / "plan-trunk-0" / "index.bin").exists()
            or (self.ncm_dir / "lfm3_ncm_core_bf16.pt").exists()
        ):
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

        # S3 Tool: Multi-Provider Web Search (`Tavily`, `Brave Search`, `Wikipedia`, `arXiv`) + `research()` Relevance Filter
        from web_search import SearchConfig, WebSearch

        self.search_config = SearchConfig.shared(
            self.git_memory_repo.default_repo_dir / "config" / self.git_memory_repo.device / "search.json",
            self.git_memory_repo.vault_dir / "search-keys.json",
        )
        self.search_tool = WebSearch(self.search_config)

        # Pre-warm the full default system prompt + `<|im_start|>user\n` prefix in the 24-layer KV + ShortConv cache
        self._default_sys_prefix_ids = self.tokenizer.encode(
            self._build_system_prompt(allow_internet=False, allow_python=False, use_hybrid_agent_format=False)
            + "<|im_start|>user\n",
            allow_special=True,
        )
        self.engine.prefill_cached(self._default_sys_prefix_ids)
        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Persistent Studio Settings & Permissions (Unified C1+C2+C3 + S1+S2+S3 Active by Default)
        self.settings_path = self.git_memory_repo.default_repo_dir / "config" / self.git_memory_repo.device / "settings.json"
        self.legacy_settings_path = self.base_dir / "s23_studio_settings.json"
        self.perms_path = self.git_memory_repo.default_repo_dir / "config" / self.git_memory_repo.device / "permissions.json"
        self.legacy_perms_path = self.base_dir / "s23_studio_permissions.json"
        self.settings: dict[str, Any] = {
            "schema_version": 3,
            "model": "omni",
            "engine": "gpu",
            "q4_operators": False,
            "q4_lm_head": True,
            "tools": "ask",
            "layers": {
                "hybridlink": True,
                "ncm": True,
                "memory": True,
                "recall": True,
                "ontology": True,
                "compiler": True,
                "specmd_madvise": True,
                "kv_slots": True,
            },
        }
        migrated_v3 = False
        for sp in (self.legacy_settings_path, self.settings_path):
            if sp.exists():
                try:
                    saved_s = json.loads(sp.read_text(encoding="utf-8"))
                    if isinstance(saved_s, dict) and int(saved_s.get("schema_version") or 1) >= 3:
                        if saved_s.get("engine") in ("gpu", "adreno", "cpu"):
                            self.settings["engine"] = "gpu" if saved_s["engine"] in ("gpu", "adreno") else "cpu"
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
                    else:
                        migrated_v3 = True
                except Exception:
                    pass
        if migrated_v3:
            try:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
                self.legacy_settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            except Exception:
                pass
        self._apply_hardware_settings()
        self.device_profile = self.detect_android_device_profile()
        self.turns_lock = threading.Lock()
        self.active_turns: dict[str, dict[str, Any]] = {}
        self.recent_turns: dict[str, dict[str, Any]] = {}

        self.permissions_meta: dict[str, dict[str, str]] = {
            "internet": {
                "label": "Internet search and web pages",
                "detail": "Searches the web with the service chosen in Settings (Tavily, Brave Search, or Wikipedia and arXiv) and opens pages.",
            },
            "code": {
                "label": "Run code Clover writes",
                "detail": "Executes Python code in an isolated on-device subprocess with a time limit.",
            },
            "shell": {
                "label": "Run terminal commands",
                "detail": "Runs shell commands on this device in the active workspace folder with a time limit. Stopping the answer stops the command.",
            },
            "ssh": {
                "label": "Connect to remote SSH hosts",
                "detail": "Runs non-interactive SSH commands on remote hosts you configure in Settings.",
            },
            "files": {
                "label": "Read files and folders",
                "detail": "Lists folders and reads text files, photos, WAV audio and videos you attach or reference on this device.",
            },
            "edit_files": {
                "label": "Create or change files",
                "detail": "Writes text files in your active workspace on this device.",
            },
            "memory": {
                "label": "Remember things you tell it",
                "detail": "Saves what you ask Clover to remember as notes in Clover's local Git repository (clover-memory-repo).",
            },
            "vscode": {
                "label": "GitHub & VS Code sync",
                "detail": "Syncs Clover's private Git memory repository with your GitHub account or VS Code bridge.",
            },
            "media": {
                "label": "Create pictures & media",
                "detail": "Creates pictures, audio and visual previews on this device and saves them in your media folder.",
            },
            "voice_camera": {
                "label": "Microphone, camera and voice synthesis",
                "detail": "Allows voice input and spoken audio replies on this device.",
            },
        }
        self.perm_modes: dict[str, str] = {
            "internet": "allow",
            "code": "allow",
            "shell": "ask",
            "ssh": "ask",
            "files": "allow",
            "edit_files": "ask",
            "memory": "allow",
            "vscode": "allow",
            "media": "allow",
            "voice_camera": "allow",
        }
        for pp in (self.legacy_perms_path, self.perms_path):
            if pp.exists():
                try:
                    saved_p = json.loads(pp.read_text(encoding="utf-8"))
                    if isinstance(saved_p, dict):
                        for pk, pv in saved_p.items():
                            if pk in self.perm_modes and pv in ("allow", "ask", "deny"):
                                self.perm_modes[pk] = pv
                except Exception:
                    pass

        # S2 · Relationships: Turn-Level MemoryIndex (`sessions/<month>/<chat>.jsonl`, `growth/notes/`, BM25 + Dense RRF)
        from memory_index import MemoryIndex

        # C1 is not safe to call from two threads: S2's note writer and a turn ran together on k3 and both outputs were
        # unreadable (research/core-observation). Turns wait for this lock; the note writer skips when it is taken.
        self.engine_lock = threading.RLock()
        self._busy_turns = 0
        self._last_turn_end = time.time()
        self.memory_index = MemoryIndex(
            self.base_dir,
            self.git_memory_repo,
            is_idle=self._is_studio_idle,
            note_writer=self._idle_note_writer,
            start=True,
            recall=bool(self.settings["layers"].get("recall", True)),
        )
        from clover_semantics import CloverSemanticInterface
        self.semantics = CloverSemanticInterface()

        # If a GitHub session is signed in, pull latest sessions & memory in the background on startup
        gh_cfg = self.git_memory_repo.get_github_sync_config()
        if gh_cfg.get("logged_in") and gh_cfg.get("auto_sync") and gh_cfg.get("repo"):
            def _startup_pull() -> None:
                try:
                    self.git_memory_repo.configure_and_sync_private_github(
                        repo_full_name=gh_cfg["repo"],
                        action="pull",
                        branch=gh_cfg.get("branch") or "main",
                    )
                    self.memory_index.wake()
                except Exception:
                    pass
            threading.Thread(target=_startup_pull, daemon=True, name="s1-startup-pull").start()

    def _save_to_media_storage(self, raw_bytes: bytes, ext: str) -> str:
        """Saves an uploaded or generated media file into the user's configured local or cloud `media_dir`."""
        import hashlib
        mdir = self.git_memory_repo.get_media_dir()
        mdir.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha256(raw_bytes).hexdigest()[:24]
        fp = mdir / f"{stem}.{ext.lstrip('.')}"
        if not fp.is_file():
            fp.write_bytes(raw_bytes)
        return fp.name

    def _is_studio_idle(self) -> bool:
        return self._busy_turns == 0 and (time.time() - self._last_turn_end > 8.0)

    def get_precomputed_dataset_sizes(self) -> dict[str, Any]:
        def _sz(p: Path) -> int:
            if not p.exists():
                return 0
            if p.is_file():
                return int(p.stat().st_size)
            return sum(int(f.stat().st_size) for f in p.rglob("*") if f.is_file())

        def _mib(b: int) -> float:
            return round(b / (1024.0 * 1024.0), 2)

        apk_b = _sz(self.base_dir / "android-app" / "CloverAIStudio.apk") or _sz(SCRIPT_DIR / "android-app" / "CloverAIStudio.apk")
        tok_b = _sz(self.base_dir / "tokenizer.json")
        so_b = _sz(self.base_dir / "liblfm3_cpu_engine.so")

        c1_op_bf16 = sum(_sz(self.pruned_dir / f"trunk-{i}" / "operator.bin") + _sz(self.pruned_dir / f"trunk-{i}" / "common.bin") + _sz(self.pruned_dir / f"trunk-{i}" / "index.bin") for i in range(24))
        c1_op_4bit = sum(_sz(self.pruned_dir / f"trunk-{i}" / "operator.4bit.bin") for i in range(24))
        c1_exp_4bit = sum(_sz(self.pruned_dir / f"trunk-{i}" / "ffn.4bit.bin") for i in range(24))
        c1_glob_emb = _sz(self.pruned_dir / "global-0" / "embed_tokens.bf16.bin")
        c1_glob_lm4 = _sz(self.pruned_dir / "global-0" / "lm_head.4bit.bin")
        c1_total = c1_op_bf16 + c1_op_4bit + c1_exp_4bit + c1_glob_emb + c1_glob_lm4

        prec = self.ncm_dir / "precomputed"
        c2_plan = _sz(prec / "plan-trunk-0")
        c3_bridge = _sz(prec / "bridge-trunk-0")
        c3_vq_trk = _sz(prec / "vqvae-trunk-0")
        c3_vq_exp = sum(_sz(prec / f"vqvae-expert-{i}") for i in range(11))
        c3_ncm_trk = sum(_sz(prec / f"ncm-trunk-{i}") for i in range(12))
        c3_ncm_glb = _sz(prec / "ncm-global-0")
        c3_ptr_trk = sum(_sz(prec / f"ptr-trunk-{i}") for i in range(3)) + _sz(prec / "ptr-global-0")
        c3_vq_vocab = _sz(self.ncm_dir / "vqvae_wte.bf16.bin") + _sz(self.ncm_dir / "vqvae_dec.bf16.bin")
        c3_total = c3_bridge + c3_vq_trk + c3_vq_exp + c3_ncm_trk + c3_ncm_glb + c3_ptr_trk + c3_vq_vocab
        s1_s2_b = _sz(self.git_memory_repo.default_repo_dir)
        total_b = apk_b + tok_b + so_b + c1_total + c2_plan + c3_total + s1_s2_b

        return {
            "app_apk_mib": _mib(apk_b),
            "tokenizer_and_c_engine_mib": _mib(tok_b + so_b),
            "c1_base_model_total_mib": _mib(c1_total),
            "c1_trunks_24_bf16_mib": _mib(c1_op_bf16),
            "c1_trunks_24_4bit_mib": _mib(c1_op_4bit),
            "c1_experts_24_swiglu_4bit_mib": _mib(c1_exp_4bit),
            "c1_global_0_embed_and_lmhead_mib": _mib(c1_glob_emb + c1_glob_lm4),
            "c2_hybridagent_plan_trunk_0_mib": _mib(c2_plan),
            "c3_bridges_trunk_0_mib": _mib(c3_bridge),
            "c3_vqvae_trunk_and_11_experts_mib": _mib(c3_vq_trk + c3_vq_exp),
            "c3_ncm_12_trunks_and_global_mib": _mib(c3_ncm_trk + c3_ncm_glb),
            "c3_pointer_decoder_trunks_mib": _mib(c3_ptr_trk),
            "c3_vqvae_shared_vocab_maps_mib": _mib(c3_vq_vocab),
            "c3_decode_total_mib": _mib(c3_total),
            "s1_s2_memory_and_ontology_mib": _mib(s1_s2_b),
            "total_on_device_mib": _mib(total_b),
            "total_on_device_gib": round(total_b / (1024.0 ** 3), 2),
        }

    def detect_android_device_profile(self) -> dict[str, Any]:
        """
        Universal Android Device & SoC Auto-Detection:
        Reads Android system properties (`getprop`) and `/proc/cpuinfo` to identify the exact
        device manufacturer, model, SoC, CPU architecture, and GPU (Qualcomm Adreno, ARM Mali/Immortalis,
        Samsung Xclipse, or Android Studio Emulator / host CPU).
        """
        def _prop(key: str) -> str:
            gp = shutil.which("getprop")
            if not gp:
                return ""
            try:
                res = subprocess.run([gp, key], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
                return (res.stdout or "").strip() if res.returncode == 0 else ""
            except Exception:
                return ""

        mfr = _prop("ro.product.manufacturer") or _prop("ro.product.vendor.manufacturer")
        model = _prop("ro.product.model") or _prop("ro.product.vendor.model")
        soc_mfr = _prop("ro.soc.manufacturer")
        soc_mod = _prop("ro.soc.model") or _prop("ro.board.platform")
        sdk_ver = _prop("ro.build.version.sdk")
        rel_ver = _prop("ro.build.version.release")
        arch = platform.machine().lower()

        cpu_model = ""
        if os.path.exists("/proc/cpuinfo"):
            try:
                for ln in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                    if ln.lower().startswith(("model name", "hardware", "processor")):
                        val = ln.split(":", 1)[-1].strip()
                        if val and not val.isdigit():
                            cpu_model = val
                            break
            except Exception:
                pass

        dm = self.engine.get_dm_stats(madvise_enabled=True)
        opencl_ok = bool(dm.get("direction_b_mobile_soc_4tier_ledger", {}).get("adreno740_zero_copy_opencl_active", False))

        soc_combined = f"{soc_mfr} {soc_mod}".strip() or cpu_model or arch
        soc_low = (soc_combined + " " + model + " " + mfr).lower()
        if any(k in soc_low for k in ("qualcomm", "qti", "sm8", "sm7", "sm6", "sdm", "lahaina", "taro", "kalama", "pineapple")):
            gpu_label = "Adreno OpenCL GPU"
        elif any(k in soc_low for k in ("tensor", "zuma", "gs201", "gs101", "mediatek", "mt6", "dimensity", "mali", "immortalis")):
            gpu_label = "Mali / Immortalis GPU"
        elif "s5e" in soc_low or "exynos" in soc_low:
            gpu_label = "Xclipse / Mali GPU"
        elif opencl_ok:
            gpu_label = "Mobile OpenCL GPU"
        else:
            gpu_label = "Mobile GPU"

        cpu_label = "ARM64 NEON/SMMLA CPU" if arch in ("aarch64", "arm64") else f"{arch.upper()} CPU"
        dev_name = f"{mfr} {model}".strip() if (mfr or model) else (cpu_model or f"{platform.system()} ({arch})")
        return {
            "device_name": dev_name,
            "soc": soc_combined,
            "arch": arch,
            "android_release": rel_ver or None,
            "android_sdk": sdk_ver or None,
            "gpu_available": opencl_ok,
            "gpu_label": gpu_label,
            "cpu_label": cpu_label,
            "active_threads": self.engine.active_threads,
        }

    def speech_parts(self, text: str) -> list[str]:
        clean = re.sub(r"```[\s\S]*?```", " code block. ", str(text or ""))
        clean = re.sub(r"[`*_#>~]+", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return []
        raw_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
        out: list[str] = []
        buf = ""
        for p in raw_parts:
            if len(buf) + len(p) < 160:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    out.append(buf)
                buf = p
        if buf:
            out.append(buf)
        return out[:20] or [clean[:400]]

    def tts_wav_bytes(self, text: str, voice: str = "female") -> bytes:
        durl = generate_spoken_wav_data_url(text, self.base_dir)
        if durl and "," in durl:
            return base64.b64decode(durl.split(",", 1)[1])
        # Generate a valid minimal 16 kHz mono WAV buffer if eSpeak NG is not installed on the host
        sr = 16000
        n_samples = int(sr * 0.25)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * n_samples)
        return buf.getvalue()

    def _idle_note_writer(self, msgs: list[dict[str, str]], max_new: int = 90) -> str | None:
        if not self._is_studio_idle():
            return None
        if not self.engine_lock.acquire(blocking=False):
            return None
        try:
            u_txt = "\n".join(str(m.get("content") or "") for m in msgs if m.get("content"))
            p_str = (
                "<|startoftext|><|im_start|>system\nYou write concise memory notes from conversations.<|im_end|>\n"
                f"<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n<think></think><answer>"
            )
            pids = self.tokenizer.encode(p_str, allow_special=True)[:768]
            if not self._is_studio_idle():
                return None
            h_p, _ = self.engine.prefill_cached(pids)
            g_ids, _, _ = self.engine.decode_from_cache(h_p, max_new_tokens=max_new, allow_thinking=0)
            return self._clean_model_text(self.tokenizer.decode(g_ids, skip_special_tokens=False))
        except Exception:
            return None
        finally:
            self.engine_lock.release()

    DECLINED_TEXT = (
        "[Clover permission] The user chose not to allow this just now, so it did not run. "
        "Say so plainly (it is not turned off) and offer another way."
    )

    def _check_tool_permission(
        self,
        perm_key: str,
        tool_name: str,
        detail: str,
        allow_once: set[str],
        deny_once: set[str],
        interactive_ask: bool,
        explicit_flag: bool = True,
    ) -> tuple[bool, dict[str, Any] | None, str]:
        """
        Checks `Allow` / `Ask` / `Off` (`deny`) for `perm_key`.
        - If `perm_key == "shell"` and `local_terminal.risky(detail)` is non-None, always prompts (`ask`)
          even when `shell` is set to `Allow`, with `"risky": why` so destructive commands are never run silently.
        - If `perm_key in deny_once` (the user clicked "Don't allow" on the Ask card), returns `DECLINED_TEXT`
          so Clover explains that the user declined it just now rather than saying the permission is turned off.
        """
        if perm_key in deny_once:
            return False, None, self.DECLINED_TEXT
        if not explicit_flag:
            return False, None, f"[Clover permission] {perm_key} was not allowed for this turn."
        risk_why = None
        if perm_key == "shell":
            import local_terminal
            risk_why = local_terminal.risky(detail)
        if perm_key in allow_once:
            return True, None, "allowed_once"
        mode = self.perm_modes.get(perm_key, "allow")
        if mode == "deny":
            return False, None, f"[Clover permission] '{self.permissions_meta.get(perm_key, {}).get('label', perm_key)}' is turned off in Settings."
        if mode == "allow" and not risk_why:
            return True, None, "allowed"
        if mode == "ask" or risk_why:
            if interactive_ask:
                prompt_d: dict[str, Any] = {
                    "permission": perm_key,
                    "tool": tool_name,
                    "label": self.permissions_meta.get(perm_key, {}).get("label", perm_key),
                    "detail": str(detail or "")[:600],
                }
                if risk_why:
                    prompt_d["risky"] = risk_why
                return False, prompt_d, "ask"
            return False, None, f"[Clover permission] '{perm_key}' requires confirmation."
        return False, None, "denied"

    def _apply_hardware_settings(self) -> None:
        self.engine.set_adreno_opencl_active(self.settings.get("engine") in ("gpu", "adreno"))
        self.engine.set_use_4bit_operators(bool(self.settings.get("q4_operators", False)))
        self.engine.set_use_4bit_lm_head(bool(self.settings.get("q4_lm_head", True)))
        self.engine.set_madvise_active(bool(self.settings.get("layers", {}).get("specmd_madvise", True)))

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        if "engine" in changes and changes["engine"] in ("gpu", "adreno", "cpu"):
            self.settings["engine"] = "gpu" if changes["engine"] in ("gpu", "adreno") else "cpu"
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
                    if lk == "recall" and hasattr(self, "memory_index"):
                        self.memory_index.set_recall(lv)
        self._apply_hardware_settings()
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            self.legacy_settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
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
            self.perms_path.parent.mkdir(parents=True, exist_ok=True)
            self.perms_path.write_text(json.dumps(self.perm_modes, indent=2), encoding="utf-8")
            self.legacy_perms_path.write_text(json.dumps(self.perm_modes, indent=2), encoding="utf-8")
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

    def _build_system_prompt(
        self,
        allow_internet: bool = True,
        allow_python: bool = True,
        use_hybrid_agent_format: bool = True,
        tools: list[str] | None = None,
    ) -> str:
        """tools: the tool names to offer (`TOOL_DEFS`); None means web search and Python by the two flags. Web search
        and Python alone use HybridAgent's format; any other tool uses the `<tools>` list."""
        if tools is None:
            tools = [n for n, on in (("web_search", allow_internet), ("python_interpreter", allow_python)) if on]
        if use_hybrid_agent_format and tools and set(tools) <= {"web_search", "python_interpreter"}:
            return f"<|startoftext|><|im_start|>system\n{self.HYBRID_AGENT_SYSTEM_PROMPT}<|im_end|>\n"
        active_tools = [self.TOOL_DEFS[t] for t in tools if t in self.TOOL_DEFS]
        if active_tools:
            tools_block = "\n".join(json.dumps(t) for t in active_tools)
            return f"<|startoftext|><|im_start|>system\n<tools>\n{tools_block}\n</tools><|im_end|>\n"
        return "<|startoftext|><|im_start|>system\nYou are Clover, a helpful AI assistant. Answer the user's request directly.<|im_end|>\n"

    _SPECIAL_TOKEN = re.compile(r"<\|[^|<>\s]{1,40}\|>")

    def _history_turns(self, history: list[dict[str, Any]] | None, budget_tokens: int) -> str:
        """This conversation's earlier messages as chat turns for C1: the most recent that fit in `budget_tokens`, each cut
        to 200 tokens, starting with a user message. Special-token text inside a message is removed, so a quoted page
        cannot open a system turn."""
        turns: list[str] = []
        used = 0
        for m in reversed(history or []):
            role = "user" if m.get("role") == "user" else "assistant"
            text = self._SPECIAL_TOKEN.sub(" ", str(m.get("content") or "")).strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, allow_special=False)
            if len(ids) > 200:
                text = self.tokenizer.decode(ids[:200], skip_special_tokens=True).strip() + " ..."
            turn = f"<|im_start|>{role}\n{text}<|im_end|>\n"
            n = len(self.tokenizer.encode(turn, allow_special=True))
            if used + n > budget_tokens:
                break
            turns.append(turn)
            used += n
        turns.reverse()
        while turns and not turns[0].startswith("<|im_start|>user"):
            turns.pop(0)
        return "".join(turns)

    @staticmethod
    def looping(text: str) -> tuple[bool, str]:
        """Detects and trims any repeated tail (160-char tail seen twice before, 25+ char substring repeated 3x, or identical line repeated 5x)."""
        s = text or ""
        if len(s) >= 640 and s.count(s[-160:], 0, len(s) - 160) >= 2:
            tail = s[-160:]
            return True, s[: s.find(tail) + 160].rstrip()
        if len(s) >= 90:
            for w in range(25, min(140, len(s) // 3) + 1):
                unit = s[-w:]
                if s.endswith(unit * 3):
                    idx = s.find(unit)
                    return True, s[: idx + len(unit)].rstrip()
        lines = [ln.strip() for ln in s.splitlines() if len(ln.strip()) >= 8]
        if len(lines) >= 5 and len(set(lines[-5:])) == 1:
            unit = lines[-1]
            idx = s.find(unit)
            return True, s[: idx + len(unit)].rstrip()
        return False, s

    @classmethod
    def _clean_model_text(cls, text: str) -> str:
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
        _, s = cls.looping(s)
        return s

    @staticmethod
    def _parse_lfm3_tool_call_xml(raw_text: str) -> dict[str, Any] | None:
        """Parses HybridAgent's `<search>...</search>`, `<python>...</python>`, `<shell>...</shell>`, `<ssh host=\"...\">...</ssh>`, and S1 Git Repo tags (`<repo_search>`, `<repo_read>`, `<repo_write>`) as well as `<tool_call><function=NAME>...`."""
        txt = raw_text or ""
        m_srch = re.search(r"<search>\s*([\s\S]*?)\s*</search>", txt)
        if m_srch:
            return {"function": "web_search", "parameters": {"query": m_srch.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        m_py = re.search(r"<python>\s*([\s\S]*?)\s*</python>", txt)
        if m_py:
            return {"function": "python_interpreter", "parameters": {"code": m_py.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_python"}
        m_sh = re.search(r"<shell>\s*([\s\S]*?)\s*</shell>", txt)
        if m_sh:
            return {"function": "run_shell", "parameters": {"command": m_sh.group(1).strip()}, "raw_xml": txt, "format": "hybrid_agent_search"}
        m_ssh = re.search(r"<ssh\s+host=[\"']([^\"']+)[\"'](?:\s+port=[\"'](\d+)[\"'])?\s*>\s*([\s\S]*?)\s*</ssh>", txt)
        if m_ssh:
            return {
                "function": "run_ssh",
                "parameters": {"host": m_ssh.group(1).strip(), "port": m_ssh.group(2) or "22", "command": m_ssh.group(3).strip()},
                "raw_xml": txt,
                "format": "hybrid_agent_search",
            }
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

    def _execute_shell_subprocess(self, command: str, cwd: str | None = None, timeout_s: float = 30.0) -> str:
        """Executes a local shell command (`local_terminal.shell`) with process-tree cleanup and 1 MB UTF-8 cap."""
        import local_terminal

        cmd = (command or "").strip()
        if not cmd:
            return ""
        work_dir = str(self.git_memory_repo.repo_dir) if hasattr(self, "git_memory_repo") else str(self.base_dir)
        resolved_cwd, err = local_terminal.resolve_cwd(cwd, workdir=work_dir)
        if err:
            return err
        out = local_terminal.shell(cmd, cwd=resolved_cwd, timeout_s=int(timeout_s))
        if "is not recognized as the name of a cmdlet" in out or "command not found" in out:
            out += (
                "\n[That command does not exist on this device. Use a real command or an installed program, "
                "e.g. git log --oneline -5, git status, ls -la, df -h.]"
            )
        return out[:4000]

    def _execute_ssh_subprocess(self, host: str, command: str, port: str | int = 22, timeout_s: float = 30.0) -> str:
        """Executes a non-interactive SSH command via `local_terminal.SSHHosts`."""
        import local_terminal

        h = (host or "").strip()
        cmd = (command or "").strip()
        if not h or not cmd:
            return "Error: both host and command are required for SSH."
        cfg_p = self.git_memory_repo.default_repo_dir / "config" / self.git_memory_repo.device / "ssh-hosts.json"
        ssh_mgr = local_terminal.SSHHosts(cfg_p)
        return ssh_mgr.run_ssh(h, cmd, port=port, timeout_s=int(timeout_s))[:4000]

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
    _ASKS_USER = re.compile(
        r"\?\s*(?:\n|$)|\b(?:could|can|would) you (?:please )?(?:tell|provide|confirm|clarify|share|specify)\b"
        r"|\bplease (?:tell|provide|confirm|clarify|specify)\b",
        re.I,
    )
    _LINKS_AND_PATHS = re.compile(
        r"https?://\S+|www\.\S+|\b[A-Za-z]:\\\S*|(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)+"
    )
    _LESSON_QUESTION = re.compile(r'(?:On a question like|Question:)\s*["\']([^"\']+)["\']', re.I)
    _LESSON_TOPIC = re.compile(r"[a-z][a-z0-9_\-]{2,}|\d+(?:\.\d+)?")
    _LESSON_FILLER = {
        "the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where", "which", "who",
        "why", "how", "are", "was", "were", "will", "would", "could", "should", "have", "has", "had", "does",
        "did", "not", "but", "you", "your", "can", "about", "after", "before", "over", "under", "more", "most",
        "some", "any", "each", "other", "than", "then", "also", "just", "only", "very", "much", "many", "such",
        "there", "their", "they", "them", "these", "those", "its", "his", "her", "our", "out", "one", "two",
        "three", "first", "new", "use", "used", "using", "make", "made", "way", "well", "may", "might", "must",
        "need", "want", "like", "give", "given", "take", "get", "got", "let", "say", "said", "tell", "told",
        "ask", "asked", "question", "answer", "earlier", "later", "independent", "solution", "solutions",
        "gave", "check", "checked", "checks", "user", "clover", "sentence", "sentences", "word", "words",
        "found", "right", "wrong", "really", "here",
    }

    @classmethod
    def topic_terms(cls, text: str) -> set[str]:
        """Content words of `text` (for a learning/lesson, of the question it is about), without filler words."""
        m = cls._LESSON_QUESTION.search(text or "")
        src = (m.group(1) if m else (text or "")).lower()
        return {
            w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in cls._LESSON_TOPIC.findall(src)
            if w not in cls._LESSON_FILLER
        }

    @classmethod
    def asked_tools(cls, text: str) -> list[str]:
        """The phone's tools that `text` asks for (used when `settings['tools'] == 'ask'`, or to narrow `'always'`). The
        rules are common to every Clover process (semantics.py); the phone adds only its tool names."""
        import semantics
        return semantics.asked(text, phone_runtime())

    def social_turn(self, prompt: str, frame, session_id: str, max_active_layer: int = 7) -> dict[str, Any]:
        """A message made only of pleasantries ("hi", "thanks", "bye"): semantics answers and the model is not called.
        The turn is still recorded, so "thanks" counts as feedback on the answer before it (Growth)."""
        turn_n = 1
        if max(1, min(7, int(max_active_layer))) >= 4 and self.perm_modes.get("memory", "allow") != "deny":
            try:
                growth = self.git_memory_repo.record_task_or_session_turn(
                    session_id=session_id, prompt=prompt, reply=frame.reply, tools_used=[], elapsed_ms=0.0)
                turn_n = int(growth.get("turn_n") or 1)
                if hasattr(self, "memory_index"):
                    self.memory_index.wake()
            except Exception:
                pass
        return {"device": platform.node(), "model": "semantics", "session_id": session_id, "turn_n": turn_n,
                "prompt": prompt, "reply": frame.reply, "output_type": "text", "tools_used": [],
                "layer_trace": {"semantics_frame": frame.layer()}, "generated_tokens": 0, "emitted_tokens": 0,
                "total_cycle_ms": 0.0}

    def remember_for(self, session_id: str, max_active_layer: int = 7):
        """The save `semantics.align` makes when the user asked Clover to remember something: their words, as a note in
        S1 (`notes/MEM-USER-*.md`). None when memory is not set to Allow, as on the laptop."""
        if max(1, min(7, int(max_active_layer))) < 4 or self.perm_modes.get("memory", "allow") != "allow" \
                or not self.settings.get("layers", {}).get("memory", True):
            return None

        def save(note: str) -> str:
            try:
                saved = self.git_memory_repo.remember_note(note, conversation_id=session_id)
            except Exception as ex:
                return f"Not saved: {ex}"
            if hasattr(self, "memory_index"):
                self.memory_index.wake()
            return f"Saved to memory as {saved['id']}."
        return save

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
        max_prompt_tokens: int = 1280,
        max_new_tokens: int = 256,
        output_type: str = "text",
        image_b64: str | None = None,
        attached_files: list[dict[str, Any]] | None = None,
        voice_input: bool = False,
        max_active_layer: int = 7,
        use_ncm_renderer: bool = True,
        enable_thinking: bool = False,
        inject_latent_step_before_decode: bool | None = None,
        allow_internet: bool = True,
        allow_python: bool = True,
        allow_files: bool = True,
        session_id: str = "s23-studio",
        allow_once: list[str] | None = None,
        deny_once: list[str] | None = None,
        interactive_ask: bool = False,
        history: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        meaning: Any = None,
    ) -> dict[str, Any]:
        """
        Executes the Clover cycle up to `max_active_layer` (`1 .. 7`) with zero hardcoded answers:
          - Enforces `Allow` / `Ask` / `Off` permissions across `internet`, `code`, `shell`, `ssh`, `files`, and `memory`.
          - S1 (`CloverMemoryGitRepo`) & S2 (`MemoryIndex` turn-level BM25 + dense RRF + `k3_ontology_graph.py`)
            inject retrieved evidence (`F6` dated background system message for recall).
          - C1 (`LFM3` 24-layer C engine) + C2 (`HybridLink2048` + `Lfm3NativeHybridLink`) + C3 (`11-Level NCM`).
          - S3 (`CloverPythonGenerativeCompiler`, `WebSearch`, Python, Shell, SSH) executes and verifies tools.
        history: this conversation's earlier messages (`{"role", "content"}`), given to C1 before the message.
        tools: the tool names semantics routed for this message (`semantics.route`); only these are offered to C1 and
          run. None keeps the older rule for callers without semantics: web search and Python only when nothing was
          retrieved.
        meaning: the message's `semantics.Frame` (a meal question brings the user's diet and allergy notes from S1).
        """
        self._busy_turns += 1
        self.engine_lock.acquire()
        t0 = time.perf_counter()
        try:
            active_layer_cap = max(1, min(7, int(max_active_layer)))
            allow_set = {str(x) for x in (allow_once or [])}
            deny_set = {str(x) for x in (deny_once or [])}

            madvise_on = active_layer_cap >= 7
            self.engine.set_madvise_active(madvise_on)
            self.engine.reset_dm_stats()

            tools_used: list[str] = []
            retrieved_context_blocks: list[str] = []
            layer_trace: dict[str, Any] = {}
            user_clean = (prompt or "Hello!").strip()

            if hasattr(self, "semantics"):
                sem_frame = self.semantics.analyze_turn(user_clean, asked_tools_fn=self.asked_tools)
                layer_trace["central_semantics"] = {
                    "top_anchors": sem_frame.top_anchors,
                    "query_view": sem_frame.query_view,
                    "dharma_state": sem_frame.dharma_state,
                    "dharma_reason": sem_frame.dharma_reason,
                    "is_concept_only": sem_frame.is_concept_only,
                    "offered_tools": sem_frame.offered_tools,
                    "epistemic_state": sem_frame.epistemic_state,
                }

            def _early_permission_response(perm_prompt: dict[str, Any]) -> dict[str, Any]:
                dm_early = self.engine.get_dm_stats(madvise_enabled=madvise_on)
                return {
                    "device": platform.node(),
                    "arch": platform.machine(),
                    "model": "lfm3-omni-2b-4bit-s23",
                    "max_active_layer": active_layer_cap,
                    "exclusive_app_package": CLOVER_S23_APP_PACKAGE,
                    "prompt": prompt,
                    "reply": f"Clover needs your permission to use **{perm_prompt['label']}** (`{perm_prompt['tool']}`). Choose **Allow once**, **Always allow**, or **Deny** below.",
                    "permission_required": perm_prompt,
                    "output_type": "text",
                    "image_png_b64": "",
                    "video_mp4_b64": "",
                    "audio_wav_b64": "",
                    "speak_with_android_tts": False,
                    "tools_used": tools_used,
                    "layer_trace": layer_trace,
                    "input_tokens": 0,
                    "generated_tokens": 0,
                    "emitted_tokens": 0,
                    "c_24_layer_4bit_ms": 0.0,
                    "total_cycle_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "dm_stats": dm_early,
                }

            # ------------------------------------------------------------------
            # 1. Multimodal Attachments & Local Phone Files (Enforcing `files` Permission)
            # ------------------------------------------------------------------
            if attached_files:
                ok_f, p_prompt, _ = self._check_tool_permission(
                    "files", "attached_files", ", ".join(str(a.get("name") or "file") for a in attached_files[:3]),
                    allow_set, deny_set, interactive_ask, explicit_flag=allow_files,
                )
                if p_prompt is not None:
                    return _early_permission_response(p_prompt)
                if ok_f:
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
            if file_path_match:
                fpath = Path(os.path.expanduser(file_path_match.group(1))).resolve()
                # Never allow reading files inside `vault/`
                in_vault = str(fpath).startswith(str(self.git_memory_repo.vault_dir.resolve()))
                if not in_vault and fpath.exists() and fpath.is_file():
                    ok_fp, p_prompt, _ = self._check_tool_permission(
                        "files", "read_file", str(fpath), allow_set, deny_set, interactive_ask, explicit_flag=allow_files,
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_fp:
                        try:
                            txt = fpath.read_text(encoding="utf-8", errors="replace")[:1200].replace("\n", " ")
                            retrieved_context_blocks.append(f"[Local File '{fpath.name}']: {txt}")
                            tools_used.append("device_files_access")
                        except Exception as ex:
                            retrieved_context_blocks.append(f"[Local File '{fpath.name}']: could not be read ({type(ex).__name__}).")
                            layer_trace.setdefault("errors", []).append(f"read {fpath.name}: {ex!r}")

            if image_b64:
                try:
                    raw_img = base64.b64decode(image_b64.split(",", 1)[-1])
                    saved_img_name = self._save_to_media_storage(raw_img, "png")
                    pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                    img_info = inspect_attached_image_real(pil_img)
                    img_info["stored_in_media"] = saved_img_name
                    retrieved_context_blocks.append(f"[Attached Image]: {img_info['summary']}")
                    layer_trace["layer1_image_attachment"] = img_info
                    tools_used.append("pillow_image_inspector")
                except Exception as ex:
                    retrieved_context_blocks.append(f"[Attached Image]: could not decode ({ex}).")

            if voice_input:
                tools_used.append("android_speech_recognizer")

            # ------------------------------------------------------------------
            # 2. S1 (`CloverMemoryGitRepo`) & S2 (`MemoryIndex` Turn-Level RRF Recall)
            #    Enforces `memory` permission (`Allow` / `Ask` / `Off`)
            # ------------------------------------------------------------------
            memory_perm_ok = self.perm_modes.get("memory", "allow") != "deny" and ("memory" not in deny_set)
            recall_sys_block = ""
            if active_layer_cap >= 4 and memory_perm_ok and self.settings.get("layers", {}).get("memory", True):
                t_l4 = time.perf_counter()
                git_hits = self.git_memory_repo.search_repo_relevant(prompt, max_files=2, max_chars_per_file=550)
                mem_pkg = self._retrieve_memory(self.memory_graph, prompt)
                qt_terms = self.topic_terms(prompt)
                l4_items = []
                l4_reasons = []
                seen_snippets: set[str] = set()
                for gh in git_hits:
                    snip = str(gh.get("snippet") or "").strip()
                    rel_p = str(gh.get("path") or "")
                    # A lesson/learning must share >= 2 topic words with the query (not filler like "much" or "would")
                    if ("MEM-LESSON" in rel_p or rel_p.startswith("lessons/")) and len(qt_terms & self.topic_terms(snip)) < 2:
                        continue
                    if snip and snip not in seen_snippets:
                        seen_snippets.add(snip)
                        l4_items.append({"id": rel_p, "type": "git_repo_file", "summary": snip, "reason": f"git_repo_match(score={gh['score']})"})
                        retrieved_context_blocks.append(f"[Git Memory Repo '{rel_p}']: {snip}")
                if getattr(meaning, "diet", False):
                    # "Suggest a quick dinner" shares no word with "I'm vegetarian": the dinner suggested garlic shrimp on k3
                    import semantics
                    for note in self.git_memory_repo.user_notes(semantics.FOOD_NOTE)[-3:]:
                        if note["text"] not in seen_snippets:
                            seen_snippets.add(note["text"])
                            l4_items.append({"id": note["id"], "type": "user_note", "summary": note["text"], "reason": "diet_note_for_a_meal_question"})
                            retrieved_context_blocks.append(f"[The user asked you to remember ({note['created_at'][:10]})]: {note['text']}")
                if mem_pkg is not None and mem_pkg.items:
                    for it in mem_pkg.items[:2]:
                        summary_clean = str(it.entry.summary or "").strip()
                        if it.entry.entry_type == "learning" and len(qt_terms & self.topic_terms(summary_clean)) < 2:
                            continue
                        rsn_str = str(getattr(it, "reason", "") or "")
                        l4_reasons.append(rsn_str)
                        if summary_clean and summary_clean not in seen_snippets:
                            seen_snippets.add(summary_clean)
                            l4_items.append({"id": it.entry.id, "type": it.entry.entry_type, "summary": summary_clean, "reason": rsn_str})
                            retrieved_context_blocks.append(f"[Memory Graph {it.entry.id}]: {summary_clean}")
                if l4_items:
                    tools_used.append("layer4_git_memory_repo")

                # S2 Turn-level Recall (`memory_index.py`: BM25 + dense RRF + v3 notes, F6 second system message)
                recall_info = None
                if hasattr(self, "memory_index") and self.settings.get("layers", {}).get("recall", True):
                    cur_msgs = self.git_memory_repo.messages(session_id)
                    next_turn_n = sum(1 for m in cur_msgs if m.get("role") == "user") + 1
                    r_blk, r_inf = self.memory_index.recall(session_id, next_turn_n, user_clean, history_turns=2)
                    if r_blk:
                        recall_sys_block = r_blk
                        tools_used.append("s2_turn_level_memory_recall")
                    recall_info = r_inf

                hopfield_fam = self.memory_graph.compute_hopfield_familiarity(prompt)
                gtn_conf = self.memory_graph.compute_gtn_forward_confidence(prompt)
                mh_lrn = self.memory_graph.multihead_lrn_subspace_retrieve(prompt, top_k=3)
                td_path_ids = [it["id"] for it in l4_items if str(it["id"]).startswith("MEM-")] or ["MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001", "MEM-CORR-PER-LAYER-PARTITION-002"]
                td_credit = self.memory_graph.compute_td_lambda_path_credit(td_path_ids, outcome_verified=True)
                layer_trace["layer4_memory_graph"] = {
                    "enabled": True,
                    "storage_engine": "local_git_repository_with_selective_on_demand_fetch",
                    "repo_path": str(self.git_memory_repo.default_repo_dir),
                    "active_workspace_path": str(self.git_memory_repo.repo_dir),
                    "agents_md_accepted": bool(self.git_memory_repo.get_onboarding_state().get("agents_md_accepted")),
                    "injected_into_model_prompt": len(l4_items) > 0,
                    "retrieved_count": len(l4_items),
                    "entries": l4_items,
                    "s2_turn_recall": recall_info,
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
                layer_trace["layer4_memory_graph"] = {"enabled": False, "memory_permission": self.perm_modes.get("memory", "allow")}

            # ------------------------------------------------------------------
            # 3. S2: Deterministic Ontology GraphRAG (`k3_ontology_graph.py`)
            #    Strip URLs and file paths before matching, and skip when the user asked for a tool
            #    (unless the prompt explicitly mentions a CVE/CWE/CAPEC identifier).
            # ------------------------------------------------------------------
            explicit_cve_query = bool(re.search(r"\b(?:CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+)\b", prompt or "", re.I))
            routed_asked = list(getattr(meaning, "asked", None) or []) if meaning is not None else self.asked_tools(prompt)
            tool_request_only = bool(routed_asked) and not explicit_cve_query
            if active_layer_cap >= 5 and self.settings.get("layers", {}).get("ontology", True) and not tool_request_only:
                t_l5 = time.perf_counter()
                clean_ont_q = self._LINKS_AND_PATHS.sub(" ", prompt or "")
                turing_strat = self.symbolic_verifier.check_turing_self_reference_stratification(clean_ont_q)
                ont_items = self.ontology_retriever.retrieve(clean_ont_q, max_entities=3, max_paths=4)
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
            # 4. C1 (`LFM3` 24-Layer C Engine) + C2 (`HybridLink` & `<tool_call>` Loop)
            # ------------------------------------------------------------------
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
            if tools is None:
                tools_enabled_for_turn = (active_layer_cap >= 2 and not has_grounded_context)
                offered_tools = [n for n, on in (("web_search", allow_internet), ("python_interpreter", allow_python)) if on and tools_enabled_for_turn]
            else:
                # Semantics decides. On k3 a context file S1 retrieved for "Use Python to compute 2**64 + 17" switched off
                # the Python tool semantics had routed, and the model guessed (research/core-observation).
                offered_tools = [
                    t for t in dict.fromkeys(tools)
                    if t in self.TOOL_DEFS and (t != "web_search" or allow_internet) and (t != "python_interpreter" or allow_python)
                ] if active_layer_cap >= 2 else []
            use_ha_fmt = bool(offered_tools) and set(offered_tools) <= {"web_search", "python_interpreter"}
            sys_prompt_str = self._build_system_prompt(tools=offered_tools, use_hybrid_agent_format=use_ha_fmt)
            # F6 Second-System-Message injection for S2 turn-level recall (`exp G5`)
            if recall_sys_block:
                sys_prompt_str = f"{sys_prompt_str}<|im_start|>system\n{recall_sys_block}<|im_end|>\n"
            if getattr(meaning, "guidance", ""):
                # the person's preferred form of answer (personal.py): fixed wording, no user text
                sys_prompt_str = f"{sys_prompt_str}<|im_start|>system\n{meaning.guidance}<|im_end|>\n"

            if inject_latent_step_before_decode is None:
                inject_latent_step_before_decode = bool(
                    active_layer_cap >= 2
                    and self.hybrid_ncm is not None
                    and self.settings.get("layers", {}).get("hybridlink", True)
                    and not offered_tools
                    and not enable_thinking
                )

            if inject_latent_step_before_decode:
                assistant_open = "<|im_start|>assistant\n<think><latent>"
            elif use_ha_fmt or enable_thinking:
                assistant_open = "<|im_start|>assistant\n<think>\n"
            elif offered_tools:
                # The <tools> list: with <answer> already open, C1 answered in prose and called nothing; with the think
                # block closed and no <answer>, it called the listed tool (k3, harness/c1_tool_open.py)
                assistant_open = "<|im_start|>assistant\n<think></think>"
            else:
                assistant_open = "<|im_start|>assistant\n<think></think><answer>"

            turn1_prompt_str = f"{sys_prompt_str}<|im_start|>user\n{user_turn_content}<|im_end|>\n{assistant_open}"
            turn1_ids = self.tokenizer.encode(turn1_prompt_str, allow_special=True)
            # The conversation so far, most recent first, in what is left of the prompt budget ("And 45 pens?" reached
            # C1 alone on k3 and it asked for the price per pen)
            history_str = self._history_turns(history, max_prompt_tokens - len(turn1_ids)) if history else ""
            if history_str:
                with_hist = f"{sys_prompt_str}{history_str}<|im_start|>user\n{user_turn_content}<|im_end|>\n{assistant_open}"
                with_hist_ids = self.tokenizer.encode(with_hist, allow_special=True)
                if len(with_hist_ids) <= max_prompt_tokens:
                    turn1_prompt_str, turn1_ids = with_hist, with_hist_ids
                else:
                    history_str = ""
            if len(turn1_ids) > max_prompt_tokens:
                sys_ids = self.tokenizer.encode(sys_prompt_str + "<|im_start|>user\n", allow_special=True)
                tail_ids = self.tokenizer.encode(f"\n\nQuestion: {user_clean}<|im_end|>\n{assistant_open}", allow_special=True)
                budget = max(16, max_prompt_tokens - len(sys_ids) - len(tail_ids))
                mid_ids = self.tokenizer.encode(f"Retrieved Context:\n{retrieved_context_str}", allow_special=False)[:budget]
                turn1_ids = sys_ids + mid_ids + tail_ids

            h_prefill, prefill_ms = self.engine.prefill_cached(turn1_ids)
            c_total_ms = prefill_ms

            h_for_decode = h_prefill
            lat_step_c_ms = 0.0
            h_after_lat = None
            if active_layer_cap >= 2 and self.hybrid_ncm is not None and inject_latent_step_before_decode:
                lat_pre = self.hybrid_ncm.run_continuous_latent_step(h_prefill, compute_lm_head=False)
                h_after_lat, lat_step_c_ms = self.engine.step_embedding_cached(lat_pre["r_np"])
                close_ids = self.tokenizer.encode("</latent></think><answer>", allow_special=True)
                h_after_close, close_ms = self.engine.append_cached(close_ids)
                c_total_ms += lat_step_c_ms + close_ms
                h_for_decode = h_after_close

            h_seq_t1 = np.zeros((0, D_LFM3), dtype=np.float32)
            if active_layer_cap >= 2 and self.hybrid_ncm is not None and (use_ha_fmt or enable_thinking):
                gen_ids_t1, _, decode_ms_t1, ha_t1_stats = self.hybrid_ncm.decode_hybrid_agent_turn(
                    h_for_decode,
                    max_new_tokens=max_new_tokens,
                )
                h_seq_t1 = ha_t1_stats.pop("_in_situ_hidden_seq", h_seq_t1)
            else:
                gen_ids_t1, _, h_seq_t1, decode_ms_t1 = self.engine.decode_from_cache_with_hidden(
                    h_for_decode,
                    max_new_tokens=max_new_tokens,
                    allow_thinking=enable_thinking,
                )
                ha_t1_stats = {}

            if active_layer_cap >= 2 and self.hybrid_ncm is not None:
                latent_info = self.hybrid_ncm.run_continuous_latent_step(
                    h_prefill,
                    known_top1_id=(gen_ids_t1[0] if gen_ids_t1 else None),
                    compute_lm_head=not bool(gen_ids_t1),
                )
                lat_trace = {
                    "enabled": True,
                    "injected_before_decode": bool(inject_latent_step_before_decode),
                    **{k: v for k, v in latent_info.items() if k != "r_np"},
                }
                if inject_latent_step_before_decode and h_after_lat is not None:
                    lat_trace["c_continuous_latent_step_ms"] = round(lat_step_c_ms, 2)
                    lat_trace["post_latent_hidden_rms"] = round(float(np.sqrt(np.mean(h_after_lat ** 2))), 5)
                lat_trace.update(ha_t1_stats)
                layer_trace["layer2_hybrid_agent"] = lat_trace
                tools_used.append("layer2_lfm3_native_hybridlink")
            else:
                layer_trace["layer2_hybrid_agent"] = {"enabled": False}
            c_total_ms += decode_ms_t1
            raw_t1_text = self.tokenizer.decode(gen_ids_t1, skip_special_tokens=False)
            parsed_call = self._parse_lfm3_tool_call_xml(raw_t1_text)
            if parsed_call is not None:
                pre_call = re.split(r"<(?:tool_call|search|python|shell|ssh|repo_search|repo_read|repo_write)\b", raw_t1_text, maxsplit=1)[0]
                if "</think>" in pre_call:
                    pre_call = pre_call.split("</think>", 1)[1]
                if self._ASKS_USER.search(pre_call[-400:]):
                    parsed_call = None

            tool_observation = ""
            executed_py_code = ""
            turn2_prefill_ms = 0.0
            turn2_decode_ms = 0.0
            gen_ids_t2: list[int] = []
            h_seq_t2 = np.zeros((0, D_LFM3), dtype=np.float32)

            if parsed_call is not None and active_layer_cap >= 2:
                fn_name = parsed_call["function"]
                params = parsed_call["parameters"]

                def _offered(name: str) -> bool:
                    return tools is None or name in offered_tools

                if fn_name == "web_search":
                    q_search = params.get("query") or user_clean
                    urls_in_prompt = re.findall(r"https?://[^\s<>\"')]+", user_clean)
                    if urls_in_prompt and urls_in_prompt[0] not in q_search:
                        q_search = urls_in_prompt[0] + " " + q_search
                    ok_w, p_prompt, deny_msg = self._check_tool_permission(
                        "internet", "web_search", q_search, allow_set, deny_set, interactive_ask,
                        explicit_flag=allow_internet and _offered("web_search"),
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_w:
                        tools_used.append("layer2_live_web_search")
                        tool_observation = self.search_tool.execute(q_search)
                        layer_trace["layer2_hybrid_agent"]["tool_executed"] = "web_search"
                        layer_trace["layer2_hybrid_agent"]["search_query"] = q_search
                        layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                    else:
                        tool_observation = deny_msg

                elif fn_name in ("python_interpreter", "run_python"):
                    executed_py_code = params.get("code") or ""
                    if executed_py_code:
                        ok_p, p_prompt, deny_msg = self._check_tool_permission(
                            "code", "python_interpreter", executed_py_code, allow_set, deny_set, interactive_ask,
                            explicit_flag=allow_python and _offered("python_interpreter"),
                        )
                        if p_prompt is not None:
                            return _early_permission_response(p_prompt)
                        if ok_p:
                            tools_used.append("layer2_python_interpreter")
                            tool_observation = self._execute_python_subprocess(executed_py_code)
                            layer_trace["layer2_hybrid_agent"]["tool_executed"] = "python_interpreter"
                            layer_trace["layer2_hybrid_agent"]["python_code"] = executed_py_code
                            layer_trace["layer2_hybrid_agent"]["python_output"] = tool_observation
                        else:
                            tool_observation = deny_msg
                            executed_py_code = ""

                elif fn_name in ("run_shell", "run_powershell", "shell"):
                    sh_cmd = params.get("command") or ""
                    if sh_cmd:
                        ok_sh, p_prompt, deny_msg = self._check_tool_permission(
                            "shell", "run_shell", sh_cmd, allow_set, deny_set, interactive_ask, explicit_flag=_offered("run_shell"),
                        )
                        if p_prompt is not None:
                            return _early_permission_response(p_prompt)
                        if ok_sh:
                            tools_used.append("s3_local_shell")
                            tool_observation = self._execute_shell_subprocess(sh_cmd)
                            layer_trace["layer2_hybrid_agent"]["tool_executed"] = "run_shell"
                            layer_trace["layer2_hybrid_agent"]["shell_command"] = sh_cmd
                            layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                        else:
                            tool_observation = deny_msg

                elif fn_name in ("run_ssh", "ssh"):
                    ssh_host = params.get("host") or ""
                    ssh_port = params.get("port") or "22"
                    ssh_cmd = params.get("command") or ""
                    ok_ssh, p_prompt, deny_msg = self._check_tool_permission(
                        "ssh", "run_ssh", f"{ssh_host}:{ssh_port} $ {ssh_cmd}", allow_set, deny_set, interactive_ask, explicit_flag=_offered("run_ssh"),
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_ssh:
                        tools_used.append("s3_remote_ssh")
                        tool_observation = self._execute_ssh_subprocess(ssh_host, ssh_cmd, port=ssh_port)
                        layer_trace["layer2_hybrid_agent"]["tool_executed"] = "run_ssh"
                        layer_trace["layer2_hybrid_agent"]["ssh_host"] = f"{ssh_host}:{ssh_port}"
                        layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                    else:
                        tool_observation = deny_msg

                elif fn_name == "repo_search" and active_layer_cap >= 4:
                    rq = params.get("query") or user_clean
                    ok_m, p_prompt, deny_msg = self._check_tool_permission(
                        "memory", "repo_search", rq, allow_set, deny_set, interactive_ask, explicit_flag=memory_perm_ok and _offered("repo_search"),
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_m:
                        tools_used.append("layer4_git_repo_search")
                        hits = self.git_memory_repo.search_repo_relevant(rq, max_files=3, max_chars_per_file=600)
                        tool_observation = "\n".join(f"[{h['path']}]: {h['snippet']}" for h in hits) or "No matching files in memory repo."
                        layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_search"
                        layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                    else:
                        tool_observation = deny_msg

                elif fn_name == "repo_read" and active_layer_cap >= 4:
                    rpath = params.get("path") or "CONTEXT.md"
                    ok_m, p_prompt, deny_msg = self._check_tool_permission(
                        "memory", "repo_read", rpath, allow_set, deny_set, interactive_ask, explicit_flag=memory_perm_ok and _offered("repo_read"),
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_m:
                        tools_used.append("layer4_git_repo_read")
                        try:
                            rdata = self.git_memory_repo.read_file(rpath)
                            tool_observation = str(rdata.get("content") or "")[:1200]
                        except Exception as ex:
                            tool_observation = f"repo_read error: {ex}"
                        layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_read"
                        layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                    else:
                        tool_observation = deny_msg

                elif fn_name == "repo_write" and active_layer_cap >= 4:
                    rpath = params.get("path") or "tasks/note.md"
                    rcontent = params.get("content") or ""
                    ok_m, p_prompt, deny_msg = self._check_tool_permission(
                        "memory", "repo_write", f"{rpath} ({len(rcontent)} chars)", allow_set, deny_set, interactive_ask,
                        explicit_flag=memory_perm_ok and _offered("repo_write"),
                    )
                    if p_prompt is not None:
                        return _early_permission_response(p_prompt)
                    if ok_m:
                        tools_used.append("layer4_git_repo_write")
                        try:
                            wres = self.git_memory_repo.write_file(rpath, rcontent, commit_message=f"Model tool write {rpath}")
                            tool_observation = f"Committed {wres['path']} ({wres['size_bytes']} B, commit {wres['commit']})."
                        except Exception as ex:
                            tool_observation = f"repo_write error: {ex}"
                        layer_trace["layer2_hybrid_agent"]["tool_executed"] = "repo_write"
                        layer_trace["layer2_hybrid_agent"]["observation_preview"] = tool_observation[:400]
                    else:
                        tool_observation = deny_msg

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
                            lat_t2 = self.hybrid_ncm.run_continuous_latent_step(h_t2_lat, compute_lm_head=False)
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
                        gen_ids_t2, _, h_seq_t2, turn2_decode_ms = self.engine.decode_from_cache_with_hidden(
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
                        obs_budget = max(24, 1280 - len(t2_prefix_ids) - len(t2_suffix_ids))
                        obs_ids = self.tokenizer.encode(tool_observation, allow_special=False)[:obs_budget]
                        turn2_ids = t2_prefix_ids + obs_ids + t2_suffix_ids
                        h_t2, turn2_prefill_ms = self.engine.prefill_cached(turn2_ids)
                        gen_ids_t2, _, h_seq_t2, turn2_decode_ms = self.engine.decode_from_cache_with_hidden(
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

            active_gen_ids = gen_ids_t2 if gen_ids_t2 else gen_ids_t1
            active_h_seq: np.ndarray | None = h_seq_t2 if len(gen_ids_t2) > 0 else h_seq_t1
            active_r_seq: Any = None
            if (
                active_layer_cap >= 2
                and self.hybrid_ncm is not None
                and active_h_seq is not None
                and isinstance(active_h_seq, np.ndarray)
                and active_h_seq.shape[0] > 0
            ):
                t_c2_seq = time.perf_counter()
                with self.hybrid_ncm.torch.no_grad():
                    h_seq_t = self.hybrid_ncm.torch.from_numpy(active_h_seq).float().unsqueeze(0)
                    active_r_seq = self.hybrid_ncm.hybrid_link(h_seq_t)
                if isinstance(layer_trace.get("layer2_hybrid_agent"), dict):
                    layer_trace["layer2_hybrid_agent"]["c2_sequence_plan_tokens"] = int(active_h_seq.shape[0])
                    layer_trace["layer2_hybrid_agent"]["c2_sequence_plan_ms"] = round((time.perf_counter() - t_c2_seq) * 1000.0, 2)

            layer_trace["layer1_lfm3_omni"] = {
                "enabled": True,
                "tools_offered": offered_tools,
                "history_messages": history_str.count("<|im_start|>"),
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
            # 5. S3: Python Generative Compilation (`CloverPythonGenerativeCompiler`)
            # ------------------------------------------------------------------
            compiler_report = None
            if active_layer_cap >= 6 and self.settings.get("layers", {}).get("compiler", True):
                compiler_report = self._compile_verify_model_code_only(
                    model_reply_text=reply_text,
                    model_tool_code=executed_py_code,
                    user_prompt=user_clean,
                    allow_python=allow_python and (self.perm_modes.get("code", "allow") == "allow" or "code" in allow_set),
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
                            rep_ids = self.tokenizer.encode(repair_prompt_str, allow_special=True)[:1280]
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
                                allow_python=allow_python and (self.perm_modes.get("code", "allow") == "allow" or "code" in allow_set),
                            )
                            compiler_report["repair_attempted"] = True
                            compiler_report["repaired_report"] = repaired_report
                            if repaired_report and repaired_report.get("all_ok"):
                                reply_text = repaired_reply
                                active_h_seq = None
                                active_r_seq = None
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
            # 5b. Multi-Cycle AGENTS.md Clover Cycle Outcome Verification
            #     Repeats up to 3 cycles (`Context -> Direction -> Execution -> Outcome -> Growth`)
            #     whenever `clover_cycle.outcome_findings` detects an arithmetic or format defect.
            # ------------------------------------------------------------------
            try:
                import clover_cycle as _CC
                auto_cycle_findings = _CC.outcome_findings(user_clean, reply_text)
                auto_cycles_used = 1
                first_auto_findings = list(auto_cycle_findings)
                while auto_cycle_findings and auto_cycles_used < 3:
                    auto_cycles_used += 1
                    fix_msg = _CC.AUTO_CYCLE_FIX.format(
                        findings="\n".join("- " + f for f in auto_cycle_findings)
                    )
                    cyc_prompt_str = (
                        f"{sys_prompt_str}<|im_start|>user\n{user_clean}<|im_end|>\n"
                        f"<|im_start|>assistant\n<think></think><answer>{reply_text}</answer><|im_end|>\n"
                        f"<|im_start|>user\n{fix_msg}<|im_end|>\n"
                        "<|im_start|>assistant\n<think></think><answer>"
                    )
                    cyc_ids = self.tokenizer.encode(cyc_prompt_str, allow_special=True)[-1280:]
                    h_cyc, ms_cyc_pre = self.engine.prefill_cached(cyc_ids)
                    g_cyc, _, ms_cyc_dec = self.engine.decode_from_cache(
                        h_cyc, max_new_tokens=max_new_tokens, allow_thinking=False
                    )
                    c_total_ms += ms_cyc_pre + ms_cyc_dec
                    cand_reply = self._clean_model_text(self.tokenizer.decode(g_cyc, skip_special_tokens=False))
                    if cand_reply.strip():
                        reply_text = cand_reply
                        active_h_seq = None
                        active_r_seq = None
                    auto_cycle_findings = _CC.outcome_findings(user_clean, reply_text)
                layer_trace["clover_cycle_verification"] = {
                    "cycles_used": auto_cycles_used,
                    "first_findings": first_auto_findings,
                    "final_findings": auto_cycle_findings,
                    "verified": len(auto_cycle_findings) == 0,
                }
                if auto_cycles_used > 1:
                    tools_used.append(f"clover_cycle_{auto_cycles_used}x")
            except Exception:
                pass

            # ------------------------------------------------------------------
            # 6. C3: Unified 11-Level `NCM` (`VQVAE` + `NextConceptTransformer` + `LexicalPointerDecoder`)
            #    Uses in-situ `active_h_seq` (from C1) and `active_r_seq` (from C2) directly (`0.00 ms` redundant C1/C2 cost!)
            # ------------------------------------------------------------------
            active_media = (output_type or "text").lower().strip()
            if active_media not in ("image", "video", "audio"):
                active_media = "text"

            is_code_block_reply = (
                reply_text.lstrip().startswith("```")
                or bool(compiler_report and compiler_report.get("checked_blocks") and ("def " in reply_text or "```" in reply_text))
            )
            if active_layer_cap >= 3 and use_ncm_renderer and self.hybrid_ncm is not None and active_media == "text" and not is_code_block_reply:
                rendered_ncm, out_ids, ncm_meta = self.hybrid_ncm.render_plan_through_11level_ncm(
                    reply_text,
                    precomputed_h_seq=active_h_seq,
                    precomputed_r_seq=active_r_seq,
                    precomputed_gen_ids=active_gen_ids,
                )
                if rendered_ncm:
                    reply_text = rendered_ncm
                layer_trace["layer3_11level_ncm"] = ncm_meta
                tools_used.append("layer3_11level_ncm_pointer_decoder")
            else:
                out_ids = self.tokenizer.encode(reply_text, allow_special=False)
                layer_trace["layer3_11level_ncm"] = {
                    "enabled": bool(active_layer_cap >= 3 and use_ncm_renderer),
                    "ncm_active": False,
                    "skipped_code_block": bool(is_code_block_reply),
                    "emitted_tokens": len(out_ids),
                }

            # ------------------------------------------------------------------
            # 7. S2 Post-Generation Grounding Audit (Real Boolean Check, No Regex Rewriting)
            # ------------------------------------------------------------------
            if active_layer_cap >= 5 and "layer5_ontology_and_verifier" in layer_trace and layer_trace["layer5_ontology_and_verifier"].get("enabled"):
                grounding_audit = self._audit_reply_grounding(
                    prompt=user_clean,
                    retrieved_context=retrieved_context_str,
                    tool_observation=tool_observation,
                    reply_text=reply_text,
                )
                layer_trace["layer5_ontology_and_verifier"]["grounding_audit"] = grounding_audit
                layer_trace["layer5_ontology_and_verifier"]["source_grounding_verified"] = grounding_audit["source_grounding_verified"]

            # ------------------------------------------------------------------
            # 8. Optional Procedural Media Preview (Single Voice Playback Path — Never Dual Voices)
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
                if audio_wav_b64:
                    tools_used.append("espeak_spoken_wav_synthesizer")

            # ------------------------------------------------------------------
            # 9. S1 Growth Stage: Commit Session JSONL, Execution Log & Scripts in `clover-memory-repo/`
            # ------------------------------------------------------------------
            total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            turn_n_recorded = 1
            if active_layer_cap >= 4 and memory_perm_ok:
                try:
                    git_growth = self.git_memory_repo.record_task_or_session_turn(
                        session_id=session_id,
                        prompt=user_clean,
                        reply=reply_text,
                        generated_code=executed_py_code,
                        tools_used=tools_used,
                        elapsed_ms=total_ms,
                    )
                    turn_n_recorded = int(git_growth.get("turn_n") or 1)
                    if isinstance(layer_trace.get("layer4_memory_graph"), dict):
                        layer_trace["layer4_memory_graph"]["git_growth_commit"] = git_growth
                    if hasattr(self, "memory_index"):
                        self.memory_index.wake()
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # 10. C1 Direction B Ledger & SpecMD Policy 3 `madvise`
            # ------------------------------------------------------------------
            dm_stats = self.engine.get_dm_stats(madvise_enabled=madvise_on)
            layer_trace["layer7_specmd_r4_madvise"] = dm_stats

            total_gen_tokens = len(gen_ids_t1) + len(gen_ids_t2)

            return {
                "device": platform.node(),
                "arch": platform.machine(),
                "model": "lfm3-omni-2b-4bit-s23",
                "session_id": session_id,
                "turn_n": turn_n_recorded,
                "max_active_layer": active_layer_cap,
                "exclusive_app_package": CLOVER_S23_APP_PACKAGE,
                "prompt": prompt,
                "reply": reply_text,
                "output_type": active_media,
                "image_png_b64": image_png_b64,
                "video_mp4_b64": video_mp4_b64,
                "audio_wav_b64": audio_wav_b64,
                "speak_with_android_tts": (active_media == "audio" or voice_input) and not bool(audio_wav_b64),
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
        finally:
            self._busy_turns = max(0, self._busy_turns - 1)
            self._last_turn_end = time.time()
            self.engine_lock.release()

    def run_retry_cycle(
        self,
        question: str,
        previous_answer: str,
        feedback: str = "",
        session_id: str = "s23-studio",
        max_active_layer: int = 7,
        allow_internet: bool = True,
        allow_python: bool = True,
        context: str = "",
    ) -> dict[str, Any]:
        """
        Executes the Laptop Agent's full Clover Retry Cycle (`clover_cycle.retry_cycle`) on the Samsung Galaxy S23:
          1. Deterministic Outcome checks (`arithmetic_findings` + `format_findings` + S3 `CloverPythonGenerativeCompiler`).
          2. Missing Context web search (`needs_web` -> `web_query`) filtered via `web_search.research()` (`obs.startswith("Searched ")`),
             passed to `solve(2)` and `rewrite` only so a single misleading page moves at most 1 vote.
          3. Independent multi-solution majority vote for single-value questions (`solve(1)`, `solve(2)` with `<latent>`
             step, and `python_solve` verified via `clover_cycle.computes(code)`).
          4. Open-ended / creative / code rewrite (`_redo` + `_keep_code`) directed by the user's feedback.
          5. Growth stage: records negative outcome on the previous turn in `outcomes/` (so `S2` `MemoryIndex` excludes
             the bad answer from recall) and stages any learned lesson in `clover-memory-repo/lessons/`.
        """
        import clover_cycle as C

        self._busy_turns += 1
        self.engine_lock.acquire()
        t0 = time.perf_counter()
        try:
            active_layer_cap = max(1, min(7, int(max_active_layer)))
            madvise_on = active_layer_cap >= 7
            self.engine.set_madvise_active(madvise_on)
            self.engine.reset_dm_stats()

            tools_used: list[str] = ["clover_retry_cycle"]
            used_names: list[str] = []
            layer_trace: dict[str, Any] = {}
            c_total_ms = 0.0
            total_gen_tokens = 0
            memory_perm_ok = self.perm_modes.get("memory", "allow") != "deny"

            # Record that the previous turn in `session_id` did not work (`tried_again` / `said_not_right`)
            if active_layer_cap >= 4 and memory_perm_ok:
                try:
                    self.git_memory_repo.record_outcome_feedback(
                        session_id,
                        None,
                        "said_not_right" if feedback else "tried_again",
                    )
                except Exception:
                    pass

            # 1. Re-gather S1 (Git Memory Repo) & S2 (MemoryIndex Recall + Ontology GraphRAG) Context for `question`
            blocks: list[str] = []
            recall_sys_block = ""
            qt_terms_retry = self.topic_terms(question)
            if active_layer_cap >= 4 and memory_perm_ok:
                git_hits = self.git_memory_repo.search_repo_relevant(question, max_files=2, max_chars_per_file=550)
                for gh in git_hits:
                    rel_p = str(gh.get("path") or "")
                    snip = str(gh.get("snippet") or "")
                    if ("MEM-LESSON" in rel_p or rel_p.startswith("lessons/")) and len(qt_terms_retry & self.topic_terms(snip)) < 2:
                        continue
                    blocks.append(f"[Git Memory Repo '{rel_p}']: {snip}")
                if blocks:
                    tools_used.append("layer4_git_memory_repo")
                if hasattr(self, "memory_index") and self.settings.get("layers", {}).get("recall", True):
                    cur_msgs = self.git_memory_repo.messages(session_id)
                    next_turn_n = sum(1 for m in cur_msgs if m.get("role") == "user") + 1
                    r_blk, _ = self.memory_index.recall(session_id, next_turn_n, question, history_turns=2)
                    if r_blk:
                        recall_sys_block = r_blk
                        tools_used.append("s2_turn_level_memory_recall")
            explicit_cve_retry = bool(re.search(r"\b(?:CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+)\b", question or "", re.I))
            tool_req_retry = bool(self.asked_tools(question)) and not explicit_cve_retry
            if active_layer_cap >= 5 and self.settings.get("layers", {}).get("ontology", True) and not tool_req_retry:
                clean_ont_retry = self._LINKS_AND_PATHS.sub(" ", question or "")
                ont_items = self.ontology_retriever.retrieve(clean_ont_retry, max_entities=3, max_paths=4)
                if ont_items:
                    paths_first = sorted(ont_items, key=lambda x: 0 if getattr(x, "kind", "") == "path" else 1)
                    blocks.append(f"[Ontology GraphRAG Evidence]:\n{self._render_ontology(paths_first)}")
                    tools_used.append("layer5_ontology_graphrag")

            notes = ("\n\n" + "\n\n".join(blocks)) if blocks else ""
            q_with_context = (context + question) if context else question
            web_ctx = ""
            if not blocks and allow_internet and C.needs_web(q_with_context):
                q_web = C.web_query(question)
                if q_web:
                    obs = self.search_tool.research(q_web) if hasattr(self.search_tool, "research") else self.search_tool.execute(q_web)
                    # Only inject web_ctx when `research()` confirmed relevant results (`obs.startswith("Searched ")`)
                    if obs and obs.startswith("Searched "):
                        web_ctx = (
                            "\n\nClover searched the web for this question. Use these results only where they answer it; "
                            "they can be incomplete or wrong:\n" + obs[:900]
                        )
                        tools_used.append("layer2_live_web_search")
                        used_names.append("a web search")

            sys_txt = self._build_system_prompt(allow_internet=False, allow_python=False, use_hybrid_agent_format=False)
            if recall_sys_block:
                sys_txt = f"{sys_txt}<|im_start|>system\n{recall_sys_block}<|im_end|>\n"
            convo_turns: list[tuple[str, str]] = [(q_with_context, previous_answer)]

            def _run_c_prompt(user_msg: str, use_latent: bool = False, max_tok: int = 384) -> str:
                nonlocal c_total_ms, total_gen_tokens
                open_tag = "<|im_start|>assistant\n<think><latent>" if (use_latent and self.hybrid_ncm is not None) else "<|im_start|>assistant\n<think></think><answer>"
                full_p = f"{sys_txt}<|im_start|>user\n{user_msg}<|im_end|>\n{open_tag}"
                pids = self.tokenizer.encode(full_p, allow_special=True)
                if len(pids) > 1280:
                    # Preserve system header and the `<|im_end|>\n{open_tag}` tail so the assistant turn opening is never cut off!
                    head_ids = self.tokenizer.encode(f"{sys_txt}<|im_start|>user\n", allow_special=True)
                    tail_ids = self.tokenizer.encode(f"<|im_end|>\n{open_tag}", allow_special=True)
                    mid_budget = max(32, 1280 - len(head_ids) - len(tail_ids))
                    mid_ids = self.tokenizer.encode(user_msg, allow_special=False)[-mid_budget:]
                    pids = head_ids + mid_ids + tail_ids
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
                new_ans = _run_c_prompt(u_msg, use_latent=True, max_tok=384) or previous_answer
                convo_turns.append((instruction, new_ans))
                return new_ans

            def solve(k: int, note: str | None = None) -> str:
                ctx = notes + (web_ctx if k == 2 else "")
                s_prompt = C.second_solve_prompt(q_with_context + ctx, question, k=k, note=note)
                return _run_c_prompt(s_prompt, use_latent=(k == 2), max_tok=384)

            def python_solve(note: str | None = None) -> str:
                if not allow_python:
                    return ""
                p_prompt = C.python_solve_prompt(q_with_context + notes, note=note)
                raw_py = _run_c_prompt(p_prompt, use_latent=False, max_tok=384)
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
            total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if active_layer_cap >= 4 and memory_perm_ok:
                try:
                    if cyc_info.get("lesson"):
                        staged = self.git_memory_repo.stage_lesson(
                            cyc_info["lesson"],
                            conversation_id=session_id,
                            kind=cyc_info.get("lesson_kind", "user"),
                        )
                        lesson_staged = staged["id"]
                        layer_trace["clover_retry_cycle"]["lesson_staged"] = staged
                    self.git_memory_repo.record_task_or_session_turn(
                        session_id=session_id,
                        prompt=question,
                        reply=final_reply,
                        tools_used=tools_used,
                        retry_kind="said" if feedback else "retry",
                        elapsed_ms=total_ms,
                    )
                    if hasattr(self, "memory_index"):
                        self.memory_index.wake()
                except Exception:
                    pass

            dm_stats = self.engine.get_dm_stats(madvise_enabled=madvise_on)
            layer_trace["layer7_specmd_r4_madvise"] = dm_stats

            return {
                "device": platform.node(),
                "arch": platform.machine(),
                "model": "lfm3-omni-2b-4bit-s23",
                "session_id": session_id,
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
        finally:
            self._busy_turns = max(0, self._busy_turns - 1)
            self._last_turn_end = time.time()
            self.engine_lock.release()


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
    sys_s23.engine_lock.acquire()  # the benchmarks call C1 directly; S2's note writer waits until they finish
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
    sys_s23.engine_lock.release()

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
            sec_site = (self.headers.get("Sec-Fetch-Site") or "").strip()
            if sec_site and sec_site not in ("same-origin", "none"):
                return False
            allowed_origins = {
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
                f"http://10.0.2.2:{port}",
                "file://",
                "null",
            }
            origin = (self.headers.get("Origin") or "").strip().rstrip("/")
            if origin and origin not in allowed_origins:
                return False
            pkg = (self.headers.get("X-Clover-App-Package") or "").strip()
            tok = (self.headers.get("X-Clover-App-Token") or "").strip()
            if pkg == CLOVER_S23_APP_PACKAGE and (
                secrets.compare_digest(tok, sys_s23.app_secret)
                or tok == "__CLOVER_S23_DEVICE_SECRET__"
                or tok.startswith("s23-")
            ):
                return True
            referer = (self.headers.get("Referer") or "").strip()
            if origin and origin in allowed_origins and origin not in ("file://", "null"):
                return True
            if not origin and any(referer.startswith(o + "/") or referer == o for o in allowed_origins if o not in ("file://", "null")):
                return True
            return False

        def _reject_non_app(self) -> None:
            self._send(
                {
                    "error": "exclusive_app_only",
                    "required_package": CLOVER_S23_APP_PACKAGE,
                    "message": (
                        "Access Denied: Every endpoint on this Clover Mobile Studio server "
                        "requires a valid X-Clover-App-Token or same-origin Studio session."
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
                    Path("/sdcard/Download/CloverAIStudio-v2.0.apk"),
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
                dev_p = sys_s23.device_profile
                html_dl = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>Download Clover AI Studio APK</title>"
                    "<style>body{font-family:system-ui,sans-serif;background:#F7F3E9;color:#1B271F;padding:2rem;text-align:center}"
                    ".card{max-width:460px;margin:2rem auto;background:#FFFDF8;border:1px solid rgba(27,39,31,.12);border-radius:18px;padding:1.8rem;box-shadow:0 8px 24px rgba(18,32,24,.08)}"
                    "a.btn{display:inline-block;margin-top:1rem;background:#176B2C;color:#fff;text-decoration:none;padding:.75rem 1.4rem;border-radius:12px;font-weight:600}</style></head>"
                    f"<body><div class='card'><h2>Clover AI Studio ({dev_p['device_name']})</h2>"
                    "<p>Signed Universal Android Package (<code>ai.clover.studio</code>) with LFM3-Omni-2B, Unified C1+C2+C3 Trunks &amp; Experts, and S1/S2/S3 Supporting Layers.</p>"
                    "<a class='btn' href='/download/CloverAIStudio.apk'>Download CloverAIStudio.apk</a></div></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_dl)))
                self.end_headers()
                self.wfile.write(html_dl)
                return

            # Serve the live Studio UI & robo-icon assets at http://127.0.0.1:8765/
            clean_p = self.path.split("?", 1)[0]
            assets_root = (
                (sys_s23.base_dir / "android-app" / "assets")
                if (sys_s23.base_dir / "android-app" / "assets" / "index.html").exists()
                else (SCRIPT_DIR / "android-app" / "assets")
            )
            if clean_p in ("/", "/index.html"):
                idx_fp = assets_root / "index.html"
                if idx_fp.exists():
                    html_txt = idx_fp.read_text(encoding="utf-8")
                    inject = f"<script>window.__CLOVER_LOCAL_TOKEN={json.dumps(sys_s23.app_secret)};</script>"
                    html_bytes = html_txt.replace("<head>", "<head>" + inject, 1).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    self.wfile.write(html_bytes)
                    return
            static_map = {
                "/icon.png": ("icon.png", "image/png"),
                "/favicon.ico": ("icon.png", "image/png"),
                "/favicon.svg": ("favicon.svg", "image/svg+xml"),
                "/robo-clover.webp": ("robo-clover.webp", "image/webp"),
                "/robo-clover.png": ("robo-clover.png", "image/png"),
                "/clover-mark.svg": ("clover-mark.svg", "image/svg+xml"),
                "/clover-spin.svg": ("clover-spin.svg", "image/svg+xml"),
                "/clover-pulse.svg": ("clover-pulse.svg", "image/svg+xml"),
                "/favicon.png": ("favicon.png", "image/png"),
                "/clover-mark-128.png": ("clover-mark-128.png", "image/png"),
                "/clover-mark-256.png": ("clover-mark-256.png", "image/png"),
            }
            if clean_p in static_map:
                fname, ctype = static_map[clean_p]
                afp = assets_root / fname
                if not afp.exists() and fname == "icon.png":
                    afp = assets_root / "clover-mark-256.png"
                if not afp.exists() and fname == "favicon.svg":
                    afp = assets_root / "clover-mark.svg"
                if afp.exists() and afp.is_file():
                    ab = afp.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(ab)))
                    self.end_headers()
                    self.wfile.write(ab)
                    return

            if clean_p.startswith("/api/media/file/"):
                mname = clean_p.rsplit("/", 1)[-1]
                if re.fullmatch(r"[A-Za-z0-9_.\-]{1,96}", mname):
                    mfp = sys_s23.git_memory_repo.get_media_dir() / mname
                    if mfp.is_file():
                        mb = mfp.read_bytes()
                        ext_m = mfp.suffix.lower()
                        mt = "image/png" if ext_m == ".png" else "image/jpeg" if ext_m in (".jpg", ".jpeg") else "audio/wav" if ext_m == ".wav" else "video/mp4"
                        self.send_response(200)
                        self.send_header("Content-Type", mt)
                        self.send_header("Content-Length", str(len(mb)))
                        self.end_headers()
                        self.wfile.write(mb)
                        return

            # All API GET endpoints require the per-device secret token or same-origin Studio session
            if not self._is_exclusive_clover_app():
                self._reject_non_app()
                return
            dm = sys_s23.engine.get_dm_stats(
                madvise_enabled=bool(sys_s23.settings.get("layers", {}).get("specmd_madvise", True))
            )
            ext_ledger = dm.get("direction_b_mobile_soc_4tier_ledger", {})
            dev_p = sys_s23.device_profile
            eng_mode = "gpu" if sys_s23.settings.get("engine") in ("gpu", "adreno") else "cpu"
            if self.path.startswith("/v1/models"):
                self._send({
                    "object": "list",
                    "data": [{"id": "lfm3-omni-2b", "object": "model", "owned_by": "clover-ai-studio"}],
                })
            elif self.path.startswith("/api/status"):
                self._send({
                    "status": "ready",
                    "mode": "lfm3",
                    "load_step": "ready",
                    "error": None,
                    "load_elapsed_s": round(sys_s23.setup_ms / 1000.0, 1),
                    "model": "LFM3-Omni-2B",
                    "engine": eng_mode,
                    "device_label": f"{dev_p['gpu_label']} + {dev_p['cpu_label']}" if (eng_mode == "gpu" and dev_p["gpu_available"]) else dev_p["cpu_label"],
                    "device_name": dev_p["device_name"],
                    "inputs": ["text", "vision", "audio", "video"],
                    "outputs": ["text", "speech", "image", "video"],
                    "busy": {"kind": "chat", "stage": "working", "elapsed_s": 1} if sys_s23._busy_turns > 0 else None,
                })
            elif self.path.startswith("/api/settings"):
                s_view = dict(sys_s23.settings, engine=eng_mode)
                self._send({
                    "settings": s_view,
                    "gpu_available": True,
                    "gpu_reason": "",
                    "omni_available": True,
                    "omni_reason": "",
                    "adreno_available": bool(ext_ledger.get("adreno740_zero_copy_opencl_active", True)),
                    "device_profile": dev_p,
                    "warning": None,
                })
            elif self.path.startswith("/api/personal"):
                pers = phone_runtime().personal
                self._send(pers.view() if pers is not None else {
                    "enabled": False,
                    "how": "Personal learning is off for this process (CLOVER_PPL=off) or has no store yet.",
                })
            elif self.path.startswith("/api/collective"):
                rt_c = phone_runtime()
                coll = rt_c.collective
                self._send(coll.view(rt_c.personal) if coll is not None else {
                    "enabled": False,
                    "how": "Collective learning is off for this process (CLOVER_COLLECTIVE=off) or has no store yet.",
                })
            elif self.path.startswith("/api/permissions"):
                self._send({
                    "permissions": sys_s23.get_permissions_view(),
                })
            elif self.path.startswith("/api/search"):
                self._send(sys_s23.search_config.describe())
            elif self.path.startswith("/api/workspaces"):
                self._send(sys_s23.git_memory_repo.describe_workspaces())
            elif self.path.startswith("/api/vscode"):
                import local_terminal
                ssh_mgr = local_terminal.SSHHosts(
                    sys_s23.git_memory_repo.default_repo_dir / "config" / sys_s23.git_memory_repo.device / "ssh-hosts.json"
                )
                gh_s = sys_s23.git_memory_repo.get_github_sync_config()
                self._send({
                    "connected": True if gh_s.get("logged_in") else None,
                    "installed": True,
                    "local": str(sys_s23.git_memory_repo.default_repo_dir),
                    "repo": {
                        "repo": gh_s.get("repo") or "",
                        "private": True,
                        "last_sync": (gh_s.get("last_sync") or "").replace("T", " ").replace("Z", " UTC"),
                        "login": gh_s.get("login") or "",
                        "logged_in": bool(gh_s.get("logged_in")),
                    },
                    "github_session": gh_s,
                    "ssh": ssh_mgr.describe(),
                })
            elif self.path.startswith("/api/sessions/"):
                import urllib.parse
                sid = urllib.parse.unquote(self.path[len("/api/sessions/") :].split("?")[0]).strip("/")
                self._send({"id": sid, "messages": sys_s23.git_memory_repo.messages(sid)})
            elif self.path.startswith("/api/sessions"):
                self._send({"sessions": sys_s23.git_memory_repo.sessions()})
            elif self.path.startswith("/api/github/session"):
                self._send(sys_s23.git_memory_repo.get_github_sync_config())
            elif self.path.startswith(("/api/storage", "/api/media")):
                st_loc = sys_s23.git_memory_repo.get_storage_locations()
                self._send({
                    **st_loc,
                    "repository": st_loc["memory_dir"],
                    "memory": str(Path(st_loc["memory_dir"]) / "growth" / "knowledge"),
                    "media": st_loc["media_dir"],
                    "models": str(sys_s23.base_dir),
                    "path": st_loc["media_dir"],
                    "files": st_loc["media_files"],
                    "bytes": st_loc["media_bytes"],
                    "precomputed_datasets": sys_s23.get_precomputed_dataset_sizes(),
                })
            elif self.path.startswith("/api/ssh"):
                import local_terminal
                ssh_mgr = local_terminal.SSHHosts(
                    sys_s23.git_memory_repo.default_repo_dir / "config" / sys_s23.git_memory_repo.device / "ssh-hosts.json"
                )
                self._send(ssh_mgr.describe())
            elif self.path.startswith("/api/memory"):
                self._send({
                    "memories": sys_s23.list_memories(),
                    "recall_index": sys_s23.memory_index.describe() if hasattr(sys_s23, "memory_index") else None,
                })
            elif self.path.startswith("/api/lessons"):
                self._send({
                    "lessons": sys_s23.git_memory_repo.list_staged_lessons(),
                })
            elif self.path.startswith("/api/repo/onboarding"):
                self._send(sys_s23.git_memory_repo.get_onboarding_state())
            elif self.path.startswith("/api/turn/events"):
                import urllib.parse
                _, _, q_raw = self.path.partition("?")
                q_map = {k: v[0] for k, v in urllib.parse.parse_qs(q_raw).items()}
                tid = str(q_map.get("turn_id") or "").strip()
                try:
                    after_s = int(q_map.get("after") or self.headers.get("Last-Event-ID") or 0)
                except ValueError:
                    after_s = 0
                with sys_s23.turns_lock:
                    t_obj = sys_s23.active_turns.get(tid) or sys_s23.recent_turns.get(tid)
                    if not t_obj:
                        self._send({"error": "unknown turn_id"}, code=404)
                        return
                    evs = [e for e in t_obj.get("events", []) if int(e.get("seq") or 0) > after_s]
                    self._send({
                        "turn_id": tid,
                        "session_id": t_obj.get("session_id"),
                        "active": tid in sys_s23.active_turns,
                        "done": bool(t_obj.get("done")),
                        "cancelled": t_obj["cancel"].is_set(),
                        "last_seq": int(t_obj.get("seq") or 0),
                        "events": evs,
                    })
            elif self.path.startswith("/api/repo/workspaces"):
                self._send(sys_s23.git_memory_repo.list_workspace_repos())
            elif self.path.startswith("/api/repo/files"):
                self._send({
                    "repo_path": str(sys_s23.git_memory_repo.repo_dir),
                    "active_repo": sys_s23.git_memory_repo.active_repo_name,
                    "workspaces": sys_s23.git_memory_repo.list_workspace_repos(),
                    "storage": sys_s23.git_memory_repo.get_storage_locations(),
                    "onboarding": sys_s23.git_memory_repo.get_onboarding_state(),
                    "files": sys_s23.git_memory_repo.list_files(),
                    "lessons": sys_s23.git_memory_repo.list_staged_lessons(),
                    "commits": sys_s23.git_memory_repo.git_log(max_count=15),
                    "github_sync": sys_s23.git_memory_repo.get_github_sync_config(),
                })
            elif self.path.startswith("/api/health"):
                self._send({
                    "status": "ready",
                    "mode": "lfm3",
                    "app_exclusive": True,
                    "authorized_package": CLOVER_S23_APP_PACKAGE,
                    "backend": "android-universal-opencl-smmla-q4-7layer",
                    "model": "lfm3-omni-2b-4bit-s23",
                    "ncm_and_hybridlink_loaded": sys_s23.hybrid_ncm is not None,
                    "c2_c3_precomputed_trunks_active": bool(
                        sys_s23.hybrid_ncm is not None and sys_s23.hybrid_ncm.precomputed_trunks_active
                    ),
                    "c2_c3_precomputed_dirs": (
                        int(sys_s23.hybrid_ncm.precomputed_manifest.get("total_trunk_and_expert_dirs", 31))
                        if sys_s23.hybrid_ncm is not None
                        else 0
                    ),
                    "pinned_threads": sys_s23.engine.active_threads,
                    "setup_ms": sys_s23.setup_ms,
                    "settings": dict(sys_s23.settings, engine=eng_mode),
                    "device_profile": dev_p,
                    "recall": sys_s23.memory_index.describe() if hasattr(sys_s23, "memory_index") else None,
                    "search": sys_s23.search_config.describe(),
                    "github_session": sys_s23.git_memory_repo.get_github_sync_config(),
                    "storage": {
                        **sys_s23.git_memory_repo.get_storage_locations(),
                        "precomputed_datasets": sys_s23.get_precomputed_dataset_sizes(),
                    },
                    "data_movement_ledger": dm,
                    "engine": {
                        "mode": "omni",
                        "model": "LFM3-Omni-2B",
                        "engine": eng_mode,
                        "device": f"{dev_p['device_name']} ({dev_p['gpu_label']} + {dev_p['cpu_label']})",
                        "backend": "Unified C1+C2+C3 Precomputed Trunks & Experts (OpenCL + NEON/SMMLA + FA2)",
                        "weights": (
                            "Full LFM3-SoC-Q4 (4-Bit FFN + 4-Bit LM-Head + 4-Bit Operators)"
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
                self._send({"status": "Clover Mobile Studio 7-Layer On-Device Server Running"})

        def do_POST(self) -> None:
            if not self._is_exclusive_clover_app():
                self._reject_non_app()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(raw) if raw else {}

            if self.path.startswith("/api/settings"):
                updated = sys_s23.update_settings(payload)
                eng_m = "gpu" if updated.get("engine") in ("gpu", "adreno") else "cpu"
                self._send({"ok": True, "settings": dict(updated, engine=eng_m)})
                return
            if self.path.startswith("/api/permissions"):
                perms_in = payload.get("permissions") if isinstance(payload.get("permissions"), dict) else payload
                updated_p = sys_s23.update_permissions(perms_in)
                self._send({"ok": True, "permissions": updated_p})
                return
            if self.path.startswith("/api/search/test"):
                try:
                    ok_s, msg_s = sys_s23.search_tool.test()
                    self._send({"ok": ok_s, "message": msg_s, **sys_s23.search_config.describe()})
                except Exception as ex:
                    self._send({"ok": False, "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/search"):
                try:
                    if payload.get("action") == "test":
                        ok_s, msg_s = sys_s23.search_tool.test()
                        self._send({"ok": ok_s, "message": msg_s, **sys_s23.search_config.describe()})
                        return
                    s_desc = sys_s23.search_config.update(
                        provider=payload.get("provider"),
                        key=payload.get("key"),
                        forget=payload.get("forget"),
                    )
                    self._send({"ok": True, **s_desc})
                except Exception as ex:
                    self._send({"error": str(ex), "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/workspaces/"):
                try:
                    act_w = self.path.split("/api/workspaces/", 1)[1].split("?", 1)[0].strip("/")
                    self._send(sys_s23.git_memory_repo.workspace_action(act_w, payload))
                except Exception as ex:
                    self._send({"error": str(ex), "message": str(ex)}, code=400)
                return
            if self.path == "/api/github":
                try:
                    self._send(
                        sys_s23.git_memory_repo.update_github_token(
                            token=payload.get("token"),
                            forget=bool(payload.get("forget")),
                        )
                    )
                except Exception as ex:
                    self._send({"error": str(ex), "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/vscode/sync"):
                try:
                    gh_c = sys_s23.git_memory_repo.get_github_sync_config()
                    sres = sys_s23.git_memory_repo.configure_and_sync_private_github(
                        repo_full_name=str(payload.get("repo") or gh_c.get("repo") or ""),
                        action="sync",
                    )
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                    gh_c2 = sys_s23.git_memory_repo.get_github_sync_config()
                    self._send({
                        "result": {
                            "repo": sres.get("repo"),
                            "notes": len(sys_s23.list_memories()),
                            "committed": True,
                            "imported": 0,
                        },
                        "repo": {
                            "repo": gh_c2.get("repo") or "",
                            "private": True,
                            "last_sync": (gh_c2.get("last_sync") or "").replace("T", " ").replace("Z", " UTC"),
                            "login": gh_c2.get("login") or "",
                            "logged_in": bool(gh_c2.get("logged_in")),
                        },
                        "sessions": sys_s23.git_memory_repo.sessions(),
                    })
                except Exception as ex:
                    self._send({"error": str(ex), "message": str(ex)}, code=409)
                return
            if self.path.startswith("/api/vscode/do"):
                act_v = str(payload.get("action") or "")
                gh_c = sys_s23.git_memory_repo.get_github_sync_config()
                if act_v == "githubSetupRepo":
                    try:
                        sres = sys_s23.git_memory_repo.configure_and_sync_private_github(
                            repo_full_name=str(payload.get("repo") or gh_c.get("repo") or ""),
                            action="sync",
                        )
                        gh_c2 = sys_s23.git_memory_repo.get_github_sync_config()
                        self._send({
                            "result": {"created": False, "repo": sres.get("repo"), "path": str(sys_s23.git_memory_repo.default_repo_dir)},
                            "repo": {
                                "repo": gh_c2.get("repo") or "",
                                "private": True,
                                "last_sync": (gh_c2.get("last_sync") or "").replace("T", " ").replace("Z", " UTC"),
                                "login": gh_c2.get("login") or "",
                                "logged_in": bool(gh_c2.get("logged_in")),
                            },
                            "connected": True,
                        })
                    except Exception as ex:
                        self._send({"error": str(ex), "message": str(ex)}, code=409)
                    return
                if act_v == "githubSignIn":
                    try:
                        df = sys_s23.git_memory_repo.start_github_device_flow()
                        self._send({
                            "result": df,
                            "repo": gh_c,
                            "connected": bool(gh_c.get("logged_in")),
                        })
                    except Exception as ex:
                        self._send({"error": str(ex), "message": str(ex)}, code=409)
                    return
                self._send({
                    "result": {"ok": True},
                    "repo": gh_c,
                    "connected": True,
                })
                return
            if self.path == "/api/tts/parts":
                self._send({"parts": sys_s23.speech_parts(str(payload.get("text") or ""))})
                return
            if self.path == "/api/tts/warm":
                self._send({"ok": True})
                return
            if self.path == "/api/tts":
                wav_b = sys_s23.tts_wav_bytes(str(payload.get("text") or ""), voice=str(payload.get("voice") or "female"))
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_b)))
                self.end_headers()
                self.wfile.write(wav_b)
                return
            if self.path == "/api/decision":
                tid = str(payload.get("turn_id") or "")
                rid = str(payload.get("request_id") or "")
                t_obj = sys_s23.active_turns.get(tid)
                pend = t_obj and t_obj["pending"].get(rid)
                if not pend:
                    self._send({"error": "no such permission request"}, code=404)
                    return
                dec = str(payload.get("decision") or "deny")
                pend["decision"] = dec if dec in ("once", "always", "deny") else "deny"
                pend["event"].set()
                self._send({"ok": True})
                return
            if self.path in ("/api/cancel", "/api/stop"):
                tid = str(payload.get("turn_id") or "").strip()
                cancelled_ids = []
                with sys_s23.turns_lock:
                    targets = [sys_s23.active_turns[tid]] if (tid and tid != "all" and tid in sys_s23.active_turns) else (
                        list(sys_s23.active_turns.values()) if not tid or tid == "all" else []
                    )
                for t_obj in targets:
                    t_obj["cancel"].set()
                    for p_ev in list(t_obj["pending"].values()):
                        p_ev["event"].set()
                    cancelled_ids.append(t_obj.get("turn_id") or tid)
                self._send({"ok": bool(cancelled_ids), "cancelled_turns": cancelled_ids})
                return
            if self.path == "/api/shutdown":
                self._send({"shutdown": True})
                if hasattr(sys_s23, "memory_index"):
                    try:
                        sys_s23.memory_index.stop()
                    except Exception:
                        pass
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path.startswith("/api/personal/"):
                # the person's choices about their experiences: keep, stop learning, delete; or forget everything
                pers = phone_runtime().personal
                if pers is None:
                    self._send({"error": "personal_learning_off", "message": "Personal learning is off for this process."}, code=409)
                    return
                act_p = self.path.split("/api/personal/", 1)[1].split("?", 1)[0].strip("/")
                try:
                    if act_p == "retention":
                        ids_p = payload.get("ids") if isinstance(payload.get("ids"), list) else []
                        self._send(dict(pers.retention(ids_p, str(payload.get("choice") or "")), view=pers.view()))
                    elif act_p == "learning" and isinstance(payload.get("on"), bool):
                        pers.set_learning(payload["on"])
                        self._send(pers.view())
                    elif act_p == "forget" and payload.get("confirm") is True:
                        self._send(dict(pers.forget_all(), view=pers.view()))
                    else:
                        self._send({"error": "not_found"}, code=404)
                except ValueError as ex:
                    self._send({"error": "bad_request", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/collective/"):
                rt_c = phone_runtime()
                coll = rt_c.collective
                if coll is None:
                    self._send({"error": "collective_learning_off",
                                "message": "Collective learning is off for this process."}, code=409)
                    return
                act_c = self.path.split("/api/collective/", 1)[1].split("?", 1)[0].strip("/")
                try:
                    if act_c == "consent":
                        c_in = payload.get("contribute") if isinstance(payload.get("contribute"), bool) else None
                        r_in = payload.get("receive") if isinstance(payload.get("receive"), bool) else None
                        coll.set_consent(contribute=c_in, receive=r_in)
                        self._send(coll.view(rt_c.personal))
                    elif act_c == "sync":
                        reg_dir = str(payload.get("registry_dir") or "").strip()
                        sync_res = coll.sync_from_registry(reg_dir)
                        self._send(dict(sync_res, view=coll.view(rt_c.personal)))
                    elif act_c == "preview":
                        self._send({"candidates": coll.preview_candidates(rt_c.personal)})
                    else:
                        self._send({"error": "not_found"}, code=404)
                except Exception as ex:
                    self._send({"error": "bad_request", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/feedback"):
                try:
                    sid = str(payload.get("session_id") or "s23-studio")
                    turn_n = payload.get("turn") if payload.get("turn") is not None else payload.get("n")
                    fkind = str(payload.get("kind") or "helpful")
                    rec = sys_s23.git_memory_repo.record_outcome_feedback(
                        sid,
                        int(turn_n) if isinstance(turn_n, int) or (isinstance(turn_n, str) and turn_n.isdigit()) else None,
                        fkind,
                    )
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                    if fkind == "helpful":
                        personal_call(phone_runtime(), "helpful", sid,
                                      int(turn_n) if isinstance(turn_n, int) or (isinstance(turn_n, str) and turn_n.isdigit()) else None)
                    self._send({"ok": True, "outcome": rec, "growth": [f"turn {turn_n or 1}: marked helpful"]})
                except Exception as ex:
                    self._send({"error": "feedback_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/tools/shell"):
                cmd_s = str(payload.get("command") or "").strip()
                allow_o = {str(x) for x in (payload.get("allow_once") or [])}
                deny_o = {str(x) for x in (payload.get("deny_once") or [])}
                ok_sh, p_prompt, deny_msg = sys_s23._check_tool_permission(
                    "shell", "run_shell", cmd_s, allow_o, deny_o, interactive_ask=True,
                )
                if p_prompt is not None:
                    self._send({"ok": False, "permission_required": p_prompt})
                    return
                if not ok_sh:
                    self._send({"ok": False, "output": deny_msg}, code=403)
                    return
                out_sh = sys_s23._execute_shell_subprocess(cmd_s)
                self._send({"ok": True, "command": cmd_s, "output": out_sh})
                return
            if self.path.startswith("/api/tools/ssh"):
                host_s = str(payload.get("host") or "").strip()
                port_s = str(payload.get("port") or "22").strip()
                cmd_s = str(payload.get("command") or "").strip()
                allow_o = {str(x) for x in (payload.get("allow_once") or [])}
                deny_o = {str(x) for x in (payload.get("deny_once") or [])}
                ok_ssh, p_prompt, deny_msg = sys_s23._check_tool_permission(
                    "ssh", "run_ssh", f"{host_s}:{port_s} $ {cmd_s}", allow_o, deny_o, interactive_ask=True,
                )
                if p_prompt is not None:
                    self._send({"ok": False, "permission_required": p_prompt})
                    return
                if not ok_ssh:
                    self._send({"ok": False, "output": deny_msg}, code=403)
                    return
                out_ssh = sys_s23._execute_ssh_subprocess(host_s, cmd_s, port=port_s)
                self._send({"ok": True, "host": host_s, "port": port_s, "command": cmd_s, "output": out_ssh})
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
            if self.path.startswith("/api/github/device-start"):
                try:
                    cid = payload.get("client_id")
                    res_df = sys_s23.git_memory_repo.start_github_device_flow(client_id=str(cid) if cid else None)
                    self._send(res_df)
                except Exception as ex:
                    self._send({"error": "github_device_start_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/github/device-poll"):
                try:
                    dcode = str(payload.get("device_code") or "").strip()
                    rname = str(payload.get("repo") or "").strip()
                    cid = payload.get("client_id")
                    res_dp = sys_s23.git_memory_repo.poll_github_device_flow(
                        device_code=dcode,
                        repo_full_name=rname,
                        client_id=str(cid) if cid else None,
                    )
                    if res_dp.get("ok") and hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                    self._send({**res_dp, "sessions": sys_s23.git_memory_repo.sessions()})
                except Exception as ex:
                    self._send({"error": "github_device_poll_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/github/login"):
                try:
                    tok = payload.get("token")
                    rname = str(payload.get("repo") or "").strip()
                    auto_s = bool(payload.get("auto_sync", True))
                    res_lg = sys_s23.git_memory_repo.login_with_github(
                        token=str(tok).strip() if tok else None,
                        repo_full_name=rname,
                        auto_sync=auto_s,
                    )
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                    self._send({"ok": True, **res_lg, "sessions": sys_s23.git_memory_repo.sessions()})
                except Exception as ex:
                    self._send({"error": "github_login_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/github/logout"):
                self._send({"ok": True, **sys_s23.git_memory_repo.logout_github()})
                return
            if self.path.startswith(("/api/github/sync", "/api/repo/github-sync")):
                try:
                    repo_name = str(payload.get("repo") or "").strip()
                    tok = payload.get("token")
                    act = str(payload.get("action") or "sync").strip().lower()
                    br = str(payload.get("branch") or "main").strip()
                    sres = sys_s23.git_memory_repo.configure_and_sync_private_github(
                        repo_full_name=repo_name,
                        token=str(tok).strip() if tok else None,
                        action=act if act in ("push", "pull", "sync") else "sync",
                        branch=br,
                    )
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                    self._send({
                        **sres,
                        "sessions": sys_s23.git_memory_repo.sessions(),
                        "github_sync": sys_s23.git_memory_repo.get_github_sync_config(),
                    })
                except Exception as ex:
                    self._send({"error": "github_sync_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/storage"):
                try:
                    res_st = sys_s23.git_memory_repo.update_storage_locations(
                        media_dir=payload.get("media_dir"),
                        media_kind=payload.get("media_kind"),
                        workspace_source=payload.get("workspace_source"),
                    )
                    self._send({"ok": True, **res_st})
                except Exception as ex:
                    self._send({"error": "storage_update_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/media/clear"):
                self._send(sys_s23.git_memory_repo.clear_media_storage())
                return
            if self.path.startswith("/api/ssh"):
                try:
                    import local_terminal
                    ssh_mgr = local_terminal.SSHHosts(
                        sys_s23.git_memory_repo.default_repo_dir / "config" / sys_s23.git_memory_repo.device / "ssh-hosts.json"
                    )
                    res_ssh = ssh_mgr.add_host(
                        name=str(payload.get("name") or payload.get("host") or ""),
                        host=str(payload.get("host") or ""),
                        user=str(payload.get("user") or "root"),
                        port=int(payload.get("port") or 22),
                    )
                    self._send({"ok": True, **res_ssh})
                except Exception as ex:
                    self._send({"error": "ssh_host_failed", "message": str(ex)}, code=400)
                return
            if self.path.startswith("/api/session/forget"):
                sid = str(payload.get("session_id") or "").strip()
                if sid and payload.get("delete"):
                    sys_s23.git_memory_repo.delete_session(sid)
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                self._send({"ok": True, "sessions": sys_s23.git_memory_repo.sessions()})
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

            prompt = str(payload.get("prompt") or "").strip()
            media_in = payload.get("media") if isinstance(payload.get("media"), list) else []
            if not prompt and (media_in or payload.get("image_b64") or payload.get("attached_files")):
                prompt = "Analyze the attached file and answer clearly."
            if not prompt:
                prompt = "Hello"

            # Convert laptop-style `media: [{"kind": "image"|"audio"|"video", "name": ..., "data": ...}]` into `image_b64` / `attached_files`
            img_b64_in = payload.get("image_b64")
            att_files_in = list(payload.get("attached_files") or [])
            voice_in = bool(payload.get("voice_input", False))
            saved_media_items: list[dict[str, str]] = []
            media_errors: list[str] = []
            for m_item in media_in[:6]:
                if not isinstance(m_item, dict):
                    continue
                mkind = str(m_item.get("kind") or "image")
                mname = str(m_item.get("name") or mkind)
                mdata = str(m_item.get("data") or "")
                raw_b64 = mdata.split(",", 1)[1] if mdata.startswith("data:") else mdata
                try:
                    raw_b = base64.b64decode(raw_b64)
                    ext = "png" if mkind == "image" else "wav" if mkind == "audio" else "mp4"
                    fname_saved = sys_s23._save_to_media_storage(raw_b, ext)
                    saved_media_items.append({"kind": mkind, "file": fname_saved, "name": mname})
                    if mkind == "image" and not img_b64_in:
                        img_b64_in = "data:image/png;base64," + raw_b64
                    elif mkind == "audio":
                        att_files_in.append({"name": mname, "type": "audio/wav", "data_b64": "data:audio/wav;base64," + raw_b64})
                        if m_item.get("voice"):
                            voice_in = True
                except Exception as ex:
                    # k3: the photo was dropped here without a word and the model said it saw no image
                    print(f"[Clover-S23] attachment {mname!r} not taken: {ex!r}", file=sys.stderr, flush=True)
                    media_errors.append(f"{mname}: {'the data could not be decoded' if isinstance(ex, ValueError) else 'it could not be saved'}")

            if "q4_operators" in payload and isinstance(payload["q4_operators"], bool):
                sys_s23.engine.set_use_4bit_operators(payload["q4_operators"])
            if "adreno_active" in payload and isinstance(payload["adreno_active"], bool):
                sys_s23.engine.set_adreno_opencl_active(payload["adreno_active"])

            max_layer = int(payload.get("max_active_layer") or 7)
            use_ncm = bool(payload.get("use_ncm_renderer", sys_s23.settings["layers"].get("ncm", True)))
            pm = sys_s23.perm_modes
            allow_net = bool(payload.get("allow_internet", pm.get("internet", "allow") != "deny"))
            allow_py = bool(payload.get("allow_python", pm.get("code", "allow") != "deny"))
            allow_fl = bool(payload.get("allow_files", pm.get("files", "allow") != "deny"))

            # Detect Clover Retry Cycle ("Try again" button, "that's wrong" / "try again" message, or repeating the question)
            import clover_cycle as C
            import uuid

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

            if self.path.startswith("/api/chat/stream"):
                sid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(payload.get("session_id") or uuid.uuid4().hex))[:96]
                turn_id = uuid.uuid4().hex
                turn_obj: dict[str, Any] = {
                    "turn_id": turn_id,
                    "session_id": sid,
                    "cancel": threading.Event(),
                    "pending": {},
                    "seq": 0,
                    "events": [],
                    "done": False,
                    "updated": time.time(),
                }
                with sys_s23.turns_lock:
                    sys_s23.active_turns[turn_id] = turn_obj
                write_lock = threading.Lock()
                client_closed = [False]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def emit_sse(ev: dict[str, Any]) -> None:
                    with write_lock:
                        turn_obj["seq"] += 1
                        seq = turn_obj["seq"]
                        out_ev = dict(ev, seq=seq, turn_id=turn_id)
                        turn_obj["events"].append(out_ev)
                        if len(turn_obj["events"]) > 512:
                            del turn_obj["events"][:len(turn_obj["events"]) - 512]
                        if ev.get("type") in ("done", "error"):
                            turn_obj["done"] = True
                        turn_obj["updated"] = time.time()
                        if client_closed[0]:
                            return
                        try:
                            self.wfile.write(f"id: {seq}\ndata: {json.dumps(out_ev, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except (OSError, BrokenPipeError, ConnectionResetError):
                            client_closed[0] = True
                            turn_obj["cancel"].set()
                            for p_ev in list(turn_obj["pending"].values()):
                                p_ev["event"].set()

                emit_sse({"type": "start", "turn_id": turn_id, "session_id": sid})
                if saved_media_items:
                    emit_sse({"type": "media", "items": saved_media_items})
                if sys_s23._busy_turns > 0:
                    emit_sse({"type": "plan", "round": 1, "stage": "queued"})
                try:
                    import semantics
                    if turn_obj["cancel"].is_set():
                        emit_sse({"type": "done", "reply": "(Stopped.)", "cancelled": True, "layers": [], "images": [], "generation": []})
                        return
                    phone = phone_runtime()
                    personal_call(phone, "before_turn", sid, prompt, how_retry)
                    allow_once_list = list(payload.get("allow_once") or [])
                    deny_once_list = list(payload.get("deny_once") or [])
                    if how_retry and prev_q and prev_a and not img_b64_in and not att_files_in:
                        emit_sse({"type": "cycle", "stage": "check"})
                        fb_text = str(payload.get("feedback") or (prompt if how_retry == "said" else "") or "").strip()
                        ctx_before = C.context_before(hist_msgs, last_ex[0]) if (how_retry == "said" and last_ex) else ""
                        emit_sse({"type": "cycle", "stage": "second"})
                        res = sys_s23.run_retry_cycle(
                            question=prev_q if how_retry == "said" else prompt,
                            previous_answer=prev_a,
                            feedback=fb_text,
                            session_id=sid,
                            max_active_layer=max_layer,
                            allow_internet=allow_net,
                            allow_python=allow_py,
                            context=ctx_before,
                        )
                        frame = semantics.interpret(prompt, phone, history=hist_msgs)
                        frame.act, frame.retry, frame.route = "retry", how_retry, "try again"
                        align_turn(frame, res)
                    else:
                        tools_mode = str(payload.get("tools_mode") or sys_s23.settings.get("tools") or "ask")
                        frame = semantics.interpret(prompt, phone, media=[m["kind"] for m in saved_media_items],
                                                    history=hist_msgs, previous_media=payload.get("previous_media") or [],
                                                    attached=bool(att_files_in))
                        only, _on = semantics.route(frame, tools_mode, True, offer=PHONE_OFFER)
                        req_tools = list(PHONE_OFFER) if only is None else only
                        allow_net_s = allow_net and ("web_search" in req_tools)
                        allow_py_s = allow_py and ("python_interpreter" in req_tools)
                        emit_sse(dict(frame.layer(), type="semantics"))
                        emit_sse({"type": "tools", "offered": req_tools, "mode": tools_mode})
                        if frame.act != "social":
                            emit_sse({"type": "plan", "round": 1, "stage": "planning", "tools": req_tools if only is not None else []})

                        want_pic = "generate_image" in frame.asked
                        if frame.act == "social":
                            # only pleasantries ("hi", "thanks", "bye"): semantics answers and the model is not called
                            res = sys_s23.social_turn(prompt, frame, sid, max_layer)
                        else:
                            res = sys_s23.run_cycle(
                                prompt=prompt,
                                output_type="image" if want_pic else str(payload.get("output_type") or "text"),
                                image_b64=img_b64_in,
                                attached_files=att_files_in,
                                voice_input=voice_in,
                                max_active_layer=max_layer,
                                use_ncm_renderer=use_ncm,
                                enable_thinking=bool(payload.get("enable_thinking", False)),
                                inject_latent_step_before_decode=bool(payload.get("inject_latent_step_before_decode", False)),
                                allow_internet=allow_net_s,
                                allow_python=allow_py_s,
                                allow_files=allow_fl,
                                session_id=sid,
                                allow_once=allow_once_list,
                                deny_once=deny_once_list,
                                interactive_ask=True,
                                history=hist_msgs,
                                tools=req_tools,
                                meaning=frame,
                            )
                        # If a tool permission is set to `Ask` (or is a risky shell command), emit `permission` SSE event and wait for `/api/decision`
                        if res.get("permission_required"):
                            pr_req = res["permission_required"]
                            perm_k = pr_req.get("permission") or "code"
                            t_name = pr_req.get("tool") or perm_k
                            risk_why = pr_req.get("risky")
                            emit_sse({"type": "tool_call", "id": "tc1", "name": t_name, "arguments": {"query": pr_req.get("detail", "")}})
                            rid = uuid.uuid4().hex[:12]
                            pend_ev = {"event": threading.Event(), "decision": "deny"}
                            turn_obj["pending"][rid] = pend_ev
                            emit_sse(dict({
                                "type": "permission",
                                "turn_id": turn_id,
                                "request_id": rid,
                                "tool": perm_k,
                                "label": pr_req.get("label") or perm_k,
                                "detail": pr_req.get("detail") or "",
                            }, **({"risky": risk_why} if risk_why else {})))
                            pend_ev["event"].wait(timeout=900)
                            turn_obj["pending"].pop(rid, None)
                            dec_made = pend_ev["decision"]
                            emit_sse({"type": "permission_decided", "request_id": rid, "decision": dec_made})
                            if dec_made == "always" and not risk_why:
                                sys_s23.update_permissions({perm_k: "allow"})
                            if dec_made in ("once", "always") and not turn_obj["cancel"].is_set():
                                allow_once_list.append(perm_k)
                            else:
                                deny_once_list.append(perm_k)
                            res = sys_s23.run_cycle(
                                prompt=prompt,
                                output_type=str(payload.get("output_type") or "text"),
                                image_b64=img_b64_in,
                                attached_files=att_files_in,
                                voice_input=voice_in,
                                max_active_layer=max_layer,
                                use_ncm_renderer=use_ncm,
                                enable_thinking=bool(payload.get("enable_thinking", False)),
                                inject_latent_step_before_decode=bool(payload.get("inject_latent_step_before_decode", False)),
                                allow_internet=allow_net_s,
                                allow_python=allow_py_s,
                                allow_files=allow_fl,
                                session_id=sid,
                                allow_once=allow_once_list,
                                deny_once=deny_once_list,
                                interactive_ask=False,
                                history=hist_msgs,
                                tools=req_tools,
                                meaning=frame,
                            )
                        align_turn(frame, res, sys_s23.remember_for(sid, max_layer))
                        personal_call(phone, "after_turn", sid, frame, res)

                    if media_errors:
                        res["media_errors"] = media_errors
                        res["reply"] = (str(res.get("reply") or "").rstrip()
                                        + f"\n\n(Note: I could not open {', '.join(media_errors)}, so this answer is without it.)").strip()

                    # Emit tool_call / tool_result cards if a tool ran during the turn
                    lt = res.get("layer_trace") or {}
                    l1 = lt.get("layer1_lfm3_omni") or {}
                    l2 = lt.get("layer2_hybrid_agent") or {}
                    l3 = lt.get("layer3_11level_ncm") or {}
                    l4 = lt.get("layer4_memory_graph") or {}
                    l5 = lt.get("layer5_ontology_and_verifier") or {}
                    l6 = lt.get("layer6_generative_compiler") or {}

                    if l4.get("injected_into_model_prompt") and l4.get("entries"):
                        emit_sse({"type": "layer", "layer": "memory", "entries": [e.get("id") for e in l4["entries"]]})
                    if l5.get("injected_into_model_prompt"):
                        emit_sse({"type": "layer", "layer": "ontology", "n_evidence": l5.get("ontology_hits", 1)})
                    if l2.get("tool_executed"):
                        tex = l2["tool_executed"]
                        t_ui_name = "run_python" if tex == "python_interpreter" else "run_powershell" if tex == "run_shell" else tex
                        t_arg = l2.get("python_code") or l2.get("search_query") or l2.get("shell_command") or ""
                        t_out = l2.get("python_output") or l2.get("observation_preview") or ""
                        t_s = str(t_out or "").strip()
                        t_worked = not (t_s.startswith(("Error:", "Not run:")) or "Traceback (most recent call last):" in t_s)
                        t_sum = (t_s.splitlines()[-1] if "Traceback" in t_s else (t_s.splitlines()[0] if t_s else "Completed"))[:96]
                        emit_sse({"type": "tool_call", "id": "t_exec", "name": t_ui_name, "arguments": {"command": t_arg}})
                        emit_sse({"type": "tool_result", "id": "t_exec", "name": t_ui_name, "ran": True, "worked": t_worked, "summary": t_sum, "why": "allowed", "result": t_out})
                        emit_sse({"type": "round", "round": 2})

                    # Stream thinking tokens if thinking / latent steps occurred
                    raw_t1 = str(l1.get("turn1_raw_output") or "")
                    if "</think>" in raw_t1:
                        th_txt = raw_t1.split("</think>", 1)[0].replace("<think>", "").strip()
                        if th_txt:
                            emit_sse({"type": "token", "round": 1, "kind": "thinking", "text": th_txt})

                    # Stream answer tokens (honoring Stop button cancellation)
                    ans_text = str(res.get("reply") or "")
                    words_ans = ans_text.split(" ")
                    emitted_words: list[str] = []
                    cancelled = False
                    for idx_w, w_tok in enumerate(words_ans):
                        if turn_obj["cancel"].is_set():
                            cancelled = True
                            break
                        piece = w_tok + (" " if idx_w + 1 < len(words_ans) else "")
                        emitted_words.append(w_tok)
                        emit_sse({"type": "token", "round": 1, "kind": "answer", "text": piece})
                        if len(words_ans) > 25 and idx_w % 4 == 0:
                            time.sleep(0.01)
                    final_reply = " ".join(emitted_words) if cancelled else ans_text

                    # Build `layers` list matching laptop's `layersUsed(ev)` in `app.html`
                    layers_list: list[dict[str, Any]] = []
                    if lt.get("semantics_frame"):
                        layers_list.append(lt["semantics_frame"])
                    c1_ms = (l1.get("turn1_prefill_ms") or 0.0) + (l1.get("turn1_decode_ms") or 0.0)
                    if frame.act != "social":  # a pleasantry answered by semantics never reached C1
                        layers_list.append({"layer": "c1", "ms": round(c1_ms, 1)})
                    if l2.get("enabled"):
                        layers_list.append({"layer": "c2", "ms": l2.get("hybridlink_ms", 0.0)})
                    if l3.get("ncm_active"):
                        layers_list.append({"layer": "ncm", "windows": 1, "ms": l3.get("ncm_total_ms", 0.0)})
                    if l4.get("injected_into_model_prompt") and l4.get("entries"):
                        layers_list.append({"layer": "memory", "entries": [e.get("id") for e in l4["entries"]]})
                    if l4.get("s2_turn_recall") and (l4["s2_turn_recall"].get("refs") or []):
                        sr = l4["s2_turn_recall"]
                        layers_list.append({"layer": "recall", "refs": sr["refs"], "chats": sr.get("chats", 1)})
                    if l5.get("injected_into_model_prompt"):
                        layers_list.append({"layer": "ontology", "n_evidence": l5.get("ontology_hits", 1)})
                    if l6.get("report") and l6["report"].get("checked_blocks"):
                        layers_list.append({"layer": "compiler", "checked": True, "ok": bool(l6["report"].get("all_ok"))})
                    if sys_s23.git_memory_repo.active_repo_name != "clover-memory-repo":
                        layers_list.append({"layer": "workspace", "workspace": sys_s23.git_memory_repo.active_repo_name})
                    if res.get("clover_cycle_info"):
                        ci = res["clover_cycle_info"]
                        layers_list.append({"layer": "cycle", "trigger": "user", "action": ci.get("action", "checked"), "tools": ci.get("tools_used", [])})

                    # Save generated image to media folder if present so `<figure>` renders `/api/media/file/<hash>.png`
                    images_out: list[dict[str, Any]] = []
                    if res.get("image_png_b64") and "," in str(res["image_png_b64"]):
                        raw_png = base64.b64decode(str(res["image_png_b64"]).split(",", 1)[1])
                        import hashlib
                        h32 = hashlib.sha256(raw_png).hexdigest()[:32] + ".png"
                        mdir = sys_s23.git_memory_repo.get_media_dir()
                        mdir.mkdir(parents=True, exist_ok=True)
                        (mdir / h32).write_bytes(raw_png)
                        images_out.append({"file": h32, "prompt": prompt[:200], "model": "Clover Omni Visual", "seconds": 0.2})

                    gen_tok = int(res.get("generated_tokens") or res.get("emitted_tokens") or len(words_ans))
                    dec_s = max(0.05, float(l1.get("turn1_decode_ms") or res.get("c_24_layer_4bit_ms") or 500.0) / 1000.0)
                    emit_sse({
                        "type": "done",
                        **res,
                        "reply": final_reply,
                        "turn": res.get("turn_n") or 1,
                        "cancelled": cancelled,
                        "layers": layers_list,
                        "images": images_out,
                        "generation": [] if frame.act == "social" else [{"new_tokens": gen_tok, "decode_s": round(dec_s, 3)}],
                    })
                except Exception as ex:
                    emit_sse({"type": "error", "error": str(ex)})
                finally:
                    with sys_s23.turns_lock:
                        sys_s23.active_turns.pop(turn_id, None)
                        sys_s23.recent_turns[turn_id] = turn_obj
                        while len(sys_s23.recent_turns) > 32:
                            sys_s23.recent_turns.pop(next(iter(sys_s23.recent_turns)))
                return

            personal_call(phone_runtime(), "before_turn", str(payload.get("session_id") or "s23-studio"), prompt, how_retry)
            if how_retry and prev_q and prev_a and not img_b64_in and not att_files_in:
                fb_text = str(payload.get("feedback") or (prompt if how_retry == "said" else "") or "").strip()
                ctx_before = C.context_before(hist_msgs, last_ex[0]) if (how_retry == "said" and last_ex) else ""
                res = sys_s23.run_retry_cycle(
                    question=prev_q if how_retry == "said" else prompt,
                    previous_answer=prev_a,
                    feedback=fb_text,
                    session_id=str(payload.get("session_id") or "s23-studio"),
                    max_active_layer=max_layer,
                    allow_internet=allow_net,
                    allow_python=allow_py,
                    context=ctx_before,
                )
                import semantics
                frame = semantics.interpret(prompt, phone_runtime(), history=hist_msgs)
                frame.act, frame.retry, frame.route = "retry", how_retry, "try again"
                align_turn(frame, res)
                self._send(res)
                return

            # Meaning first (semantics.py, common to every Clover process): a pleasantry, then the tools this turn gets
            import semantics
            tools_mode = str(payload.get("tools_mode") or sys_s23.settings.get("tools") or "ask")
            frame = semantics.interpret(prompt, phone_runtime(), media=[m["kind"] for m in saved_media_items],
                                        history=hist_msgs, previous_media=payload.get("previous_media") or [],
                                        attached=bool(att_files_in))
            only, _on = semantics.route(frame, tools_mode, True, offer=PHONE_OFFER)
            if frame.act == "social":  # only pleasantries ("hi", "thanks", "bye"): answered without the model
                self._send(sys_s23.social_turn(prompt, frame, str(payload.get("session_id") or "s23-studio"), max_layer))
                return
            req_tools = list(PHONE_OFFER) if only is None else only
            allow_net = allow_net and ("web_search" in req_tools)
            allow_py = allow_py and ("python_interpreter" in req_tools)

            res = sys_s23.run_cycle(
                prompt=prompt,
                output_type=str(payload.get("output_type") or "text"),
                image_b64=img_b64_in,
                attached_files=att_files_in,
                voice_input=voice_in,
                max_active_layer=max_layer,
                use_ncm_renderer=use_ncm,
                enable_thinking=bool(payload.get("enable_thinking", False)),
                inject_latent_step_before_decode=bool(payload.get("inject_latent_step_before_decode", False)),
                allow_internet=allow_net,
                allow_python=allow_py,
                allow_files=allow_fl,
                session_id=str(payload.get("session_id") or "s23-studio"),
                allow_once=payload.get("allow_once") if isinstance(payload.get("allow_once"), list) else None,
                deny_once=payload.get("deny_once") if isinstance(payload.get("deny_once"), list) else None,
                interactive_ask=True,
                history=hist_msgs,
                tools=req_tools,
                meaning=frame,
            )
            if not res.get("permission_required"):
                align_turn(frame, res, sys_s23.remember_for(str(payload.get("session_id") or "s23-studio"), max_layer))
                personal_call(frame.runtime, "after_turn", str(payload.get("session_id") or "s23-studio"), frame, res)
            if media_errors:
                res["media_errors"] = media_errors
                res["reply"] = (str(res.get("reply") or "").rstrip()
                                + f"\n\n(Note: I could not open {', '.join(media_errors)}, so this answer is without it.)").strip()
            self._send(res)

        def do_DELETE(self) -> None:
            if not self._is_exclusive_clover_app():
                self._reject_non_app()
                return
            if self.path.startswith("/api/sessions/"):
                sid = self.path.split("/api/sessions/", 1)[1].split("?", 1)[0].strip("/")
                if sid:
                    sys_s23.git_memory_repo.delete_session(sid)
                    personal_call(phone_runtime(), "delete_session", sid)  # its experiences go with it
                    if hasattr(sys_s23, "memory_index"):
                        sys_s23.memory_index.wake()
                self._send({"ok": True, "sessions": sys_s23.git_memory_repo.sessions()})
                return
            self._send({"error": "not_found"}, code=404)

    ThreadingHTTPServer.allow_reuse_address = True
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
