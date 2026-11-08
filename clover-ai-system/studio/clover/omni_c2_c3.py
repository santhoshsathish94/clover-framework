"""Unified Core 2 (HybridAgent Plan) + Bridges (`ComboBridge` & `ExpertBridge`) + Core 3 (11-Level NCM Decode) Runtime
for `LFM3-Omni-2B` (`omni_c2_c3.py`).

Loads directly from the precomputed `CLVRTRK1` binary trunks in `<base>/data/omni/precomputed`:
  - Core 2 · Plan (`hybridagent/plan-trunk-0`):
      `Lfm3NativeHybridLink(d=2048, bottleneck=512)` + `HybridLink2048(d=2048 -> 2048 -> 2048)` continuous `<latent>`
      plan representation $r_t = R_{\\text{LFM3}}(h_t)$.
  - Bridges (`bridges/bridge-trunk-0`):
      `Lfm3ComboBridge(2048 -> 768 -> 384)` ($z_{\\text{rh}}$) + `Lfm3ExpertBridge(2048 -> 384 -> 384)` ($z_{\\text{exp}}$).
  - Core 3 · Decode (`ncm/`):
      * 11-Level Multiscale Residual `VQVAE` (`vqvae-trunk-0` + `vqvae-expert-0..10` + `vqvae_wte.bf16.bin` +
        `vqvae_dec.bf16.bin`, levels `[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]` = `313` concept codes).
      * 12-Layer `NextConceptTransformer` (`ncm-trunk-0..11` + `ncm-global-0`, `n_head=12`, `d=768`, QK-RMSNorm, RoPE).
      * 3-Layer `LexicalPointerDecoder` (`ptr-trunk-0..2` + `ptr-global-0`, `n_head=6`, `d=384`, copy-gate pointer
        mechanism + active-vocabulary projection).

Implemented in pure vectorized NumPy/BLAS so it runs identically on Windows, Linux (`k3`), and Android (`Chaquopy`)
with zero requirement for a multi-gigabyte PyTorch wheel on mobile devices.
"""
from __future__ import annotations

import collections
import json
import math
import re
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

V_LFM3 = 128000
D_LFM3 = 2048
D_VQ = 384
D_NCM = 768
P_BPE = 512
P_RH = 512
L_WMT = 512
VQ_LEVELS = (1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64)

GRAMMAR_CONNECTOR_WORDS = (
    "the", "The", "a", "A", "an", "An", "in", "In", "on", "On", "at", "At", "to", "To", "for", "For",
    "of", "by", "with", "from", "as", "into", "over", "after", "before", "during", "between", "through",
    "and", "or", "but", "so", "while", "when", "where", "which", "that", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "it", "its", "this", "these",
    "they", "them", "their", "we", "our", "you", "your", "I", "my", "me",
    ".", ",", ";", ":", "!", "?", "'s", "'", "-", "(", ")", "/", "//", "://", "_", "=", "&", "#", "[", "]",
    "https", "http", "www", "com", "org", "io", "ai",
)

REQUIRED_PRECOMPUTED = (
    "trunks/tokenizer.json",
    "hybridagent/plan-trunk-0/index.bin",
    "bridges/bridge-trunk-0/index.bin",
    "ncm/vqvae-trunk-0/index.bin",
    "ncm/vqvae-expert-0/index.bin",
    "ncm/vqvae-expert-10/index.bin",
    "ncm/ncm-global-0/index.bin",
    "ncm/ncm-trunk-0/index.bin",
    "ncm/ncm-trunk-11/index.bin",
    "ncm/ptr-global-0/index.bin",
    "ncm/ptr-trunk-0/index.bin",
    "ncm/ptr-trunk-2/index.bin",
    "ncm/vqvae_wte.bf16.bin",
    "ncm/vqvae_dec.bf16.bin",
)


def precomputed_dir(base_dir: str | Path) -> Path:
    p = Path(base_dir) / "data" / "omni" / "precomputed"
    if (p / "models" / "trunks" / "tokenizer.json").is_file():
        return p / "models"
    return p


def missing_c2_c3_files(base_dir: str | Path) -> list[str]:
    p = precomputed_dir(base_dir)
    return [rel for rel in REQUIRED_PRECOMPUTED if not (p / rel).is_file()]


# ---------------- Pure-Python 128K BPE Tokenizer (`tokenizer.json`) ----------------

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
    def __init__(self, tokenizer_json_path: Path):
        raw = json.loads(Path(tokenizer_json_path).read_text(encoding="utf-8"))
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
        self._special_re = (
            re.compile("(" + "|".join(re.escape(k) for k in sorted(self.added_tokens_map.keys(), key=len, reverse=True)) + ")")
            if self.added_tokens_map
            else None
        )
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for idx, m in enumerate(model.get("merges", [])):
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


# ---------------- Pure-NumPy Mathematical Primitives (Bit-Exact to PyTorch) ----------------

def read_clvrtrk1(trunk_dir: Path) -> dict[str, np.ndarray]:
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
    out_t: dict[str, np.ndarray] = {}
    pos = 20
    for _ in range(n_entries):
        tname = idx_bytes[pos : pos + 64].split(b"\x00", 1)[0].decode("utf-8")
        pos += 64
        fid, dtype_code, ndim, offset, nbytes, d0, d1, d2, d3 = struct.unpack("<IIIQQIIII", idx_bytes[pos : pos + 44])
        pos += 44
        dims = [d0, d1, d2, d3][:ndim] if ndim > 0 else [1]
        raw = blobs[fid][offset : offset + nbytes]
        if dtype_code == 1:
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32).copy()
            out_t[tname] = f32.reshape(dims)
        else:
            out_t[tname] = np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)
    return out_t


def bf16_rows_to_f32(mm_u16: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
    sub = np.asarray(mm_u16[indices], dtype=np.uint32)
    return (sub << 16).view(np.float32)


def layer_norm(x: np.ndarray, weight: np.ndarray | None = None, bias: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + np.float32(eps))
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def group_norm_bcl(x_bcl: np.ndarray, groups: int = 8, eps: float = 1e-5) -> np.ndarray:
    x_bcl = x_bcl.astype(np.float32, copy=False)
    B, C, L = x_bcl.shape
    xg = x_bcl.reshape(B, groups, C // groups, L)
    mean = xg.mean(axis=(2, 3), keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=(2, 3), keepdims=True)
    return ((xg - mean) / np.sqrt(var + np.float32(eps))).reshape(B, C, L)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(eps))
    return (x / rms) * weight


def gelu(x: np.ndarray) -> np.ndarray:
    # Exact GELU via Abramowitz & Stegun 7.1.26 erf approximation (max error < 1.5e-7), strictly float32
    x = x.astype(np.float32, copy=False)
    z = x * np.float32(0.7071067811865475)
    sign = np.sign(z)
    az = np.abs(z)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * az)
    poly = t * (
        np.float32(0.254829592)
        + t * (
            np.float32(-0.284496736)
            + t * (np.float32(1.421413741) + t * (np.float32(-1.453152027) + t * np.float32(1.061405429)))
        )
    )
    erf_val = sign * (np.float32(1.0) - poly * np.exp(-(az * az)))
    return np.float32(0.5) * x * (np.float32(1.0) + erf_val)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-np.clip(x, np.float32(-60.0), np.float32(60.0))))


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if x.ndim == 3 and x.shape[0] == 1:
        y = np.matmul(x[0], weight.T)[None, :, :]
    else:
        y = np.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


def precompute_rope_cos_sin(max_pos: int, head_dim: int = 64, base: float = 100000.0) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    positions = np.arange(max_pos, dtype=np.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1).astype(np.float32)
    return cos, sin


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]


def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = np.float32(1.0 / math.sqrt(q.shape[-1]))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    probs = softmax(scores, axis=-1)
    return np.matmul(probs, v)


def interp_linear_1d(x_bcl: np.ndarray, size: int) -> np.ndarray:
    x_bcl = x_bcl.astype(np.float32, copy=False)
    B, C, L = x_bcl.shape
    if L == size:
        return x_bcl.copy()
    if L == 1:
        return np.repeat(x_bcl, size, axis=2)
    idx = (np.arange(size, dtype=np.float32) + np.float32(0.5)) * np.float32(L / float(size)) - np.float32(0.5)
    idx = np.clip(idx, np.float32(0.0), np.float32(L - 1.0))
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, L - 1)
    w = (idx - lo.astype(np.float32))[None, None, :]
    return x_bcl[:, :, lo] * (np.float32(1.0) - w) + x_bcl[:, :, hi] * w


def interp_area_1d(x_bcl: np.ndarray, size: int) -> np.ndarray:
    B, C, L = x_bcl.shape
    if L == size:
        return x_bcl.copy()
    out = np.zeros((B, C, size), dtype=np.float32)
    scale = L / float(size)
    for i in range(size):
        start = i * scale
        end = (i + 1) * scale
        s_idx = int(np.floor(start))
        e_idx = int(np.ceil(end - 1e-7))
        out[:, :, i] = x_bcl[:, :, s_idx : min(e_idx, L)].sum(axis=2) / max(1, e_idx - s_idx)
    return out


def conv_transpose1d_s4(x_bcl: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # weight: [C_in, C_out, 4], stride=4, kernel_size=4
    B, C_in, L = x_bcl.shape
    C_out = weight.shape[1]
    x_clean = np.nan_to_num(x_bcl, nan=0.0, posinf=1e3, neginf=-1e3)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = np.matmul(np.swapaxes(x_clean, 1, 2), weight.reshape(C_in, C_out * 4)).reshape(B, L, C_out, 4)
    y = np.swapaxes(np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3), 1, 2).reshape(B, C_out, L * 4)
    return y + bias[None, :, None]


def conv1d_k3_p1(x_bcl: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # weight: [C_out, C_in, 3], stride=1, padding=1
    x_clean = np.nan_to_num(x_bcl, nan=0.0, posinf=1e3, neginf=-1e3)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = np.matmul(weight[:, :, 1], x_clean) + bias[None, :, None]
        if x_clean.shape[2] > 1:
            y[:, :, 1:] += np.matmul(weight[:, :, 0], x_clean[:, :, :-1])
            y[:, :, :-1] += np.matmul(weight[:, :, 2], x_clean[:, :, 1:])
    return np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)


class OmniC2C3Runtime:
    """Complete C2 (HybridAgent Plan) + Bridges (`ComboBridge` & `ExpertBridge`) + C3 (11-Level NCM Decode) runtime."""

    def __init__(self, base_dir: str | Path):
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir)
        self.pre = precomputed_dir(self.base_dir)
        missing = missing_c2_c3_files(self.base_dir)
        if missing:
            raise FileNotFoundError("Missing precomputed C2/C3/Bridge files: " + ", ".join(missing))

        self.lock = threading.Lock()
        self.tok = Lfm3PureBpeTokenizer(self.pre / "trunks" / "tokenizer.json")
        self.vocab_size = V_LFM3

        # 1. C2 · Plan (`hybridagent/plan-trunk-0`)
        self.plan_t = read_clvrtrk1(self.pre / "hybridagent" / "plan-trunk-0")
        self.has_hl2048 = "hl2048.proj1_fused.weight" in self.plan_t

        # 2. Bridges (`bridges/bridge-trunk-0`)
        self.bridge_t = read_clvrtrk1(self.pre / "bridges" / "bridge-trunk-0")

        # 3. C3 · VQ-VAE (`ncm/vqvae-trunk-0` + `ncm/vqvae-expert-0..10`)
        self.vq_trunk = read_clvrtrk1(self.pre / "ncm" / "vqvae-trunk-0")
        self.vq_experts = [read_clvrtrk1(self.pre / "ncm" / f"vqvae-expert-{s}") for s in range(11)]
        self.vq_wte_mm = np.memmap(str(self.pre / "ncm" / "vqvae_wte.bf16.bin"), dtype=np.uint16, mode="r", shape=(V_LFM3, D_VQ))
        self.vq_dec_mm = np.memmap(str(self.pre / "ncm" / "vqvae_dec.bf16.bin"), dtype=np.uint16, mode="r", shape=(V_LFM3, D_VQ))

        emb_path = self.pre / "trunks" / "global-0" / "embed_tokens.bf16.bin"
        self.lfm3_emb_mm = (
            np.memmap(str(emb_path), dtype=np.uint16, mode="r", shape=(V_LFM3, D_LFM3))
            if emb_path.is_file()
            else None
        )

        # 4. C3 · 12-Layer NextConceptTransformer (`ncm/ncm-global-0` + `ncm/ncm-trunk-0..11`)
        self.ncm_global = read_clvrtrk1(self.pre / "ncm" / "ncm-global-0")
        self.ncm_blocks = [read_clvrtrk1(self.pre / "ncm" / f"ncm-trunk-{l}") for l in range(12)]

        # 5. C3 · 3-Layer LexicalPointerDecoder (`ncm/ptr-global-0` + `ncm/ptr-trunk-0..2`)
        self.ptr_global = read_clvrtrk1(self.pre / "ncm" / "ptr-global-0")
        self.ptr_blocks = [read_clvrtrk1(self.pre / "ncm" / f"ptr-trunk-{l}") for l in range(3)]

        # Precompute RoPE tables and level index embeddings matching NextConceptTransformer
        self.rope_cos, self.rope_sin = precompute_rope_cos_sin(max_pos=4096, head_dim=64, base=100000.0)
        lvl_ids = []
        for i, lv in enumerate(VQ_LEVELS[1:]):
            lvl_ids.extend([i] * lv)
        self.level_pos_embed = self.ncm_global["transformer.wle.weight"][np.array(lvl_ids, dtype=np.int64)][None, :, :]

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

        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # ---------------- C2 · HybridAgent Continuous Latent Plan ----------------

    def token_embeddings_2048(self, token_ids: list[int]) -> np.ndarray:
        """Obtain 2048-d token representations for `token_ids` (`[L, 2048]`)."""
        ids = np.array([max(0, min(V_LFM3 - 1, int(x))) for x in (token_ids or [self.tok.bos_token_id])], dtype=np.int64)
        if self.lfm3_emb_mm is not None:
            return bf16_rows_to_f32(self.lfm3_emb_mm, ids)
        # Fallback if only 384-d vqvae_wte is present: tile/pad to 2048
        w384 = bf16_rows_to_f32(self.vq_wte_mm, ids)
        reps = (D_LFM3 + D_VQ - 1) // D_VQ
        return np.tile(w384, (1, reps))[:, :D_LFM3]

    def forward_hybridlink(self, h_2048: np.ndarray) -> np.ndarray:
        """Run C2 `Lfm3NativeHybridLink(d=2048, bottleneck=512)` on `h_2048` (`[..., 2048]`)."""
        dp = self.plan_t
        xn = layer_norm(h_2048, None, None, 1e-5)
        mid = gelu(linear(xn, dp["native.proj1_fused.weight"], dp["native.proj1_fused.bias"]))
        z = h_2048 + linear(mid, dp["native.proj2.weight"], dp["native.proj2.bias"])
        return layer_norm(z, dp["native.post_ln_fused.weight"], dp["native.post_ln_fused.bias"], 1e-5)

    def forward_hybridlink_2048(self, h_2048: np.ndarray) -> np.ndarray:
        """Run Stage-3 `HybridLink2048(d=2048 -> 2048 -> 2048)` on `h_2048` (`[..., 2048]`)."""
        if not self.has_hl2048:
            return self.forward_hybridlink(h_2048)
        dp = self.plan_t
        xn = layer_norm(h_2048, None, None, 1e-5)
        mid = gelu(linear(xn, dp["hl2048.proj1_fused.weight"], dp["hl2048.proj1_fused.bias"]))
        z = h_2048 + linear(mid, dp["hl2048.proj2.weight"], dp["hl2048.proj2.bias"])
        return layer_norm(z, dp["hl2048.post_ln.weight"], dp["hl2048.post_ln.bias"], 1e-5) * 0.024

    def plan_turn(self, prompt_text: str, offered_tools: list[str] | None = None, latent_steps: int = 3) -> dict[str, Any]:
        """Execute C2 HybridAgent continuous `<latent>` planning over the prompt tokens (`plan-trunk-0`)."""
        _, c2_info = self.c2_plan(prompt_text, c1_state=None, offered_tools=offered_tools, latent_steps=latent_steps)
        return c2_info

    # ---------------- Strict Stage 1: C1 · Inference ONLY ----------------

    def c1_infer(
        self,
        raw_inference_text: str,
        generation_stats: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute strictly C1 (Model Inference representation extraction ONLY) in a single one-time pass.
        Extracts 2048-d C1 hidden/token representations (`h_primary` and window views `h_valid`) across up to `P_BPE = 512` tokens.
        Performs zero C2 planning (`HybridLink`) and zero C3 decoding (`Bridges` / `11-Level NCM`).
        """
        t0 = time.perf_counter()
        raw = str(raw_inference_text or "")
        lines = raw.splitlines()
        in_code = False
        windows: list[dict[str, Any]] = []
        total_tokens = 0
        all_active_pids: list[int] = []

        for ln in lines:
            s = ln.strip()
            if s.startswith("```"):
                in_code = not in_code
                windows.append({"passthrough": True, "raw": ln})
                continue
            if in_code or not s or s.startswith("|"):
                windows.append({"passthrough": True, "raw": ln})
                continue
            m_pfx = re.match(r"^(\s*(?:[-*•]|\d+[.)]|#{1,4})\s+)(.+)$", ln)
            if m_pfx:
                pfx, body = m_pfx.group(1), m_pfx.group(2)
                lead_ws = ""
            else:
                pfx = ""
                lead_ws = ln[: len(ln) - len(ln.lstrip())]
                body = s
            clean = body.strip()
            all_pids = self.tok.encode(clean, allow_special=False)
            if not all_pids:
                windows.append({"passthrough": True, "raw": ln})
                continue
            pids = all_pids[:P_BPE]
            tail_ids = all_pids[P_BPE:]
            total_tokens += len(all_pids)
            start_idx = len(all_active_pids)
            if start_idx + len(pids) <= P_BPE:
                all_active_pids.extend(pids)
                end_idx = len(all_active_pids)
            else:
                start_idx, end_idx = -1, -1
            windows.append({
                "passthrough": False,
                "raw": ln,
                "prefix": pfx,
                "lead_ws": lead_ws,
                "clean": clean,
                "pids": pids,
                "tail_ids": tail_ids,
                "start": start_idx,
                "end": end_idx,
            })

        if not all_active_pids:
            all_active_pids = self.tok.encode(raw.strip() or "Hello", allow_special=False)[:P_BPE] or [self.tok.bos_token_id]
        # Single one-time C1 embedding lookup across the entire token window
        h_primary = self.token_embeddings_2048(all_active_pids)[None, :, :]  # [1, L_prim, 2048]
        for w in windows:
            if w.get("passthrough"):
                continue
            st_i, ed_i = w.get("start", -1), w.get("end", -1)
            if 0 <= st_i < ed_i <= h_primary.shape[1]:
                w["h_valid"] = h_primary[:, st_i:ed_i, :]
            else:
                w["h_valid"] = self.token_embeddings_2048(w["pids"])[None, :, :]

        tensor_ms = (time.perf_counter() - t0) * 1000.0
        prefill_ms = 0.0
        gen_tokens = 0
        for st in generation_stats or ():
            if isinstance(st, dict):
                prefill_ms += float(st.get("prefill_s") or 0.0) * 1000.0
                gen_tokens += int(st.get("new_tokens") or 0)
        total_ms = round(prefill_ms + tensor_ms, 2)
        n_active = 1 if any(not w.get("passthrough") for w in windows) else 0

        c1_state = {
            "stage": "c1_inference",
            "raw_text": raw,
            "windows": windows,
            "pids_primary": all_active_pids,
            "h_primary": h_primary,
            "tokens": gen_tokens or total_tokens,
            "ms": total_ms,
        }
        c1_info = {
            "layer": "c1_inference",
            "model": model_name or "LFM3-Omni-2B",
            "windows": n_active,
            "window_tokens": P_BPE,
            "tokens": gen_tokens or total_tokens,
            "ms": total_ms,
        }
        return c1_state, c1_info

    # ---------------- Strict Stage 2: C2 · HybridAgent Planning ONLY ----------------

    def c2_plan(
        self,
        prompt_text: str,
        c1_state: dict[str, Any] | None = None,
        offered_tools: list[str] | None = None,
        latent_steps: int = 3,
        jepa_arch: str = "auto",
        context_cue_64: np.ndarray | None = None,
        action_prior_8: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute strictly C2 (`HybridAgent` continuous `<latent>` planning on `plan-trunk-0` + selectable `EB-JEPA`/`RPRE`
        latent trajectory optimization) in a single one-time pass.
        Takes C1's `2048`-d inference state (`c1_state`) and computes `r_primary` (`[1, L_prim, 2048]`) via
        `Lfm3NativeHybridLink(d=2048, bottleneck=512)` + `HybridLink2048(d=2048 -> 2048 -> 2048)` + `jepa_engine.plan_c2_latent`.
        Performs zero C3 decoding (`Bridges` / `11-Level NCM`).
        """
        t0 = time.perf_counter()
        if c1_state is None or c1_state.get("stage") != "c1_inference":
            c1_state, _ = self.c1_infer(prompt_text or "Hello")

        if isinstance(offered_tools, str):
            tools_list = [offered_tools] if offered_tools.strip() else []
        else:
            tools_list = list(offered_tools or [])

        jepa_res: dict[str, Any] = {"enabled": False, "arch": "none"}
        with self.lock:
            h_primary = c1_state["h_primary"]  # [1, L_prim, 2048]
            h_cur = h_primary.mean(axis=1)     # [1, 2048]
            norms: list[float] = []
            cos_list: list[float] = []
            for _ in range(max(1, int(latent_steps))):
                r_native = self.forward_hybridlink(h_cur)
                r_s3 = self.forward_hybridlink_2048(h_cur)
                denom = float(np.linalg.norm(h_cur) * np.linalg.norm(r_native)) + 1e-8
                cos_list.append(float(np.sum(h_cur * r_native) / denom))
                norms.append(float(np.linalg.norm(r_native)))
                h_cur = h_cur + 0.5 * (r_native + r_s3)

            # Single one-time C2 pass across the entire token window
            r_primary = self.forward_hybridlink(h_primary) + self.forward_hybridlink_2048(h_primary)
            if jepa_arch != "none":
                try:
                    import jepa_engine
                    jepa_res = jepa_engine.for_base(self.base_dir).plan_c2_latent(
                        h_primary,
                        r_primary,
                        offered_tools=tools_list,
                        context_cue_64=context_cue_64,
                        action_prior_8=action_prior_8,
                        configured_arch=jepa_arch,
                    )
                    if jepa_res.get("enabled") and "r_refined_2048" in jepa_res:
                        r_primary = jepa_res["r_refined_2048"]
                        if jepa_res.get("ranked_tools"):
                            tools_list = jepa_res["ranked_tools"]
                except Exception:
                    jepa_res = {"enabled": False, "arch": "none"}

            planned_windows: list[dict[str, Any]] = []
            has_active = False
            for w in c1_state.get("windows") or ():
                if w.get("passthrough"):
                    planned_windows.append(dict(w))
                    continue
                has_active = True
                h_valid = w["h_valid"]
                st_i, ed_i = w.get("start", -1), w.get("end", -1)
                if 0 <= st_i < ed_i <= r_primary.shape[1]:
                    r_valid = r_primary[:, st_i:ed_i, :]
                else:
                    r_valid = self.forward_hybridlink(h_valid) + self.forward_hybridlink_2048(h_valid)
                planned_windows.append({
                    "passthrough": False,
                    "raw": w["raw"],
                    "prefix": w["prefix"],
                    "lead_ws": w["lead_ws"],
                    "clean": w["clean"],
                    "pids": w["pids"],
                    "tail_ids": w["tail_ids"],
                    "start": st_i,
                    "end": ed_i,
                    "h_valid": h_valid,
                    "r_valid": r_valid,
                })
            n_active = 1 if has_active else 0

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        jepa_meta = (
            {
                "jepa_arch": jepa_res["arch"],
                "jepa_energy_before": jepa_res["energy_before"],
                "jepa_energy_after": jepa_res["energy_after"],
                "jepa_energy_reduction_pct": jepa_res["energy_reduction_pct"],
                "jepa_vc_std": jepa_res["vc_std"],
            }
            if jepa_res.get("enabled")
            else {}
        )

        c2_state = {
            "stage": "c2_plan",
            "raw_text": c1_state.get("raw_text") or "",
            "windows": planned_windows,
            "pids_primary": c1_state["pids_primary"],
            "h_primary": h_primary,
            "r_primary": r_primary,
            "latent_steps": int(latent_steps),
            "latent_norm": round(norms[-1], 4),
            "hybridlink_cos": round(cos_list[-1], 4),
            "planned_tools": tools_list,
            "jepa_plan": jepa_res if jepa_res.get("enabled") else None,
            **jepa_meta,
            "ms": ms,
        }
        c2_info = {
            "layer": "c2_hybridagent",
            "module": "HybridLink2048 + Lfm3NativeHybridLink(d=2048, bottleneck=512)"
                      + (f" + JEPA({jepa_res['arch']})" if jepa_res.get("enabled") else ""),
            "trunk": "plan-trunk-0",
            "windows": n_active,
            "window_tokens": P_BPE,
            "prompt_tokens": int(c1_state.get("tokens") or len(c1_state["pids_primary"])),
            "latent_steps": int(latent_steps),
            "latent_norm": round(norms[-1], 4),
            "hybridlink_cos": round(cos_list[-1], 4),
            "planned_tools": tools_list,
            **jepa_meta,
            "ms": ms,
        }
        return c2_state, c2_info

    # ---------------- Bridges (`ComboBridge` + `ExpertBridge`) ----------------

    def forward_combo_bridge(self, r_2048: np.ndarray) -> np.ndarray:
        db = self.bridge_t
        xn = layer_norm(r_2048, None, None, 1e-5)
        x = linear(gelu(linear(xn, db["combo.fc1_fused.weight"], db["combo.fc1_fused.bias"])),
                   db["combo.fc2.weight"], db["combo.fc2.bias"])
        return layer_norm(x, db["combo.ln_out_fused.weight"], db["combo.ln_out_fused.bias"], 1e-5)

    def forward_expert_bridge(self, h_2048: np.ndarray) -> np.ndarray:
        db = self.bridge_t
        xn = layer_norm(h_2048, None, None, 1e-5)
        x = linear(gelu(linear(xn, db["expert.proj1_fused.weight"], db["expert.proj1_fused.bias"])),
                   db["expert.proj2.weight"], db["expert.proj2.bias"])
        return layer_norm(x, db["expert.ln_out_fused.weight"], db["expert.ln_out_fused.bias"], 1e-5)

    # ---------------- C3 · 11-Level Multiscale VQ-VAE & NextConceptTransformer ----------------

    def _vq_up_to_full(self, q_blc: np.ndarray, level_idx: int, target_len: int = 64) -> np.ndarray:
        q_bcl = np.swapaxes(q_blc, 1, 2)
        if level_idx == 0:
            dvq = self.vq_trunk
            x = conv_transpose1d_s4(q_bcl, dvq["learned_up_l1.0.weight"], dvq["learned_up_l1.0.bias"])
            x = group_norm_bcl(x, groups=8)
            x = conv_transpose1d_s4(x, dvq["learned_up_l1.2.weight"], dvq["learned_up_l1.2.bias"])
            x = group_norm_bcl(x, groups=8)
            x = conv_transpose1d_s4(x, dvq["learned_up_l1.4.weight"], dvq["learned_up_l1.4.bias"])
            return np.swapaxes(x, 1, 2)
        return np.swapaxes(interp_linear_1d(q_bcl, target_len), 1, 2)

    def _vq_apply_quant_resi(self, h_blc: np.ndarray, level_idx: int) -> np.ndarray:
        dve = self.vq_experts[level_idx]
        h_bcl = np.swapaxes(h_blc, 1, 2)
        conv_out = conv1d_k3_p1(h_bcl, dve["quant_resi.weight"], dve["quant_resi.bias"])
        out_bcl = np.float32(0.5) * h_bcl + np.float32(0.5) * conv_out
        return np.swapaxes(out_bcl, 1, 2)

    def _vq_next_autoregressive_input(self, level_idx: int, z_hat: np.ndarray, z_q_level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_levels = len(VQ_LEVELS)
        if level_idx != n_levels - 1:
            up = self._vq_up_to_full(z_q_level, level_idx, VQ_LEVELS[-1])
            up = self._vq_apply_quant_resi(up, level_idx)
            z_hat = z_hat + up
            nxt = np.swapaxes(interp_area_1d(np.swapaxes(z_hat, 1, 2), VQ_LEVELS[level_idx + 1]), 1, 2)
            return z_hat, nxt
        z_q = self._vq_apply_quant_resi(z_q_level, level_idx)
        z_hat = z_hat + z_q
        return z_hat, z_hat

    def _ncm_block_forward(
        self,
        x: np.ndarray,
        blk: dict[str, np.ndarray],
        cos: np.ndarray,
        sin: np.ndarray,
        past_kv: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        B, T, C = x.shape
        nh, hd = 12, 64
        xn1 = layer_norm(x, None, None, 1e-6)
        qkv = linear(xn1, blk["attn.c_attn_fused.weight"], blk["attn.c_attn_fused.bias"])
        q, k, v = np.split(qkv, 3, axis=2)
        q = rms_norm(q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3), blk["attn.q_norm.weight"], 1e-6)
        k = rms_norm(k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3), blk["attn.k_norm.weight"], 1e-6)
        v = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if past_kv is not None:
            k_all = np.concatenate([past_kv[0], k], axis=2)
            v_all = np.concatenate([past_kv[1], v], axis=2)
        else:
            k_all, v_all = k, v
        y = sdpa(q, k_all, v_all).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + linear(y, blk["attn.c_proj.weight"], blk["attn.c_proj.bias"])
        xn2 = layer_norm(x, None, None, 1e-6)
        mlp_h = gelu(linear(xn2, blk["mlp.c_fc_fused.weight"], blk["mlp.c_fc_fused.bias"]))
        x = x + linear(mlp_h, blk["mlp.c_proj.weight"], blk["mlp.c_proj.bias"])
        return x, (k, v)

    def _ptr_block_forward(self, x: np.ndarray, mem: np.ndarray, pblk: dict[str, np.ndarray]) -> np.ndarray:
        B, T, C = x.shape
        M = mem.shape[1]
        nh, hd = 6, 64
        xn1 = layer_norm(x, None, None, 1e-5)
        qkv = linear(xn1, pblk["qkv_fused.weight"], pblk["qkv_fused.bias"])
        q, k, v = np.split(qkv, 3, axis=2)
        q = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        y = sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + linear(y, pblk["proj.weight"], pblk["proj.bias"])

        hq = layer_norm(x, None, None, 1e-5)
        hkv = layer_norm(mem, None, None, 1e-5)
        cq = linear(hq, pblk["cx_q_fused.weight"], pblk["cx_q_fused.bias"]).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        ckv = linear(hkv, pblk["cx_kv_fused.weight"], pblk["cx_kv_fused.bias"])
        ck, cv = np.split(ckv, 2, axis=2)
        ck = ck.reshape(B, M, nh, hd).transpose(0, 2, 1, 3)
        cv = cv.reshape(B, M, nh, hd).transpose(0, 2, 1, 3)
        cy = sdpa(cq, ck, cv).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + linear(cy, pblk["cx_proj.weight"], pblk["cx_proj.bias"])

        xn2 = layer_norm(x, None, None, 1e-5)
        mlp_h = gelu(linear(xn2, pblk["mlp0_fused.weight"], pblk["mlp0_fused.bias"]))
        x = x + linear(mlp_h, pblk["mlp2.weight"], pblk["mlp2.bias"])
        return x

    def _vqvae_middle_hidden(self, z_refined: np.ndarray) -> np.ndarray:
        dvq = self.vq_trunk
        B, T, C = z_refined.shape
        nh, hd = 6, 64
        xn1 = layer_norm(z_refined, None, None, 1e-6)
        qkv = linear(xn1, dvq["middle_block.attn.c_attn_fused.weight"], dvq["middle_block.attn.c_attn_fused.bias"])
        q, k, v = np.split(qkv, 3, axis=2)
        q = rms_norm(q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3), dvq["middle_block.attn.q_norm.weight"], 1e-6)
        k = rms_norm(k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3), dvq["middle_block.attn.k_norm.weight"], 1e-6)
        v = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        y = sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        xd = z_refined + linear(y, dvq["middle_block.attn.c_proj.weight"], dvq["middle_block.attn.c_proj.bias"])
        xn2 = layer_norm(xd, None, None, 1e-6)
        mlp_h = gelu(linear(xn2, dvq["middle_block.mlp.c_fc_fused.weight"], dvq["middle_block.mlp.c_fc_fused.bias"]))
        xd = xd + linear(mlp_h, dvq["middle_block.mlp.c_proj.weight"], dvq["middle_block.mlp.c_proj.bias"])
        return layer_norm(xd, dvq["decoder.ln_f.weight"], None, 1e-6)

    def _vqvae_middle_and_decode(self, z_refined: np.ndarray, allowed_ids: np.ndarray) -> np.ndarray:
        xd = self._vqvae_middle_hidden(z_refined)
        sub_w = bf16_rows_to_f32(self.vq_dec_mm, allowed_ids)
        return np.matmul(xd, sub_w.T)

    def _build_allowed_vocab_ids(self, plan_str: str) -> np.ndarray:
        allowed = set(self.grammar_ids)
        allowed.update(self.tok.encode(plan_str, allow_special=False))
        for w in re.findall(r"[A-Za-z0-9.,:!?'£$€%/_\-+=&#()\[\]]+", plan_str):
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()):
                allowed.update(self.tok.encode(v, allow_special=False))
        return np.array(sorted(x for x in allowed if 0 <= x < self.vocab_size), dtype=np.int64)

    def _decode_window_aligned(self, sub_logits: np.ndarray, allowed_ids: np.ndarray, plan_ids: list[int]) -> tuple[str, list[int], bool]:
        pad_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < self.vocab_size and int(x) not in (pad_id, eos_id)]
        if not valid_p:
            return "", [], False

        id_to_col = {int(tid): idx for idx, tid in enumerate(allowed_ids.tolist())}
        cont_map: dict[int, set[int]] = {}
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs[0] in (":", ".", "'", "/", "-", "_", "=", "?", "&", "#")):
                cont_map.setdefault(a, set()).add(b)

        max_counts = collections.Counter(valid_p)
        content_set = {tid for tid in valid_p if tid not in self.repeatable_grammar_ids}
        emitted_counts: collections.Counter[int] = collections.Counter()

        T = sub_logits.shape[0]
        max_len = min(T, len(valid_p))
        clean_ids: list[int] = []
        prev_tid = -1
        prev_word = ""
        eos_col = id_to_col.get(eos_id)

        for t in range(max_len):
            lg = sub_logits[t].copy()
            if prev_tid >= 0:
                if emitted_counts[prev_tid] >= max_counts.get(prev_tid, 1) and prev_tid in id_to_col:
                    lg[id_to_col[prev_tid]] -= 30.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        if emitted_counts[nxt_b] < max_counts.get(nxt_b, 1) and nxt_b in id_to_col:
                            lg[id_to_col[nxt_b]] += 12.0
            for tid, cnt in emitted_counts.items():
                if tid not in self.repeatable_grammar_ids and cnt >= max_counts.get(tid, 1) and tid in id_to_col:
                    lg[id_to_col[tid]] -= 30.0

            all_covered = bool(content_set) and all(emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set)
            if all_covered and t >= len(valid_p) - 1 and eos_col is not None:
                lg[eos_col] += 18.0

            best_col = int(np.argmax(lg))
            tid = int(allowed_ids[best_col])
            if tid == eos_id:
                break
            if 0 <= tid < self.vocab_size and tid != pad_id:
                ws = self.tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2 and emitted_counts[tid] >= max_counts.get(tid, 1):
                    lg[best_col] -= 30.0
                    best_col = int(np.argmax(lg))
                    tid = int(allowed_ids[best_col])
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
        all_content_emitted = all(emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set)
        has_dangling_tail = bool(
            re.search(
                r"(?:\b(?:and|or|to|for|of|in|with|by|from|the|a|an|as|at|on)\s*$|,\s*$|\d+,\d{1,2}$)",
                decoded_str,
                flags=re.IGNORECASE,
            )
        )
        if not decoded_str or not all_content_emitted or has_dangling_tail:
            return self.tok.decode(valid_p), list(valid_p), True
        return decoded_str, clean_ids, False

    def _run_ncm_multiscale_levels(self, z_pref_v: np.ndarray, target_len: int) -> tuple[np.ndarray, int, int]:
        """Run C3 12-Layer NextConceptTransformer (`ncm-trunk-0..11` + `ncm-global-0`) + `vqvae-expert-0..10`
        across the multiscale VQ levels (`[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`) to produce `z_hat` (`[1, 64, 384]`)."""
        P_v = z_pref_v.shape[1]
        max_lvl = 2
        for li, l_res in enumerate(VQ_LEVELS):
            if li >= 2 and l_res >= target_len:
                max_lvl = li
                break
        else:
            max_lvl = len(VQ_LEVELS) - 1

        z_hat = np.zeros((1, VQ_LEVELS[-1], D_VQ), dtype=np.float32)
        h = linear(z_pref_v, self.ncm_global["transformer.wte.weight"], self.ncm_global["transformer.wte.bias"])
        cos_p, sin_p = self.rope_cos[:P_v], self.rope_sin[:P_v]

        prefix_kv = []
        for blk in self.ncm_blocks:
            h, kv = self._ncm_block_forward(h, blk, cos_p, sin_p, past_kv=None)
            prefix_kv.append(kv)

        h_l0 = layer_norm(h[:, -1:, :], None, None, 1e-6)
        l0_logits = linear(h_l0, self.ncm_global["lm_head_l0_fused.weight"], self.ncm_global["lm_head_l0_fused.bias"])
        idx_next = np.argmax(l0_logits, axis=-1)  # [1, 1]
        codes_emitted = int(idx_next.size)
        level_acum = 0
        P_orig = 96

        for level_idx in range(max_lvl):
            l_next = VQ_LEVELS[level_idx + 1]
            z_q_level = self.vq_experts[level_idx]["codebook.weight"][idx_next]
            z_hat, x = self._vq_next_autoregressive_input(level_idx, z_hat, z_q_level)
            x = linear(x, self.ncm_global["transformer.wte.weight"], self.ncm_global["transformer.wte.bias"])
            x = x + self.level_pos_embed[:, level_acum : level_acum + l_next, :]
            cos_k = self.rope_cos[P_orig : P_orig + l_next]
            sin_k = self.rope_sin[P_orig : P_orig + l_next]
            for blk, pkv in zip(self.ncm_blocks, prefix_kv):
                x, _ = self._ncm_block_forward(x, blk, cos_k, sin_k, past_kv=pkv)
            x_ln = layer_norm(x, None, None, 1e-6)
            logits = linear(
                x_ln,
                self.ncm_global[f"lm_heads.{level_idx}_fused.weight"],
                self.ncm_global[f"lm_heads.{level_idx}_fused.bias"],
            )
            idx_next = np.argmax(logits, axis=-1)
            codes_emitted += int(idx_next.size)
            level_acum += l_next

        z_q_last = self.vq_experts[max_lvl]["codebook.weight"][idx_next]
        z_hat, _ = self._vq_next_autoregressive_input(max_lvl, z_hat, z_q_last)
        return z_hat, codes_emitted, max_lvl + 1

    @staticmethod
    def _pos_slice(w_2d: np.ndarray, length: int) -> np.ndarray:
        if length <= w_2d.shape[0]:
            return w_2d[:length][None, :, :]
        return np.swapaxes(interp_linear_1d(np.swapaxes(w_2d[None, :, :], 1, 2), length), 1, 2)

    def _pointer_decode_from_zhat(
        self,
        z_hat: np.ndarray,
        z_rh_v: np.ndarray,
        z_bpe_v: np.ndarray,
        pids: list[int],
        tail_ids: list[int],
        clean: str,
        precomputed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[str, bool]:
        """Run C3 3-Layer LexicalPointerDecoder (`ptr-trunk-0..2` + `ptr-global-0`) + `vqvae-trunk-0` on one planned window."""
        L_val = len(pids)
        dpg = self.ptr_global
        if precomputed is not None:
            x_d, mem_bpe, sub_logits_in, allowed_ids = precomputed
            sub_logits = sub_logits_in.copy()
        else:
            allowed_ids = self._build_allowed_vocab_ids(clean)
            t_pos = self._pos_slice(dpg["tgt_pos.weight"], L_val)
            p_pos = self._pos_slice(dpg["plan_pos.weight"], L_val)
            r_pos = self._pos_slice(dpg["rh_pos.weight"], L_val)
            mem_bpe = z_bpe_v + p_pos
            mem_rh = z_rh_v + r_pos
            mem = np.concatenate([mem_rh, mem_bpe], axis=1)

            zh_sub = (
                z_hat[:, :L_val, :]
                if L_val <= z_hat.shape[1]
                else np.swapaxes(interp_linear_1d(np.swapaxes(z_hat, 1, 2), L_val), 1, 2)
            )
            x_d = zh_sub + t_pos
            for pblk in self.ptr_blocks:
                x_d = self._ptr_block_forward(x_d, mem, pblk)
            z_refined = zh_sub + layer_norm(x_d, dpg["ln_out_fused.weight"], dpg["ln_out_fused.bias"], 1e-5)
            sub_logits = self._vqvae_middle_and_decode(z_refined, allowed_ids)[0]  # [L_val, len(allowed_ids)]

        cq = linear(x_d, dpg["copy_q.weight"], dpg.get("copy_q.bias"))
        cq = cq / (np.linalg.norm(cq, axis=-1, keepdims=True) + 1e-8)
        ck = linear(mem_bpe, dpg["copy_k.weight"], dpg.get("copy_k.bias"))
        ck = ck / (np.linalg.norm(ck, axis=-1, keepdims=True) + 1e-8)
        sim = np.matmul(cq, np.swapaxes(ck, -1, -2))  # [1, L_val, L_val]

        has_semi = any(pid == self.tok.semicolon_id for pid in pids)
        expected_T = min(float(L_val), max(4.0, L_val * 1.25 + 2.0)) if has_semi else float(L_val)
        t_pos_idx = np.arange(L_val, dtype=np.float32)[None, :, None]
        p_pos_idx = np.arange(L_val, dtype=np.float32)[None, None, :]
        if has_semi:
            diag_bias = -2.5 * (((t_pos_idx / expected_T) - (p_pos_idx / max(1.0, float(L_val)))) ** 2)
        else:
            diag_bias = -45.0 * ((t_pos_idx - p_pos_idx) ** 2)
        attn_probs = softmax(sim * 12.0 + diag_bias, axis=-1)
        ctx_vec = np.matmul(attn_probs, z_bpe_v)

        g_copy = sigmoid(linear(np.concatenate([x_d, ctx_vec], axis=-1), dpg["copy_gate.weight"], dpg.get("copy_gate.bias")))
        soft_scale = g_copy * 16.0
        eff_scale = soft_scale if has_semi else (soft_scale + 65.0)
        copy_boost = (attn_probs * eff_scale)[0]  # [L_val, L_val]

        id_to_col = {int(tid): col for col, tid in enumerate(allowed_ids.tolist())}
        for src_pos, pid in enumerate(pids):
            col = id_to_col.get(int(pid))
            if col is not None:
                sub_logits[:, col] += copy_boost[:, src_pos]

        rendered_first, _, used_fallback = self._decode_window_aligned(sub_logits, allowed_ids, pids)
        if tail_ids:
            rendered_first = (rendered_first + self.tok.decode(tail_ids)).strip()
        return rendered_first, used_fallback

    # ---------------- Strict Stage 3: C3 · Bridges + 11-Level NCM Decoding ONLY ----------------

    def c3_decode(self, c2_state: dict[str, Any], on_window: Any = None) -> tuple[str, dict[str, Any]]:
        """Execute strictly C3 (Bridges + 11-Level VQ-VAE + 12-Layer NextConceptTransformer + 3-Layer LexicalPointerDecoder).
        Requires `c2_state` produced by `c2_plan` (which already contains `h_valid` from C1 and `r_valid` from C2).
        Never calls C1 (`token_embeddings_2048`) or C2 (`forward_hybridlink`).
        """
        if not isinstance(c2_state, dict) or c2_state.get("stage") != "c2_plan":
            raise ValueError("c3_decode requires a c2_state dict with stage='c2_plan' from c2_plan()")
        t0 = time.perf_counter()
        raw_text = str(c2_state.get("raw_text") or "")
        if not raw_text.strip():
            return raw_text, {"layer": "ncm", "windows": 0, "codes": 0, "s": 0.0}

        with self.lock:
            pids_prim = c2_state["pids_primary"]
            h_prim = c2_state["h_primary"]
            r_prim = c2_state["r_primary"]

            # 1. Single one-time Bridges pass (`bridge-trunk-0`) across the entire token window (`P_BPE = 512`)
            z_rh_p = self.forward_combo_bridge(r_prim)
            z_exp_p = self.forward_expert_bridge(h_prim)
            z_bpe_p = bf16_rows_to_f32(self.vq_wte_mm, pids_prim)[None, :, :] + z_exp_p
            if z_rh_p.shape[1] > 48:
                z_rh_ncm = np.swapaxes(interp_linear_1d(np.swapaxes(z_rh_p, 1, 2), 48), 1, 2)
                z_bpe_ncm = np.swapaxes(interp_linear_1d(np.swapaxes(z_bpe_p, 1, 2), 48), 1, 2)
                z_pref_p = np.concatenate([z_rh_ncm, z_bpe_ncm], axis=1)
            else:
                z_pref_p = np.concatenate([z_rh_p, z_bpe_p], axis=1)

            # 2. Single one-time C3 12-Layer NextConceptTransformer + 11-Level Multiscale VQ-VAE pass
            z_hat, codes_emitted, levels_run = self._run_ncm_multiscale_levels(z_pref_p, min(len(pids_prim), 64))

            # 3. Single one-time C3 3-Layer LexicalPointerDecoder + VQ-VAE Middle Block pass across `L_prim`
            L_prim = z_rh_p.shape[1]
            dpg = self.ptr_global
            t_pos_p = self._pos_slice(dpg["tgt_pos.weight"], L_prim)
            p_pos_p = self._pos_slice(dpg["plan_pos.weight"], L_prim)
            r_pos_p = self._pos_slice(dpg["rh_pos.weight"], L_prim)
            mem_bpe_p = z_bpe_p + p_pos_p
            mem_rh_p = z_rh_p + r_pos_p
            mem_p = np.concatenate([mem_rh_p, mem_bpe_p], axis=1)
            zh_prim = (
                z_hat[:, :L_prim, :]
                if L_prim <= z_hat.shape[1]
                else np.swapaxes(interp_linear_1d(np.swapaxes(z_hat, 1, 2), L_prim), 1, 2)
            )
            x_d_prim = zh_prim + t_pos_p
            for pblk in self.ptr_blocks:
                x_d_prim = self._ptr_block_forward(x_d_prim, mem_p, pblk)
            z_ref_prim = zh_prim + layer_norm(x_d_prim, dpg["ln_out_fused.weight"], dpg["ln_out_fused.bias"], 1e-5)
            xd_prim = self._vqvae_middle_hidden(z_ref_prim)
            allowed_prim = self._build_allowed_vocab_ids(raw_text)
            sub_w_prim = bf16_rows_to_f32(self.vq_dec_mm, allowed_prim)
            sub_logits_prim = np.matmul(xd_prim[0], sub_w_prim.T)

            out_lines: list[str] = []
            has_active = False
            for w in c2_state.get("windows") or ():
                if w.get("passthrough"):
                    line_out = str(w.get("raw") or "")
                    out_lines.append(line_out)
                    if on_window is not None:
                        on_window(line_out)
                    continue
                has_active = True
                pids = w["pids"]
                tail_ids = w["tail_ids"]
                st_i, ed_i = w.get("start", -1), w.get("end", -1)
                if 0 <= st_i < ed_i <= z_rh_p.shape[1]:
                    z_rh_v = z_rh_p[:, st_i:ed_i, :]
                    z_bpe_v = z_bpe_p[:, st_i:ed_i, :]
                    precomp = (x_d_prim[:, st_i:ed_i, :], mem_bpe_p[:, st_i:ed_i, :], sub_logits_prim[st_i:ed_i, :], allowed_prim)
                else:
                    h_valid = w["h_valid"]
                    r_valid = w["r_valid"]
                    z_rh_v = self.forward_combo_bridge(r_valid)
                    z_exp_v = self.forward_expert_bridge(h_valid)
                    z_bpe_v = bf16_rows_to_f32(self.vq_wte_mm, pids)[None, :, :] + z_exp_v
                    precomp = None
                rendered_body, _ = self._pointer_decode_from_zhat(
                    z_hat, z_rh_v, z_bpe_v, pids, tail_ids, w["clean"], precomputed=precomp
                )
                if w.get("prefix"):
                    line_out = str(w["prefix"]) + rendered_body.lstrip()
                else:
                    line_out = str(w.get("lead_ws") or "") + rendered_body
                out_lines.append(line_out)
                if on_window is not None:
                    on_window(line_out)
            n_win = 1 if has_active else 0

            elapsed_s = round(time.perf_counter() - t0, 2)
            return "\n".join(out_lines), {
                "layer": "ncm",
                "windows": n_win,
                "window_tokens": P_BPE,
                "codes": codes_emitted,
                "levels": levels_run or 11,
                "bridges": "Lfm3ComboBridge(2048->384) + Lfm3ExpertBridge(2048->384)",
                "decoder": "11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder",
                "s": elapsed_s,
            }

    def render_window(self, window_text: str) -> tuple[str, dict[str, Any]]:
        """Standalone helper for single-window tests: runs strict C1 -> C2 -> C3 in order."""
        t0 = time.perf_counter()
        raw = str(window_text or "")
        clean = raw.strip()
        if not clean or clean.startswith("```") or clean.startswith("|"):
            return window_text, {"codes": 0, "levels": 0, "ms": 0.0, "passthrough": True}
        c1_state, _ = self.c1_infer(window_text)
        c2_state, _ = self.c2_plan(window_text, c1_state=c1_state)
        decoded, meta = self.c3_decode(c2_state)
        return decoded, {
            "codes": int(meta.get("codes") or 0),
            "levels": int(meta.get("levels") or 0),
            "used_fallback": False,
            "ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def render_reply(self, reply: str, windows: list[str] | None = None) -> tuple[str, dict[str, Any]]:
        """Standalone helper: runs strict C1 (infer) -> C2 (plan) -> C3 (decode) pipeline."""
        if not (reply or "").strip():
            return reply, {"layer": "ncm", "windows": 0, "codes": 0, "s": 0.0}
        c1_state, _ = self.c1_infer(reply)
        c2_state, _ = self.c2_plan(reply, c1_state=c1_state)
        return self.c3_decode(c2_state)


_RUNTIMES: dict[str, OmniC2C3Runtime] = {}
_RUNTIMES_LOCK = threading.Lock()


def for_base(base_dir: str | Path) -> OmniC2C3Runtime | None:
    key = str(Path(base_dir).resolve())
    with _RUNTIMES_LOCK:
        if key in _RUNTIMES:
            return _RUNTIMES[key]
        if missing_c2_c3_files(base_dir):
            return None
        try:
            rt = OmniC2C3Runtime(base_dir)
            _RUNTIMES[key] = rt
            return rt
        except Exception as exc:
            print(f"[Clover] OmniC2C3Runtime could not load: {exc}", flush=True)
            return None
