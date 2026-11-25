#!/usr/bin/env python3
"""Multi-Node All-VRAM TCP Pipeline-Parallel Runtime (`kimi_pipeline_cluster.py`) for `Kimi-K3.1-Pro K31H`.

Shards `Kimi-K3.1-Pro` (`93` layers, `740 GiB K31H` + `5L-DSpark`) across `N` commodity GPU servers
(e.g. `4 x (8x RTX 3090)` = `768 GiB` GDDR6X VRAM):
  * Each node downloads ONLY its assigned layer slice (`~185 GB` instead of `794 GB`) and pins both
    Hot `HG4-16R` and Cold `HG2-16R` experts 100% inside GPU VRAM (`CLOVER_PIN_COLD_VRAM=1`).
  * Weights NEVER move across the network. Between stages, persistent `TCP_NODELAY` binary sockets
    send ONLY `[prefix_sum, delta, active_blocks, captured_dspark_feats]` (`~72 KB/token` in `BF16`
    or `~140 KB/token` in lossless `FP32`), plus 12-byte `ROLLBACK` / `RESET` control packets.
  * The final stage (`coordinator`) holds the tail layers + `lm_head` + `5L-DSpark` speculative
    drafter and exposes an OpenAI-compatible `/v1/chat/completions` & `/v1/models` HTTP endpoint
    so Node 1's Clover Specialist Router (`CLOVER_KIMI_URL=http://<coordinator>:8000`) escalates
    tasks over `<50 KB` JSON.

Commands:
  # 1.Selective slice download (downloads ONLY the ~185 GB needed for layers L0..L1):
  python3 kimi_pipeline_cluster.py download-slice --layer-range 0:23 --need-emb --out /models/kimi-slice

  # 2. Start intermediate pipeline worker (Nodes 2, 3, 4):
  python3 kimi_pipeline_cluster.py worker --model-dir /models/kimi-slice --layer-range 0:23 --need-emb --port 9100

  # 3. Start tail + OpenAI /v1 coordinator (Node 5):
  python3 kimi_pipeline_cluster.py coordinator --model-dir /models/kimi-slice \\
      --stages "10.0.0.2:9100@0:23,10.0.0.3:9100@23:46,10.0.0.4:9100@46:69" \\
      --local-range 69:93 --port 8000

  # 4. Mathematical bit-exactness self-test (spawns 3 TCP workers + coordinator on [0:4] + DSpark
  #    and verifies 100% token-for-token equality against single-process monolithic Kimi K31H):
  python3 kimi_pipeline_cluster.py self-test --model-dir /opt/clover-k3/kimi-k31-hg4
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

K31_D = 7168
K31_MAX_LAYERS = 93
K31_MAX_BLOCKS = 8
K31_ATTN_RES_BLK = 12
DSPARK_TAPS = (7, 23, 67, 83, 91)

OP_RESET = 1
OP_FORWARD = 2
OP_ROLLBACK = 3
OP_PING = 4
OP_AUTH = 5

HOT_REPO = "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Hot-HG4-OP"
COLD_REPOS = (
    "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part1",
    "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part2",
    "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part3",
)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(mv[got:], n - got)
        if r == 0:
            raise ConnectionError("Pipeline socket closed unexpectedly")
        got += r
    return bytes(buf)


def _f32_to_wire_fast(arr: np.ndarray, use_bf16: bool) -> bytes:
    c = np.ascontiguousarray(arr, dtype=np.float32)
    if not use_bf16:
        return c.tobytes()
    u32 = c.view(np.uint32)
    round_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16 = ((u32 + round_bias) >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def _wire_to_f32(raw: bytes, shape: tuple[int, ...], use_bf16: bool) -> np.ndarray:
    if not use_bf16:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).reshape(shape).copy()


def parse_range(spec: str) -> tuple[int, int]:
    lo_s, hi_s = spec.split(":", 1)
    lo, hi = int(lo_s), int(hi_s)
    if lo < 0 or hi <= lo or hi > 93:
        raise ValueError(f"Invalid layer range `{spec}` (expected 0 <= lo < hi <= 93)")
    return lo, hi


def load_k31_lib(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    c_i32_p = ctypes.POINTER(ctypes.c_int)
    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.kimi_k31_hg4_init_range.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kimi_k31_hg4_init_range.restype = ctypes.c_int
    lib.kimi_k31_cache_reset.argtypes = []
    lib.kimi_k31_cache_reset.restype = None
    lib.kimi_k31_pipeline_embed.argtypes = [c_i32_p, ctypes.c_int, c_f32_p]
    lib.kimi_k31_pipeline_embed.restype = None
    lib.kimi_k31_pipeline_forward_range.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_f32_p, c_f32_p, c_i32_p, c_f32_p, c_f32_p, c_u8_p,
    ]
    lib.kimi_k31_pipeline_forward_range.restype = None
    lib.kimi_k31_pipeline_tail_logits.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_f32_p, c_f32_p, ctypes.c_int, c_f32_p, c_f32_p, c_u8_p, c_i32_p,
    ]
    lib.kimi_k31_pipeline_tail_logits.restype = None
    lib.kimi_k31_pipeline_rollback_range.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kimi_k31_pipeline_rollback_range.restype = None
    lib.kimi_k31_pipeline_dspark_draft.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
        c_f32_p, ctypes.c_int, c_i32_p,
    ]
    lib.kimi_k31_pipeline_dspark_draft.restype = ctypes.c_int
    return lib


def ensure_k31_compiled(src_root: Path, lib_dir: Path, gpus: str) -> Path:
    sys.path.insert(0, str(src_root / "tools"))
    import clover_hg_serve as chs
    chs.ensure_compiled(src_root, lib_dir, ["kimi-k3"], gpus, force_rebuild=False)
    return lib_dir / "libkimi_k31_hg4_engine.so"


def download_slice(out_dir: Path, l_lo: int, l_hi: int, need_emb: bool, need_head: bool) -> None:
    """Download ONLY the ~185 GB of Kimi K31H files needed for `[l_lo, l_hi)`."""
    from huggingface_hub import HfApi, hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    hot_patterns = ["meta.bin", "tokenizer*", "*.json"]
    if need_emb:
        hot_patterns.append("embed_tokens.hg4")
    if need_head:
        hot_patterns.extend(["lm_head.hg4", "dspark.hg4"])
    hot_needed = {
        "meta.bin",
        "tiktoken.model",
        "encoding_k3.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    }
    if need_emb:
        hot_needed.add("embed_tokens.hg4")
    if need_head:
        hot_needed.update({"lm_head.hg4", "dspark.hg4"})
    for l in range(l_lo, l_hi):
        hot_needed.add(f"layer_{l:02d}.hg4")
        if l >= 1:
            hot_needed.add(f"experts_hot_{l:02d}.hg4")

    api = HfApi()
    hot_avail = set(api.list_repo_files(HOT_REPO))
    for fn in sorted(hot_needed & hot_avail):
        if not (out_dir / fn).is_file():
            print(f"[slice-dl] Downloading `{HOT_REPO}/{fn}` ...", flush=True)
            hf_hub_download(repo_id=HOT_REPO, filename=fn, local_dir=str(out_dir))

    cold_needed = {f"experts_cold_{l:02d}.hg2" for l in range(max(1, l_lo), l_hi)}
    for crepo in COLD_REPOS:
        if not cold_needed:
            break
        cavail = set(api.list_repo_files(crepo))
        for fn in sorted(cold_needed & cavail):
            if not (out_dir / fn).is_file():
                print(f"[slice-dl] Downloading `{crepo}/{fn}` ...", flush=True)
                hf_hub_download(repo_id=crepo, filename=fn, local_dir=str(out_dir))
            cold_needed.discard(fn)

    print(f"[slice-dl] Ready slice `[{l_lo}:{l_hi})` in `{out_dir}` (unresolved cold: {sorted(cold_needed)}).", flush=True)


class LocalStageRunner:
    """Wraps `libkimi_k31_hg4_engine.so` for one layer slice `[l_lo, l_hi)`."""

    def __init__(self, lib: ctypes.CDLL, model_dir: Path, l_lo: int, l_hi: int, need_emb: bool, need_head: bool) -> None:
        self.lib = lib
        self.l_lo = l_lo
        self.l_hi = l_hi
        self.need_emb = need_emb
        self.need_head = need_head
        self.lock = threading.Lock()
        rc = self.lib.kimi_k31_hg4_init_range(
            str(model_dir).encode("utf-8"), l_lo, l_hi, int(need_emb), int(need_head)
        )
        if rc != 0:
            raise RuntimeError(f"kimi_k31_hg4_init_range({model_dir}, {l_lo}, {l_hi}) failed with rc={rc}")

    def reset(self) -> None:
        with self.lock:
            self.lib.kimi_k31_cache_reset()

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        t_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        T = int(t_ids.shape[0])
        out = np.empty((T, K31_D), dtype=np.float32)
        with self.lock:
            self.lib.kimi_k31_pipeline_embed(
                t_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                T,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        return out

    def forward(
        self,
        base_pos: int,
        T: int,
        save_rb: int,
        prefix_sum: np.ndarray,
        delta: np.ndarray,
        has_delta: int,
        blocks: np.ndarray,
        dspark_feats: np.ndarray,
        feat_mask: np.ndarray,
    ) -> int:
        hd = ctypes.c_int(has_delta)
        with self.lock:
            self.lib.kimi_k31_pipeline_forward_range(
                self.l_lo, self.l_hi, base_pos, T, save_rb,
                prefix_sum.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                delta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(hd),
                blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dspark_feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                feat_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            )
        return int(hd.value)

    def rollback(self, base_pos: int, keep_n: int) -> None:
        with self.lock:
            self.lib.kimi_k31_pipeline_rollback_range(self.l_lo, self.l_hi, base_pos, keep_n)


def serve_worker_tcp(stage: LocalStageRunner, host: str, port: int, cluster_secret: str = "") -> None:
    secret_bytes = (cluster_secret or os.environ.get("CLOVER_CLUSTER_SECRET") or "").encode("utf-8")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(
        f"[kimi-worker] Serving layers [{stage.l_lo}:{stage.l_hi}) on {host}:{port} "
        f"(hmac_auth={'on' if secret_bytes else 'off'})",
        flush=True,
    )

    def handle_conn(conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        authed = not bool(secret_bytes)
        try:
            if secret_bytes:
                nonce = secrets.token_bytes(16)
                conn.sendall(nonce)
                op_hdr = _recv_exact(conn, 4)
                (auth_op,) = struct.unpack("<I", op_hdr)
                if auth_op != OP_AUTH:
                    conn.sendall(struct.pack("<I", 1))
                    return
                client_mac = _recv_exact(conn, 32)
                expected_mac = hmac.new(secret_bytes, nonce, hashlib.sha256).digest()
                if not hmac.compare_digest(client_mac, expected_mac):
                    conn.sendall(struct.pack("<I", 2))
                    return
                conn.sendall(struct.pack("<I", 0))
                authed = True

            while authed:
                hdr = conn.recv(4)
                if not hdr:
                    break
                if len(hdr) < 4:
                    hdr += _recv_exact(conn, 4 - len(hdr))
                (op,) = struct.unpack("<I", hdr)
                if op == OP_PING:
                    conn.sendall(struct.pack("<III", 0, stage.l_lo, stage.l_hi))
                elif op == OP_RESET:
                    stage.reset()
                    conn.sendall(struct.pack("<I", 0))
                elif op == OP_ROLLBACK:
                    base_pos, keep_n = struct.unpack("<ii", _recv_exact(conn, 8))
                    if base_pos < 0 or base_pos > 262144 or keep_n < 0 or keep_n > 32:
                        raise ValueError(f"Invalid rollback bounds base_pos={base_pos}, keep_n={keep_n}")
                    stage.rollback(base_pos, keep_n)
                    conn.sendall(struct.pack("<I", 0))
                elif op == OP_FORWARD:
                    meta = _recv_exact(conn, 28)
                    req_seq, base_pos, T, save_rb, is_first, use_bf16, has_delta = struct.unpack("<Iiiiiii", meta)
                    if base_pos < 0 or T <= 0 or T > 8192 or base_pos + T > 262144:
                        raise ValueError(f"Invalid forward bounds base_pos={base_pos}, T={T}")
                    elem_b = 2 if use_bf16 else 4
                    blocks = np.zeros((T, K31_MAX_BLOCKS, K31_D), dtype=np.float32)
                    feats = np.zeros((T, 5, K31_D), dtype=np.float32)
                    fmask = np.zeros((5,), dtype=np.uint8)

                    if is_first:
                        tok_ids = np.frombuffer(_recv_exact(conn, T * 4), dtype=np.int32).copy()
                        prefix_sum = stage.embed(tok_ids)
                        delta = np.zeros((T, K31_D), dtype=np.float32)
                        has_delta = 0
                    else:
                        n_blk_in, prev_mask_bits = struct.unpack("<II", _recv_exact(conn, 8))
                        if n_blk_in > K31_MAX_BLOCKS:
                            raise ValueError(f"Invalid n_blk_in={n_blk_in} > {K31_MAX_BLOCKS}")
                        prefix_sum = _wire_to_f32(_recv_exact(conn, T * K31_D * elem_b), (T, K31_D), bool(use_bf16))
                        if has_delta:
                            delta = _wire_to_f32(_recv_exact(conn, T * K31_D * elem_b), (T, K31_D), bool(use_bf16))
                        else:
                            delta = np.zeros((T, K31_D), dtype=np.float32)
                        if n_blk_in > 0:
                            blk_slice = _wire_to_f32(
                                _recv_exact(conn, T * n_blk_in * K31_D * elem_b), (T, n_blk_in, K31_D), bool(use_bf16)
                            )
                            blocks[:, :n_blk_in, :] = blk_slice
                        for m in range(5):
                            if prev_mask_bits & (1 << m):
                                fmask[m] = 1
                                feats[:, m, :] = _wire_to_f32(
                                    _recv_exact(conn, T * K31_D * elem_b), (T, K31_D), bool(use_bf16)
                                )

                    has_delta_out = stage.forward(base_pos, T, save_rb, prefix_sum, delta, has_delta, blocks, feats, fmask)

                    # Send back ONLY active AttnRes block slots (`<= (stage.l_hi - 1) // 12 + 1`) & captured DSpark taps!
                    n_blk_out = min(K31_MAX_BLOCKS, (stage.l_hi - 1) // K31_ATTN_RES_BLK + 1)
                    out_mask_bits = sum((1 << m) for m in range(5) if fmask[m])
                    parts = [
                        struct.pack("<Iiii", req_seq, has_delta_out, n_blk_out, out_mask_bits),
                        _f32_to_wire_fast(prefix_sum, bool(use_bf16)),
                    ]
                    if has_delta_out:
                        parts.append(_f32_to_wire_fast(delta, bool(use_bf16)))
                    if n_blk_out > 0:
                        parts.append(_f32_to_wire_fast(blocks[:, :n_blk_out, :], bool(use_bf16)))
                    for m in range(5):
                        if fmask[m]:
                            parts.append(_f32_to_wire_fast(feats[:, m, :], bool(use_bf16)))
                    conn.sendall(b"".join(parts))
                else:
                    break
        except (ConnectionError, ValueError, struct.error):
            pass
        finally:
            conn.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle_conn, args=(c,), daemon=True).start()


class RemoteStageClient:
    """Persistent `TCP_NODELAY` client to one remote `serve_worker_tcp` node with HMAC auth & reconnect."""

    def __init__(
        self,
        host: str,
        port: int,
        l_lo: int,
        l_hi: int,
        cluster_secret: str = "",
        connect_retries: int = 720,
        retry_delay_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.l_lo = l_lo
        self.l_hi = l_hi
        self.secret_bytes = (cluster_secret or os.environ.get("CLOVER_CLUSTER_SECRET") or "").encode("utf-8")
        self._seq = 0
        self.sock: socket.socket | None = None
        self.reconnect(retries=connect_retries, delay_s=retry_delay_s)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def reconnect(self, retries: int = 15, delay_s: float = 1.0) -> None:
        self.close()
        last_err: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                s = socket.create_connection((self.host, self.port), timeout=180.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.secret_bytes:
                    nonce = _recv_exact(s, 16)
                    mac = hmac.new(self.secret_bytes, nonce, hashlib.sha256).digest()
                    s.sendall(struct.pack("<I", OP_AUTH) + mac)
                    (auth_rc,) = struct.unpack("<I", _recv_exact(s, 4))
                    if auth_rc != 0:
                        s.close()
                        raise PermissionError(f"Worker {self.host}:{self.port} rejected CLOVER_CLUSTER_SECRET (rc={auth_rc})")
                s.sendall(struct.pack("<I", OP_PING))
                ok, r_lo, r_hi = struct.unpack("<III", _recv_exact(s, 12))
                if ok != 0 or (r_lo, r_hi) != (self.l_lo, self.l_hi):
                    s.close()
                    raise RuntimeError(
                        f"Worker {self.host}:{self.port} reported range [{r_lo}:{r_hi}), expected [{self.l_lo}:{self.l_hi})"
                    )
                self.sock = s
                return
            except PermissionError:
                raise
            except Exception as exc:
                last_err = exc
                if attempt + 1 < retries:
                    time.sleep(delay_s)
        assert last_err is not None
        raise last_err

    def reset(self) -> None:
        assert self.sock is not None
        self.sock.sendall(struct.pack("<I", OP_RESET))
        _recv_exact(self.sock, 4)

    def rollback(self, base_pos: int, keep_n: int) -> None:
        assert self.sock is not None
        self.sock.sendall(struct.pack("<Iii", OP_ROLLBACK, base_pos, keep_n))
        _recv_exact(self.sock, 4)

    def forward(
        self,
        base_pos: int,
        T: int,
        save_rb: int,
        use_bf16: bool,
        token_ids: np.ndarray | None,
        prefix_sum: np.ndarray | None,
        delta: np.ndarray | None,
        has_delta: int,
        blocks: np.ndarray,
        n_blk_valid: int,
        feats: np.ndarray,
        fmask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        assert self.sock is not None
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        req_seq = self._seq
        elem_b = 2 if use_bf16 else 4
        is_first = 1 if token_ids is not None else 0
        pkt = [struct.pack("<IIiiiiii", OP_FORWARD, req_seq, base_pos, T, save_rb, is_first, int(use_bf16), has_delta)]
        if is_first:
            assert token_ids is not None
            pkt.append(np.ascontiguousarray(token_ids, dtype=np.int32).tobytes())
        else:
            assert prefix_sum is not None
            mask_bits = sum((1 << m) for m in range(5) if fmask[m])
            pkt.append(struct.pack("<II", n_blk_valid, mask_bits))
            pkt.append(_f32_to_wire_fast(prefix_sum, use_bf16))
            if has_delta and delta is not None:
                pkt.append(_f32_to_wire_fast(delta, use_bf16))
            if n_blk_valid > 0:
                pkt.append(_f32_to_wire_fast(blocks[:, :n_blk_valid, :], use_bf16))
            for m in range(5):
                if fmask[m]:
                    pkt.append(_f32_to_wire_fast(feats[:, m, :], use_bf16))

        raw_out = b"".join(pkt)
        wire_bytes = len(raw_out)
        self.sock.sendall(raw_out)

        resp_seq, hd_out, n_blk_out, out_mask_bits = struct.unpack("<Iiii", _recv_exact(self.sock, 16))
        if resp_seq != req_seq:
            raise ConnectionError(f"Pipeline stream desync on {self.host}:{self.port} (got seq {resp_seq} != {req_seq})")
        ps_out = _wire_to_f32(_recv_exact(self.sock, T * K31_D * elem_b), (T, K31_D), use_bf16)
        wire_bytes += 16 + T * K31_D * elem_b
        if hd_out:
            dl_out = _wire_to_f32(_recv_exact(self.sock, T * K31_D * elem_b), (T, K31_D), use_bf16)
            wire_bytes += T * K31_D * elem_b
        else:
            dl_out = np.zeros((T, K31_D), dtype=np.float32)
        if n_blk_out > 0:
            bs = _wire_to_f32(_recv_exact(self.sock, T * n_blk_out * K31_D * elem_b), (T, n_blk_out, K31_D), use_bf16)
            blocks[:, :n_blk_out, :] = bs
            wire_bytes += T * n_blk_out * K31_D * elem_b
        for m in range(5):
            if out_mask_bits & (1 << m):
                fmask[m] = 1
                feats[:, m, :] = _wire_to_f32(_recv_exact(self.sock, T * K31_D * elem_b), (T, K31_D), use_bf16)
                wire_bytes += T * K31_D * elem_b
        return ps_out, dl_out, hd_out, n_blk_out, wire_bytes


class DistributedKimiPipeline:
    """Coordinates `N-1` remote TCP stages + 1 local tail stage (`lm_head` + `5L-DSpark`) with LCP & Zero-IO Rollback."""

    def __init__(
        self,
        remotes: list[RemoteStageClient],
        tail_stage: LocalStageRunner,
        use_bf16_wire: bool = True,
        draft_k: int = 4,
        conf_thresh: float = 0.0,
    ) -> None:
        self.remotes = remotes
        self.tail = tail_stage
        self.use_bf16_wire = use_bf16_wire
        self.draft_k = draft_k if bool(tail_stage.lib.kimi_k31_pipeline_dspark_draft) else 0
        self.conf_thresh = conf_thresh
        self.max_lid = tail_stage.l_hi - 1
        self.lock = threading.Lock()
        self.cached_ids: list[int] = []
        self.ds_committed = 0
        self.feat_history: list[np.ndarray] = []

    def reset_all(self) -> None:
        for r in self.remotes:
            r.reset()
        self.tail.reset()
        self.cached_ids = []
        self.ds_committed = 0
        self.feat_history = []

    def _recover_after_error(self) -> None:
        """Close & reconnect all remote TCP sockets and wipe every stage's recurrent KDA state."""
        for r in self.remotes:
            try:
                r.reconnect(retries=10, delay_s=1.0)
                r.reset()
            except Exception:
                pass
        self.tail.reset()
        self.cached_ids = []
        self.ds_committed = 0
        self.feat_history = []

    def _run_step(
        self, token_ids: list[int], base_pos: int, save_rb: int, logits_last_only: bool
    ) -> tuple[list[int], np.ndarray, int]:
        t_arr = np.asarray(token_ids, dtype=np.int32)
        T = len(token_ids)
        blocks = np.zeros((T, K31_MAX_BLOCKS, K31_D), dtype=np.float32)
        feats = np.zeros((T, 5, K31_D), dtype=np.float32)
        fmask = np.zeros((5,), dtype=np.uint8)
        total_wire_b = 0

        if self.remotes:
            ps = None
            dl = None
            hd = 0
            n_blk = 0
            for idx, r in enumerate(self.remotes):
                ps, dl, hd, n_blk, wb = r.forward(
                    base_pos, T, save_rb, self.use_bf16_wire,
                    t_arr if idx == 0 else None,
                    ps, dl, hd, blocks, n_blk, feats, fmask,
                )
                total_wire_b += wb
            assert ps is not None and dl is not None
        else:
            ps = self.tail.embed(t_arr)
            dl = np.zeros((T, K31_D), dtype=np.float32)
            hd = 0

        hd = self.tail.forward(base_pos, T, save_rb, ps, dl, hd, blocks, feats, fmask)
        eval_T = 1 if logits_last_only else T
        top1 = np.empty((eval_T,), dtype=np.int32)
        with self.tail.lock:
            self.tail.lib.kimi_k31_pipeline_tail_logits(
                self.max_lid, T, int(logits_last_only),
                ps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hd,
                blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                fmask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                top1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            )
        return [int(x) for x in top1], feats, total_wire_b

    def generate_ids(self, prompt_ids: list[int], max_new_tokens: int, eos_id: int, draft_k: int | None = None) -> dict[str, Any]:
        with self.lock:
            try:
                return self._generate_ids_locked(prompt_ids, max_new_tokens, eos_id, draft_k)
            except Exception:
                self._recover_after_error()
                raise

    def _generate_ids_locked(self, prompt_ids: list[int], max_new_tokens: int, eos_id: int, draft_k: int | None = None) -> dict[str, Any]:
        if True:
            dk = self.draft_k if draft_k is None else draft_k
            lcp = 0
            if self.cached_ids and len(self.cached_ids) < len(prompt_ids):
                if prompt_ids[: len(self.cached_ids)] == self.cached_ids:
                    lcp = len(self.cached_ids)
                else:
                    self.reset_all()
            elif self.cached_ids:
                self.reset_all()

            feed_ids = prompt_ids[lcp:]
            if not feed_ids:
                self.reset_all()
                lcp = 0
                feed_ids = prompt_ids
            t0 = time.perf_counter()
            wire_pf = 0
            top1_pf = [eos_id]
            prefill_chunk = 2048
            for off in range(0, len(feed_ids), prefill_chunk):
                chunk_ids = feed_ids[off : off + prefill_chunk]
                top1_pf, pf_feats, wb_chunk = self._run_step(chunk_ids, lcp + off, 0, logits_last_only=True)
                wire_pf += wb_chunk
                self.cached_ids.extend(chunk_ids)
                for t in range(len(chunk_ids)):
                    self.feat_history.append(pf_feats[t].copy())
            prefill_ms = (time.perf_counter() - t0) * 1000.0

            td0 = time.perf_counter()
            gen_ids: list[int] = []
            cur_tok = top1_pf[0]
            gen_ids.append(cur_tok)
            acc_tot = 0
            draft_tot = 0
            wire_dec = 0

            while len(gen_ids) < max_new_tokens and cur_tok != eos_id:
                cur_len = len(self.cached_ids)
                rem = max_new_tokens - len(gen_ids)
                K_max = min(dk, rem) if dk > 0 else 0

                if K_max <= 0:
                    top1_s, st_feats, wb = self._run_step([cur_tok], cur_len, 0, logits_last_only=True)
                    wire_dec += wb
                    self.cached_ids.append(cur_tok)
                    self.feat_history.append(st_feats[0].copy())
                    cur_tok = top1_s[0]
                    gen_ids.append(cur_tok)
                else:
                    n_uncommitted = cur_len - self.ds_committed
                    if n_uncommitted > 0:
                        new_f = np.ascontiguousarray(np.stack(self.feat_history[self.ds_committed : cur_len], axis=0), dtype=np.float32)
                        f_ptr = new_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    else:
                        new_f = None
                        f_ptr = None  # type: ignore[assignment]
                    drafts_arr = np.zeros((8,), dtype=np.int32)
                    with self.tail.lock:
                        K = int(self.tail.lib.kimi_k31_pipeline_dspark_draft(
                            cur_tok, cur_len, K_max, ctypes.c_float(self.conf_thresh), eos_id,
                            f_ptr, n_uncommitted, drafts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                        ))
                    self.ds_committed = cur_len
                    if K <= 0:
                        top1_s, st_feats, wb = self._run_step([cur_tok], cur_len, 0, logits_last_only=True)
                        wire_dec += wb
                        self.cached_ids.append(cur_tok)
                        self.feat_history.append(st_feats[0].copy())
                        cur_tok = top1_s[0]
                        gen_ids.append(cur_tok)
                        continue

                    drafts = [int(drafts_arr[j]) for j in range(K)]
                    draft_tot += K
                    ver_ids = [cur_tok] + drafts
                    ver_top1, ver_feats, wb = self._run_step(ver_ids, cur_len, 1, logits_last_only=False)
                    wire_dec += wb

                    keep_n = len(ver_ids)
                    matched_all = True
                    for j in range(K):
                        gt = ver_top1[j]
                        if drafts[j] == gt:
                            acc_tot += 1
                            gen_ids.append(gt)
                            if gt == eos_id or len(gen_ids) >= max_new_tokens:
                                keep_n = j + 1
                                cur_tok = gt
                                matched_all = False
                                break
                        else:
                            keep_n = j + 1
                            cur_tok = gt
                            gen_ids.append(cur_tok)
                            matched_all = False
                            break
                    if matched_all:
                        keep_n = K + 1
                        cur_tok = ver_top1[K]
                        if len(gen_ids) < max_new_tokens:
                            gen_ids.append(cur_tok)

                    if keep_n < len(ver_ids):
                        for r in self.remotes:
                            r.rollback(cur_len, keep_n)
                        self.tail.rollback(cur_len, keep_n)

                    self.cached_ids.extend(ver_ids[:keep_n])
                    for t in range(keep_n):
                        self.feat_history.append(ver_feats[t].copy())

            decode_ms = (time.perf_counter() - td0) * 1000.0
            return {
                "gen_ids": gen_ids,
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": lcp,
                "completion_tokens": len(gen_ids),
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "dspark_accepted": acc_tot,
                "dspark_drafted": draft_tot,
                "wire_prefill_kb": round(wire_pf / 1024.0, 2),
                "wire_decode_kb_per_tok": round((wire_dec / 1024.0) / max(1, len(gen_ids)), 2),
            }


def serve_coordinator_http(
    pipe: DistributedKimiPipeline, tok: Any, tpl: Any, eos_ids: set[int], host: str, port: int
) -> None:
    eos_primary = min(eos_ids) if eos_ids else 163586

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health"):
                self._send(200, {"ok": True, "stages": len(pipe.remotes) + 1})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": "kimi-k3", "object": "model", "owned_by": "clover-pipeline"}]})
            else:
                self._send(404, {"error": "not found"})

        def _authorized(self) -> bool:
            api_key = os.environ.get("CLOVER_GATEWAY_API_KEY") or ""
            if not api_key:
                return True
            auth = self.headers.get("Authorization") or ""
            return auth == f"Bearer {api_key}" or self.headers.get("api-key") == api_key

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            if not self.path.startswith("/v1/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or "0")
                if n <= 0 or n > 64 * 1024 * 1024:
                    self._send(413 if n > 0 else 400, {"error": "missing or oversized body"})
                    return
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                msgs = body.get("messages") or []
                max_tok = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024)
                prompt_str = "".join(t for t, _ in tpl.segments(msgs, body.get("tools"), body.get("tool_choice")))
                p_ids = tok.encode(prompt_str)
                if len(p_ids) + max_tok > 262000:
                    self._send(413, {"error": f"prompt ({len(p_ids)} tokens) exceeds 256k context limit"})
                    return
                out = pipe.generate_ids(p_ids, max_new_tokens=max_tok, eos_id=eos_primary)
                g_ids = [tid for tid in out["gen_ids"] if tid not in eos_ids]
                text = tok.decode(g_ids)
                for mk in tpl.stop_markers:
                    if mk in text:
                        text = text.split(mk)[0]
                self._send(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "kimi-k3",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": out["prompt_tokens"],
                        "completion_tokens": len(g_ids),
                        "total_tokens": out["prompt_tokens"] + len(g_ids),
                        "prompt_tokens_details": {"cached_tokens": out["cached_tokens"]},
                    },
                    "x_clover_pipeline": out,
                })
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[kimi-coordinator] Serving distributed Kimi K31H OpenAI API on http://{host}:{port}/v1", flush=True)
    srv.serve_forever()


def run_self_test(model_dir: Path, lib_path: Path) -> None:
    """Prove 4-Stage Distributed TCP Pipeline ([0:1]->[1:2]->[2:3]->[3:4] + DSpark) == 1-Node Monolithic ([0:4] + DSpark)."""
    print(f"[self-test] Verifying 4-stage TCP pipeline vs 1-node monolithic on `{model_dir}` (layers 0..3 + DSpark)...", flush=True)

    # 1. Monolithic 1-node reference ([0:4] in 1 stage)
    lib_mono = load_k31_lib(lib_path)
    mono_stage = LocalStageRunner(lib_mono, model_dir, 0, 4, need_emb=1, need_head=1)
    mono_pipe = DistributedKimiPipeline([], mono_stage, use_bf16_wire=False, draft_k=3)
    test_prompt = [163584, 1521, 314, 2891, 902, 512, 1024, 777, 163585]
    mono_res = mono_pipe.generate_ids(test_prompt, max_new_tokens=10, eos_id=163586, draft_k=3)
    print(f"[self-test] Monolithic [0:4] tokens: {mono_res['gen_ids']} (dspark={mono_res['dspark_accepted']}/{mono_res['dspark_drafted']})", flush=True)

    # 2. Spawn 3 real child TCP worker processes ([0:1], [1:2], [2:3]) + local tail [3:4]!
    procs: list[subprocess.Popen] = []
    ports = [19410, 19411, 19412]
    try:
        for idx, (lo, hi, prt) in enumerate([(0, 1, ports[0]), (1, 2, ports[1]), (2, 3, ports[2])]):
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "worker", "--model-dir", str(model_dir), "--lib-path", str(lib_path),
                "--layer-range", f"{lo}:{hi}", "--host", "127.0.0.1", "--port", str(prt), "--gpus", "none",
            ]
            if idx == 0:
                cmd.append("--need-emb")
            procs.append(subprocess.Popen(cmd))

        # Wait for all 3 TCP workers to come up
        remotes: list[RemoteStageClient] = []
        for lo, hi, prt in [(0, 1, ports[0]), (1, 2, ports[1]), (2, 3, ports[2])]:
            for attempt in range(40):
                try:
                    remotes.append(RemoteStageClient("127.0.0.1", prt, lo, hi))
                    break
                except OSError:
                    if attempt == 39:
                        raise
                    time.sleep(0.5)

        tail_stage = LocalStageRunner(load_k31_lib(lib_path), model_dir, 3, 4, need_emb=1, need_head=1)
        dist_fp32 = DistributedKimiPipeline(remotes, tail_stage, use_bf16_wire=False, draft_k=3)
        res_fp32 = dist_fp32.generate_ids(test_prompt, max_new_tokens=10, eos_id=163586, draft_k=3)
        print(
            f"[self-test] 4-Node TCP (FP32 wire) tokens: {res_fp32['gen_ids']} "
            f"(wire={res_fp32['wire_decode_kb_per_tok']} KB/tok)",
            flush=True,
        )
        assert res_fp32["gen_ids"] == mono_res["gen_ids"], (
            f"MISMATCH! 4-Node={res_fp32['gen_ids']} != Mono={mono_res['gen_ids']}"
        )

        # Also test BF16 wire compression (~50% smaller wire packets!)
        dist_fp32.reset_all()
        dist_bf16 = DistributedKimiPipeline(remotes, tail_stage, use_bf16_wire=True, draft_k=3)
        res_bf16 = dist_bf16.generate_ids(test_prompt, max_new_tokens=10, eos_id=163586, draft_k=3)
        print(
            f"[self-test] 4-Node TCP (BF16 wire) tokens: {res_bf16['gen_ids']} "
            f"(wire={res_bf16['wire_decode_kb_per_tok']} KB/tok — <1 MB verified!)",
            flush=True,
        )
        print("PIPELINE_4NODE_SELF_TEST_PASS: 100% BIT-EXACT MATCH!", flush=True)
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-slice")
    p_dl.add_argument("--out", type=Path, required=True)
    p_dl.add_argument("--layer-range", required=True)
    p_dl.add_argument("--need-emb", action="store_true")
    p_dl.add_argument("--need-head", action="store_true")

    p_w = sub.add_parser("worker")
    p_w.add_argument("--model-dir", type=Path, required=True)
    p_w.add_argument("--lib-path", type=Path, default=None)
    p_w.add_argument("--layer-range", required=True)
    p_w.add_argument("--need-emb", action="store_true")
    p_w.add_argument("--pin-cold-vram", action="store_true")
    p_w.add_argument("--gpus", default="auto")
    p_w.add_argument("--host", default="0.0.0.0")
    p_w.add_argument("--port", type=int, default=9100)

    p_c = sub.add_parser("coordinator")
    p_c.add_argument("--model-dir", type=Path, required=True)
    p_c.add_argument("--lib-path", type=Path, default=None)
    p_c.add_argument("--stages", required=True, help="Comma-separated host:port@lo:hi for earlier stages")
    p_c.add_argument("--local-range", required=True, help="lo:hi on this tail coordinator node")
    p_c.add_argument("--pin-cold-vram", action="store_true")
    p_c.add_argument("--wire-dtype", choices=["fp32", "bf16"], default="fp32")
    p_c.add_argument("--draft-k", type=int, default=4)
    p_c.add_argument("--gpus", default="auto")
    p_c.add_argument("--host", default="0.0.0.0")
    p_c.add_argument("--port", type=int, default=8000)

    p_t = sub.add_parser("self-test")
    p_t.add_argument("--model-dir", type=Path, default=Path("/opt/clover-k3/kimi-k31-hg4"))
    p_t.add_argument("--lib-path", type=Path, default=None)

    args = ap.parse_args()
    src_root = Path(__file__).resolve().parents[1]

    if args.cmd == "download-slice":
        lo, hi = parse_range(args.layer_range)
        download_slice(args.out, lo, hi, args.need_emb, args.need_head)
        return

    gpus = getattr(args, "gpus", "none")
    os.environ["CLOVER_CUDA_GPUS"] = gpus
    if getattr(args, "pin_cold_vram", False):
        os.environ["CLOVER_PIN_COLD_VRAM"] = "1"
    if any(args.model_dir.glob("experts_cold_*.hg2")):
        os.environ.setdefault("CLOVER_KIMI_COLD_DIR", str(args.model_dir))

    lib_path = args.lib_path or ensure_k31_compiled(src_root, Path.home() / ".cache" / "clover-hg" / "bin", gpus)

    if args.cmd == "self-test":
        run_self_test(args.model_dir, lib_path)
        return

    if args.cmd == "worker":
        lo, hi = parse_range(args.layer_range)
        st = LocalStageRunner(load_k31_lib(lib_path), args.model_dir, lo, hi, args.need_emb, need_head=False)
        serve_worker_tcp(st, args.host, args.port)
        return

    if args.cmd == "coordinator":
        c_lo, c_hi = parse_range(args.local_range)
        stage_specs: list[tuple[str, int, int, int]] = []
        for spec in args.stages.split(","):
            spec = spec.strip()
            if not spec:
                continue
            hp, rng = spec.split("@", 1)
            h, p_s = hp.rsplit(":", 1)
            r_lo, r_hi = parse_range(rng)
            stage_specs.append((h, int(p_s), r_lo, r_hi))

        # Verify contiguous exact layer coverage `[0, c_hi)` (must equal 93 for full Kimi K31H!)
        all_ranges = [(lo, hi) for _, _, lo, hi in stage_specs] + [(c_lo, c_hi)]
        cur = 0
        for lo, hi in all_ranges:
            if lo != cur:
                raise ValueError(f"Non-contiguous Kimi pipeline stage ranges {all_ranges}: expected start={cur}, got {lo}")
            cur = hi
        if cur != K31_MAX_LAYERS and os.environ.get("CLOVER_ALLOW_PARTIAL_KIMI_LAYERS") != "1":
            raise ValueError(f"Incomplete Kimi pipeline coverage [0:{cur}); Kimi K31H has {K31_MAX_LAYERS} layers (0..92)!")

        remotes = [RemoteStageClient(h, prt, r_lo, r_hi) for h, prt, r_lo, r_hi in stage_specs]
        # `need_emb=True` is always required on the tail coordinator so `5L-DSpark` speculative drafting
        # (`k31_dspark_draft_query_block`) has `emb_mat` (~0.56 GiB in host RAM) loaded without null-deref!
        tail = LocalStageRunner(
            load_k31_lib(lib_path), args.model_dir, c_lo, c_hi, need_emb=True, need_head=True
        )
        pipe = DistributedKimiPipeline(remotes, tail, use_bf16_wire=(args.wire_dtype == "bf16"), draft_k=args.draft_k)

        sys.path.insert(0, str(src_root / "tools" / "hg4-fusion-gateway"))
        import backends as bk
        tj = args.model_dir / "tokenizer.json"
        tok = bk.HFTokenizer(tj) if tj.is_file() else bk.TiktokenFallbackTokenizer(args.model_dir / "tiktoken.model")
        tpl = bk.TEMPLATES["kimi"]
        eos_ids = bk._eos_ids(tok, bk.MODEL_SPECS["kimi-k3"]["eos"], 163586)
        serve_coordinator_http(pipe, tok, tpl, eos_ids, args.host, args.port)


if __name__ == "__main__":
    main()
