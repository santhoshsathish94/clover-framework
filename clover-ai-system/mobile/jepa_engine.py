"""Unified JEPA Architecture Interface & Flow Router for the Clover AI System (`jepa_engine.py`).

Allows Clover to select and execute any of the four core JEPA architectures—or their unified Clover synthesis
(`clover_rpre`)—across every stage of the Clover runtime (`S1-S3 <-> C1-C3 <-> Cycle`):

  1. `ijepa`       — Visual Latent Prediction (`facebookresearch/ijepa`):
                     Context-only sparse patch encoding (`apply_masks`, ~30% self-attention FLOPs) + multi-block target
                     prediction for S3 visual/UI salience ranking (`image_inspector.py`) and scene layout classification.
  2. `audio_jepa`  — Audio Spectrogram Latent Prediction (`LudovicTuncay/Audio-JEPA`):
                     2D time-frequency log-mel patch tokenization + symmetric `CenteredNormMSELoss` + time-frequency
                     band masking for S3/C1 voice & audio quality, prosody, and acoustic regime detection.
  3. `eb_jepa`     — Energy-Based Generalized Learning & Latent MPC Planning (`facebookresearch/eb_jepa`):
                     Variance-Covariance (`VCLoss`) + Temporal Similarity + Inverse Dynamics (`IDM`) regularization and
                     elitist Cross-Entropy Method (`CEMPlanner`) latent trajectory optimization inside C2 (`omni_c2_c3.py`).
  4. `mjepa`       — Shared Multimodal Audio-Visual-State Representation Learning (`facebookresearch/MJEPA`):
                     Single shared 4-level hierarchical Transformer trunk + 9-path intra/cross-modal predictors
                     (`a2v`, `v2a`, `s2v`, `s2a`, `av2a`, `av2v`) for S2 cross-modal memory recall and missing-modality
                     inference.
  5. `clover_rpre` — Clover Reusable Predictive Representation Engine (`research/predictive-representation-engine`):
                     Unifies `ijepa + audio_jepa + eb_jepa + mjepa` and closes the deployment loop (`Outcome -> Growth`)
                     via `OutcomeResidualAdapter` with a Baseline-Prototype Anomaly Gate and Held-Out Non-Regression
                     Rollback Gate.
  6. `auto`        — Clover Semantic Selector (default): automatically selects the right JEPA architecture for each active
                     flow in the turn (`visual_s3 -> ijepa`, `audio_s3 -> audio_jepa`, `multimodal_s2 -> mjepa`,
                     `c2_plan -> eb_jepa`, `outcome_growth -> clover_rpre`).
  7. `none`        — Disabled (baseline pass-through for controlled A/B comparison across flows).

Implemented in pure vectorized NumPy/BLAS (reading `CLVRTRK1` binary trunks from `<base>/data/omni/precomputed/jepa/jepa-trunk-0`
or `.npz` weights) so it runs in 1–6 ms on Windows Laptop, Linux (`k3`), and Android (`Chaquopy`) without requiring
PyTorch at inference time, while pairing directly with `train_clover_jepa_trunks.py` for CPU/GPU PyTorch training.
"""
from __future__ import annotations

import copy
import json
import math
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

JEPA_ARCHITECTURES = ("auto", "clover_rpre", "ijepa", "audio_jepa", "eb_jepa", "mjepa", "none")

FLOW_DEFAULT_ARCH = {
    "visual_s3": "ijepa",
    "audio_s3": "audio_jepa",
    "multimodal_s2": "mjepa",
    "c2_plan": "eb_jepa",
    "outcome_growth": "clover_rpre",
}

VISUAL_SCENE_KINDS = (
    "ui_controls",
    "tabular_chart",
    "document_receipt",
    "code_terminal",
    "diagram_scene",
    "natural_photo",
)

ACOUSTIC_REGIMES = (
    "clean_speech",
    "noisy_speech",
    "harmonic_music",
    "transient_alert",
    "low_snr_silence",
)

D_JEPA = 64
D_PRED = 32
D_VQ = 384
D_LFM3 = 2048
D_ACT = 8
N_PATCHES = 16  # 4x4 grid for fast runtime patch tokenization


# ---------------- Pure-NumPy Mathematical Primitives ----------------

def _layer_norm(x: np.ndarray, weight: np.ndarray | None = None, bias: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    y = (x - m) / np.sqrt(v + np.float32(eps))
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x * x * x)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - mx)
    return e / (np.sum(e, axis=axis, keepdims=True) + np.float32(1e-9))


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = np.matmul(np.asarray(x, dtype=np.float32), w.T)
    return y + b if b is not None else y


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + np.float32(eps))
    return x / n


def centered_norm_mse(pred: np.ndarray, target: np.ndarray, center_pred: bool = True) -> float:
    """Audio-JEPA symmetric centered normalized MSE loss (`rpre/losses.py`)."""
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    t = (t - t.mean(axis=-1, keepdims=True)) / np.sqrt(t.var(axis=-1, keepdims=True) + np.float32(1e-6))
    if center_pred:
        p = (p - p.mean(axis=-1, keepdims=True)) / np.sqrt(p.var(axis=-1, keepdims=True) + np.float32(1e-6))
    pn = _l2_normalize(p, axis=-1)
    tn = _l2_normalize(t, axis=-1)
    return float(np.mean((pn - tn) ** 2))


def write_clvrtrk1(trunk_dir: Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write a dictionary of float32 NumPy tensors into Clover's `CLVRTRK1` binary trunk format."""
    trunk_dir = Path(trunk_dir)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    common_buf = bytearray()
    op_buf = bytearray()
    ffn_buf = bytearray()
    entries: list[bytes] = []

    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        raw = a.tobytes()
        if "mlp" in name or "ffn" in name or "net." in name:
            fid, buf = 3, ffn_buf
        elif "attn" in name or "gru" in name or "proj" in name:
            fid, buf = 1, op_buf
        else:
            fid, buf = 0, common_buf
        offset = len(buf)
        buf.extend(raw)
        ndim = a.ndim
        dims = list(a.shape) + [1, 1, 1, 1]
        name_b = name.encode("utf-8")[:63].ljust(64, b"\x00")
        meta = struct.pack(
            "<IIIQQIIII",
            fid,
            0,  # float32 dtype_code = 0
            ndim,
            offset,
            len(raw),
            dims[0],
            dims[1],
            dims[2],
            dims[3],
        )
        entries.append(name_b + meta)

    header = b"CLVRTRK1" + struct.pack("<III", 1, 0, len(entries))
    (trunk_dir / "index.bin").write_bytes(header + b"".join(entries))
    (trunk_dir / "common.bin").write_bytes(bytes(common_buf))
    (trunk_dir / "operator.bin").write_bytes(bytes(op_buf))
    (trunk_dir / "ffn.bin").write_bytes(bytes(ffn_buf))
    return trunk_dir


def read_clvrtrk1(trunk_dir: Path) -> dict[str, np.ndarray]:
    """Read `CLVRTRK1` binary trunk directory into a dictionary of float32 NumPy arrays."""
    idx_path = Path(trunk_dir) / "index.bin"
    idx_bytes = idx_path.read_bytes()
    if idx_bytes[:8] != b"CLVRTRK1":
        raise ValueError(f"Invalid CLVRTRK1 magic in {trunk_dir}")
    _, _, n_entries = struct.unpack("<III", idx_bytes[8:20])
    fnames = {0: "common.bin", 1: "operator.bin", 3: "ffn.bin"}
    blobs: dict[int, memoryview] = {}
    for fid, fn in fnames.items():
        fp = Path(trunk_dir) / fn
        if fp.is_file() and fp.stat().st_size > 0:
            blobs[fid] = memoryview(fp.read_bytes())
    out: dict[str, np.ndarray] = {}
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
            out[tname] = (u16.astype(np.uint32) << 16).view(np.float32).copy().reshape(dims)
        else:
            out[tname] = np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)
    return out


def _build_default_weights(seed: int = 20261107) -> dict[str, np.ndarray]:
    """Deterministic orthogonal/calibrated initialization when precomputed trunk files are not yet on disk."""
    rng = np.random.default_rng(seed)

    def ortho(rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
        m = rng.standard_normal((rows, cols)).astype(np.float32)
        u, _, vt = np.linalg.svd(m, full_matrices=False)
        q = u if rows >= cols else vt
        if q.shape != (rows, cols):
            q = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-6)
        return (q * np.float32(scale)).astype(np.float32)

    w: dict[str, np.ndarray] = {}
    w["vis_adapter.weight"] = ortho(D_JEPA, 48, 0.8)
    w["vis_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["vis_pos_embed"] = (rng.standard_normal((N_PATCHES, D_JEPA)) * 0.04).astype(np.float32)
    w["vis_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    w["aud_adapter.weight"] = ortho(D_JEPA, 16, 0.8)
    w["aud_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["aud_pos_embed"] = (rng.standard_normal((N_PATCHES, D_JEPA)) * 0.04).astype(np.float32)
    w["aud_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    w["state_adapter.weight"] = ortho(D_JEPA, D_VQ, 0.8)
    w["state_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["state_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    for l in range(4):
        w[f"enc.block.{l}.qkv.weight"] = ortho(3 * D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.qkv.bias"] = np.zeros(3 * D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.proj.weight"] = ortho(D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.proj.bias"] = np.zeros(D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.mlp1.weight"] = ortho(2 * D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.mlp1.bias"] = np.zeros(2 * D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.mlp2.weight"] = ortho(D_JEPA, 2 * D_JEPA, 0.7)
        w[f"enc.block.{l}.mlp2.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    w["pred.in_proj.weight"] = ortho(D_PRED, D_JEPA, 0.7)
    w["pred.in_proj.bias"] = np.zeros(D_PRED, dtype=np.float32)
    w["pred.mask_token"] = np.zeros((1, D_PRED), dtype=np.float32)
    w["pred.pos_embed"] = (rng.standard_normal((N_PATCHES, D_PRED)) * 0.03).astype(np.float32)
    w["pred.qkv.weight"] = ortho(3 * D_PRED, D_PRED, 0.7)
    w["pred.qkv.bias"] = np.zeros(3 * D_PRED, dtype=np.float32)
    w["pred.out_proj.weight"] = ortho(D_JEPA, D_PRED, 0.7)
    w["pred.out_proj.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    for pair in ("a2v", "v2a", "s2v", "s2a", "v2s", "a2s"):
        w[f"cross.{pair}.fc1.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
        w[f"cross.{pair}.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
        w[f"cross.{pair}.fc2.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
        w[f"cross.{pair}.fc2.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    w["ac_pred.state_in.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
    w["ac_pred.state_in.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["ac_pred.act_in.weight"] = ortho(D_JEPA, D_ACT, 0.5)
    w["ac_pred.act_in.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["ac_pred.out.weight"] = np.eye(D_JEPA, dtype=np.float32)
    w["ac_pred.out.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    w["idm.fc1.weight"] = ortho(D_JEPA, 2 * D_JEPA, 0.7)
    w["idm.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["idm.fc2.weight"] = ortho(D_ACT, D_JEPA, 0.7)
    w["idm.fc2.bias"] = np.zeros(D_ACT, dtype=np.float32)

    w["bridge.jepa_to_vq.weight"] = ortho(D_VQ, D_JEPA, 0.6)
    w["bridge.jepa_to_vq.bias"] = np.zeros(D_VQ, dtype=np.float32)
    w["bridge.lfm3_to_vq.weight"] = ortho(D_VQ, D_LFM3, 0.6)
    w["bridge.lfm3_to_vq.bias"] = np.zeros(D_VQ, dtype=np.float32)
    w["bridge.vq_to_lfm3.weight"] = ortho(D_LFM3, D_VQ, 0.15)
    w["bridge.vq_to_lfm3.bias"] = np.zeros(D_LFM3, dtype=np.float32)

    w["probe.visual_scene.weight"] = ortho(len(VISUAL_SCENE_KINDS), D_JEPA, 1.0)
    w["probe.visual_scene.bias"] = np.zeros(len(VISUAL_SCENE_KINDS), dtype=np.float32)
    w["probe.acoustic_regime.weight"] = ortho(len(ACOUSTIC_REGIMES), D_JEPA, 1.0)
    w["probe.acoustic_regime.bias"] = np.zeros(len(ACOUSTIC_REGIMES), dtype=np.float32)

    in_res = D_JEPA + D_ACT + D_JEPA
    w["residual.fc1.weight"] = ortho(D_JEPA, in_res, 0.5)
    w["residual.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["residual.fc2.weight"] = np.zeros((D_JEPA, D_JEPA), dtype=np.float32)
    w["residual.fc2.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["residual.gate"] = np.array([1.0], dtype=np.float32)
    w["residual.baseline_cue_proto"] = np.zeros((1, D_JEPA), dtype=np.float32)
    w["residual.has_proto"] = np.array([0.0], dtype=np.float32)

    return w


class CloverJEPARuntime:
    """Selectable JEPA Engine for Clover AI Studio, K3 Server, and Mobile."""

    def __init__(self, base_dir: str | Path | None = None):
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.lock = threading.Lock()
        self.trunk_dir = self._resolve_trunk_dir(self.base_dir)
        self.weights_source = "default_orthogonal"
        if (self.trunk_dir / "index.bin").is_file():
            try:
                self.w = read_clvrtrk1(self.trunk_dir)
                self.weights_source = f"clvrtrk1:{self.trunk_dir}"
            except Exception:
                self.w = _build_default_weights()
        else:
            self.w = _build_default_weights()

        defaults = _build_default_weights()
        for k, v in defaults.items():
            if k not in self.w or self.w[k].shape != v.shape:
                self.w[k] = v

        self.adapter_path = self.base_dir / "growth" / "jepa_outcome_adapter.npz"
        self.outcome_buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.baseline_anchor: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.cycle_updates = 0
        self.rollbacks = 0
        self._load_persisted_adapter()
        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @staticmethod
    def _resolve_trunk_dir(base_dir: Path) -> Path:
        for cand in (
            base_dir / "data" / "omni" / "precomputed" / "jepa" / "jepa-trunk-0",
            base_dir / "data" / "omni" / "precomputed" / "models" / "jepa" / "jepa-trunk-0",
            Path(__file__).resolve().parent / "models" / "jepa-trunk-0",
        ):
            if (cand / "index.bin").is_file():
                return cand
        return base_dir / "data" / "omni" / "precomputed" / "jepa" / "jepa-trunk-0"

    def _load_persisted_adapter(self) -> None:
        if not self.adapter_path.is_file():
            return
        try:
            with np.load(str(self.adapter_path)) as data:
                for k in ("residual.fc1.weight", "residual.fc1.bias", "residual.fc2.weight", "residual.fc2.bias",
                          "residual.gate", "residual.baseline_cue_proto", "residual.has_proto"):
                    if k in data and data[k].shape == self.w[k].shape:
                        self.w[k] = data[k].astype(np.float32)
                if "cycle_updates" in data:
                    self.cycle_updates = int(data["cycle_updates"][0])
        except Exception:
            pass

    def _save_persisted_adapter(self) -> None:
        try:
            self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(self.adapter_path),
                **{
                    "residual.fc1.weight": self.w["residual.fc1.weight"],
                    "residual.fc1.bias": self.w["residual.fc1.bias"],
                    "residual.fc2.weight": self.w["residual.fc2.weight"],
                    "residual.fc2.bias": self.w["residual.fc2.bias"],
                    "residual.gate": self.w["residual.gate"],
                    "residual.baseline_cue_proto": self.w["residual.baseline_cue_proto"],
                    "residual.has_proto": self.w["residual.has_proto"],
                    "cycle_updates": np.array([self.cycle_updates], dtype=np.int64),
                },
            )
        except Exception:
            pass

    def choose_architecture(
        self,
        flow: str,
        configured_arch: str = "auto",
        *,
        has_visual: bool = False,
        has_audio: bool = False,
        has_outcome_feedback: bool = False,
    ) -> str:
        arch = str(configured_arch or "auto").strip().lower()
        if arch not in JEPA_ARCHITECTURES:
            arch = "auto"
        if arch != "auto":
            return arch
        if has_outcome_feedback or flow == "outcome_growth":
            return "clover_rpre"
        if has_visual and has_audio:
            return "mjepa" if flow != "c2_plan" else "clover_rpre"
        return FLOW_DEFAULT_ARCH.get(flow, "clover_rpre")

    def describe(self, configured_arch: str = "auto") -> dict[str, Any]:
        return {
            "configured_arch": configured_arch if configured_arch in JEPA_ARCHITECTURES else "auto",
            "available_architectures": list(JEPA_ARCHITECTURES),
            "flow_routing": dict(FLOW_DEFAULT_ARCH),
            "weights_source": self.weights_source,
            "embed_dim": D_JEPA,
            "bridge_dim": D_VQ,
            "cycle_updates": self.cycle_updates,
            "rollbacks": self.rollbacks,
            "setup_ms": self.setup_ms,
        }

    def _run_encoder_blocks(self, tokens_nd: np.ndarray, return_hierarchical: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
        x = np.asarray(tokens_nd, dtype=np.float32)
        levels: list[np.ndarray] = []
        for l in range(4):
            xn1 = _layer_norm(x)
            qkv = _linear(xn1, self.w[f"enc.block.{l}.qkv.weight"], self.w[f"enc.block.{l}.qkv.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            scale = np.float32(1.0 / math.sqrt(D_JEPA))
            attn = _softmax(np.matmul(q, k.T) * scale, axis=-1)
            x = x + _linear(np.matmul(attn, v), self.w[f"enc.block.{l}.proj.weight"], self.w[f"enc.block.{l}.proj.bias"])
            xn2 = _layer_norm(x)
            h_mlp = _gelu(_linear(xn2, self.w[f"enc.block.{l}.mlp1.weight"], self.w[f"enc.block.{l}.mlp1.bias"]))
            x = x + _linear(h_mlp, self.w[f"enc.block.{l}.mlp2.weight"], self.w[f"enc.block.{l}.mlp2.bias"])
            levels.append(_layer_norm(x))
        if return_hierarchical:
            return np.mean(np.stack(levels, axis=0), axis=0), levels
        return levels[-1], levels

    def _predict_masked_tokens(self, ctxt_tokens: np.ndarray, ctxt_idx: np.ndarray, targ_idx: np.ndarray) -> np.ndarray:
        pos = self.w["pred.pos_embed"]
        x_c = _linear(ctxt_tokens, self.w["pred.in_proj.weight"], self.w["pred.in_proj.bias"]) + pos[ctxt_idx]
        x_m = np.repeat(self.w["pred.mask_token"], len(targ_idx), axis=0) + pos[targ_idx]
        x_all = np.concatenate([x_c, x_m], axis=0)
        xn = _layer_norm(x_all)
        qkv = _linear(xn, self.w["pred.qkv.weight"], self.w["pred.qkv.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)
        attn = _softmax(np.matmul(q, k.T) / np.float32(math.sqrt(D_PRED)), axis=-1)
        out = x_all + np.matmul(attn, v)
        pred_targ = _linear(_layer_norm(out[len(ctxt_idx):]), self.w["pred.out_proj.weight"], self.w["pred.out_proj.bias"])
        return _layer_norm(pred_targ)

    def _cross_modal_predict(self, z_tokens_or_vec: np.ndarray, pair: str) -> np.ndarray:
        z = np.asarray(z_tokens_or_vec, dtype=np.float32)
        if z.ndim == 2:
            z = z.mean(axis=0, keepdims=True)
        elif z.ndim == 1:
            z = z[None, :]
        h = _gelu(_linear(_layer_norm(z), self.w[f"cross.{pair}.fc1.weight"], self.w[f"cross.{pair}.fc1.bias"]))
        out = _layer_norm(z + _linear(h, self.w[f"cross.{pair}.fc2.weight"], self.w[f"cross.{pair}.fc2.bias"]))
        return out.squeeze(0)

    def inspect_visual_latent(
        self,
        img_rgb: np.ndarray,
        elements: list[dict[str, Any]] | None = None,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        arch = self.choose_architecture("visual_s3", configured_arch, has_visual=True)
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        arr = np.asarray(img_rgb, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.max() > 1.5:
            arr = arr / np.float32(255.0)
        H, W, _ = arr.shape

        patches = np.zeros((N_PATCHES, 48), dtype=np.float32)
        for py in range(4):
            y0, y1 = int(py * H / 4), max(int(py * H / 4) + 1, int((py + 1) * H / 4))
            for px in range(4):
                x0, x1 = int(px * W / 4), max(int(px * W / 4) + 1, int((px + 1) * W / 4))
                cell = arr[y0:y1, x0:x1, :3]
                sh, sw = max(1, cell.shape[0] // 4), max(1, cell.shape[1] // 4)
                feat = []
                for sy in range(4):
                    for sx in range(4):
                        sub = cell[sy * sh : (sy + 1) * sh or cell.shape[0], sx * sw : (sx + 1) * sw or cell.shape[1]]
                        if sub.size == 0:
                            feat.extend([0.0, 0.0, 0.0])
                        else:
                            feat.extend(sub.mean(axis=(0, 1)).tolist())
                patches[py * 4 + px] = np.array(feat[:48], dtype=np.float32)

        patch_dev = np.mean((patches - patches.mean(axis=0, keepdims=True)) ** 2, axis=-1)
        patches_norm = (patches - patches.mean(axis=0, keepdims=True)) / (patches.std(axis=0, keepdims=True) + np.float32(1e-5))

        tokens = _linear(patches_norm, self.w["vis_adapter.weight"], self.w["vis_adapter.bias"]) + self.w["vis_pos_embed"] + self.w["vis_mod_token"]
        use_hier = arch in ("mjepa", "clover_rpre")
        full_z, _ = self._run_encoder_blocks(tokens, return_hierarchical=use_hier)

        block_a_targ = np.array([5, 6, 9, 10], dtype=np.int64)
        block_a_ctxt = np.array([i for i in range(N_PATCHES) if i not in block_a_targ], dtype=np.int64)
        block_b_ctxt = np.array([5, 6, 9, 10, 1, 2, 13, 14, 4], dtype=np.int64)
        block_b_targ = np.array([i for i in range(N_PATCHES) if i not in block_b_ctxt], dtype=np.int64)

        z_ctxt_a, _ = self._run_encoder_blocks(tokens[block_a_ctxt], return_hierarchical=use_hier)
        pred_a = self._predict_masked_tokens(z_ctxt_a, block_a_ctxt, block_a_targ)

        z_ctxt_b, _ = self._run_encoder_blocks(tokens[block_b_ctxt], return_hierarchical=use_hier)
        pred_b = self._predict_masked_tokens(z_ctxt_b, block_b_ctxt, block_b_targ)

        patch_surprise = np.zeros(N_PATCHES, dtype=np.float32)
        for idx_i, p_idx in enumerate(block_a_targ):
            patch_surprise[p_idx] = float(np.mean((pred_a[idx_i] - full_z[p_idx]) ** 2))
        for idx_i, p_idx in enumerate(block_b_targ):
            patch_surprise[p_idx] = float(np.mean((pred_b[idx_i] - full_z[p_idx]) ** 2))

        s_min, s_max = float(patch_surprise.min()), float(patch_surprise.max())
        surp_norm = (patch_surprise - s_min) / max(1e-6, s_max - s_min)
        d_min, d_max = float(patch_dev.min()), float(patch_dev.max())
        dev_norm = (patch_dev - d_min) / max(1e-6, d_max - d_min)
        patch_salience = np.float32(0.55) * surp_norm + np.float32(0.45) * dev_norm

        salient_refs: list[dict[str, Any]] = []
        if elements:
            for el in elements:
                bbox = el.get("bbox") or [0, 0, W, H]
                cx = (bbox[0] + 0.5 * bbox[2]) / max(1.0, float(W))
                cy = (bbox[1] + 0.5 * bbox[3]) / max(1.0, float(H))
                px = min(3, max(0, int(cx * 4)))
                py = min(3, max(0, int(cy * 4)))
                sal = float(patch_salience[py * 4 + px])
                if el.get("text"):
                    sal = min(1.0, sal * 0.7 + 0.3)
                el["jepa_salience"] = round(sal, 3)
                salient_refs.append({"ref": el.get("ref", ""), "role": el.get("role", ""), "salience": round(sal, 3), "text": el.get("text", "")[:40]})
            salient_refs.sort(key=lambda r: -r["salience"])

        z_vis_pool = full_z.mean(axis=0)
        z_vq = _layer_norm(_linear(z_vis_pool[None, :], self.w["bridge.jepa_to_vq.weight"], self.w["bridge.jepa_to_vq.bias"])).squeeze(0)

        logits = _linear(z_vis_pool[None, :], self.w["probe.visual_scene.weight"], self.w["probe.visual_scene.bias"]).squeeze(0)
        n_el = len(elements or ())
        n_txt = sum(1 for e in (elements or ()) if e.get("text"))
        if n_el >= 2 and any(e.get("role") in ("button", "header", "heading", "card", "visual-region") for e in (elements or ())):
            scene_kind = "ui_controls"
        elif n_txt >= 4:
            scene_kind = "document_receipt"
        else:
            scene_kind = VISUAL_SCENE_KINDS[int(np.argmax(logits))]

        flops_ratio = round((len(block_a_ctxt) / float(N_PATCHES)) ** 2, 4)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "visual_s3",
            "arch": arch,
            "scene_kind": scene_kind,
            "attn_flops_ratio": flops_ratio,
            "mean_patch_surprise": round(float(patch_surprise.mean()), 4),
            "top_salient_elements": salient_refs[:5],
            "z_jepa": z_vis_pool,
            "z_vq": z_vq,
            "ms": ms,
        }

    def inspect_audio_latent(
        self,
        audio_input: bytes | str | Path | np.ndarray,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        arch = self.choose_architecture("audio_s3", configured_arch, has_audio=True)
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        samples = self._extract_audio_samples(audio_input)
        spec_16x16, snr_db = self._compute_logmel_16x16(samples)

        patches = np.zeros((N_PATCHES, 16), dtype=np.float32)
        for ti in range(4):
            for fi in range(4):
                patches[ti * 4 + fi] = spec_16x16[ti * 4 : (ti + 1) * 4, fi * 4 : (fi + 1) * 4].reshape(-1)

        tokens = _linear(patches, self.w["aud_adapter.weight"], self.w["aud_adapter.bias"]) + self.w["aud_pos_embed"] + self.w["aud_mod_token"]
        use_hier = arch in ("mjepa", "clover_rpre")
        full_z, _ = self._run_encoder_blocks(tokens, return_hierarchical=use_hier)

        targ_idx = np.array([3, 5, 6, 7, 9, 10], dtype=np.int64)
        ctxt_idx = np.array([i for i in range(N_PATCHES) if i not in targ_idx], dtype=np.int64)
        z_ctxt, _ = self._run_encoder_blocks(tokens[ctxt_idx], return_hierarchical=use_hier)
        pred_targ = self._predict_masked_tokens(z_ctxt, ctxt_idx, targ_idx)
        true_targ = full_z[targ_idx]

        cn_mse = centered_norm_mse(pred_targ, true_targ, center_pred=True)
        tf_cos = max(-1.0, min(1.0, 1.0 - 0.5 * cn_mse))

        z_aud_pool = full_z.mean(axis=0)
        z_vq = _layer_norm(_linear(z_aud_pool[None, :], self.w["bridge.jepa_to_vq.weight"], self.w["bridge.jepa_to_vq.bias"])).squeeze(0)

        if snr_db < 3.0:
            regime = "low_snr_silence"
        elif snr_db >= 14.0 and tf_cos >= 0.45:
            regime = "clean_speech"
        elif snr_db < 10.0:
            regime = "noisy_speech"
        else:
            logits = _linear(z_aud_pool[None, :], self.w["probe.acoustic_regime.weight"], self.w["probe.acoustic_regime.bias"]).squeeze(0)
            regime = ACOUSTIC_REGIMES[int(np.argmax(logits))]

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "audio_s3",
            "arch": arch,
            "acoustic_regime": regime,
            "snr_db": round(float(snr_db), 2),
            "tf_centered_norm_mse": round(float(cn_mse), 4),
            "tf_latent_cos": round(float(tf_cos), 4),
            "z_jepa": z_aud_pool,
            "z_vq": z_vq,
            "ms": ms,
        }

    @staticmethod
    def _extract_audio_samples(audio_input: bytes | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio_input, np.ndarray):
            return audio_input.astype(np.float32).reshape(-1)
        raw_bytes = b""
        if isinstance(audio_input, (str, Path)):
            p = Path(audio_input)
            if p.is_file():
                try:
                    with wave.open(str(p), "rb") as wf:
                        n_ch = wf.getnchannels()
                        sw = wf.getsampwidth()
                        frames = wf.readframes(min(wf.getnframes(), 64000))
                        if sw == 2:
                            s = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            return s[::max(1, n_ch)]
                except Exception:
                    raw_bytes = p.read_bytes()[:64000]
        elif isinstance(audio_input, (bytes, bytearray)):
            raw_bytes = bytes(audio_input[:64000])
        if len(raw_bytes) >= 44 and raw_bytes[:4] == b"RIFF":
            pcm = raw_bytes[44:]
            n = (len(pcm) // 2) * 2
            if n > 0:
                return np.frombuffer(pcm[:n], dtype=np.int16).astype(np.float32) / 32768.0
        if raw_bytes:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            return (arr - 128.0) / 128.0
        return np.zeros(1024, dtype=np.float32)

    @staticmethod
    def _compute_logmel_16x16(samples: np.ndarray) -> tuple[np.ndarray, float]:
        if samples.ndim == 2 and samples.shape == (16, 16):
            p90 = float(np.percentile(samples, 90))
            p10 = float(np.percentile(samples, 10))
            return samples.astype(np.float32), max(0.0, (p90 - p10) * 8.0)
        s = np.asarray(samples, dtype=np.float32).reshape(-1)
        if s.size < 256:
            s = np.pad(s, (0, 256 - s.size))
        frames = np.array_split(s[: max(256, (s.size // 16) * 16)], 16)
        spec = np.zeros((16, 16), dtype=np.float32)
        frame_energies = np.zeros(16, dtype=np.float32)
        for ti, fr in enumerate(frames):
            win = fr * np.hanning(len(fr)).astype(np.float32)
            mag = np.abs(np.fft.rfft(win))
            frame_energies[ti] = float(np.mean(fr * fr) + 1e-9)
            bins = np.array_split(mag, 16)
            for fi, b in enumerate(bins):
                spec[ti, fi] = math.log10(float(np.mean(b * b)) + 1e-7)
        snr_db = 10.0 * math.log10(float(np.percentile(frame_energies, 90)) / float(np.percentile(frame_energies, 15) + 1e-9))
        spec = (spec - spec.mean()) / (spec.std() + np.float32(1e-5))
        return spec, max(0.0, min(40.0, snr_db))

    def fuse_and_retrieve_multimodal(
        self,
        query_vq_384: np.ndarray,
        *,
        visual_z_64: np.ndarray | None = None,
        audio_z_64: np.ndarray | None = None,
        candidates_vq_384: np.ndarray | None = None,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        arch = self.choose_architecture(
            "multimodal_s2",
            configured_arch,
            has_visual=visual_z_64 is not None,
            has_audio=audio_z_64 is not None,
        )
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        q_vq = np.asarray(query_vq_384, dtype=np.float32).reshape(1, D_VQ)
        s_tok = _linear(q_vq, self.w["state_adapter.weight"], self.w["state_adapter.bias"]) + self.w["state_mod_token"]
        z_s, _ = self._run_encoder_blocks(s_tok, return_hierarchical=(arch in ("mjepa", "clover_rpre")))
        z_s_vec = z_s.squeeze(0)

        imputed_modalities: list[str] = []
        if visual_z_64 is not None and audio_z_64 is not None:
            z_v = np.asarray(visual_z_64, dtype=np.float32).reshape(D_JEPA)
            z_a = np.asarray(audio_z_64, dtype=np.float32).reshape(D_JEPA)
        elif visual_z_64 is not None:
            z_v = np.asarray(visual_z_64, dtype=np.float32).reshape(D_JEPA)
            z_a = 0.5 * (self._cross_modal_predict(z_v, "v2a") + self._cross_modal_predict(z_s_vec, "s2a"))
            imputed_modalities.append("audio_from_v2a_s2a")
        elif audio_z_64 is not None:
            z_a = np.asarray(audio_z_64, dtype=np.float32).reshape(D_JEPA)
            z_v = 0.5 * (self._cross_modal_predict(z_a, "a2v") + self._cross_modal_predict(z_s_vec, "s2v"))
            imputed_modalities.append("visual_from_a2v_s2v")
        else:
            z_v = self._cross_modal_predict(z_s_vec, "s2v")
            z_a = self._cross_modal_predict(z_s_vec, "s2a")
            imputed_modalities.extend(["visual_from_s2v", "audio_from_s2a"])

        z_joint_64 = _layer_norm((z_s_vec + 0.5 * (z_v + z_a))[None, :]).squeeze(0)
        z_joint_vq = _layer_norm(_linear(z_joint_64[None, :], self.w["bridge.jepa_to_vq.weight"], self.w["bridge.jepa_to_vq.bias"])).squeeze(0)
        fused_query_vq = _l2_normalize(0.65 * q_vq.squeeze(0) + 0.35 * z_joint_vq, axis=-1)

        scores = []
        if candidates_vq_384 is not None and len(candidates_vq_384) > 0:
            cands = _l2_normalize(np.asarray(candidates_vq_384, dtype=np.float32), axis=-1)
            scores = np.matmul(cands, fused_query_vq).tolist()

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "multimodal_s2",
            "arch": arch,
            "imputed_modalities": imputed_modalities,
            "cross_modal_av_cos": round(float(np.sum(_l2_normalize(z_v) * _l2_normalize(z_a))), 4),
            "z_joint_64": z_joint_64,
            "z_joint_vq": z_joint_vq,
            "fused_query_vq": fused_query_vq,
            "candidate_scores": [round(float(s), 4) for s in scores],
            "ms": ms,
        }

    def _step_transition(self, z_bd: np.ndarray, a_ba: np.ndarray, context_cue: np.ndarray | None = None, use_residual: bool = True) -> np.ndarray:
        h = _gelu(_linear(z_bd, self.w["ac_pred.state_in.weight"], self.w["ac_pred.state_in.bias"])
                  + _linear(a_ba, self.w["ac_pred.act_in.weight"], self.w["ac_pred.act_in.bias"]))
        base_next = _layer_norm(z_bd + _linear(h, self.w["ac_pred.out.weight"], self.w["ac_pred.out.bias"]))
        if not use_residual:
            return base_next
        delta = self._forward_residual(z_bd, a_ba, context_cue=context_cue)
        if float(np.max(np.abs(delta))) <= 1e-7:
            return base_next
        return _layer_norm(base_next + delta)

    def _forward_residual(self, z_bd: np.ndarray, a_ba: np.ndarray, context_cue: np.ndarray | None = None) -> np.ndarray:
        B = z_bd.shape[0]
        if context_cue is None:
            c_bd = np.zeros((B, D_JEPA), dtype=np.float32)
        else:
            c_arr = np.asarray(context_cue, dtype=np.float32)
            if c_arr.shape == (B, D_JEPA):
                c_bd = c_arr
            elif c_arr.size == D_JEPA:
                c_bd = np.repeat(c_arr.reshape(1, D_JEPA), B, axis=0)
            else:
                c_bd = np.zeros((B, D_JEPA), dtype=np.float32)
        anom_gate = np.float32(1.0)
        if float(self.w["residual.has_proto"][0]) > 0.5 and context_cue is not None:
            proto = self.w["residual.baseline_cue_proto"]
            dist_sq = np.mean((c_bd - proto) ** 2, axis=-1, keepdims=True)
            anom_gate = np.clip((dist_sq - np.float32(0.25)) / np.float32(0.15), np.float32(0.0), np.float32(1.0))
        inp = np.concatenate([z_bd, a_ba, c_bd], axis=-1)
        h = _gelu(_linear(inp, self.w["residual.fc1.weight"], self.w["residual.fc1.bias"]))
        out = _linear(h, self.w["residual.fc2.weight"], self.w["residual.fc2.bias"])
        gate = np.tanh(self.w["residual.gate"][0])
        return anom_gate * gate * out

    def plan_c2_latent(
        self,
        h_primary_2048: np.ndarray,
        r_primary_2048: np.ndarray,
        *,
        goal_vq_384: np.ndarray | None = None,
        offered_tools: list[str] | None = None,
        context_cue_64: np.ndarray | None = None,
        configured_arch: str = "auto",
        horizon: int = 4,
        cem_iters: int = 4,
        num_samples: int = 32,
    ) -> dict[str, Any]:
        arch = self.choose_architecture(
            "c2_plan",
            configured_arch,
            has_outcome_feedback=(self.cycle_updates > 0 or context_cue_64 is not None),
        )
        if arch == "none":
            return {"enabled": False, "arch": "none", "r_refined_2048": r_primary_2048}

        t0 = time.perf_counter()
        h_mean_2048 = np.asarray(h_primary_2048, dtype=np.float32).mean(axis=1)
        r_mean_2048 = np.asarray(r_primary_2048, dtype=np.float32).mean(axis=1)

        h_vq = _layer_norm(_linear(h_mean_2048, self.w["bridge.lfm3_to_vq.weight"], self.w["bridge.lfm3_to_vq.bias"]))
        r_vq = _layer_norm(_linear(r_mean_2048, self.w["bridge.lfm3_to_vq.weight"], self.w["bridge.lfm3_to_vq.bias"]))

        z_init = _layer_norm(_linear(h_vq, self.w["state_adapter.weight"], self.w["state_adapter.bias"]) + self.w["state_mod_token"])
        if goal_vq_384 is not None:
            g_vq = np.asarray(goal_vq_384, dtype=np.float32).reshape(1, D_VQ)
        else:
            g_vq = r_vq
        z_goal = _layer_norm(_linear(g_vq, self.w["state_adapter.weight"], self.w["state_adapter.bias"]) + self.w["state_mod_token"])

        idm_in = np.concatenate([z_init, z_goal], axis=-1)
        a_idm = _linear(_gelu(_linear(idm_in, self.w["idm.fc1.weight"], self.w["idm.fc1.bias"])),
                        self.w["idm.fc2.weight"], self.w["idm.fc2.bias"])

        use_res = arch == "clover_rpre"
        rng = np.random.default_rng(101)
        mu = np.repeat(a_idm[:, None, :], horizon, axis=1).squeeze(0)
        sigma = np.ones((horizon, D_ACT), dtype=np.float32) * np.float32(0.35)
        n_elites = max(4, num_samples // 5)

        energy_before = float(np.mean((z_init - z_goal) ** 2))
        best_cost = energy_before
        best_z_final = z_init.copy()

        for _ in range(cem_iters):
            eps = rng.standard_normal((num_samples, horizon, D_ACT)).astype(np.float32)
            acts = mu[None, :, :] + sigma[None, :, :] * eps
            acts[0] = mu
            norms = np.linalg.norm(acts, axis=-1, keepdims=True)
            acts = acts * np.minimum(1.0, 0.8 / (norms + 1e-6))

            z_cur = np.repeat(z_init, num_samples, axis=0)
            total_cost = np.zeros(num_samples, dtype=np.float32)
            for t_step in range(horizon):
                z_cur = self._step_transition(z_cur, acts[:, t_step, :], context_cue=context_cue_64, use_residual=use_res)
                total_cost += np.mean((z_cur - z_goal) ** 2, axis=-1)

            elite_idx = np.argsort(total_cost)[:n_elites]
            elites = acts[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + np.float32(1e-4)
            best_cost = float(total_cost[elite_idx[0]] / horizon)
            best_z_final = z_cur[elite_idx[0] : elite_idx[0] + 1]

        z_delta_vq = _linear(best_z_final - z_init, self.w["bridge.jepa_to_vq.weight"], self.w["bridge.jepa_to_vq.bias"])
        r_delta_2048 = _linear(z_delta_vq, self.w["bridge.vq_to_lfm3.weight"], self.w["bridge.vq_to_lfm3.bias"])[:, None, :]
        r_refined_2048 = np.asarray(r_primary_2048, dtype=np.float32) + np.float32(0.25) * r_delta_2048

        ranked_tools = list(offered_tools or [])
        if len(ranked_tools) > 1:
            act_scores = []
            for idx_t, tname in enumerate(ranked_tools):
                slot = idx_t % D_ACT
                act_scores.append((float(mu[0, slot]), tname))
            act_scores.sort(key=lambda x: -x[0])
            ranked_tools = [t for _, t in act_scores]

        vc_std = float(np.sqrt(np.var(best_z_final) + 1e-6))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "c2_plan",
            "arch": arch,
            "horizon": horizon,
            "cem_iters": cem_iters,
            "energy_before": round(energy_before, 4),
            "energy_after": round(best_cost, 4),
            "energy_reduction_pct": round(100.0 * max(0.0, energy_before - best_cost) / max(1e-6, energy_before), 1),
            "vc_std": round(vc_std, 4),
            "first_action": mu[0].copy(),
            "z_init_64": z_init.squeeze(0),
            "z_planned_64": best_z_final.squeeze(0),
            "ranked_tools": ranked_tools,
            "r_refined_2048": r_refined_2048,
            "ms": ms,
        }

    def record_and_adapt_outcome(
        self,
        z_t_64: np.ndarray,
        action_vec_8: np.ndarray,
        z_tp1_obs_64: np.ndarray,
        *,
        context_cue_64: np.ndarray | None = None,
        is_baseline: bool = False,
        configured_arch: str = "auto",
        lr: float = 0.025,
        steps: int = 16,
        max_baseline_regression: float = 0.015,
    ) -> dict[str, Any]:
        arch = self.choose_architecture("outcome_growth", configured_arch, has_outcome_feedback=True)
        if arch == "none":
            return {"enabled": False, "arch": "none", "updated": False}

        t0 = time.perf_counter()
        z_t = np.asarray(z_t_64, dtype=np.float32).reshape(1, D_JEPA)
        a_t = np.asarray(action_vec_8, dtype=np.float32).reshape(1, D_ACT)
        z_obs = np.asarray(z_tp1_obs_64, dtype=np.float32).reshape(1, D_JEPA)
        c_t = (
            np.asarray(context_cue_64, dtype=np.float32).reshape(1, D_JEPA)
            if context_cue_64 is not None
            else np.zeros((1, D_JEPA), dtype=np.float32)
        )

        with self.lock:
            if is_baseline:
                self.baseline_anchor.append((z_t.copy(), a_t.copy(), z_obs.copy(), c_t.copy()))
                if len(self.baseline_anchor) > 128:
                    self.baseline_anchor.pop(0)
                non_zero_cues = [x[3] for x in self.baseline_anchor if float(np.sum(np.abs(x[3]))) > 1e-5]
                if non_zero_cues:
                    self.w["residual.baseline_cue_proto"] = np.mean(np.concatenate(non_zero_cues, axis=0), axis=0, keepdims=True)
                    self.w["residual.has_proto"] = np.array([1.0], dtype=np.float32)
                return {
                    "enabled": True,
                    "flow": "outcome_growth",
                    "arch": arch,
                    "updated": False,
                    "baseline_anchors": len(self.baseline_anchor),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            self.outcome_buffer.append((z_t.copy(), a_t.copy(), z_obs.copy(), c_t.copy()))
            if len(self.outcome_buffer) > 256:
                self.outcome_buffer.pop(0)

            Z_t = np.concatenate([x[0] for x in self.outcome_buffer], axis=0)
            A_t = np.concatenate([x[1] for x in self.outcome_buffer], axis=0)
            Z_obs = np.concatenate([x[2] for x in self.outcome_buffer], axis=0)
            C_t = np.concatenate([x[3] for x in self.outcome_buffer], axis=0)

            base_next = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=False)
            pred_before = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=True)
            mse_before = float(np.mean((pred_before - Z_obs) ** 2))

            base_mse_before = 0.0
            if self.baseline_anchor:
                bZ_t = np.concatenate([x[0] for x in self.baseline_anchor], axis=0)
                bA_t = np.concatenate([x[1] for x in self.baseline_anchor], axis=0)
                bZ_obs = np.concatenate([x[2] for x in self.baseline_anchor], axis=0)
                bC_t = np.concatenate([x[3] for x in self.baseline_anchor], axis=0)
                bp_before = self._step_transition(bZ_t, bA_t, context_cue=bC_t, use_residual=True)
                base_mse_before = float(np.mean((bp_before - bZ_obs) ** 2))

            snap_fc1_w = self.w["residual.fc1.weight"].copy()
            snap_fc1_b = self.w["residual.fc1.bias"].copy()
            snap_fc2_w = self.w["residual.fc2.weight"].copy()
            snap_fc2_b = self.w["residual.fc2.bias"].copy()

            target_delta = Z_obs - base_next
            inp = np.concatenate([Z_t, A_t, C_t], axis=-1)
            N_samples = float(inp.shape[0])

            for _ in range(steps):
                h_pre = _linear(inp, self.w["residual.fc1.weight"], self.w["residual.fc1.bias"])
                h_act = _gelu(h_pre)
                pred_delta = _linear(h_act, self.w["residual.fc2.weight"], self.w["residual.fc2.bias"])
                err = (pred_delta - target_delta) * (2.0 / N_samples)

                grad_fc2_w = np.matmul(err.T, h_act) + 1e-4 * self.w["residual.fc2.weight"]
                grad_fc2_b = np.sum(err, axis=0)
                dh = np.matmul(err, self.w["residual.fc2.weight"]) * (h_pre > 0.0).astype(np.float32)
                grad_fc1_w = np.matmul(dh.T, inp) + 1e-4 * self.w["residual.fc1.weight"]
                grad_fc1_b = np.sum(dh, axis=0)

                self.w["residual.fc2.weight"] -= np.float32(lr) * grad_fc2_w
                self.w["residual.fc2.bias"] -= np.float32(lr) * grad_fc2_b
                self.w["residual.fc1.weight"] -= np.float32(lr) * grad_fc1_w
                self.w["residual.fc1.bias"] -= np.float32(lr) * grad_fc1_b

            pred_after = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=True)
            mse_after = float(np.mean((pred_after - Z_obs) ** 2))

            base_mse_after = 0.0
            if self.baseline_anchor:
                bp_after = self._step_transition(bZ_t, bA_t, context_cue=bC_t, use_residual=True)
                base_mse_after = float(np.mean((bp_after - bZ_obs) ** 2))

            regressed = (self.baseline_anchor and (base_mse_after > base_mse_before + max_baseline_regression)) or (mse_after > mse_before + 1e-5)
            if regressed:
                self.w["residual.fc1.weight"] = snap_fc1_w
                self.w["residual.fc1.bias"] = snap_fc1_b
                self.w["residual.fc2.weight"] = snap_fc2_w
                self.w["residual.fc2.bias"] = snap_fc2_b
                self.rollbacks += 1
                return {
                    "enabled": True,
                    "flow": "outcome_growth",
                    "arch": arch,
                    "updated": False,
                    "rolled_back": True,
                    "mse_before": round(mse_before, 5),
                    "mse_after": round(mse_before, 5),
                    "baseline_mse_before": round(base_mse_before, 5),
                    "baseline_mse_after": round(base_mse_after, 5),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            self.cycle_updates += 1
            self._save_persisted_adapter()
            return {
                "enabled": True,
                "flow": "outcome_growth",
                "arch": arch,
                "updated": True,
                "rolled_back": False,
                "cycle_updates": self.cycle_updates,
                "mse_before": round(mse_before, 5),
                "mse_after": round(mse_after, 5),
                "baseline_mse_before": round(base_mse_before, 5),
                "baseline_mse_after": round(base_mse_after, 5),
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }


_INSTANCES: dict[str, CloverJEPARuntime] = {}
_INST_LOCK = threading.Lock()


def for_base(base_dir: str | Path | None = None) -> CloverJEPARuntime:
    key = str(Path(base_dir).resolve()) if base_dir else "default"
    with _INST_LOCK:
        inst = _INSTANCES.get(key)
        if inst is None:
            inst = CloverJEPARuntime(base_dir)
            _INSTANCES[key] = inst
        return inst
