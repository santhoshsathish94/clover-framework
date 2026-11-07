"""PyTorch CPU/GPU Training & `CLVRTRK1` Binary Trunk Exporter for the Clover JEPA Engine (`train_clover_jepa_trunks.py`).

Trains the unified Clover JEPA trunk (`jepa-trunk-0`) supporting all 5 selectable architectures (`ijepa`, `audio_jepa`,
`eb_jepa`, `mjepa`, `clover_rpre`) and exports the trained weights directly into Clover's `CLVRTRK1` binary format
(`index.bin`, `common.bin`, `operator.bin`, `ffn.bin`) for `clover/jepa_engine.py`.

Usage:
  - CPU / k3 pilot training (default):
      python scripts/train_clover_jepa_trunks.py --device auto --scale pilot
  - Full GPU training (when GPU server is switched on):
      python scripts/train_clover_jepa_trunks.py --device cuda --scale full --epochs 120 --batch-size 256
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
_CAND_A = ROOT.parent.parent / "clover-ai-system" / "studio" / "clover"
_CAND_B = ROOT / "clover-ai-system" / "studio" / "clover"
STUDIO_CLOVER = _CAND_A if (_CAND_A / "jepa_engine.py").is_file() else _CAND_B
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDIO_CLOVER))

from jepa_engine import (  # noqa: E402
    ACOUSTIC_REGIMES,
    D_ACT,
    D_JEPA,
    D_LFM3,
    D_PRED,
    D_VQ,
    N_PATCHES,
    VISUAL_SCENE_KINDS,
    write_clvrtrk1,
)
from rpre.losses import CenteredNormMSELoss, HingeStdLoss, CovarianceLoss  # noqa: E402


class TorchCloverJEPATrunk(nn.Module):
    """Exact PyTorch counterpart of `CloverJEPARuntime` (`clover/jepa_engine.py`) for CPU/GPU training and `CLVRTRK1` export."""

    def __init__(self):
        super().__init__()
        self.vis_adapter = nn.Linear(48, D_JEPA)
        self.vis_pos_embed = nn.Parameter(torch.randn(N_PATCHES, D_JEPA) * 0.04)
        self.vis_mod_token = nn.Parameter(torch.randn(1, D_JEPA) * 0.04)

        self.aud_adapter = nn.Linear(16, D_JEPA)
        self.aud_pos_embed = nn.Parameter(torch.randn(N_PATCHES, D_JEPA) * 0.04)
        self.aud_mod_token = nn.Parameter(torch.randn(1, D_JEPA) * 0.04)

        self.state_adapter = nn.Linear(D_VQ, D_JEPA)
        self.state_mod_token = nn.Parameter(torch.randn(1, D_JEPA) * 0.04)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(D_JEPA, 3 * D_JEPA),
                "proj": nn.Linear(D_JEPA, D_JEPA),
                "mlp1": nn.Linear(D_JEPA, 2 * D_JEPA),
                "mlp2": nn.Linear(2 * D_JEPA, D_JEPA),
            })
            for _ in range(4)
        ])

        self.pred_in_proj = nn.Linear(D_JEPA, D_PRED)
        self.pred_mask_token = nn.Parameter(torch.zeros(1, D_PRED))
        self.pred_pos_embed = nn.Parameter(torch.randn(N_PATCHES, D_PRED) * 0.03)
        self.pred_qkv = nn.Linear(D_PRED, 3 * D_PRED)
        self.pred_out_proj = nn.Linear(D_PRED, D_JEPA)

        self.cross = nn.ModuleDict({
            pair: nn.ModuleDict({
                "fc1": nn.Linear(D_JEPA, D_JEPA),
                "fc2": nn.Linear(D_JEPA, D_JEPA),
            })
            for pair in ("a2v", "v2a", "s2v", "s2a", "v2s", "a2s")
        })

        self.ac_state_in = nn.Linear(D_JEPA, D_JEPA)
        self.ac_act_in = nn.Linear(D_ACT, D_JEPA)
        self.ac_out = nn.Linear(D_JEPA, D_JEPA)

        self.idm_fc1 = nn.Linear(2 * D_JEPA, D_JEPA)
        self.idm_fc2 = nn.Linear(D_JEPA, D_ACT)

        self.bridge_jepa_to_vq = nn.Linear(D_JEPA, D_VQ)
        self.bridge_lfm3_to_vq = nn.Linear(D_LFM3, D_VQ)
        self.bridge_vq_to_lfm3 = nn.Linear(D_VQ, D_LFM3)

        self.probe_visual_scene = nn.Linear(D_JEPA, len(VISUAL_SCENE_KINDS))
        self.probe_acoustic_regime = nn.Linear(D_JEPA, len(ACOUSTIC_REGIMES))

        self.res_fc1 = nn.Linear(D_JEPA + D_ACT + D_JEPA, D_JEPA)
        self.res_fc2 = nn.Linear(D_JEPA, D_JEPA)
        nn.init.zeros_(self.res_fc2.weight)
        nn.init.zeros_(self.res_fc2.bias)

    @staticmethod
    def _ln(x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), eps=1e-5)

    def encode_tokens(self, x: torch.Tensor, hierarchical: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode `[B, N, D_JEPA]` through 4 shared Transformer blocks."""
        levels: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(D_JEPA)
        for blk in self.blocks:
            xn1 = self._ln(x)
            qkv = blk["qkv"](xn1)
            q, k, v = qkv.chunk(3, dim=-1)
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            x = x + blk["proj"](torch.matmul(attn, v))
            xn2 = self._ln(x)
            x = x + blk["mlp2"](F.gelu(blk["mlp1"](xn2)))
            levels.append(self._ln(x))
        if hierarchical:
            return torch.stack(levels, dim=0).mean(dim=0), levels
        return levels[-1], levels

    def predict_masked(self, z_ctxt: torch.Tensor, ctxt_idx: torch.Tensor, targ_idx: torch.Tensor) -> torch.Tensor:
        """Predict target tokens `[B, N_targ, D_JEPA]` from visible context `[B, N_ctxt, D_JEPA]`."""
        B = z_ctxt.size(0)
        x_c = self.pred_in_proj(z_ctxt) + self.pred_pos_embed[ctxt_idx].unsqueeze(0)
        x_m = self.pred_mask_token.unsqueeze(0).expand(B, targ_idx.numel(), -1) + self.pred_pos_embed[targ_idx].unsqueeze(0)
        x_all = torch.cat([x_c, x_m], dim=1)
        xn = self._ln(x_all)
        q, k, v = self.pred_qkv(xn).chunk(3, dim=-1)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_PRED), dim=-1)
        out = x_all + torch.matmul(attn, v)
        pred_targ = self.pred_out_proj(self._ln(out[:, ctxt_idx.numel() :, :]))
        return self._ln(pred_targ)

    def cross_predict(self, z_pool: torch.Tensor, pair: str) -> torch.Tensor:
        m = self.cross[pair]
        h = F.gelu(m["fc1"](self._ln(z_pool)))
        return self._ln(z_pool + m["fc2"](h))

    def step_transition(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ac_state_in(z_t) + self.ac_act_in(a_t))
        return self._ln(z_t + self.ac_out(h))

    def export_tensors(self, baseline_cue_proto: np.ndarray | None = None) -> dict[str, np.ndarray]:
        sd = {k: v.detach().cpu().float().numpy() for k, v in self.state_dict().items()}
        out: dict[str, np.ndarray] = {
            "vis_adapter.weight": sd["vis_adapter.weight"],
            "vis_adapter.bias": sd["vis_adapter.bias"],
            "vis_pos_embed": sd["vis_pos_embed"],
            "vis_mod_token": sd["vis_mod_token"],
            "aud_adapter.weight": sd["aud_adapter.weight"],
            "aud_adapter.bias": sd["aud_adapter.bias"],
            "aud_pos_embed": sd["aud_pos_embed"],
            "aud_mod_token": sd["aud_mod_token"],
            "state_adapter.weight": sd["state_adapter.weight"],
            "state_adapter.bias": sd["state_adapter.bias"],
            "state_mod_token": sd["state_mod_token"],
            "pred.in_proj.weight": sd["pred_in_proj.weight"],
            "pred.in_proj.bias": sd["pred_in_proj.bias"],
            "pred.mask_token": sd["pred_mask_token"],
            "pred.pos_embed": sd["pred_pos_embed"],
            "pred.qkv.weight": sd["pred_qkv.weight"],
            "pred.qkv.bias": sd["pred_qkv.bias"],
            "pred.out_proj.weight": sd["pred_out_proj.weight"],
            "pred.out_proj.bias": sd["pred_out_proj.bias"],
            "ac_pred.state_in.weight": sd["ac_state_in.weight"],
            "ac_pred.state_in.bias": sd["ac_state_in.bias"],
            "ac_pred.act_in.weight": sd["ac_act_in.weight"],
            "ac_pred.act_in.bias": sd["ac_act_in.bias"],
            "ac_pred.out.weight": sd["ac_out.weight"],
            "ac_pred.out.bias": sd["ac_out.bias"],
            "idm.fc1.weight": sd["idm_fc1.weight"],
            "idm.fc1.bias": sd["idm_fc1.bias"],
            "idm.fc2.weight": sd["idm_fc2.weight"],
            "idm.fc2.bias": sd["idm_fc2.bias"],
            "bridge.jepa_to_vq.weight": sd["bridge_jepa_to_vq.weight"],
            "bridge.jepa_to_vq.bias": sd["bridge_jepa_to_vq.bias"],
            "bridge.lfm3_to_vq.weight": sd["bridge_lfm3_to_vq.weight"],
            "bridge.lfm3_to_vq.bias": sd["bridge_lfm3_to_vq.bias"],
            "bridge.vq_to_lfm3.weight": sd["bridge_vq_to_lfm3.weight"],
            "bridge.vq_to_lfm3.bias": sd["bridge_vq_to_lfm3.bias"],
            "probe.visual_scene.weight": sd["probe_visual_scene.weight"],
            "probe.visual_scene.bias": sd["probe_visual_scene.bias"],
            "probe.acoustic_regime.weight": sd["probe_acoustic_regime.weight"],
            "probe.acoustic_regime.bias": sd["probe_acoustic_regime.bias"],
            "residual.fc1.weight": sd["res_fc1.weight"],
            "residual.fc1.bias": sd["res_fc1.bias"],
            "residual.fc2.weight": sd["res_fc2.weight"],
            "residual.fc2.bias": sd["res_fc2.bias"],
            "residual.gate": np.array([1.0], dtype=np.float32),
            "residual.baseline_cue_proto": (
                baseline_cue_proto.astype(np.float32).reshape(1, D_JEPA)
                if baseline_cue_proto is not None
                else np.zeros((1, D_JEPA), dtype=np.float32)
            ),
            "residual.has_proto": np.array([1.0 if baseline_cue_proto is not None else 0.0], dtype=np.float32),
        }
        for l in range(4):
            for k in ("qkv", "proj", "mlp1", "mlp2"):
                out[f"enc.block.{l}.{k}.weight"] = sd[f"blocks.{l}.{k}.weight"]
                out[f"enc.block.{l}.{k}.bias"] = sd[f"blocks.{l}.{k}.bias"]
        for pair in ("a2v", "v2a", "s2v", "s2a", "v2s", "a2s"):
            out[f"cross.{pair}.fc1.weight"] = sd[f"cross.{pair}.fc1.weight"]
            out[f"cross.{pair}.fc1.bias"] = sd[f"cross.{pair}.fc1.bias"]
            out[f"cross.{pair}.fc2.weight"] = sd[f"cross.{pair}.fc2.weight"]
            out[f"cross.{pair}.fc2.bias"] = sd[f"cross.{pair}.fc2.bias"]
        return out


def generate_clover_multimodal_batch(batch_size: int, g: torch.Generator, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate paired Clover visual patches `[B, 16, 48]`, audio spectrogram patches `[B, 16, 16]`,
    concept/state vectors `[B, 384]`, actions `[B, 8]`, and next concept states `[B, 384]` across 6 semantic classes.
    """
    vis_y = torch.randint(0, len(VISUAL_SCENE_KINDS), (batch_size,), generator=g).to(device)
    aud_y = vis_y % len(ACOUSTIC_REGIMES)

    vis = torch.randn((batch_size, N_PATCHES, 48), generator=g).to(device) * 0.20
    aud = torch.randn((batch_size, N_PATCHES, 16), generator=g).to(device) * 0.20
    state_vq = torch.randn((batch_size, D_VQ), generator=g).to(device) * 0.20

    for i in range(batch_size):
        vc = int(vis_y[i].item())
        ac = int(aud_y[i].item())
        # Structured foreground patches (patches 5, 6, 9, 10 carry the salient UI/chart/document signature)
        vis[i, [5, 6, 9, 10], (vc * 8) : (vc + 1) * 8] += 2.2
        vis[i, [0, 1, 2, 3], (vc * 6) % 40 : ((vc * 6) % 40) + 6] += 1.4
        # Structured time-frequency harmonic bands in audio patches
        aud[i, [3, 5, 6, 7, 9, 10], (ac * 3) : (ac * 3) + 3] += 2.2
        state_vq[i, (vc * 64) : (vc + 1) * 64] += 1.8

    actions = (torch.rand((batch_size, D_ACT), generator=g).to(device) - 0.5) * 1.2
    # Action-conditioned shift in VQ concept space
    act_proj = F.pad(actions, (0, D_VQ - D_ACT))
    # Roll each action dimension across a 48-d block of D_VQ
    next_state_vq = state_vq.clone()
    for a_i in range(D_ACT):
        next_state_vq[:, a_i * 48 : (a_i + 1) * 48] += 0.85 * actions[:, a_i : a_i + 1]

    return {
        "vis": vis,
        "aud": aud,
        "state_vq": state_vq,
        "next_state_vq": next_state_vq,
        "actions": actions,
        "vis_y": vis_y,
        "aud_y": aud_y,
    }


def train_and_export(
    device_str: str = "auto",
    scale: str = "pilot",
    epochs: int | None = None,
    batch_size: int = 96,
    seed: int = 20261107,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if epochs is None:
        epochs = 120 if scale == "full" else 45

    model = TorchCloverJEPATrunk().to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False

    std_loss_fn = HingeStdLoss(std_margin=1.0)
    cov_loss_fn = CovarianceLoss()
    cn_loss_fn = CenteredNormMSELoss(center_pred=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)

    ctxt_idx = torch.tensor([0, 1, 2, 4, 7, 8, 11, 12, 13, 14, 15], dtype=torch.long, device=device)
    targ_idx = torch.tensor([3, 5, 6, 9, 10], dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    steps_per_epoch = 16 if scale == "full" else 8

    for ep in range(1, epochs + 1):
        ep_losses = []
        for _ in range(steps_per_epoch):
            batch = generate_clover_multimodal_batch(batch_size, g, device)
            vis, aud = batch["vis"], batch["aud"]
            s_vq, ns_vq, acts = batch["state_vq"], batch["next_state_vq"], batch["actions"]

            v_tok = model.vis_adapter(vis) + model.vis_pos_embed.unsqueeze(0) + model.vis_mod_token.unsqueeze(0)
            a_tok = model.aud_adapter(aud) + model.aud_pos_embed.unsqueeze(0) + model.aud_mod_token.unsqueeze(0)
            s_tok = (model.state_adapter(s_vq) + model.state_mod_token).unsqueeze(1)
            ns_tok = (model.state_adapter(ns_vq) + model.state_mod_token).unsqueeze(1)

            with torch.no_grad():
                tv_tok = ema_model.vis_adapter(vis) + ema_model.vis_pos_embed.unsqueeze(0) + ema_model.vis_mod_token.unsqueeze(0)
                ta_tok = ema_model.aud_adapter(aud) + ema_model.aud_pos_embed.unsqueeze(0) + ema_model.aud_mod_token.unsqueeze(0)
                ts_tok = (ema_model.state_adapter(s_vq) + ema_model.state_mod_token).unsqueeze(1)
                tv_z, _ = ema_model.encode_tokens(tv_tok, hierarchical=True)
                ta_z, _ = ema_model.encode_tokens(ta_tok, hierarchical=True)
                ts_z, _ = ema_model.encode_tokens(ts_tok, hierarchical=True)

            # 1. I-JEPA visual sparse context -> target prediction
            zv_ctxt, _ = model.encode_tokens(v_tok[:, ctxt_idx, :], hierarchical=True)
            pred_v_targ = model.predict_masked(zv_ctxt, ctxt_idx, targ_idx)
            loss_ijepa = F.smooth_l1_loss(pred_v_targ, tv_z[:, targ_idx, :])

            # 2. Audio-JEPA spectrogram sparse context -> target prediction with symmetric CenteredNormMSELoss
            za_ctxt, _ = model.encode_tokens(a_tok[:, ctxt_idx, :], hierarchical=True)
            pred_a_targ = model.predict_masked(za_ctxt, ctxt_idx, targ_idx)
            loss_audio_jepa = cn_loss_fn(pred_a_targ, ta_z[:, targ_idx, :])

            # 3. MJEPA Cross-Modal 6-path alignment (`a2v`, `v2a`, `s2v`, `s2a`, `v2s`, `a2s`)
            zv_full, _ = model.encode_tokens(v_tok, hierarchical=True)
            za_full, _ = model.encode_tokens(a_tok, hierarchical=True)
            zs_full, _ = model.encode_tokens(s_tok, hierarchical=True)
            zv_p = zv_full.mean(dim=1)
            za_p = za_full.mean(dim=1)
            zs_p = zs_full.squeeze(1)

            tv_p = tv_z.mean(dim=1)
            ta_p = ta_z.mean(dim=1)
            ts_p = ts_z.squeeze(1)

            loss_mjepa_cross = (
                F.smooth_l1_loss(model.cross_predict(za_p, "a2v"), tv_p)
                + F.smooth_l1_loss(model.cross_predict(zv_p, "v2a"), ta_p)
                + F.smooth_l1_loss(model.cross_predict(zs_p, "s2v"), tv_p)
                + F.smooth_l1_loss(model.cross_predict(zs_p, "s2a"), ta_p)
                + F.smooth_l1_loss(model.cross_predict(zv_p, "v2s"), ts_p)
                + F.smooth_l1_loss(model.cross_predict(za_p, "a2s"), ts_p)
            ) / 6.0

            # 4. EB-JEPA Action-Conditioned Transition + Inverse Dynamics (`IDM`) + VCLoss
            zns_full, _ = model.encode_tokens(ns_tok, hierarchical=True)
            zns_p = zns_full.squeeze(1)
            pred_ns = model.step_transition(zs_p, acts)
            pred_act = model.idm_fc2(F.gelu(model.idm_fc1(torch.cat([zs_p, zns_p], dim=-1))))
            loss_eb_pred = F.mse_loss(pred_ns, zns_p.detach())
            loss_idm = F.mse_loss(pred_act, acts)
            loss_vc = (
                std_loss_fn(zv_p) + std_loss_fn(za_p) + std_loss_fn(zs_p)
                + 0.1 * (cov_loss_fn(zv_p) + cov_loss_fn(za_p) + cov_loss_fn(zs_p))
            )

            # 5. Bridge reconstruction & semantic probe supervision
            recon_vq = model.bridge_jepa_to_vq(zs_p)
            loss_bridge = F.mse_loss(recon_vq, s_vq)
            loss_probes = (
                F.cross_entropy(model.probe_visual_scene(zv_p), batch["vis_y"])
                + F.cross_entropy(model.probe_acoustic_regime(za_p), batch["aud_y"])
            )

            total_loss = (
                loss_ijepa
                + 0.5 * loss_audio_jepa
                + loss_mjepa_cross
                + loss_eb_pred
                + loss_idm
                + 0.5 * loss_vc
                + 0.5 * loss_bridge
                + 0.25 * loss_probes
            )

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            with torch.no_grad():
                for p_o, p_t in zip(model.parameters(), ema_model.parameters()):
                    p_t.data.mul_(0.96).add_(p_o.data, alpha=0.04)

            ep_losses.append(float(total_loss.item()))

        if ep in (1, epochs // 2, epochs):
            history.append({"epoch": ep, "loss": round(float(np.mean(ep_losses)), 4)})

    # Compute baseline context cue prototype for Clover RPRE Anomaly Gate
    with torch.no_grad():
        eval_b = generate_clover_multimodal_batch(128, g, device)
        v_t = model.vis_adapter(eval_b["vis"]) + model.vis_pos_embed.unsqueeze(0) + model.vis_mod_token.unsqueeze(0)
        a_t = model.aud_adapter(eval_b["aud"]) + model.aud_pos_embed.unsqueeze(0) + model.aud_mod_token.unsqueeze(0)
        zv_e, _ = model.encode_tokens(v_t, hierarchical=True)
        za_e, _ = model.encode_tokens(a_t, hierarchical=True)
        vis_acc = float((model.probe_visual_scene(zv_e.mean(dim=1)).argmax(dim=-1) == eval_b["vis_y"]).float().mean().item())
        aud_acc = float((model.probe_acoustic_regime(za_e.mean(dim=1)).argmax(dim=-1) == eval_b["aud_y"]).float().mean().item())
        cue_proto = (0.5 * (zv_e.mean(dim=1) + za_e.mean(dim=1))).mean(dim=0).cpu().numpy()

    tensors = model.export_tensors(baseline_cue_proto=cue_proto)
    trunk_out = STUDIO_CLOVER / "models" / "jepa-trunk-0"
    write_clvrtrk1(trunk_out, tensors)

    report = {
        "date": "2026-11-07",
        "device": str(device),
        "scale": scale,
        "epochs": epochs,
        "batch_size": batch_size,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "history": history,
        "eval_visual_scene_acc": round(vis_acc, 4),
        "eval_acoustic_regime_acc": round(aud_acc, 4),
        "exported_trunk": str(trunk_out),
        "num_tensors": len(tensors),
    }
    rep_path = ROOT / "results" / "jepa_trunk_training_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export Clover JEPA CLVRTRK1 trunks (CPU or GPU)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--scale", default="pilot", choices=("pilot", "full"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    train_and_export(device_str=args.device, scale=args.scale, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
