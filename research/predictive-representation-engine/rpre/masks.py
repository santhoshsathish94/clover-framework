"""Masking strategies & token-selection utilities from I-JEPA, Audio-JEPA, and MJEPA (`rpre/masks.py`).

Implements:
- `apply_masks(x, masks, concat=True)`: exact gather operation from `ijepa/src/masks/utils.py`,
  `audio_jepa/src/masks/components/utils.py`, and `MJEPA/src/masks/utils.py`.
- `sample_multiblock_2d_masks(...)`: I-JEPA / Audio-JEPA 2D multi-block semantic target sampling
  with disjoint context complement (`acceptable_regions = masks_C`).
- `sample_random_masks(...)`: Audio-JEPA random patch masking (`random_block.py`).
- `sample_audio_tf_masks(...)`: Anisotropic Time-Frequency band + block masking tailored for
  spectrograms where harmonics span frequency and formants span time.
"""
from __future__ import annotations

import math
import torch


def apply_masks(
    x: torch.Tensor,
    masks: torch.Tensor | list[torch.Tensor],
    concat: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    """Gather token subsets indexed by `masks` from `x` (`[B, N, D]`).

    Matches `ijepa/src/masks/utils.py:apply_masks` and `MJEPA/src/masks/utils.py:apply_masks`.
    When `masks` is a list of $M$ mask tensors of shape `[B, K]`:
    - If `concat=True`, returns `[B * M, K, D]`.
    - If `concat=False`, returns a list of $M$ tensors each of shape `[B, K, D]`.
    """
    if not isinstance(masks, list):
        masks = [masks]
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat each batch block of size `B` `repeat` times (`ijepa/src/utils/tensors.py`)."""
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


def sample_random_masks(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: tuple[float, float] = (0.4, 0.6),
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Audio-JEPA `random_block.py` unstructured random patch masking.

    Returns `(masks_enc, masks_pred)` where each is a 1-element list of `[B, K]` index tensors.
    """
    num_patches = grid_h * grid_w
    r = mask_ratio[0] + torch.rand(1, generator=generator).item() * (mask_ratio[1] - mask_ratio[0])
    num_keep = max(2, min(num_patches - 2, int(num_patches * (1.0 - r))))
    enc_list, pred_list = [], []
    for _ in range(batch_size):
        perm = torch.randperm(num_patches, generator=generator)
        enc_list.append(torch.sort(perm[:num_keep]).values)
        pred_list.append(torch.sort(perm[num_keep:]).values)
    masks_enc = [torch.stack(enc_list, dim=0).to(device=device, dtype=torch.long)]
    masks_pred = [torch.stack(pred_list, dim=0).to(device=device, dtype=torch.long)]
    return masks_enc, masks_pred


def sample_multiblock_2d_masks(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    enc_mask_scale: tuple[float, float] = (0.75, 0.95),
    pred_mask_scale: tuple[float, float] = (0.15, 0.25),
    aspect_ratio: tuple[float, float] = (0.75, 1.5),
    nenc: int = 1,
    npred: int = 4,
    min_keep: int = 2,
    allow_overlap: bool = False,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """I-JEPA (`ijepa/src/masks/multiblock.py`) & Audio-JEPA (`multiblock.py`) 2D multi-block mask sampler.

    Samples `npred` rectangular semantic target blocks and `nenc` context blocks constrained to lie
    outside the target blocks (`acceptable_regions = masks_C`) when `allow_overlap=False`.
    """
    num_patches = grid_h * grid_w

    def _sample_size(scale: tuple[float, float], ar_scale: tuple[float, float]) -> tuple[int, int]:
        u = torch.rand(1, generator=generator).item()
        s = scale[0] + u * (scale[1] - scale[0])
        max_keep = max(min_keep, int(round(num_patches * s)))
        ar = ar_scale[0] + u * (ar_scale[1] - ar_scale[0])
        bh = max(1, min(grid_h - 1, int(round(math.sqrt(max_keep * ar)))))
        bw = max(1, min(grid_w - 1, int(round(math.sqrt(max_keep / ar)))))
        return bh, bw

    p_size = _sample_size(pred_mask_scale, aspect_ratio)
    e_size = _sample_size(enc_mask_scale, (1.0, 1.0))

    def _sample_one(b_size: tuple[int, int], acceptable: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bh, bw = b_size
        tries = 0
        for _ in range(60):
            top = int(torch.randint(0, max(1, grid_h - bh + 1), (1,), generator=generator).item())
            left = int(torch.randint(0, max(1, grid_w - bw + 1), (1,), generator=generator).item())
            m = torch.zeros((grid_h, grid_w), dtype=torch.int32)
            m[top : top + bh, left : left + bw] = 1
            if acceptable is not None:
                n_acc = max(0, len(acceptable) - tries)
                for k in range(n_acc):
                    m = m * acceptable[k]
            idx = torch.nonzero(m.flatten(), as_tuple=False).squeeze(-1)
            if idx.numel() >= min_keep:
                comp = torch.ones((grid_h, grid_w), dtype=torch.int32)
                comp[top : top + bh, left : left + bw] = 0
                return idx, comp
            tries += 1
        # Fallback if grid is very small
        all_idx = torch.arange(num_patches, dtype=torch.long)
        if acceptable:
            comp_flat = acceptable[0].flatten()
            valid = torch.nonzero(comp_flat, as_tuple=False).squeeze(-1)
            if valid.numel() >= min_keep:
                return valid[: max(min_keep, valid.numel())], acceptable[0]
        return all_idx[: max(min_keep, bh * bw)], torch.ones((grid_h, grid_w), dtype=torch.int32)

    batch_pred, batch_enc = [], []
    min_kp = num_patches
    min_ke = num_patches
    for _ in range(batch_size):
        mp_list, mc_list = [], []
        for _ in range(npred):
            m_p, m_c = _sample_one(p_size, None)
            mp_list.append(m_p)
            mc_list.append(m_c)
            min_kp = min(min_kp, int(m_p.numel()))
        batch_pred.append(mp_list)
        acc = None if allow_overlap else mc_list
        me_list = []
        comp_union = torch.ones((grid_h, grid_w), dtype=torch.int32)
        for mc in mc_list:
            comp_union = comp_union * mc
        for _ in range(nenc):
            if not allow_overlap:
                valid_c = torch.nonzero(comp_union.flatten(), as_tuple=False).squeeze(-1)
                if valid_c.numel() >= min_keep:
                    u_scale = enc_mask_scale[0] + torch.rand(1, generator=generator).item() * (enc_mask_scale[1] - enc_mask_scale[0])
                    n_sel = max(min_keep, min(int(valid_c.numel()), int(round(valid_c.numel() * u_scale))))
                    perm_c = torch.randperm(valid_c.numel(), generator=generator)[:n_sel]
                    m_e = torch.sort(valid_c[perm_c]).values
                    me_list.append(m_e)
                    min_ke = min(min_ke, int(m_e.numel()))
                    continue
            m_e, _ = _sample_one(e_size, acc)
            me_list.append(m_e)
            min_ke = min(min_ke, int(m_e.numel()))
        batch_enc.append(me_list)

    masks_pred = [
        torch.stack([batch_pred[b][p][:min_kp] for b in range(batch_size)], dim=0).to(device=device, dtype=torch.long)
        for p in range(npred)
    ]
    masks_enc = [
        torch.stack([batch_enc[b][e][:min_ke] for b in range(batch_size)], dim=0).to(device=device, dtype=torch.long)
        for e in range(nenc)
    ]
    return masks_enc, masks_pred


def sample_audio_tf_masks(
    batch_size: int,
    time_steps: int,
    freq_bins: int,
    npred: int = 2,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Structured Time-Frequency Spectrogram Masking for Audio-JEPA.

    Combines a contiguous time-step band mask (forcing temporal formant prediction across all
    harmonics) and a contiguous frequency-band mask (forcing harmonic overtone prediction),
    keeping the non-overlapping complement as context.
    """
    num_patches = time_steps * freq_bins
    t_band = max(1, time_steps // 4)
    f_band = max(1, freq_bins // 4)
    target_len = t_band * freq_bins

    batch_pred = [[] for _ in range(npred)]
    batch_enc = []
    min_ke = num_patches

    for _ in range(batch_size):
        comp = torch.ones((time_steps, freq_bins), dtype=torch.int32)
        for p in range(npred):
            if p % 2 == 0:
                t0 = int(torch.randint(0, max(1, time_steps - t_band + 1), (1,), generator=generator).item())
                m = torch.zeros((time_steps, freq_bins), dtype=torch.int32)
                m[t0 : t0 + t_band, :] = 1
                comp[t0 : t0 + t_band, :] = 0
            else:
                f0 = int(torch.randint(0, max(1, freq_bins - f_band + 1), (1,), generator=generator).item())
                m = torch.zeros((time_steps, freq_bins), dtype=torch.int32)
                m[:, f0 : f0 + f_band] = 1
                comp[:, f0 : f0 + f_band] = 0
            idx = torch.nonzero(m.flatten(), as_tuple=False).squeeze(-1)
            if idx.numel() < target_len:
                idx = idx.repeat((target_len + idx.numel() - 1) // idx.numel())[:target_len]
            else:
                idx = idx[:target_len]
            batch_pred[p].append(idx)

        enc_idx = torch.nonzero(comp.flatten(), as_tuple=False).squeeze(-1)
        if enc_idx.numel() < 2:
            enc_idx = torch.arange(min(4, num_patches), dtype=torch.long)
        min_ke = min(min_ke, int(enc_idx.numel()))
        batch_enc.append(enc_idx)

    masks_pred = [torch.stack(plist, dim=0).to(device=device, dtype=torch.long) for plist in batch_pred]
    masks_enc = [torch.stack([e[:min_ke] for e in batch_enc], dim=0).to(device=device, dtype=torch.long)]
    return masks_enc, masks_pred
