---
license: mit
task_categories:
  - text-generation
  - image-text-to-text
  - audio-text-to-text
language:
  - en
tags:
  - clover
  - clover-35b-a3b
  - kimi-k3
  - kimi-k3.1-pro
  - nemotron
  - liquid-lfm3
  - jepa
  - hybrid-agent
  - next-concept-modeling
pretty_name: Clover AI Studio — Unified Precomputed Trunks, Experts, HybridAgent, NCM, Bridges & JEPA
---

# Clover AI Studio (`Santhoshsathish94/clover-ai-studio`)

Unified precomputed weights, algebraic operator trunks (`CLVRTRK1` / `CLV35TRK` / `LFM3TRK1` / `NEMOTRK1`), 4-bit / direct experts (`CLVR4B01` / `experts.direct` / `experts.4bit.direct`), `HybridAgent` (`C2 · Plan`), `11-Level Next-Concept Modeling` (`C3 · Decode`), latent bridges, and selectable `JEPA` predictive representation trunks (`I-JEPA`, `Audio-JEPA`, `EB-JEPA`, `MJEPA`, `Clover RPRE`) across all four Clover model families.

Every model family in this repository follows the **identical 6-folder structure**:

```text
<model>/
├── experts/       # Precomputed routed MoE experts, dense SwiGLU FFN experts & 11-level VQ codebook experts
├── trunks/        # Precomputed per-layer trunks (index.bin, common.bin, operator.bin, bd_gdn.bin / indexer.bin) & global-0/
├── hybridagent/   # Core 2 (Plan): precomputed plan-trunk-0/ + Stage-1..3 HybridLink / HybridAgent trunks & checkpoints
├── ncm/           # Core 3 (Decode): precomputed vqvae-trunk-0/, ncm-trunk-0..11/, ptr-trunk-0..2/, mmap vocab bins & NCM weights
├── bridges/       # Core 3 Bridges: precomputed bridge-trunk-0/ (ComboBridge + ExpertBridge) & cross-model latent bridge trunks
└── jepa/          # Selectable JEPA Predictive Representation Engine: jepa-trunk-0/ (CLVRTRK1) + multimodal dataset & adapters
```

---

## 1. `kimi/` — Kimi K3.1 Pro (`moonshotai/Kimi-K3.1-Pro`, `2.8T` MoE / `104B` Active, `93` Layers: `69 KDA + 24 Gated DSA` with 5-Tensor Lightning Indexer + `MoonViT-V2` + `Kimi-K3.1-Pro-DSpark`, Server Tier)

- `kimi/experts/` (`434` files, `1,347.30 GiB`): `expert-1/` .. `expert-92/` (`constants.bin`, `experts.bin`, `experts.direct`, `maps.json`) + `vqvae-expert-0/` .. `vqvae-expert-10/`
- `kimi/trunks/`: `global-0/` (`embed_tokens.bf16.bin`, `embedding_norm.bf16.bin`, `lm_head.bf16.bin`, `lm_head.4bit.bin`, `output_attn_res_norm.bf16.bin`, `output_attn_res_proj.bf16.bin`) + `trunk-0/` .. `trunk-92/` (`index.bin`, `common.bin`, `operator.bin`, `routed-shared.bin`, plus `indexer.bin` `K31IDX01` on all 24 Gated DSA layers `trunk-3, 7, 11, ..., 91, 92`)
- `kimi/hybridagent/`: `plan-trunk-0/` (`CLVRTRK1` + `CLVR4B01`) + `kimi_k31_hybridagent_final.pt` + `kimi_k31_hybrid_ncm_bridges.pt`
- `kimi/ncm/` (`627` files, `15.37 GiB`): `vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin`, `vqvae-trunk-0/`, `vqvae-expert-0..10/`, `ncm-trunk-0..11/`, `ncm-global-0/`, `ptr-trunk-0..2/`, `ptr-global-0/`
- `kimi/bridges/` (`18` files, `0.89 GiB`): `bridge-trunk-0/` + `q2k3-bridge-trunk-0/` (`CLVRTRK1` + `CLVR4B01`)

---

## 2. `nemotron/` — NVIDIA Nemotron 3.5 Super VL (`120B-A12B`, All `88` Backbone + `2` MTP + `32` C-RADIO Vision Layers, Workstation Tier)

- `nemotron/experts/` (`191` files, `66.37 GiB`): All `40` backbone MoE layers (`expert-1, 3, 5, ..., 87/`) + `expert-mtp-1/` (`constants.bin`, `experts.4bit.direct`, `maps.json`, plus `experts.direct` on `expert-1` & `expert-mtp-1`) + `vqvae-expert-0/` .. `vqvae-expert-10/`
- `nemotron/trunks/` (`476` files, `17.41 GiB`): `global-0/` + all `88` backbone trunks (`trunk-0/` .. `trunk-87/`) + `trunk-mtp-0/` & `trunk-mtp-1/` + `vision-trunk-0/` .. `vision-trunk-31/` & `vision-global-0/`
- `nemotron/hybridagent/` (`35` files, `1.41 GiB`): `plan-trunk-0/` + `plan-stage-1/` .. `plan-stage-3/` (`CLVRTRK1` + `CLVR4B01`) + `nemotron_hybridagent_final.pt`
- `nemotron/ncm/` (`374` files, `3.17 GiB`): `vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin`, `vqvae-trunk-0/`, `vqvae-expert-0..10/`, `ncm-trunk-0..11/`, `ncm-global-0/`, `ptr-trunk-0..2/`, `ptr-global-0/`
- `nemotron/bridges/` (`8` files, `1.22 GiB`): `bridge-trunk-0/` (`CLVRTRK1` + `CLVR4B01`) + `nemotron_hybrid_ncm_bridges.pt`

---

## 3. `clover-35b/` — DiffuSpace Clover-35B-A3B (`Diffuspace/Clover-35B-A3B-Preview`, `35B` MoE / `3B` Active, `40` Layers: `30` Bidirectional Gated DeltaNet `BD-GDN` + `10` Gated GQA + Block Diffusion `block_size=32`, Apache-2.0)

- `clover-35b/experts/`: `expert-0/` & `expert-3/` (`constants.bin`, `experts.direct`, `experts.4bit.direct`, `maps.json` with `top_p=0.9` block coreset selection) + `vqvae-expert-0/` .. `vqvae-expert-10/`
- `clover-35b/trunks/`: `global-0/` (`embed_tokens.bf16.bin`, `lm_head.bf16.bin`, `lm_head.4bit.bin`, `norm.bf16.bin` over `248,320` vocab, `D_model = 2048`) + `trunk-0/` (`BD-GDN` + `operator.bin` `CLV35OP1` + `bd_gdn.bin` `CLV35BD1`) + `trunk-3/` (`Gated GQA` + `operator.bin` with `attention.bin` pruned, verified `cos_sim = 1.0`, `max_abs_diff = 0.0` in both causal commit and block-diffusion preview modes) + `bd_gdn.bin` across all 30 Bidirectional GDN layers (`trunk-0, 1, 2, 4, 5, 6, ...`) + `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `clover-35b/hybridagent/`: `plan-trunk-0/` (`CLVRTRK1`: `Clover35BNativeHybridLink` $2048 \to 512 \to 2048$, $2048 \to 1024 \to 2048$, and `HybridLink2048`) + `clover_35b_hybridagent_final.pt` + `clover_35b_hybridagent_training_summary.json`
- `clover-35b/ncm/`: `vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin` (`[248320, 384]` BF16), `vqvae-trunk-0/`, `vqvae-expert-0..10/`, `ncm-trunk-0..11/`, `ncm-global-0/`, `ptr-trunk-0..2/`, `ptr-global-0/`, `manifest.json`
- `clover-35b/bridges/`: `bridge-trunk-0/` (`CLVRTRK1`: `Clover35BComboBridge` + `Clover35BExpertBridge` $2048 \to 384$) + `clover_35b_hybrid_ncm_bridges.pt`
- `clover-35b/jepa/`: `jepa-trunk-0/` (`CLVRTRK1` 106 tensors aligned to Clover-35B-A3B `2048 <-> 384 <-> 64`) + `clover_35b_jepa_report.json`

---

## 4. `liquid-lfm3/` — Liquid AI LFM3-Omni-2B / LFM3-1.7B (`24` Tri-Hybrid Layers, Mobile & Laptop Tier)

- `liquid-lfm3/experts/` (`138` files, `0.63 GiB`): `expert-0/` .. `expert-23/` (`constants.bin`, `maps.json`, `ffn.4bit.bin`) + `vqvae-expert-0/` .. `vqvae-expert-10/`
- `liquid-lfm3/trunks/` (`126` files, `1.96 GiB`): `global-0/` (`embed_tokens.bf16.bin`, `embedding_norm.bf16.bin`, `lm_head.4bit.bin`, `rope_inv_freq.bin`) + `trunk-0/` .. `trunk-23/` (`index.bin`, `common.bin`, `operator.bin`, `operator.4bit.bin`, `ffn.4bit.bin`) + `prune_lfm3_summary.json` + `tokenizer.json`
- `liquid-lfm3/hybridagent/` (`7` files, `0.04 GiB`): `plan-trunk-0/` (`CLVRTRK1` + `CLVR4B01`) + `hybridlink_adapter_2048.pt`
- `liquid-lfm3/ncm/` (`188` files, `0.93 GiB`): `vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin`, `vqvae-trunk-0/`, `vqvae-expert-0..10/`, `ncm-trunk-0..11/`, `ncm-global-0/`, `ptr-trunk-0..2/`, `ptr-global-0/`, `manifest.json` + `lfm3_ncm_core_bf16.pt`
- `liquid-lfm3/bridges/` (`9` files, `0.60 GiB`): `bridge-trunk-0/` (`CLVRTRK1` + `CLVR4B01`) + `lfm3_hybrid_ncm_bridges.pt`, `lfm3_s23_bridges.npz`, `lfm3_s23_bridges_fp16.npz`
- `liquid-lfm3/jepa/` (`12` files, `12.35 MB`): `jepa-trunk-0/` (`CLVRTRK1`: `index.bin`, `common.bin`, `operator.bin`, `ffn.bin` trained across 8x NVIDIA RTX 3090 GPUs for `ijepa`, `audio_jepa`, `eb_jepa`, `mjepa`, and `clover_rpre`), `real_clover_jepa_dataset.npz`, `jepa_outcome_adapter.npz`, `jepa_collective_growth.json`, and benchmark/self-training reports.
