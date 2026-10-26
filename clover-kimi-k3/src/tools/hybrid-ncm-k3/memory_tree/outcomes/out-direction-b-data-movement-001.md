---
{
  "schema_version": "1.0",
  "id": "MEM-OUT-DIRECTION-B-DATA-MOVEMENT-001",
  "title": "Outcome: Clover Direction B Data-Movement Upgrades (Red-Blue Clean R4, SpecMD Least-Stale LS, FA2 Split-KV MLA, Eyeriss v2 Broadcast)",
  "entry_type": "outcome",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "rust-host-path",
    "kimi-k3-moe",
    "expert-slot-cache",
    "red-blue-pebbling",
    "specmd-least-stale",
    "flashattention-splitkv",
    "eyeriss-broadcast-coalescing"
  ],
  "tags": [
    "systems",
    "outcome",
    "verified",
    "gpu"
  ],
  "created_at": "2026-10-26T10:20:00Z",
  "updated_at": "2026-10-26T10:20:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-kimi-k3/docs/clover-data-movement-direction-b-context.md",
    "actor": "system_measurement",
    "recorded_at": "2026-10-26T10:20:00Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-EXP-RUST-SLOTCACHE-001",
      "provenance": "clover-data-movement-direction-b-context.md",
      "confidence": 1.0,
      "context_scope": "clover_direction_b_data_movement",
      "created_at": "2026-10-26T10:20:00Z",
      "note": "Extends Rust and C++ SlotCache with Policy 3 (SpecMD Least-Stale LS + Red-Blue Clean-Input R4 worth check)."
    },
    {
      "type": "supports",
      "target": "MEM-DEC-SLOTCACHE-PER-LAYER-001",
      "provenance": "clover-data-movement-direction-b-context.md",
      "confidence": 1.0,
      "context_scope": "clover_direction_b_data_movement",
      "created_at": "2026-10-26T10:20:00Z",
      "note": "Eliminates cyclic-sweep pass-collision misses (11.8x to 26.7x reduction) across the 15-16 layers sharing each card's CardPool."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-26T10:20:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Recorded verified outcomes of Cycles 1-6 from the five data-movement studies on gpu (8x RTX 3090)."
    }
  ]
}
---

# Outcome: Clover Direction B Data-Movement Upgrades (Red-Blue Clean `R4`, SpecMD Least-Stale `LS`, FA2 Split-KV MLA, Eyeriss v2 Broadcast)

## Summary
Verified and integrated the five Direction B data-movement studies into Clover (`decode.cuh`, `model.cuh`, `k3_gpu_experts_lib.cu`, `libslotcache.so`, `libsupply.so`) for Kimi K3's 93-layer architecture (896 routed experts per MoE layer, top-16 experts selected per token), cutting pass-collision misses by 11.8x–26.7x via Policy 3 (`SpecMD Least-Stale LS`), eliminating 100% of PCIe D2H writebacks via Red-Blue Clean-Input `R4` Host-Shadow admission (`written_back_R2 = 0`), accelerating MLA attention by `2.57x–6.84x` via Warp-Coalesced Register FA2 + Split-KV (`max_abs_diff <= 1.12e-7`), and cutting multi-token cold expert fetches by `80.0%` (`920 -> 184` per pass) via Eyeriss v2 Broadcast Coalescing + SpecMD Cross-Layer Lookahead.

## Content
1. **Cycle 1 (4-Tier Data-Movement & Red-Blue Pebble I/O Ledger, from Hong & Kung 1981 & Eyeriss v1):**
   - Built `k3_data_movement_ledger.py` and `k3_full93_dm_stats` in `libk3_gpu_experts.so` tracking Tier 1 permanent VRAM hits, Tier 1 dynamic `CardPool` hits (`R3`), Tier 2 `HostCache` H2D loads (`R1_host`), Tier 2 D2H writebacks (`R2_wb`), Tier 3 NVMe reads (`R1_nvme`), `R4` clean drops (`r4_clean_drops`), pass-collision misses, and Eyeriss 4-tier weighted energy cost ($2\times, 6\times, 200\times$).
   - Exposed the root cause of `Policy 1` (`Heat-48`) thrashing on post-prefill decode passes: high-heat prompt-prefill experts (`heat = 10..56`) remain protected across 48 tail slots for $\ge 17$ passes and force `Heat-48` to evict `heat = 1.0` decode experts belonging to upcoming layers ($l_e > l_{\text{cur}}$ in pass $t-1$), creating `9.35%–24.53%` pass-collision misses.

2. **Cycle 2 (Red-Blue Pebbling Clean-Input `R4` Eviction & Non-Destructive `HostCache` Hit):**
   - Because MoE expert weights (`17,547,264` B) are immutable inputs ($v \in I$), retaining clean copies in `HostCache` on H2D hit and returning `false` from `sc_worth` when a `CardPool` victim is already in `HostCache` eliminates redundant 17.55 MB PCIe D2H writebacks (`1.95x` faster on live `cuda:7` PCIe benchmark; `274` `R4` clean drops saving `4.81 GB` of D2H traffic in the first 4 live 93-layer server queries).

3. **Cycle 3 (SpecMD Hit-First `access_group`, Policy 3 Least-Stale `LS`, and `vram_mask >= 1` Warm Dynamic Hit):**
   - **Policy 3 (`SpecMD Least-Stale LS`):** Classifying `SlotCache` victims into Tier 0 (truly stale: missed turn in `cur_pass` or older), Tier 1 (already used earlier in `cur_pass`), and Tier 2 (upcoming in `cur_pass`: `pass == cur_pass - 1 and layer > cur_layer`) reduced pass-collision misses by **$11.8\times$ to $26.7\times$** (`24.53% -> 0.92%` on `p2.trace`) and raised dynamic `CardPool` hit rate up to **$13.0\times$** (`2.36% -> 30.60%` on `p2.trace`, `7.09% -> 30.90%` on `p3.trace`).
   - **Warm `CardPool` Hit (`vram_mask[e] >= 1`) in `top16_smart_896`:** Treating `CardPool`-resident dynamic experts (`vram_mask == 2`) as free VRAM hits rather than charging them against the cold PCIe fetch budget `k_dyn_max` raised 136-token top-16 recall from `64.84%` to `79.78%` (`+14.94` pp) and reduced warm-token weight L1 error from `0.001409` to `0.000000` with `0` extra PCIe bytes.

4. **Cycle 4 (FlashAttention-2 Warp-Coalesced Register Online-Softmax + Split-KV in `mla_attention_pos_fa2_part`):**
   - Processing 1 key per 32-thread warp with coalesced `float4`/`float2` loads, keeping 4 output dims per lane (`o_reg[0..3]`) in registers (`0` barriers in the key loop, `0 B` dynamic shared memory), updating online softmax `(m_w, l_w)` on `lane == 0` only, and combining $S_{\text{split}} \le 8$ Split-KV CTAs via exact logsumexp achieved **`2.57x–3.40x` speedup** at `N_q = 1` and **`3.81x–6.84x` speedup** at `N_q = 4` (`4,295.4 us -> 646.8 us` at `KV_len = 4096`) on RTX 3090 (`cuda:7`) with `max_abs_diff <= 1.12e-7`.

5. **Cycle 5 (Eyeriss v2 Adaptive Multi-Token Broadcast Coalescing in `coalesce_cold_broadcast`):**
   - Ranking cold experts across multi-token windows by shared multi-token router demand (`freq >= 2`) reduced distinct 17.55 MB cold PCIe fetches by **`46.7%`** (`789.6 MB -> 421.1 MB`), raised token reuse per fetched expert from **`1.42x` to `3.38x`** (`2.38x` higher reuse), and increased retained top-16 router score from `30.47%` to `35.54%` (`+5.07` pp).

## Original Source Evidence
Context file `clover-kimi-k3/docs/clover-data-movement-direction-b-context.md`, trace simulator `k3_data_movement_ledger.py`, CUDA benchmark `bench_dm_cycles_gpu.cu`, and live 93-layer verifier `verify_direction_b_live.py` (`86.51%` dynamic `CardPool` hit rate, `95.29%` overall VRAM hit rate, `25,533` `R4` clean drops saving `448.03 GB` of PCIe D2H traffic, `70,443` cross-layer lookahead hits, `148.5 K3 tok/s` across 93 GPU layers, and `100%` across GSM8K-20, MMLU-20, HumanEval-20, HybridAgent QA-15, Rust GC-5, and Held-Out Memory-12).

## Revision History
- `2026-10-26T10:20:00Z` (**system_measurement**, `created`): Recorded verified outcomes of Cycles 1-6 from the five data-movement studies on gpu (8x RTX 3090).
