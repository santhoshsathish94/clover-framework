# Clover Reusable Predictive Representation Engine (`RPRE`)

> **Guiding Architectural Question**: *Can we build a reusable predictive representation engine that processes different kinds of information more efficiently, learns their underlying relationships, and improves through observed outcomes?*

**Yes—when three mechanisms from the four JEPA papers are unified with Clover's closed-loop `Outcome → Growth` cycle:**
1. **Context-Only Sparse Multi-Modal Encoding (`I-JEPA` + `Audio-JEPA` + `MJEPA`)**: Modality-specific patch/vector adapters drop masked tokens *before* the shared Transformer trunk (`apply_masks`), cutting self-attention FLOPs by **$3.3\times$ (`29.9%` of full grid)** while predicting in normalized latent space rather than reconstructing noisy pixels or waveforms (**`97.8%` vs. `64.7%`** under heavy noise in `E1`; **`0.8362` vs. `-0.0125`** masked latent cosine alignment in `E2`).
2. **Cross-Modal & Energy-Regularized Relational Learning (`MJEPA` + `EB-JEPA`)**: Combining MJEPA's 9-path intra/cross-modal prediction (`v2v`, `a2a`, `av2av`, `a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`) and 4-level hierarchical target extraction with EB-JEPA's Variance-Covariance + Temporal Similarity + Inverse Dynamics regularization (`VC_IDM_Sim_Regularizer`) prevents both representation collapse (`std: 0.0135 -> 1.0225`) and modality competition (`joint acc: 61.1% -> 68.6%`), lifting closed-loop latent MPC planning success from **`6.7%` (unregularized) and `60.0%` (`VCLoss` only) to `85.0%` (`VCLoss + Sim + IDM`)** (`E3`, `E4`).
3. **Closed-Loop Outcome-Driven Residual Growth (`Clover RPRE` `OutcomeResidualAdapter` + `CloverCycleAgent`)**: All four upstream JEPA codebases are open-loop at deployment—their world models never update when real execution outcomes $z_{t+1}^{\text{obs}}$ diverge from predictions $\hat{z}_{t+1}$. Adding `OutcomeResidualAdapter` conditioned on MJEPA's multimodal context cues with a baseline-prototype anomaly gate and held-out non-regression rollback gate reduces shifted-regime prediction MSE by **$3.7\times$ (`0.1162 -> 0.0316`)**, lifts goal-reaching success from **`60.4%` (Cycle 1) to `79.2%`–`85.4%` (Cycles 3–5)** (vs. `54.2%` static JEPA and `29.2%` state-only), preserves **`93.3%` (`100%` of baseline)** on unshifted dynamics, and achieves **`80.0%` zero-shot audio-only transfer** when visual input is occluded (`E5`).

---

## Documents & Artifacts

1. [**Stage 1 — Source-Level Audit of the Four Papers & Repositories**](STAGE1-FOUR-PAPERS-AND-REPOS-AUDIT.md)
   - Line-by-line architectural analysis of [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa), [`LudovicTuncay/Audio-JEPA`](https://github.com/LudovicTuncay/Audio-JEPA), [`facebookresearch/eb_jepa`](https://github.com/facebookresearch/eb_jepa), and [`facebookresearch/MJEPA`](https://github.com/facebookresearch/MJEPA).
   - Documents the asymmetric target-only centering defect discovered in `Audio-JEPA` (`loss.py:42–46`) and the open-loop deployment limitation in `EB-JEPA` (`eb_jepa/planning.py:258–295`).
2. [**Stage 2 — Reusable Predictive Representation Engine (`RPRE`) Architecture**](STAGE2-REUSABLE-PREDICTIVE-ENGINE-ARCHITECTURE.md)
   - Full mathematical specification and system diagram of `RPRE` (`rpre/`).
3. [**Stage 3 — Controlled Multi-Seed Experiments (`E1`–`E5`) & Results**](STAGE3-EXPERIMENTS-AND-RESULTS.md)
   - Complete experimental tables and ablation analysis across seeds `[101, 202, 303]`.
4. [**Stage 4 — Clover AI System Integration, Selectable JEPA Interfaces & 5-Flow Benchmark**](STAGE4-CLOVER-AI-SYSTEM-JEPA-INTEGRATION.md)
   - Unified selectable JEPA interface ([`clover/jepa_engine.py`](../../clover-ai-system/studio/clover/jepa_engine.py)), trained `CLVRTRK1` binary trunk (`models/jepa-trunk-0`), GPU training pipeline ([`scripts/train_clover_jepa_trunks.py`](scripts/train_clover_jepa_trunks.py)), and 5-flow benchmark ([`test_jepa_clover_flows.py`](../../clover-ai-system/studio/clover/test_jepa_clover_flows.py)).
5. [**Code & Reproducibility**](rpre/)
   - [`rpre/masks.py`](rpre/masks.py), [`rpre/losses.py`](rpre/losses.py), [`rpre/engine.py`](rpre/engine.py), [`rpre/planner.py`](rpre/planner.py)
   - [`tests/test_rpre.py`](tests/test_rpre.py) (15 unit & integration checks)
   - [`scripts/run_all_jepa_experiments.py`](scripts/run_all_jepa_experiments.py) & [`results/jepa_investigation_results.json`](results/jepa_investigation_results.json)
   - [`scripts/train_clover_jepa_trunks.py`](scripts/train_clover_jepa_trunks.py), [`results/jepa_trunk_training_report.json`](results/jepa_trunk_training_report.json), & [`results/clover_jepa_flow_integration_results.json`](results/clover_jepa_flow_integration_results.json)
