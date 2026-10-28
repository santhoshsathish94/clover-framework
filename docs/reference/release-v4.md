# Clover AI v4 (`v4.1.2`)

**Status:** Released (`VERSION` = `4.1.2`).  
**Scope:** The verified seven-layer Clover AI system; the next research experiment is tracked separately as [Observe](../../research/observe/README.md) and [Meaning Development](../experiments/meaning-development/experiment-design.md).

## What v4 represents

Clover AI v4 is the current seven-layer system described in the repository. Its seven layers cover:

1. Governance and discernment.
2. File-based memory and relational knowledge graph.
3. Ontology-guided neuro-symbolic GraphRAG and symbolic verification.
4. HybridAgent reasoning and tools.
5. On-the-fly Python and Rust generative compilation.
6. The Kimi K3 CPU/GPU engine and validated Rust host path.
7. Parallel Next-Concept Modeling and web/VS Code agent surfaces.

The authoritative implementation details, supported claims, evidence, and limitations remain in the root [README](../../README.md), [Clover-Kimi-K3 documentation](../../clover-kimi-k3/README.md), [Foundations-to-Clover & 70-Task Evaluation Context](../runtime-context/engine-and-architecture/clover-foundations-to-clover-gpu-context.md), and the committed benchmark bundles under [`benchmarks/results/1936d4eb0d53/gpu/`](../../benchmarks/verified-runs/run-baseline-before-foundations/gpu/). A version label does not replace those evidence records.

## Verified v4 baseline and boundaries

1. **Zero-state reproducible setup (`1936d4eb0d53`):**
   - Verified on the `8 × NVIDIA RTX 3090` server (`gpu`) after wiping all legacy directories under `/home/usergpu/` and bootstrapping fresh from GitHub (`santhoshsathish94/clover-framework`) and Hugging Face (`Santhoshsathish94/clover-kimi-k3`).
2. **What the 70-task head-to-head and 3-arm ablation prove ([`clover_70_ablation_report.json`](../../benchmarks/verified-runs/run-baseline-before-foundations/gpu/clover_70_ablation_report.json), [`frontier_70_scored_report.json`](../../benchmarks/verified-runs/run-baseline-before-foundations/gpu/frontier_70_scored_report.json)):**
   - **Scorer version & SHA-256:** [`run_real_ai_benchmark_suite.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/run_real_ai_benchmark_suite.py) (`sha256: 3736b986a27ca358...`, recorded in `provenance` inside `clover_70_ablation_report.json`).
   - **Clover's underlying model alone (`Arm 0`, no Clover mechanisms):** `33 / 70` (`47.14%` pass rate, `965.41 s` sum of task latencies, `234.04 GPU-s / solved task`).
   - **Adding Tool Orchestration (`<python>`, `<search>`) & Generative Compilation (`Arm 1`):** `57 / 70` (`81.43%` pass rate, `+24 tasks`, `1,015.40 s` sum of task latencies, `142.51 GPU-s / solved task`).
   - **Adding Ontology GraphRAG & Persistent Memory Graph (`Arm 2`, Full Clover v4):** `70 / 70` (`100.00%` pass rate, `99.08%` mean score, `+37 tasks` total over `Arm 0`, `1,002.26 s` sum of per-task latencies in `results.jsonl` / `1,005.50 s` suite wall envelope, `114.54 GPU-s / solved task`).
   - **Head-to-head against frontier reference models on the exact same 70 prompts:** On the 58 public/general tasks (where our harness called each model directly without attaching Clover's tools, compiler, ontology, or memory layers), **Claude Opus 5.5** (`57 / 58 = 98.28%`) and **GPT-6 Astra** (`56 / 58 = 96.55%`) are already near-perfect in-weights (vs. `30 / 58 = 51.72%` for Clover's raw model alone), and rightly abstain on private organizational ontology/memory records that were not attached to their prompt (`65 / 70 = 92.86%` and `63 / 70 = 90.00%` overall). Clover's value is the `+52.86 pp` system lift around a local open-weights model, not a claim that its raw model outperforms frontier models on full public leaderboards.
3. **CPU & GPU verification bundles preserved:**
   - **GPU (`gpu`, `8 × RTX 3090`):** Full 70-task suite (`70 / 70 = 100.00%` in `1,005.50 s`), 3-arm ablation, and 2-arm Foundations A/B (`0/16` stress FP, `6/6` tight-budget E2E) preserved under [`benchmarks/results/1936d4eb0d53/gpu/run_clean_v2/`](../../benchmarks/verified-runs/run-baseline-before-foundations/gpu/run_clean_v2/summary.md).
   - **CPU (`k3`, `Ryzen 9 7950X3D`, `124 GB` RAM):** Bit-identical C engine (`run-model.sh`), unified 5-gate CPU verification (`5 / 5 PASS` in `156.02 s`), 7-category representative task battery (`7 / 7 = 100.0%` across all 7 categories in `971.38 s` task latency / `978.59 s` suite envelope), and 2-arm Foundations A/B on CPU (`16/16 -> 0/16` stress FP, `0/6 -> 6/6` tight-budget E2E) preserved under [`benchmarks/results/1936d4eb0d53/cpu/run_cpu_v4_verified/`](../../benchmarks/verified-runs/run-baseline-before-foundations/cpu/run_cpu_v4_verified/summary.md).

## Release boundary

- v4 names the current seven-layer Clover AI system.
- Observe is the next experiment, not an eighth layer added to v4.
- Observe explores reciprocal observation and change between participating systems, then extends that experiment toward internet-wide access to shared learnings, distributed memory, model resources, and compute.
- Observe begins with proof of correctness and interaction, not a latency target. Performance is measured, but time is not an initial pass/fail constraint.
- Experimental hypotheses must remain distinguishable from implemented capabilities and verified outcomes.

## Release verification

Before publishing a GitHub Release/tag, verify the intended release commit, ensure the seven-layer description matches the implementation, and attach the relevant test and benchmark evidence. Do not imply that the distributed Observe network is already implemented in v4.
