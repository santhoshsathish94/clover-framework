# Research: DeepSeek Super-Agent, OpenJev Decision Engine & NVIDIA Dory 1.0

> **Review, 2026-11-09:** see [05-review-and-held-out-retest.md](../../docs/runtime-context/superagent-jev-dory/05-review-and-held-out-retest.md) and the pre-registered held-out re-test in [02-openjev-decision/heldout-review/](02-openjev-decision/heldout-review/PREREGISTRATION.md).

Executable probes, standalone model runtimes, and empirical JSON telemetry for the three-project Clover integration study:

- [01-deepseek-superagent/](01-deepseek-superagent/clover-dsh-probe.spec.ts) — Live Cordis architecture probe (`clover-dsh-probe.spec.ts`) and telemetry ([dsh_live_probe_results.json](01-deepseek-superagent/dsh_live_probe_results.json))
- [02-openjev-decision/](02-openjev-decision/run_openjev_live_experiments.py) — Live `GPT-AGI/OpenJev` & `Braedennn/OpenJev` evaluation and `CloverUpgradedOpenJevBackend` ([openjev_live_results.json](02-openjev-decision/openjev_live_results.json))
- [03-nvidia-dory/](03-nvidia-dory/run_dory_live_experiments.py) — Standalone 507-tensor PyTorch engine for `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` across `1, 2, 3, 4` loops ([dory_live_results.json](03-nvidia-dory/dory_live_results.json))
- [04-clover-unified-integration/](04-clover-unified-integration/run_unified_clover_ablation.py) — Jev-Adaptive Dory loop controller & 5-way combined system ablation ([unified_clover_ablation_results.json](04-clover-unified-integration/unified_clover_ablation_results.json))
- [05-plain-message-routing/](05-plain-message-routing/PREREGISTRATION.md) — Pre-registered 200-message plain-message routing benchmark, the gate for a caution-only routing reading: FAIL ([results.json](05-plain-message-routing/results.json)). Follow-up item 2.
- [06-dory-parity/](06-dory-parity/PREREGISTRATION.md) — Pre-registered parity check of the standalone Dory engine against the llama.cpp Dory PR: PASS at fp32 ([results.json](06-dory-parity/results.json), [claims.json](06-dory-parity/claims.json)). Follow-up item 3; the GGUFs and logit dumps stay on k3.
- Context records: [docs/runtime-context/superagent-jev-dory/README.md](../../docs/runtime-context/superagent-jev-dory/README.md)
