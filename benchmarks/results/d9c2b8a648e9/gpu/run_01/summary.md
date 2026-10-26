# Clover Real AI Benchmark Summary — `GPU` (`d9c2b8a648e9`)

- **Tested Commit SHA:** `d9c2b8a648e985a6dde5853c893fe04ed9f2c58b`
- **Checkout Path:** `/home/usergpu/clover-framework-foundations` (`working_tree_clean = False`)
- **All Imports Resolve to Frozen Checkout:** `True`
- **Hardware:** `8 × NVIDIA GeForce RTX 3090`
- **Total Standardized Tasks Executed:** `67 / 70` (`95.71%` pass rate, `96.07%` mean canonical score)
- **Total Wall Duration:** `1042.03 s`
- **Peak System RAM Used:** `80638 MiB` | **Peak Aggregate GPU VRAM Used:** `171512 MiB` (`167.49 GiB` across 8 GPUs) | **Peak GPU Util:** `100%`

## 1. Results by Standardized AI Benchmark Category

| Category | Tasks (`Passed / Total`) | Pass Rate | Mean Canonical Score | Mean Wall Latency (`ms`) | Mean 93-Layer K3 GPU (`ms`) |
| :--- | :---: | :---: | :---: | ---: | ---: |
| `1_general_knowledge_mmlu` | `10 / 10` | **`100.00%`** | `100.00%` | `12639.8 ms` | `3410.6 ms` |
| `2_math_reasoning_gsm8k` | `10 / 10` | **`100.00%`** | `100.00%` | `21790.4 ms` | `5820.7 ms` |
| `3_coding_executable_tests` | `15 / 15` | **`100.00%`** | `100.00%` | `9350.1 ms` | `2918.2 ms` |
| `4_instruction_and_structured_output` | `13 / 15` | **`86.67%`** | `90.55%` | `8030.7 ms` | `4169.9 ms` |
| `5_evidence_grounded_ontology` | `5 / 6` | **`83.33%`** | `77.78%` | `21210.7 ms` | `5674.4 ms` |
| `6_persistent_memory_and_distractors` | `6 / 6` | **`100.00%`** | `100.00%` | `24849.5 ms` | `8811.1 ms` |
| `7_tool_use_and_agent_multihop` | `8 / 8` | **`100.00%`** | `100.00%` | `19711.6 ms` | `4934.0 ms` |
| **OVERALL (7 Categories, `N=70`)** | **`67 / 70`** | **`95.71%`** | **`96.07%`** | **`14843.8 ms`** | **`4643.2 ms`** |

## 2. Separate Mechanism-Level Foundations A/B Test (`bench_foundations_e2e_ab.py`)

- **Arm A (Foundations Disabled):** Stress FP = `16/16` (`9244` polluted tokens), Positive Recall = `16/16`, Tight-Budget E2E = `0/6 (0.0%)`
- **Arm B (Foundations Enabled):** Stress FP = **`0/16`** (`0` polluted tokens), Positive Recall = **`16/16`**, Tight-Budget E2E = **`6/6 (100.0%)`**

