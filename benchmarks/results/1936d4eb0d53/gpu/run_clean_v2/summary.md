# Clover Real AI Benchmark Summary — `GPU` (`1936d4eb0d53`)

- **Tested Commit SHA:** `1936d4eb0d531fee5802a71379423c06dd9cdaa7`
- **Checkout Path:** `/opt/clover-k3/clover-framework` (`working_tree_clean = True`)
- **All Imports Resolve to Frozen Checkout:** `True`
- **Hardware:** `8 × NVIDIA GeForce RTX 3090`
- **Total Standardized Tasks Executed:** `70 / 70` (`100.00%` pass rate, `99.08%` mean canonical score)
- **Total Wall Duration:** `1005.50 s`
- **Peak System RAM Used:** `80441 MiB` | **Peak Aggregate GPU VRAM Used:** `171590 MiB` (`167.57 GiB` across 8 GPUs) | **Peak GPU Util:** `100%`

## 1. Results by Standardized AI Benchmark Category

| Category | Tasks (`Passed / Total`) | Pass Rate | Mean Canonical Score | Mean Wall Latency (`ms`) | Mean 93-Layer K3 GPU (`ms`) |
| :--- | :---: | :---: | :---: | ---: | ---: |
| `1_general_knowledge_mmlu` | `10 / 10` | **`100.00%`** | `100.00%` | `12238.3 ms` | `3310.2 ms` |
| `2_math_reasoning_gsm8k` | `10 / 10` | **`100.00%`** | `100.00%` | `21202.9 ms` | `5878.1 ms` |
| `3_coding_executable_tests` | `15 / 15` | **`100.00%`** | `100.00%` | `9088.8 ms` | `2858.7 ms` |
| `4_instruction_and_structured_output` | `15 / 15` | **`100.00%`** | `97.92%` | `7354.0 ms` | `3652.3 ms` |
| `5_evidence_grounded_ontology` | `6 / 6` | **`100.00%`** | `94.45%` | `20560.6 ms` | `5369.1 ms` |
| `6_persistent_memory_and_distractors` | `6 / 6` | **`100.00%`** | `100.00%` | `23987.2 ms` | `8439.0 ms` |
| `7_tool_use_and_agent_multihop` | `8 / 8` | **`100.00%`** | `100.00%` | `19239.6 ms` | `4869.1 ms` |
| **OVERALL (7 Categories, `N=70`)** | **`70 / 70`** | **`100.00%`** | **`99.08%`** | **`14318.0 ms`** | **`4447.8 ms`** |

## 2. Separate Mechanism-Level Foundations A/B Test (`bench_foundations_e2e_ab.py`)

- **Arm A (Foundations Disabled):** Stress FP = `16/16` (`9244` polluted tokens), Positive Recall = `16/16`, Tight-Budget E2E = `0/6 (0.0%)`
- **Arm B (Foundations Enabled):** Stress FP = **`0/16`** (`0` polluted tokens), Positive Recall = **`16/16`**, Tight-Budget E2E = **`6/6 (100.0%)`**

