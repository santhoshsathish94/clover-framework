# Clover Real AI Benchmark Summary — `GPU` (`5615959136dd`)

- **Tested Commit SHA:** `5615959136dd90622105b1084c755e673635ec0c`
- **Checkout Path:** `/home/usergpu/clover-framework-foundations` (`working_tree_clean = True`)
- **All Imports Resolve to Frozen Checkout:** `True`
- **Hardware:** `8 × NVIDIA GeForce RTX 3090`
- **Total Standardized Tasks Executed:** `70 / 70` (`100.00%` pass rate, `99.08%` mean canonical score)
- **Total Wall Duration:** `1016.90 s`
- **Peak System RAM Used:** `80409 MiB` | **Peak Aggregate GPU VRAM Used:** `171512 MiB` (`167.49 GiB` across 8 GPUs) | **Peak GPU Util:** `100%`

## 1. Results by Standardized AI Benchmark Category

| Category | Tasks (`Passed / Total`) | Pass Rate | Mean Canonical Score | Mean Wall Latency (`ms`) | Mean 93-Layer K3 GPU (`ms`) |
| :--- | :---: | :---: | :---: | ---: | ---: |
| `1_general_knowledge_mmlu` | `10 / 10` | **`100.00%`** | `100.00%` | `12468.1 ms` | `3519.6 ms` |
| `2_math_reasoning_gsm8k` | `10 / 10` | **`100.00%`** | `100.00%` | `20690.3 ms` | `5674.3 ms` |
| `3_coding_executable_tests` | `15 / 15` | **`100.00%`** | `100.00%` | `9110.5 ms` | `3029.9 ms` |
| `4_instruction_and_structured_output` | `15 / 15` | **`100.00%`** | `97.92%` | `7530.2 ms` | `3841.1 ms` |
| `5_evidence_grounded_ontology` | `6 / 6` | **`100.00%`** | `94.45%` | `20279.4 ms` | `5355.8 ms` |
| `6_persistent_memory_and_distractors` | `6 / 6` | **`100.00%`** | `100.00%` | `23686.6 ms` | `8379.3 ms` |
| `7_tool_use_and_agent_multihop` | `8 / 8` | **`100.00%`** | `100.00%` | `21134.3 ms` | `4754.4 ms` |
| **OVERALL (7 Categories, `N=70`)** | **`70 / 70`** | **`100.00%`** | **`99.08%`** | **`14486.7 ms`** | **`4506.5 ms`** |

## 2. Separate Mechanism-Level Foundations A/B Test (`bench_foundations_e2e_ab.py`)

- **Arm A (Foundations Disabled):** Stress FP = `16/16` (`9244` polluted tokens), Positive Recall = `16/16`, Tight-Budget E2E = `0/6 (0.0%)`
- **Arm B (Foundations Enabled):** Stress FP = **`0/16`** (`0` polluted tokens), Positive Recall = **`16/16`**, Tight-Budget E2E = **`6/6 (100.0%)`**

