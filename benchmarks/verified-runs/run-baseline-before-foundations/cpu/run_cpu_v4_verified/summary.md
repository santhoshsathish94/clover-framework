# Clover AI v4 CPU Verification Summary (`k3`, Ryzen 9 7950X3D)

- **Tested Commit SHA:** `0391598fb12aeb953882efd87f52842e4f5301a7`
- **Scorer Script & SHA-256:** `run_real_ai_benchmark_suite.py` (`3736b986a27ca358...`)
- **7-Category Representative Tasks Passed:** **`7 / 7` (`100.0%`)**
- **Sum of Task Latencies:** `971.38 s` | **Suite Wall Envelope:** `978.59 s`
- **Unified 5-Gate CPU Verification (`verify_cpu_or_gpu_e2e.py`):** `5 / 5 PASS` (`156.02 s`)
- **Foundations 2-Arm A/B on CPU:** Arm A Stress FP = `16/16` (`9244` polluted tokens), Positive Recall = `16/16`, Tight-Budget E2E = `0/6 (0.0%)` vs. Arm B Stress FP = **`0/16` (`0` tokens)**, Positive Recall = **`16/16`**, Tight-Budget E2E = **`6/6 (100.0%)`**

## Per-Category CPU Results

| Category | Task ID | Passed | Score | Detail | Wall Latency (`ms`) | K3 CPU + NCM (`ms`) |
| :--- | :--- | :---: | :---: | :--- | ---: | ---: |
| `1_general_knowledge_mmlu` | `mmlu_college_physics_1` | `PASS` | `1.00` | `C` | `59847.8 ms` | `4777.5 ms` |
| `2_math_reasoning_gsm8k` | `gsm8k_test_1` | `PASS` | `1.00` | `3.0` | `268951.3 ms` | `13216.7 ms` |
| `3_coding_executable_tests` | `HumanEval_2` | `PASS` | `1.00` | `OK` | `76058.2 ms` | `2761.9 ms` |
| `4_instruction_and_structured_output` | `sudoku_easy_0` | `PASS` | `1.00` | `81/81 cells` | `10642.5 ms` | `10628.6 ms` |
| `5_evidence_grounded_ontology` | `onto_qa_2_harbourside_council` | `PASS` | `1.00` | `3/3 terms` | `142575.2 ms` | `6199.3 ms` |
| `6_persistent_memory_and_distractors` | `mem_qa_0_flat_response_partition` | `PASS` | `1.00` | `req=3/3, forb=0` | `299588.5 ms` | `18937.5 ms` |
| `7_tool_use_and_agent_multihop` | `agent_ha_q2` | `PASS` | `1.00` | `matched` | `113714.8 ms` | `7214.0 ms` |
