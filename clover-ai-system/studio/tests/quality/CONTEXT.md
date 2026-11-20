# Clover Studio Quality Engine & x-cruncher Integration — Working Context

## Direction (human, 2026-11-20)
- Apply `x-cruncher v1.6.1` engineering breakthroughs (`runtime ISA dispatch`, `1-cycle LUT16 vpermb/vpshufb`,
  `lock-free 2 MB hugepage scratch cache`, `atomic/thread-local C11 concurrency`, `Radix-4/2 STFT`, and `exact math / Pi`
  engine) across Clover's native CPU engine and Studio tools.
- Build and run the **Clover Studio Quality Engine** (`PR #3` foundation + full E2E specification) in an isolated
  branch and git worktree (`feature/xcruncher-engine-and-quality-gate`), without rushing any step and without touching
  live user state (`C:\Users\santhoshn\CloverAI`) or other active worktrees.

## Boundaries
- Existing regression suites (`test_semantics.py`, `test_tool_routing.py`, `test_clover_repo.py`,
  `test_regression_fixes.py`, `test_page_fetch.py`, `test_image_gen.py`, `test_local_tool_packs.py`) remain untouched.
- `mobile/semantics.py` and `studio/clover/semantics.py` remain byte-for-byte identical; all three copies of
  `lfm3_cpu_engine.c` (`mobile/`, `src/models/lfm3-1.7b/`, `studio/clover/models/lfm3-1.7b/`) remain byte-for-byte
  identical.
- Unconfigured hosted endpoints (`clover-35b`, `nemotron`, `kimi`) and unconfigured generative image backends are
  never simulated or reported as `PASS`; they report `NOT_RUN` or raise `ImageBackendUnavailableError`.

## Execution & Outcomes Observed
1. **`lfm3_cpu_engine.c` (`x-cruncher` port):**
   - Added single-binary runtime CPU dispatch (`AVX-512 VNNI+VBMI` -> `AVX-VNNI` -> `AVX2+FMA` -> `ARM64` -> `Scalar`),
     vectorized group-64 activation quantization, lock-free 1-slot `2 MB` THP scratch cache (`0` `malloc/free` per
     pass), cgroup/affinity-aware thread topology detection, exported 4-bit `LUT16` codebook kernel
     (`lfm3_linear_4bit_lut16_omp`), and zero-dependency C11 Radix-4/2 STFT log-mel audio extractor
     (`lfm3_compute_log_mel_f32`: `1.62 ms` on `Zen 4` / `11.05 ms` on `i7-1355U` for `10s` of `16 kHz` audio).
   - Verified on `24`-layer `LFM3-Omni-2B` (`soc-q4`): **`18.5x` faster prefill** (`5,787 ms -> 313 ms`, `204.5 tok/s`)
     and **`5.3x` faster decode** (`1,730 ms -> 328 ms`, `48.8 tok/s`) with bit-identical output tokens; clean under
     `ASan+UBSan+LSan` and `ThreadSanitizer`.
2. **Studio `x-cruncher v1.6.1` Exact Math & 3-Tier Image Gateway:**
   - Vendored `x-cruncher v1.6.1` (`clover/_xcruncher_c/`) with `xcruncher_bridge.py` into Studio's `calculate` tool
     and `"math.exact"` routing (`pi_hex`, binary-splitting `factorial`, exact `51`-bit NTT / Wave big-int products).
   - Replaced silent procedural Matplotlib canvas fallback in `clover/image_gen.py` with a 3-tier Image Generation
     Gateway (`high_quality` OpenAI-compatible Images API + edits + transparent background, `local_diffusion` `sd-cli`
     txt2img + img2img, opt-in `diagram` labeled `is_generative=False`), `/api/status` `image_capabilities`, and UI
     provenance captions in `clover/web/app.html`.
3. **Real Studio Defects Surfaced & Repaired by the Quality Engine (`29 PASS, 0 FAIL, 2 READY, 3 NOT_RUN`):**
   - **Offline baseline regressions on `PR #3`:** Fixed `omni_native.looping()` false-positive on short 3-fold phrases
     and `server.tool_inventory()` counting internal `manage_todo_list` in `base` instead of `conditional`.
   - **Integrated-GPU Vulkan OOM in `omni_native._c1_server()`:** Added automatic CPU fallback (`ctx <= 8192`) when a
     second Vulkan `llama-server` fails `vk::Device::allocateMemory` alongside `self.gpu`.
   - **Multi-step tool stripping in `omni_native.plan_s3_pre_calls()`:** Restricted terminal `read_file` S3 pre-calls
     to read-only tool sets (`<= {read_file, list_directory, search_files, file_info}`) so `write_file`, `run_python`,
     `analyze_table`, and `chart_table` are never stripped before multi-step execution.
   - **Natural-order file edit & table routing in `semantics.py`:** Extended `files.write`, `python`, `table.analyze`,
     and `table.chart` patterns so `edit/update/change/fix <file>`, `run <script>.py`, and `.csv` filenames preceding
     `group by` / `bar chart` route cleanly (`155/155` `test_semantics.py` and `129/129` `test_tool_routing.py` pass).
   - **Empty filter bug in `local_tool_packs.analyze_table()`:** Ignored `filter_column` when `filter_value` is blank
     instead of filtering `column == ""` to zero rows.
   - **Outcome persistence check in `clover_cycle.verify_turn_outcome()`:** Added Cycle-2 verification when a user asks
     to modify a file (`files.write`), `read_file` succeeded, and the 2B model drafted updated text without calling
     `write_file`.
   - **Remote smart-switch resilience in `server.lfm3_turn()`:** Added seamless fallback to local `S.omni_engine` when a
     smart-routed remote `k3` connection resets or is unreachable offline.

## Growth Carried Forward
- Testing against a **disposable `--base` linked read-only to real local weights (`LFM3-Omni-2B`)** with `smart_switch`
  disabled exposed six real-world multi-step, routing, tool-argument, and offline-resilience bugs that unit mocks and
  remote synthesis hid.
- Always run `python tests/quality_gate.py --offline --spawn-isolated --seed-base <CloverAI> --live --playwright --playwright-e2e --e2e`
  before Studio releases, and verify `safety:seed_base_user_data_untouched == PASS`.
