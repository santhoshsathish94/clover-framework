# Full-system test of main 59f56d33 on the dev PC (2026-11-11)

Worktree `clover-fullsys-wt`, branch `test/full-system-20261111` (= origin/main 59f56d33, Studio 5.0.5). The real
`server.py` ran with LFM3-Omni-2B (llama.cpp Vulkan, Q4_0) on the Intel UHD GPU, the C2/C3 precomputed parts, CSI, all
20 tools and the default layers. Data folder: a scratch copy in %TEMP% (models hard-linked; the user's `~/CloverAI`
untouched). PC: 10 cores, 15.7 GB RAM, 3.0 GB free and CPU 76% busy from other sessions at the start, so timings are
for this shared machine only. Raw results: `%TEMP%\te-study\fullsys` (JSON per phase).

## What ran and what came back

| Run | Result | Time |
|---|---|---|
| `clover/test_human_usefulness.py` (unchanged) | 18/18 checks | 16 turns, 252 s, median 11.9 s |
| `tests/natural_regression.py` (unchanged, 22 scenarios) | 39/49 turns; file scenarios re-run from a path without `\t`: 7/10 | 1006 s, median 12.6 s, max 98 s |
| 12 prompts, setting `c1_inference_only` off (default) | 9 right, 1 wrong translation, 1 factual errors, 1 cut at 512 tokens | median 5.5 s |
| same 12 prompts, setting on, GPU | 12/12 error: second model copy `ErrorOutOfDeviceMemory` (exit 0xC0000409) | 4-20 s |
| 6 prompts, setting on, CPU engine | ran (C1 = model states, 3-22 s); replies = the user's own message, or the clock tool's text | median 11.5 s |
| GSM8K first 20 (official test set) | 11/20 by hand (benchmark extractor said 8: misreads "= 3 bolts", "= 260 sheep", "= 7 dozens") | median 8.5 s |
| MMLU 4 x 5 subjects | 11/20 | median 2.9 s |
| Studio page in the browser | page, stages, answer, buttons work; "5 s · 64 tokens · 15.6 tok/s" | 6 s |

Decode 12-16 tok/s (7 tok/s during the first, busiest run). The answer text arrives only at the end of a turn
(first answer event = turn end in every turn), so a 46 s story shows nothing for 46 s.

## Findings (each observed in this run)

1. `c1_inference_only` on the GPU: a second Vulkan llama-server (ctx 8192) cannot allocate on the Intel UHD while the
   chat model is loaded; every turn ends with an error and no reply.
2. `c1_inference_only` on the CPU: runs; the reply is the user's message (C3 reproduces C1's text) or the S3 tool output.
3. 512-token answer cap (`_plan_token_budget`, 59f56d33): a 600-word story stopped mid-sentence (finish `length`);
   in excel-sales 1 six Python-writing rounds hit 512 and the cut code printed nothing (381 s turn).
4. Fixed-format replies: the two "rewrite these notes" prompts return the notes with `;` -> `,` (0 tokens, 0.3 s).
5. Personal learning cannot be switched off: `Settings.update` stores `personal = True` for any value (c32cc5c7,
   2026-11-06); natural_regression personal-habit 6 fails on it.
6. Stop does not stop: answers are emitted only after C3 (562351a7, 2026-11-06); natural_regression stop 1 fails.
7. Recall pulled another chat into an answer: the manager email in excel-sales 5 was written from an earlier, separate
   chat ("meeting moved to Friday 3 pm, room 204").
8. Wrong facts and arithmetic in fluent replies: bitcoin "24 hours ago $7,002,613.57" (consistent with the INR price
   labelled $), "$13.80" for 3 x 4.50 + 1.20, pens "$20.01", "rainbow above the sun", engine "four phases per
   revolution", Hindi "हलो हो"; the cat picture reply describes the prompt, not the drawn picture; Fibonacci reply says
   8 numbers and lists 9.
9. A model-written path lost the character after a backslash (`\te-study` -> `\e-study`) in 7 of 7 Python calls; the
   attachment text given to the model was correct (checked with `documents.block`).
10. An MMLU question made the model call `read_file` on `C:\Users\santhoshn` (53 s, failed harmlessly).

## Not covered

k3-hosted models (Kimi, 35B, Nemotron; web only), the Fused engine (not installed here: no
`lfm3_ncm_core_bf16.pt` / `liblfm3_cpu_engine.dll`), phone, voice output, a machine without other load.
