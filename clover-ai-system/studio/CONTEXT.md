# Studio Model Response Integrity

## Direction - 2026-11-16

The human requested Clover operating rules, no hardcoded answers or response
fallbacks, and visible real model responses from Clover AI Studio.

Canonical AGENTS.md and the packaged clover/AGENTS.md were read in full. Main
was clean at the start. Existing verification controls, user data, installed
applications, and production services must remain unchanged without approval.

## Cycle 1 - Missing Image Model

- Observed: ImageGen.generate called a predefined Matplotlib scene renderer
  whenever its diffusion executable or weights were absent. It reported success
  with model name "Clover Visual Canvas".
- Changed: unavailable diffusion models now raise an explicit RuntimeError.
- Checked locally: sd15-lcm, sd15, and an unknown preset with an empty temporary
  model directory. All three raised model-unavailable errors. A patched renderer
  would fail the probe if called; it was never called. No media directory existed
  afterwards. Editor diagnostics reported no errors.
- This verifies missing-model behavior only, not successful model inference or
  a shipped/live Studio installation. No existing tests were modified.
- Remaining: trace chat response substitutions;
  exercise genuine model responses through Studio and preserve their provenance.
- Growth: more varied procedural scenes do not make a missing model real. Model
  unavailability must stay distinguishable from successful generated content.

## Cycle 2 - Remove Procedural Image Content

- Searched the AI system source for ImageGen and _render_canvas_image callers.
  Only ImageGen had used the renderer; the server tool already reports failures.
- Removed the complete unused renderer, including its subject templates.
- Unchanged tests/test_image_gen.py: 15 checks passed locally. They cover CLI
  arguments, content-named files, failed generation, cancellation, empty prompts,
  low memory, and the image tool error contract, using a stand-in executable.
- No real diffusion inference or installed application has been checked yet.
- Next: follow server engine selection and answer assembly, then check the
  actual running model in an isolated conversation without changing user data.

## Cycle 3 - Conversational Model Bypass

- Observed: server.lfm3_turn returned semantics.Frame.reply for social messages,
  reporting zero rounds and no generation. Existing test_semantics.py and
  test_self_description.py explicitly require this old fixed-answer behavior.
  The current human direction conflicts with those expectations; they were not
  changed or used to redefine acceptance.
- Removed the server bypass. Social messages retain no-tool routing but now use
  the same selected-engine run_turn path as ordinary chat.
- An isolated recording-engine probe passed for Hi, Thanks, How are you?,
  Are you conscious?, and Explain a checksum. Each reached the engine exactly
  once, preserved its output, and reported one round and that engine's identity.
- This is orchestration verification with a test engine, not live inference.
- Remaining: obsolete fixed reply data and metadata in semantics, direct reply
  plans in Omni, engine selection fallbacks, and actual model/UI verification.

## Cycle 4 - Observe Answers Without Rewriting Them

- Removed the fixed social/self-description answer text and its generator from
  semantics. Social classification still controls no-tool routing; it no longer
  supplies an answer or claims the model was bypassed.
- Semantic alignment now records checks instead of replacing the answer with
  corrected arithmetic, restored paths, formatting, or stock explanatory text.
  Tool outcomes, injection flags, and discrepancy notes remain in metadata.
- Focused local checks passed: four social frames have no canned reply;
  matching arithmetic is verified; incorrect and missing arithmetic is contested;
  original text is preserved in all cases; untrusted tool instructions remain
  flagged without appending a claim about what the model did.
- No live inference established by these offline checks. Next: remove engine
  canned plans and silent model replacements, then verify actual inference.

## Cycle 5 - Selected Hosted Model Must Run

- Observed: K3HostEngine delegated answers to Omni, rewrote its model labels,
  skipped hosted synthesis on the web and for some outputs, and swallowed hosted
  worker failures. Retries also delegated to Omni.
- Removed those delegations. Chat and retry now enter the existing selected
  worker path. Omni may still supply separately identified speech/input helpers.
- Local recording-worker checks passed in desktop and web modes and on retry.
  A backing Omni object raised if asked to answer; it was never called. An
  unavailable selected worker raised its error rather than returning an answer.
- These tests use controlled doubles, not live models. Remaining shortcuts in
  the standalone path (direct plans, identity rewriting, optional-stage error
  handling) still need removal before the hosted path is considered complete.

## Cycle 6 - No Fabricated C1 State

- Removed _EmbeddingStateServer, which substituted token embeddings or zero
  vectors for full C1 inference, and removed remote engines' delegation of C1
  inference to a different local Omni model.
- Missing selected-model providers/GGUF files now raise an explicit error.
- Local checks passed for absent and nonexistent model paths, no alternate-model
  invocation, removal of the synthetic provider class, and preservation of an
  already configured provider. This does not measure real hidden states.
- Unsupported hosted/fused inference-only configurations can now expose a real
  error instead of silently producing an answer from substitute state.

## Verification Direction Confirmed

- The public /api/health request returned auth_required: approved sign-in is
  required. No authentication was bypassed and no live web inference was run.
- The human explicitly approved updating existing tests that require canned
  replies or bypassed inference to require model calls and explicit failures.
- The human chose an isolated local Studio using installed models for live
  verification. Keep its conversations/data separate. No production deployment,
  installed-app replacement, or user conversation modification is authorized.

## Cycle 7 - Omni Answers Must Come From Inference

- Removed Omni's canned image-capability answer, direct S3 answer bypass,
  arithmetic replacement in automatic retries, and post-answer text trimming.
- Removed the TypeError catch that could retry a failed _chat call with a
  different signature instead of surfacing the original failure.
- Recording-model probe passed: an image-capability question called the model
  once and retained its output and counters. A model returning incorrect 390 for
  17 * 23 was called three times; 390 remained visible with a failed Outcome.
- Local installed data/tools contain Omni and llama.cpp, but no diffusion
  component. Real image generation is unavailable on this installation; test
  that absence as an explicit failure, not a synthetic successful image.

## Cycle 8 - Explicit Model and Backend Selection

- Unknown model IDs now raise instead of selecting Omni; an unavailable Omni
  reference no longer returns whichever previous engine remains in S.engine.
- GPU and fused startup failures now report errors without retrying on CPU.
  Smart switching occurs only for an explicitly selected Auto model.
- Focused local checks passed for exact model selection, unknown IDs, unavailable
  Omni, GPU failure, and fused failure. Failed startups left no selected engine.
- This verifies selection/error control flow with constructor doubles, not
  hardware execution. Existing fixtures that only set S.engine need to set the
  selected S.omni_engine explicitly under the approved test update.

## Cycle 9 - Hosted Output and Metrics

- Removed hosted identity rewriting, zero-inference direct tool replies, the
  alternate precomputed-draft response path, and fabricated NCM labels/counters.
  Constructing a hosted engine no longer changes the backing Omni's identity.
- Empty worker output and output-stage errors now fail explicitly.
- Focused checks passed: a successful image tool still invokes the selected
  worker, even a wrong identity claim remains the model's actual text, missing
  counters stay null, an unrun NCM stage is not reported, and empty output errors.
- Worker doubles establish control flow only; hosted model inference is outside
  the chosen isolated-local verification target.

## Cycle 10 - Shared Direct-Answer Bypass Removed

- Deleted the shared direct S3 answer generator and the remaining native/fused
  callers. Successful tool results now supply context rather than answer text.
- Fused control-flow probe passed: a completed clock tool still invoked the
  model, preserved its response and reported counters, and empty model output
  raised an error. Native and Omni modules also imported successfully.
- No fused weights were executed in this probe. Optional-layer missing-model
  behavior and persisted regression tests remain to be checked.

## Cycle 11 - Missing Neural Stages Are Errors

- LayerKit no longer falls back from a missing hosted runtime to Omni, fabricates
  C1/C2 states, hashes words into neural state, or passes input through a missing
  C3 model. C3 rejects text-only plans without model token IDs.
- Focused provider checks passed for all missing stages and exact preservation
  of supplied embedding values in the meaning bridge. These are controlled
  provider tests, not proof of real C2/C3 inference.
- Startup inspection found automatic optional-component installation. Isolated
  verification must disable uninstalled optional stages explicitly and avoid
  automatic setup touching the shared installed model/tool directories.

## Cycle 12 - Model Installation Isolation and Manifest Check

- Removed hardcoded external and home-directory searches from precomputed_dir.
  An empty supplied installation now remains missing for Omni and all three
  hosted model keys instead of borrowing another installation's assets.
- Actual installed required-file checks returned no missing entries for Omni,
  C2/C3, or CSI. The earlier concern that C2/C3 might be absent was not established
  by the folder listing; no always-on core policy change is needed for this run.
  Manifest presence does not yet prove payload validity or inference success.
- The unchanged memory-only routing suite passed, including its fault-injection
  liveness check, after neural output stages were gated on their explicit layer.
- Isolated verification will use the complete installed core with separate
  writable data. The diffusion component is still absent.

## Cycles 13-16 - Decoder, Retry, Grounding, and Startup

- C3 no longer echoes input tokens when its decoded answer is incomplete. A
  discriminating logits probe selected different tokens and retained them.
  Empty decoding, missing C1 embeddings, and corrupt model payloads now raise.
  Training code also contains the pointer copy-bias constants; these were not
  mistaken for response templates or removed. Runtime metadata reports actual
  executed NCM levels/blocks rather than fixed maximum counts.
- Retry voting/checks remain; returned text is an actual model candidate without
  stock success prose or copied prior answers. Empty rewrites fail. An indentation
  error in the edit was caught, repaired, and the same focused checks passed.
- Ontology grounding reports conflicts without substituting model text.
- JEPA no longer uses random/default weights when checkpoints are missing,
  corrupt, or incomplete. Startup no longer marks a partial model stack ready.
  Missing-stage startup checks passed and closed the failed instance's model.

## Approved Tests and Live Local Run

- Human authorized replacing canned-reply/bypass expectations in existing tests.
  Self-description suite passed (24 checks), semantics passed (155 checks), and
  test_regression_fixes.py passed including its installed C2/C3 checks. The image
  suite (15 checks) and memory routing suite also passed earlier in this task.
- Test indentation defects were caught before execution. After three failed
  patch-tool repairs, the human authorized a two-prefix mechanical repair in
  test_semantics.py; the complete suite then passed. Full surrounding context
  fixed a subsequent indentation defect in test_regression_fixes.py.
- Human then authorized stopping the existing local Studio to free memory. Its
  PID 23116 process tree was confirmed gone; original data was not deleted.
- Real isolated server: http://127.0.0.1:8799/, launched by
  tests/live_model_integrity.py, PID 27828, terminal
  d33fa869-342e-41bc-a638-973e7cec0025. Base:
  C:/Users/santhoshn/AppData/Local/Temp/clover-studio-integrity-7a4x5vih.
  Copied only data/ and tools/, verified all 742 files by SHA-256; no vault,
  credentials, or original conversations copied. All core stages and CSI loaded.
- Actual GPU: Intel UHD on i7-1355U, 15.7 GB RAM. Novel bicycle prompt: 23 C1
  generated tokens; real C2 and C3 ran; response was prompt-specific. Browser Hi:
  10 generated tokens. Browser umbrella prompt: 23 tokens, 3.77 s whole turn.
- Umbrella response SHA-256 was identical at C1 output, C3 output, final SSE
  response, and visible browser paragraph:
  1223a3ebc6886272e11e6c1588615d13ff13fa4c9709302ed7cae788669526b7.
  Trace: model-output-trace.jsonl in the isolated base. C3 used 4 levels and
  3 transformer blocks for these short replies; do not claim all 11/12 ran.
- Chromium discards the response body when the app closes finished SSE. Captured
  chunks as the browser consumed them instead of replaying a completed prompt.
- Live negative image probe produced no picture but the model offered a textual
  substitute. This was not accepted as an error-path pass. Required image-tool
  failure now raises before inference; fake Pillow 3D previews, preset 3D plan
  execution, and invented installation claims were also removed. Focused checks
  passed; restart and repeat the live failure probe before calling it complete.
- Saved model/backend choices now survive settings reload; focused checks passed.

## Final Local Outcome

- The isolated Studio was restarted with --reuse after all chat-path fixes,
  re-verifying its assets. Current terminal:
  ae1b3855-84c9-4519-8c3f-356afe99d033. It remains running at
  http://127.0.0.1:8799/ with the same isolated base above.
- The repeated image request returned HTTP 500 with the missing diffusion model
  error, no reply/image, and zero new C1/C3 trace records. An unknown selected
  model returned HTTP 400, likewise without a generated substitute.
- A fresh successful API request generated 26 tokens and executed C2/C3. Its
  returned text exactly matched the recorded C3 SHA-256. The final browser
  request generated 24 tokens and displayed:
  "Ice floats because it is less dense than liquid water, so it rises to the top
  of the water column."
  Browser/SSE/C3 SHA-256:
  1956ebf7500f87fa2ad3feff92bc8ae6d4b8f032de2f2611de108e943dbd6f5b.
- Browser screenshot showed the failed image request as an explicit error and
  the subsequent real model response with its measured token count. An earlier
  browser wait timed out because the polling options were in the wrong argument;
  the final check used the proper Playwright options and completed.
- All 742 original installed data/tool files were re-hashed at handoff and still
  matched the initial manifest. Original app files, credentials, and conversations
  were not replaced. Only the previously running local process was stopped with
  the human's approval.
- Evidence in the isolated base: asset-hashes.json, model-output-trace.jsonl,
  and live-verification.json. These contain local test evidence, not production
  certification or proof of correctness for every possible model answer.
- Final targeted suites all passed: test_model_response_integrity.py (13 tests),
  test_self_description.py, test_semantics.py (155 checks), test_image_gen.py
  (15 checks), test_memory_turn_tools_follow_route.py, test_regression_fixes.py.
  Repository gates passed: 29 validator tests; 399 Markdown files/1,802 relative
  links with zero broken links/anchors; 10 site pages with zero errors.
- Shared mobile semantics and JEPA sources were synchronized. Only the matching
  retry_cycle function was synchronized in mobile/clover_cycle.py because the
  surrounding mobile and Studio files already differed at HEAD. Learning callers
  no longer use the removed hash-state helper. No removed response helper has a
  remaining reference under studio/clover.
- Not run: the entire Studio test collection, live hosted-model inference,
  phone/voice workflows, or successful diffusion inference. Those environments
  are not covered by the local text checks. No production deployment, package
  release, commit, or push was performed; changes remain in the worktree.
- Continue from the recorded local URL and evidence. For production or packaged
  apps, obtain deployment direction, run the corresponding real-model workflows,
  and preserve the explicit failures. Do not restore canned replies or alter
  verification outcomes to conceal unavailable models or inaccurate output.

## Image Installation Follow-up - 2026-11-16

- Human tried image generation under every model and asked why required
  components were not installed. All selected chat models share the same local
  image tool. Stopping at an honest missing-model error was an incomplete usable
  outcome; complete the actual picture setup and verify real images.
- Scope remains the running isolated base at port 8799. The existing pinned
  pictures component contains stable-diffusion.cpp Vulkan, SD 1.5 (4.265 GB), and
  LCM-LoRA (134.6 MB), each with its publisher SHA-256 in launch/manifest.json.
- Fixed setup_fetch status: a requested picture-component failure now produces
  a failed completion event and nonzero exit, even if core files succeeded.
  Focused failure/success checks passed.
- Actual setup run installed and hash-verified stable-diffusion.cpp. Both pinned
  Hugging Face weight downloads failed with SSLV3_ALERT_HANDSHAKE_FAILURE; the
  installer correctly exited 1. Existing core/CSI/C2/C3 files were verified and
  reused. Do not call image generation installed or verified yet.
- Next: distinguish Python HTTPS transport from endpoint availability without
  relaxing TLS or hash checks. Current free memory was about 1.8 GB, below the
  existing image-tool threshold; address capacity after downloads, do not bypass
  the memory guard or stop unrelated user processes.

### Verified Source Copy In Progress

- Windows curl also rejected the pinned weight URL with SEC_E_UNTRUSTED_ROOT.
  No certificate checks or pinned hashes were relaxed.
- Human explicitly authorized copying the existing model files from k3 over
  SSH, read-only: no remote service restart or modification.
- K3 files under /opt/clover-k3/web-studio/data/image/ have the exact pinned
  sizes and SHA-256 values: SD 1.5 is 4,265,146,304 bytes with
  6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa;
  LCM-LoRA is 134,621,556 bytes with
  8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4.
- SCP is still running in terminal 0466a176-3ac4-46b7-97c2-d3629ace787c, writing
  only to the isolated base's downloads/pictures-from-k3/ staging directory.
  Last observed 299 MB (7 percent), about 2.9 MB/s; not a completion claim.
  Do not overwrite the partial transfer or retry it as if nothing happened.
- Await the terminal completion notification. Then verify both local hashes
  against launch/manifest.json, move only verified files to their manifest
  destinations, rerun picture setup to record success, and test real generation.
- The memory question did not return a human selection: the tool reported that
  the human was replying to terminal prompts directly and required no more
  questions or terminal input until the transfer completes. Do not treat this as
  approval to stop unrelated apps or bypass the 3.5 GB image-memory guard.

### Picture Installation Complete; Reload Approved

- SCP completed both files. Local SHA-256 values matched the pinned manifest
  exactly before the files were moved from staging into data/image/sd15/ and
  data/image/lora/. No existing target was overwritten.
- setup_fetch --components pictures completed with exit 0 at 2026-11-16 19:53:44.
  config/installed.json reports core, pictures, CSI, and C2/C3 all OK. The live
  /api/health at port 8799 now lists generate_image as installed, none unavailable.
- A real request through /api/chat (session image-install-real-20261116, model
  omni, prompt "Generate an image of a yellow teapot on a white table.") hit the
  real image tool but was refused by its existing memory guard: 2.0 GB free vs
  about 3.4 GB required. No image was generated; do not claim success yet.
- Human authorized desktop cleanup. Only one VS Code window exists, containing
  this session. A separate Chrome window received a normal CloseMainWindow()
  request, not a forced kill. Chrome showed another What's New window afterward;
  free memory remained about 2.2 GB. No user files or VS Code work were discarded.
- Largest memory users were this window's VS Code extension host (PID 22012)
  and renderer (PID 18752), roughly 5 GB combined. Do not kill them individually.
- Human explicitly approved: "Yes, save context and reload this window" to free
  editor/extension heaps. Reload is the next action. It may interrupt this chat;
  the Studio's terminal/server should be checked after VS Code returns.
- Resume from here: recheck port 8799 health and actual free memory. Do not
  re-download models, bypass the memory guard, relax TLS/hash checks, or stop
  unrelated apps. Once memory permits, repeat the real image request, inspect
  the PNG and model metadata, then verify it appears in the browser. Shared
  image generation applies to all chat-model choices; hosted chat inference was
  not established by merely installing the image tool.

### Reload Confirmation Identified

- Desktop automation initially could not focus VS Code, then successfully
  requested Reload Window after the desktop became available. The old extension
  host remained because VS Code was waiting for confirmation, not because a
  different reload command should be tried.
- Read the actual accessibility dialog: "A session is in progress. Are you sure
  you want to reload the window?" / "The session will stop if you reload the
  window." Buttons: Yes, Cancel. This is the exact reload already approved by
  the human. The next action is to invoke Yes on that verified dialog.
- The active chat will stop on reload. Human was asked to reopen this chat and
  say continue afterward. Do not claim reload or image generation completed
  until the resumed session checks the real process/memory/server state.
## Image generation installed and verified - 2026-11-17

- Pinned weights copied from k3 (read-only, authorized) and re-hashed locally: SD 1.5 6ce01616..., LCM-LoRA 8f90d840... both match launch/manifest.json. HTTPS to huggingface.co fails here (SEC_E_UNTRUSTED_ROOT); no TLS or hash check was relaxed.
- setup_fetch.py now exits 1 and reports failure when any requested component (including pictures) fails; it previously reported success if core passed.
- Setup restamped data/omni/precomputed/.clover-dataset.json revision (5f886993 -> a783d937, manifest-pinned); all 141 file entries identical. Harness asset-hashes.json updated for that one entry plus the 3 picture files (745 entries). Original ~/CloverAI untouched.
- Studio (port 8799) listed generate_image as installed. Real chat request was refused by the memory guard: 1.4 GB free, 3.4 GB needed (Studio's Omni llama-server holds ~3 GB, VS Code ~3.7 GB). Guard left unchanged.
- With the isolated Studio stopped (5.1 GB free), the real ImageGen code path generated a 512x512 image in 51.2 s (seed 7, SD 1.5 + LCM 4 steps): media/0ad41f593edb7ee09fd94fac50562e2c.png in the isolated base. Viewed: a yellow teapot on a wooden shelf.
- NOT verified: image generation through a running Studio chat on this 16 GB laptop. Chat model and diffusion together do not fit while VS Code is open. Options for the human: close VS Code/other apps before asking for an image, or have Studio unload the chat model while diffusion runs (a design change, not made).
- Isolated Studio is currently stopped. Restart: tests/live_model_integrity.py --base <isolated base> --port 8799 --reuse.
## k3 consumer reconciliation - 2026-11-17
- Context: compared the repo with k3 (/opt/clover-k3/studio-android/src/laptop/clover and /opt/clover-k3/web-studio/clover), LF-normalized on both sides. Raw hashes said 61 files differ; the real number is 14 (the files this work edited). k3 holds 8 files with changes that are not in git HEAD 06141cd6: the image client (sd-server + k3 worker), /api/worker/image, DNS cache, routing regexes, removal of hardcoded tech notes.
- Outcome: three-way merge (base=HEAD, mine, k3) written into the working tree. Kept k3 image path and its worker endpoint; replaced its silent sd-server pass with an explicit error; dropped k3's canned direct-answer plan, embedding-state fallback, 3D preview and (Clover finished without writing an answer.).
- Not verified: image through k3 worker end to end, APK build, phone. Flagged: k3 
un_python tool text in lfm3_native hardcodes a pack-cost recipe.
- Next (needs approval): back up k3 trees, push merged sources, build APK on k3.

### APK built on k3 - 2026-11-17 (approved by the human)
- Backup: /opt/clover-k3/studio-android/backup-nofallback-20261117T042351Z (src tree tar + previous 5.0.6 and beta apk/aab).
- Pushed 8 merged files into src/laptop/clover only (hash-verified, CRLF); web-studio service not touched. Ran build_apk.sh phone release (about 14 s).
- Verified on k3: 117 bundled .py files == k3 source; k3 worker image client and /api/worker/image present; canned plan, embedding-state fallback and hardcoded tech notes absent; beta apk signature verifies. 5.0.6 apk sha256 a6b9954e657a73a5e3c0a23dfb140ed2c16cb1f3498e30c5ee790800cba8047a.
- Version NOT bumped (same 5.0.6 name, outputs overwritten; old ones are in the backup). Not run: install on phone, a real image from the phone, other models from the phone, laptop Studio against k3.

## C1 own decode (A) vs C1-states + C2 + C3 (B) on k3 - 2026-11-17
Research on k3, no real users (human approved stopping `clover-k3-chat` for it; it was started again afterwards and answers health checks).
Setup: temperature 0, seed 1, one run per cell, k3 CPU (Ryzen 9 7950X3D). A = each model's own decoder: llama.cpp for Omni, Clover-35B and Nemotron; Clover's own whole-model engine for Kimi.
Prompts: math = "What is 17 multiplied by 23? Answer in one short sentence."; explain = "Explain in two sentences why the sky is blue."

### A: C1 inference + decoding (the model's own text)

| Model | Prompt | Prefill + decode = total (tokens) | Output text |
|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0) | math | 0.15 s + 0.30 s = 0.79 s (14 tok) | The product of 17 and 23 is 391. |
| Omni (LFM3-Omni-2B Q4_0) | explain | 0.13 s + 1.11 s = 1.71 s (50 tok) | The sky appears blue because sunlight, which is white, is scattered by the molecules in the atmosphere. This scattering is most effective for the shorter wavelengths of blue light, which are scattered in all directions, making the sky look blue to us. |
| Clover-35B-A3B (Q4_K_M) | math | 0.37 s + 2.98 s = 3.7 s (43 tok) | 17 multiplied by 23 equals 391. |
| Clover-35B-A3B (Q4_K_M) | explain | 0.34 s + 10.86 s = 11.68 s (153 tok) | Sunlight contains all colors, and Earth's atmosphere scatters sunlight in all directions. Blue light has a shorter wavelength and is scattered much more strongly than longer red or orange wavelengths, so the sky appears blue. |
| Nemotron 3.5 Super VL (Q4_0) | math | 1.86 s + 31.23 s = 33.47 s (120 tok) | 17 multiplied by 23 is 391. |
| Nemotron 3.5 Super VL (Q4_0) | explain | 1.39 s + 95.75 s = 97.71 s (365 tok) | Sunlight is made up of many colors of light, and as it enters Earth’s atmosphere, gas molecules scatter shorter wavelengths like blue and violet much more strongly than longer wavelengths like red. Because our eyes are more sensitive to blue than violet, and some violet light is absorbed by the upper atmosphere, the scattered light reaching our eyes from all directions makes the sky appear blue. |
| Kimi K3.1 Pro (Clover engine) | math | 48.0 s + 43.7 s = 91.7 s (10 tok) | 17 multiplied by 23 is 391. |
| Kimi K3.1 Pro (Clover engine) | explain | 48.2 s + 181.4 s = 229.7 s (44 tok) | The sky appears blue because sunlight is scattered by the molecules in Earth's atmosphere, and shorter blue wavelengths scatter more strongly than other colors. This scattered blue light reaches our eyes from all directions, making the sky look blue. |

### B1: A's text sent through k3's C1 -> C2 -> C3 stage route (what the Studio does today)

| Model | Prompt | C1 / C2 / C3 (ms) | Stage wall | Output text | Same as A |
|---|---|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0) | math | 0.43/15.39/120.0 | 0.6 s | The product of 17 and 23 is 391. | True |
| Omni (LFM3-Omni-2B Q4_0) | explain | 3.01/17.0/120.0 | 0.61 s | The sky appears blue because sunlight, which is white, is scattered by the molecules in the atmosphere. This scattering is most effective for the shorter wavelengths of blue light, which are scattered in all directions, making the sky look blue to us. | True |
| Clover-35B-A3B (Q4_K_M) | math | 2.04/13.19/120.0 | 0.6 s | 17 multiplied by 23 equals 391. | True |
| Clover-35B-A3B (Q4_K_M) | explain | 7.4/14.8/130.0 | 0.62 s | Sunlight contains all colors, and Earth's atmosphere scatters sunlight in all directions. Blue light has a shorter wavelength and is scattered much more strongly than longer red or orange wavelengths, so the sky appears blue. | True |
| Nemotron 3.5 Super VL (Q4_0) | math | 526.32/59.5/60.0 | 1.37 s | 17 multiplied by 23 is 391. | True |
| Nemotron 3.5 Super VL (Q4_0) | explain | 447.43/56.83/80.0 | 1.18 s | Sunlight is made up of many colors of light, and as it enters Earth’s atmosphere, gas molecules scatter shorter wavelengths like blue and violet much more strongly than longer wavelengths like red. Because our eyes are more sensitive to blue than violet, and some violet light is absorbed by the upper atmosphere, the scattered light reaching our eyes from all directions makes the sky appear blue. | True |
| Kimi K3.1 Pro (Clover engine) | math | 2.68/54.57/130.0 | 0.53 s | 17 multiplied by 23 is 391. | True |
| Kimi K3.1 Pro (Clover engine) | explain | 7.68/69.91/140.0 | 0.67 s | The sky appears blue because sunlight is scattered by the molecules in Earth's atmosphere, and shorter blue wavelengths scatter more strongly than other colors. This scattered blue light reaches our eyes from all directions, making the sky look blue. | True |

### B2: strict C1 states of the PROMPT only (no decoded reply) -> C2 plan -> C3 decode

| Model | Prompt | C1 / C2 / C3 (ms) | Total | Output text | Equals the prompt |
|---|---|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0) | math | 6.2 / 26.5 / 109.9 | 0.143 s | What is 17 multiplied by 23? Answer in one short sentence. | True |
| Omni (LFM3-Omni-2B Q4_0) | explain | 1.5 / 13.6 / 104.8 | 0.12 s | Explain in two sentences why the sky is blue. | True |
| Clover-35B-A3B (Q4_K_M) | math | 3.2 / 13.2 / 107.9 | 0.124 s | What is 17 multiplied by 23? Answer in one short sentence. | True |
| Clover-35B-A3B (Q4_K_M) | explain | 1.3 / 12.0 / 105.9 | 0.119 s | Explain in two sentences why the sky is blue. | True |
| Nemotron 3.5 Super VL (Q4_0) | math | 2.3 / 37.2 / 109.3 | 0.149 s | What is 17 multiplied by 23? Answer in one short sentence. | True |
| Nemotron 3.5 Super VL (Q4_0) | explain | 1.2 / 36.0 / 107.1 | 0.144 s | Explain in two sentences why the sky is blue. | True |
| Kimi K3.1 Pro (Clover engine) | math | 2.0 / 51.0 / 111.1 | 0.164 s | What is 17 multiplied by 23? Answer in one short sentence. | True |
| Kimi K3.1 Pro (Clover engine) | explain | 1.1 / 48.4 / 107.3 | 0.157 s | Explain in two sentences why the sky is blue. | True |

### What this shows, and what it does not
- B2 returned the prompt unchanged in 8/8 cases: C3 renders the text C1 was given, so with only the prompt there is no answer. No model's C3 wrote an answer from C1 states.
- B1 returned A's text unchanged in 8/8 cases. That shows C3 did not alter these 8 texts, not that C3 never alters text.
- C1 times of 1-6 ms in B1/B2 cannot be a forward pass of a 2.8T model; I did not verify what that runtime computes. Treat B's C1 time as not a model-inference time.
- A column timings are single runs on a CPU box that also ran other services; Kimi turn 2 continued from turn 1's context (prefill 38 tokens, not cold).
- Nemotron A uses a Q4_0 GGUF (ggml-org, sha256 fe82c76e...); Clover's own Nemotron runtime on k3 is a renderer over pruned data and does not decode text.
- Raw files: %TEMP%\clover-adb\c1-vs-c1c2c3\*.json (A, B1) and c1states_c2_c3.json (B2); k3 /tmp/c1states_c2_c3.json.
- k3 changes: Kimi scripts chmod +x; Nemotron GGUF kept at /opt/clover-k3/four-node-20261112/nemotron (72 GB); temporary servers stopped.

## Correction and real B2 (C1 hidden states -> C2 -> C3) - 2026-11-17, same day, supersedes the B2 table above
Direction (human): "we have already done all of this ... follow clover and look at main and older commits". It had been done: branch `feature/c1-inference-only` (commit 3ae993ea), token-economizer cycle 6 (d1704dc1) and cycle 7 (0f785f9f), in `docs/runtime-context/token-economizer/01-token-economizer-context.md` (lines 638-939 at 0f785f9f). I read that only after being told.

### Correction
- The earlier B2 table in this file used `c1_infer_stage(prompt)`. For these models that function is an embedding-table lookup (Kimi: embedding + RMS norm; Nemotron: a bounded mix), not a forward pass. Its 1-6 ms C1 times were lookups. That table is not C1 inference; treat it as superseded. (My earlier sentence that I had not verified this is now verified in `omni_c2_c3.py` `get_c1_states_fn`.)
- k3 deployed `omni_c2_c3.py` already contains the branch's `c1_infer_states` and `full_ncm`; my repo copy lacks the per-model stage table (removed as a fallback), so this run used k3's runtime.

### Real B2: C1 inference only (one forward pass of the model's own GGUF, llama-server `--embeddings --pooling none`, no token sampled) -> C2 (`c2_plan`, JEPA off) -> C3 (`c3_decode` full_ncm, all 12 blocks)
Method is cycle 7's probe (`cycle-7/k3/c1i_probe.py`) with the same controls: the same C2/C3 given all-zero states and random states of the same norm. One run per cell, k3 CPU, chat template prefix as the Studio builds it.

| Model | Prompt | C1 forward (ms) / C2 (ms) / C3 (ms) | Output text from C3 | Equals the prompt | Same text for real, zero and random states |
|---|---|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0, d=2048) | math | 126.2 / 9.3 / 478.6 | What is 17 multiplied by 23? Answer in one short sentence. | True | True |
| Omni (LFM3-Omni-2B Q4_0, d=2048) | explain | 118.8 / 8.4 / 202.0 | Explain in two sentences why the sky is blue. | True | True |
| Clover-35B-A3B (Q4_K_M, d=2048) | math | 469.4 / 9.2 / 485.7 | What is 17 multiplied by 23? Answer in one short sentence. | True | True |
| Clover-35B-A3B (Q4_K_M, d=2048) | explain | 417.2 / 8.5 / 207.1 | Explain in two sentences why the sky is blue. | True | True |
| Nemotron 3.5 Super VL (Q4_0, d=4096) | math | 1586.4 / 35.1 / 831.3 | What is 17 multiplied by 23? Answer in one short sentence. | True | True |
| Nemotron 3.5 Super VL (Q4_0, d=4096) | explain | 1377.6 / 32.8 / 212.2 | Explain in two sentences why the sky is blue. | True | True |
| Kimi K3.1 Pro (Clover engine, d=7168) | math, explain | not measured | not measured | - | - |

### A (unchanged, measured earlier in this file): each model's own decode, for comparison with B2's texts
Texts are in the A table above. B2's C3 text differs from every A text in all 6 measured cells; it is the prompt.

### Reading
- With real C1 states, C3's text is the prompt word for word in 6 of 6 cells, and identical for real, zero and random states in 6 of 6. This reproduces cycle 7 (Omni, 8 of 8 prompts) on Clover-35B and Nemotron too.
- So no model's reply in this pipeline comes from C1's inference or C2's plan today: C3 was trained to copy or re-render a sentence it is given (cycle 6: no question-answer pairs in its training), and the decode limits output to the input's words.
- C1 forward times are real: 118-126 ms Omni, 417-469 ms Clover-35B, 1,378-1,586 ms Nemotron (prompt of about 40-60 tokens including the chat prefix). C2 8-35 ms. C3 200-830 ms (the first call per model is slower).
- What would make C3 answer is stated in cycle 6/7 and was left to the human: a C3 trained to write from C1 states, or a C2 that writes plans. Not built here.

### Not covered
- Kimi: the engine saves layer-92 rows only for decode steps (`PF_SAVE_ROWS`, `whole-model.c` near line 1100), not for prompt positions. A prompt-state export would mean modifying the C engine; not done. Kimi's B2 is therefore not measured. Kimi's A is in the A table.
- Two prompts per model, one run each. C3 trained on Stage-3 HybridSFT Omni states; Studio and this run use the official Omni GGUF (cycle 6 caveat). C3/C2 for Clover-35B and Nemotron come from their precomputed stage folders on k3 and were not trained on these GGUFs' states.
- Raw: `%TEMP%\clover-adb\b2_*.json` (copied from k3 `/tmp/b2_*.json`), scripts `b2_real.py`, `k3-b2-run.sh` there.
- k3: `clover-k3-chat` stopped and started again twice for this (health OK each time); my llama-servers were launched by the script and stopped by it; the 72 GB Nemotron GGUF is still in `/opt/clover-k3/four-node-20261112/nemotron`.

## Final: A (C1 inference + decoding) vs B (C1 inference only + C2 plan + C3 decode), real prompts, outputs and timings - 2026-11-17
Direction (human): add a flag to the engine if needed, and show real numbers, prompts and outputs for both approaches. Settings: k3 CPU, temperature 0 seed 1 for A, one run per cell.
A = the model's own decoder (llama.cpp GGUF for Omni, Clover-35B, Nemotron; Clover's whole-model engine for Kimi). B = C1's real last-layer hidden states for the same prompt (no token sampled) -> `c2_plan` -> `c3_decode` (12 NCM blocks); Omni, Clover-35B, Nemotron via `llama-server --embeddings --pooling none`, Kimi via the engine change below. B's C1 times are forward-pass times of the real model; for Kimi it is the engine's whole request time (93 layers plus the 1-token head).

### Prompt `math`: "What is 17 multiplied by 23? Answer in one short sentence."

| Model | A: C1 own decode output | A time | B: C3 output | B time (C1 / C2 / C3) | B equals prompt | B same for real / zero / random states |
|---|---|---|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0) | The product of 17 and 23 is 391. | 0.79 s total (prefill 0.15 s + decode 0.30 s, 14 tok) | What is 17 multiplied by 23? Answer in one short sentence. | C1 126.2 ms / C2 9.3 ms / C3 478.6 ms = 0.61 s | True | True |
| Clover-35B-A3B (Q4_K_M) | 17 multiplied by 23 equals 391. | 3.7 s total (prefill 0.37 s + decode 2.98 s, 43 tok) | What is 17 multiplied by 23? Answer in one short sentence. | C1 469.4 ms / C2 9.2 ms / C3 485.7 ms = 0.96 s | True | True |
| Nemotron 3.5 Super VL (Q4_0) | 17 multiplied by 23 is 391. | 33.47 s total (prefill 1.86 s + decode 31.23 s, 120 tok) | What is 17 multiplied by 23? Answer in one short sentence. | C1 1586.4 ms / C2 35.1 ms / C3 831.3 ms = 2.45 s | True | True |
| Kimi K3.1 Pro (Clover engine) | 17 multiplied by 23 is 391. | 91.7 s total (prefill 48.0 s + decode 43.7 s, 10 tok) | What is 17 multiplied by 23? Answer in one short sentence. | C1 37.89 s / C2 49.1 ms / C3 487.3 ms = 38.43 s | True | True |

### Prompt `explain`: "Explain in two sentences why the sky is blue."

| Model | A: C1 own decode output | A time | B: C3 output | B time (C1 / C2 / C3) | B equals prompt | B same for real / zero / random states |
|---|---|---|---|---|---|---|
| Omni (LFM3-Omni-2B Q4_0) | The sky appears blue because sunlight, which is white, is scattered by the molecules in the atmosphere. This scattering is most effective for the shorter wavelengths of blue light, which are scattered in all directions, making the sky look blue to us. | 1.71 s total (prefill 0.13 s + decode 1.11 s, 50 tok) | Explain in two sentences why the sky is blue. | C1 118.8 ms / C2 8.4 ms / C3 202.0 ms = 0.33 s | True | True |
| Clover-35B-A3B (Q4_K_M) | Sunlight contains all colors, and Earth's atmosphere scatters sunlight in all directions. Blue light has a shorter wavelength and is scattered much more strongly than longer red or orange wavelengths, so the sky appears blue. | 11.68 s total (prefill 0.34 s + decode 10.86 s, 153 tok) | Explain in two sentences why the sky is blue. | C1 417.2 ms / C2 8.5 ms / C3 207.1 ms = 0.63 s | True | True |
| Nemotron 3.5 Super VL (Q4_0) | Sunlight is made up of many colors of light, and as it enters Earth’s atmosphere, gas molecules scatter shorter wavelengths like blue and violet much more strongly than longer wavelengths like red. Because our eyes are more sensitive to blue than violet, and some violet light is absorbed by the upper atmosphere, the scattered light reaching our eyes from all directions makes the sky appear blue. | 97.71 s total (prefill 1.39 s + decode 95.75 s, 365 tok) | Explain in two sentences why the sky is blue. | C1 1377.6 ms / C2 32.8 ms / C3 212.2 ms = 1.62 s | True | True |
| Kimi K3.1 Pro (Clover engine) | The sky appears blue because sunlight is scattered by the molecules in Earth's atmosphere, and shorter blue wavelengths scatter more strongly than other colors. This scattered blue light reaches our eyes from all directions, making the sky look blue. | 229.7 s total (prefill 48.2 s + decode 181.4 s, 44 tok) | Explain in two sentences why the sky is blue. | C1 34.27 s / C2 42.5 ms / C3 204.5 ms = 34.52 s | True | True |

### Reading
- B's output is the prompt itself in 8 of 8 cells, and does not change when C1's states are replaced by zeros or random vectors (8 of 8). A's outputs are the models' own, different answers. This matches cycle 7 (`feature/c1-inference-only`, 0f785f9f): C3 copies the text it is given; with C1 stopping at inference nothing writes an answer.
- B is cheaper only because nothing is written (sum of C1 + C2 + C3 per prompt, math / explain, against A's total): Omni 0.61 s / 0.33 s vs 0.79 s / 1.71 s, Clover-35B 0.96 s / 0.63 s vs 3.70 s / 11.68 s, Nemotron 2.45 s / 1.62 s vs 33.47 s / 97.71 s, Kimi 38.43 s / 34.5 s (prompt pass through 93 layers, no decode) vs 91.7 s / 229.7 s. B's time buys no answer.
- One run per cell; Kimi's A turn 2 continued from turn 1's context; C3's first call per model is slower (479-831 ms) than later ones (200-212 ms).

### Engine change (Kimi K3, `clover-ai-system/src/code/common/whole-model.c` and `src/scripts/run-model.sh`, uncommitted in the working tree)
- Two off-by-default environment flags, both added to `run-model.sh`'s `env -i` allow-list (without that they are dropped):
  - `PF_SAVE_PROMPT_STATES=file`: after the prompt pass, writes the layer-92 output row of every prompt position (N x 7168 float32, before the final normalisation; the runtime applies the model's `norm.bf16.bin` RMSNorm).
  - `PF_IDS_FILE=file`: the prompt is exactly the token ids in the file (the ids the C2/C3 runtime builds), through `ids_for_run`, which falls back to `prompt_ids` when unset.
- Flag-off identity check: the pre-change `whole-model.c` (from the backup, sha256 c370a06b8fd532e1, equal to the repo file before the edit) was built in a scratch tree and run next to the new build, chat template, same prompt, 1 decoded token. Prompt ids, next token (id 1810, "17"), generated ids and the engine's arithmetic line were identical: {'prompt_ids': True, 'next_id': True, 'generated_ids': True, 'arithmetic': True}. This is not the repo's recorded 1000-token check against the pod's chain (the recorded inputs are not on k3), so identity is shown for this one prompt, not the pod's full reference.
- k3: backup in `/opt/clover-k3/clover-kimi-k3/backup-promptstates-20261117T080713Z`; the new source and rebuilt binary are in place there; `clover-k3-chat` was stopped for the runs and started again (health OK); scratch trees removed.
- Not done: nothing committed or pushed; the flags are not used by the Studio.
- Raw files: `%TEMP%\clover-adb\c1-vs-c1c2c3\*.json` (A), `b2_*.json` (B), `kimi_identity.json`, `kimi_export.json`; scripts `b2_real.py`, `kimi_b2.py` there.

## Correction 2: the full wired route (HybridAgent-8B plan -> NCM render with the model's states) answers; it is the real B - 2026-11-17
Direction (human): C1 hands its state to C2 which plans and C3 decodes the plan; this is the HybridAgent + NCM concept, already trained, assets already on k3. Read on k3 (`clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-agent-engine.py`): `run_agent_planning` (HybridAgent-8B, `run_hybridlink` latent reasoning, writes the plan text) and `render_ncm_with_k3_experts` (the selected model's real engine runs on the plan's tokens -> its 7168/4096/2048-d states -> its native HybridLink + bridge -> NCM + pointer decoder). The k3 `/api/worker/chat` plain call runs exactly this.
The two earlier B tables in this file (embedding lookup; Studio C2/C3 on C1 prompt states) tested the Studio's lightweight C2 and a copy-trained C3, not this route. Their finding (C3 copies its input) stands for those parts only.

### Measured earlier today (k3 `/api/worker/chat`, no draft; raw in `%TEMP%\clover-adb\k3-stage-matrix\*.json`), one run per cell

| Model | Prompt | Wall | C2 HybridAgent plan | C3 NCM render (incl. the model's states) | Reply (verbatim) |
|---|---|---|---|---|---|
| Omni | math | 105.28 s | 97.0 s | 7.7 s | Hi there! Let me help you with this math problem.  The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation.  I'll use the Python interpreter to compute this accurately.  The calculation 17 * 23 = 391 is correct.  So the answer is 391. |
| Omni | explain | 70.19 s | 63.4 s | 6.2 s | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering.  So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer |
| Clover-35B-A3B | math | 100.16 s | 97.3 s | 2.3 s | Hi there! Let me help you with this math problem.  The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation.  I'll use the Python interpreter to compute this accurately.  The calculation 17 * 23 = 391 is correct.  So the answer is 391. |
| Clover-35B-A3B | explain | 65.32 s | 63.3 s | 1.5 s | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering.  So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer |
| Nemotron 3.5 Super VL | math | 100.33 s | 96.9 s | 2.9 s | Hi there! Let me help you with this math problem.  The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation.  I'll use the Python interpreter to compute this accurately.  The calculation 17 * 23 = 391 is correct.  So the answer is 391. |
| Nemotron 3.5 Super VL | explain | 66.36 s | 63.6 s | 2.1 s | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering.  So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer |
| Kimi K3.1 Pro | math | 101.09 s | 96.8 s | 3.7 s | Hi there! Let me help you with this math problem.  The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation.  I'll use the Python interpreter to compute this accurately.  The calculation 17 * 23 = 391 is correct.  So the answer is 391. |
| Kimi K3.1 Pro | explain | 66.89 s | 63.7 s | 2.7 s | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering.  So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer |

### What this shows
- This route writes text. The reply is not the prompt. C2's plan time (63-97 s) is almost the whole wall time; C3 render is 1.5-7.7 s.
- The reply is byte-identical across Omni, Clover-35B, Nemotron and Kimi for the same prompt (math sha e4063bfe, explain 20526883). So the text comes from HybridAgent-8B, which reads the prompt text itself (`run_agent_planning`: `backbone(input_ids=...)` of the HybridAgent model, its own `last_h`). The selected model's states enter only in the render step, over the plan's tokens. In this route the choice of model does not change what is said.
- That differs from "C1's state handed to C2": the planner is not conditioned on the selected model's hidden states for the question. Whether a HybridLink that maps a model's states into HybridAgent's latent space is meant to drive planning is in the research context (cycles 49-53) and was not tested here.
- The explain replies are cut mid-sentence ("...scatter more than the longer"), and no tool call ran for the math prompt although the plan says it will use Python. Neither was investigated.
- A still gives each model's own answer in 0.8-230 s; this route gives one shared answer in 65-105 s. On these 2 prompts, A is faster and the answers differ by model; B-full is slower and identical across models. Quality was not scored.

## Per-model Clover flow, set up and run on k3 - 2026-11-17 (each model's own engine + HybridLink + bridges + NCM)
Direction (human): every model has its own HybridAgent bridges and NCM data trained for it and uploaded from k3; set up the flow clearly, do not conclude early, show what it produces and where the issues are compared to C1 alone.
Context read first: `src/models/benchmark_3_models_70_tasks.py` (`ModelSpecificNCMRenderer`: tokenises plan windows in the model's vocabulary, runs the model's compiled engine for `h_backbone`, then its trained `HybridLink` + `ComboBridge` + `ExpertBridge` into the 11-level NCM + pointer decoder, decoded in the model's own vocabulary); per-model files exist on k3 for all three (`/opt/clover-k3/{lfm3,nemotron,clover-35b}-computed/...bridges.pt`, `pruned/`), HF dataset `Santhoshsathish94/clover-ai-studio` has folders liquid-lfm3, nemotron, clover-35b, kimi. In that benchmark the plan text comes from a separate baseline run (`raw_k3_out`) and C3 renders it.
Run: `/tmp/flow_model.py <model>` on k3 (script copy `%TEMP%\clover-adb\flow_model.py`), chat worker stopped for memory then started again (health OK). Two plan sources per model: H = the plan windows HybridAgent-8B wrote on k3 for the prompt (worker route), A = that model's own C1 text from the earlier run, split into windows as the benchmark does. One run per cell; render time is the second of two calls (the first includes warm-up).

| Model | Prompt | Plan source | Plan text (words) | Render output (that model's own HybridLink + bridges + NCM) | Render time | Render == plan |
|---|---|---|---|---|---|---|
| LFM3-1.7B chain | math | H HybridAgent-8B plan | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. (51) | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. | 7.287 s | True |
| LFM3-1.7B chain | math | A C1's own text | The product of 17 and 23 is 391. (8) | The product of 17 and 23 is 391. | 1.555 s | True |
| LFM3-1.7B chain | explain | H HybridAgent-8B plan | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer (45) | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer | 5.831 s | True |
| LFM3-1.7B chain | explain | A C1's own text | The sky appears blue because sunlight, which is white, is scattered by the molecules in the atmosphere. This scattering is most effective for the shorter wavelengths of blue light, which are scattered in all directions, making the sky look blue to us. (42) | The sky appears blue because sunlight, which is white, is scattered by the molecules in the atmosphere. This scattering is most effective for the shorter wavelengths of blue light, which are scattered in all directions, making the sky look blue to us. | 4.873 s | True |
| Clover-35B-A3B | math | H HybridAgent-8B plan | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. (51) | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. | 2.169 s | True |
| Clover-35B-A3B | math | A C1's own text | 17 multiplied by 23 equals 391. (6) | 17 multiplied by 23 equals 391. | 0.648 s | True |
| Clover-35B-A3B | explain | H HybridAgent-8B plan | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer (45) | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer | 1.377 s | True |
| Clover-35B-A3B | explain | A C1's own text | Sunlight contains all colors, and Earth's atmosphere scatters sunlight in all directions. Blue light has a shorter wavelength and is scattered much more strongly than longer red or orange wavelengths, so the sky appears blue. (35) | Sunlight contains all colors, and Earth's atmosphere scatters sunlight in all directions. Blue light has a shorter wavelength and is scattered much more strongly than longer red or orange wavelengths, so the sky appears blue. | 1.015 s | True |
| Nemotron 3.5 Super VL | math | H HybridAgent-8B plan | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. (51) | Hi there! Let me help you with this math problem. The question is asking for the result of 17 multiplied by 23. This is a straightforward multiplication calculation. I'll use the Python interpreter to compute this accurately. The calculation 17 * 23 = 391 is correct. So the answer is 391. | 2.653 s | True |
| Nemotron 3.5 Super VL | math | A C1's own text | 17 multiplied by 23 is 391. (6) | 17 multiplied by 23 is 391. | 0.729 s | True |
| Nemotron 3.5 Super VL | explain | H HybridAgent-8B plan | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer (45) | First, maybe I should recall the basic explanation. I think it's something called Rayleigh scattering. So when sunlight enters the Earth's atmosphere, the molecules and small particles in the air scatter the light. The shorter wavelengths, like blue and violet, scatter more than the longer | 1.83 s | True |
| Nemotron 3.5 Super VL | explain | A C1's own text | Sunlight is made up of many colors of light, and as it enters Earth’s atmosphere, gas molecules scatter shorter wavelengths like blue and violet much more strongly than longer wavelengths like red. Because our eyes are more sensitive to blue than violet, and some violet light is absorbed by the upper atmosphere, the scattered light reaching our eyes from all directions makes the sky appear blue. (66) | Sunlight is made up of many colors of light, and as it enters Earth’s atmosphere, gas molecules scatter shorter wavelengths like blue and violet much more strongly than longer wavelengths like red. Because our eyes are more sensitive to blue than violet, and some violet light is absorbed by the upper atmosphere, the scattered light reaching our eyes from all directions makes the sky appear blue. | 2.351 s | True |

Render equals its plan after whitespace normalisation in 12 of 12 cells (word ratio 1.0000 each). Note: in cells where the raw strings differ, the strings are equal after whitespace normalisation, so the difference is whitespace only (window joins).

### What it produces, and the issues seen against C1 alone
1. **C3 renders; it does not write.** For all three models, with their own trained bridges and NCM, the render is the plan, word for word (12 of 12). Whatever the plan says is what the reply says. This holds for plans from HybridAgent-8B and for plans from the model's own C1 text. So a different answer can only come from a different plan.
2. **The plan writer decides the content, and it is not the selected model.** HybridAgent-8B wrote byte-identical plans for Omni, Clover-35B, Nemotron and Kimi (the worker route); the per-model engine, HybridLink and NCM then only render them. Choosing a model changes the render time, not the answer.
3. **HybridAgent's plan is longer and chattier than C1's answer.** math: 51 words ("Hi there! Let me help you with this math problem. ... The calculation 17 * 23 = 391 is correct. So the answer is 391.") against 6-8 words from C1 ("17 multiplied by 23 equals 391."). The 391 is correct in both.
4. **The explain plan is its thinking text, cut off.** "First, maybe I should recall the basic explanation..." ends mid-sentence ("scatter more than the longer"). `run_agent_planning` in `clover-agent-engine.py` has `max_new: int = 360` as its default, which is the cause I can see in the code; I did not change it and did not re-run with a larger value. C1's explain answers are complete two-sentence answers.
5. **The math plan says "I'll use the Python interpreter" but no tool call ran** (`tool_calls` empty in the worker response); the plan promises something that did not happen.
6. **Cost.** Render adds 0.6-7.3 s per reply (LFM3 chain slowest, Clover-35B fastest) and changes nothing in these 12 cells. The plan step is the expensive part: HybridAgent-8B took 63 s (explain) and 97 s (math) on k3's CPU, against C1 alone 0.8-1.7 s (Omni), 3.7-11.7 s (Clover-35B), 33-98 s (Nemotron), 92-230 s (Kimi).
7. **Not an end-to-end win here.** On these two prompts, C1 alone gives each model's own, shorter, complete answer faster for three of four models; the HybridAgent + NCM route gives one shared, longer answer, one of them truncated. Answer quality was not scored beyond reading the texts.

### Not covered / unknown
- Kimi: `ModelSpecificNCMRenderer` has no Kimi branch; Kimi's render ran through the worker route (K3 engine) earlier, whose reply equalled the plan (math exact; explain differs by one character in length (291 vs 292), not inspected). Its `q2k3-bridge-trunk-0` and per-model `hybridagent` files were not exercised separately.
- The LFM3 chain here is LFM3-1.7B (its own bridges and NCM), not the Omni 2B the Studio uses; Omni's A text was used as the plan source for it. The worker maps the `omni` key to `lfm3-1.7b`.
- HybridAgent's intended latent path was not exercised: `NativeHybridLink` was trained to map a model's hidden states toward next-token embeddings and a `<latent>` readout, but this flow gives it the states of the plan tokens, so the plan text is fixed before it runs. Whether a plan can be read out of the model's own states for the question (the concept described by the human) was not tested; the research context (cycles 49-53) and `*_hybrid_ncm.py` are where that would be built.
- Per-model `jepa/` folders and `clover_35b_hybridagent_final.pt` were not used. One run per cell, two prompts, no quality scoring.
- Raw: `%TEMP%\clover-adb\flow_*.json`, `flow_inputs.json`, `flow-compare.py`.

## Model-native latent plan (C1 states of the prompt -> the model's own HybridLink -> latent steps + cosine readout), LFM3-1.7B - 2026-11-17
Direction (human): build the case where the model's own C1 states for the prompt drive C2 (HybridLink) and C3 renders the plan; dataset `Santhoshsathish94/clover-ai-studio` holds per-model `hybridagent/`, `bridges/`, `ncm/`, `trunks/`, `jepa/` (six folders per model).
Context read: the dataset card; `lfm3_hybrid_ncm.py` (the LFM3 `Lfm3NativeHybridLink` is trained by `train_and_verify_lfm3_native_hybrid_and_ncm`: 250 AdamW steps on the hidden states of six hardcoded sentences, loss 1.0*cos + 0.2*mse to the next token's embedding); Clover-35B's `hybridagent/clover_35b_hybridagent_final.pt` holds `clover_35b_native_link_512/1024`, `clover_35b_hybridlink_2048` and `qwen_to_clover_35b_bridge`, trained on 8 GPUs in 3 stages (summary: eval_cos_2048 0.9812); every `inputs_embeds` latent loop in the repo (`clover-agent-engine.py`, `conversational-chat-engine.py`, `train_*`) runs the Qwen HybridAgent backbone, not the target model.
Run (k3, `/tmp/latent_lfm3.py`, copy `%TEMP%\clover-adb\latent_lfm3.py`): LFM3-1.7B bf16 via HF transformers on CPU (accepts `inputs_embeds`), chat template, 48 steps, full recompute per step, one run per cell. Per step: h = last-layer state; R = the checkpoint's `lfm3_native_hybrid_link`(h); readout = nearest embedding by cosine. Three modes on the same weights: `lmhead` (the model's own greedy decode, control), `link_discrete` (feed the readout token), `link_latent` (feed R(h) as the next input embedding).

| Prompt | Mode | Steps / time (median ms per step) | Readout token == lm_head top-1 | Mean readout cosine | Output (48 steps max) |
|---|---|---|---|---|---|
| math | lmhead | 48 / 9.43 s (204.7 ms) | 16/48 | 0.4316 | We need to compute 17 * 23. Let's do it: 17 * 20 = 340, 17 * 3 = 51, total = 391. So answer is 391. The instruction: " |
| math | link_discrete | 48 / 9.23 s (200.5 ms) | 10/48 | 0.6453 | We need.4 million pounds for a new cycle lane along Mill Road. Harbours Road.. equalize differential pressure and verify downstream telemetry.. equalize downstream telemetry.. SpecMD Least-Stale slot c |
| math | link_latent | 48 / 9.37 s (204.6 ms) | 6/48 | 0.3518 | We and3bours AIAI AI.4ide AI.,...,....... Liquid Liquid Liquid Liquid Liquid Liquid switches Liquid Liquid switches switches switches switches switches Liquid Liquid Liquid switches switches Liquid Liquid Liquid Liquid switches |
| explain | lmhead | 48 / 9.23 s (202.4 ms) | 13/48 | 0.3271 | We need to answer: "Explain in two sentences why the sky is blue." This is a classic question about Rayleigh scattering. The user wants exactly two sentences. So I'll craft a concise but accurate explanation in two sentences. First sentence |
| explain | link_discrete | 48 / 9.3 s (204.0 ms) | 25/48 | 0.5258 | We. LFM3. Harbourside CounciltById3_LocalPosition232243333, 2,3,3,3,3,3,3,3,3,3,3,3, |
| explain | link_latent | 48 / 9.23 s (202.3 ms) | 1/48 | 0.319 | We AIron AI,系统 Liquid Liquid Liquid N N....2222222222....................... |

### What it shows
- The model's own `lm_head` decode answers ("We need to compute 17 * 23 ... total = 391"; "...classic question about Rayleigh scattering"), as a thinking-style start, cut by the 48-step cap.
- The LFM3 link used as the plan writer does not: `link_discrete` and `link_latent` output mostly words from its six training sentences (word counts from those sentences: 25/29, 14/17 discrete; 26/30, 4/9 latent) or repeated tokens. The readout equals `lm_head`'s top-1 on only 1-25 of 48 steps, mean cosine 0.32-0.65.
- So, for LFM3-1.7B, this deployed link cannot write a plan from the model's own states. This follows from how it was trained (250 steps on six sentences, next-token embedding as target); it is a result about this checkpoint, not about the HybridLink idea or the other models' links.
- Cost per step was about 200 ms for all three modes (a full recompute; no KV cache), so the latent loop itself is not faster than `lm_head` decode here.

### Not covered
- Clover-35B, Nemotron and Kimi links (8-GPU, 3-stage training) were not run: their engines take token ids, not embeddings (`forward_ids`), so a latent loop needs an embeddings entry point added, or the Qwen HybridAgent + `qwen_to_*_bridge` path that the checkpoints suggest is the intended C2.
- The C3 render of these readouts was not run: the plans are not usable text.
- `porting-models/` (trainer `train_clover_35b_hybrid_ncm_gpu.py`) was not read; whether its 3-stage HybridSFT tunes the backbone is unknown to me.
- Raw: `%TEMP%\clover-adb\latent_lfm3.json`, `latent-check.py`.

## Hugging Face dataset assets vs what was run on k3, and a re-run with the dataset's checkpoint - 2026-11-17
Direction (human): the HF dataset `Santhoshsathish94/clover-ai-studio` holds each model's own trained HybridAgent/bridges/NCM; use those and see the results.
- Contents (file listing via the HF API, 3,655 files, 1,657 GB): per model `trunks/ experts/ hybridagent/ ncm/ bridges/ jepa/`. No file path contains qwen / hybridagent-8b / togethercomputer; no file over 3 GB under any `hybridagent/` folder. The Qwen3-8B-based HybridAgent-8B that wrote the plans in the worker route is NOT in the dataset (it is in k3 `/root/hf`). `hybridagent/` holds small trained adapters: Clover-35B 0.32 GB, Nemotron 1.5 GB, LFM3 0.05 GB, Kimi 12 GB (about 2 GB checkpoints). The Clover-35B checkpoint contains `qwen_to_clover_35b_bridge`.
- No training corpora or question-answer data are in the dataset (weights, trunks, experts, bridges, NCM, JEPA adapters plus `real_clover_jepa_dataset.npz`); this matches cycle 6's finding that no question-answer data for C3 exists on HF or k3.
- Byte comparison (sha256 against the HF-published oid, run on k3): IDENTICAL for `lfm3_hybrid_ncm_bridges.pt`, `nemotron_hybrid_ncm_bridges.pt`, `clover_35b_hybrid_ncm_bridges.pt`, `clover_35b_hybridagent_final.pt`, `nemotron_hybridagent_final.pt`, `kimi_k31_hybridagent_final.pt`. DIFFERENT for `checkpoint-clover-true-hybrid-agent-ncm.pt` (HF 2,069,336,623 B sha c2c7fc88...; the k3 copy used by the renderer 2,069,331,463 B sha fc72938b...; 5,160 bytes smaller). The k3 stage folder has no `kimi_k31_hybrid_ncm_bridges.pt`.
- Re-run: downloaded the HF copy of that checkpoint to `/opt/clover-k3/hf-dataset-check/` (sha256 verified), ran `ModelSpecificNCMRenderer` for LFM3-1.7B, Clover-35B and Nemotron with it (`flow_model_hf.py`, same two plan sources and prompts as the earlier flow). Result: the render equals the earlier render in 12 of 12 cells and equals its plan in 12 of 12 (whitespace-normalised); render times within about 0.1 s of the earlier run. So the 5 KB difference between the two copies made no difference to these outputs.
- What this does not show: the earlier finding (render equals plan) holds with the dataset's own assets for these 12 cells, two prompts, one run. It says nothing about the dataset's HybridAgent checkpoints as plan writers for the other models (Clover-35B, Nemotron, Kimi links were not run as latent planners; their engines take token ids). `jepa/` folders were not used.
- Raw: `%TEMP%\clover-adb\flowhf_*.json`, `flowhf-compare.py`; k3 `/tmp/hf_vs_k3.json`.

## Kimi only: C1 states of the prompt -> Kimi's own HybridLink (C2) -> Kimi's NCM (C3) - 2026-11-17
Direction (human): run only Kimi; C1 inference hands its state to C2 planning and C3 decodes the plan; also: how can the models give identical output, and compare the two copies of the shared checkpoint.
### Checkpoint copies
- `checkpoint-clover-true-hybrid-agent-ncm.pt`: the HF copy and the k3 copy are identical tensor by tensor (all 14 top-level sections, no key missing from either; `torch.equal` on every tensor). The 5,160-byte file-size difference is not in any value.
### What was run (k3, one run per cell)
- C1: Kimi's own engine (`whole-model`, 93 layers) on the prompt with its chat template, ending where the answer starts (math prompt ids end `... <|open|>response<|sep|>`; the engine's next token is `17` (id 1810) for math and `The` (id 1008) for explain). Engine change: `PF_SAVE_PROMPT_STATES` now also writes `<file>.snaps` (the 8 snapshots, 8 x N x 7168 float32); still off by default; backup `backup-promptstates-20261117T080713Z`.
- Final state: the engine's own `normalization_process` (exported by `pipeline-stage.so`, called through ctypes) on the residual row and the 8 snapshots, reading `output_attn_res_norm`, `output_attn_res_proj` and `norm` from `model-00094-of-000096.safetensors`: a softmax-weighted aggregate of 9 sources then RMS-normalised. A plain RMSNorm of the layer-92 rows (my first attempt) is the wrong input and gave the same punctuation, so it was not the cause.
- C2: `k31_native_hybrid_link` (7168 -> 3584 -> 7168) from `kimi_k31_hybridagent_final.pt`, then cosine readout over Kimi's 163,840 embedding rows. Teacher-forced over the prompt positions (the engine takes token ids only, so no latent feedback loop).
- C3: the runtime's NCM chain (`c1_infer_stage` + `c2_plan` + `c3_decode`, full NCM) on the readout text.

| Prompt | Positions | Final state (ms, all positions) | Link + readout (ms) | Next-token agreement inside the prompt | Mean cosine | Readout at the last position (top 5) | Engine's first answer token and its rank under the link | C3 render of the readout plan |
|---|---|---|---|---|---|---|---|---|
| math | 36 | 4.9 | 840.6 | 0/35 | 0.2498 | [',', '.', ':', ' ', '-'] | `17` rank 767 of 163,840 | .., , and,,,,,,,,,,,,,,,,,,,,,,,,.,,, and, (equals the readout plan: True) |
| explain | 31 | 4.3 | 810.7 | 0/30 | 0.2563 | [',', '.', ':', ' ', '-'] | `The` rank 107 of 163,840 | .., , and,,,.,,,,,,.,,,,,,,,,,,,,, (equals the readout plan: True) |

Readout per position, math: `.`, `.`, `,`, ` `, `,`, ` and`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,`, `,` ...

### What it shows
- Kimi's own link does not write a plan from Kimi's own states: the readout is commas, full stops and `and`; it agrees with the prompt's next token at 0 of 65 positions; the engine's actual first answer token is at rank 767 (`17`) and 107 (`The`) under the link.
- C3 then renders that readout unchanged (`equals the readout plan: True`, 2 of 2), as it did for every plan it was given in the other runs: it re-renders, it does not write.
- This agrees with the checkpoint's own training summary (`kimi_k31_hybridagent_training_summary.json`): final-stage eval `eval_cos_nat` 0.276 (my mean readout cosine is 0.25), `eval_ntp` 4.24, perplexity 69; the three stages ran 40, 60 and 60 steps (52 s, 76 s, 76 s on 8 GPUs). Nemotron's summary is similar in scale (stages of 40/60/60 steps, `eval_cos_nat` 0.63, perplexity 863). Clover-35B's summary reports `ntp_loss_after` 0.0032 and cosine 0.98 at 2048 with token accuracy 1.0, which I read as fitting a small set; I did not test it on new text.
- Correction to my earlier wording: I described the Clover-35B, Nemotron and Kimi links as multi-stage 8-GPU trained and said they "may behave differently" from LFM3's. The summaries show smoke-scale training for Kimi and Nemotron (tens of steps per stage); for Kimi the result above is the same failure as LFM3's, with the right input.
### Why the four models gave identical output earlier
- In the worker route and in the per-model renders the text is the same because the plan text is the same (HybridAgent-8B wrote it, independent of the chosen model) and each model's C3 re-renders its input word for word. Different file sizes in the dataset come from vocabulary size, hidden width, saved training stages and extra checkpoints, not from different behaviour on text.
### Not covered
- No Kimi latent feedback loop (engine takes ids only) and no use of `qwen_to_k31_bridge` (the Qwen HybridAgent-8B route). The 4-bit `lm_head` was not decoded, so I could not run the head as a control on these states; the engine's own next token (`17`, `The`) is the reference used instead.
- Two prompts, one run each. Raw: `%TEMP%\clover-adb\kimi_chain2.json`, `kimi_c3_readout.json`, `kimi_chain2.py`, `kimi_c3_readout.py`; k3 `/tmp/kimi_b2/chat_rows_*`.


