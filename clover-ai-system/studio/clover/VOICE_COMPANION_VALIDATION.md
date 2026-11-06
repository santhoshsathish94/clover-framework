# Clover AI Studio conversation mode — validation and handoff

## Intended outcome

Keep ordinary chat as the default. Use the Clover flower beside Settings to enter a full-screen robot conversation with the supplied interactive 3D character, natural voice pauses, spoken responses, transparent status/results over the robot, and the same Clover flower available as a toggle back to chat.

The work is in the canonical Studio source on `feat/clover-natural-voice-companion`, merged with `origin/main` through `cdffec92`. The installed application under `C:\Users\santh\CloverAI` was not modified.

## Implemented

- The Clover flower is an accessible button immediately beside Settings. The separate composer Voice button was removed; upstream hold-to-speak remains available in ordinary chat.
- Conversation mode fills the viewport and hides/inerts the chat application. The same Clover button moves into the conversation dialog and remains visible and operable; leaving moves that same node back beside Settings.
- Voice capture starts automatically after the user enters and grants microphone access. Listening, thinking, speaking, and errors are shown as lightweight text without a status card.
- Voice turns and streaming assistant replies appear in a transparent, non-blocking overlay above the lower part of the robot. The overlay retains up to eight entries and uses text nodes rather than HTML injection.
- The supplied robot retains pointer tracking, idle animation, waving, blinking, thinking, and answering reactions. Clicking/tapping it produces a wave; moving the pointer changes its gaze. Results and status use `pointer-events: none`, so they do not block robot interaction.
- The Clover toggle and Escape return to chat. Double-clicking/double-tapping the robot also returns to chat on devices without a keyboard. These exit paths are handled inside the iframe as well as the parent page.
- Voice startup selects `wave`, waiting selects `sleepy`, thinking selects `thinking`, and playback selects `answering`. Reactions are triggered on actual state transitions and scene readiness, not every monitor tick.
- Continuous Voice sends audio with a conversation marker. Turns remain in session/model history and are visible in the chat transcript after returning; the session is titled “Voice conversation.” Spoken replies use Clover's actual response.
- Conversation turns carry an explicit `conversation` flag and a server-enforced 256-token ceiling. The Omni system instruction asks for one or two natural, short sentences unless the user requests detail. Ordinary chat retains its configured budget.
- Conversation turns do not warm the separate CPU speech engine while C3 is running. Their speech requests reuse the resident GPU model after the answer is ready, avoiding a second resident 2+ GB model on this laptop. Ordinary voice messages retain their existing warm-up and CPU speech path.
- Entering conversation is explicit consent to spoken replies, independent of the separate “Speak replies to voice messages” setting. Speech playback failures are added to the visible results and listening resumes instead of leaving a silent, apparently stuck robot.
- Voice waits for **3,000 ms of silence** after speech. The maximum utterance is 45 seconds and the no-speech timeout is 20 seconds.
- Ordinary text turns and hold-to-speak voice messages retain their existing visible transcript behavior. Existing 16 kHz WAV conversion, audio-chat/TTS APIs, and microphone cleanup are retained.
- Leaving conversation stops microphone tracks and speech, restores chat, and returns the iframe to `about:blank` to release WebGL. The main page CSP permits only same-origin frames.
- The refined source from `Clover_Robo_Refined_Source.zip` is now maintained in `robot/scene.js`, building on the earlier user-supplied robot. It adds the curved five-leaf antenna, refined ivory/green materials, woven cape and scarf, fuller activity props, articulated arm posing, softer idle/action blending, and irregular blinks. `npm run build:robot` produces the local `web/clover-robo.js` bundle from Three.js 0.180.0; no hosted robot runtime is loaded.

## Verification evidence

### Controlled browser/integration regression

Command: `npm run test:voice`

The final run passed all **23 checks** using Edge/Playwright, a synthetic microphone, generated PCM audio, and mocked chat/TTS responses. It verifies:

- Studio startup on isolated test port 7117;
- the Clover flower immediately beside Settings and no composer Voice button;
- visible chat and default-off capture before entering conversation;
- a full-viewport robot scene with hidden/inert chat, no ring, a visible Clover toggle, and transparent/non-blocking status and results;
- a ready local 3D scene with viewport-sized canvas, nonzero draw calls/triangles, WebGL alpha, and hidden ground/distant scenery/contact shadow;
- only the local `clover-robo.js` robot resource, with no remote robot runtime requests;
- continued capture after 3.2 seconds with no premature chat request, followed by a real WAV audio request after the completed pause;
- the conversation request flag, 256-token budget, absent speech warm-up, and resident-model speech request;
- the user voice-turn marker and streaming reply over the robot, with both retained in the session transcript;
- `thinking` during processing and speech preparation, `answering` during playback, and listening resumption afterward;
- spoken replies in conversation mode even when ordinary voice-message autoplay is disabled;
- Clover-toggle, Escape, and robot double-click exit paths, chat/transcript restoration, complete microphone shutdown, and WebGL release;
- unchanged ordinary text chat and hold-to-speak turns, including their full response budget and existing speech warm-up;
- a 375 × 812 robot/toggle/status layout without horizontal overflow;
- visible speech-playback failure feedback followed by automatic listening recovery;
- permission-denial accessibility and recovery, with no browser exceptions, console errors, failed requests, or HTTP errors.

Artifacts: `test-artifacts/results.json`, `voice-desktop.png`, `voice-speaking.png`, `voice-mobile.png`, and `robot-panel.png`.

Visual inspection covered the refined robot's desktop listening/speaking and mobile states. The first desktop capture clipped the antenna's Clover at the top; the desktop camera target was corrected from 1.85 to 2.3 and the complete robot was then verified uncropped. Mobile framing remained unchanged. Status sits at the upper left, the Clover toggle at the upper right, and the voice-turn/reply text is transparent and readable along the bottom without a card or background.

### Live-preview observation and outcome-driven corrections

- The live preview reported `ready: true`, `omni: true`, `audio: true`, and active microphone capture in `listening` phase. This rules out a not-ready model or an immediate permission/startup failure; it does not establish physical speech recognition or speaker output.
- The previous interface hid its own status and Clover toggle, so the user had no visible feedback or direct toggle back. The toggle now remains in the dialog, and status/results are visible over the robot.
- The previous speech pipeline discarded playback errors. Errors now remain visible in the conversation results and listening resumes. A full invalid-audio regression passed without suppressing the failure.
- Conversation previously depended on the ordinary voice-message autoplay setting. Entering conversation now explicitly enables spoken replies for that mode.
- Earlier full-screen cycles also fixed iframe-local Escape handling and a denial-test scene-loading race; those exit and denial checks remain green.
- The refined standalone scene was adapted without weakening those controls: transparency and hidden scenery were retained, iframe double-click/Escape exits were restored, and the scene diagnostics still expose the transparency and viewport assertions used by the regression suite.

### Additional checks

- `npm run build:robot` bundled the refined source successfully, the final `npm run test:voice` passed all 23 checks, and `git diff --check` passed. Earlier JavaScript, inline Studio JavaScript, and Python compilation checks also passed; this asset-only cycle did not modify the Python backend.
- `python scripts/check-links.py` passed with 1,327 relative links, zero broken links, and zero dead anchors after removing the generated isolated runtime directories.
- `python ../tests/test_strict_c1_c2_c3.py` passed all strict stage-order and S1/S2/S3 checks after the performance changes. C1 still emits no answer tokens; C2 and C3 retain their required inputs and ordering.
- The live port 7100 preview was previously reloaded and inspected: model/audio readiness, robot readiness, microphone capture, visible in-dialog Clover toggle, non-blocking overlays, and toggle-back to chat all matched the implementation.
- The isolated fixture intentionally has no `llama-server.exe`; its expected startup exception is unrelated to the mocked browser checks.

### Current real local LFM3-Omni regression

The final `node test-real-backend.js` run passed all **six checks** against the installed model and tools. It synthesized a nonempty 253,484-byte question WAV, accepted it as real audio input, returned “Hello! I'm Clover, your assistant running on your Windows laptop.”, and synthesized a nonempty 46,124-byte conversation reply through the resident model. The conversation speech request took **9.569 seconds** in that contended two-server test environment.

Latest main adds SSE event IDs. The first real-backend run returned a valid reply but the test parser looked only at blocks beginning with `data:` and missed frames beginning with `id:`. The parser now reads `data:` lines independently; its requirement for a nonempty real reply was not weakened.

### Latency evidence and remaining diagnosis

The live measurement script sends isolated requests, records stage timings, and deletes its sessions. It is configurable with `CLOVER_TEST_ORIGIN` and `CLOVER_LATENCY_OUT`.

- Initial pre-fix run: text **2.720 s**, voice **44.824 s**; voice C3 consumed about **40.5 s**.
- Controlled pre-fix voice turn requesting exactly “Hello”: **23.169 s**. A longer text reply took **56.500 s**, proving that response length contributes.
- Repeated pre-fix short turns with identical 13-code/3-level/1-window C3 work: text **25.892 s**, voice **31.180 s**, following text **36.622 s**. Their C3 times were **18.22 s**, **26.80 s**, and **28.78 s**, so neither audio recognition nor response length alone explains the delay.
- A later run against the existing 7100 process, with only the reloaded frontend contract active, measured text **41.533 s**, voice **43.924 s**, and following text **24.100 s**. This is not a measurement of the new Python backend.
- The isolated real C3 runtime decoded the same 15-token/47-code/5-level reply in **0.890 s**. In the full app, the CPU speech server held about **2.46 GB**, free memory fell to about **1.83 GB**, and equal C3 workloads varied widely. This supports resource contention as the remaining cause; it does not prove that every slow turn is caused by speech alone.

Artifacts: `test-artifacts/live-voice-latency-before.json`, `live-voice-latency-after-frontend.json`, `c3-profile.json`, `c3-profile.txt`, and `real-backend-results.json`.

## Recommended talking-visual improvements

In priority order:

1. **Playback-amplitude mouth motion:** connect the actual TTS audio analyser to mouth openness, with smoothing and a noise floor. Keep it active for the entire audio stream instead of relying only on the fixed 3.8-second answering reaction. This is the most useful near-term improvement and does not require phoneme data.
2. **Viseme synchronization:** for higher fidelity, obtain phoneme/viseme timestamps from a compatible TTS engine or a local speech aligner, then blend a small set of mouth shapes. The current LFM3-Omni timing supply has not been established, so this needs an API/engine investigation first.
3. **Prosody-aware expression:** use pitch, energy, sentence boundaries, and pauses to drive subtle head nods, eye shape, cheek glow, and hand emphasis. Avoid continuous random movement while the robot is speaking.
4. **Turn-taking face cues:** use restrained face-only cues for hearing, end-of-sentence detection, thinking, and speaking. This preserves the robot-only interface while making the conversation state understandable.
5. **Interruption control:** add a clear tap-to-interrupt action or carefully tested barge-in using echo cancellation and VAD. The current implementation intentionally pauses capture during playback to avoid transcribing Clover's own voice.
6. **Performance and accessibility:** measure frame time on the laptop and phone, adapt pixel ratio/shadow quality, suspend rendering while hidden, and provide reduced-motion behavior without disabling the actual voice controls.

## Limits and next cycle

- Browser tests use a fake microphone, not this laptop's physical microphone or speakers. Real-room pause detection, speaker delay, mouth/audio alignment, and conversational quality still need a human listening check.
- The answering animation is currently state-driven and time-limited; it is not amplitude- or phoneme-accurate lip synchronization.
- The 7100 Python process predates the merged backend and these performance changes. Its static frontend can reload from disk, but the new conversation instruction, server ceiling, and resident-model speech path require a client-managed preview restart. No final post-restart end-to-end timing is claimed.
- This is not a full regression of Clover's other models or non-streamed Clover-layers mode.
- Temporary test servers and generated runtime directories were stopped/removed. The client manages the port 7100 preview lifecycle; it was not killed.

The next useful cycle is reopening/restarting the managed preview, repeating the same latency measurement, and validating physical microphone/speaker conversation. Do not repeat the old warm-up-plus-second-model path as a performance comparison.
