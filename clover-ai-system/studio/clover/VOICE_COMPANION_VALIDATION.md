# Clover AI Studio voice companion — validation and handoff

## Intended outcome

Keep the supplied robot and clockwise circle, remove every visible companion background, and make continuous Voice feel like a spoken conversation rather than a read-aloud chat transcript. Allow natural pauses before sending, while leaving ordinary text chat and hold-to-speak unchanged.

The repository was fast-forwarded from Clover main through commit `213937de` (12 incoming commits). The tested Voice bridge, controller, assets, and regression suite are integrated into the canonical Studio source. The installed application under `C:\Users\santh\CloverAI` was not modified.

## Implemented

- The supplied robot pixels are preserved in `robo-clover-transparent.webp`; the warm backdrop, halo, and floor shadow are removed. The static artwork remains at `opacity: 0.58`.
- Both the companion and its compact status have transparent backgrounds, with no card, blur, border, shadow, canvas, WebGL mesh, body deformation, mouth, or eye-glow layer.
- One SVG ring with a white dashed segment rotates clockwise through 360 degrees every 2.4 seconds during listening and playback. It stays still during speech preparation, waiting, errors, and reduced-motion use.
- The icon-only microphone control remains inside `#composer`, beside Send, with accessible names and state descriptions.
- Continuous Voice sends audio with a conversation marker. Its user and assistant turns remain in session/model history but are filtered from the visible chat; the welcome view stays visible and the session is titled “Voice conversation.” Spoken replies still use Clover's actual response.
- Continuous Voice waits for **3,000 ms of silence** after speech. The maximum utterance is 45 seconds and the no-speech timeout is 20 seconds.
- Ordinary text turns and hold-to-speak voice messages retain their existing visible transcript behavior.
- Existing 16 kHz WAV conversion, audio-chat/TTS APIs, and microphone cleanup are retained. The static-file allowlist includes the transparent artwork, voice CSS, and controller.

## Verification evidence

### Controlled browser/integration regression

Command: `npm run test:voice`

The final run passed all **16 checks** using Edge/Playwright, a synthetic microphone, generated PCM audio, and mocked chat/TTS responses. It verifies:

- updated Studio startup on isolated test port 7117;
- icon-only microphone control in the composer and default-off capture;
- transparent corner pixels in the loaded robot, transparent companion/status styling, opacity `0.58`, and no robot canvas, mesh, body, or mouth animation;
- continued capture after 3.2 seconds with no premature chat request, followed by a real WAV audio request after the completed pause;
- no visible voice-message bubble, assistant transcript, or “reading your message” turn; the welcome view remains and the title becomes “Voice conversation”;
- no ring animation during the deliberate 650 ms speech-preparation delay, then clockwise `rotate(360deg)` animation during actual playback;
- listening and ring resumption after playback, and complete microphone shutdown when Voice is disabled;
- unchanged visible ordinary text chat and hold-to-speak turns;
- a 375 px layout without horizontal overflow;
- permission-denial recovery and no browser exceptions, console errors, failed requests, or HTTP errors.

Artifacts: `test-artifacts/results.json`, `voice-desktop.png`, `voice-speaking.png`, `voice-mobile.png`, and `robot-panel.png`.

Visual inspection covered the full desktop/mobile views, companion panel, thumbnail, enlarged robot/ring crop, and the isolated artwork against a dark contrasting background. The first color-distance mask left a faint halo and floor shadow, so it was replaced with foreground segmentation and inspected again. The final captures show only the robot, intended circle, and transparent status over the page.

JavaScript syntax checks and Python compilation passed. An initial regression assertion raced the asynchronous WAV conversion; the test now waits for the actual chat response. Production behavior was not weakened to address that test race. The isolated fixture intentionally has no `llama-server.exe`; its expected startup exception is unrelated to the mocked browser checks.

### Earlier real local LFM3-Omni regression

An earlier run of `node test-real-backend.js`, before this main-branch merge and voice simplification, passed all five checks against the installed model and tools. It produced a nonempty 238,124-byte WAV and the model replied: “Hello! How can I help you today?” This is historical evidence only; the real backend was not rerun for this presentation and timing cycle.

## Limits and next cycle

- Browser interaction tests use a fake microphone, not this laptop's physical microphone or speakers. Real-room pause detection, speaker delay, and conversational quality still need a human listening check.
- The ring reflects voice state; it is not an audio-amplitude or phoneme visualization.
- This is not a full regression of Clover's other models or non-streamed Clover-layers mode.
- Temporary test servers were stopped. The port 7100 preview configuration was not changed; the client manages its server lifecycle.

The next useful check is a short physical microphone/speaker conversation in the preview, including a mid-sentence pause, to judge the three-second timing and spoken response in the actual room.
