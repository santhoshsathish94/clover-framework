# Clover Studio Quality Engineering

## Why this exists

Clover has useful component and regression suites, but a passing offline suite is not proof that a person can complete a task in a running Studio. The quality gate keeps those evidence classes separate and makes failures reproducible.

## First milestone: repeatable baseline

From `clover-ai-system/studio`:

```powershell
# Existing offline regression suites (no live model required)
python tests/quality_gate.py --offline --json logs/quality/offline.json

# Read-only checks against a running local Studio
python tests/quality_gate.py --live --base-url http://127.0.0.1:8765 --json logs/quality/live.json

# Optional browser-shell smoke check
python -m pip install playwright
python -m playwright install chromium
python tests/quality_gate.py --live --playwright --base-url http://127.0.0.1:8765
```

The runner never sends chat messages, grants permissions, changes settings, or invokes tools. Live endpoints are read-only. Reports deliberately omit tokens, chat content, user files, and credentials.

## Evidence classes

- **offline_or_simulated**: unit/regression suites, fake engines, stand-in model output, stub image files.
- **live_http_read_only**: actual HTTP responses from the running Studio process.
- **live_browser_shell**: a real browser loaded the Studio UI shell and observed JavaScript page errors.
- **live_model_e2e** (future): actual prompt sent to the selected model and its complete streamed response captured.
- **live_tool_e2e** (future): a real tool executed under explicit test permissions and its external effect independently verified.
- **live_media_e2e** (future): a real generated media artifact was decoded and evaluated; a successful tool-call string is not enough.

A test may only claim the evidence class it actually exercised. Missing laptop, phone, K3, or GPU access is reported as NOT RUN, never as PASS.

## Scenario coverage to add next

The scenario catalogue is `scenarios.json`. Each scenario describes the user-visible task, required outcome, evidence needed, and risk class. The next increments should add:

1. Browser-driven chat scenarios against a disposable test profile.
2. Model matrix execution: local LFM3, Clover-35B, Nemotron, Kimi, and Auto routing where each endpoint is available.
3. Tool scenarios with explicit one-time permissions and reversible fixtures.
4. Image generation checks that require a real decodable image and score prompt adherence, composition, artifacts, resolution, and edit consistency.
5. Audio/video attachment and response checks.
6. Device runners for Windows laptop and Android phone, plus remote K3/GPU runners.

## Image generation: honest capability gate

A PNG existing on disk proves only that a PNG was written. The existing offline image test intentionally substitutes a fake `sd-cli`; the UI harness also uses a stand-in image. These tests are valuable for routing, file handling, error handling, and Stop behavior, but they do not measure image quality.

For every real image run, retain (with secrets stripped): requested prompt, selected backend/model, model availability, dimensions, generation time, output path/hash, decode result, and evaluator scores. Keep procedural illustrations labelled as such. Do not silently present a canvas fallback as general-purpose photorealistic generation.

## Failure record format

Each failed scenario should produce a JSON record with:
- scenario ID and version;
- device/OS, Studio build, model/backend and runtime configuration;
- expected outcome and observed outcome;
- evidence class;
- failure layer: UI, transport, orchestration, model, tool, permission, media pipeline, persistence, or evaluator;
- timestamp, duration, trace/screenshot/artifact references;
- minimal reproduction steps and whether the run is safe to retry.

Do not upload private prompts, personal files, access tokens, or generated personal media to shared CI artifacts by default.
