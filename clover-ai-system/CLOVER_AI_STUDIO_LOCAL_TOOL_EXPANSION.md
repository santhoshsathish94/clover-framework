# Clover AI Studio — Local-First Tool Expansion Assessment

**Assessment date:** 2026-11-05  
**Application inspected:** Clover AI Studio 0.4.1 on Windows 11  
**Local model/runtime:** LFM3-Omni-2B, 4-bit, llama.cpp; CPU and Intel Iris Xe Vulkan paths available  
**Scope:** what the installed Windows app actually supports, which tools are feasible on this laptop, where they fit, and how to verify them. Hosted Kimi 3.1 Pro and GPU-server work are deliberately deferred.

## 1. Executive recommendation

Clover already has the important foundation for expansion: native function schemas, a multi-round tool loop, per-tool permission checks, bounded subprocesses, cancellation, output clipping, prompt-injection warnings, and execution records. The main opportunity is therefore **not** to add a second agent framework. It is to add small, deterministic tools behind the existing contract.

The recommended first implementation is a **Reliability and Inspection Pack**:

1. `calculate` — exact arithmetic without asking the 2B model to do arithmetic internally.
2. `convert_units` — deterministic conversion across an explicit unit catalogue.
3. `search_files` — bounded file-name and literal-content search in an authorized folder.
4. `file_info` — metadata and optional streaming checksums.
5. `git_inspect` — fixed read-only `status`, `diff`, `log`, and `show` operations.

These tools need no additional AI model and little GPU work. They directly address the observed arithmetic failure and reduce the need to generate arbitrary Python or PowerShell for routine tasks.

The next pack should improve the app's existing strengths: paginated document reading, fixed-operation table analysis/charts, and FFmpeg media inspection/transformation. Screen capture, background automation, MCP, SSH, hosted accounts, and remote model execution should remain later phases because their main cost is authority and security design, not local compute.

**No application source was changed in this assessment.** The installed implementation was copied into `testing/runtime-src/` before inspection.

## 2. Evidence standard and boundaries

This assessment follows the repository's [AGENTS.md](AGENTS.md):

- **Observed:** read from the installed runtime, the current repository, health/settings evidence, or regression results.
- **Repository claim:** documented or implemented for another surface, but not established for this Windows installation.
- **Proposed:** a recommendation in this report, not an existing capability.
- **Unknown:** the authoritative Windows source location and release path are not present in the checked-out repository.

The requested direction is local-first expansion on this laptop. This report does not authorize implementation, new permissions, remote execution, credential access, or hosted Kimi integration.

## 3. What Clover AI Studio currently supports

### 3.1 Base structured tools

The installed `/api/health` inventory and [lfm3_native.py](testing/runtime-src/lfm3_native.py) lines 48–82 agree on nine base tools for Windows.

| Tool | Current behavior | Permission | Important bounds |
|---|---|---|---|
| `run_python` | Executes model-written Python; can process attached documents and create PNG charts | `code` | 60 s; 4 GB per-process memory limit on Windows; bounded output; not network-isolated |
| `web_search` | Searches the configured provider and returns up to five results | `internet` | Query limited to 400 characters; per-turn web-call cap; provider timeouts |
| `fetch_url` | Retrieves a page's main text | `internet` | Up to 5,000 characters returned by the normal tool path |
| `run_powershell` | Runs arbitrary PowerShell and installed programs | `shell` | 60 s default, 600 s maximum; 1 MB subprocess capture; process-tree stop; risky commands still ask |
| `read_file` | Reads a text file, optionally by 1-based line range | `files` | Rejects binary-looking files; vault reads blocked |
| `list_directory` | Lists one directory | `files` | First 300 entries; no recursive search |
| `write_file` | Creates or overwrites a UTF-8 text file | `edit_files` | 200,000-character limit; existing-file backup; Clover repository and vault writes blocked |
| `get_current_time` | Returns local date/time and UTC offset | None | Pure local read |
| `laptop_status` | Reports disk, battery, memory, and top memory-using apps | `shell` | Fixed status report rather than arbitrary status commands |

Tool results are clipped to 6,000 characters before being returned to the model. The UI event receives a 3,000-character preview and the turn record retains 2,000 characters. The normal loop permits up to eight rounds, suppresses repeated identical calls, and can force a final answer after the tool budget is exhausted. See [lfm3_native.py](testing/runtime-src/lfm3_native.py) lines 256–258 and 496–657, and [omni_native.py](testing/runtime-src/omni_native.py) lines 433–616.

### 3.2 Conditional tools built by the server

These do not appear in the base health list, but the installed server can register them through its `extra_tools` contract.

| Tool or family | Registration condition | Permission |
|---|---|---|
| `remember` | Memory is enabled or a workspace is active | `memory`; the observed default is `allow` |
| `vscode_open` | VS Code bridge object exists | `vscode` |
| `vscode_clone` | Same | `vscode` |
| `vscode_install` | Same | `vscode` |
| `vscode_run` | Same | `shell` |
| `vscode_github_setup` | Same | `vscode` |
| `vscode_github_sync` | Same | `vscode` |
| `generate_image` | `tools/sd.cpp/vulkan/sd-cli.exe` exists | `media` |

On this laptop, `sd-cli.exe` is absent, so local picture generation is **not currently installed**. The optional picture implementation supports 256–768 px output, requires about 3.5 GB free RAM, and has a 600 s limit. See [server.py](testing/runtime-src/server.py) lines 519–545 and 676–730, and [image_gen.py](testing/runtime-src/image_gen.py).

### 3.3 Adjacent capabilities that are not separate model tools

| Capability | What the installed app actually does |
|---|---|
| Document ingestion | Accepts up to three documents per message, 25 MB each: XLSX/XLSM/XLS, CSV/TSV, PDF, DOCX, PPTX, JSON, TXT, and MD. It saves the original and adds a bounded preview to the prompt. |
| Spreadsheet previews | Shows dimensions, columns, sample rows, and exact whole-column count/total/min/max/mean; up to six sheets and 200,000 scanned rows per table. |
| Full data analysis | Uses `run_python` on the saved document for grouped, filtered, ranked, or charted results. |
| Media understanding | LFM3-Omni accepts image, audio, and video input; FFmpeg supports attachment conversion and video decoding. |
| Speech | Produces local speech through a separate CPU llama.cpp instance when chat uses the GPU. |
| Voice recording | The user explicitly starts a browser microphone recording; this is not a model-controlled microphone tool. |
| Workspace context | Scans a bounded workspace index and supplies relevant context; this is separate from recursive content search. |
| VS Code workspace tools | The VS Code path has core file/search/edit/diagnostic tools and deferred `search_tools`; these are not equivalent native Studio tools. |
| Execution history | Tool offers, calls, permission decisions, results, and errors are written to local execution logs. |

Document behavior is implemented in [documents.py](testing/runtime-src/documents.py), media ingestion in [server.py](testing/runtime-src/server.py) lines 466–516, and execution records in [clover_repo.py](testing/runtime-src/clover_repo.py) lines 215–246.

### 3.4 What is not established as current Windows support

- **MCP:** repository architecture documents mention MCP, but no MCP client or server integration was found in the 67 copied installed Python modules or in tracked implementation code. Treat MCP as a proposed extension, not a current feature.
- **Native recursive file/content search:** only the VS Code bridge exposes such search. The base Studio tools do not.
- **Deterministic calculator or unit converter:** absent. The current choices are direct model arithmetic or generated Python.
- **Dedicated read-only Git tools:** Git is available through PowerShell or VS Code, not through a fixed read-only schema.
- **User-callable FFmpeg tools:** FFmpeg, FFprobe, and FFplay are installed, but only internal attachment conversion/decoding paths currently use them.
- **Screenshot/screen capture, OCR, scanned-PDF rendering, local scheduled tasks, and model-controlled camera/microphone:** not found in the installed Studio tool registry.
- **Windows SSH tool:** the mobile repository contains an SSH reference implementation, but the installed Windows runtime has neither its host manager nor an `ssh` permission category.

## 4. Runtime and safety constraints that new tools must preserve

### 4.1 Routing is part of the tool

The default **When I ask** mode does not offer every schema on every turn. [semantics.py](testing/runtime-src/semantics.py) maps message intent to capabilities, and [lfm3_native.py](testing/runtime-src/lfm3_native.py) maps capabilities to tool names. A new schema without corresponding routing rules will often never reach the model.

This gating is intentional: source comments record that always exposing tools degraded 6 of 65 ladder tasks and worsened arithmetic. Expansion should therefore add a small number of clear capabilities, not a large undifferentiated tool catalogue.

### 4.2 Existing permission and execution model

- Internet, code, shell, file read, file edit, VS Code, and media default to `ask`; memory defaults to `allow`.
- Permission checks occur at execution, after routing and model planning. A planner cannot legitimately bypass a denial.
- Risky PowerShell operations require confirmation even when shell is generally allowed.
- PowerShell and image generation support cancellation and process-tree termination.
- Python uses a Windows Job Object for memory and process-tree control.
- Tool-originated text is passed through prompt-injection flagging.
- `write_file` makes a backup and protects Clover's own repository and vault.

These are strong reusable mechanisms. However, existing general file tools can operate outside the active workspace when the user names another path, and Python is not network-isolated. New fixed tools should be **more narrowly scoped**, not inherit broad authority merely because a general tool already has it. See [permissions.py](testing/runtime-src/permissions.py), [local_terminal.py](testing/runtime-src/local_terminal.py), [windows_python_tool.py](testing/runtime-src/windows_python_tool.py), and [paths.py](testing/runtime-src/paths.py).

### 4.3 Laptop fit

Observed hardware is an i5-1335U, 15.7 GB RAM, and Intel Iris Xe graphics with 8,033 MiB reported by Vulkan. The local chat model is a 2B 4-bit model with a 32,768-token context.

| Addition type | Laptop fit |
|---|---|
| Exact arithmetic, unit conversion, file metadata, search, read-only Git | Good; CPU/I/O work, no extra model |
| Document pagination and fixed table operations | Good with row, sheet, page, memory, and output limits |
| Charts | Good through existing matplotlib/PNG artifact path |
| FFmpeg inspection, frame extraction, trim, and transcode | Good for bounded user-selected files; CPU-intensive jobs need timeouts and cancellation |
| Scanned-PDF page rendering | Feasible; may need one pinned rendering dependency |
| OCR | Feasible as an optional CPU component, but accuracy and language coverage need separate tests |
| Screenshot | Light compute; needs an explicit native/browser capture bridge and fresh consent |
| MCP | Compute is manageable; dynamic trust, secrets, process lifecycle, and prompt cost are the hard parts |
| Optional Stable Diffusion pictures | Possible only after installing the absent component; memory pressure makes it unsuitable as a dependency of the first pack |

## 5. Prioritized local-first additions

### 5.1 Phase 1 — Reliability and Inspection Pack

| Proposed tool | Value | Cost | Permission | Implementation notes |
|---|---|---:|---|---|
| `calculate` | High | Low | None | Safe arithmetic parser/evaluator; never Python `eval`; return exact numeric text |
| `convert_units` | Medium | Low | None | Explicit unit catalogue and aliases; decimal arithmetic; reject incompatible dimensions |
| `search_files` | High | Low–medium | `files` | File-name glob and literal text modes; authorized root required; bounded depth/files/results |
| `file_info` | Medium | Low | `files` | Size, timestamps, type/extension, and optional streamed SHA-256 |
| `git_inspect` | High for repository work | Low–medium | `files` | Fixed operation enum; active/authorized repository only; no arbitrary Git flags or mutation |

#### `calculate`

**Suggested schema:** `expression: string`.

Support parentheses, `+`, `-`, `*`, `/`, `//`, `%`, exponentiation, and a small documented set of constants/functions if needed. Use a safe AST evaluator or equivalent decimal parser. Cap expression length, literal digits, exponent magnitude, and operation count so inputs such as enormous nested powers fail quickly.

Route explicit arithmetic expressions and requests such as “calculate”, “sum”, “total”, and “what is 137 × 29 + 46?” to this tool. A more reliable alternative for a fully parseable arithmetic-only message is to evaluate it directly in the semantic layer and bypass generation entirely.

**Regression acceptance:** the existing failing case returns exactly `4019`, without requiring the user to select Try again.

#### `convert_units`

**Suggested schema:** `value: number`, `from_unit: string`, `to_unit: string`.

Start with length, mass, time, temperature, data size, and common volume units. Temperature needs affine conversion rather than a multiplier. Return both the normalized result and the exact conversion used when helpful. Unknown or incompatible units are errors, not model guesses.

#### `search_files`

**Suggested schema:** `root: string`, `query: string`, `mode: "name" | "content"`, `glob: string`, `max_results: integer`.

Resolve the root against the active workspace or an explicitly named user folder. Enforce containment after resolution; do not follow directory symlinks/reparse points by default. Skip the vault, Clover records, `.git`, and binary/oversized files. Prefer literal content search first; unrestricted model-generated regular expressions add avoidable complexity.

Return relative paths, line numbers, short snippets, counts, and a `truncated` flag. Do not return whole files.

#### `file_info`

**Suggested schema:** `path: string`, `sha256: boolean`.

Use streamed hashing rather than loading the whole file. Put a size/time bound on hashing and make the permission detail say that the file will be read. Return metadata as compact structured text; do not infer file safety from a hash.

#### `git_inspect`

**Suggested schema:** `operation: "status" | "diff" | "log" | "show"`, `path: string`, `target: string`, `limit: integer`.

Map each operation to a fixed argument list. No arbitrary revisions-as-options, configuration overrides, hooks, pager, editor, credential prompt, push, commit, checkout, restore, clean, or reset. Run with the existing quiet Git environment and output bounds.

This is intentionally a read-only complement to `run_powershell`, not a general Git agent. Mutating Git operations require a separate later direction and risk design.

### 5.2 Phase 2 — Documents, Data, and Media Pack

| Proposed tool | Value | Cost | Permission | Integration |
|---|---|---:|---|---|
| `read_document` | High | Medium | `files` for arbitrary paths | Reuse `documents.py`; page/slide/sheet/range pagination beyond the 6,000-character preview |
| `analyze_table` | High | Medium | `files`; `edit_files` only for export | Fixed filter/group/aggregate/sort operations using installed pandas/openpyxl, not generated code |
| `chart_table` | Medium–high | Medium | `files` + artifact-write/media consent | Fixed chart types and columns; save through the existing media/out PNG path |
| `media_info` | Medium | Low | `files` | Fixed FFprobe JSON query with stream/format allowlisting and output cap |
| `extract_frame` | Medium–high | Low | `files` + artifact-write/media consent | FFmpeg at a specified timestamp; dimension and file-size limits |
| `trim_media` | Medium | Medium | `files` + `edit_files` | Fixed start/duration/end schema; no overwrite; bounded output folder |
| `convert_media` | Medium | Medium | `files` + `edit_files` | Explicit audio/video/image preset allowlist; no arbitrary FFmpeg arguments |
| `render_pdf_page` | High for scanned material | Medium | `files` + artifact-write/media consent | Render selected pages to bounded PNGs for Omni vision; report when OCR/text is unavailable |

The data tools should return machine-checkable values and a `truncated`/rows-read indicator. They should not claim exactness when a row cap was reached. Artifact tools should use fresh non-colliding output names and show the resulting file in chat.

All FFmpeg work should use `local_terminal.run`, not the current attachment helper's bare `subprocess.run`, so user cancellation, process-tree termination, and consistent output limits apply. FFprobe and FFmpeg are already installed locally; no new media model is required.

### 5.3 Phase 3 — Native capture and local automation

These are feasible but should follow the first two packs.

- **Screenshot:** add a new `screen` permission, default `ask`; ideally require fresh consent for every capture. Capture only the user-selected screen/window/region, save into media, and make the image available to the current Omni turn. Never start background or periodic capture from a general `allow` setting.
- **Scanned-document OCR:** optional component with pinned engine/models, page caps, language reporting, and an explicit distinction between OCR text and visual interpretation.
- **Local reminders/tasks:** requires persistent storage, an app-running policy, cancellation/deletion controls, and a separate `automation` permission. Do not create Windows Task Scheduler entries silently. A task must remain disabled until the user explicitly enables it.
- **User-initiated camera capture:** possible through a native/browser consent flow, but not a model-invoked always-on camera tool. Existing microphone recording should remain user-initiated.

### 5.4 Phase 4 — Extensibility and remote access

**Local MCP client** is the appropriate long-term way to avoid hard-coding every integration, but it should come after a threat model and source-target clarification.

Minimum design requirements:

- stdio/local servers first; explicit executable and argument allowlist;
- no automatic installation, startup, or server discovery execution;
- namespaced tool names and collision checks;
- deferred catalog search rather than injecting every server schema into every prompt;
- per-server and per-tool permission decisions, with server-supplied annotations treated as untrusted hints;
- scoped paths, network egress disclosure, secret redaction, timeouts, cancellation, and output caps;
- independent tests for malicious tool descriptions/results and server process termination.

**SSH** can draw on the mobile repository's host-management ideas, but that code is not current Windows support. It needs a distinct permission, configured-host allowlist, host-key verification, non-interactive execution, risky-command checks, and explicit human direction. It belongs with the later GPU-server discussion, not the first local pack.

**Generic HTTP requests, browser automation with account sessions, email/calendar/messaging writes, package installation, process control, and credentialed GitHub mutations** should not be added as general-purpose local tools in this pass.

## 6. Exact integration points in the installed architecture

The following are implementation targets in the copied installed source, not instructions to patch only the installed copy.

1. **Tool pack module:** put deterministic implementations in a focused module such as `local_tool_packs.py`; keep parsing/execution separate from model code.
2. **Registration:** build entries with `schema`, `permission`, `detail(args)`, and `run(args)` and append them to `extra` in [server.py](testing/runtime-src/server.py) `lfm3_turn`, lines 1040–1054. This reuses the existing dynamic-tool seam.
3. **Schemas:** use [lfm3_native.py](testing/runtime-src/lfm3_native.py) `_fn`, lines 43–45. Base tools may instead be added to `STUDIO_TOOLS`, but the server extra-tool path is cleaner for an optional pack.
4. **Dispatch and permission:** keep [lfm3_native.py](testing/runtime-src/lfm3_native.py) `execute_tool`, lines 496–546, as the single dispatch path. Do not call pack runners directly from routing.
5. **Capability routing:** extend `RUNTIME.tools` and the relevant `ASKS`/`DECLINES` rules in [semantics.py](testing/runtime-src/semantics.py). Add only intent patterns with positive and negative tests.
6. **Path enforcement:** add reusable authorized-root resolution beside [paths.py](testing/runtime-src/paths.py), including post-resolution containment and symlink/reparse-point behavior.
7. **Subprocesses:** reuse [local_terminal.py](testing/runtime-src/local_terminal.py) for FFmpeg and read-only Git; use argument arrays and fixed flags.
8. **Documents/artifacts:** reuse [documents.py](testing/runtime-src/documents.py) and the existing `media/out` artifact flow in [lfm3_native.py](testing/runtime-src/lfm3_native.py) lines 548–576.
9. **UI:** add labels/icons to `TOOL_UI` in [web/app.html](testing/runtime-src/web/app.html) lines 807–814. Add new permission categories to [permissions.py](testing/runtime-src/permissions.py) only when an existing category would misrepresent the action.
10. **Audit:** preserve the existing `tool_call`, `tool_result`, and permission events emitted through the server and recorded by [clover_repo.py](testing/runtime-src/clover_repo.py).
11. **Health/status:** expose base, conditional, installed, and unavailable tools distinctly. The current engine description lists only `STUDIO_TOOLS`, so it is not a complete conditional-tool inventory.

### Source/deployment blocker

The checked-out repository does not contain the installed Windows `server.py`, `lfm3_native.py`, or `omni_native.py`. No tracked history for the two native modules was found in the locally available branches. The mobile [local_terminal.py](clover-ai-system/mobile/local_terminal.py) says it was ported from `laptop/clover/local_terminal.py`, but that laptop source directory is absent.

Therefore `testing/runtime-src/` is **analysis evidence, not an authoritative upstream edit target**. A production implementation should wait until the Windows source and installer build path are identified. Patching only `C:\Users\santh\CloverAI` would create an untracked local fork and would not produce a reproducible release.

## 7. Verification matrix

Existing validation controls must remain unchanged. New tests should be added beside the authoritative Windows source once located.

| Layer | Required checks |
|---|---|
| Calculator | Exact arithmetic including `137 × 29 + 46 = 4019`; precedence; decimals; division by zero; malformed input; huge exponent rejection; proof that names/imports/calls cannot execute |
| Units | Precision, aliases, temperature offsets, incompatible units, unknown units, extreme values |
| File search/info | Name/content hits, no matches, Unicode, binary skip, file/result/time caps, vault/records exclusion, root escape, symlink/reparse escape, streamed hash against a known digest |
| Git inspection | Fixed status/diff/log/show outputs; invalid repository; argument injection rejected; before/after refs, index, and working tree unchanged; no pager/editor/network |
| Documents/data | Page/sheet/range selection; row-cap disclosure; known grouped totals; empty/missing columns; chart artifact opens; current attachment remains isolated from earlier attachments |
| Media | FFprobe JSON bounds; frame at known timestamp; trim duration; preset conversion; no overwrite; malformed media; cancellation and timeout stop the whole process tree |
| Permissions | Every category defaults as designed; Ask performs no work before a decision; Deny and non-interactive modes fail closed; Allow once does not persist; new screen capture cannot silently become background capture |
| Routing | Positive phrases offer the intended tool; explanations and creative requests do not; “don't use/search/run” remains declined; unknown tools and malformed arguments never execute |
| Tool loop | No duplicate-call storms; result clipping; final answer from tool evidence; cancellation during model generation and during a tool |
| Regression | Repeat the existing arithmetic and cross-attachment cases; rerun broad CPU and Vulkan chat/attachment/settings checks; compare execution logs with actual effects |
| Resource behavior | Measure elapsed time, peak memory, output size, and leftover processes on this laptop; do not infer performance from tool completion alone |

### Phase 1 acceptance criteria

- All deterministic results match independent expected values.
- No new tool performs network access.
- File/Git tools cannot escape their authorized root or mutate the repository.
- Every permission denial results in zero tool side effects.
- New tools appear honestly in health/UI only when available.
- Existing repository validators and Windows regression checks pass without weakening acceptance criteria.
- Unfavorable results are recorded with the exact input and environment, not retried blindly until a favorable answer appears.

## 8. Deferred hosted Kimi / GPU-server boundary

Hosted Kimi 3.1 Pro is intentionally outside this assessment's implementation scope. When that work starts, the local tool contract should remain reusable: a hosted model may propose a call, but local dispatch must still enforce routing, user permission, path/network boundaries, timeouts, and audit. Model choice must never silently expand tool authority or send local content off-device.

The immediate next cycle does not need the GPU server. It needs the authoritative Windows source target and a small deterministic pack validated on this laptop.

## 9. Recommended next step

Locate or restore the authoritative Windows Studio source and its installer build path, then implement only **Phase 1: `calculate`, `convert_units`, `search_files`, `file_info`, and `git_inspect`**. Validate the pack on CPU and Vulkan, preserve the execution evidence, and use those outcomes to decide whether Phase 2 should begin with documents/data or media.
