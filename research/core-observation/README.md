# How the Clover layers work: observed on k3, 2026-11-01

Observation only, as directed: what each layer received, what it handed on, and how the layers interacted.
Nothing here is a fix, a ranking or a recommendation. Each statement points to the evidence in `traces/`.
Where something was not established, it says so.

## What ran

| | Framework runtime | Laptop stack |
|---|---|---|
| Code | clover-framework `75f8c5d`, `clover-ai-system/mobile/s23_on_device_clover.py` (the phone runtime) | scratch-agents `lfm3-hybridagent/laptop/clover` (the laptop Studio), unchanged |
| Model | our trained LFM3-Omni-2B S3: 24 pruned trunks with 4-bit FFN, from k3's `s23-omni2b` export | LiquidAI LFM3-Omni-2B Q4_0 GGUF (SHA-256 matches LiquidAI's) + mmproj Q8_0 |
| C1 engine | Clover C engine `liblfm3_cpu_engine.so`, 8 threads | llama.cpp b12012 `llama-server`, CPU, 8 threads |
| Settings | the runtime's defaults (tools "ask", every layer on); engine cpu | the Studio's defaults (tools "ask"; memory, ontology, compiler on; recall, ncm, kv off); engine cpu |
| Permissions | internet, code, files, memory allowed; shell, ssh, file changes denied | same |
| Device part replaced | none | the Python runner: the POSIX `PythonTool` its Windows runner was ported from (`harness/linux_device`) |

Both ran as unprivileged sandboxed services on k3 (`harness/k3.sh`), with their own CPUs and loopback ports. The
live K3 server and other sessions' folders were not touched. Both received the same 15 messages in 10
conversations (`harness/drive.py`), as their web pages send them. Every layer boundary was wrapped from outside
the code (`harness/tracer.py`): results are passed through unchanged.

The two stacks run different weights, so their answers are not a comparison of models. What they show is how
each stack moves a message through its layers.

## The path of one message

### Framework runtime
1. **Input.** The HTTP handler parses the payload and converts attached media into image or audio fields. It
   turns the client's `history` into messages, which it uses only for retry detection and semantics. `run_cycle`
   has no history parameter (`s23_on_device_clover.py` 2845).
2. **Semantics.** `interpret` reads the message into a frame, and `route` picks the tool names. A message made only
   of pleasantries goes to `social_turn`; no model call is made, and S1 records the turn (turns 1-3).
3. **S1, S2.** `run_cycle` then runs, in order, every turn:
   - S1 searches the git memory repository (`search_repo_relevant`) and the memory graph
     (`retrieve_relevant_memory_package`, which returned nothing in all 11 turns);
   - S2 recall looks for earlier messages above its similarity threshold (`recall`);
   - S2 ontology retrieves entities and runs its formula check (`retrieve`, `match_formula_csm_lmc`); skipped in
     the 4 turns that asked for a tool.
   Anything found becomes a "Retrieved Context" block (cut at 420 tokens) or, for recall, a second system message.
4. **C1.** The system prompt is chosen by one rule (`s23_on_device_clover.py` 3149-3151): tools are offered only
   when nothing was retrieved in step 3. With retrieved context, C1 gets the plain prompt without the tool
   format, whatever semantics routed. C1 prefills the prompt (`prefill_cached`, 1.3-46 s for 42-555 tokens), then
   decodes up to 256 tokens (`decode_from_cache_with_hidden`).
5. **C2.** Only when the tool format is on: `decode_hybrid_agent_turn` opens `<latent>` spans. Each step feeds the
   HybridLink vector back into C1's cache (`step_embedding_cached`) and decodes on. A tool tag (`<search>`)
   is parsed; the permission check runs; the tool runs; its result is appended to C1's cache (`append_cached`);
   C1 decodes the answer. `run_continuous_latent_step` also runs once per turn after decoding and records the
   predicted next token (always C1's first token in these turns).
6. **C3.** `_dec_model_aligned` receives C1's token ids (with logits). In 8 of 11 turns it returned C1's text; in
   turns 4, 5 and 8 it returned a shorter prefix of it. In all 11 turns the reply sent was C1's text
   (`harness/c3_vs_c1.py`). The NCM generation functions (`_generate_from_zpref_kv_exact`, `_forward_pointer_decoder`)
   were called 0 times.
7. **After the answer.** The compiler looks for Python in the reply (none in these turns); the grounding audit
   lists numbers not found in context or tool output (turns 10, 11). S1 commits the turn to the repository.
   `align_turn` maps the turn's evidence into `semantics.align` and stores the frame as the first trace layer.
8. **Background.** After each turn, S2's memory index writes a note about the user with C1, on its own thread
   (`s2-memory-index`). It is not part of any turn. Over the run it used 271 s of C1, against 255 s for all
   the turns together (`counts.py`).

### Laptop stack
1. **Input.** The handler saves media to the media folder. The server keeps each conversation's history and gives
   it to the engine.
2. **Semantics.** `interpret` and `route`, the same file as the framework's; social messages are answered with no
   model call. The Try-again check (`last_exchange`, `retry_requested`) runs on every turn with history.
3. **S1 records.** `turn_start` records the message before the engine runs.
4. **C2 loop with C1 inside.** `run_turn` first collects layer blocks:
   - S1 memory (`memory_context`): the user's stored notes, matched to the question;
   - S1 records (`records_context`);
   - S2 ontology (`ontology_context`), skipped when a tool was asked for.
   Each C1 request is one `/v1/chat/completions` stream to llama-server (`_chat`), carrying the conversation, the
   tool schemas semantics routed, and any image as an image part. A tool call comes back as a structured call;
   permission is checked (`_permit`), the tool runs (`_run_tool`), its output is added as a message, and C1 is
   asked again (2-3 requests in the tool turns).
5. **C3.** Text is decoded by llama-server. `ncm_render` was not called (the setting is off; it is trained for
   LFM3-1.7B only).
6. **After the answer.** `non_answer`, the compiler check, the S2 grounding filter (in the 7 turns where the
   ontology step ran), `semantics.align`, then `turn_end`, which commits the records.
7. **Try again.** `retry_cycle` runs two independent solutions and a Python solution, then compares them (turn 7:
   three C1 requests and one Python run).
8. **Background.** Nothing ran in the background during these turns (recall is off by default).

## What each message showed

| # | Message | Framework runtime | Laptop stack |
|---|---|---|---|
| 1-3 | Hi / How are you? / Thanks | semantics replies, no model, S1 records each | same replies, no model, records each |
| 4 | What is the capital of Australia? | unreadable text (see "C1 shared by two threads") | "Canberra", 1 C1 request |
| 5 | Pens, 3 for $2, 12 pens? | steps to the answer, no tools | "$8", 1 C1 request |
| 6 | And 45 pens? | C1 got only "And how much would 45 pens cost?": asks for the price per pen | C1 got the conversation (4 messages): "$30.15" |
| 7 | That's wrong, try again | handler error: `module 'clover_cycle' has no attribute 'context_before'` | Clover cycle: solutions 30.15, 30.15, Python 30 |
| 8 | Search the web for Jaipur's population | latent spans, `<search>`, Wikipedia, answer "3.1 million (2011)" | web_search, then fetch_url, answer "3,046,163 (2011)" |
| 9 | Use Python to compute 2**64 + 17 | S1 found a repository file, so no tool format: "4,194,304,001" from the model | run_python ran: 18446744073709551633 |
| 10 | Remember that I'm vegetarian | no remember tool in the runtime; recall's earlier-chat block entered C1; reply ends with "the population of Jaipur: 1,552,000" | `remember` tool saved MEM-USER-...-012cc8 |
| 11 | Suggest a quick dinner | the memory graph returned nothing for the note: "garlic shrimp" | S1 memory gave "[M1] The user is vegetarian": chickpea curry |
| 12 | Photo: what is in it? | the image was dropped before the turn (see "Import path"); "I don't see an image" | image part reached C1: "Invoice 4721", "Milk 3 x 180" (the image says 1.80) |
| 13 | What number is written on it? | frame marked `seen: image`, C1 had no image: "1" | "4721" |
| 14 | Read meeting-notes.txt and summarize | S1 found a repository file, so no tool format; C1 never saw the file: a generic sentence | read_file ran: the five points |
| 15 | Delete the old report files | no tool format (same rule); the model declined in text | model proposed `Remove-Item` with the path rewritten as `C:\opt\...`; permission refused it; nothing ran |

## Interactions observed

### C1 shared by two threads (framework)
S2's note writer started a C1 call at 476.3 s (`traces/framework.jsonl`, thread `s2-memory-index`). The message
"What is the capital of Australia?" arrived at 481.5 s. From 481.5 s to 527.1 s both threads called the same C1
engine (`overlap.py`: 3 overlapping spans). Both outputs from that window are unreadable: the reply, and the note.
No other turn overlapped a background call, and every other reply reads normally.

Tested afterwards in a separate process (`harness/c1_concurrency.py`), with the two prompts from that window:
- **Alone**, twice each: identical output every time ("Canberra is the capital of Australia." and "User: Hi").
- **From two threads at once**: all 6 outputs garbled, in 3 of 3 runs.

The C1 engine is not safe to call from two threads. The note writer checks that the studio is idle only before it
starts (`_idle_note_writer`, `_is_studio_idle`: no turn running and 8 s since the last one). Nothing holds the
engine while it runs, so a message that arrives during its 14 s prefill runs alongside it.

In the laptop stack, the turn holds a gate (`SideGate`) that stops side completions, and llama-server serves one
request at a time. No overlapping C1 calls were recorded.

### Retrieval decides the tools (framework)
Semantics routed a tool in turns 9, 14 and 15: `python_interpreter`, `repo_read`, then `run_shell` + `repo_read`.
In all three, S1's repository search returned a file (its own context index, or a knowledge folder's context
file). The rule "tools only when nothing was retrieved" then built the prompt without the tool format. The routed
tool was not offered to the model, and no tool ran. In turn 8, nothing was retrieved; the tool format was on and
the search ran.

### Conversation reaches the model differently
- **Framework:** the history is used for retry detection and semantics. C1 receives earlier messages only through S2
  recall, and only when a stored message passes its similarity threshold. Turns 6 and 13: recall found no match
  (best cosine 0.277 and 0.372), and C1 got the message alone. Turns 10 and 15: recall passed earlier
  conversations from other sessions ("Hi", "How are you?"), marked as possibly unrelated.
- **Laptop:** the conversation is part of every C1 request.

### Import path (framework on Linux)
The runtime puts several folders on `sys.path` (`s23_on_device_clover.py` 85-96). On k3 that loaded:
- `clover_cycle` from `src/tools/hybrid-ncm-k3/`, not `mobile/`;
- `clover_git_memory_repo` from `src/tools/hybrid-ncm-k3/`, not `mobile/` (startup record in the trace).

The handler uses names the loaded copies do not have:
- `clover_cycle.context_before`, which raised in turn 7;
- `CloverMemoryGitRepo.get_media_dir`, which exists only in `mobile/clover_git_memory_repo.py` 1920. Saving the
  photo raised inside a `try` with an empty `except`, so turn 12 ran without the image, and nothing was logged.

The runtime also loads C3's model code (`ncm_runtime.py`, `models/`) from the base folder's `ncm/`. In the
repository, `ncm_runtime.py` is absent and `models/*.py` differs from `src/tools/hybrid-ncm-k3/models/`; the copy
used came from k3's `s23-omni2b` staging folder.

The cause is the loop at `s23_on_device_clover.py` 92-94: it inserts each folder at position 0, so folders listed
later come before the runtime's own. This applies to the repository layout, which is what any server or desktop
running from a checkout gets. The phone-format folder on k3 has no module names that clash
(`harness/shadow_check.sh`).

### Semantics is common; the hand-off around it is not
Both stacks loaded the same `semantics.py`. The laptop's copy differs only in line endings (CRLF), so a
byte-for-byte check reports a difference between the two machines.

- **Laptop:** the frame's tool choice becomes the tool list C1 sees (`only_tools`).
- **Framework:** the routed tools set two flags: `allow_internet` and `allow_python`. Then the retrieval rule above
  decides the prompt. `frame.seen` (turn 13) and `frame.diet` (turn 11) do not reach C1 or S1 in this runtime.

### Memory write and read
- **Laptop:** the model called `remember`; S1 saved the note. The next conversation's `memory_context` put it into
  the prompt (turns 10-11).
- **Framework:** the runtime has no remember tool. S1 recorded turn 10 in sessions and under `tasks/`. The memory
  graph returned nothing for the dinner question.

### Laptop stack on Linux
- The tool descriptions and system prompt still say "Windows laptop" and offer `run_powershell`. The model wrote a
  PowerShell command with the Linux path rewritten as `C:\opt\...` (turn 15).
- `clover_loop.py` imports the Windows Python runner at module level, so S1 memory, S2 ontology and the Python tool
  fail to load on Linux without a POSIX runner. The first start logged
  `repo update failed: AttributeError module 'ctypes' has no attribute 'WinDLL'`.
- `server.py` puts its own folder first on `sys.path` (line 28), ahead of anything inserted before it is imported.

## Timing (CPU, 8 threads each, k3)
- **Framework C1:** prefill 1.3-1.8 s for 42-46 tokens and 12-46 s for 154-555 tokens. Two short prompts took
  longer: 43 tokens 13.7 s (the overlapping turn) and 55 tokens 5.3 s (the next turn). Decode 0.3-32 s per turn, at
  most 256 new tokens.
- **Laptop C1:** 20 requests. Prefill 0.44-1.9 s for 135-659 tokens, 5.5 s for 2,301 (a fetched web page); decode
  32.5-40.6 tokens/s (`harness/c1_stats.py`). A turn took 0.7-14.5 s.

These are this machine and these settings; they are not the phone or the laptop.

## Not established
- Why S2 recall's threshold passed "Hi" and "How are you?" for unrelated questions (turns 10, 15).
- Whether `_dec_model_aligned` ever changes C1's text in a way that reaches the reply (shorter in 3 turns here; the
  reply was C1's text in all 11).
- The framework's voice, video and file-attachment paths, and the laptop's TTS: not exercised.
- How either stack behaves with tools set to "always", or with recall on in the laptop stack.

## Evidence
- `traces/framework.jsonl`, `traces/laptop.jsonl`: every recorded layer call (arguments, results, timings,
  thread). `traces/*-turns.jsonl`: what each stack streamed back per message. `traces/*-summary.md`: per-turn
  sequences (`harness/summarize.py`).
- `harness/`: tracer, launchers, driver, k3 service script, and the analysis scripts (`overlap.py`, `counts.py`,
  `show_turn.py`, `c1_stats.py`, `c3_vs_c1.py`, `c1_concurrency.py`, `shadow_check.sh`).
- k3: `/opt/clover-k3/core-observe` (services `clover-observe-fw`, `clover-observe-laptop`; stop with
  `harness/k3.sh stop`).
