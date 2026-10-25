# Clover Research & Architecture Context — VS Code Agentic Tool-Calling Loop (`ToolCallingLoop`, OpenAI/Anthropic/Gemini Adapters & Native `HybridAgent-8B` `<tools>`/`<tool_call>` Protocol)

- **Date:** 2026-10-25
- **Direction (creator):**
  *"do complete regression or find an agentic code of any existing AI agents like gemini 4, opus or gpts then based on that understanding develop the clover ai do the analysis work properly only that can give use the right context to start our work follow clover"*
- **Primary Sources Read & Inspected:**
  1. **VS Code Copilot Chat (`v0.70.0`) Agent Engine (`C:\Users\santhoshn\AppData\Local\Programs\Microsoft VS Code\6d62b1d5da\resources\app\extensions\copilot\dist\extension.js`):**
     - `ToolCallingLoop` (`_runLoop`, `runOne`, `validateToolMessagesCore`, `stripInternalToolCallIds`, indices `12800635–12891000`).
     - `RH` (`LanguageModelWrapper._provideLanguageModelResponse`, index `12413309`).
     - `Ok` (Extension-Contributed `LanguageModelChatProvider` Endpoint Adapter, `makeChatRequest2` & `Dgn`, indices `19406875–19412000`).
     - `Cl` / `nF` (`CustomEndpoint` & OpenAI Endpoint Adapter, `sze`, `SI`, `uSt`, `lSt`, indices `12099692–12456000`).
     - `rx` (`AnthropicBYOK`, `IBn`, `EBn`, `PSt`, indices `12187140–12435000`).
     - `HI` (`GeminiBYOK`, `y3n`, `gJi`, indices `12678000–12689000`).
  2. **Official `HybridAgent-8B-Agentic` / `Qwen3` Native Tokenizer & Engine (`oss/HybridAgent/hybrid_agent/engine.py`, `prompts.py`, and `tokenizer_config.json` on `gpu`):**
     - Read `HybridGenerationEngine` (`DECODE`, `LATENT`, `FEED`, `IDLE`, `DONE` state machine over a persistent KV working memory with `_tool_barrier`) and the exact Jinja `chat_template` embedded in `togethercomputer/HybridAgent-8B-Agentic`.

---

## 1. How Existing Frontier AI Agents (GPT, Claude Opus, Gemini 4) Execute the Agentic Loop in VS Code

### 1.1 The Outer Loop (`ToolCallingLoop._runLoop` & `runOne` in `extension.js`)
In VS Code's `</> Agent` mode, the agent loop is driven by `ToolCallingLoop`:
1. **Round $r = 0, 1, 2, \dots$ (`_runLoop`):**
   - Calls `getAvailableTools()` to collect the active workspace tools (`read_file`, `list_dir`, `file_search`, `grep_search`, `run_in_terminal`, `replace_string_in_file`, `create_file`, `get_errors`, etc.), each with `{ name, description, inputSchema }`.
   - Calls `buildPrompt2()` to assemble `A.messages`:
     - `[0] role: System` (coding agent instructions + tool-use instructions),
     - `[1..k] Prior conversation turns`,
     - `[current] role: User` (containing `<environment_info>`, `<workspace_info>`, `<attachments>`, `<context>`, `<reminderInstructions>`, and `<userRequest>...</userRequest>`),
     - Plus, for every completed tool round $j < r$ in the current turn:
       - `role: Assistant` with `toolCalls: [{ id: callId, function: { name, arguments } }]` (and optional thinking/text prefix), followed immediately by
       - `role: Tool` with `toolCallId: callId` and `content: [TextPart(tool_output)]`.
   - **Strict Pairing Invariant (`validateToolMessagesCore`):** Every `role: Tool` message **must** be preceded by a `role: Assistant` message whose `toolCalls` array contains a matching `id === toolCallId` (after stripping VS Code's internal `__vscode-<N>` suffix via `stripInternalToolCallIds`). Any orphaned tool message is dropped!
2. **Dispatch to the Selected Model (`this.fetch(we)`):**
   - **Path A — Built-in OpenAI / `CustomEndpoint` (`Cl` / `nF`):**
     - Calls `sze(n, this.model)` $\to$ `SI(messages)` to serialize `messages` and `tools: [{ type: "function", function: { name, description, parameters } }]` into an OpenAI `/v1/chat/completions` POST request.
     - Streams SSE chunks through `uSt` (`this.toolCalls.update(A)`):
       - When `delta.tool_calls` arrives (`{ id, function: { name, arguments } }`), `uSt` accumulates `id`, `name`, and `arguments`.
       - When `finish_reason === "tool_calls"` or `"stop"` arrives and `this.toolCalls.hasToolCalls()` is true, `finishedCb` receives `copilotToolCalls: [{ id, name, arguments }]`.
   - **Path B — Extension-Contributed `LanguageModelChatProvider` (`Ok.makeChatRequest2`):**
     - Calls `Dgn(messages)` to convert the internal transcript into `vscode.LanguageModelChatMessage[]`:
       - `Assistant` tool calls become `vscode.LanguageModelToolCallPart(callId, name, parsedInputObject)`.
       - `Tool` results become `vscode.LanguageModelChatMessage.User([new vscode.LanguageModelToolResultPart2(toolCallId, contentParts)])`.
     - Calls `this.languageModel.sendRequest(A, { tools, modelOptions }, token)` and iterates `for await (let P of _.stream)`:
       - If `P instanceof vscode.LanguageModelToolCallPart`, it records `{ name: P.name, arguments: JSON.stringify(P.input), id: P.callId }` into `copilotToolCalls` (`B`).
       - If `P instanceof vscode.LanguageModelTextPart`, it appends `P.value` to the assistant response `x`.
3. **Tool Execution & Loop Continuation vs. Termination:**
   - Inside `rP.processResponse`, any `copilotToolCalls` emitted during the round are invoked against VS Code's workspace tools (`vscode.lm.invokeTool`), and their outputs are stored in `this.toolCallResults[toolCallId]`.
   - Back in `_runLoop`:
     - **If `round.toolCalls.length > 0`:** The loop increments `s++` and immediately starts **Round $r + 1$**, feeding the assistant's `toolCalls` + the new `Tool` result messages back to the model so it can inspect the file/directory/terminal output and decide the next tool call!
     - **If `round.toolCalls.length === 0`:** The loop runs the stop hook (`executeStopHook`) and **terminates**, displaying the assistant's final text response to the user.

---

## 2. Why Our Previous Implementation Failed to Loop Like a True Agent

Comparing the architecture above against what we had in `k3_agent_engine.py` and `k3_chat_server.py` revealed **four root flaws**:

1. **We Did Not Use `HybridAgent-8B-Agentic`'s Native Trained `<tools>` / `<tool_call>` / `<tool_response>` Template:**
   - Inspecting `tok.chat_template` in `togethercomputer/HybridAgent-8B-Agentic` on `gpu` showed that the backbone (`Qwen3-8B`) was explicitly trained with:
     - `tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)`, which formats tool schemas inside `<tools>\n...\n</tools>`, formats prior assistant tool calls as `<tool_call>\n{"name": "...", "arguments": ...}\n</tool_call>`, and formats tool outputs as `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>`.
   - Instead of passing `tools=tools` and the multi-turn `assistant` (`tool_calls`) + `tool` (`<tool_response>`) messages into `ha_tok.apply_chat_template`, we had invented a custom prompt string (`decide_vscode_agent_step`) and forced `bridge_str = "...<tool_call>{\"name\": \""`, which fought the model's trained distribution!
2. **We Collapsed Multi-Round Tool History into a Single String Instead of Preserving `assistant` $\to$ `tool` Rounds:**
   - In `k3_chat_server.py`, we flattened `messages` into `user_prompt` + `tool_rounds` instead of passing the true multi-step `[user, assistant(tool_calls), tool(output), assistant(tool_calls), tool(output), ...]` sequence into `ha_tok.apply_chat_template(chat_msgs, tools=tools)`.
   - Because the model could not see its own prior `<tool_call>` turns paired with `<tool_response>` turns in its native format, it could not chain **Step 1 (`list_dir` / `file_search`) $\to$ Step 2 (`read_file`) $\to$ Step 3 (`grep_search` / `run_in_terminal`) $\to$ Step 4 (Final Answer)**.
3. **Why `/tmp/vscode_clover_live_verify.json` Was Not Created When Launching `code` on `gpu`:**
   - Two reasons:
     1. Our PowerShell heredoc had Windows `\r\n` line endings (`cat: '/tmp/vscode_clover_live_verify.json\r': No such file or directory`).
     2. In `vscode-extension/package.json`, `"activationEvents"` had `"onStartupFinished"`, which fires in VS Code only after workbench startup completes, and our `extension.js` checked `instanceof` / `part.callId` without handling `LanguageModelToolResultPart2` (used by `Dgn` in `copilot-chat v0.70.0`).
4. **Why Final Synthesis After Tool Rounds Lost Exact File Paths / Hashes / Line Numbers:**
   - When `tool_rounds` were present, feeding the response through `render_ncm_with_k3_experts`'s 32-token pointer-copy windows stripped code blocks and exact file paths unless code/path tokens were preserved.

---

## 3. Universal Agentic Architecture for Clover Kimi K3 (`k3_agent_engine.py`, `k3_chat_server.py`, `vscode-extension/extension.js`)

Based on this analysis, here is the clean, universal design:

1. **Native Multi-Turn `<tools>` + `<latent>` + `<tool_call>` Engine (`run_vscode_agent_turn` in `k3_agent_engine.py`):**
   - Pass the exact OpenAI `tools` list and the full multi-step message history (`system`, `user`, `assistant` with `tool_calls`, and `tool` with `content`) directly into **`self.ha_tok.apply_chat_template(chat_msgs, tools=selected_tools, tokenize=False, add_generation_prompt=True)`**.
   - Run `HybridAgent-8B`'s continuous `<latent>` reasoning block (`ENTRY_ANCHOR -> run_hybridlink(last_h) -> EXIT_ANCHOR`) inside `<think>...</think>`, then let the model naturally emit either:
     - **One or more `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` blocks** (when it needs to inspect a directory, search for files, read a file's lines, search text with grep, edit a file, or run a terminal command), OR
     - **A direct answer** (once the `<tool_response>` blocks in the conversation history contain the verified facts from the workspace).
   - Validate every emitted tool call against the tool's JSON schema (`required` properties, parameter types, and automatic workspace path resolution relative to `<workspace_info>`), and prevent duplicate calls to the exact same `(tool_name, arguments)` pair in the same turn.
2. **Grounded Final Answer Synthesis via 93-Layer Kimi K3 MoE + NCM:**
   - When the model finishes its tool rounds and emits the final answer, run `parse_and_split_plans` + `render_ncm_with_k3_experts` (93-Layer Kimi K3 MoE + `K3NativeHybridLink` + 11-Pass NCM + `K3LexicalPointerDecoder`) to render the final response, while preserving exact code blocks, file paths, and command outputs gathered during the tool loop.
3. **Full Compatibility in `clover-vscode-provider` (`vscode-extension/extension.js`):**
   - Support both `LanguageModelToolResultPart` and `LanguageModelToolResultPart2` (which `Dgn` in `copilot-chat v0.70.0` constructs), forward `options.tools` and `options.toolMode` to `/v1/chat/completions`, and report `vscode.LanguageModelToolCallPart` back to VS Code's `ToolCallingLoop`.

---

## 4. Live Execution & Verification Inside Real VS Code (`v1.143.1`) on `gpu` (`/tmp/vscode_clover_live_verify.json`)

Installed official VS Code `1.143.1` (`code_1.143.1-1792691907_amd64.deb`) and `xvfb` directly on `gpu`, installed `clover-framework.clover-vscode-provider-1.0.0` in `/home/usergpu/.vscode/extensions/` and `chatLanguageModels.json` in `/tmp/vscode-usergpu-data/User/`, launched `/usr/share/code/code` under `xvfb-run`, and executed three end-to-end agent tasks inside the live VS Code Extension Host (`allTasksPassed: true`):

| VS Code Agent Task (`vscode.version = 1.143.1`) | Tool Rounds Executed | Tools Invoked Step-by-Step | Final Grounded Reply |
|---|---:|---|---|
| **Task 1 (`task1_multi_step_explore_and_git`):** *"First list the directory `/home/usergpu/clover-framework`, then read `/home/usergpu/clover-framework/VERSION` and check how many git working trees exist in `/home/usergpu/clover-framework`, and summarize what you found."* | **`3` rounds** | **Round 1:** `list_dir({"path": "/home/usergpu/clover-framework"})` $\to$ `.git/`, `AGENTS.md`, `README.md`, `VERSION`, etc.<br>**Round 2:** `run_in_terminal({"command": "git -C \"/home/usergpu/clover-framework\" worktree list", "mode": "sync"})` $\to$ `/home/usergpu/clover-framework 0f8fad2 [main]`<br>**Round 3:** `read_file({"filePath": "/home/usergpu/clover-framework/VERSION", "startLine": 1, "endLine": 150})` $\to$ `3.5.1` | Summarizes the directory contents (`AGENTS.md`, `CHANGELOG.md`, `README.md`, `VERSION`, etc.), states the exact version (`"3.5.1"`), and reports `1` git working tree (`0f8fad2 [main]`). |
| **Task 2 (`task2_read_file_and_analyze`):** *"Read the file `/home/usergpu/clover-framework/VERSION` and tell me the exact version number."* | **`2` rounds** | **Round 1:** `list_dir({"path": "/home/usergpu/clover-framework"})`<br>**Round 2:** `read_file({"filePath": "/home/usergpu/clover-framework/VERSION", "startLine": 1, "endLine": 150})` $\to$ `3.5.1` | *"The version number from the /home/usergpu/clover-framework/VERSION file is 3.5.1. This is the exact version number as specified in the file content provided."* |
| **Task 3 (`task3_direct_math_no_false_tool`):** *"What is 25 * 4?"* | **`0` workspace rounds** | Uses internal `<python>` arithmetic without false workspace tool calls | *"The calculation of 25 * 4 is straightforward. • Final Answer: #### 25 * 4 = 100"* |

