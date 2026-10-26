"use strict";

const vscode = require("vscode");
const http = require("http");
const https = require("https");

/**
 * Clover Kimi K3 LanguageModelChatProvider for VS Code (`vscode.lm.registerLanguageModelChatProvider('clover', ...)`).
 * Registers "Clover" as a native first-class vendor in the VS Code Chat & Agent model picker for any user who installs
 * the extension, without requiring Enterprise BYOK policy flags or manual JSON configuration.
 */
class CloverLanguageModelChatProvider {
  constructor() {
    this._onDidChange = new vscode.EventEmitter();
    this.onDidChangeLanguageModelChatInformation = this._onDidChange.event;
  }

  dispose() {
    this._onDidChange.dispose();
  }

  /**
   * Returns the list of Clover Kimi K3 models displayed in the VS Code model picker.
   */
  async provideLanguageModelChatInformation(_options, _token) {
    return [
      {
        id: "clover-kimi-k3",
        name: "Clover Kimi K3 (93L MoE + HybridAgent + NCM)",
        vendor: "clover",
        family: "clover-kimi-k3",
        version: "3.5.1",
        maxInputTokens: 246144,
        maxOutputTokens: 16000,
        isUserSelectable: true,
        isDefault: false,
        tooltip:
          "Clover Kimi K3 — 93-Layer Kimi K3 MoE + HybridAgent-8B + 11-Pass Next-Concept Modeling (NCM) + Ontology-Guided Neuro-Symbolic GraphRAG (256K context window)",
        capabilities: {
          toolCalling: true,
          imageInput: true,
          editTools: ["find-replace", "multi-find-replace"],
        },
      },
    ];
  }

  /**
   * Converts VS Code LanguageModelChatRequestMessage[] into OpenAI-compatible messages,
   * preserving assistant `tool_calls` (`LanguageModelToolCallPart`) and `tool` results (`LanguageModelToolResultPart`).
   */
  _convertMessages(messages) {
    const out = [];
    for (const msg of messages || []) {
      let role = "user";
      if (msg.role === vscode.LanguageModelChatMessageRole.Assistant) {
        role = "assistant";
      } else if (msg.role === vscode.LanguageModelChatMessageRole.System) {
        role = "system";
      }

      const contentParts = [];
      const assistantToolCalls = [];
      let hasToolResults = false;
      const parts = Array.isArray(msg.content) ? msg.content : [msg.content];
      for (const part of parts) {
        if (!part) {
          continue;
        }
        if (typeof part === "string") {
          contentParts.push({ type: "text", text: part });
        } else if (typeof part.value === "string") {
          contentParts.push({ type: "text", text: part.value });
        } else if (part.callId && part.name && part.input !== undefined) {
          // LanguageModelToolCallPart
          assistantToolCalls.push({
            id: String(part.callId),
            type: "function",
            function: {
              name: String(part.name),
              arguments: typeof part.input === "string" ? part.input : JSON.stringify(part.input || {}),
            },
          });
        } else if (part.callId && part.content !== undefined) {
          // LanguageModelToolResultPart
          hasToolResults = true;
          const toolTxt = Array.isArray(part.content)
            ? part.content.map((p) => (typeof p.value === "string" ? p.value : JSON.stringify(p))).join("\n")
            : String(part.content);
          out.push({
            role: "tool",
            tool_call_id: String(part.callId),
            content: toolTxt,
          });
        } else if (part.mimeType && part.data) {
          // LanguageModelDataPart (image/attachment)
          const b64 = Buffer.from(part.data).toString("base64");
          contentParts.push({
            type: "image_url",
            image_url: { url: `data:${part.mimeType};base64,${b64}` },
          });
        }
      }

      if (assistantToolCalls.length > 0) {
        out.push({
          role: "assistant",
          content: contentParts.length === 1 ? contentParts[0].text : null,
          tool_calls: assistantToolCalls,
        });
      } else if (!hasToolResults && contentParts.length === 1 && contentParts[0].type === "text") {
        out.push({ role, content: contentParts[0].text });
      } else if (!hasToolResults && contentParts.length > 0) {
        out.push({ role, content: contentParts });
      }
    }
    return out;
  }

  /**
   * Sends the chat request (including `options.tools`) to the Clover Kimi K3 `/v1/chat/completions` endpoint
   * and reports `LanguageModelToolCallPart` or `LanguageModelTextPart` back to VS Code.
   */
  async provideLanguageModelChatResponse(model, messages, options, progress, token) {
    const cfg = vscode.workspace.getConfiguration("clover");
    const endpointUrl = cfg.get("endpointUrl", "https://api.cloverframework.com/v1/chat/completions");
    const apiKey = cfg.get("apiKey", "");
    const cands = cfg.get("candidates", 8);

    const openAiTools = Array.isArray(options?.tools)
      ? options.tools.map((t) => ({
          type: "function",
          function: {
            name: t.name,
            description: t.description || "",
            parameters: t.inputSchema || { type: "object", properties: {} },
          },
        }))
      : [];

    const payload = JSON.stringify({
      model: model?.id || "clover-kimi-k3",
      messages: this._convertMessages(messages),
      tools: openAiTools,
      cands,
      stream: false,
    });

    const parsedUrl = new URL(endpointUrl);
    const isHttps = parsedUrl.protocol === "https:";
    const transport = isHttps ? https : http;

    const headers = {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(payload),
      "User-Agent": "Clover-VSCode-Provider/1.0.0",
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const responseJson = await new Promise((resolve, reject) => {
      const req = transport.request(
        {
          protocol: parsedUrl.protocol,
          hostname: parsedUrl.hostname,
          port: parsedUrl.port || (isHttps ? 443 : 80),
          path: parsedUrl.pathname + parsedUrl.search,
          method: "POST",
          headers,
          timeout: 120000,
        },
        (res) => {
          const chunks = [];
          res.on("data", (chunk) => chunks.push(chunk));
          res.on("end", () => {
            const raw = Buffer.concat(chunks).toString("utf-8");
            if (res.statusCode && res.statusCode >= 400) {
              reject(new Error(`Clover Kimi K3 HTTP ${res.statusCode}: ${raw}`));
              return;
            }
            try {
              resolve(JSON.parse(raw));
            } catch (err) {
              reject(new Error(`Invalid JSON from Clover Kimi K3: ${err.message}`));
            }
          });
        }
      );

      req.on("error", (err) => reject(err));
      if (token) {
        token.onCancellationRequested(() => {
          req.destroy();
          reject(new Error("Request cancelled"));
        });
      }
      req.write(payload);
      req.end();
    });

    const choice = (responseJson.choices && responseJson.choices[0]) || {};
    const msg = choice.message || {};
    const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
    if (toolCalls.length > 0 && vscode.LanguageModelToolCallPart) {
      for (const tc of toolCalls) {
        const fn = tc.function || {};
        let parsedArgs = {};
        try {
          parsedArgs = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : fn.arguments || {};
        } catch (_e) {
          parsedArgs = {};
        }
        progress.report(
          new vscode.LanguageModelToolCallPart(
            tc.id || `call_${Date.now()}`,
            fn.name || "run_in_terminal",
            parsedArgs
          )
        );
      }
      return;
    }
    const text = msg.content || "";
    if (text && vscode.LanguageModelTextPart) {
      progress.report(new vscode.LanguageModelTextPart(text));
    }
  }

  async provideTokenCount(_model, text, _token) {
    if (typeof text === "string") {
      return Math.max(1, Math.ceil(text.length / 4));
    }
    const raw = JSON.stringify(text || "");
    return Math.max(1, Math.ceil(raw.length / 4));
  }
}

function activate(context) {
  let provider = null;
  if (vscode.lm && typeof vscode.lm.registerLanguageModelChatProvider === "function") {
    provider = new CloverLanguageModelChatProvider();
    context.subscriptions.push(vscode.lm.registerLanguageModelChatProvider("clover", provider));
    context.subscriptions.push(provider);
  }

  // Register a self-test command that runs inside the live VS Code extension host via `vscode.lm.selectChatModels({ vendor: 'clover' })`
  // and verifies multi-step agent tool-calling (`LanguageModelToolCallPart` -> tool execution -> `LanguageModelTextPart`).
  const verifyCmd = vscode.commands.registerCommand("clover.verifyLiveAgentLoop", async () => {
    const fs = require("fs");
    const cp = require("child_process");
    const models = await vscode.lm.selectChatModels({ vendor: "clover" });
    let model = models && models[0];
    if (!model && provider) {
      const infoList = await provider.provideLanguageModelChatInformation({ silent: true }, new vscode.CancellationTokenSource().token);
      const info = infoList[0];
      model = {
        id: info.id,
        name: info.name,
        vendor: info.vendor,
        maxInputTokens: info.maxInputTokens,
        sendRequest: async (msgs, opts) => {
          const parts = [];
          const progress = { report: (p) => parts.push(p) };
          await provider.provideLanguageModelChatResponse(info, msgs, opts || {}, progress, new vscode.CancellationTokenSource().token);
          return {
            stream: (async function* () {
              for (const p of parts) {
                yield p;
              }
            })(),
          };
        },
      };
    }
    if (!model) {
      throw new Error("No Clover models registered in vscode.lm");
    }

    const tools = [
      {
        name: "list_dir",
        description: "List the contents of a directory.",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
      {
        name: "read_file",
        description: "Read the contents of a file.",
        inputSchema: {
          type: "object",
          properties: {
            filePath: { type: "string" },
            startLine: { type: "number" },
            endLine: { type: "number" },
          },
          required: ["filePath", "startLine", "endLine"],
        },
      },
      {
        name: "grep_search",
        description: "Do a fast text search in the workspace.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            isRegexp: { type: "boolean" },
            includePattern: { type: "string" },
          },
          required: ["query", "isRegexp"],
        },
      },
      {
        name: "run_in_terminal",
        description: "Execute shell commands in a persistent terminal session.",
        inputSchema: {
          type: "object",
          properties: {
            command: { type: "string" },
            explanation: { type: "string" },
            goal: { type: "string" },
            mode: { type: "string", enum: ["sync", "async"] },
          },
          required: ["command", "explanation", "goal", "mode"],
        },
      },
      {
        name: "create_file",
        description: "Create a new file in the workspace.",
        inputSchema: {
          type: "object",
          properties: {
            filePath: { type: "string" },
            content: { type: "string" },
          },
          required: ["filePath", "content"],
        },
      },
    ];

    function executeWorkspaceTool(tcPart) {
      try {
        const inp = tcPart.input || {};
        if (tcPart.name === "list_dir") {
          const dirPath = inp.path || "/home/usergpu/clover-framework";
          const entries = fs.readdirSync(dirPath, { withFileTypes: true }).map((d) => (d.isDirectory() ? d.name + "/" : d.name));
          return entries.join("\n");
        }
        if (tcPart.name === "read_file") {
          const fp = inp.filePath || "/home/usergpu/clover-framework/VERSION";
          if (!fs.existsSync(fp)) {
            return `Error: File ${fp} does not exist`;
          }
          const st = fs.statSync(fp);
          if (st.isDirectory()) {
            return `Error: ${fp} is a directory, use list_dir instead`;
          }
          return fs.readFileSync(fp, "utf-8");
        }
        if (tcPart.name === "grep_search") {
          const q = inp.query || "Clover";
          const out = cp.execSync(`grep -rn -i "${q}" /home/usergpu/clover-framework/VERSION /home/usergpu/clover-framework/README.md 2>/dev/null || true`, {
            encoding: "utf-8",
            timeout: 10000,
          }).trim();
          return out || "No matches";
        }
        if (tcPart.name === "create_file") {
          const fp = inp.filePath || "/tmp/clover_agent_created.txt";
          fs.writeFileSync(fp, String(inp.content || ""), "utf-8");
          return `Successfully created ${fp}`;
        }
        if (tcPart.name === "run_in_terminal") {
          const cmd = inp.command || "pwd";
          return cp.execSync(cmd, { encoding: "utf-8", timeout: 15000 }).trim();
        }
        return "OK";
      } catch (err) {
        return `Error executing ${tcPart.name}: ${err.message || String(err)}`;
      }
    }

    async function runAgentTask(taskId, userRequestText) {
      const prompt =
        "<workspace_info>\nI am working in a workspace with the following folders:\n- /home/usergpu/clover-framework\n</workspace_info>\n" +
        `<userRequest>\n${userRequestText}\n</userRequest>`;
      const messages = [vscode.LanguageModelChatMessage.User(prompt)];
      const trajectory = [];
      let finalReply = "";
      for (let round = 0; round < 5; round++) {
        const resp = await model.sendRequest(messages, { tools });
        const toolCalls = [];
        const textChunks = [];
        for await (const part of resp.stream) {
          if (part && part.name && part.callId) {
            toolCalls.push(part);
          } else if (part && typeof part.value === "string") {
            textChunks.push(part.value);
          }
        }
        if (toolCalls.length === 0) {
          finalReply = textChunks.join("");
          break;
        }
        const tcPart = toolCalls[0];
        const toolOut = executeWorkspaceTool(tcPart);
        trajectory.push({
          round: round + 1,
          callId: tcPart.callId,
          tool: tcPart.name,
          input: tcPart.input,
          output: toolOut,
        });
        messages.push(vscode.LanguageModelChatMessage.Assistant([tcPart]));
        messages.push(
          vscode.LanguageModelChatMessage.User([
            new vscode.LanguageModelToolResultPart(tcPart.callId, [new vscode.LanguageModelTextPart(toolOut)]),
          ])
        );
      }
      return {
        taskId,
        userRequest: userRequestText,
        roundsExecuted: trajectory.length,
        toolsCalled: trajectory.map((t) => t.tool),
        trajectory,
        finalReply,
      };
    }

    const tasks = [
      await runAgentTask(
        "task1_multi_step_explore_and_git",
        "First list the directory /home/usergpu/clover-framework, then read /home/usergpu/clover-framework/VERSION and check how many git working trees exist in /home/usergpu/clover-framework, and summarize what you found."
      ),
      await runAgentTask(
        "task2_read_file_and_analyze",
        "Read the file /home/usergpu/clover-framework/VERSION and tell me the exact version number."
      ),
      await runAgentTask(
        "task3_direct_math_no_false_tool",
        "What is 25 * 4?"
      ),
    ];

    const resultObj = {
      vscodeVersion: vscode.version,
      modelId: model.id,
      modelName: model.name,
      vendor: model.vendor,
      maxInputTokens: model.maxInputTokens,
      allTasksPassed:
        tasks[0].roundsExecuted >= 2 &&
        tasks[0].finalReply.includes("3.5.1") &&
        tasks[1].roundsExecuted >= 1 &&
        tasks[1].finalReply.includes("3.5.1") &&
        tasks[2].roundsExecuted === 0 &&
        tasks[2].finalReply.includes("100"),
      tasks,
    };
    fs.writeFileSync("/tmp/vscode_clover_live_verify.json", JSON.stringify(resultObj, null, 2), "utf-8");
    return resultObj;
  });
  context.subscriptions.push(verifyCmd);

  if (process.env.CLOVER_VSCODE_AUTOVERIFY === "1" || require("fs").existsSync("/tmp/clover_vscode_autoverify_trigger")) {
    setTimeout(() => {
      vscode.commands.executeCommand("clover.verifyLiveAgentLoop").catch((err) => {
        require("fs").writeFileSync(
          "/tmp/vscode_clover_live_verify.json",
          JSON.stringify({ error: String(err && err.stack ? err.stack : err) }, null, 2),
          "utf-8"
        );
      });
    }, 3000);
  }
}

function deactivate() {}

module.exports = { activate, deactivate };
