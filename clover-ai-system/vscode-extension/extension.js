"use strict";

const vscode = require("vscode");
const http = require("http");
const https = require("https");

/**
 * ============================================================================
 * 1. CloverModelCatalogFactory (Factory Pattern — Single Responsibility)
 * ============================================================================
 * Centralizes registration, alias resolution, and capability descriptors for all
 * three switchable Layer-1 Foundation Model Backbones in the Clover AI System:
 *   1. `nemotron-3.5-super-vl` (NVIDIA Nemotron 3.5 Super VL 120B-A12B, Default)
 *   2. `lfm3-1.7b`             (Liquid AI LFM3-1.7B 24-Layer Tri-Hybrid)
 *   3. `clover-kimi-k3`        (MoonshotAI Kimi K3 93-Layer 2.78T MoE)
 */
class CloverModelCatalogFactory {
  static getSupportedModels() {
    const sharedCapabilities = {
      toolCalling: true,
      imageInput: true,
      editTools: ["find-replace", "multi-find-replace"],
    };

    return [
      {
        id: "nemotron-3.5-super-vl",
        name: "Clover — Nemotron 3.5 Super VL (120B-A12B)",
        vendor: "clover",
        family: "clover-nemotron",
        version: "4.1.2",
        maxInputTokens: 1032576,
        maxOutputTokens: 16000,
        isUserSelectable: true,
        isDefault: true,
        tooltip:
          "Clover 7-Layer AI System with NVIDIA Nemotron 3.5 Super VL (88-Layer 124.45B-A12B Mamba-2 + 512-Expert Latent ReLU² MoE + NoPE 16:1 GQA + C-RADIO Vision + 2-Layer Native MTP + HybridAgent + 11-Level NCM + Ontology GraphRAG + Generative Compilation + Memory Graph, 1M context)",
        capabilities: { ...sharedCapabilities },
      },
      {
        id: "lfm3-1.7b",
        name: "Clover — Liquid LFM3-Omni (1.7B • Samsung S23 / Edge & Server)",
        vendor: "clover",
        family: "clover-lfm3",
        version: "4.1.2",
        maxInputTokens: 1032576,
        maxOutputTokens: 16000,
        isUserSelectable: true,
        isDefault: false,
        tooltip:
          "Clover 7-Layer AI System with Liquid AI LFM3-1.7B Omni (24-Layer Tri-Hybrid Width-7 Gated ShortConv + Sliding-256 GQA + Full GQA + 4-Bit Planar FFN + Snapdragon 8 Gen 2 / Samsung S23 Mobile Optimization + HybridAgent + 11-Level NCM + Ontology GraphRAG + Generative Compilation + Memory Graph, 1.71 GiB total mobile footprint, 1M session context)",
        capabilities: { ...sharedCapabilities },
      },
      {
        id: "clover-kimi-k3",
        name: "Clover — Kimi K3 (2.78T MoE)",
        vendor: "clover",
        family: "clover-kimi-k3",
        version: "4.1.2",
        maxInputTokens: 1032576,
        maxOutputTokens: 16000,
        isUserSelectable: true,
        isDefault: false,
        tooltip:
          "Clover 7-Layer AI System with MoonshotAI Kimi K3 (93-Layer 2.78T MoE + MoonViT3d Vision + Rust Host Path + HybridAgent + 11-Level NCM + Ontology GraphRAG + Generative Compilation + Memory Graph, 1M session context)",
        capabilities: { ...sharedCapabilities },
      },
    ];
  }

  static resolveModelId(requestedId, fallbackDefaultId = "nemotron-3.5-super-vl") {
    const raw = typeof requestedId === "string" ? requestedId.trim().toLowerCase() : "";
    if (!raw) {
      return fallbackDefaultId;
    }
    if (raw.includes("nemotron")) {
      return "nemotron-3.5-super-vl";
    }
    if (raw.includes("lfm3") || raw.includes("lmf3") || raw.includes("liquid") || raw.includes("s23") || raw.includes("omni")) {
      return "lfm3-1.7b";
    }
    if (raw.includes("kimi") || raw.includes("k3")) {
      return "clover-kimi-k3";
    }
    const catalog = CloverModelCatalogFactory.getSupportedModels();
    const exact = catalog.find((m) => m.id === raw);
    return exact ? exact.id : fallbackDefaultId;
  }
}

/**
 * ============================================================================
 * 2. CloverOpenAIMessageAdapter (Adapter Pattern — Null-Safe Message & Tool Mapping)
 * ============================================================================
 * Converts VS Code `LanguageModelChatRequestMessage[]` and tool definitions into
 * validated OpenAI-compatible `/v1/chat/completions` payloads with strict null checks.
 */
class CloverOpenAIMessageAdapter {
  static convertTools(vscodeTools) {
    if (!Array.isArray(vscodeTools)) {
      return [];
    }
    const converted = [];
    for (const tool of vscodeTools) {
      if (!tool || typeof tool !== "object" || typeof tool.name !== "string" || !tool.name.trim()) {
        continue;
      }
      converted.push({
        type: "function",
        function: {
          name: tool.name.trim(),
          description: typeof tool.description === "string" ? tool.description : "",
          parameters:
            tool.inputSchema && typeof tool.inputSchema === "object"
              ? tool.inputSchema
              : { type: "object", properties: {} },
        },
      });
    }
    return converted;
  }

  static convertMessages(messages) {
    if (!Array.isArray(messages) || messages.length === 0) {
      return [{ role: "user", content: "Hello" }];
    }

    const out = [];
    for (const msg of messages) {
      if (!msg || typeof msg !== "object") {
        continue;
      }
      let role = "user";
      if (msg.role === vscode.LanguageModelChatMessageRole.Assistant) {
        role = "assistant";
      } else if (msg.role === vscode.LanguageModelChatMessageRole.System) {
        role = "system";
      }

      const contentParts = [];
      const assistantToolCalls = [];
      let hasToolResults = false;
      const rawContent = msg.content !== undefined && msg.content !== null ? msg.content : "";
      const parts = Array.isArray(rawContent) ? rawContent : [rawContent];

      for (const part of parts) {
        if (part === null || part === undefined) {
          continue;
        }
        if (typeof part === "string") {
          if (part.length > 0) {
            contentParts.push({ type: "text", text: part });
          }
        } else if (typeof part.value === "string") {
          if (part.value.length > 0) {
            contentParts.push({ type: "text", text: part.value });
          }
        } else if (part.callId && part.name && part.input !== undefined) {
          let serializedArgs = "{}";
          try {
            serializedArgs =
              typeof part.input === "string" ? part.input : JSON.stringify(part.input || {});
          } catch (_err) {
            serializedArgs = "{}";
          }
          assistantToolCalls.push({
            id: String(part.callId),
            type: "function",
            function: {
              name: String(part.name),
              arguments: serializedArgs,
            },
          });
        } else if (part.callId && part.content !== undefined) {
          hasToolResults = true;
          let toolTxt = "";
          try {
            toolTxt = Array.isArray(part.content)
              ? part.content
                  .filter((p) => p !== null && p !== undefined)
                  .map((p) => (typeof p.value === "string" ? p.value : JSON.stringify(p)))
                  .join("\n")
              : String(part.content ?? "");
          } catch (_err) {
            toolTxt = String(part.content || "");
          }
          out.push({
            role: "tool",
            tool_call_id: String(part.callId),
            content: toolTxt,
          });
        } else if (part.mimeType && part.data) {
          try {
            const b64 = Buffer.from(part.data).toString("base64");
            contentParts.push({
              type: "image_url",
              image_url: { url: `data:${part.mimeType};base64,${b64}` },
            });
          } catch (_err) {
            // Skip malformed binary attachment safely
          }
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

    return out.length > 0 ? out : [{ role: "user", content: "Hello" }];
  }
}

/**
 * ============================================================================
 * 3. CloverHttpTransportClient (Strategy / Resilient Transport Client)
 * ============================================================================
 * Handles HTTP/HTTPS transport, URL validation, timeouts, cancellation tokens,
 * and structured JSON error handling.
 */
class CloverHttpTransportClient {
  static async postJson({ endpointUrl, apiKey, payloadObj, timeoutMs, cancellationToken }) {
    if (!endpointUrl || typeof endpointUrl !== "string") {
      throw new Error("Clover endpointUrl configuration is missing or invalid.");
    }

    let parsedUrl;
    try {
      parsedUrl = new URL(endpointUrl.trim());
    } catch (urlErr) {
      throw new Error(`Invalid Clover endpointUrl "${endpointUrl}": ${urlErr.message}`);
    }

    const isHttps = parsedUrl.protocol === "https:";
    const transport = isHttps ? https : http;
    const payloadStr = JSON.stringify(payloadObj || {});
    const safeTimeoutMs =
      typeof timeoutMs === "number" && Number.isFinite(timeoutMs) && timeoutMs >= 5000
        ? timeoutMs
        : 120000;

    const headers = {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Length": Buffer.byteLength(payloadStr),
      "User-Agent": "Clover-VSCode-Provider/4.1.2",
    };
    if (typeof apiKey === "string" && apiKey.trim().length > 0) {
      headers["Authorization"] = `Bearer ${apiKey.trim()}`;
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let cancelDisposable = null;

      const finish = (err, result) => {
        if (settled) {
          return;
        }
        settled = true;
        if (cancelDisposable && typeof cancelDisposable.dispose === "function") {
          cancelDisposable.dispose();
        }
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      };

      const req = transport.request(
        {
          protocol: parsedUrl.protocol,
          hostname: parsedUrl.hostname,
          port: parsedUrl.port || (isHttps ? 443 : 80),
          path: parsedUrl.pathname + parsedUrl.search,
          method: "POST",
          headers,
          timeout: safeTimeoutMs,
        },
        (res) => {
          const chunks = [];
          res.on("data", (chunk) => {
            if (chunk) {
              chunks.push(chunk);
            }
          });
          res.on("end", () => {
            const raw = Buffer.concat(chunks).toString("utf-8");
            if (res.statusCode && res.statusCode >= 400) {
              finish(new Error(`Clover AI HTTP ${res.statusCode}: ${raw.slice(0, 600)}`));
              return;
            }
            if (!raw || !raw.trim()) {
              finish(new Error("Empty response body from Clover AI server."));
              return;
            }
            try {
              finish(null, JSON.parse(raw));
            } catch (parseErr) {
              finish(new Error(`Invalid JSON from Clover AI server: ${parseErr.message}`));
            }
          });
        }
      );

      req.on("timeout", () => {
        req.destroy();
        finish(new Error(`Clover AI request timed out after ${safeTimeoutMs} ms.`));
      });

      req.on("error", (err) => {
        finish(new Error(`Clover AI network error: ${err && err.message ? err.message : String(err)}`));
      });

      if (cancellationToken && typeof cancellationToken.onCancellationRequested === "function") {
        cancelDisposable = cancellationToken.onCancellationRequested(() => {
          req.destroy();
          finish(new Error("Request cancelled by user."));
        });
      }

      req.write(payloadStr);
      req.end();
    });
  }
}

/**
 * ============================================================================
 * 4. CloverLanguageModelChatProvider (VS Code LanguageModelChatProvider)
 * ============================================================================
 * Registers "Clover" as a native first-class vendor in the VS Code Chat & Agent
 * model picker with all three switchable foundation models (`nemotron-3.5-super-vl`,
 * `lfm3-1.7b`, and `clover-kimi-k3`).
 */
class CloverLanguageModelChatProvider {
  constructor() {
    this._onDidChange = new vscode.EventEmitter();
    this.onDidChangeLanguageModelChatInformation = this._onDidChange.event;
  }

  dispose() {
    if (this._onDidChange) {
      this._onDidChange.dispose();
    }
  }

  async provideLanguageModelChatInformation(_options, _token) {
    return CloverModelCatalogFactory.getSupportedModels();
  }

  _convertMessages(messages) {
    return CloverOpenAIMessageAdapter.convertMessages(messages);
  }

  async provideLanguageModelChatResponse(model, messages, options, progress, token) {
    if (!progress || typeof progress.report !== "function") {
      throw new Error("VS Code progress reporter is required.");
    }

    const cfg = vscode.workspace.getConfiguration("clover");
    const endpointUrl = cfg.get("endpointUrl", "https://api.cloverframework.com/v1/chat/completions");
    const defaultModel = cfg.get("defaultModel", "nemotron-3.5-super-vl");
    const apiKey = cfg.get("apiKey", "");
    const cands = cfg.get("candidates", 8);
    const timeoutMs = cfg.get("requestTimeoutMs", 120000);

    const resolvedModelId = CloverModelCatalogFactory.resolveModelId(model && model.id, defaultModel);
    const openAiTools = CloverOpenAIMessageAdapter.convertTools(options && options.tools);
    const convertedMessages = CloverOpenAIMessageAdapter.convertMessages(messages);

    const payloadObj = {
      model: resolvedModelId,
      messages: convertedMessages,
      tools: openAiTools,
      cands: typeof cands === "number" && cands >= 1 ? Math.min(16, Math.floor(cands)) : 8,
      stream: false,
    };

    const responseJson = await CloverHttpTransportClient.postJson({
      endpointUrl,
      apiKey,
      payloadObj,
      timeoutMs,
      cancellationToken: token,
    });

    const choices = Array.isArray(responseJson && responseJson.choices) ? responseJson.choices : [];
    const choice = choices[0] && typeof choices[0] === "object" ? choices[0] : {};
    const msg = choice.message && typeof choice.message === "object" ? choice.message : {};
    const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];

    if (toolCalls.length > 0 && vscode.LanguageModelToolCallPart) {
      for (const tc of toolCalls) {
        if (!tc || typeof tc !== "object") {
          continue;
        }
        const fn = tc.function && typeof tc.function === "object" ? tc.function : {};
        let parsedArgs = {};
        try {
          parsedArgs =
            typeof fn.arguments === "string"
              ? JSON.parse(fn.arguments)
              : fn.arguments && typeof fn.arguments === "object"
                ? fn.arguments
                : {};
        } catch (_e) {
          parsedArgs = {};
        }
        progress.report(
          new vscode.LanguageModelToolCallPart(
            tc.id ? String(tc.id) : `call_${Date.now()}`,
            fn.name ? String(fn.name) : "run_in_terminal",
            parsedArgs
          )
        );
      }
      return;
    }

    const text = typeof msg.content === "string" ? msg.content : "";
    if (text && vscode.LanguageModelTextPart) {
      progress.report(new vscode.LanguageModelTextPart(text));
    } else if (vscode.LanguageModelTextPart) {
      progress.report(
        new vscode.LanguageModelTextPart(
          typeof responseJson?.reply === "string" && responseJson.reply
            ? responseJson.reply
            : "Clover completed the turn."
        )
      );
    }
  }

  async provideTokenCount(_model, text, _token) {
    if (typeof text === "string") {
      return Math.max(1, Math.ceil(text.length / 4));
    }
    try {
      const raw = JSON.stringify(text ?? "");
      return Math.max(1, Math.ceil(raw.length / 4));
    } catch (_err) {
      return 1;
    }
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
          const dirPath = inp.path || "/opt/clover-k3/clover-framework";
          const entries = fs.readdirSync(dirPath, { withFileTypes: true }).map((d) => (d.isDirectory() ? d.name + "/" : d.name));
          return entries.join("\n");
        }
        if (tcPart.name === "read_file") {
          const fp = inp.filePath || "/opt/clover-k3/clover-framework/VERSION";
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
          const out = cp.execSync(`grep -rn -i "${q}" /opt/clover-k3/clover-framework/VERSION /opt/clover-k3/clover-framework/README.md 2>/dev/null || true`, {
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
        "<workspace_info>\nI am working in a workspace with the following folders:\n- /opt/clover-k3/clover-framework\n</workspace_info>\n" +
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
        "First list the directory /opt/clover-k3/clover-framework, then read /opt/clover-k3/clover-framework/VERSION and check how many git working trees exist in /opt/clover-k3/clover-framework, and summarize what you found."
      ),
      await runAgentTask(
        "task2_read_file_and_analyze",
        "Read the file /opt/clover-k3/clover-framework/VERSION and tell me the exact version number."
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
        tasks[0].finalReply.includes("4.0.0") &&
        tasks[1].roundsExecuted >= 1 &&
        tasks[1].finalReply.includes("4.0.0") &&
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

module.exports = {
  activate,
  deactivate,
  CloverModelCatalogFactory,
  CloverOpenAIMessageAdapter,
  CloverHttpTransportClient,
  CloverLanguageModelChatProvider,
};
