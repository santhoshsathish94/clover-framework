"""VS Code agent step for the laptop: picks the next VS Code tool call with LFM3's own tool-call format.

Port of CloverK3AgentEngine.decide_vscode_agent_step (clover-agent-engine.py at d0c535a, lines 3319-4113).
  kept:    the routing gates (greeting, inline attachment, standalone code spec, self-contained math, workspace
           intent), tool normalisation (_normalize_tool_args), skipping calls already made, direct read_file for an
           explicit file path, the git-inspect command, the fallbacks after the model step;
  dropped: paths of the creator's machines (C:\\personal fallback, clover-framework redirect, /app, `go get` lines),
           the visual-output gate (it can execute chart code just to test a prompt), and the multi-file PR synthesis
           loop (_synthesize_file_implementation: GPU-sized, many minutes per file on this CPU);
  changed: the model step uses LFM3's chat template (`<tools>` JSON lines in the system turn; the model answers
           `<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>`) instead of
           HybridAgent-8B's Hermes JSON calls, without latent steps (base LFM3 has no HybridLink).
VS Code runs the returned tool itself, with its own confirmation prompts.
"""
import json
import re

import torch

import clover_loop as CL

MAX_ROUNDS = 6        # k3 allows 24; each round costs a planner prefill on this CPU
MAX_TOOLS = 8         # k3 keeps 12; every tool schema is prompt the CPU must prefill
MAX_NEW_TOKENS = 160

PRIORITY_TOOLS = ["read_file", "list_dir", "file_search", "grep_search", "run_in_terminal",
                  "replace_string_in_file", "create_file", "get_errors"]
FILE_RE = (r"(?:[A-Za-z]:[\\/][^\s\"'`<>]+|(?<![A-Za-z0-9_.\-])/(?:home|opt|tmp|var|etc|usr|root|mnt)/[a-zA-Z0-9_.\-/]+"
           r"|\b(?:[a-zA-Z0-9_.\-]+/)+[a-zA-Z0-9_.\-]+|\b[a-zA-Z0-9_\-]+\.(?:go|py|rs|ts|tsx|js|jsx|json|md|toml|yaml|yml|c|cpp|h|cu|cuh|txt|sh|ps1|cmd|bat|cs|java|kt|html|css)\b"
           r"|(?<![%{<])\b(?:VERSION|README\.md|AGENTS\.md|LICENSE|Makefile|Dockerfile)\b(?!\}))")


def _compact_schema(params):
    """Tool parameter schema without per-property descriptions (fewer prompt tokens on CPU)."""
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    props = {}
    for k, v in (params.get("properties") or {}).items():
        if isinstance(v, dict):
            props[k] = {kk: vv for kk, vv in v.items() if kk in ("type", "enum", "items")}
    out = {"type": "object", "properties": props}
    if params.get("required"):
        out["required"] = list(params["required"])
    return out


def _coerce(value, schema):
    t = (schema or {}).get("type")
    v = value.strip("\n")
    if t in ("integer", "number"):
        try:
            return int(v) if t == "integer" or re.fullmatch(r"-?\d+", v.strip()) else float(v)
        except ValueError:
            return v
    if t == "boolean":
        return v.strip().lower() == "true"
    if t in ("object", "array"):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def parse_lfm3_tool_call(text, tool_map):
    """`<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>...` -> (name, args); Hermes JSON as a fallback."""
    m = re.search(r"<function=([A-Za-z0-9_.\-]+)>([\s\S]*?)(?:</function>|$)", text)
    if m and m.group(1) in tool_map:
        name = m.group(1)
        props = tool_map[name]["function"]["parameters"].get("properties") or {}
        args = {}
        for pm in re.finditer(r"<parameter=([A-Za-z0-9_.\-]+)>\n?([\s\S]*?)\n?</parameter>", m.group(2)):
            args[pm.group(1)] = _coerce(pm.group(2), props.get(pm.group(1)))
        return name, args
    m_json = re.search(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"[\s\S]*\})", text)
    if m_json:
        for suffix in ("", "}", "}}", "\"}}"):
            try:
                obj = json.loads(m_json.group(1) + suffix)
            except ValueError:
                continue
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if obj.get("name") in tool_map and isinstance(args, dict):
                return obj["name"], args
            break
    return None, None


@torch.inference_mode()
def generate_tool_turn(eng, chat_msgs, tools, force_call, max_new=MAX_NEW_TOKENS):
    """Greedy decode of the next assistant turn with the planner (KV cache, same precision as planning)."""
    tok = eng.ha_tok
    prompt = tok.apply_chat_template(chat_msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    if force_call:
        prompt += "<tool_call>\n<function="
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids  # template already starts with BOS
    backbone = eng.ha_model.backbone
    out = backbone(input_ids=ids, use_cache=True, return_dict=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}
    gen = []
    for _ in range(max_new):
        nxt = int(logits.argmax(dim=-1).item())
        if nxt in stop:
            break
        gen.append(nxt)
        text = tok.decode(gen, skip_special_tokens=False)
        if "</tool_call>" in text or (not force_call and len(gen) >= 12 and "<tool_call>" not in text):
            break
        out = backbone(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True, return_dict=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
    text = tok.decode(gen, skip_special_tokens=False)
    return ("<tool_call>\n<function=" + text) if force_call else text, ids.shape[1], len(gen)


def decide_tool_step(eng, user_prompt, workspace_info, history, tool_rounds, available_tools, trace=None):
    """Returns {"name", "arguments" (JSON string)} for the next VS Code tool call, or None to write the final answer."""
    trace = trace if trace is not None else {}
    if not available_tools or len(tool_rounds) >= MAX_ROUNDS:
        trace["route"] = "no tools or round limit"
        return None
    raw_q = CL._extract_raw_user_query(user_prompt).strip()
    low_q = raw_q.lower()
    if not tool_rounds and CL.is_short_greeting_or_ack(raw_q):
        trace["route"] = "greeting"
        return None

    effective_q = raw_q
    if history and (len(raw_q.split()) <= 16 or any(fw in low_q for fw in (
            "see it", "check it", "read it", "do it", "use tools", "available local", "we have cloned",
            "it's available", "this branch", "that branch", "same repo"))):
        prev_users = [CL._extract_raw_user_query(str(h.get("user") or "")).strip() for h in history[-2:]
                      if str(h.get("user") or "").strip()]
        if prev_users:
            effective_q = " | ".join(prev_users + [raw_q])
    effective_low = effective_q.lower()

    has_inline_attachment = (
        "Using the attached content / session context above, answer the user's request clearly and thoroughly:\n" in user_prompt
        and "Verified Workspace Tool Outputs:" not in user_prompt)
    has_explicit_fs_path = bool(re.search(FILE_RE, effective_q) and not re.search(r"https?://", effective_q))
    is_standalone_code_spec = bool(not has_explicit_fs_path and (
        re.search(r"\b(?:def|pub\s+fn|fn)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<[^>]*>)?\s*\(", raw_q) or ">>>" in raw_q
        or low_q.startswith(("write a python function", "write a rust function", "implement a python function",
                             "implement a rust function"))))
    is_self_contained_math = bool(not has_explicit_fs_path and (
        CL.needs_agent_python_tool(raw_q, history=history) is not None
        or (len(re.findall(r"\b\d+(?:\.\d+)?\b", raw_q)) >= 3 and any(mw in low_q for mw in (
            "how many", "how much", "how long", "how far", "what is the total", "daily cost", "round to",
            "ordered triple", "trailing zeros", "kwh", "miles per hour", "compound interest")))))
    if not tool_rounds and (has_inline_attachment or is_standalone_code_spec or is_self_contained_math):
        trace["route"] = "answer without tools (attachment / code spec / math)"
        return None

    has_workspace_intent = bool(tool_rounds or has_explicit_fs_path or any(kw in effective_low for kw in (
        "repo", "repository", "worktree", "working tree", "branch", "branches", "commit", "commits",
        "recent changes", "what changed", "diff", "status", "log", "git", "file", "files", "folder", "folders",
        "directory", "directories", "path", "read the", "read file", "look at", "inspect", "see it", "local", "cloned",
        "workspace", "search the workspace", "grep", "where is", "list the dir", "list dir", "edit the file",
        "modify the file", "create a file", "run in terminal", "terminal", "command", "project", "code in",
        "this code", "my code", "function in", "class in", "bug in", "error in", "test", "build", "compile")))
    if not has_workspace_intent:
        trace["route"] = "no workspace intent"
        return None

    tool_map, tool_order = {}, []
    for t in available_tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = str(fn.get("name") or "").strip()
        if not name or name == "vscode_askQuestions":
            continue
        desc_lines = str(fn.get("description") or "").splitlines()
        desc = re.split(r"(?<=[.!?])\s", desc_lines[0])[0][:160] if desc_lines else ""
        tool_map[name] = {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": _compact_schema(fn.get("parameters") or fn.get("inputSchema") or {})}}
        tool_order.append(name)
    if not tool_map:
        trace["route"] = "no usable tools"
        return None
    selected = [n for n in PRIORITY_TOOLS if n in tool_map]
    for n in tool_order:
        if n not in selected and len(selected) < MAX_TOOLS:
            selected.append(n)
    selected_tools = [tool_map[n] for n in selected[:MAX_TOOLS]]

    m_root = re.search(r"-\s*([A-Za-z]:\\[^\r\n]+|/[^\r\n]+)", workspace_info or "")
    ws_root = m_root.group(1).strip().rstrip("\\/") if m_root else ""
    sep_ch = "\\" if ("\\" in ws_root or not ws_root) else "/"

    def _resolve_abs_path(p_str):
        p_str = str(p_str or "").strip().strip("'\"`")
        if not p_str or p_str in (".", "./", ".\\"):
            return ws_root or p_str
        if re.match(r"^[A-Za-z]:[\\/]|^/", p_str) or not ws_root:
            return p_str
        return ws_root + sep_ch + p_str.replace("/", sep_ch).replace("\\", sep_ch).lstrip("\\/")

    def _normalize_tool_args(t_name, raw_args):
        args = dict(raw_args or {})
        if t_name == "read_file":
            fp = args.get("filePath") or args.get("path") or args.get("file") or ""
            args = {"filePath": _resolve_abs_path(fp), "startLine": int(args.get("startLine") or 1),
                    "endLine": int(args.get("endLine") or 350)}
        elif t_name == "list_dir":
            dp = args.get("path") or args.get("dirPath") or args.get("directory") or ws_root
            args = {"path": _resolve_abs_path(dp)}
        elif t_name == "file_search":
            args = {"query": str(args.get("query") or args.get("pattern") or "**/*")}
        elif t_name == "grep_search":
            out_g = {"query": str(args.get("query") or args.get("pattern") or ""),
                     "isRegexp": bool(args.get("isRegexp", True))}
            if args.get("includePattern"):
                out_g["includePattern"] = str(args["includePattern"])
            args = out_g
        elif t_name == "run_in_terminal":
            cmd_s = str(args.get("command") or args.get("cmd") or "").strip()
            if sep_ch == "\\" and "&&" in cmd_s:
                cmd_s = cmd_s.replace("&&", ";")
            args = {"command": cmd_s, "explanation": str(args.get("explanation") or f"Run command for: {raw_q[:80]}"),
                    "goal": str(args.get("goal") or raw_q[:60] or "Inspect workspace"), "mode": "sync"}
        elif t_name == "create_file":
            args = {"filePath": _resolve_abs_path(args.get("filePath") or args.get("path") or ""),
                    "content": str(args.get("content") or "")}
        elif t_name == "replace_string_in_file":
            args = {"filePath": _resolve_abs_path(args.get("filePath") or args.get("path") or ""),
                    "oldString": str(args.get("oldString") or ""), "newString": str(args.get("newString") or "")}
        elif t_name == "manage_todo_list":
            import clover_cycle as C
            raw_list = args.get("todoList") or args.get("todos") or args.get("items") or []
            args = {"todoList": C.normalize_todo_list(raw_list)}
        return args

    already = set()
    for rd in tool_rounds:
        try:
            a_obj = json.loads(rd.get("arguments") or "{}")
        except ValueError:
            a_obj = {}
        already.add((str(rd.get("name") or ""), json.dumps(_normalize_tool_args(str(rd.get("name") or ""), a_obj), sort_keys=True)))
    called_names = {rd.get("name") for rd in tool_rounds}

    def _call(name, args):
        norm = _normalize_tool_args(name, args)
        if (name, json.dumps(norm, sort_keys=True)) in already:
            return None
        if name == "run_in_terminal" and not norm.get("command"):
            return None
        if name in ("read_file", "create_file", "replace_string_in_file") and not norm.get("filePath"):
            return None
        return {"name": name, "arguments": json.dumps(norm)}

    wants_explicit_list_dir = any(w in effective_low for w in ("list the dir", "list dir", "files in", "what files are in"))
    wants_worktree = any(w in effective_low for w in ("worktree", "working tree", "worktrees", "working trees"))
    wants_git_inspect = wants_worktree or any(w in effective_low for w in (
        "branch", "branches", "commit", "commits", "recent changes", "what changed", "git log", "git status",
        "git diff", "git show"))
    repo_dir = ws_root or "."

    def _build_git_inspect_cmd():
        if wants_worktree and not any(w in effective_low for w in ("recent changes", "git log", "git show")):
            return f'git -C "{repo_dir}" worktree list'
        if sep_ch == "\\":
            return f'git -C "{repo_dir}" status -sb ; git -C "{repo_dir}" branch -a ; git -C "{repo_dir}" log -n 8 --oneline'
        return f'git -C "{repo_dir}" status -sb && git -C "{repo_dir}" branch -a && git -C "{repo_dir}" log -n 8 --oneline'

    m_fpaths = [p.rstrip(".,;:)") for p in re.findall(FILE_RE, effective_q)]
    m_fpaths = [p for p in m_fpaths if not p.startswith(("http:", "https:", "//"))]

    def _file_like(abs_p):
        base = abs_p.replace("\\", "/").rstrip("/").split("/")[-1]
        return (("." in base) or base in ("VERSION", "README", "LICENSE", "Makefile", "Dockerfile")) and \
            abs_p.rstrip("\\/") != ws_root.rstrip("\\/")

    if m_fpaths and not wants_explicit_list_dir and "read_file" in tool_map and "read_file" not in called_names:
        for fp in m_fpaths:
            abs_fp = _resolve_abs_path(fp)
            if _file_like(abs_fp):
                c = _call("read_file", {"filePath": abs_fp, "startLine": 1, "endLine": 350})
                if c:
                    trace["route"] = "explicit file path"
                    return c

    if wants_git_inspect and (not wants_explicit_list_dir or "list_dir" in called_names) and \
            "run_in_terminal" in tool_map and "run_in_terminal" not in called_names:
        c = _call("run_in_terminal", {"command": _build_git_inspect_cmd()})
        if c:
            trace["route"] = "git inspect"
            return c

    sys_prompt = (
        "You are Clover, a coding agent inside VS Code, running on this laptop. "
        "Inspect the real workspace with the tools before answering; never guess file contents, folder listings "
        "or git state. "
        + (f"The workspace root is `{ws_root}`; pass full absolute paths. " if ws_root else "")
        + "When the tool results already answer the request, write the answer without another tool call.")
    chat_msgs = [{"role": "system", "content": sys_prompt}]
    for h in (history or [])[-2:]:
        if h.get("user") and h.get("reply"):
            chat_msgs.append({"role": "user", "content": str(h["user"])[:400]})
            chat_msgs.append({"role": "assistant", "content": str(h["reply"])[:400]})
    ws_header = f"Workspace info:\n{workspace_info[:1200]}\n\n" if workspace_info else ""
    chat_msgs.append({"role": "user", "content": f"{ws_header}{raw_q[:1500]}"})
    for idx, rd in enumerate(tool_rounds[-4:]):
        t_name = str(rd.get("name") or "").strip()
        try:
            a_obj = json.loads(rd.get("arguments") or "{}")
        except ValueError:
            a_obj = {}
        call_id = rd.get("id") or f"call_{idx + 1}"
        chat_msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": t_name, "arguments": _normalize_tool_args(t_name, a_obj)}}]})
        chat_msgs.append({"role": "tool", "tool_call_id": call_id, "content": str(rd.get("output") or "")[:900]})

    text, n_prompt, n_gen = generate_tool_turn(eng, chat_msgs, selected_tools, force_call=not tool_rounds)
    trace.update({"route": "model", "prompt_tokens": n_prompt, "generated_tokens": n_gen, "model_text": text[:600],
                  "tools_offered": selected})
    name, args = parse_lfm3_tool_call(text, tool_map)
    if name:
        c = _call(name, args)
        if c:
            return c
        trace["model_call_rejected"] = f"{name} (repeat or empty arguments)"
    if tool_rounds:
        trace["route"] = "model answers (no further tool)"
        return None

    read_paths = set()
    for rd in tool_rounds:
        if rd.get("name") == "read_file":
            try:
                read_paths.add(_resolve_abs_path(json.loads(rd.get("arguments") or "{}").get("filePath")).lower())
            except (ValueError, AttributeError):
                pass
    if wants_git_inspect and "run_in_terminal" in tool_map and "run_in_terminal" not in called_names:
        c = _call("run_in_terminal", {"command": _build_git_inspect_cmd()})
        if c:
            trace["route"] = "fallback: git inspect"
            return c
    if wants_explicit_list_dir and "list_dir" in tool_map and "list_dir" not in called_names and ws_root:
        c = _call("list_dir", {"path": ws_root})
        if c:
            trace["route"] = "fallback: list_dir"
            return c
    if m_fpaths and "read_file" in tool_map:
        for fp in m_fpaths:
            abs_fp = _resolve_abs_path(fp)
            if _file_like(abs_fp) and abs_fp.lower() not in read_paths:
                c = _call("read_file", {"filePath": abs_fp, "startLine": 1, "endLine": 150})
                if c:
                    trace["route"] = "fallback: read_file"
                    return c
    return None
