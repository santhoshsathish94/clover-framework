"""Automated test-suite for the Clover HG4 Fusion Gateway (Phase 1a).

    python clover-ai-system\\src\\tools\\hg4-fusion-gateway\\test_hg4_fusion_gateway.py

Runs fully offline against deterministic mock engines, plus a real `liblfm3_hg4_engine.dll` smoke test when the
native library and `%USERPROFILE%\\CloverAI\\data\\omni\\hg4` weights are present.
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends as B  # noqa: E402
import fusion_router as FR  # noqa: E402
import gateway as G  # noqa: E402
import tool_grammar as tg  # noqa: E402

BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                    "timeout": {"type": "integer"}},
                   "required": ["command"]}}}
WRITE_TOOL = {"type": "function", "function": {
    "name": "write", "description": "Write a file",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "content": {"type": "string"},
        "overwrite": {"type": "boolean"}, "lines": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["create", "append"]}},
        "required": ["filePath", "content"]}}}
TOOLS = [BASH_TOOL, WRITE_TOOL]
REPO_SYSTEM = "You are OpenCode, an agentic coding assistant.\n" + "".join(
    f"src/module_{i}.py: def handler_{i}(x): return x * {i}\n" for i in range(150))


def mock_router() -> FR.FusionRouter:
    return FR.FusionRouter(B.build_backends("mock"))


# ====================================================================================================================
class TestToolGrammar(unittest.TestCase):
    def test_render_prompt_json_and_xml(self):
        j = tg.render_tool_system_prompt(TOOLS, "auto", style="json")
        x = tg.render_tool_system_prompt(TOOLS, "required", style="xml")
        self.assertIn("<tools>", j)
        self.assertIn('{"name": <function-name>, "arguments": <args-json-object>}', j)
        self.assertIn("<function=example_function_name>", x)
        self.assertIn("MUST call at least one function", x)
        self.assertEqual(tg.render_tool_system_prompt(TOOLS, "none"), "")
        forced = tg.render_tool_system_prompt(TOOLS, {"type": "function", "function": {"name": "write"}})
        self.assertIn('"name": "write"', forced)
        self.assertNotIn('"name": "bash"', forced)

    def test_five_gnarly_malformed_tool_outputs(self):
        gnarly = [
            # 1. trailing commas + single quotes + Python literals + snake_case key + string-typed numbers
            ("<tool_call>\n{'name': 'write', 'arguments': {'file_path': 'a.py', 'content': 'x = 1', "
             "'overwrite': True, 'lines': '42',},}\n</tool_call>",
             {"filePath": "a.py", "content": "x = 1", "overwrite": True, "lines": 42}),
            # 2. raw newlines + unescaped double quotes inside a code string
            ('<tool_call>\n{"name": "write", "arguments": {"filePath": "hello.py", "content": "def hi():\n'
             '    print("hello, world")\n    return {"ok": True}\n"}}\n</tool_call>',
             {"filePath": "hello.py", "content": 'def hi():\n    print("hello, world")\n    return {"ok": True}\n'}),
            # 3. truncated: missing closing braces and </tool_call>
            ('Let me run the build.\n<tool_call>\n{"name": "bash", "arguments": {"command": "lake build", '
             '"timeout": 120',
             {"command": "lake build", "timeout": 120}),
            # 4. XML function/parameter tags with JSON-typed values needing coercion
            ("<function=write>\n<parameter=filePath>\nsrc/x.ts\n</parameter>\n<parameter=content>\n"
             "const a = {b: 1};\n</parameter>\n<parameter=overwrite>\nyes\n</parameter>\n<parameter=tags>\n"
             '["ts", "gen"]\n</parameter>\n<parameter=mode>\nAPPEND\n</parameter>\n</function>',
             {"filePath": "src/x.ts", "content": "const a = {b: 1};", "overwrite": True,
              "tags": ["ts", "gen"], "mode": "append"}),
            # 5. arguments as a stringified, glitchy JSON blob + unquoted keys + missing comma, fenced
            ('```json\n{"name": "Bash", "arguments": "{command: \'pytest -q tests/\' timeout: 30.0}"}\n```',
             {"command": "pytest -q tests/", "timeout": 30}),
        ]
        for raw, expected in gnarly:
            with self.subTest(raw=raw[:40]):
                res = tg.parse_tool_calls(raw, TOOLS)
                self.assertTrue(res.all_valid, res.problems)
                calls = res.openai_tool_calls()
                self.assertEqual(len(calls), 1)
                self.assertTrue(calls[0]["id"].startswith("call_"))
                self.assertEqual(calls[0]["type"], "function")
                args = json.loads(calls[0]["function"]["arguments"])  # always valid JSON
                self.assertEqual(args, expected)
        self.assertEqual(tg.parse_tool_calls(gnarly[2][0], TOOLS).content, "Let me run the build.")

    def test_json_repair_edge_cases(self):
        self.assertEqual(tg.repair_json('{"a": , "b": 2}'), {"a": None, "b": 2})
        self.assertEqual(tg.repair_json('{"k": "v"} trailing {'), {"k": "v"})
        self.assertEqual(tg.repair_json("{path: src/main.py, n: [1 2 3,],}"), {"path": "src/main.py", "n": [1, 2, 3]})
        self.assertEqual(tg.repair_json('{"code": "if x == "y":\n    f("a", b)\n"}'),
                         {"code": 'if x == "y":\n    f("a", b)\n'})
        with self.assertRaises(tg.JsonRepairError):
            tg.repair_json("no json here")

    def test_schema_validation_and_unknown_tools(self):
        res = tg.parse_tool_calls('<tool_call>{"name": "write", "arguments": {"content": "x"}}</tool_call>', TOOLS)
        self.assertFalse(res.all_valid)
        self.assertEqual(res.openai_tool_calls(), [])
        self.assertIn("missing required parameter 'filePath'", " ".join(res.problems))
        res2 = tg.parse_tool_calls('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', TOOLS)
        self.assertIn("unknown tool 'rm_rf'", res2.errors)
        plain = tg.parse_tool_calls("Just an answer with {braces}.", TOOLS)
        self.assertEqual(plain.tool_calls, [])
        self.assertEqual(plain.content, "Just an answer with {braces}.")

    def test_multiple_calls_hermes(self):
        raw = ('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>\n'
               '<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>')
        calls = tg.parse_tool_calls(raw, TOOLS).openai_tool_calls()
        self.assertEqual([json.loads(c["function"]["arguments"])["command"] for c in calls], ["ls", "pwd"])
        self.assertEqual([c["index"] for c in calls], [0, 1])


# ====================================================================================================================
class TestPrefixCache(unittest.TestCase):
    def _turn1(self, be):
        msgs = [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": "MOCK_CALL_BASH build it"}]
        r1 = be.generate(msgs, TOOLS, "auto", B.GenParams(max_tokens=200))
        parsed = tg.parse_tool_calls(r1.text, TOOLS)
        calls = parsed.openai_tool_calls()
        be.remember_assistant(parsed.content, calls, r1.gen_ids)
        return msgs, r1, calls

    def test_tool_result_append_zero_reprefill(self):
        for mid in ("clover-35b-a3b", "nemotron-3.5-super-vl", "lfm3-omni-2b"):
            with self.subTest(model=mid):
                be = B.make_mock_backend(mid)
                eng = be.engine
                msgs, r1, calls = self._turn1(be)
                self.assertEqual(r1.cached_tokens, 0)
                self.assertGreater(r1.prompt_tokens, 3000)
                cached_before = list(be.last_cached_ids)
                fed_before, resets_before = eng.total_fed, eng.resets
                msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": calls},
                                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Build OK: 0 errors"}]
                r2 = be.generate(msgs2, TOOLS, "auto", B.GenParams(max_tokens=200))
                prompt2 = be.build_prompt_ids(msgs2, TOOLS, "auto")
                self.assertEqual(prompt2[:len(cached_before)], cached_before)  # strict token-level extension
                self.assertEqual(r2.cached_tokens, len(cached_before))
                self.assertEqual(r2.prefill_tokens, r2.prompt_tokens - len(cached_before))
                self.assertLess(r2.prefill_tokens, 120)
                self.assertEqual(eng.resets, resets_before)  # zero re-prefill of the 3K+ token repo prefix
                # eos-terminated: every sampled token was fed exactly once, nothing else was re-fed
                self.assertEqual(r2.finish_reason, "stop")
                self.assertEqual(eng.total_fed - fed_before, r2.prefill_tokens + r2.completion_tokens)
                self.assertIn("observed tool result: Build OK", r2.text)

    def test_divergent_prefix_resets_recurrent_models(self):
        be = B.make_mock_backend("clover-35b-a3b")
        msgs, r1, _ = self._turn1(be)
        resets = be.engine.resets
        edited = [{"role": "system", "content": REPO_SYSTEM.replace("handler_3(", "handler_three(")},
                  {"role": "user", "content": "hello"}]
        r2 = be.generate(edited, TOOLS, "auto", B.GenParams(max_tokens=50))
        self.assertEqual(r2.cached_tokens, 0)
        self.assertEqual(r2.prefill_tokens, r2.prompt_tokens)
        self.assertEqual(be.engine.resets, resets + 1)

    def test_omni_truncates_to_lcp(self):
        be = B.make_mock_backend("lfm3-omni-2b")
        msgs = [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": "first question"}]
        be.generate(msgs, None, None, B.GenParams(max_tokens=50))
        resets = be.engine.resets
        r2 = be.generate(msgs[:1] + [{"role": "user", "content": "second question"}], None, None,
                         B.GenParams(max_tokens=50))
        self.assertGreater(r2.cached_tokens, 3000)  # cache_truncate(lcp) keeps the shared system prefix
        self.assertLess(r2.prefill_tokens, 40)
        self.assertEqual(be.engine.resets, resets)
        self.assertEqual(be.engine.state, be.last_cached_ids)

    def test_identical_prompt_reuses_logits(self):
        be = B.make_mock_backend("clover-35b-a3b")
        msgs = [{"role": "user", "content": "MOCK_CALL_BASH x"}]
        be.generate(msgs, TOOLS, "auto", B.GenParams(max_tokens=1))
        r = be.generate(msgs, TOOLS, "auto", B.GenParams(max_tokens=1))
        self.assertEqual((r.cached_tokens, r.prefill_tokens), (r.prompt_tokens, 0))

    def test_concurrent_requests_serialize(self):
        be = B.make_mock_backend("clover-35b-a3b")
        eng = be.engine
        state = {"active": 0, "max": 0}
        lock = threading.Lock()
        orig = eng.feed

        def guarded(ids):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.0005)
            try:
                return orig(ids)
            finally:
                with lock:
                    state["active"] -= 1
        eng.feed = guarded
        errors = []

        def worker(i):
            try:
                r = be.generate([{"role": "user", "content": f"subagent {i}"}], None, None, B.GenParams(max_tokens=40))
                if f"subagent {i}" not in r.text:
                    errors.append(r.text)
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(state["max"], 1)


# ====================================================================================================================
class TestFusionRouter(unittest.TestCase):
    def setUp(self):
        self.r = mock_router()

    def _conv_with_tool(self, tool_output):
        return [{"role": "system", "content": "Lean 4 prover agent."},
                {"role": "user", "content": "Prove add_comm."},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "lake build"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": tool_output}]

    def test_cascade_escalates_only_on_objective_failure(self):
        fail = self.r.complete({"model": "clover-fusion-high", "tools": TOOLS,
                                "messages": self._conv_with_tool("Main.lean:4:2: error: unsolved goals\n⊢ a + b = b + a")})
        self.assertEqual(fail.telemetry["served_by"], "nemotron-3.5-super-vl")
        self.assertEqual(fail.telemetry["escalations"], 1)
        self.assertEqual(fail.telemetry["escalation_reason"], "tool_failure")
        self.assertIn("error:", fail.telemetry["escalation_marker"])
        ok = self.r.complete({"model": "clover-fusion-high", "tools": TOOLS,
                              "messages": self._conv_with_tool("Build completed successfully. 0 errors, 12 passed.")})
        self.assertEqual(ok.telemetry["served_by"], "clover-35b-a3b")
        self.assertEqual(ok.telemetry["escalations"], 0)
        # an old failure followed by a new user turn is not a trailing tool failure
        later = self._conv_with_tool("error: unsolved goals") + [{"role": "user", "content": "Thanks, now explain."}]
        c = self.r.complete({"model": "clover-fusion-high", "messages": later})
        self.assertEqual(c.telemetry["served_by"], "clover-35b-a3b")
        self.assertIsNone(FR.objective_failure(self._conv_with_tool("I am not confident about this proof.")))
        for marker in ("Traceback (most recent call last):\n  File", "FAILED tests/test_x.py::test_a", "2 failed, 3 passed"):
            self.assertIsNotNone(FR.objective_failure(self._conv_with_tool(marker)), marker)

    def test_cascade_escalates_on_invalid_tool_call_after_retry(self):
        c = self.r.complete({"model": "clover-fusion-high", "tools": TOOLS,
                             "messages": [{"role": "user", "content": "MOCK_BAD_TOOL_STUBBORN run the build"}]})
        self.assertEqual(c.telemetry["escalation_reason"], "invalid_tool_call_after_retry")
        self.assertEqual(c.telemetry["served_by"], "nemotron-3.5-super-vl")
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertEqual(json.loads(c.tool_calls[0]["function"]["arguments"]), {"command": "lake build"})

    def test_direct_model_retry_fixes_invalid_tool_call(self):
        c = self.r.complete({"model": "clover-35b-a3b", "tools": TOOLS,
                             "messages": [{"role": "user", "content": "MOCK_BAD_TOOL go"}]})
        self.assertEqual(c.telemetry["tool_call_retries"], 1)
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertEqual(len(c.telemetry["steps"]), 2)

    def test_goldfish_token_cap_and_structure(self):
        c = self.r.complete({"model": "clover-egm-goldfish", "max_tokens": 4000,
                             "messages": [{"role": "user", "content": "Review this diff MOCK_LONG"}]})
        self.assertLessEqual(c.usage["completion_tokens"], 400)
        self.assertEqual(c.finish_reason, "length")
        self.assertEqual(c.telemetry["max_tokens_cap"], 400)
        obj = json.loads(c.content)
        self.assertEqual(obj["verdict"], "non-veto")
        self.assertIsInstance(obj["findings"], list)
        self.assertEqual(c.telemetry["served_by"], "clover-35b-a3b")

    def test_standard_routing(self):
        triv = self.r.complete({"model": "clover-fusion-standard",
                                "messages": [{"role": "user", "content": "Format this list as bullets: a, b, c"}]})
        self.assertEqual(triv.telemetry["served_by"], "lfm3-omni-2b")
        code = self.r.complete({"model": "clover-fusion-standard", "tools": TOOLS,
                                "messages": [{"role": "user", "content": "Fix the bug in parser.py"}]})
        self.assertEqual(code.telemetry["served_by"], "clover-35b-a3b")

    def test_critique_audit_and_revise(self):
        flawed = self.r.complete({"model": "clover-fusion-max",
                                  "messages": [{"role": "user", "content": "Sum 1..n in Python (BUG expected)"}]})
        self.assertTrue(flawed.telemetry["audit"]["flaw"])
        self.assertEqual([s["role"] for s in flawed.telemetry["steps"]], ["draft", "auditor", "revise"])
        self.assertLessEqual(flawed.telemetry["steps"][1]["completion_tokens"], FR.AUDITOR_MAX_TOKENS)
        self.assertIn("revised answer", flawed.content)
        clean = self.r.complete({"model": "clover-fusion-max",
                                 "messages": [{"role": "user", "content": "Explain list comprehensions"}]})
        self.assertFalse(clean.telemetry["audit"]["flaw"])
        self.assertEqual([s["role"] for s in clean.telemetry["steps"]], ["draft", "auditor"])

    def test_elephant_and_aliases(self):
        e = self.r.complete({"model": "clover-egm-elephant", "messages": [{"role": "user", "content": "Write a spec"}]})
        self.assertEqual(e.telemetry["served_by"], "nemotron-3.5-super-vl")
        k = self.r.complete({"model": "kimi-k3.1-pro", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(k.telemetry["served_by"], "kimi-k3")
        with self.assertRaises(FR.RouterError):
            self.r.complete({"model": "gpt-9", "messages": [{"role": "user", "content": "hi"}]})


# ====================================================================================================================
class TestGatewayHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = mock_router()
        cls.srv, cls.th = G.serve_in_thread(cls.router)
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.th.join(timeout=5)

    def _req(self, method, path, body=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            payload = resp.read().decode("utf-8")
            return resp.status, resp.getheader("Content-Type"), payload
        finally:
            conn.close()

    def _sse(self, body):
        status, ctype, payload = self._req("POST", "/v1/chat/completions", dict(body, stream=True))
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/event-stream"))
        events = [ln[len("data: "):] for ln in payload.split("\n") if ln.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]

    def test_health_and_models(self):
        st, _, p = self._req("GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(p)["status"], "ok")
        st, _, p = self._req("GET", "/v1/models")
        ids = {m["id"] for m in json.loads(p)["data"]}
        for want in ("lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3", "kimi-k3.1-pro",
                     "clover-fusion-standard", "clover-fusion-high", "clover-fusion-max",
                     "clover-egm-elephant", "clover-egm-goldfish"):
            self.assertIn(want, ids)
        st, _, p = self._req("GET", "/v1/models/clover-fusion-high")
        self.assertEqual(json.loads(p)["x_clover"]["virtual"], True)

    def test_non_streaming_completion(self):
        st, _, p = self._req("POST", "/v1/chat/completions", {
            "model": "nemotron-3.5-super-vl", "messages": [{"role": "user", "content": "Hello there"}],
            "temperature": 0, "max_completion_tokens": 64})
        self.assertEqual(st, 200)
        d = json.loads(p)
        self.assertEqual(d["object"], "chat.completion")
        self.assertEqual(d["choices"][0]["message"]["content"], "[nemotron-3.5-super-vl] Hello there")
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        self.assertGreater(d["usage"]["prompt_tokens"], 0)
        self.assertIn("x_clover_telemetry", d)

    def test_max_tokens_and_stop(self):
        st, _, p = self._req("POST", "/v1/chat/completions", {
            "model": "clover-35b-a3b", "messages": [{"role": "user", "content": "MOCK_LONG"}], "max_tokens": 10})
        d = json.loads(p)
        self.assertEqual(d["usage"]["completion_tokens"], 10)
        self.assertEqual(d["choices"][0]["finish_reason"], "length")
        st, _, p = self._req("POST", "/v1/chat/completions", {
            "model": "clover-35b-a3b", "messages": [{"role": "user", "content": "alpha STOPHERE beta"}],
            "stop": ["STOP"]})
        self.assertEqual(json.loads(p)["choices"][0]["message"]["content"], "[clover-35b-a3b] alpha")

    def test_streaming_content(self):
        body = {"model": "lfm3-omni-2b", "messages": [{"role": "user", "content": "stream me please"}]}
        chunks = self._sse(body)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        self.assertEqual(text, "[lfm3-omni-2b] stream me please")
        self.assertGreater(sum(1 for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")), 3)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("x_clover_telemetry", chunks[-1])

    def test_streaming_tool_calls(self):
        chunks = self._sse({"model": "clover-35b-a3b", "tools": TOOLS, "tool_choice": "auto",
                            "stream_options": {"include_usage": True},
                            "messages": [{"role": "user", "content": 'MOCK_TOOL_RAW: Writing now.\n<tool_call>\n'
                                          '{"name": "write", "arguments": {"filePath": "a.py", "content": '
                                          '"print("x")\n",}}\n</tool_call>'}]})
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        self.assertEqual(content, "Writing now.")
        self.assertNotIn("<tool_call>", content)
        tc_deltas = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls") or []]
        self.assertTrue(tc_deltas[0]["id"].startswith("call_"))
        self.assertEqual(tc_deltas[0]["function"]["name"], "write")
        args = "".join(t["function"].get("arguments") or "" for t in tc_deltas)
        self.assertEqual(json.loads(args), {"filePath": "a.py", "content": 'print("x")\n'})
        finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(finals[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertIn("usage", chunks[-1])

    def test_lcp_over_http_and_telemetry(self):
        msgs = [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": "MOCK_CALL_BASH build"}]
        st, _, p = self._req("POST", "/v1/chat/completions", {"model": "clover-fusion-high", "tools": TOOLS,
                                                               "messages": msgs})
        d1 = json.loads(p)
        msg = d1["choices"][0]["message"]
        self.assertEqual(d1["choices"][0]["finish_reason"], "tool_calls")
        t1 = d1["x_clover_telemetry"]["steps"][0]
        msgs2 = msgs + [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "ok: 0 errors"}]
        st, _, p = self._req("POST", "/v1/chat/completions", {"model": "clover-fusion-high", "tools": TOOLS,
                                                               "messages": msgs2})
        t2 = json.loads(p)["x_clover_telemetry"]
        step = t2["steps"][0]
        self.assertEqual(step["model"], "clover-35b-a3b")
        self.assertEqual(step["cached_tokens"], t1["prompt_tokens"] + t1["completion_tokens"])
        self.assertLess(step["prefill_tokens"], 120)
        self.assertGreater(t2["prefill_cache_hit_pct"], 95.0)
        self._req("POST", "/v1/chat/completions", {"model": "clover-fusion-high",
                                                    "messages": msgs2[:-1] + [{"role": "tool", "content": "FAILED (errors=1)"}]})
        st, _, p = self._req("GET", "/v1/telemetry")
        tel = json.loads(p)
        self.assertGreaterEqual(tel["cascade"]["escalations"], 1)
        self.assertGreater(tel["totals"]["prefill_cache_hit_pct"], 0)
        self.assertIn("decode_tokens_per_sec", tel["totals"])

    def test_errors(self):
        st, _, p = self._req("POST", "/v1/chat/completions", {"model": "nope", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(st, 404)
        self.assertIn("error", json.loads(p))
        st, _, _ = self._req("POST", "/v1/chat/completions", raw=b"{not json")
        self.assertEqual(st, 400)
        st, _, _ = self._req("POST", "/v1/chat/completions", {"model": "clover-35b-a3b", "messages": []})
        self.assertEqual(st, 400)
        st, _, _ = self._req("GET", "/v1/nothing")
        self.assertEqual(st, 404)

    def test_economizer_skills_and_output_budget(self):
        ext_tools = list(TOOLS) + [
            {"type": "function", "function": {"name": "webfetch", "description": "Fetch web URL over HTTP", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "todowrite", "description": "Manage todo items", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ]
        msgs = [
            {"role": "system", "content": "Local workspace assistant."},
            {"role": "user", "content": "Read local Demo.lean"},
        ]
        for k in range(1, 4):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{k}", "type": "function", "function": {"name": "webfetch", "arguments": "{\"url\":\"http://x\"}"}}]})
            # Deliberately omit `name` on role="tool" to match real OpenCode / @ai-sdk/openai-compatible payloads
            msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": "Error: 404 Not Found\n" + ("X" * 4000)})

        # Default-on check (no x_clover_* body flags required)
        st, _, p = self._req("POST", "/v1/chat/completions", {
            "model": "lfm3-omni-2b",
            "messages": msgs,
            "tools": ext_tools,
        })
        self.assertEqual(st, 200)
        tel = json.loads(p)["x_clover_telemetry"]
        self.assertTrue(tel["economizer"]["loop_detected"])
        self.assertEqual(tel["economizer"]["blocked_tool"], "webfetch")
        self.assertIn("webfetch", tel["economizer"]["dropped"])
        self.assertGreaterEqual(tel["output_budget_trimmed"], 3)
        self.assertIn("lean4-mathlib-proving", tel["skills_indexed"])

        # Consecutive-reset check: a succeeding tool call resets the streak so later turns are NOT falsely blocked
        msgs_reset = list(msgs) + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c_ok", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "c_ok", "content": "Demo.lean"},
        ]
        st2, _, p2 = self._req("POST", "/v1/chat/completions", {
            "model": "lfm3-omni-2b",
            "messages": msgs_reset,
            "tools": ext_tools,
        })
        self.assertEqual(st2, 200)
        self.assertFalse(json.loads(p2)["x_clover_telemetry"]["economizer"]["loop_detected"])


# ====================================================================================================================
_OMNI = B.discover_paths()["lfm3-omni-2b"]
_HAS_OMNI = bool(_OMNI.get("dir") and _OMNI.get("lib") and (Path(_OMNI["dir"]) / "tokenizer.json").is_file())


@unittest.skipUnless(_HAS_OMNI, "native liblfm3_hg4_engine + Omni HG4 weights not present")
class TestRealOmniDLL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.be = B.make_native_backend("lfm3-omni-2b", _OMNI)
        cls.router = FR.FusionRouter({"lfm3-omni-2b": cls.be})

    def test_generate_and_lcp_reuse(self):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+3? Answer with just the number."}]
        c1 = self.router.complete({"model": "lfm3-omni-2b", "messages": msgs, "max_tokens": 16})
        self.assertIn("5", c1.content)
        msgs2 = msgs + [{"role": "assistant", "content": c1.content},
                        {"role": "user", "content": "And 10+5? Just the number."}]
        c2 = self.router.complete({"model": "lfm3-omni-2b", "messages": msgs2, "max_tokens": 16})
        self.assertIn("15", c2.content)
        s1, s2 = c1.telemetry["steps"][0], c2.telemetry["steps"][0]
        self.assertGreaterEqual(s2["cached_tokens"], s1["prompt_tokens"])
        self.assertLess(s2["prefill_tokens"], s2["prompt_tokens"])

    def test_native_tool_call(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
                                                   "parameters": {"type": "object",
                                                                  "properties": {"city": {"type": "string"}},
                                                                  "required": ["city"]}}}]
        c = self.router.complete({"model": "lfm3-omni-2b", "tools": tools, "max_tokens": 64,
                                  "messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertEqual(c.tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(c.tool_calls[0]["function"]["arguments"])["city"], "Paris")


if __name__ == "__main__":
    warnings.simplefilter("error")
    unittest.main(verbosity=2, warnings="error")
