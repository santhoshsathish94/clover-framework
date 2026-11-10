"""Offline checks of what Clover adopted from TokenEconomizer (no engine, no VS Code), item by item:
1 the /v1 reply's usage is the engine's own count, and every VS Code step and turn is in the usage ledger;
5 usage evidence for the offered VS Code tools, used only when the "tool_deferral" setting is on.

Usage: python tests/test_token_economy.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import server  # noqa: E402
import usage_ledger as U  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# ---- 1: real usage and the ledger ----
gens = [{"prompt_tokens": 4196, "reused_tokens": 4133, "new_tokens": 31, "prefill_s": 0.4, "decode_s": 0.9},
        {"prompt_tokens": 4300, "reused_tokens": 0, "new_tokens": 12, "prefill_s": 4.66, "decode_s": 0.31}, None]
t = U.totals(gens)
check("1 totals add every model call", t == {"model_calls": 2, "prompt_tokens": 8496, "cached_tokens": 4133,
                                             "completion_tokens": 43, "prefill_s": 5.06, "decode_s": 1.21})
u = U.openai_usage(t)
check("1 OpenAI usage carries the cached prompt tokens",
      u == {"prompt_tokens": 8496, "completion_tokens": 43, "total_tokens": 8539,
            "prompt_tokens_details": {"cached_tokens": 4133}})
check("1 no model call, no usage", U.openai_usage(U.totals([])) is None)

tmp = Path(tempfile.mkdtemp(prefix="clover-usage-"))
led = U.Ledger(tmp, "te test")
rec = led.record("vscode_step", t, tool="run_in_terminal", ms=5310, model=None)
check("1 ledger line written under logs/usage", led.path.parent == tmp / "logs" / "usage"
      and led.path.name.startswith("te-test-") and rec["tool"] == "run_in_terminal" and "model" not in rec)
check("1 ledger reads back", led.read()[-1]["prompt_tokens"] == 8496)


class Reply:
    """server.Handler._openai_reply on a handler that only collects what it would send."""
    def __init__(self):
        self.sent = None

    def _send(self, code, payload, ctype=None):
        self.sent = payload


h = Reply()
server.Handler._openai_reply(h, "done", None, False, "five words in the prompt", usage=u)
check("1 a reply with counts reports them", h.sent["usage"] == u)
server.Handler._openai_reply(h, "done", None, False, "five words in the prompt")
check("1 a reply without counts keeps the old estimate", h.sent["usage"]["prompt_tokens"] == 5
      and h.sent["usage"]["completion_tokens"] == 24)
server.Handler._openai_reply(h, None, {"name": "read_file", "arguments": "{}"}, True, "q", usage=u)
last = [json.loads(x[6:]) for x in h.sent.decode("utf-8").split("\n\n") if x.startswith("data: {")][-1]
check("1 streamed: the finishing chunk carries the counts", last["usage"] == u)

saved = server.S.usage
server.S.usage = U.Ledger(tmp, "rec")
try:
    server.record_usage("turn", gens, turn_kind="VS Code answer", ms=900)
    got = server.S.usage.read()[-1]
finally:
    server.S.usage = saved
check("1 record_usage writes the turn with its kind", got["kind"] == "turn" and got["turn_kind"] == "VS Code answer"
      and got["model_calls"] == 2 and got["completion_tokens"] == 43)
server.S.usage = None
server.record_usage("turn", gens)
check("1 no ledger (tests, stand-ins): nothing breaks", True)
server.S.usage = saved

# ---- 5: usage evidence for the offered tools, behind the "tool_deferral" setting ----
import lfm3_native as N  # noqa: E402
import tool_usage as TU  # noqa: E402

cfg = Path(tempfile.mkdtemp(prefix="clover-settings-")) / "settings.json"
check("5 the setting is off by default", server.Settings(cfg).get()["tool_deferral"] is False)
cfg.write_text(json.dumps({"tool_deferral": True}), encoding="utf-8")
st5 = server.Settings(cfg)
check("5 a saved true is kept", st5.get()["tool_deferral"] is True)
try:
    st5.update({"tool_deferral": "yes"})
    bad_rejected = False
except ValueError:
    bad_rejected = True
st5.update({"tool_deferral": False})
check("5 only true/false is accepted", bad_rejected and st5.get()["tool_deferral"] is False)

use = TU.ToolUsage(Path(tempfile.mkdtemp(prefix="clover-tools-")))
offer = {"click_element", "drag_element", "read_file", "run_in_terminal"}
for k in range(TU.EVIDENCE_TURNS - 1):
    use.record(f"turn{k}", offer, "click_element" if k == 0 else None)
check("5 under the evidence: nothing deferred", use.deferred(N.VSCODE_CORE) == set())
use.record("turn9", offer)
use.record("turn9", offer, None)
check("5 a turn's steps count once", use.deferred(N.VSCODE_CORE) == {"drag_element"})
check("5 the core is never deferred", "read_file" not in use.deferred(N.VSCODE_CORE)
      and "run_in_terminal" not in use.deferred(N.VSCODE_CORE))
use.record("turn10", offer, "drag_element")
check("5 one call makes it ordinary again", use.deferred(N.VSCODE_CORE) == set())
page = "Use Playwright on the shared page: type Ana in the name box and click Greet."
check("5 a deferred browser tool leaves the full set", "drag_element" in N.active_vscode_tools(page)
      and "drag_element" not in N.active_vscode_tools(page, "", (), {"drag_element"}))
check("5 ...but not when this turn already used it", "drag_element" in
      N.active_vscode_tools(page, "", [{"name": "drag_element"}], {"drag_element"}))
check("5 the core stays even if named", "read_file" in N.active_vscode_tools(page, "", (), {"read_file"}))
ws5 = "I am working in a workspace with the following folders:\n- C:\\work\\calc \n"
check("5 a turn's steps share one id, other requests do not",
      TU.turn_id(ws5, "run the tests") == TU.turn_id(ws5.replace("calc ", "Calc"), "run the tests")
      and TU.turn_id(ws5, "run the tests") != TU.turn_id(ws5, "fix the test"))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
