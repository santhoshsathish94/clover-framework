"""C1 inference only (setting c1_inference_only): C1 runs one forward pass and samples no token; its hidden states for
the turn's message go to C2 (plan) and C3 (decode), and C3's text is the reply.

With stand-ins for the inference server and the C2/C3 kit:
1. run_turn never calls C1's text generation (_chat); it asks the inference server for hidden states once, over the
   chat template up to this turn's message followed by the turn's tokens.
2. Stage order C1 -> C2 -> C3, one answer event after C3; the reply is C3's text; C3 runs the full NCM; 0 new tokens.
3. S3's pre-call results join the turn's text without their instruction header, and leave the prefix.
4. A retry in this mode is read the same way (no C1 writing).
With the precomputed C2/C3 parts on this machine:
5. c1_infer_states puts the given states at each window's tokens (after the prefix), and c3_decode(full_ncm=True)
   decodes them with all 12 NCM blocks.
6. When C1 reads a multi-window turn with S1/S2/S3 evidence + headers/question, C2 (c2_plan) selects the factual
   evidence windows using C1's existing h_valid states (zero calls to C1 token_embeddings_2048) and C3 decodes the
   answer without echoing the user's question.
"""
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import omni_c2_c3  # noqa: E402
import omni_native  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


class FakeServer:
    ctx = 8192

    def __init__(self):
        self.templates, self.state_calls = [], []

    def apply_template(self, messages):
        self.templates.append(messages)
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def tokenize(self, text):
        return [len(w) for w in text.split()]

    def hidden_states(self, ids):
        self.state_calls.append(list(ids))
        return [[float(i)] * omni_c2_c3.D_LFM3 for i in range(len(ids))]


class FakeKit:
    def __init__(self):
        self.seen = {}

    def c1_infer_states(self, turn_text, states_fn, prefix_ids=(), model=None, layers_desc=None):
        h = states_fn(list(prefix_ids) + [5, 6, 7])
        self.seen.update(turn_text=turn_text, prefix=list(prefix_ids))
        return {"stage": "c1_inference", "raw_text": turn_text}, {"layer": "c1_inference", "tokens": len(h), "ms": 1.0}

    def c2_plan_state(self, q, c1_state=None, offered_tools=None):
        self.seen["c2_from"] = c1_state
        return {"stage": "c2_plan", "raw_text": c1_state["raw_text"]}, {"layer": "c2_hybridagent", "ms": 1.0}

    def c3_decode(self, c2_state, on_window=None, full_ncm=False):
        self.seen["full_ncm"] = full_ncm
        return "C3 decoded: " + c2_state["raw_text"][:60], {"layer": "ncm", "s": 0.01}

    def grounding_filter(self, q, reply):
        return reply, False


def engine():
    eng = object.__new__(omni_native.OmniNative)
    eng.lock = threading.Lock()
    eng.stage = {"name": "idle"}
    eng.media_dir = Path(tempfile.mkdtemp())
    eng._extra = {}
    eng.workdir, eng.workname = None, ""
    eng.c1_inference_only = True
    eng.srv = FakeServer()
    eng._c1_server = lambda: eng.srv

    def no_writing(*a, **k):
        raise AssertionError("C1 was asked to write text")
    eng._chat = no_writing
    return eng


# 1-2. a plain turn (the Studio keeps earlier exchanges on the same topic)
eng, kit, events = engine(), FakeKit(), []
res = eng.run_turn([{"role": "user", "content": "We want a heat pump at home."},
                    {"role": "assistant", "content": "A heat pump suits most homes."}],
                   "What does a heat pump do?", tools=False, layers=("ncm",), kit=kit, on_event=events.append,
                   system="SYS")
check("one hidden-state request, no text generation", len(eng.srv.state_calls) == 1)
check("C1 reads the conversation before the turn (system and history in the template)",
      [m["role"] for m in eng.srv.templates[-1]] == ["system", "user", "assistant", "user"]
      and "suits most homes" in eng.srv.templates[-1][2]["content"])
check("the turn's text is the user's message", kit.seen["turn_text"] == "What does a heat pump do?")
check("the prefix stops where the turn's message starts", kit.seen["prefix"] == eng.srv.tokenize(
    "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nWe want a heat pump at home.<|im_end|>\n"
    "<|im_start|>assistant\nA heat pump suits most homes.<|im_end|>\n<|im_start|>user\n"))
stages = [e["stage"] for e in events if e.get("type") == "plan" and e.get("stage")]
check("stage order C1 -> C2 -> C3", stages == ["c1_inference", "c1_done", "c2_plan", "c2_done", "ncm_decode", "ncm_done"])
answers = [i for i, e in enumerate(events) if e.get("type") == "token" and e.get("kind") == "answer"]
done = next(i for i, e in enumerate(events) if e.get("stage") == "ncm_done")
check("one answer event, after C3", len(answers) == 1 and answers[0] > done)
check("the reply is C3's text", res["reply"] == "C3 decoded: What does a heat pump do?")
check("C3 decodes with the full NCM", kit.seen["full_ncm"] is True)
check("C2 gets C1's state", kit.seen["c2_from"]["stage"] == "c1_inference")
check("no token was generated", res["generation"][0]["new_tokens"] == 0 and res["generation"][0]["c1"] == "inference only")
check("layers record C1, C2, C3 and the outcome check",
      [x["layer"] for x in res["layers"] if x["layer"] in ("c1_inference", "c2_hybridagent", "ncm", "outcome")]
      == ["c1_inference", "c2_hybridagent", "ncm", "outcome"])

# 3. S3 ran a search before C1
eng, kit = engine(), FakeKit()
eng.execute_tool = lambda name, args, ask=None, allow=(): (True, "1. Argentina won the 2022 final on penalties", "allowed")
res = eng.run_turn([], "Search the web: who won the 2022 FIFA World Cup final?", tools=True, only_tools=["web_search"],
                   layers=("ncm",), kit=kit, system="SYS")
check("the search ran before C1", [r["tool"] for r in res["tool_calls"]] == ["web_search"])
check("its results are part of the turn's text", "Argentina won the 2022 final" in kit.seen["turn_text"]
      and kit.seen["turn_text"].endswith("who won the 2022 FIFA World Cup final?"))
check("without the header that tells C1 how to answer", "already executed" not in kit.seen["turn_text"])
check("and not again in the prefix", not any("Argentina" in m["content"] for m in eng.srv.templates[-1]))

# 4. trying again
eng, kit = engine(), FakeKit()
res = eng.retry_turn([{"role": "user", "content": "3 pens cost $2. What do 30 cost?"}, {"role": "assistant", "content": "$10"}],
                     "3 pens cost $2. What do 30 cost?", "$10", "That's not right, try again.", layers=("ncm",), kit=kit,
                     system="SYS")
check("a retry is read like a turn (no C1 writing)", kit.seen["turn_text"] == "That's not right, try again."
      and len(eng.srv.state_calls) == 1)

# 5-6. the real C2/C3 parts, when this machine has them
for cand in (Path("/opt/clover-k3/web-studio"), Path.home() / "CloverAI"):
    if cand.is_dir() and not omni_c2_c3.missing_c2_c3_files(cand):
        rt = omni_c2_c3.for_base(cand)
        asked = []

        def states(ids):
            asked.append(list(ids))
            return np.repeat(np.arange(len(ids), dtype=np.float32)[:, None], omni_c2_c3.D_LFM3, axis=1)

        text = "Heat pumps move heat from outside air into a home.\n- They work in winter\n```\nkept as is\n```"
        st, info = rt.c1_infer_states(text, states, prefix_ids=[11, 12, 13])
        act = [w for w in st["windows"] if not w.get("passthrough")]
        nl = len(rt.tok.encode("\n", allow_special=False))
        check(f"{cand}: one forward pass over prefix and every line", len(asked) == 1 and len(asked[0]) == info["tokens"])
        check("the first line's states start right after the prefix", act[0]["h_valid"][0, 0, 0] == 3.0)
        check("the next line's states follow its newline", act[1]["h_valid"][0, 0, 0] == 3.0 + len(act[0]["pids"]) + nl)
        check("h_primary holds the windows' states", np.array_equal(st["h_primary"][0, :len(act[0]["pids"])], act[0]["h_valid"][0]))
        c2, _ = rt.c2_plan("What does a heat pump do?", c1_state=st, jepa_arch="none")
        out, meta = rt.c3_decode(c2, full_ncm=True)
        check("c3_decode(full_ncm=True) decodes with all 12 NCM blocks",
              rt._ncm_blocks_all is not None and len(rt._ncm_blocks_all) == 12 and bool(out.strip()))

        # 6. Verify C2 State -> Plan filters out prompt headers and echoed question lines without calling C1
        ev_turn = (
            "Live web_search results:\n"
            "1. Argentina won the 2022 FIFA World Cup final on penalties.\n\n"
            "Search the web: who won the 2022 FIFA World Cup final?"
        )
        st_ev, _ = rt.c1_infer_states(ev_turn, states, prefix_ids=[1, 2])
        orig_emb = rt.token_embeddings_2048
        rt.token_embeddings_2048 = lambda *a, **k: (_ for _ in ()).throw(AssertionError("C2 called C1 token_embeddings_2048"))
        try:
            c2_ev, _ = rt.c2_plan("Search the web: who won the 2022 FIFA World Cup final?", c1_state=st_ev, jepa_arch="none")
            out_ev, _ = rt.c3_decode(c2_ev, full_ncm=True)
        finally:
            rt.token_embeddings_2048 = orig_emb
        check("C2 State -> Plan selects evidence window and drops echoed question without calling C1",
              "Argentina won the 2022 FIFA World Cup final" in out_ev
              and "who won the 2022 FIFA World Cup final?" not in out_ev)
        break
else:
    print("SKIP the real C2/C3 parts are not on this machine")
print("ALL PASS")
