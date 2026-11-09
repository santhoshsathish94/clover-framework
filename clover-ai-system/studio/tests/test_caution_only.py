"""Review item 1: the routing reading is used for caution only. An act reading never authorizes anything and never
removes a question; a confident gather/clarify/verify reading only makes an action ask first. caution.py's exact label
probabilities are rebuilt correctly from llama-server's two passes (stub server).
Usage: python tests/test_caution_only.py"""
import json
import math
import sys
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import caution  # noqa: E402
import lfm3_native  # noqa: E402
import semantics  # noqa: E402
import superagent_jev_dory as SJ  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True

# 1. the decision engine's injection policy
eng = SJ.CloverDecisionEngine(min_confidence=0.6)
act_hi = SJ.Decision(action="act", confidence=0.97, alternatives={}, reason="", recommended_loops=1, calibrated=True)
ver_hi = SJ.Decision(action="verify", confidence=0.9, alternatives={}, reason="", recommended_loops=4, calibrated=True)
act_lo = SJ.Decision(action="act", confidence=0.2, alternatives={}, reason="", recommended_loops=4, calibrated=True)
for step in (1, 2):
    inj = eng.should_inject_authoritative(step=step, decision=act_hi, prev_action="gather")
    ok &= check(f"a confident act decision is never injected (step {step})", not inj["inject"] and inj["mode"] != "authoritative")
inj = eng.should_inject_authoritative(step=2, decision=ver_hi, prev_action="gather")
ok &= check("a confident verify decision can still be injected", inj["inject"] and inj["mode"] == "authoritative")
inj = eng.should_inject_authoritative(step=1, decision=act_lo)
ok &= check("a low-confidence act decision only escalates uncertainty", inj["mode"] == "uncertainty_escalation")


# 2. CautionEngine readings from fixed scores
class Fixed:
    def __init__(self, scores, prior=(0.0, 0.0, 0.0, 0.0)):
        self.scores, self.prior = list(scores), list(prior)

    def raw_scores(self, state, instructions=caution.INSTRUCTIONS):
        return self.prior if state == caution.NULL_STATE else self.scores


r = caution.CautionEngine(Fixed([0.0, -9.0, -9.0, -9.0]), alpha=0.0).reading("x")
ok &= check("a confident act reading adds no caution", r.label == "act" and not r.caution)
r = caution.CautionEngine(Fixed([-9.0, -9.0, 0.0, -9.0]), alpha=0.0).reading("x")
ok &= check("a confident clarify reading adds caution", r.label == "clarify" and r.caution and r.confidence >= 0.6)
r = caution.CautionEngine(Fixed([-0.1, -0.2, 0.0, -0.15]), alpha=0.0).reading("x")
ok &= check("an unsure clarify reading adds none", r.label == "clarify" and not r.caution)
r = caution.CautionEngine(Fixed([-9.0, -9.0, 0.0, -9.0], prior=[-9.0, -9.0, 2.0, -9.0]), alpha=1.0).reading("x")
ok &= check("the null-state prior is removed before the reading", r.probabilities["clarify"] < 0.9)

# the same arithmetic as the module the study's engine uses
raw, prior = [-1.0, -2.5, -0.7, -3.0], [-0.5, -1.0, -2.0, -2.5]
p1 = caution.softmax(caution.debias(raw, prior, 0.8), 0.7)
p2, c2 = SJ.debias_and_calibrate_logits(raw, prior, prior_alpha=0.8, temperature=0.7)
ok &= check("debias + softmax + confidence equal the study module's", max(abs(a - b) for a, b in zip(p1, p2)) < 1e-12
            and abs(caution.confidence(p1) - c2) < 1e-12)

# 3. apply(): caution only
rt = lfm3_native.RUNTIME
caut = caution.Reading("clarify", 0.9, {}, True, 0.0)
calm = caution.Reading("act", 0.99, {}, False, 0.0)
f = semantics.interpret("Run git status in C:\\work\\repo", rt)
before = (f.act, f.discernment)
ok &= check("an action that would run without asking now asks first", before == ("action", "proceed")
            and caution.apply(f, caut) and f.discernment == "ask" and "asks before it runs" in f.why)
f = semantics.interpret("Delete the file C:\\work\\old.txt", rt)
ok &= check("an action that already asks keeps asking, act reading or not", f.discernment == "ask"
            and not caution.apply(f, calm) and f.discernment == "ask")
f = semantics.interpret("What is the capital of Canada?", rt)
ok &= check("a question is left exactly as it was", not caution.apply(f, caut) and f.discernment == "proceed")
f = semantics.interpret("Run git status in C:\\work\\repo", rt)
ok &= check("an act reading changes nothing", not caution.apply(f, calm) and f.discernment == "proceed")


# 4. LlamaServerScorer: exact label log-probabilities from two passes (stub llama-server)
LP = {598: -2.6, 79: -0.5, 11595: -3.7, 53550: -3.2}  # true next-token log-probabilities of the first tokens
SECOND = {1483: -0.01, 2062: -0.02}  # continuation tokens of gather (79 1483) and clarify (11595 2062)


class StubServer:
    def __init__(self, top_has):
        self.top_has, self.prompts = set(top_has), []

    def request(self, method, path, body, timeout):
        if path == "/props":
            return 200, json.dumps({"bos_token": "<|startoftext|>"}).encode()
        if path == "/tokenize":
            text = body["content"]
            table = {"<|startoftext|>": [124894], "act": [598], "gather": [79, 1483], "clarify": [11595, 2062],
                     "verify": [53550]}
            return 200, json.dumps({"tokens": table.get(text, [7, 8, 9])}).encode()
        if path == "/completion":
            ids = body["prompt"]
            self.prompts.append(ids)
            if body.get("post_sampling_probs"):
                z = sum(math.exp(LP[t]) for t in [t for t, _ in body["logit_bias"]])
                top = [{"id": t, "prob": math.exp(LP[t]) / z} for t, _ in body["logit_bias"]]
                return 200, json.dumps({"completion_probabilities": [{"top_probs": top}]}).encode()
            if ids[-1] in (79, 11595):
                top = [{"id": t, "logprob": v} for t, v in SECOND.items()] + [{"id": 5, "logprob": -6.0}]
            else:
                top = [{"id": t, "logprob": LP[t]} for t in self.top_has] + [{"id": 597, "logprob": -0.4}]
            return 200, json.dumps({"completion_probabilities": [{"top_logprobs": top}]}).encode()
        raise AssertionError(path)


srv = StubServer(top_has=[79])  # only gather's first token is among the most likely; the others are rebuilt
sc = caution.LlamaServerScorer(srv.request, template="markdown")
got = sc.raw_scores("Make it faster.")
want = [LP[598], (LP[79] + SECOND[1483]) / 2, (LP[11595] + SECOND[2062]) / 2, LP[53550]]
ok &= check("labels missing from the top list are rebuilt exactly", max(abs(a - b) for a, b in zip(got, want)) < 1e-9
            and sc.floors == 0)
ok &= check("every prompt starts with the model's BOS token", all(p[0] == 124894 for p in srv.prompts))
srv2 = StubServer(top_has=[])
sc2 = caution.LlamaServerScorer(srv2.request, template="markdown")
sc2.raw_scores("Make it faster.")
ok &= check("with no label among the most likely, the estimate is counted, not hidden", sc2.floors > 0)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
