"""Plain-message routing benchmark (review item 2): the gate for using a routing reading in Clover, caution only.

  python run_benchmark.py freeze          -> split.json (seeded, stratified) + hashes of every input
  python run_benchmark.py baseline        -> baseline.json: what Studio's semantics already does (run on Windows)
  python run_benchmark.py run             -> results.json on k3: Omni arms (caution.py on llama-server), HF arms
See PREREGISTRATION.md for the protocol and the decision rule.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIO = Path(os.environ.get("STUDIO_CLOVER", str(HERE / "studio-clover")))
STUDY = Path(os.environ.get("STUDY_DIR", "/opt/clover-k3/superagent-jev-dory-study"))
LLAMA_SERVER = Path(os.environ.get("LLAMA_SERVER", "/opt/clover-k3/core-observe/laptop-base/tools/llama.cpp/cpu/llama-server"))
OMNI_GGUF = Path(os.environ.get("OMNI_GGUF", "/opt/clover-k3/core-observe/laptop-base/data/omni/gguf/LFM3-Omni-2B-Q4_0.gguf"))
LFM3_HF = os.environ.get("LFM3_HF", "/opt/clover-k3/models/LFM3-1.7B")
THREADS = int(os.environ.get("THREADS", "16"))
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
OMNI_ARMS = [  # (name, template, alpha); temperature 0.7 for all, as in the study
    ("O1_markdown_a0.8", "markdown", 0.8),
    ("O2_chat_a0.55", "chat", 0.55),
    ("O3_markdown_a0.0", "markdown", 0.0),
    ("O4_chat_a0.8", "chat", 0.8),
]
HF_ARMS = [("H0_raw_hfbackend", "raw", None, None), ("H1_markdown_a0.8", "upg", 0.8, False),
           ("H2_chat_a0.55", "upg", 0.55, True)]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks():
    spec = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    return spec["tasks"]


def leak(text):
    t = text.lower()
    for ch in ".,'\"?!:;()\u2014-":
        t = t.replace(ch, " ")
    return [w for w in t.split() if w == "act" or w.startswith(("gather", "clarif", "verif"))]


def freeze():
    ts = tasks()
    assert len(ts) == 200 and len({t["id"] for t in ts}) == 200
    for lbl in ("act", "gather", "clarify", "verify"):
        assert sum(1 for t in ts if t["gold"] == lbl) == 50, lbl
    bad = [(t["id"], leak(t["text"])) for t in ts if leak(t["text"])]
    assert not bad, bad
    split = {}
    for lbl in ("act", "gather", "clarify", "verify"):
        ids = sorted(t["id"] for t in ts if t["gold"] == lbl)
        random.Random(f"split-{lbl}-20261109").shuffle(ids)
        split.update({i: ("test" if k < 25 else "calibration") for k, i in enumerate(ids)})
    (HERE / "split.json").write_text(json.dumps(split, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("tasks.json", "split.json", "PREREGISTRATION.md", "run_benchmark.py"):
        print(sha256(HERE / name), name)
    print(sha256(STUDIO / "caution.py"), "caution.py")


def baseline():
    """Studio's own semantics on each message (laptop runtime, personal/collective/CSI off): does Clover already ask?"""
    sys.path.insert(0, str(STUDIO))
    import lfm3_native
    import semantics
    out = {}
    for t in tasks():
        f = semantics.interpret(t["text"], lfm3_native.RUNTIME)
        out[t["id"]] = {"act": f.act, "discernment": f.discernment, "asked": list(f.asked)}
    (HERE / "baseline.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    asks = sum(1 for v in out.values() if v["discernment"] in ("ask", "refuse"))
    print(f"baseline written: {asks} of {len(out)} messages already ask or refuse")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self):
        self.port = _free_port()
        self.log = open(HERE / "llama-server-omni.log", "w")
        self.proc = subprocess.Popen([str(LLAMA_SERVER), "-m", str(OMNI_GGUF), "--host", "127.0.0.1", "--port",
                                      str(self.port), "-c", "4096", "-np", "1", "-t", str(THREADS), "--no-webui",
                                      "-ngl", "0"], stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, LD_LIBRARY_PATH=str(LLAMA_SERVER.parent)))
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                if self.request("GET", "/health", None, 2)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server did not start")

    def request(self, method, path, body=None, timeout=60):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None,
                         headers={"Content-Type": "application/json"} if body is not None else {})
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=30)


def summarize(recs, split, base, tau):
    def caution(r):
        return r["label"] != "act" and r["confidence"] >= tau
    out = {}
    for part in ("calibration", "test", "all"):
        rs = [r for r in recs if part == "all" or split[r["id"]] == part]
        act = [r for r in rs if r["gold"] == "act"]
        non = [r for r in rs if r["gold"] != "act"]
        unflagged = [r for r in non if base[r["id"]]["discernment"] not in ("ask", "refuse")]
        out[part] = {
            "n": len(rs), "correct_4way": sum(1 for r in rs if r["label"] == r["gold"]),
            "false_caution": sum(1 for r in act if caution(r)), "n_act": len(act),
            "caution_recall": sum(1 for r in non if caution(r)), "n_non_act": len(non),
            "added_caution": sum(1 for r in unflagged if caution(r)), "n_not_already_asking": len(unflagged),
            "per_label_correct": {lbl: sum(1 for r in rs if r["gold"] == lbl and r["label"] == lbl)
                                  for lbl in ("act", "gather", "clarify", "verify")},
            "predicted": {lbl: sum(1 for r in rs if r["label"] == lbl) for lbl in ("act", "gather", "clarify", "verify")},
        }
    return out


def gate(arms, split, base):
    """Selection on the calibration half only; the decision on the test half (PREREGISTRATION.md)."""
    best = None
    for order, (name, recs) in enumerate(arms.items()):
        for tau in TAUS:
            cal = summarize(recs, split, base, tau)["calibration"]
            if cal["false_caution"] <= 1:
                key = (cal["caution_recall"], tau, -order)
                if best is None or key > best[0]:
                    best = (key, name, tau)
    if best is None:
        return {"selected": None, "verdict": "FAIL", "reason": "no arm keeps calibration false caution at 1 of 25 or less"}
    _, name, tau = best
    t = summarize(arms[name], split, base, tau)["test"]
    g1 = t["false_caution"] <= 2
    g2 = t["caution_recall"] >= 45
    g3 = t["n_not_already_asking"] > 0 and t["added_caution"] * 2 >= t["n_not_already_asking"]
    return {"selected": {"arm": name, "tau": tau}, "test": t, "G1_false_caution_le_2_of_25": g1,
            "G2_caution_recall_ge_45_of_75": g2, "G3_added_caution_ge_half": g3,
            "verdict": "PASS" if (g1 and g2 and g3) else "FAIL"}


def two_pass_consistency(scorer, texts):
    """Liveness of the exact-probability method: where a label's first token is among the n_probs most likely, the
    value rebuilt from the biased pass must equal the directly reported one."""
    import math
    worst, checked = 0.0, 0
    for text in texts:
        ids = scorer.ids(text)
        top = scorer._top(ids)
        firsts = [s[0] for s in scorer._seqs]
        q = scorer._among(ids, firsts)
        known = [i for i, t in enumerate(firsts) if t in top and q[i] > 0]
        if len(known) < 2:
            continue
        k = max(known, key=lambda i: q[i])
        log_z = top[firsts[k]] - math.log(q[k])
        for i in known:
            if i != k:
                worst = max(worst, abs(math.log(q[i]) + log_z - top[firsts[i]]))
                checked += 1
    return {"pairs_checked": checked, "max_abs_logprob_difference": worst}


def run_omni(ts):
    sys.path.insert(0, str(STUDIO))
    import caution
    srv = Server()
    arms, diag = {}, {}
    try:
        for name, template, alpha in OMNI_ARMS:
            scorer = caution.LlamaServerScorer(srv.request, template=template)
            eng = caution.CautionEngine(scorer, alpha=alpha, temperature=0.7, threshold=0.6)
            recs = []
            for t in ts:
                r = eng.reading(t["text"])
                recs.append({"id": t["id"], "gold": t["gold"], "label": r.label, "confidence": r.confidence,
                             "probabilities": r.probabilities, "ms": round(r.ms, 1)})
            arms[name] = recs
            diag[name] = {"label_token_counts": [len(s) for s in scorer._seqs], "bos": scorer._bos,
                          "floors": scorer.floors, "prompt_example_tokens": len(scorer.ids(ts[0]["text"])),
                          "prior_raw": eng.prior(),
                          "two_pass_consistency": two_pass_consistency(scorer, [t["text"] for t in ts[::10]])}
            print(name, "done", flush=True)
        props = json.loads(srv.request("GET", "/props")[1])
        diag["server"] = {"build": props.get("build_info"), "model": OMNI_GGUF.name}
    finally:
        srv.close()
    return arms, diag


def run_hf(ts):
    sys.path.insert(0, str(STUDY))
    import torch
    torch.set_num_threads(THREADS)
    import run_openjev_live_experiments as S
    from openjev.backends.hf import HFBackend
    from openjev.core.primitives import Choice, SystemOneRequest
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LFM3_HF, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LFM3_HF, dtype=torch.bfloat16, trust_remote_code=True).eval()
    raw = HFBackend(model_id=LFM3_HF, device="cpu")
    raw._torch, raw._tok, raw._model, raw.device = torch, tok, model, "cpu"
    sys.path.insert(0, str(STUDIO))
    import caution
    arms = {}
    for name, kind, alpha, chat in HF_ARMS:
        if kind == "raw":
            backend = raw
        else:
            backend = S.CloverUpgradedOpenJevBackend(model=model, tokenizer=tok, model_name="LFM3-1.7B", device="cpu",
                                                     prior_alpha=alpha, temperature=0.7)
            backend.use_chat_template = chat
        recs = []
        for t in ts:
            req = SystemOneRequest(state=t["text"], questions={"direction": Choice(
                instructions=caution.INSTRUCTIONS, criteria=dict(caution.CRITERIA))})
            a = backend.decide(req).answers["direction"]
            recs.append({"id": t["id"], "gold": t["gold"], "label": a.choice, "confidence": a.confidence,
                         "probabilities": a.probabilities})
        arms[name] = recs
        print(name, "done", flush=True)
    return arms


def run():
    ts = tasks()
    split = json.loads((HERE / "split.json").read_text(encoding="utf-8"))
    base = json.loads((HERE / "baseline.json").read_text(encoding="utf-8"))
    res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": THREADS,
           "sha256": {n: sha256(HERE / n) for n in ("tasks.json", "split.json", "baseline.json", "PREREGISTRATION.md",
                                                     "run_benchmark.py")} | {"caution.py": sha256(STUDIO / "caution.py"),
                                                                             "omni_gguf": sha256(OMNI_GGUF)}}
    omni, diag = run_omni(ts)
    res["omni_diagnostics"] = diag
    res["omni"] = {name: {"by_tau": {str(tau): summarize(recs, split, base, tau) for tau in TAUS}, "records": recs}
                   for name, recs in omni.items()}
    res["gate"] = gate(omni, split, base)
    print("GATE", json.dumps(res["gate"]), flush=True)
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    hf = run_hf(ts)
    res["hf_reference"] = {name: {"by_tau": {str(tau): summarize(recs, split, base, tau) for tau in TAUS},
                                  "records": recs} for name, recs in hf.items()}
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    {"freeze": freeze, "baseline": baseline, "run": run}[sys.argv[1]]()
