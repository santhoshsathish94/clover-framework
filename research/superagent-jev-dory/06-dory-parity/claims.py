"""Pre-registered "also reported" checks: the study's single-case Dory claims on both implementations.
w7-style verification routing (option probabilities at K = 1, 2, 4) and the chain-of-thought case (greedy, K = 1, 2, 4),
with the study's chunking and cache mode (last_loop), on the standalone engine (bf16 as the study ran it, and fp32) and
on the llama.cpp PR (BF16 and F32 GGUF)."""
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_parity as RP  # noqa: E402

sys.path.insert(0, str(RP.STUDY))
LOOPS = (1, 2, 4)
EOS = [2, 11]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def ref_score(kind, k, ids, opts):
    with tempfile.TemporaryDirectory() as td:
        p, o = Path(td) / "p.txt", Path(td) / "o.txt"
        p.write_text(" ".join(map(str, ids)) + "\n")
        o.write_text("".join(" ".join(map(str, s)) + "\n" for s in opts))
        out = subprocess.run([str(HERE / "dory_claims"), "score", str(RP.GGUF[kind]), str(k), "last_loop",
                              str(RP.THREADS), str(p), str(o)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=True).stdout.decode("utf-8", "replace")
    return [float(re.search(r"mean=(-?[\d.]+)", ln).group(1)) for ln in out.splitlines() if ln.startswith("OPTION")]


def ref_generate(kind, k, ids, max_new):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.txt"
        p.write_text(" ".join(map(str, ids)) + "\n")
        out = subprocess.run([str(HERE / "dory_claims"), "generate", str(RP.GGUF[kind]), str(k), "last_loop",
                              str(RP.THREADS), str(p), str(max_new), ",".join(map(str, EOS))],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.decode("utf-8", "replace")
    line = next(ln for ln in out.splitlines() if ln.startswith("GENERATED"))
    return [int(x) for x in line.split()[1:]]


def final_number(text):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def main():
    import torch
    import run_dory_live_experiments as D
    import run_unified_clover_ablation as U
    torch.set_num_threads(RP.THREADS)
    tok = RP.tokenizer()
    cases = [("mixed:" + t["id"], t) for t in U.MIXED_WORKLOAD_TASKS if t["id"] == "w7_hard_verify_route"]
    cases += [("decision:" + t["id"], t) for t in D.DORY_DECISION_TASKS if "agent_verify" in t["id"]]
    cot = tok.apply_chat_template([{"role": "user", "content": "If x = 3, then x = x * 2 + 1, and finally x = x * 2, "
                                    "what is the final value of x? Reply concisely with the final number at the end."}],
                                  add_generation_prompt=True, tokenize=False)
    cot_ids = tok.encode(cot, add_special_tokens=True)
    partial = HERE / "claims_partial.json"
    res = json.loads(partial.read_text()) if partial.exists() else {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "routing": {}, "cot": {"gold": 14}, "phases": []}
    save = lambda: partial.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for kind in ("bf16", "f32"):
        if f"reference_{kind}" in res["phases"]:
            continue
        for cid, t in cases:
            ids = tok.encode(t["prompt"], add_special_tokens=True)
            opts = [tok.encode(o, add_special_tokens=False) for o in t["options"]]
            for k in LOOPS:
                res["routing"].setdefault(cid, {"options": t["options"], "gold": t["gold"]})[f"reference_{kind}_k{k}"] = dict(
                    zip(t["options"], [round(p, 4) for p in softmax(ref_score(kind, k, ids, opts))]))
        for k in LOOPS:
            gen = ref_generate(kind, k, cot_ids, 200)
            text = tok.decode(gen, skip_special_tokens=True)
            res["cot"][f"reference_{kind}_k{k}"] = {"text": text[-300:], "final_number": final_number(text), "tokens": len(gen)}
        res["phases"].append(f"reference_{kind}")
        save()
        print(kind, "reference done", flush=True)
    model, _ = D.load_dory_from_safetensors(RP.MODEL_DIR)
    for dtype in ("bf16", "fp32"):
        if dtype == "fp32":
            model = model.float()
        if f"standalone_{dtype}" in res["phases"]:
            continue
        for cid, t in cases:
            ids = tok.encode(t["prompt"], add_special_tokens=True)
            for k in LOOPS:
                sc, _ = D.score_dory_options_multitoken(model, tok, ids, t["options"], num_loops=k)
                res["routing"].setdefault(cid, {"options": t["options"], "gold": t["gold"]})[f"standalone_{dtype}_k{k}"] = \
                    dict(zip(t["options"], [round(p, 4) for p in softmax(sc)]))
        for k in LOOPS:
            g = model.generate_greedy(cot_ids, max_new_tokens=200, num_loops=k, eos_token_ids=EOS, kv_mode="last_loop")
            text = tok.decode(g["generated_ids"], skip_special_tokens=True)
            res["cot"][f"standalone_{dtype}_k{k}"] = {"text": text[-300:], "final_number": final_number(text),
                                                     "tokens": len(g["generated_ids"])}
        res["phases"].append(f"standalone_{dtype}")
        save()
        print(dtype, "standalone done", flush=True)
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "claims.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("CLAIMS_DONE", flush=True)


if __name__ == "__main__":
    main()
