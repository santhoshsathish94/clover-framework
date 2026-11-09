"""Dory parity: the study's standalone engine against llama.cpp PR #30306. See PREREGISTRATION.md.

  python run_parity.py prompts   -> prompts.json + ids.txt (freeze these before any model run)
  python run_parity.py run       -> reference logits (C++ tool), standalone logits, mutants, results.json
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = Path("/opt/clover-k3/superagent-jev-dory-study")
MODEL_DIR = Path("/opt/clover-k3/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
LOOPS = (1, 2, 4)
THREADS = int(os.environ.get("THREADS", "16"))
GGUF = {"f32": HERE / "dory-4b-f32.gguf", "bf16": HERE / "dory-4b-bf16.gguf"}
sys.path.insert(0, str(STUDY))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tokenizer():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=str(MODEL_DIR / "tokenizer.json"), bos_token="<s>",
                                  eos_token="<|im_end|>", pad_token="<|im_end|>", unk_token="<unk>")
    tok.chat_template = (MODEL_DIR / "chat_template.jinja").read_text(encoding="utf-8")
    return tok


def generic_texts(tok):
    chat = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    stations = " ".join(f"Station {i} recorded {(i * 37) % 91} millimetres of rain and a high of {(i * 13) % 35 + 5} "
                        f"degrees on day {i}; the wind came from the {['north', 'east', 'south', 'west'][i % 4]}."
                        for i in range(1, 76)) + " Which station recorded the most rain?"
    animals = ["ant", "bear", "cat", "dog", "eel", "fox", "goat", "hen", "ibis", "jay", "kiwi", "lion", "mole", "newt",
               "owl", "pig", "quail", "rat", "seal", "toad", "urchin", "vole", "wolf", "yak"]
    turns = []
    for i in range(1, 25):
        turns.append({"role": "user", "content": f"Step {i}: add {i * 3} to the running total and tell me the new "
                      f"total. Also name one animal that starts with the letter {animals[i - 1][0].upper()}."})
        turns.append({"role": "assistant", "content": f"The running total is now {sum(k * 3 for k in range(1, i + 1))}. "
                      f"An animal starting with {animals[i - 1][0].upper()}: {animals[i - 1]}."})
    turns.append({"role": "user", "content": "What is the running total now, and which animal did you name for the letter F?"})
    return [
        ("generic:prose", "The river ran fast after the storm, carrying branches and leaves past the old mill and "
                          "under the stone bridge where the children used to fish."),
        ("generic:question", "What is the capital of France, and why did it become the capital?"),
        ("generic:code", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
                         "# The value of fib(10) is"),
        ("generic:list", "Shopping list for the week: apples, bread, milk, rice, lentils, tomatoes, onions, yoghurt, "
                         "and"),
        ("generic:math", "A train leaves at 9:15 and arrives at 11:40. How long is the trip in minutes? Answer:"),
        ("generic:chat_haiku", chat([{"role": "user", "content": "Write a haiku about autumn."}])),
        ("generic:long_stations", stations),
        ("generic:long_chat", chat(turns)),
    ]


def build_prompts():
    import run_dory_live_experiments as D
    import run_unified_clover_ablation as U
    tok = tokenizer()
    items = [("mixed:" + t["id"], t["prompt"]) for t in U.MIXED_WORKLOAD_TASKS]
    items += [("decision:" + t["id"], t["prompt"]) for t in D.DORY_DECISION_TASKS]
    chat = lambda c: tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, tokenize=False)
    items += [("study:gen", chat("Compute 15 * 14 + 6. Reply with only the final integer.")),
              ("study:cot", chat("If x = 3, then x = x * 2 + 1, and finally x = x * 2, what is the final value of x? "
                                 "Reply concisely with the final number at the end."))]
    items += generic_texts(tok)
    out = [{"id": pid, "text": text, "ids": tok.encode(text, add_special_tokens=True)} for pid, text in items]
    (HERE / "prompts.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    (HERE / "ids.txt").write_text("".join(" ".join(map(str, p["ids"])) + "\n" for p in out), encoding="utf-8")
    for p in out:
        print(f"{p['id']:34} {len(p['ids']):5} tokens")
    print("prompts.json", sha256(HERE / "prompts.json"))


def reference_logits(kind, loops, n_prompts, n_vocab):
    out = HERE / f"ref_{kind}_k{loops}.f32"
    if not out.exists():
        subprocess.run([str(HERE / "dory_logits"), str(GGUF[kind]), str(loops), "per_loop", str(THREADS),
                        str(HERE / "ids.txt"), str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32).reshape(n_prompts, n_vocab)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def compare(ref, std):
    lr, ls = log_softmax(ref), log_softmax(std)
    pr = np.exp(lr)
    a = int(np.argmax(ref))
    top5r, top5s = set(np.argsort(-ref)[:5].tolist()), set(np.argsort(-std)[:5].tolist())
    cos = float(np.dot(ref.astype(np.float64), std.astype(np.float64)) /
                (np.linalg.norm(ref.astype(np.float64)) * np.linalg.norm(std.astype(np.float64))))
    return {"top1_ref": a, "top1_std": int(np.argmax(std)), "top1_agree": a == int(np.argmax(std)),
            "top5_overlap": len(top5r & top5s), "kl": float(np.sum(pr * (lr - ls))), "cos": round(cos, 8),
            "dlogp_ref_top": float(lr[a] - ls[a])}


def verdict(rows):
    misses = sum(1 for r in rows if not r["top1_agree"])
    kls = sorted(r["kl"] for r in rows)
    med = float(np.median(kls))
    p90 = float(np.percentile(kls, 90))
    if misses <= 1 and med <= 0.01 and p90 <= 0.1:
        v = "PASS"
    elif misses > 4 or med > 0.1:
        v = "FAIL"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "top1_misses": misses, "n": len(rows), "median_kl": med, "p90_kl": p90, "max_kl": kls[-1]}


def forward_variant(model, input_ids, positions, num_loops, reinject=True):
    """StandaloneDoryModel.forward without a cache; reinject=False drops the input re-injection (mutant M1)."""
    import torch
    import run_dory_live_experiments as D
    with torch.inference_mode():
        h = model.embeddings(input_ids)
        n_in, n_rec = model.cfg["n_input_layers"], model.cfg["n_recurrent_layers"]
        for i in range(n_in):
            h = model.layers[i](positions, h)
        input_h = h
        state = input_h
        for _ in range(num_loops):
            h = D.justnorm(state + input_h) if reinject else D.justnorm(state)
            for i in range(n_in, n_in + n_rec):
                h = model.layers[i](positions, h)
            state = D.justnorm(h)
        h = state
        for i in range(n_in + n_rec, model.cfg["num_hidden_layers"]):
            h = model.layers[i](positions, h)
        logits = torch.nn.functional.linear(h[-1:], model.lm_head.weight).float() * (
            model.logit_scale.float() / model.cfg["initializer_range"])
    return logits.squeeze(0)


def standalone_logits(model, prompts, loops, tag):
    import torch
    out = HERE / f"std_{tag}_k{loops}.npy"
    if out.exists():
        return np.load(out)
    rows = []
    for p in prompts:
        ids = torch.tensor(p["ids"], dtype=torch.long)
        pos = torch.arange(len(p["ids"]), dtype=torch.long)
        cache = model.create_kv_cache(num_loops=loops, mode="per_loop")
        logits, _ = model.forward(ids, pos, num_loops=loops, kv_cache=cache, kv_mode="per_loop")
        rows.append(logits.float().numpy())
    arr = np.stack(rows).astype(np.float32)
    np.save(out, arr)
    return arr


def swap_rope(model):
    """Mutant M2: each attention layer gets the other layer type's RoPE theta."""
    import run_dory_live_experiments as D
    thetas = {lt: float(v["rope_theta"]) for lt, v in model.cfg["rope_parameters"].items()
              if lt in ("sliding_attention", "full_attention") and isinstance(v, dict) and "rope_theta" in v}
    assert len(thetas) == 2 and len(set(thetas.values())) == 2, thetas
    for i, layer in enumerate(model.layers):
        if layer.is_attention:
            lt = model.cfg["layer_types"][i]
            other = next(t for t in thetas if t != lt)
            layer.mixer.rope = D.StandaloneRoPE(layer.mixer.head_dim, thetas[other])


def run():
    import torch
    import run_dory_live_experiments as D
    torch.set_num_threads(THREADS)
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    n = len(prompts)
    model, census = D.load_dory_from_safetensors(MODEL_DIR)
    n_vocab = model.cfg["vocab_size"]
    res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": THREADS,
           "torch": torch.__version__,
           "sha256": {"prompts.json": sha256(HERE / "prompts.json"), "PREREGISTRATION.md": sha256(HERE / "PREREGISTRATION.md"),
                      "run_parity.py": sha256(Path(__file__)), "dory_logits.cpp": sha256(HERE / "dory_logits.cpp")},
           "gguf_sha256": {k: (HERE / f"{v.name}.sha256").read_text().split()[0] for k, v in GGUF.items()},
           "pr_head": (HERE / "pr_head.txt").read_text().strip(), "n_prompts": n, "per_k": {}}
    std = {"bf16": {k: standalone_logits(model, prompts, k, "bf16") for k in LOOPS}}
    print("standalone bf16 done", flush=True)
    model = model.float()
    std["fp32"] = {k: standalone_logits(model, prompts, k, "fp32") for k in LOOPS}
    print("standalone fp32 done", flush=True)
    # the copy used for M1 must equal the engine's forward exactly
    p0 = prompts[0]
    ids0, pos0 = torch.tensor(p0["ids"]), torch.arange(len(p0["ids"]))
    copy_diff = float((forward_variant(model, ids0, pos0, 2) - model.forward(ids0, pos0, num_loops=2)[0]).abs().max())
    res["forward_copy_max_abs_diff"] = copy_diff
    m1 = np.stack([forward_variant(model, torch.tensor(p["ids"]), torch.arange(len(p["ids"])), 2, reinject=False)
                   .numpy() for p in prompts]).astype(np.float32)
    swap_rope(model)
    m2 = np.stack([model.forward(torch.tensor(p["ids"]), torch.arange(len(p["ids"])), num_loops=2,
                                 kv_cache=model.create_kv_cache(2), kv_mode="per_loop")[0].float().numpy()
                   for p in prompts]).astype(np.float32)
    print("mutants done", flush=True)
    for k in LOOPS:
        ref32, ref16 = reference_logits("f32", k, n, n_vocab), reference_logits("bf16", k, n, n_vocab)
        prim = [dict(compare(ref32[i], std["fp32"][k][i]), id=prompts[i]["id"]) for i in range(n)]
        sec = [dict(compare(ref16[i], std["bf16"][k][i]), id=prompts[i]["id"]) for i in range(n)]
        ctl = [dict(compare(std["fp32"][k][i], std["bf16"][k][i]), id=prompts[i]["id"]) for i in range(n)]
        refctl = [dict(compare(ref32[i], ref16[i]), id=prompts[i]["id"]) for i in range(n)]
        res["per_k"][str(k)] = {"primary": {"summary": verdict(prim), "rows": prim},
                                "secondary": {"summary": verdict(sec), "rows": sec},
                                "control_std_fp32_vs_bf16": {"summary": verdict(ctl), "rows": ctl},
                                "control_ref_f32_vs_bf16": {"summary": verdict(refctl), "rows": refctl}}
        if k == 2:
            res["mutants"] = {
                "M1_no_reinjection": verdict([dict(compare(ref32[i], m1[i]), id=prompts[i]["id"]) for i in range(n)]),
                "M2_swapped_rope_theta": verdict([dict(compare(ref32[i], m2[i]), id=prompts[i]["id"]) for i in range(n)]),
            }
        print(k, res["per_k"][str(k)]["primary"]["summary"], flush=True)
    passes = all(res["per_k"][str(k)]["primary"]["summary"]["verdict"] == "PASS" for k in LOOPS)
    mutants_fail = all(v["verdict"] == "FAIL" for v in res["mutants"].values())
    res["consistent_with_reference"] = bool(passes and mutants_fail and copy_diff == 0.0)
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("PARITY_DONE consistent_with_reference =", res["consistent_with_reference"], flush=True)


if __name__ == "__main__":
    {"prompts": build_prompts, "run": run}[sys.argv[1]]()
