"""Pairwise story judge: Qwen3-8B (bf16, CPU), one prefill pass per call, compares next-token logits of "A" and "B".
Usage: judge.py controls [n] | judge.py main [n_plans]   (all paths absolute, everything under /root/judge)
"""
import json, math, os, random, sys, time

os.environ["HF_HOME"] = "/root/hf"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("JUDGE_ROOT", "/root/judge")
MODEL = "Qwen/Qwen3-8B"
torch.set_num_threads(16)

PROMPT = """You are judging two short story openings that were written for a children's story plan.

PLAN: {plan}

Story A:
{a}

Story B:
{b}

Which story is better? A better story is (a) more fluent and coherent English, with no garbled or misspelled words and no repeated phrases, AND (b) follows the plan.
Answer with a single letter: A or B."""


def trunc(s, n=110):
    return " ".join(s.split()[:n])


class Judge:
    def __init__(self):
        t = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL, cache_dir="/root/hf/hub")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir="/root/hf/hub", dtype=torch.bfloat16)
        self.model.eval()
        ia = self.tok.encode("A", add_special_tokens=False)
        ib = self.tok.encode("B", add_special_tokens=False)
        assert len(ia) == 1 and len(ib) == 1, (ia, ib)
        self.ia, self.ib = ia[0], ib[0]
        print(f"loaded in {time.time()-t:.1f}s; token ids A={self.ia} B={self.ib}", flush=True)

    def text(self, plan, a, b):
        msgs = [{"role": "user", "content": PROMPT.format(plan=plan, a=trunc(a), b=trunc(b))}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def call(self, plan, a, b):
        ids = self.tok(self.text(plan, a, b), return_tensors="pt", add_special_tokens=False)
        t = time.time()
        lg = self.model(**ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        dt = time.time() - t
        lp = torch.log_softmax(lg, -1)
        top = int(lg.argmax())
        return dict(logit_A=float(lg[self.ia]), logit_B=float(lg[self.ib]),
                    pAB=float(lp[self.ia].exp() + lp[self.ib].exp()), top=self.tok.decode([top]),
                    ntok=int(ids["input_ids"].shape[1]), sec=round(dt, 2))


def shuffled(s, seed):
    w = s.split()
    random.Random(seed).shuffle(w)
    return " ".join(w)


def controls(n):
    d = json.load(open(f"{ROOT}/data.json", encoding="utf-8"))
    j = Judge()
    print("PROMPT EXAMPLE (first control item):\n" + j.text(d["plans"][0], d["base"][0], shuffled(d["base"][0], 0)) + "\n<<<END PROMPT", flush=True)
    out = open(f"{ROOT}/controls.jsonl", "w", encoding="utf-8")
    rows = []
    for i in range(n):
        plan, s = d["plans"][i], d["base"][i]
        # negative control: same text as A and as B
        r = j.call(plan, s, s); r.update(ctrl="neg", i=i, order="same"); rows.append(r)
        out.write(json.dumps(r) + "\n"); out.flush()
        # positive control: real story vs word-shuffled copy, both orders
        sh = shuffled(s, 1000 + i)
        for order, (a, b) in (("real_first", (s, sh)), ("shuf_first", (sh, s))):
            r = j.call(plan, a, b); r.update(ctrl="pos", i=i, order=order); rows.append(r)
            out.write(json.dumps(r) + "\n"); out.flush()
        print(i, rows[-3]["sec"], rows[-2]["sec"], rows[-1]["sec"], "sec", flush=True)
    neg = [r for r in rows if r["ctrl"] == "neg"]
    pos = [r for r in rows if r["ctrl"] == "pos"]
    negA = sum(r["logit_A"] > r["logit_B"] for r in neg)
    lines = [f"NEG control (identical story as A and B), n={len(neg)}: judge picks A in {negA}/{len(neg)}; "
             f"mean (logit_A-logit_B)={sum(r['logit_A']-r['logit_B'] for r in neg)/len(neg):.3f}; "
             f"mean P(A or B)={sum(r['pAB'] for r in rows)/len(rows):.3f}"]
    wins = 0; ties = 0; losses = 0; bothA = 0
    for i in range(n):
        rf = next(r for r in pos if r["i"] == i and r["order"] == "real_first")
        sf = next(r for r in pos if r["i"] == i and r["order"] == "shuf_first")
        w1 = rf["logit_A"] > rf["logit_B"]          # real wins when shown first
        w2 = sf["logit_B"] > sf["logit_A"]          # real wins when shown second
        if w1 and w2: wins += 1
        elif (not w1) and (not w2): losses += 1
        else: ties += 1
    allpos = len(pos)
    pA = sum(r["logit_A"] > r["logit_B"] for r in pos)
    real_pref = sum((r["logit_A"] > r["logit_B"]) == (r["order"] == "real_first") for r in pos)
    lines.append(f"POS control (real vs word-shuffled), n={n} pairs x 2 orders: real wins both orders {wins}, tie {ties}, shuffled wins both {losses}; "
                 f"single-call real preferred {real_pref}/{allpos}; picks position A {pA}/{allpos}")
    secs = [r["sec"] for r in rows]
    lines.append(f"seconds per call: first call {secs[0]:.1f}, median {sorted(secs)[len(secs)//2]:.1f}, mean {sum(secs)/len(secs):.1f}, n={len(secs)}")
    print("\n".join(lines), flush=True)
    open(f"{ROOT}/controls_summary.txt", "w").write("\n".join(lines) + "\n")


PAIRS = [tuple(p.split(":")) for p in os.environ["JUDGE_PAIRS"].split(",")] if os.environ.get("JUDGE_PAIRS") else \
    [("base", "t08_ng3p8"), ("base", "t06_adj10_mbr8"), ("t08_ng3p8", "t06_adj10_mbr8")]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def sign_p(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n)


def summarize():
    rows = [json.loads(l) for l in open(f"{ROOT}/results.jsonl", encoding="utf-8")]
    out = []
    for x, y in PAIRS:
        name = f"{x}_vs_{y}"
        by = {}
        for r in rows:
            if r["pair"] == name:
                by.setdefault(r["plan_idx"], {})[r["order"]] = r
        w = t = l = inc = 0
        for i, o in by.items():
            if "XY" not in o or "YX" not in o:
                inc += 1; continue
            xwin1 = o["XY"]["logit_A"] > o["XY"]["logit_B"]
            xwin2 = o["YX"]["logit_B"] > o["YX"]["logit_A"]
            if xwin1 and xwin2: w += 1
            elif (not xwin1) and (not xwin2): l += 1
            else: t += 1
        n = w + t + l
        # position-bias-free margin: mean of (A-B logit when X is shown first) and (B-A logit when X is shown second); > 0 favours X
        ds = [(o["XY"]["logit_A"] - o["XY"]["logit_B"] + o["YX"]["logit_B"] - o["YX"]["logit_A"]) / 2 for o in by.values() if "XY" in o and "YX" in o]
        if ds:
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(ds) for _ in ds) / len(ds) for _ in range(2000))
            out.append(f"{name}: margin (positive favours {x}) mean {sum(ds)/len(ds):+.3f}, bootstrap 95% CI {bs[50]:+.3f} to {bs[1949]:+.3f}, {x} ahead in {sum(v > 0 for v in ds)}/{len(ds)} plans")
        lw, hw = wilson(w, n); ll, hl = wilson(l, n)
        out.append(f"{name}: plans judged in both orders n={n} (incomplete {inc}); {x} wins {w} ({w/max(n,1):.3f}, 95% CI {lw:.3f}-{hw:.3f}), "
                   f"tie {t} ({t/max(n,1):.3f}), {y} wins {l} ({l/max(n,1):.3f}, 95% CI {ll:.3f}-{hl:.3f}); two-sided sign test on non-ties p={sign_p(w, l):.4f}")
    nA = sum(r["logit_A"] > r["logit_B"] for r in rows)
    out.append(f"position bias: judge picks A in {nA}/{len(rows)} main-run calls = {nA/max(len(rows),1):.3f}")
    secs = [r["sec"] for r in rows if "sec" in r]
    if secs: out.append(f"seconds per call (forward pass only): median {sorted(secs)[len(secs)//2]:.1f}, mean {sum(secs)/len(secs):.1f}")
    out.append(f"mean P(A)+P(B) over all calls = {sum(r['pAB'] for r in rows)/max(len(rows),1):.3f}")
    try:
        out.append("--- controls ---\n" + open(f"{ROOT}/controls_summary.txt").read().strip())
    except FileNotFoundError:
        pass
    open(f"{ROOT}/summary.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out), flush=True)


def main(nplans):
    d = json.load(open(f"{ROOT}/data.json", encoding="utf-8"))
    path = f"{ROOT}/results.jsonl"
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); done.add((r["pair"], r["plan_idx"], r["order"]))
    j = Judge()
    f = open(path, "a", encoding="utf-8")
    total = nplans * len(PAIRS) * 2
    k = 0
    for i in range(nplans):
        for x, y in PAIRS:
            name = f"{x}_vs_{y}"
            for order, (a, b) in (("XY", (d[x][i], d[y][i])), ("YX", (d[y][i], d[x][i]))):
                k += 1
                if (name, i, order) in done: continue
                r = j.call(d["plans"][i], a, b)
                r.update(pair=name, plan_idx=i, order=order)
                f.write(json.dumps(r) + "\n"); f.flush()
                if k % 6 == 0: print(f"progress {k}/{total} last {r['sec']}s", flush=True)
    summarize()
    open(f"{ROOT}/main.done", "w").write("done\n")


if __name__ == "__main__":
    if sys.argv[1] == "controls": controls(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif sys.argv[1] == "main": main(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif sys.argv[1] == "summary": summarize()
