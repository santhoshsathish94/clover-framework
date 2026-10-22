"""Read trace1.json and print what was observed (no interpretation beyond the numbers)."""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/root/runs/hybrid/trace1.json"))
S = d["steps"]
close, opn = d["ids"]["close"], d["ids"]["open"]


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else float("nan")


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else float("nan")


print("budget", d["budget"], "wall_s", round(d["wall_s"], 1), "prefill_s", round(d["prefill_s"], 2), "prompt_len", d["prompt_len"])
print("result", d["result"])
print("gate (same seed and settings as the earlier 1,024-step run; instrument must not change the text):", d["gate"])
print("steps", len(S))

kinds = {}
for s in S:
    k = s["kind"]
    if k == "token":
        k = "token:exit(</latent>)" if s["in_id"] == close else ("token:<latent>" if s["in_id"] == opn else "token")
    kinds.setdefault(k, []).append(s)
print("\n== steps by kind of input fed")
for k, v in kinds.items():
    print(f"{k:24s} n={len(v)}")

print("\n== stage times in ms (mean over steps of that kind; net = total minus my own observation work)")
print(f"{'kind':24s} {'total':>7s} {'net':>7s} {'forward':>8s} {'layers':>7s} {'norm':>6s} {'head':>6s} {'link':>6s} {'sample':>7s} {'other':>7s}")
for k, v in kinds.items():
    tot = mean([s["t_total"] for s in v]) * 1e3
    obs = mean([s["obs"] for s in v]) * 1e3
    net = tot - obs
    fw = mean([s["t_fwd"] for s in v]) * 1e3
    ly = mean([s["t_layers"] for s in v]) * 1e3
    nm = mean([s["t_norm"] for s in v]) * 1e3
    hd = mean([s["t_head"] for s in v]) * 1e3
    lk = mean([s.get("t_link", 0.0) for s in v]) * 1e3
    sm = mean([s.get("t_sample", 0.0) for s in v]) * 1e3
    other = net - fw - lk - sm
    print(f"{k:24s} {tot:7.1f} {net:7.1f} {fw:8.1f} {ly:7.1f} {nm:6.2f} {hd:6.1f} {lk:6.2f} {sm:7.2f} {other:7.1f}")
print("median total ms per kind:", {k: round(med([s['t_total'] for s in v]) * 1e3, 1) for k, v in kinds.items()})

print("\n== per-layer profile at the last position: mean output norm / relative update ||out-in||/||in|| / cos(in,out)")
tx = kinds.get("token", [])
lt = kinds.get("latent", [])
print(f"text-step n={len(tx)}, latent-step n={len(lt)}")
NL = len(S[0]["layers"])
print(f"{'layer':>5s} | {'T norm':>8s} {'T upd':>7s} {'T cos':>6s} | {'L norm':>8s} {'L upd':>7s} {'L cos':>6s}")
for i in sorted(set([0, 1, 2, 3, 5, 8, 12, 16, 20, 24, 28, 30, 32, 33, 34, 35]) & set(range(NL))):
    def m(v, j):
        return mean([s["layers"][i][j] for s in v])
    print(f"{i:5d} | {m(tx,0):8.1f} {m(tx,1):7.3f} {m(tx,2):6.3f} | {m(lt,0):8.1f} {m(lt,1):7.3f} {m(lt,2):6.3f}")

print("\n== norms")
print("fed vector norm: token steps mean", round(mean([s['emb_norm'] for s in tx]), 3), "| latent steps (R(h)) mean", round(mean([s['emb_norm'] for s in lt]), 3))
print("final normed hidden norm (what the head and HybridLink read): token steps", round(mean([s['out_norm'] for s in tx]), 2), "| latent steps", round(mean([s['out_norm'] for s in lt]), 2))
print("max |last_hidden (engine, hidden_states[-1]) - output of final norm| over all steps:", max(s["hid_vs_normed_maxabs"] for s in S))

print("\n== latent steps: readouts of the same hidden state two ways")
ls = [s for s in S if s["mode_before"] == "LATENT"]
print("latent-mode decode steps (state read before the step):", len(ls))
ex = [s for s in ls if s.get("tok") == close]
non = [s for s in ls if s.get("tok") != close]
print("steps that ended the span (engine fed </latent>):", len(ex), "| steps that continued:", len(non))
print("cosine of R(h) to its nearest token embedding: continued mean", round(mean([s['link_nn_cos'] for s in non]), 3), "min", round(min([s['link_nn_cos'] for s in non]), 3) if non else None, "max", round(max([s['link_nn_cos'] for s in non]), 3) if non else None)
print("cosine of R(h) to Emb(</latent>): continued mean", round(mean([s['link_cos_close'] for s in non]), 3), "| at exit", [round(s['link_cos_close'], 3) for s in ex])
print("LM-head P(</latent>) at the step: continued mean", round(mean([s['lm_p_close'] for s in non]), 4), "max", round(max([s['lm_p_close'] for s in non]), 4) if non else None, "| at exit", [round(s['lm_p_close'], 3) for s in ex])
agree = {"lm_top1==close": 0, "nn==close": 0, "both": 0, "neither": 0}
for s in ls:
    a = s["lm_top1"] == close
    b = s["link_nn_id"] == close
    agree["both" if a and b else ("lm_top1==close" if a else ("nn==close" if b else "neither"))] += 1
print("per latent step, which readout says </latent> (before the minimum-steps rule):", agree)
print("LM-head top-1 probability in latent steps mean", round(mean([s['lm_p1'] for s in ls]), 3), "| entropy (plain softmax) mean", round(mean([s['lm_entropy'] for s in ls]), 3))
print("sampling-distribution entropy in text steps (temperature, top-k, top-p applied) mean", round(mean([s.get('samp_entropy') for s in tx]), 3), "| P(sampled token) mean", round(mean([s.get('samp_p_tok') for s in tx]), 3))
op = [s for s in S if s.get("tok") == opn and s["mode_before"] == "DECODE"]
print("P(<latent>) at the steps where <latent> was sampled:", [round(s['samp_p_open'], 3) for s in op], "| mean P(<latent>) at all text steps", round(mean([s.get('samp_p_open') for s in tx]), 4))

print("\n== spans")
spans = []
cur = None
for s in S:
    if s["mode_before"] == "DECODE" and s.get("tok") == opn:
        cur = {"len": 0}
    if s["mode_before"] == "LATENT" and cur is not None:
        if s.get("tok") == close:
            spans.append(cur)
            cur = None
        else:
            cur["len"] += 1
if cur is not None:
    cur["open"] = True
    spans.append(cur)
print("span lengths (latent steps):", [x["len"] for x in spans], "| open at the end:", [bool(x.get("open")) for x in spans].count(True))
