"""One-line-per-question summary of trace-<id>.json files, plus readout agreement and layer-35 medians."""
import json
import re
import statistics as st
import sys

exp = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("/root/hybridagent/examples/questions.jsonl")}
for qid in sys.argv[1:]:
    if ".json" in qid:
        p = qid
        _m = re.search(r"(q\d)", p.split("/")[-1]); qkey = _m.group(1) if _m else "q4"
        qid = qkey + " [" + p.split("/")[-1] + "]"
    else:
        qkey = qid
        p = "/root/runs/hybrid/trace-%s.json" % qid if qid != "q4" else "/root/runs/hybrid/trace2.json"
    d = json.load(open(p))
    S, r = d["steps"], d["result"]
    close, opn = d["ids"]["close"], d["ids"]["open"]
    tx = [s for s in S if s["kind"] == "token" and s["in_id"] not in (close, opn)]
    lt = [s for s in S if s["kind"] == "latent"]
    net = lambda v: round(st.mean([(s["t_total"] - s["obs"]) * 1e3 for s in v]), 1) if v else None
    ls = [s for s in S if s["mode_before"] == "LATENT"]
    only_nn = [s for s in ls if s["link_nn_id"] == close and s["lm_top1"] != close]
    only_lm = [s for s in ls if s["lm_top1"] == close and s["link_nn_id"] != close]
    exits = [s for s in ls if s.get("tok") == close]
    out = d["output"]
    ans = re.findall(r"<answer>(.*?)</answer>", out, re.S)
    tools = r.get("python_calls", 0)
    spans = []
    cur = None
    for s in S:
        if s["mode_before"] == "DECODE" and s.get("tok") == opn:
            cur = 0
        elif s["mode_before"] == "LATENT" and cur is not None:
            if s.get("tok") == close:
                spans.append(cur)
                cur = None
            else:
                cur += 1
    print(f"== {qid} (expected {exp[qkey]!r})")
    print(f"  steps {len(S)}, latent steps {r['latent_steps']} ({100 * r['latent_steps'] / len(S):.0f}% of steps), spans {r['latent_spans']} {spans}, python calls {tools}, end {r['end_reason']}, wall {d['wall_s']:.0f}s, prompt_len {d['prompt_len']}")
    print(f"  answer block: {[a.strip()[:140] for a in ans]}")
    print(f"  net step ms: text {net(tx)}, latent {net(lt)} | exits {len(exits)} (min P(close) {min([s['lm_p_close'] for s in exits], default=None)}), cosine-only close {len(only_nn)}, LM-head-only close {len(only_lm)}, max P(close) on continued {round(max([s['lm_p_close'] for s in ls if s.get('tok') != close], default=0), 3)}")
    if tx and lt:
        print(f"  layer-35 out norm median: text {st.median([s['layers'][35][0] for s in tx]):.0f}, latent {st.median([s['layers'][35][0] for s in lt]):.0f}; fed norm text {st.mean([s['emb_norm'] for s in tx]):.2f}, latent {st.mean([s['emb_norm'] for s in lt]):.2f}; hid-vs-normed max {max(s['hid_vs_normed_maxabs'] for s in S)}")
    big = [s for s in tx if s['layers'][16][1] > 0.9]
    print(f"  layer-16 text steps with rel update >0.9: {len(big)} of {len(tx)}; ids {[s['in_id'] for s in big][:5]}")
