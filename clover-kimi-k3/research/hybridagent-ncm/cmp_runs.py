"""Compare runs (default, one-step, blocked, interventions...). Usage: cmp_runs.py name [name ...]. Reads trace-*.json in /root/runs/hybrid."""
import collections, json, re, sys, statistics as st
J = "/root/runs/hybrid/"
for n in sys.argv[1:]:
    d = json.load(open(J + n + ".json")); S, r = d["steps"], d["result"]
    gen, lat = r["generation_steps"], r["latent_steps"]
    lens, cur = [], 0
    for s in S:
        if s["mode_before"] == "LATENT" and s["kind"] == "latent":
            cur += 1
        elif s["mode_before"] == "LATENT" and s["kind"] == "token" and s.get("tok") == d["ids"]["close"]:
            lens.append(cur); cur = 0
    m = re.findall(r"\\boxed\{([^{}]*)\}", d["output"])
    ans = m[-1] if m else ("<answer>" if "<answer>" in d["output"] else None)
    t_gen = sum(s["t_total"] for s in S if s["mode_before"] != "FEED")
    rel = [s["iv_rel"] for s in S if "iv_rel" in s]
    print(f"{n}: gen {gen} text {gen-lat} latent {lat} spans {r['latent_spans']} (anchor {r['anchor_exits']} forced {r['forced_exits']}) "
          f"span median {st.median(lens) if lens else '-'} max {max(lens) if lens else '-'} end {r['end_reason']} wall {d['wall_s']:.0f}s t_gen {t_gen:.0f}s "
          f"py {r.get('python_calls',0)} se {r.get('search_calls',0)} ans {ans!r}" + (f" iv_rel median {st.median(rel):.3f}" if rel else ""))
