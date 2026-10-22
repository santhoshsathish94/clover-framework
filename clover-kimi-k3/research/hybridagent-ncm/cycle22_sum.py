import json, sys
J = "/root/runs/hybrid/"
groups = [  # label, b16, default, blocked
 ("q4 s0","trace-q4-b16-s0","trace-q4-search","trace-q4-nolat-search"),
 ("q4 s1","trace-q4-b16-s1","trace-q4-s1","trace-q4-nolat-s1"),
 ("q4 s2","trace-q4-b16-s2","trace-q4-s2","trace-q4-nolat-s2"),
 ("q5 s0","trace-q5-b16-s0","trace-q5-search","trace-q5-nolat"),
 ("q5 s1","trace-q5-b16-s1","trace-q5-s1","trace-q5-nolat-s1"),
 ("q5 s2","trace-q5-b16-s2","trace-q5-s2","trace-q5-nolat-s2"),
 ("q1 s0","trace-q1-b16-s0","trace-q1-search","trace-q1-nolat-search"),
 ("q1 s1","trace-q1-b16-s1",None,None),
 ("q1 s2","trace-q1-b16-s2",None,None),
]
import re
def L(n):
    d = json.load(open(J+n+".json")); r = d["result"]
    m = re.findall(r"\\boxed\{([^{}]*)\}", d["output"])
    return dict(gen=r["generation_steps"], lat=r["latent_steps"], text=r["generation_steps"]-r["latent_steps"], sp=r["latent_spans"], fo=r["forced_exits"], end=r["end_reason"], ans=m[-1] if m else None)
tot = {"b16":[0,0], "def":[0,0], "blk":[0,0]}
tot_f = {"b16":[0,0], "blk":[0,0]}
for lab,a,b,c in groups:
    A = L(a); s = f"{lab}: b16 gen {A['gen']} text {A['text']} lat {A['lat']} spans {A['sp']} forced {A['fo']} end {A['end']} ans {A['ans']!r}"
    if b:
        B, C = L(b), L(c)
        s += f" | default gen {B['gen']} text {B['text']} end {B['end']} | blocked gen {C['gen']} text {C['text']} end {C['end']} ans {C['ans']!r}"
        for k,X in (("b16",A),("def",B),("blk",C)):
            tot[k][0] += X["gen"]; tot[k][1] += X["text"]
        if C["end"] == "eos" and A["end"] == "eos":
            for k,X in (("b16",A),("blk",C)):
                tot_f[k][0] += X["gen"]; tot_f[k][1] += X["text"]
    print(s)
print("matched 7 (gen, text):", tot)
print("both-finished (gen, text):", tot_f)
