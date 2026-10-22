"""Step trace of HybridAgent on one question (observation only, nothing in the repository is changed).

Hooks and wrappers read values; they do not alter the computation. What is recorded for every generation step
(row 0, batch 1): the kind of input fed (token or latent vector), times of the stages (forward, the 36 layers,
final norm, LM head, HybridLink, sampling), per-layer output norm / relative update / cosine to the layer input
at the last position, the norms of the fed vector, and for latent steps the LM-head readout and the cosine
readout of R(h). Settings equal those of hybrid_inference.run_batch (search tool off, as in the earlier run).
usage: python trace_hybrid.py BUDGET OUT.json
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from hybrid_agent import HybridGenerationEngine, SYSTEM_PROMPT, build_tool_executor, load_hybrid_agent
import hybrid_agent.engine as E
from hybrid_agent.readout import normalized_embeddings

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 450
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/runs/hybrid/trace1.json"
QID = sys.argv[3] if len(sys.argv) > 3 else "q4"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OLD = "/root/runs/hybrid/q4.jsonl" if QID == "q4" and SEED == 0 else None
QUESTION = next(json.loads(l)["question"] for l in open("/root/hybridagent/examples/questions.jsonl") if json.loads(l)["id"] == QID)

model, tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", device=torch.device("cpu"))
bb = model.backbone
layers = bb.model.layers
NL = len(layers)
bank = normalized_embeddings(bb.get_input_embeddings().weight)  # [V, D] unit rows, fp32

H = {"rec": False, "in": [None] * NL, "lay": [None] * NL}


def mk_pre(i):
    def pre(mod, args, kwargs):
        if not H["rec"]:
            return
        hs = args[0] if args else kwargs["hidden_states"]
        if i == 0:
            H["t_l0"] = time.perf_counter()
        H["in"][i] = hs[0, -1].detach().float()
    return pre


def mk_post(i):
    def post(mod, args, kwargs, output):
        if not H["rec"]:
            return
        out = output[0] if isinstance(output, tuple) else output
        xo = out[0, -1].detach().float()
        xi = H["in"][i]
        H["lay"][i] = (float(xo.norm()), float((xo - xi).norm() / xi.norm()), float(F.cosine_similarity(xo, xi, dim=0)))
        if i == NL - 1:
            H["t_lN"] = time.perf_counter()
    return post


for i, layer in enumerate(layers):
    layer.register_forward_pre_hook(mk_pre(i), with_kwargs=True)
    layer.register_forward_hook(mk_post(i), with_kwargs=True)


def norm_hook(mod, args, output):
    if H["rec"]:
        H["normed"] = output[0, -1].detach().float()
        H["t_norm1"] = time.perf_counter()


def head_pre(mod, args):
    if H["rec"]:
        H["t_head0"] = time.perf_counter()


def head_post(mod, args, output):
    if H["rec"]:
        H["t_head1"] = time.perf_counter()


bb.model.norm.register_forward_hook(norm_hook)
bb.lm_head.register_forward_pre_hook(head_pre)
bb.lm_head.register_forward_hook(head_post)


class Tr(HybridGenerationEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.steps = []
        self.cur = None
        self.prefill_s = None

    def _init_generation(self):
        t = time.perf_counter()
        ok = super()._init_generation()
        self.prefill_s = time.perf_counter() - t
        if CACHECHECK:
            CC["prompt"] = list(self.rows[0].kv_ledger)
            CC["h0"] = self.last_hidden[0].detach().float().clone()
        return ok

    def _decode_step(self, rng):
        r = self.rows[0]
        c = {"mode_before": r.mode, "obs": 0.0}
        self.cur = c
        n0 = len(r.generated_ids)
        if r.mode == E.LATENT:
            t = time.perf_counter()
            p = torch.softmax(self.next_logits[0].float(), -1)
            top = torch.topk(p, 2)
            c["lm_top1"] = int(top.indices[0])
            c["lm_p1"] = float(top.values[0])
            c["lm_p2"] = float(top.values[1])
            c["lm_p_close"] = float(p[self.latent_close_id])
            c["lm_entropy"] = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
            c["h_norm"] = float(self.last_hidden[0].float().norm())
            c["obs"] += time.perf_counter() - t
        H["rec"] = True
        t0 = time.perf_counter()
        super()._decode_step(rng)
        c["t_total"] = time.perf_counter() - t0
        H["rec"] = False
        c["mode_after"] = r.mode
        if len(r.generated_ids) > n0:
            c["tok"] = int(r.generated_ids[-1])
        self.steps.append(c)

    def _lockstep_forward(self, inputs_embeds, real_rows):
        c = self.cur
        r = self.rows[0]
        led = r.kv_ledger[-1] if r.kv_ledger else None
        c["kind"] = "pad" if not real_rows else ("latent" if led is None else "token")
        c["in_id"] = led
        c["emb_norm"] = float(inputs_embeds[0, 0].float().norm())
        H["lay"] = [None] * NL
        t = time.perf_counter()
        super()._lockstep_forward(inputs_embeds, real_rows)
        c["t_fwd"] = time.perf_counter() - t
        c["t_layers"] = H["t_lN"] - H["t_l0"]
        c["t_norm"] = H["t_norm1"] - H["t_lN"]
        c["t_head"] = H["t_head1"] - H["t_head0"]
        c["layers"] = [[round(v, 5) for v in x] for x in H["lay"]]
        c["out_norm"] = float(H["normed"].norm())
        c["hid_vs_normed_maxabs"] = float((self.last_hidden[0].float() - H["normed"]).abs().max())
        if CACHECHECK and real_rows:
            CC["steps"].append((inputs_embeds[0, 0].detach().float().clone(), self.last_hidden[0].detach().float().clone(), c["kind"]))


import os
CACHECHECK = os.environ.get("CACHECHECK", "0") == "1"
CC = {"steps": []}
INTERV = os.environ.get("INTERV", "")  # noise:<d> | random | nearest ; applied to every latent input after it is read out
ivg = torch.Generator().manual_seed(1234 + int(sys.argv[4]) if len(sys.argv) > 4 else 1234)
SEARCH = os.environ.get("SEARCH", "0") == "1"
tools = build_tool_executor(120, 10, 500, require_search=SEARCH)
eng = Tr(
    model, tok, torch.device("cpu"), tools, SYSTEM_PROMPT, QUESTION,
    batch_size=1, max_response_tokens=BUDGET, count_tool_tokens=False,
    max_tool_calls=5, max_search_calls=5 if SEARCH else 0, max_python_calls=5, terminate_on_tool_cap=True,
    max_turns=30, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1,
    min_latent_steps=int(os.environ.get("MINLAT", "4")), max_latent_steps=int(os.environ.get("MAXLAT", "200")), record_latent_readout=False,
)

orig_link = model.run_hybridlink
LATENT = os.environ.get("LATENT", "1") == "1"
if not LATENT:
    eng._blocked_masks["in"][eng.latent_open_id] = True


def link(hs, output_dtype=None):
    c = eng.cur
    t = time.perf_counter()
    out = orig_link(hs, output_dtype=output_dtype)
    c["t_link"] = time.perf_counter() - t
    t = time.perf_counter()
    z = out[0].float()
    h = hs[0].float()
    c["link_out_norm"] = float(z.norm())
    c["link_cos_in"] = float(F.cosine_similarity(z, h, dim=0))
    sims = bank @ F.normalize(z, dim=0)
    best = int(sims.argmax())
    c["link_nn_id"] = best
    c["link_nn_cos"] = float(sims[best])
    c["link_cos_close"] = float(sims[eng.latent_close_id])
    if INTERV:
        nz = z.norm()
        if INTERV.startswith("noise:"):
            g = torch.randn(z.shape, generator=ivg)
            new = z + float(INTERV[6:]) * nz * g / g.norm()
        elif INTERV == "random":
            new = bank[int(torch.randint(0, bank.shape[0], (1,), generator=ivg))] * nz
        elif INTERV == "nearest":
            new = bank[best] * nz
        else:
            raise SystemExit("unknown INTERV " + INTERV)
        c["iv_rel"] = float((new - z).norm() / nz)
        out = out.clone()
        out[0] = new.to(out.dtype)
    c["obs"] += time.perf_counter() - t
    return out


model.run_hybridlink = link

orig_tlp, orig_sr = E.token_log_probs, E.sample_rows


def tlp(logits, temperature, top_p, top_k, blocked):
    t = time.perf_counter()
    lp = orig_tlp(logits, temperature, top_p, top_k, blocked)
    eng.cur["t_sample"] = eng.cur.get("t_sample", 0.0) + time.perf_counter() - t
    return lp


def sr(log_probs, rows, rng):
    c = eng.cur
    t = time.perf_counter()
    out = orig_sr(log_probs, rows, rng)
    c["t_sample"] = c.get("t_sample", 0.0) + time.perf_counter() - t
    if 0 in out:
        t = time.perf_counter()
        lp = log_probs[0]
        p = lp.exp()
        c["samp_p_tok"] = float(p[out[0]])
        c["samp_entropy"] = float(-(torch.where(p > 0, p * lp, torch.zeros_like(p))).sum())
        c["samp_p_open"] = float(p[eng.latent_open_id])
        c["obs"] += time.perf_counter() - t
    return out


E.token_log_probs, E.sample_rows = tlp, sr

rng = torch.Generator(device="cpu")
rng.manual_seed(SEED)
t0 = time.time()
eng.run(rng)
wall = time.time() - t0
res = eng.row_result(eng.rows[0])

gate = {}
try:
    if OLD is None:
        raise RuntimeError("no earlier record for " + QID)
    old = json.loads(open(OLD).read())["output"]
    new = res["output"]
    n = min(len(old), len(new))
    d = next((i for i in range(n) if old[i] != new[i]), None)
    gate = {"old_len": len(old), "new_len": len(new), "first_diff": d, "new_is_prefix_of_old": old.startswith(new)}
except Exception as e:  # the earlier record may be missing
    gate = {"error": str(e)}

json.dump({
    "budget": BUDGET, "wall_s": wall, "prefill_s": eng.prefill_s, "prompt_len": eng.prompt_len,
    "result": {k: v for k, v in res.items() if k != "output"}, "output": res["output"], "gate": gate,
    "ids": {"open": eng.latent_open_id, "close": eng.latent_close_id},
    "steps": eng.steps,
}, open(OUT, "w"))
print("DONE wall", round(wall, 1), "steps", len(eng.steps), "prefill", round(eng.prefill_s, 2), "gate", gate)

if CACHECHECK:
    with torch.no_grad():
        P = len(CC["prompt"])
        pe = bb.get_input_embeddings()(torch.tensor([CC["prompt"]]))
        embs = torch.stack([e for e, _, _ in CC["steps"]]).unsqueeze(0).to(pe.dtype)
        t = time.time()
        full = bb.model(inputs_embeds=torch.cat([pe, embs], 1), use_cache=False)
        hs = full.last_hidden_state[0].float()
        ref = torch.cat([CC["h0"].unsqueeze(0), torch.stack([h for _, h, _ in CC["steps"]])], 0)
        got = hs[P - 1:]
        n = ref.shape[0]
        maxabs = (got - ref).abs().max(dim=1).values
        cos = F.cosine_similarity(got, ref, dim=1)
        rel = (got - ref).norm(dim=1) / ref.norm(dim=1)
        kinds = ["prompt-end"] + [k for _, _, k in CC["steps"]]
        am_same = 0
        for a in range(0, n, 128):
            la = bb.lm_head(got[a:a + 128].to(pe.dtype)).float().argmax(-1)
            lb = bb.lm_head(ref[a:a + 128].to(pe.dtype)).float().argmax(-1)
            am_same += int((la == lb).sum())
        stats = {"prompt_len": P, "real_steps": n - 1, "full_forward_s": time.time() - t, "dtype": str(pe.dtype),
                 "max_abs_over_positions": float(maxabs.max()), "median_abs": float(maxabs.median()),
                 "min_cos": float(cos.min()), "max_rel": float(rel.max()), "median_rel": float(rel.median()),
                 "lm_head_argmax_same": am_same, "positions": n,
                 "by_kind": {k: {"n": sum(1 for x in kinds if x == k), "max_rel": float(max([float(rel[i]) for i, x in enumerate(kinds) if x == k] or [0]))} for k in set(kinds)}}
        # noise floor: the same inputs through a float32 copy of the model; both bf16 results are compared with it
        bb.float()
        t = time.time()
        h32 = bb.model(inputs_embeds=torch.cat([pe, embs], 1).float(), use_cache=False).last_hidden_state[0].float()[P - 1:]
        def am(h):
            r = []
            for a in range(0, n, 128):
                r.append(bb.lm_head(h[a:a + 128]).argmax(-1))
            return torch.cat(r)
        a32 = am(h32)
        rel_c = (ref - h32).norm(dim=1) / h32.norm(dim=1)
        rel_f = (got - h32).norm(dim=1) / h32.norm(dim=1)
        stats["fp32"] = {"forward_s": time.time() - t,
                         "cached_vs_fp32": {"median_rel": float(rel_c.median()), "max_rel": float(rel_c.max()), "argmax_same": int((am(ref) == a32).sum())},
                         "fullprefix_vs_fp32": {"median_rel": float(rel_f.median()), "max_rel": float(rel_f.max()), "argmax_same": int((am(got) == a32).sum())}}
        json.dump(stats, open(OUT + ".cache.json", "w"))
        print("CACHECHECK", json.dumps(stats))
