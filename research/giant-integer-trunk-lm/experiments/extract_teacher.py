#!/usr/bin/env python3
"""Extract & cache LFM3-Omni-2B teacher signals (identical corpus/tokenization to distill_omni_to_block_trunklm.py).

Adds full-vocab teacher log-sum-exp + teacher target NLL so that (a) teacher PPL can be reproduced and
(b) a tail-aware Top-32+tail KL can be evaluated exactly.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

hf_dir = Path("/opt/clover-k3/lfm3-omni-2b-hf")
repo_dir = Path("/opt/clover-k3/clover-framework")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/xc_variant_d/teacher_cache.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
tok = Tokenizer.from_file(str(hf_dir / "tokenizer.json"))
tok_cfg = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
bos_id = tok.token_to_id(tok_cfg.get("bos_token") or "<|startoftext|>") or 1

corpus_texts = []
for pat in ("AGENTS.md", "docs/**/*.md", "clover-ai-system/studio/clover/*.py"):
    for p in sorted(repo_dir.glob(pat)):
        if p.is_file():
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
                if len(t) > 400:
                    corpus_texts.append(t[:12000])
            except OSError:
                pass
        if sum(len(x) for x in corpus_texts) > 350_000:
            break
all_ids = tok.encode("\n\n".join(corpus_texts)).ids
seq_len, n_chunks = 128, int(sys.argv[1]) if len(sys.argv) > 1 else 160
raw = np.array(all_ids[:n_chunks * seq_len], dtype=np.int64).reshape(n_chunks, seq_len)
full = np.concatenate([np.full((n_chunks, 1), bos_id, dtype=np.int64), raw], axis=1)
input_ids = torch.from_numpy(full[:, :-1]).contiguous()
target_ids = torch.from_numpy(full[:, 1:]).contiguous()
print("corpus docs", len(corpus_texts), "tokens", len(all_ids), "bos", bos_id)

pkg_name = "lfm3_omni_pkg"
spec_pkg = importlib.util.spec_from_file_location(pkg_name, hf_dir / "__init__.py", submodule_search_locations=[str(hf_dir)])
pkg = importlib.util.module_from_spec(spec_pkg)
sys.modules[pkg_name] = pkg
for m_name in ("configuration_lfm3", "lfm3_conv", "modeling_lfm3"):
    sp = importlib.util.spec_from_file_location(f"{pkg_name}.{m_name}", hf_dir / f"{m_name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[f"{pkg_name}.{m_name}"] = mod
    sp.loader.exec_module(mod)
raw_cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
lm_cfg = sys.modules[f"{pkg_name}.configuration_lfm3"].Lfm3Config(**(raw_cfg.get("language_config") or raw_cfg.get("text_config")))
teacher = sys.modules[f"{pkg_name}.modeling_lfm3"].Lfm3ForCausalLM(lm_cfg).to(torch.bfloat16)
sd = {}
with safe_open(str(hf_dir / "model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith("model.language_model."):
            sd["model." + k[len("model.language_model."):]] = f.get_tensor(k)
sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
missing, unexpected = teacher.load_state_dict(sd, strict=False)
print("missing", len(missing), missing[:5], "unexpected", len(unexpected))
teacher.eval()

ids_l, val_l, h12_l, h24_l, lse_l, nll_l = [], [], [], [], [], []
t0 = time.time()
with torch.no_grad():
    for i in range(0, n_chunks, 8):
        o = teacher(input_ids=input_ids[i:i + 8], output_hidden_states=True, use_cache=False)
        lg = o.logits.float()
        v, ix = torch.topk(lg, k=64, dim=-1)
        lse = torch.logsumexp(lg, dim=-1)
        nll = lse - torch.gather(lg, -1, target_ids[i:i + 8].unsqueeze(-1)).squeeze(-1)
        ids_l.append(ix.to(torch.int32)); val_l.append(v.to(torch.float16))
        h12_l.append(o.hidden_states[12].to(torch.float16)); h24_l.append(o.hidden_states[-1].to(torch.float16))
        lse_l.append(lse); nll_l.append(nll)
print("teacher fwd s", time.time() - t0)
cache = dict(input_ids=input_ids, target_ids=target_ids, top_ids=torch.cat(ids_l), top_vals=torch.cat(val_l),
             h12=torch.cat(h12_l), h24=torch.cat(h24_l), lse=torch.cat(lse_l), nll=torch.cat(nll_l),
             embed=sd["model.embed_tokens.weight"].clone())
nll = cache["nll"]
top_vals_f = cache["top_vals"].float()
mass128 = None
print("teacher val PPL (chunks 128..159, full 128K vocab):", float(torch.exp(nll[128:160].mean())))
if n_chunks > 160:
    print("teacher fresh-test PPL (chunks 160..):", float(torch.exp(nll[160:].mean())))
print("teacher train PPL:", float(torch.exp(nll[:128].mean())))
print("teacher top-32 mass:", float(torch.exp(top_vals_f[:, :, :32] - cache["lse"].unsqueeze(-1)).sum(-1).mean()),
      " top-64 mass:", float(torch.exp(top_vals_f - cache["lse"].unsqueeze(-1)).sum(-1).mean()))
torch.save(cache, out)
print("saved", out)
