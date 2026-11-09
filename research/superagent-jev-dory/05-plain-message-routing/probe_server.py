"""Probe llama-server b12012 details the caution scorer depends on (not part of the benchmark)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_benchmark as B  # noqa: E402

srv = B.Server()
try:
    props = json.loads(srv.request("GET", "/props")[1])
    print("props keys:", sorted(props.keys()))
    print("bos_token:", repr(props.get("bos_token")), "eos_token:", repr(props.get("eos_token")))
    tok = lambda c, sp: json.loads(srv.request("POST", "/tokenize", {"content": c, "add_special": sp, "parse_special": True})[1])["tokens"]
    print("tokenize('<|startoftext|>')", tok("<|startoftext|>", False), "tokenize('', True)", tok("", True), "tokenize('Hi', True)", tok("Hi", True))
    md = "You are a decision engine.\n\n### Answer\n"
    ids = [1] + tok(md, False)
    labels = [598, 79, 11595, 53550]
    body = {"prompt": ids, "n_predict": 1, "n_probs": 4, "temperature": 1.0, "post_sampling_probs": True,
            "samplers": ["top_k", "temperature"], "top_k": 4, "logit_bias": [[t, 100.0] for t in labels], "cache_prompt": True}
    out = json.loads(srv.request("POST", "/completion", body)[1])
    print("post-sampling with bias:", json.dumps(out.get("completion_probabilities"), indent=None)[:900])
    body2 = {"prompt": ids, "n_predict": 1, "n_probs": 5, "temperature": 0, "cache_prompt": True}
    out2 = json.loads(srv.request("POST", "/completion", body2)[1])
    print("pre-sampling:", json.dumps(out2.get("completion_probabilities"), indent=None)[:900])
finally:
    srv.close()
cfg = json.loads(Path("/opt/clover-k3/models/LFM3-1.7B/tokenizer_config.json").read_text())
print("LFM3-1.7B HF add_bos_token:", cfg.get("add_bos_token"), "bos_token:", cfg.get("bos_token"))
