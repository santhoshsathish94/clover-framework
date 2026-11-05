"""Checks the laptop AVX2 build of the LFM3 C engine against the current (scalar) build, on this laptop.

1. Full forward, old vs new DLL, same tokens: final hidden differences, next-token agreement (tied head), time.
2. New DLL cache path: one-shot prefill and token-by-token feeding must equal the new full forward exactly.
3. Prefill and decode speed of the cache path (greedy decode from the engine's logits).
Usage: python tests/engine_avx2_check.py <out.jsonl>
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("CLOVER_BASE", Path.home() / "CloverAI"))
ENGINE = BASE / "engine"
PRUNED = BASE / "data" / "lfm3" / "pruned"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "verify" / "engine_avx2_check.jsonl")
D, V = 2048, 128000


def log(rec):
    def conv(o):
        return o.item() if hasattr(o, "item") else str(o)
    line = json.dumps(rec, default=conv)
    print(line[:2000], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load(name):
    os.add_dll_directory(str(ENGINE))
    lib = ctypes.CDLL(str(ENGINE / name))
    lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
    lib.lfm3_forward_24_layers.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.lfm3_forward_24_layers.restype = ctypes.c_float
    rc = lib.lfm3_engine_init(str(PRUNED).encode("utf-8"))
    assert rc == 0, (name, rc)
    return lib


def forward(lib, ids):
    T = len(ids)
    out = np.zeros((T, D), dtype=np.float32)
    c_ids = (ctypes.c_int * T)(*ids)
    t0 = time.perf_counter()
    lib.lfm3_forward_24_layers(c_ids, T, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out, time.perf_counter() - t0


EMB = None


def head_logits(h):
    """Tied head in fp32 from the bf16 embedding file (reference for argmax checks)."""
    global EMB
    if EMB is None:
        EMB = np.memmap(PRUNED / "global-0" / "embed_tokens.bf16.bin", dtype=np.uint16, mode="r", shape=(V, D))
    hb = (np.frombuffer(np.float32(h).tobytes(), dtype=np.uint32) >> 16).astype(np.uint16)  # truncate to bf16 bits
    hf = (hb.astype(np.uint32) << 16).view(np.float32)
    logits = np.empty(V, dtype=np.float32)
    for s in range(0, V, 16000):
        w = (EMB[s:s + 16000].astype(np.uint32) << 16).view(np.float32)
        logits[s:s + 16000] = w @ hf
    return logits


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE / "data" / "lfm3" / "hf"), trust_remote_code=True)
    prompts = {
        "short": "The capital of France is",
        "medium": "Explain in two sentences why the sky looks blue during the day and red at sunset, for a ten year old.",
        "long": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number using an iterative loop.\"\"\"\n" * 6)
                + "# Now write a function that checks whether a string is a palindrome, ignoring case and spaces.\n",
    }
    old = load("liblfm3_cpu_engine.dll")
    new = load("liblfm3_cpu_engine_avx2.dll")
    new.lfm3_get_num_threads.restype = ctypes.c_int
    new.lfm3_has_avx2.restype = ctypes.c_int
    log({"check": "new dll", "avx2": new.lfm3_has_avx2(), "threads": new.lfm3_get_num_threads()})

    full_new = {}
    for name, text in prompts.items():
        ids = tok(text, add_special_tokens=False).input_ids
        h_old, t_old = forward(old, ids)
        h_new, t_new = forward(new, ids)
        full_new[name] = (ids, h_new)
        diff = np.abs(h_old - h_new)
        lo, ln = head_logits(h_old[-1]), head_logits(h_new[-1])
        top5_o, top5_n = list(np.argsort(-lo)[:5]), list(np.argsort(-ln)[:5])
        cos = float(np.dot(h_old[-1], h_new[-1]) / (np.linalg.norm(h_old[-1]) * np.linalg.norm(h_new[-1])))
        log({"check": "full forward old vs new", "prompt": name, "tokens": len(ids), "old_s": round(t_old, 3),
             "new_s": round(t_new, 3), "speedup": round(t_old / max(t_new, 1e-9), 1),
             "equal_fraction": float(np.mean(h_old == h_new)), "max_abs_diff": float(diff.max()),
             "mean_abs_diff": float(diff.mean()), "last_cos": round(cos, 6), "next_token_same": top5_o[0] == top5_n[0],
             "top5_overlap": len(set(top5_o) & set(top5_n)), "next_old": tok.decode([int(top5_o[0])]),
             "next_new": tok.decode([int(top5_n[0])])})

    new.lfm3_cache_init.argtypes = [ctypes.c_int]
    new.lfm3_cache_feed.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    new.lfm3_cache_truncate.argtypes = [ctypes.c_int]
    assert new.lfm3_cache_init(4096) == 0
    hid = np.zeros(D, dtype=np.float32)
    lg = np.zeros(V, dtype=np.float32)
    P = ctypes.POINTER(ctypes.c_float)

    def feed(ids):
        arr = (ctypes.c_int * len(ids))(*ids)
        rc = new.lfm3_cache_feed(arr, len(ids), hid.ctypes.data_as(P), lg.ctypes.data_as(P))
        assert rc > 0, rc
        return rc

    for name, (ids, h_full) in full_new.items():
        new.lfm3_cache_reset()
        feed(ids)
        same_once = bool(np.array_equal(hid, h_full[-1]))
        lg_once = lg.copy()
        new.lfm3_cache_reset()
        for t in ids:
            feed([t])
        same_steps = bool(np.array_equal(hid, h_full[-1]))
        new.lfm3_cache_truncate(len(ids) // 2)
        feed(ids[len(ids) // 2:])
        same_rollback = bool(np.array_equal(hid, h_full[-1]))
        ref = head_logits(h_full[-1])
        log({"check": "cache path == full forward", "prompt": name, "prefill_once": same_once, "token_by_token": same_steps,
             "after_rollback": same_rollback, "logits_argmax_same_as_numpy_head": int(np.argmax(lg_once)) == int(np.argmax(ref)),
             "max_logit_diff": float(np.abs(lg_once - ref).max())})

    # speed: a ~600-token prompt, then greedy decode of 48 tokens
    text = (prompts["medium"] + " ") * 24
    ids = tok(text, add_special_tokens=False).input_ids[:600]
    new.lfm3_cache_reset()
    t0 = time.perf_counter()
    feed(ids)
    t_pre = time.perf_counter() - t0
    gen = []
    t0 = time.perf_counter()
    for _ in range(48):
        nxt = int(np.argmax(lg))
        gen.append(nxt)
        feed([nxt])
    t_dec = time.perf_counter() - t0
    log({"check": "cache speed", "prefill_tokens": len(ids), "prefill_s": round(t_pre, 2),
         "prefill_tok_per_s": round(len(ids) / t_pre, 1), "decode_tokens": 48, "decode_s": round(t_dec, 2),
         "decode_tok_per_s": round(48 / t_dec, 2), "sample": tok.decode(gen)[:200]})
    log({"check": "done"})


if __name__ == "__main__":
    main()
