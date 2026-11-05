"""Thread scaling of the laptop AVX2 LFM3 engine: cached prefill (128 tokens) and decode (16 tokens) per thread count.
Usage: python tests/engine_threads.py <out.jsonl>"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path.home() / "CloverAI"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "verify" / "engine_threads.jsonl")
THROTTLE_OFF = "--throttle-off" in sys.argv
if THROTTLE_OFF:
    sys.path.insert(0, str(BASE / "clover"))
    import winperf
    print("power throttling off:", winperf.disable_power_throttling(), flush=True)
os.add_dll_directory(str(BASE / "engine"))
lib = ctypes.CDLL(str(BASE / "engine" / "liblfm3_cpu_engine.dll"))
lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
assert lib.lfm3_engine_init(str(BASE / "data" / "lfm3" / "pruned").encode()) == 0
PF = ctypes.POINTER(ctypes.c_float)
lib.lfm3_cache_feed.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, PF, PF]
lib.lfm3_set_num_threads.argtypes = [ctypes.c_int]
assert lib.lfm3_cache_init(4096) == 0
hid = np.zeros(2048, np.float32)
lg = np.zeros(128000, np.float32)
ids = [int(x) for x in np.random.default_rng(0).integers(1000, 60000, 128)]


def feed(seq):
    arr = (ctypes.c_int * len(seq))(*seq)
    assert lib.lfm3_cache_feed(arr, len(seq), hid.ctypes.data_as(PF), lg.ctypes.data_as(PF)) > 0


feed(ids[:8])  # warm the mapping
for nt in (12, 10, 8, 6, 4, 2, 1):
    lib.lfm3_set_num_threads(nt)
    lib.lfm3_cache_reset()
    t0 = time.perf_counter()
    feed(ids)
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(16):
        feed([int(np.argmax(lg))])
    t_dec = time.perf_counter() - t0
    rec = {"threads": nt, "throttle_off": THROTTLE_OFF, "prefill_tok_s": round(len(ids) / t_pre, 2),
           "prefill_gmac_s": round(len(ids) * 1.47 / t_pre, 1),
           "decode_tok_s": round(16 / t_dec, 2), "decode_gb_s": round(16 * 3.45 / t_dec, 1)}
    print(json.dumps(rec), flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
print("THREADS_DONE", flush=True)
