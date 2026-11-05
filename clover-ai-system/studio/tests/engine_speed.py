"""Clover's C engine (bf16 weights) on the same shape as llama-bench pp512 / tg128."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
import winperf  # noqa: E402

winperf.disable_power_throttling()
import lfm3_native as N  # noqa: E402

base = HERE.parent
cache = N.EngineCache(base / "engine", base / "data" / "lfm3" / "pruned", 2048)
ids = list(range(1000, 1512))
t0 = time.perf_counter()
cache.prefill(ids)
t1 = time.perf_counter()
for _ in range(128):
    cache.step(int(np.argmax(cache.logits[:125033])))
t2 = time.perf_counter()
print(f"CLOVER_C_ENGINE bf16 threads={cache.lib.lfm3_get_num_threads()} pp512 {512 / (t1 - t0):.1f} tok/s  tg128 {128 / (t2 - t1):.2f} tok/s")
