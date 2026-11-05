"""How much of a growing prompt llama-server reuses for LFM3 (hybrid conv/attention), by physical batch size."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
import winperf  # noqa: E402

winperf.disable_power_throttling()
from transformers import AutoTokenizer  # noqa: E402

import llamacpp_engine as L  # noqa: E402
import lfm3_native as N  # noqa: E402

base = HERE.parent
tok = AutoTokenizer.from_pretrained(str(base / "data" / "lfm3" / "hf"))
text = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog near river {i % 17}." for i in range(160))
ids = tok("<|startoftext|><|im_start|>user\n" + text, add_special_tokens=False).input_ids
a, b = ids[:1400], ids[:1400] + tok(" And one more line about the fox and the dog.<|im_end|>\n<|im_start|>assistant\n",
                                     add_special_tokens=False).input_ids
for ub in (512, 256, 128):
    t0 = time.perf_counter()
    srv = L.LlamaServer(base, "vulkan", 8192, ubatch=ub)
    load = time.perf_counter() - t0
    res = []
    for p in (a, b):
        fin = [f for _, f in srv.stream(p, L.sampling_params(N.SAMPLING["greedy"], 16, 0)) if f][-1]
        tm = fin["timings"]
        res.append((len(p), tm["prompt_n"], round(tm["prompt_per_second"]), round(tm["predicted_per_second"], 1)))
    srv.close()
    (la, na, ppa, tga), (lb, nb, ppb, tgb) = res
    print(f"ubatch {ub:4d}: load {load:5.1f} s | first {la} tokens read at {ppa} tok/s, write {tga} tok/s | "
          f"second {lb} tokens: reused {lb - nb}, read {nb} new")
