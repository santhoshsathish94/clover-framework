"""TODO step 5b: Studio-sized prompts (system + tools + a long message, about 2,500 tokens) on LFM3-Omni-2B.
Iris Xe (Vulkan, all layers), CPU build, and the Vulkan build with weights on the CPU (-ngl 0: llama.cpp can still send
large prompt batches to the GPU). Run with the Studio server stopped. One JSON line per configuration.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import llamacpp_engine as L  # noqa: E402
import lfm3_native as N  # noqa: E402
import omni_native as O  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TEXT = " ".join(f"Note {i}: the warehouse on row {i} keeps {i * 7 % 50 + 10} crates of apples, {i * 3 % 20 + 5} crates "
                f"of pears and a ledger signed by clerk number {i * 11 % 97}." for i in range(1, 61))
QUESTION = "\n\nHow many crates of pears are kept on row 17? Answer in one sentence."


def run(srv):
    body = dict(O.GREEDY, messages=[{"role": "system", "content": "You are Clover, an assistant on the user's laptop."},
                                    {"role": "user", "content": TEXT + QUESTION}],
                tools=N.STUDIO_TOOLS, max_tokens=120, cache_prompt=False)
    t0 = time.perf_counter()
    tm, text = {}, ""
    for ch in srv.chat_stream(body):
        tm = ch.get("timings") or tm
        for c in ch.get("choices") or ():
            text += (c.get("delta") or {}).get("content") or ""
    return {"wall_s": round(time.perf_counter() - t0, 2), "prompt_n": tm.get("prompt_n"),
            "prompt_tok_s": round(tm.get("prompt_per_second") or 0, 1), "gen_n": tm.get("predicted_n"),
            "gen_tok_s": round(tm.get("predicted_per_second") or 0, 1), "text": text.strip()[:90]}


def case(label, kind, extra=(), quant="q4_0"):
    paths, _ = O.omni_files(BASE, quant)
    srv = L.LlamaServer(BASE, kind, 8192, model=paths["model"], mmproj=paths["mmproj"],
                        extra_args=["--reasoning", "off", *extra], name="prefill-" + label.replace(" ", "-"), timeout_s=300)
    try:
        run(srv)  # warm-up
        print(json.dumps({"case": label, **run(srv)}), flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    case("gpu ngl99 ub256", "vulkan")
    case("gpu ngl99 ub512", "vulkan", ["-ub", "512"])
    case("cpu build", "cpu")
    case("vulkan build ngl0 (op offload)", "vulkan", ["-ngl", "0"])
    case("vulkan build ngl0 ub512 (op offload)", "vulkan", ["-ngl", "0", "-ub", "512"])
    print(json.dumps({"case": "done"}), flush=True)
