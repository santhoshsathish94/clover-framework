"""TODO step 5: how LFM3-Omni-2B uses this laptop. Iris Xe (llama.cpp Vulkan) and CPU instances, alone and at the same
time, and Q4_0 against Q4_K_M. Run with the Studio server stopped (it holds the GPU instance). One JSON line each.
"""
import io
import json
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import llamacpp_engine as L  # noqa: E402
import omni_native as O  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SENT = ("Clover runs on this laptop. It can read your files when you allow it, look at pictures, and answer out loud. "
        "Ask it anything you like.")
PROMPT = "Explain in about 150 words how a refrigerator keeps food cold."


def out(**kw):
    print(json.dumps(kw), flush=True)


def start(kind, quant="q4_0", threads=None):
    paths, missing = O.omni_files(BASE, quant)
    if missing:
        raise SystemExit("missing " + ", ".join(missing))
    extra = ["--reasoning", "off"] + (["-t", str(threads)] if threads else [])
    t0 = time.perf_counter()
    srv = L.LlamaServer(BASE, kind, 8192, model=paths["model"], mmproj=paths["mmproj"], extra_args=extra,
                        name=f"probe-{kind}-{quant}-{threads or 'auto'}", timeout_s=300)
    return srv, round(time.perf_counter() - t0, 1)


def chat(srv, n=220):
    body = dict(O.GREEDY, messages=[{"role": "user", "content": PROMPT}], max_tokens=n, cache_prompt=False)
    t0 = time.perf_counter()
    first, timings, text, n = None, None, "", 0
    for ch in srv.chat_stream(body):
        if ch.get("timings"):
            timings = ch["timings"]
        for c in ch.get("choices") or ():
            d = (c.get("delta") or {}).get("content")
            if d:
                first = first or time.perf_counter()
                text += d
                n += 1
    t = timings or {}
    end = time.perf_counter()
    return {"wall_s": round(end - t0, 2), "first_s": round((first or t0) - t0, 2),
            "gen_tok": t.get("predicted_n", n), "gen_tok_s": round(t.get("predicted_per_second") or
                                                                   (n - 1) / max(end - (first or end), 1e-6), 1),
            "prompt_tok_s": round(t.get("prompt_per_second") or 0, 1), "text": text[:80]}


def tts(srv):
    t0 = time.perf_counter()
    data = srv.tts(SENT)
    wall = time.perf_counter() - t0
    with wave.open(io.BytesIO(data)) as w:
        audio = w.getnframes() / w.getframerate()
    return {"wall_s": round(wall, 2), "audio_s": round(audio, 2), "x_realtime": round(audio / wall, 2)}


def alone(kind, quant="q4_0", threads=None):
    srv, load_s = start(kind, quant, threads)
    try:
        out(case=f"{kind} {quant} t={threads or 'auto'} alone", load_s=load_s, device=srv.device,
            chat=chat(srv), tts=tts(srv))
    finally:
        srv.close()


def together(threads=None):
    gpu, _ = start("vulkan")
    cpu, _ = start("cpu", threads=threads)
    try:
        res = {}
        th = threading.Thread(target=lambda: res.update(cpu_tts=tts(cpu)))
        th.start()
        time.sleep(0.5)
        res["gpu_chat"] = chat(gpu, 300)
        th.join()
        out(case=f"GPU chat while CPU (t={threads or 'auto'}) speaks", **res)
        res = {}
        th = threading.Thread(target=lambda: res.update(gpu_tts=tts(gpu)))
        th.start()
        time.sleep(0.5)
        res["cpu_chat"] = chat(cpu, 120)
        th.join()
        out(case=f"CPU (t={threads or 'auto'}) chat while GPU speaks", **res)
    finally:
        cpu.close()
        gpu.close()


if __name__ == "__main__":
    alone("vulkan")
    alone("cpu")
    alone("cpu", threads=4)
    alone("cpu", threads=8)
    together()
    alone("vulkan", "q4_k_m")
    out(case="done")
