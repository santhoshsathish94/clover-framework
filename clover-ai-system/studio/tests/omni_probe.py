"""LFM3-Omni-2B bring-up on this laptop through llama-server: what loads, what each input and output does, how fast.

Text, a tool call and its result, speech out (/tts), that speech back in (ASR), an image, and a video with a
soundtrack (made here with ffmpeg from two drawn frames and the TTS audio). Report: verify/omni-probe/report.json.
Run: python tests/omni_probe.py [q4_0|q4_k_m] [vulkan|cpu]
"""
import json
import struct
import subprocess
import sys
import time
import wave
import zlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "clover"))
import llamacpp_engine as L  # noqa: E402

QUANT = {"q4_0": "LFM3-Omni-2B-Q4_0.gguf", "q4_k_m": "LFM3-Omni-2B-Q4_K_M.gguf"}[sys.argv[1] if len(sys.argv) > 1 else "q4_0"]
KIND = sys.argv[2] if len(sys.argv) > 2 else "vulkan"
OUT = BASE / "verify" / "omni-probe"
MEDIA = BASE / "cache" / "media"
OUT.mkdir(parents=True, exist_ok=True)
MEDIA.mkdir(parents=True, exist_ok=True)
FF = (BASE / "tools" / "ffmpeg" / "BIN.txt").read_text().strip()
report = {"quant": QUANT, "kind": KIND}


def png(path, shape, rgb):
    w = h = 256
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            inside = (64 <= x < 192 and 64 <= y < 192) if shape == "square" else ((x - 128) ** 2 + (y - 128) ** 2 < 70 ** 2)
            row += bytes(rgb if inside else (255, 255, 255))
        rows.append(bytes(row))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
    path.write_bytes(data)


def chat(srv, messages, tools=None, max_tokens=256):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0, "cache_prompt": True}
    if tools:
        body["tools"] = tools
    content, calls, finish, timings, t0 = "", {}, None, {}, time.perf_counter()
    for ev in srv.chat_stream(body):
        timings = ev.get("timings") or timings
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return {"content": content, "tool_calls": list(calls.values()), "finish": finish, "wall_s": round(time.perf_counter() - t0, 2),
            "timings": {k: timings.get(k) for k in ("cache_n", "prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")}}


def step(name, fn):
    t0 = time.perf_counter()
    try:
        report[name] = fn()
    except Exception as exc:  # keep going: the report says which input failed and how
        report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    report[name]["step_s"] = round(time.perf_counter() - t0, 2) if isinstance(report[name], dict) else None
    print(name, json.dumps(report[name], ensure_ascii=False)[:700], flush=True)


t0 = time.perf_counter()
srv = L.LlamaServer(BASE, KIND, 32768, model=BASE / "data" / "omni" / "gguf" / QUANT,
                    mmproj=BASE / "data" / "omni" / "gguf" / "mmproj-LFM3-Omni-2B-Q8_0.gguf",
                    extra_args=["--reasoning", "off", "--media-path", str(MEDIA), "--video-fps", "2", "--video-ffmpeg-dir", FF],
                    name="omni-probe", timeout_s=300)
report["load_s"] = round(time.perf_counter() - t0, 1)
report["device"] = srv.device
try:
    p = srv.props()
    report["props"] = {"modalities": p.get("modalities"), "media_marker": bool(p.get("media_marker")),
                       "build": p.get("build_info"), "caps": p.get("chat_template_caps")}
    st, data = srv._request("GET", "/v1/models")
    report["models"] = [m.get("architecture") for m in json.loads(data).get("data", [])] if st == 200 else st
    print("loaded", json.dumps({k: report[k] for k in ("load_s", "device", "props", "models")}), flush=True)

    step("text", lambda: chat(srv, [{"role": "user", "content": "What is 17 times 23? Reply with the number only."}], max_tokens=32))
    tools = [{"type": "function", "function": {"name": "get_current_time", "description": "Current local date and time.",
                                               "parameters": {"type": "object", "properties": {}, "required": []}}},
             {"type": "function", "function": {"name": "run_python", "description": "Run Python code and return what it prints.",
                                               "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                              "required": ["code"]}}}]
    msgs = [{"role": "user", "content": "Use run_python to compute the sum of the squares of 1 to 100."}]

    def tool_round():
        first = chat(srv, msgs, tools)
        out = {"first": first}
        if first["tool_calls"]:
            c = first["tool_calls"][0]
            msgs2 = msgs + [{"role": "assistant", "content": first["content"], "tool_calls": [
                {"id": c["id"] or "call_1", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]},
                {"role": "tool", "tool_call_id": c["id"] or "call_1", "content": "338350"}]
            out["second"] = chat(srv, msgs2, tools)
        return out
    step("tools", tool_round)

    def speak():
        wav = srv.tts("The quick brown fox jumps over the lazy dog. Clover runs on this laptop.")
        (MEDIA / "probe-tts.wav").write_bytes(wav)
        with wave.open(str(MEDIA / "probe-tts.wav")) as w:
            return {"bytes": len(wav), "rate": w.getframerate(), "channels": w.getnchannels(),
                    "seconds": round(w.getnframes() / w.getframerate(), 2)}
    step("tts", speak)
    step("asr", lambda: chat(srv, [{"role": "system", "content": "Perform ASR in English."},
                                   {"role": "user", "content": [{"type": "input_audio", "input_audio": {"url": "file://probe-tts.wav"}}]}],
                             max_tokens=64))
    png(MEDIA / "probe-a.png", "square", (220, 30, 30))
    png(MEDIA / "probe-b.png", "circle", (30, 60, 220))
    step("image", lambda: chat(srv, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://probe-a.png"}},
        {"type": "text", "text": "What shape and colour do you see? One sentence."}]}], max_tokens=48))

    def video():
        mp4 = MEDIA / "probe-video.mp4"
        cmd = [str(Path(FF) / "ffmpeg.exe"), "-y", "-loglevel", "error", "-loop", "1", "-t", "3", "-i", str(MEDIA / "probe-a.png"),
               "-loop", "1", "-t", "3", "-i", str(MEDIA / "probe-b.png"), "-i", str(MEDIA / "probe-tts.wav"),
               "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]", "-map", "[v]", "-map", "2:a",
               "-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-shortest", str(mp4)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120, creationflags=0x08000000)
        r = chat(srv, [{"role": "system", "content": "Answer questions about the video using what you see and hear."},
                       {"role": "user", "content": [{"type": "input_video", "input_video": {"url": "file://probe-video.mp4"}},
                                                    {"type": "text", "text": "What do you see change, and what is said?"}]}],
                 max_tokens=96)
        r["video_bytes"] = mp4.stat().st_size
        return r
    step("video", video)
    step("speed", lambda: chat(srv, [{"role": "user", "content": "Write about 150 words on why the sky is blue."}], max_tokens=220))
finally:
    srv.close()
    (OUT / f"report-{QUANT.split('-')[-1].split('.')[0].lower()}-{KIND}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PROBE_DONE", flush=True)
