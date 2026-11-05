"""Studio API check for LFM3-Omni-2B through the tunnel (streamed turns, as the Studio window sends them):
text, a tool, a photo, a follow-up about that photo, Clover's voice (/api/tts), that voice sent back as a voice message,
a video with a soundtrack, and a rejected fake attachment. Run on the dev machine: python tests/studio_omni_api_test.py VIDEO.mp4"""
import base64
import http.client
import json
import struct
import sys
import time
import urllib.request
import zlib

HOST, PORT = "127.0.0.1", 8765


def call(method, path, body=None, timeout=600, raw=False):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return (data, r.headers.get("Content-Type")) if raw else json.loads(data)


def stream(sid, prompt, media=()):
    """One streamed turn; permission requests are answered 'once'. -> (done event, event types, seconds)."""
    t0 = time.time()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=900)
    conn.request("POST", "/api/chat/stream", body=json.dumps({"prompt": prompt, "session_id": sid, "media": list(media)}),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {r.read()[:300]!r}"}, [], round(time.time() - t0, 1)
    kinds, done, buf = [], None, b""
    while done is None:
        line = r.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        kinds.append(ev["type"])
        if ev["type"] == "permission":
            call("POST", "/api/decision", {"turn_id": ev["turn_id"], "request_id": ev["request_id"], "decision": "once"})
        if ev["type"] in ("done", "error"):
            done = ev
    conn.close()
    return done or {"error": "stream ended"}, kinds, round(time.time() - t0, 1)


def show(name, done, kinds, secs, expect=None):
    reply = done.get("reply") or done.get("error") or ""
    gens = [(g["prompt_tokens"], g["reused_tokens"], g["new_tokens"], g.get("decode_tok_s")) for g in done.get("generation", [])]
    ok = (expect is None) or any(e.lower() in reply.lower() for e in expect)
    print(f"[{'ok' if ok else 'CHECK'}] {name} {secs} s\n    reply={reply[:220]!r}\n    tools={[t['tool'] for t in done.get('tool_calls', [])]} "
          f"gen={gens} events={sorted(set(kinds))}", flush=True)
    return ok


def png(shape, rgb):
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
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")


st = call("GET", "/api/status?passive=1")
print("status:", {k: st.get(k) for k in ("status", "model", "engine", "inputs")})
results = []
sid = "omni-test-" + str(int(time.time()))
results.append(show("text", *stream(sid + "-a", "What is 17 times 23? Reply with the number."), expect=["391"]))
results.append(show("tool", *stream(sid + "-b", "Use Python to compute the sum of the squares of 1 to 100."), expect=["338350", "338,350"]))
img = base64.b64encode(png("circle", (30, 60, 220))).decode()
results.append(show("photo", *stream(sid + "-c", "What shape and colour is in this picture?",
                                     [{"kind": "image", "name": "circle.png", "data": img}]), expect=["blue"]))
results.append(show("follow-up (photo in history)", *stream(sid + "-c", "What colour was the shape? One word."), expect=["blue"]))
t0 = time.time()
wav, ctype = call("POST", "/api/tts", {"text": "What is the capital of France?", "voice": "female"}, raw=True)
print(f"[{'ok' if wav[:4] == b'RIFF' else 'CHECK'}] tts {round(time.time() - t0, 1)} s {ctype} {len(wav)} bytes", flush=True)
results.append(wav[:4] == b"RIFF")
results.append(show("voice message", *stream(sid + "-d", "", [{"kind": "audio", "name": "Voice message", "data": base64.b64encode(wav).decode()}]),
                    expect=["paris"]))
if len(sys.argv) > 1:
    vid = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
    results.append(show("video", *stream(sid + "-e", "What do you see change, and what is said?",
                                         [{"kind": "video", "name": "probe.mp4", "data": vid}]), expect=["circle", "blue"]))
done, kinds, secs = stream(sid + "-f", "Describe this.", [{"kind": "image", "name": "fake.png", "data": base64.b64encode(b"not an image").decode()}])
results.append(show("fake attachment refused", done, kinds, secs, expect=["unsupported image"]))
print(f"RESULT {sum(map(bool, results))}/{len(results)} checks passed", flush=True)
