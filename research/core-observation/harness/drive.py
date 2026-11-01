"""Sends the same conversations to both stacks, as their web pages do, and records every event each one streams back.
Usage: python drive.py --stack framework|laptop --port N --out <turns.jsonl> --files <dir the file scenario reads>"""
import argparse
import base64
import io
import json
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--stack", choices=("framework", "laptop"), required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--files", required=True)
ap.add_argument("--only", default="")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"
HEAD = {"Content-Type": "application/json", "Referer": URL + "/"}  # what the stack's own page sends (same origin)


def photo():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 620, 380), outline="black", width=4)
    d.text((60, 70), "INVOICE 4721", fill="black")
    d.text((60, 140), "Coffee beans 2 x 12.50", fill="black")
    d.text((60, 190), "Milk 3 x 1.80", fill="black")
    d.text((60, 260), "TOTAL 30.40", fill="black")
    d.ellipse((470, 230, 590, 350), outline="red", width=6)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


FILE = a.files.rstrip("/") + "/meeting-notes.txt"
SCENARIOS = [
    ("social", ["Hi", "How are you?", "Thanks"]),
    ("knowledge", ["What is the capital of Australia?"]),
    ("pens", ["A shop sells pens at 3 for $2. How much do 12 pens cost?", "And how much would 45 pens cost?",
              "That's wrong, try again"]),
    ("web", ["Search the web for the population of Jaipur"]),
    ("python", ["Use Python to compute 2**64 + 17"]),
    ("remember", ["Remember that I'm vegetarian."]),
    ("dinner", ["Suggest a quick dinner I can cook tonight."]),
    ("photo", [("What is in this picture?", "photo"), "What number is written on it?"]),
    ("file", [f"Read the file {FILE} and summarize it."]),
    ("delete", [f"Delete the old report files in {a.files.rstrip('/')}/reports"]),
]


def post(path, body, stream=False, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=HEAD, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def turn(sid, text, media, history, prev_media, out):
    body = {"prompt": text, "session_id": sid, "history": history, "attached_files": [], "max_new_tokens": 1024,
            "media": media, "previous_media": prev_media}
    t0 = time.time()
    events, done = [], None
    with post("/api/chat/stream", body, stream=True) as r:
        buf = b""
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            ev["_t"] = round(time.time() - t0, 2)
            events.append(ev)
            if ev.get("type") == "permission":  # refuse anything that asks: the observation runs nothing risky
                post("/api/decision", {"turn_id": ev.get("turn_id"), "request_id": ev.get("request_id"),
                                       "decision": "deny"}).read()
            if ev.get("type") in ("done", "error"):
                done = ev
    tokens = "".join(e.get("text", "") for e in events if e.get("type") == "token" and e.get("kind") == "answer")
    rec = {"stack": a.stack, "session": sid, "prompt": text, "media": [m["kind"] for m in media],
           "seconds": round(time.time() - t0, 2), "reply": (done or {}).get("reply"), "streamed_answer": tokens,
           "event_types": [e.get("type") for e in events],
           "events": [e for e in events if e.get("type") not in ("token",)]}
    out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    out.flush()
    return (done or {}).get("reply") or ""


with open(a.out, "a", encoding="utf-8") as out:
    for name, msgs in SCENARIOS:
        if a.only and name not in a.only.split(","):
            continue
        sid = f"obs-{name}-{uuid.uuid4().hex[:8]}"
        history, prev_media = [], []
        for m in msgs:
            text, kind = (m if isinstance(m, tuple) else (m, None))
            media = [{"kind": "image", "name": "invoice.png", "data": photo()}] if kind == "photo" else []
            try:
                reply = turn(sid, text, media, history[-6:], prev_media, out)
            except Exception as exc:  # recorded, then the next scenario
                out.write(json.dumps({"stack": a.stack, "session": sid, "prompt": text,
                                      "driver_error": f"{type(exc).__name__}: {exc}"}) + "\n")
                out.flush()
                break
            history.append({"user": text, "reply": reply})
            prev_media = [x["kind"] for x in media]
            print(f"{a.stack} {name}: {text[:50]!r} -> {reply[:90]!r}", flush=True)
print("DRIVE_DONE", flush=True)
