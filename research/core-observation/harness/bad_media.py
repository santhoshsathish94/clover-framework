"""A6 probe: an attachment the handler cannot take must be said, not dropped. Sends a photo whose data is not base64.
Usage: python bad_media.py <port>"""
import json
import sys
import urllib.request
import uuid

URL = f"http://127.0.0.1:{int(sys.argv[1])}"
body = {"prompt": "What is in this picture?", "session_id": f"obs-badmedia-{uuid.uuid4().hex[:8]}", "history": [],
        "attached_files": [], "previous_media": [],
        "media": [{"kind": "image", "name": "broken.png", "data": "data:image/png;base64,not-base64!!"}]}
req = urllib.request.Request(URL + "/api/chat/stream", data=json.dumps(body).encode(), method="POST",
                             headers={"Content-Type": "application/json", "Referer": URL + "/"})
done = {}
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        if line.startswith(b"data: "):
            ev = json.loads(line[6:])
            if ev.get("type") in ("done", "error"):
                done = ev
print(json.dumps({"type": done.get("type"), "media_errors": done.get("media_errors"), "reply": done.get("reply"),
                  "error": done.get("error")}, ensure_ascii=False, indent=1))
