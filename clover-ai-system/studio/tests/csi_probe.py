"""Ask the running Studio's CSI service how it reads a few sentences (declined tools): which propositions, polarity.
Usage: python tests/csi_probe.py [CSI URL]"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import semantics  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/health").read())["csi"]["url"]
read = semantics.csi_reader(url, timeout=30)
for s in ("Without searching the web, explain what photosynthesis is in three sentences.",
          "Don't search the web, just explain what photosynthesis is in three sentences.",
          "Do not look it up online; tell me what you know about photosynthesis.",
          "Explain photosynthesis without using the internet.",
          "No web search please, just explain photosynthesis."):
    a = read(s)
    props = [(p["predicate"]["lemma"] if "lemma" in p["predicate"] else a["readings"][0]["tokens"][p["predicate"]["token"]]["text"],
              p["polarity"], p.get("mood"), p.get("embedded_under")) for p in (a or {"readings": [{"propositions": []}]})["readings"][0]["propositions"]]
    print(json.dumps({"text": s, "declined": sorted(semantics.declined_caps(s, a)), "propositions": props}, default=str))
