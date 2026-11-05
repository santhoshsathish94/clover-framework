"""Read-only, laptop: outcome field kinds and session classes over the records repo's whole git history (identifiers,
counts and categorical values only; no message text). Usage: python <this> <CloverAI dir>"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1]) / "clover-ai-studio"


def git(*a):
    return subprocess.run(["git", "-C", str(R), *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def sclass(sid):
    if sid.startswith("obs-"):
        return "observation-drive"
    if sid.startswith("live-recall"):
        return "recall-test"
    if re.fullmatch(r"[a-z0-9]{14,26}", sid):
        return "studio-ui-id"
    return re.sub(r"[0-9a-f]{6,}|\d+", "#", sid)[:20]


print("== outcomes: categorical values")
recs = [json.loads(l) for p in sorted((R / "outcomes").rglob("*.jsonl")) for l in open(p, encoding="utf-8") if l.strip()]
c = collections.defaultdict(collections.Counter)
for r in recs:
    for k in ("result", "feedback", "worked", "by"):
        if k in r and not (k == "by" and isinstance(r[k], int)):
            c[k][str(r[k])[:40]] += 1
    for k in ("tools", "used", "checks"):
        for x in r.get(k) or []:
            c[k][(json.dumps(x)[:50] if not isinstance(x, str) else re.sub(r"\d+", "#", x)[:50])] += 1
    if "lesson" in r:
        c["lesson"]["present (" + str(len(str(r["lesson"]))) + " chars)"] += 1
for k, v in c.items():
    print(f"    {k}: {dict(v.most_common(25))}")
print("== outcomes by session class")
by = collections.defaultdict(collections.Counter)
for r in recs:
    s = sclass(r.get("session", ""))
    by[s]["records"] += 1
    if "feedback" in r:
        by[s]["fb:" + r["feedback"]] += 1
    if r.get("tools"):
        by[s]["turns with tools"] += 1
for s, v in by.items():
    print(f"    {s:20s} {dict(v)}")
print("== sessions over history (from file names in git log)")
names = set(re.findall(r"sessions/\d{4}-\d\d/([^\s/]+)\.jsonl", git("log", "--all", "--name-only", "--format=")))
cls = collections.Counter(sclass(n) for n in names)
print(f"    distinct session files ever: {len(names)} by class: {dict(cls)}")
print("== turn commits by day and session class")
days = collections.defaultdict(collections.Counter)
for line in git("log", "--format=%ci|%s").splitlines():
    d, _, s = line.partition("|")
    m = re.match(r"Turn (\d+) of (\S+): (\S+)", s)
    if m:
        days[d[:10]][sclass(m.group(2)) + ":" + m.group(3)] += 1
for d in sorted(days):
    print(f"    {d}: {dict(days[d])}")
print("== growth notes: non-empty notes / all, by session class")
gn = [json.loads(l) for p in sorted((R / "growth").rglob("*.jsonl")) for l in open(p, encoding="utf-8") if l.strip()]
g = collections.Counter((sclass(r.get("session", "")), bool((r.get("note") or "").strip())) for r in gn)
print("   ", dict(g), "by:", dict(collections.Counter(r.get("by") for r in gn)))
print("== growth knowledge changes over history")
print("   ", dict(collections.Counter(re.sub(r"MEM-[A-Z]+-[\w-]+", "MEM-*", s)[:70] for s in git("log", "--format=%s", "--", "growth/knowledge").splitlines())))
print("== execution log: event words per line (after the session id)")
ev = collections.Counter()
for p in sorted((R / "execution").rglob("*.log")):
    for line in open(p, encoding="utf-8", errors="replace"):
        parts = line.split(" ", 3)
        if len(parts) == 4:
            ev[re.split(r"[:(]", parts[3], 1)[0].strip()[:30]] += 1
print("   ", dict(ev.most_common(20)))
