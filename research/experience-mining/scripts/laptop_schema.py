"""Read-only schema inventory of the laptop's Clover records: key sets, value types, counts, date ranges and identifiers.
Prints no message text, no file contents beyond field names and short structural markers. Usage: python <this> <CloverAI dir>"""
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

B = Path(sys.argv[1])
R = B / "clover-ai-studio"


def kind(v):
    if isinstance(v, str):
        return "str"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict(" + ",".join(sorted(v)[:12]) + ")"
    return type(v).__name__


def jsonl_schema(paths, label, show_values_for=()):
    keys, n, bad, vals = collections.Counter(), 0, 0, collections.defaultdict(collections.Counter)
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            n += 1
            for k, v in (r.items() if isinstance(r, dict) else []):
                keys[(k, kind(v))] += 1
                if k in show_values_for and isinstance(v, (str, bool, int)) and len(str(v)) <= 40:
                    vals[k][str(v)] += 1
    print(f"== {label}: files={len(paths)} records={n} unparsable={bad}")
    for (k, t), c in sorted(keys.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {t[:110]:110s} {c}")
    for k, c in vals.items():
        print(f"    values[{k}]: {dict(c.most_common(15))}")


def git(*args):
    return subprocess.run(["git", "-C", str(R), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


print("== records repo history (read-only git log)")
log = git("log", "--name-status", "--format=@@%ci|%s")
commits, status, subjects, top = 0, collections.Counter(), collections.Counter(), collections.Counter()
deleted_sessions, added_sessions, months = set(), set(), collections.Counter()
for line in log.splitlines():
    if line.startswith("@@"):
        commits += 1
        date, _, subj = line[2:].partition("|")
        months[date[:10]] += 1
        subjects[re.sub(r"[0-9a-f]{6,}|\d+", "#", subj)[:60]] += 1
    elif "\t" in line:
        st, path = line.split("\t", 1)
        status[st[0]] += 1
        top[path.split("/")[0] + ("/" + path.split("/")[1] if path.count("/") >= 2 else "")] += 1
        if path.startswith("sessions/"):
            (deleted_sessions if st.startswith("D") else added_sessions).add(path)
print(f"    commits={commits} file-changes={dict(status)}")
print(f"    commits per day: {dict(sorted(months.items()))}")
print(f"    subject patterns: {dict(subjects.most_common(15))}")
print(f"    changes by area: {dict(top.most_common(20))}")
print(f"    session files ever added={len(added_sessions)} deleted later={len(deleted_sessions & added_sessions)} "
      f"in tree now={len(list((R / 'sessions').rglob('*.jsonl')))}")

jsonl_schema(sorted((R / "sessions").rglob("*.jsonl")), "sessions/*.jsonl (current tree)", ("role", "kind", "type", "event"))
jsonl_schema(sorted((R / "outcomes").rglob("*.jsonl")), "outcomes/*.jsonl", ("feedback", "kind", "type", "outcome", "signal"))
jsonl_schema(sorted((R / "growth").rglob("*.jsonl")), "growth/**/*.jsonl", ("kind", "type", "tag"))
print("== execution/*.log line shapes")
shapes, n = collections.Counter(), 0
for p in sorted((R / "execution").rglob("*.log")):
    for line in open(p, encoding="utf-8", errors="replace"):
        n += 1
        shapes[re.sub(r"\d", "#", line[:40]).split(" ")[0:3].__str__()] += 1
print(f"    lines={n}; leading shapes: {dict(shapes.most_common(12))}")
for p in sorted((R / "execution").rglob("*.log"))[:1]:
    first = open(p, encoding="utf-8", errors="replace").readline()
    print("    field markers in first line:", re.findall(r"\b([a-z_]+)=", first)[:20], "| separators:", sorted(set(re.findall(r"[|\t]", first))))
print("== growth/notes and other files (names, sizes)")
for p in sorted(R.rglob("*")):
    if p.is_file() and ".git" not in p.parts and p.suffix in (".md", ".json"):
        print(f"    {str(p.relative_to(R)):60s} {p.stat().st_size}")
print("== CONTEXT.md headings (no body)")
for line in open(R / "CONTEXT.md", encoding="utf-8", errors="replace"):
    if line.startswith("#"):
        print("   ", line.rstrip()[:100])

print("== logs/*.log line shapes")
shapes, n, files = collections.Counter(), 0, sorted((B / "logs").glob("*.log"))
for p in files:
    for line in open(p, encoding="utf-8", errors="replace"):
        n += 1
        m = re.match(r"\s*(\[[^\]]{0,30}\]|\d{4}-\d\d-\d\d[ T][\d:.,]+|[A-Za-z_]+[:=]?)", line)
        shapes[re.sub(r"\d", "#", m.group(1)) if m else line[:12]] += 1
print(f"    files={len(files)} lines={n} names={sorted({re.sub(r'[-_.]?\d.*', '', p.stem) for p in files})}")
print(f"    leading shapes: {dict(shapes.most_common(15))}")

print("== verify/ result files (schemas of jsonl)")
groups = collections.defaultdict(list)
for p in (B / "verify").rglob("*.jsonl"):
    groups[str(p.parent.relative_to(B))].append(p)
for g, ps in sorted(groups.items()):
    if "recall-memory" in g:
        print(f"== {g}: {len(ps)} files (test sessions; schema skipped)")
        continue
    jsonl_schema(sorted(ps), g, ("arm", "step", "ok", "pass", "correct", "category", "kind"))

print("== data/ record-like files (not model weights)")
for p in sorted((B / "data").rglob("*")):
    if p.is_file() and p.suffix in (".jsonl", ".json", ".db", ".sqlite", ".sqlite3", ".md", ".txt", ".csv") and p.stat().st_size < 200_000_000:
        print(f"    {str(p.relative_to(B)):80s} {p.stat().st_size:>12}")
