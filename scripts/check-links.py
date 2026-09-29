"""Sweep every markdown relative link and in-page anchor in the repo."""
import os
import re
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)


def slug(text):
    """GitHub replaces each space with a hyphen, so an em dash between spaces
    leaves a double hyphen. Collapsing runs here would reject valid links."""
    s = text.strip().lower()
    s = re.sub(r"`|\*|_|\[|\]|\(|\)|<[^>]*>", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")


def anchors(path):
    try:
        t = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    out = set()
    fence = None
    for line in t.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = HEAD.match(line)
        if heading:
            base = slug(heading.group(2))
            anchor = base
            suffix = 0
            while anchor in out:
                suffix += 1
                anchor = "%s-%d" % (base, suffix)
            out.add(anchor)
    for m in re.finditer(r'<a\s+[^>]*(?:name|id)="([^"]+)"', t):
        out.add(m.group(1))
    return out


md = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "build")]
    for f in filenames:
        if f.endswith(".md"):
            md.append(os.path.join(dirpath, f))

anchor_cache = {}
bad_link, bad_anchor, checked = [], [], 0

for path in sorted(md):
    base = os.path.dirname(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        checked += 1
        frag = ""
        if "#" in target:
            target, frag = target.split("#", 1)
        if target == "":
            dest = path
        else:
            dest = os.path.normpath(os.path.join(base, target.replace("/", os.sep)))
        if not os.path.exists(dest):
            bad_link.append("%s -> %s" % (rel, target + ("#" + frag if frag else "")))
            continue
        if frag and dest.endswith(".md"):
            if dest not in anchor_cache:
                anchor_cache[dest] = anchors(dest)
            if frag.lower() not in anchor_cache[dest]:
                bad_anchor.append("%s -> %s#%s" % (rel, target, frag))

print("markdown files scanned : %d" % len(md))
print("relative links checked : %d" % checked)
print("broken links           : %d" % len(bad_link))
for b in bad_link:
    print("   " + b)
print("dead anchors           : %d" % len(bad_anchor))
for b in bad_anchor:
    print("   " + b)
if __name__ == "__main__":
    sys.exit(1 if (bad_link or bad_anchor) else 0)
