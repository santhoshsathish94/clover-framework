#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/specmd/w2"
PAPER_DIR="$W2/paper"
rm -rf "$PAPER_DIR"
mkdir -p "$PAPER_DIR/tex"

echo "=== 1. Fetch arXiv API metadata ==="
curl -sL "https://export.arxiv.org/api/query?id_list=2602.03921" -o "$PAPER_DIR/arxiv_meta.xml"
/root/ncm-cpu/venv/bin/python - << 'PY'
import xml.etree.ElementTree as ET
tree = ET.parse("/root/dm-study/specmd/w2/paper/arxiv_meta.xml")
root = tree.getroot()
ns = {"a": "http://www.w3.org/2005/Atom"}
entry = root.find("a:entry", ns)
if entry is not None:
    print("Title:", " ".join(entry.find("a:title", ns).text.split()))
    print("Published:", entry.find("a:published", ns).text)
    print("Updated:", entry.find("a:updated", ns).text)
    authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)]
    print("Authors:", ", ".join(authors))
PY

echo ""
echo "=== 2. Fetch arXiv e-print TeX bundle ==="
curl -sL "https://arxiv.org/e-print/2602.03921" -o "$PAPER_DIR/2602.03921.tar.gz"
ls -lh "$PAPER_DIR/2602.03921.tar.gz"
sha256sum "$PAPER_DIR/2602.03921.tar.gz"
tar -xzf "$PAPER_DIR/2602.03921.tar.gz" -C "$PAPER_DIR/tex"

echo ""
echo "=== 3. TeX file inventory (line counts) ==="
wc -l "$PAPER_DIR/tex"/*.tex "$PAPER_DIR/tex"/*/*.tex 2>/dev/null || wc -l "$PAPER_DIR/tex"/*.tex

echo ""
echo "=== 4. Extract key sections from TeX files ==="
/root/ncm-cpu/venv/bin/python - << 'PY'
import glob, os

tex_dir = "/root/dm-study/specmd/w2/paper/tex"
files = sorted(glob.glob(os.path.join(tex_dir, "**", "*.tex"), recursive=True))
for f in files:
    rel = os.path.relpath(f, tex_dir)
    with open(f, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    print(f"--- FILE: {rel} ({len(lines)} lines) ---")
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if any(k in s.lower() for k in [
            "\\title", "\\author", "\\section", "\\subsection", "\\subsubsection",
            "algorithm", "collision", "least-stale", "least stale", "overfetch",
            "cache-aware", "olmoe", "bandwidth", "1\\%", "5\\%", "25\\%", "open-sourc"
        ]):
            print(f"  L{i}: {s[:140]}")
PY
