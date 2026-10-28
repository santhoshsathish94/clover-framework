#!/usr/bin/env bash
set -euo pipefail

TEX_DIR="/root/dm-study/specmd/w2/paper/tex"
OUT_DIR="/root/dm-study/specmd/w2/paper"

/root/ncm-cpu/venv/bin/python - << 'PY'
import os

tex_dir = "/root/dm-study/specmd/w2/paper/tex"
part1_files = ["0_main.tex", "1_intro.tex", "2_related_work.tex", "3_methodology.tex"]
part2_files = ["4_evaluation.tex", "5_conclusion.tex", "6_appendix.tex"]

def dump_files(flist, out_path):
    with open(out_path, "w", encoding="utf-8") as out:
        for fn in flist:
            p = os.path.join(tex_dir, fn)
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            out.write(f"================ {fn} ({len(lines)} lines) ================\n")
            for idx, line in enumerate(lines, 1):
                out.write(f"{idx:03d}: {line}")
            out.write("\n")

dump_files(part1_files, "/root/dm-study/specmd/w2/paper/tex_part1.txt")
dump_files(part2_files, "/root/dm-study/specmd/w2/paper/tex_part2.txt")
for p in ["/root/dm-study/specmd/w2/paper/tex_part1.txt", "/root/dm-study/specmd/w2/paper/tex_part2.txt"]:
    print(p, os.path.getsize(p), "bytes")
PY
