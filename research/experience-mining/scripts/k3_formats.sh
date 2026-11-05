#!/usr/bin/env bash
# k3, read-only: history and record formats of the sandbox memory repositories (synthetic drive messages only).
O=/opt/clover-k3/core-observe
for r in fw-base fw-base-csi-off fw-base-csi-on archive/before-a/fw-base archive/after-a-pass1/fw-base; do
  R=$O/$r/clover-memory-repo
  echo "== $r/clover-memory-repo: $(git -c safe.directory='*' -C $R rev-list --count HEAD 2>&1) commits, $(git -c safe.directory='*' -C $R log --reverse --format=%ci 2>/dev/null | head -1) .. $(git -c safe.directory='*' -C $R log -1 --format=%ci 2>/dev/null)"
  git -c safe.directory='*' -C $R log --format=%s 2>/dev/null | sed -E 's/[0-9a-f]{6,}|[0-9]+/#/g' | sort | uniq -c | sort -rn | head -8
done
R=$O/laptop-base/clover-ai-studio
echo "== laptop-base/clover-ai-studio: $(git -c safe.directory='*' -C $R rev-list --count HEAD 2>&1) commits"
git -c safe.directory='*' -C $R log --format=%s 2>/dev/null | sed -E 's/[0-9a-f]{6,}|[0-9]+/#/g' | sort | uniq -c | sort -rn | head -8
R=$O/fw-base-csi-on/clover-memory-repo
echo "== record formats (csi-on run; synthetic drive)"
for f in $(ls $R/sessions/2026-11/*.jsonl | head -2); do echo "-- $f"; head -c 700 $f; echo; done
echo "-- sessions md:"; ls $R/sessions | head -5; head -c 600 "$R/sessions/$(ls $R/sessions | grep -v 2026 | head -1)"; echo
for f in $R/outcomes/*/*.jsonl; do echo "-- $f"; head -n 4 $f | cut -c1-400; done
for f in $R/execution/*/*.log; do echo "-- $f ($(wc -l < $f) lines)"; head -n 6 $f | cut -c1-300; done
echo "-- growth/knowledge, notes, tasks (headings)"; for f in $R/growth/knowledge/*.md $R/notes/*.md $R/tasks/*.md; do echo "   $f: $(grep -m3 '^#' $f | tr '\n' ' ' | cut -c1-160)"; done
for f in $O/fw-base/clover-memory-repo/growth/notes/*/*.jsonl; do echo "-- $f"; head -n 2 $f | cut -c1-400; done
echo "-- memory index"; ls -la $O/fw-base-csi-on/cache/memory-index; head -c 500 $O/fw-base-csi-on/cache/memory-index/*.json; echo
R=$O/laptop-base/clover-ai-studio
echo "== laptop stack on k3: formats"
for f in $(ls $R/sessions/2026-11/*.jsonl | head -1); do echo "-- $f"; head -c 500 $f; echo; done
for f in $R/outcomes/*/*.jsonl; do echo "-- $f"; head -n 3 $f | cut -c1-400; done
for f in $R/execution/*/*.log; do echo "-- $f ($(wc -l < $f) lines)"; head -n 8 $f | cut -c1-300; done
