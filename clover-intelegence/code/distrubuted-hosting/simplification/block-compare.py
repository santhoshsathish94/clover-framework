#!/usr/bin/env python3
"""Test the single-prompt block claims against every prompt in the census.

Four claims came out of folding one prompt by its 12-pod block. Each is stated here as
something that can fail, and the script reports where it does rather than how often it
holds. A claim that survives twelve prompts is worth more than one that was never put at
risk; a claim that breaks on two of them is worth knowing about before it gets written
down as structure.

  1  the fold's three largest relative movements are exactly the MLA offsets 3, 7, 11
  2  offset 12 folds nothing - relative movement exactly 0
  3  a full block multiplies the state by about 1.38 end to end
  4  the residual grows monotonically from block offset 1 to 11
  5  routing is not organised by block offset

usage: block-compare.py ROWS.tsv [TRACE-COUNTS.tsv]
"""
import collections
import math
import sys

BLOCK = 12
MLA_OFFSETS = {3, 7, 11}


def median(values):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load(path):
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            for key in ("position", "block", "offset", "pod"):
                row[key] = int(row[key])
            for key in ("out_norm", "rel_attn", "rel_fold", "rel_moe", "in_norm",
                        "spread", "ratio"):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = float("nan")
            rows.append(row)
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    rows = load(sys.argv[1])
    prompts = sorted({r["prompt"] for r in rows})
    positions = {p: sorted({r["position"] for r in rows if r["prompt"] == p}) for p in prompts}
    print("%d rows, %d prompts, %d pod-positions"
          % (len(rows), len(prompts), len(rows) // 92))
    print("   " + "  ".join("%s(%d)" % (p, len(positions[p])) for p in prompts))

    print("\n== claim 1: the fold's three largest movements are offsets 3, 7, 11 ==")
    print("   %-10s %-28s %s" % ("prompt", "top three offsets by median", "holds"))
    broken = []
    for prompt in prompts:
        by_offset = collections.defaultdict(list)
        for row in rows:
            if row["prompt"] == prompt and row["offset"] != BLOCK \
                    and not math.isnan(row["rel_fold"]):
                by_offset[row["offset"]].append(row["rel_fold"])
        ranked = sorted(by_offset, key=lambda o: -median(by_offset[o]))
        top3 = set(ranked[:3])
        ok = top3 == MLA_OFFSETS
        if not ok:
            broken.append(prompt)
        print("   %-10s %-28s %s"
              % (prompt, ",".join(str(o) for o in ranked[:3]), "yes" if ok else "NO"))
    print("   fails on %d of %d prompts%s"
          % (len(broken), len(prompts), (": " + ", ".join(broken)) if broken else ""))

    print("\n== claim 2: offset 12 folds nothing ==")
    nonzero = [r for r in rows if r["offset"] == BLOCK and r["rel_fold"] != 0.0
               and not math.isnan(r["rel_fold"])]
    total = len([r for r in rows if r["offset"] == BLOCK])
    print("   %d of %d offset-12 rows have non-zero fold movement" % (len(nonzero), total))
    for row in nonzero[:10]:
        print("     %s position %d pod %d  rel_fold %.6f"
              % (row["prompt"], row["position"], row["pod"], row["rel_fold"]))

    print("\n== claim 3: a full block multiplies the state by about 1.38 ==")
    print("   %-10s %9s %9s %9s %9s" % ("prompt", "geo mean", "min block", "max block", "blocks"))
    all_ratios = []
    for prompt in prompts:
        ratios = []
        for position in positions[prompt]:
            frame = {(r["block"], r["offset"]): r for r in rows
                     if r["prompt"] == prompt and r["position"] == position}
            for block in range(1, 8):
                start = frame.get((block, 1))
                end = frame.get((block, BLOCK))
                if start and end and start["in_norm"] > 0:
                    ratios.append(end["out_norm"] / start["in_norm"])
        if not ratios:
            continue
        all_ratios.extend(ratios)
        geo = math.exp(sum(math.log(r) for r in ratios if r > 0) / len(ratios))
        print("   %-10s %9.3f %9.3f %9.3f %9d"
              % (prompt, geo, min(ratios), max(ratios), len(ratios)))
    if all_ratios:
        geo = math.exp(sum(math.log(r) for r in all_ratios if r > 0) / len(all_ratios))
        print("   all %d full blocks: geometric mean %.3f, range %.4f to %.2f"
              % (len(all_ratios), geo, min(all_ratios), max(all_ratios)))

    print("\n== claim 4: the residual grows monotonically from offset 1 to 11 ==")
    monotone = strict = checked = 0
    worst = []
    for prompt in prompts:
        for position in positions[prompt]:
            frame = {(r["block"], r["offset"]): r for r in rows
                     if r["prompt"] == prompt and r["position"] == position}
            for block in range(1, 9):
                series = [frame[(block, o)]["out_norm"] for o in range(1, BLOCK)
                          if (block, o) in frame]
                if len(series) < 3:
                    continue
                checked += 1
                drops = sum(1 for a, b in zip(series, series[1:]) if b < a)
                if drops == 0:
                    monotone += 1
                    strict += all(b > a for a, b in zip(series, series[1:]))
                else:
                    worst.append((drops, prompt, position, block))
    print("   %d of %d block-positions rise without a single drop (%d strictly)"
          % (monotone, checked, strict))
    worst.sort(reverse=True)
    for drops, prompt, position, block in worst[:8]:
        print("     %s position %d block %d: %d drops" % (prompt, position, block, drops))

    print("\n== claim 5: routing is not organised by block offset ==")
    print("   median router spread by offset, one column per prompt")
    print("   %-7s %s" % ("offset", "".join("%9s" % p[:8] for p in prompts)))
    columns = {}
    for prompt in prompts:
        by_offset = collections.defaultdict(list)
        for row in rows:
            if row["prompt"] == prompt and not math.isnan(row["spread"]):
                by_offset[row["offset"]].append(row["spread"])
        columns[prompt] = {o: median(v) for o, v in by_offset.items()}
    for offset in range(1, BLOCK + 1):
        cells = "".join("%9.2f" % columns[p][offset] if offset in columns[p] else "%9s" % "-"
                        for p in prompts)
        print("   %-7d %s" % (offset, cells))
    print("\n   if routing were offset-organised, each prompt would rank the offsets alike.")
    ranks = {}
    for prompt in prompts:
        order = sorted(columns[prompt], key=lambda o: -columns[prompt][o])
        ranks[prompt] = {o: i for i, o in enumerate(order)}
    pairs = []
    for i, a in enumerate(prompts):
        for b in prompts[i + 1:]:
            common = set(ranks[a]) & set(ranks[b])
            if len(common) < 4:
                continue
            n = len(common)
            diff = sum((ranks[a][o] - ranks[b][o]) ** 2 for o in common)
            pairs.append(1 - 6.0 * diff / (n * (n * n - 1)))
    if pairs:
        print("   rank correlation of the offset ordering between prompt pairs:")
        print("     %d pairs, min %.3f, median %.3f, max %.3f"
              % (len(pairs), min(pairs), median(pairs), max(pairs)))

    print("\n== for contrast: the same rank correlation for the FOLD ==")
    fold_columns = {}
    for prompt in prompts:
        by_offset = collections.defaultdict(list)
        for row in rows:
            if row["prompt"] == prompt and row["offset"] != BLOCK \
                    and not math.isnan(row["rel_fold"]):
                by_offset[row["offset"]].append(row["rel_fold"])
        fold_columns[prompt] = {o: median(v) for o, v in by_offset.items()}
    fold_ranks = {p: {o: i for i, o in enumerate(sorted(c, key=lambda o: -c[o]))}
                  for p, c in fold_columns.items()}
    pairs = []
    for i, a in enumerate(prompts):
        for b in prompts[i + 1:]:
            common = set(fold_ranks[a]) & set(fold_ranks[b])
            if len(common) < 4:
                continue
            n = len(common)
            diff = sum((fold_ranks[a][o] - fold_ranks[b][o]) ** 2 for o in common)
            pairs.append(1 - 6.0 * diff / (n * (n * n - 1)))
    if pairs:
        print("     %d pairs, min %.3f, median %.3f, max %.3f"
              % (len(pairs), min(pairs), median(pairs), max(pairs)))

    print("\n== attention movement, MLA against KDA ==")
    for kind in ("M", "K"):
        values = [r["rel_attn"] for r in rows
                  if r["attn"] == kind and not math.isnan(r["rel_attn"])]
        if values:
            ordered = sorted(values)
            print("   %s  n=%-6d min %.3f  p25 %.3f  median %.3f  p75 %.3f  max %.3f"
                  % (kind, len(values), ordered[0], ordered[len(ordered) // 4],
                     median(values), ordered[3 * len(ordered) // 4], ordered[-1]))

    if len(sys.argv) > 2:
        print("\n== claim 6: the per-position arithmetic count, from the trace ==")
        counts = collections.defaultdict(dict)
        with open(sys.argv[2]) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    counts[parts[0]][int(parts[1])] = int(parts[2])
        print("   %-10s %8s %8s %8s %8s %8s %10s"
              % ("prompt", "l2", "l6", "l9", "l20", "l116", "arithmetic"))
        for prompt in sorted(counts):
            row = counts[prompt]
            arithmetic = row.get(6, 0) + row.get(9, 0) + row.get(20, 0) + row.get(116, 0)
            print("   %-10s %8d %8d %8d %8d %8d %10d"
                  % (prompt, row.get(2, 0), row.get(6, 0), row.get(9, 0),
                     row.get(20, 0), row.get(116, 0), arithmetic))
        print("   (counts are summed over every position of that prompt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
