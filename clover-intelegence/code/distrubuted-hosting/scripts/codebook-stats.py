#!/usr/bin/env python3
"""How much repetition is there in the expert codes?

Layout per 64-row block, from root_project:
  codes     at  row * width/2        one byte per two columns
  selectors at  64*width/2 + row*width/32   one byte per 32 columns

A regrouping that accumulates x per code and multiplies once per distinct code only
pays if a group of 32 columns uses markedly fewer than 16 distinct code bytes. This
counts what is actually there. It proposes nothing.
"""
import collections
import struct
import sys

ROOT_RAW = 121856          # matrices 0 and 1, width 3584
ROOT_DOWN_RAW = 104448     # matrix 2, width 3072
ROOT_EXPERT_RAW = 17547264
PATH = "/opt/clover-k3/clover-intelegence/dataset/root-46/experts.direct"


def block_offset(expert, matrix, block):
    base = expert * ROOT_EXPERT_RAW
    if matrix < 2:
        return base + (matrix * 48 + block) * ROOT_RAW
    return base + 96 * ROOT_RAW + block * ROOT_DOWN_RAW


def main():
    samples = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (451, 0, 7), (818, 2, 13), (100, 1, 25)]
    print(f"{'expert':>7s} {'matrix':>7s} {'block':>6s} {'width':>6s} "
          f"{'distinct/group':>15s} {'distinct/row':>13s} {'distinct selectors':>19s}")

    overall_group = []
    with open(PATH, "rb") as handle:
        for expert, matrix, block in samples:
            width = 3072 if matrix == 2 else 3584
            size = ROOT_DOWN_RAW if matrix == 2 else ROOT_RAW
            handle.seek(block_offset(expert, matrix, block))
            data = handle.read(size)
            if len(data) != size:
                print("short read")
                return

            code_bytes = width // 2
            sel_bytes = width // 32
            per_group, per_row, per_sel = [], [], []
            for row in range(64):
                codes = data[row * code_bytes:(row + 1) * code_bytes]
                sel_start = 64 * code_bytes + row * sel_bytes
                selectors = data[sel_start:sel_start + sel_bytes]
                per_row.append(len(set(codes)))
                per_sel.append(len(set(selectors)))
                # one group is 32 columns = 16 code bytes
                for g in range(0, code_bytes, 16):
                    per_group.append(len(set(codes[g:g + 16])))
            overall_group.extend(per_group)
            print(f"{expert:7d} {matrix:7d} {block:6d} {width:6d} "
                  f"{sum(per_group)/len(per_group):15.2f} {sum(per_row)/len(per_row):13.1f} "
                  f"{sum(per_sel)/len(per_sel):19.1f}")

    print()
    hist = collections.Counter(overall_group)
    print("distinct code bytes within a 32-column group (16 bytes available)")
    for distinct in sorted(hist):
        share = hist[distinct] / len(overall_group)
        bar = "#" * int(share * 60)
        print(f"  {distinct:2d} distinct : {hist[distinct]:7d}  {share:6.3f} {bar}")
    mean = sum(overall_group) / len(overall_group)
    print()
    print(f"mean distinct per group : {mean:.3f} of 16")
    # Each code byte carries two weights, so a distinct byte still costs two multiplies.
    print(f"multiplies now          : 32 per group, one per column")
    print(f"multiplies if regrouped : {2 * mean:.2f} per group, plus ~32 adds to bucket")
    print(f"multiplies saved        : {32 - 2 * mean:.2f} per group, bought with ~32 adds")


if __name__ == "__main__":
    main()
