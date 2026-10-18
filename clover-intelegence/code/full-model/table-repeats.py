"""Distinct 16-bit values in the embedding table (seed.bin) and the vocabulary head (fruit.bin), assuming 163,840 rows of 7,168 bf16.
usage: table-repeats.py FILE [ROWS_TO_SAMPLE]   (rows are taken evenly across the file)"""
import os, sys
import numpy as np

path = sys.argv[1]
sample = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
size = os.path.getsize(path)
width = 7168
rows = size // (width * 2)
print(f"{path}: {size} bytes = {rows} rows of {width} x 16-bit (remainder {size - rows * width * 2})")
m = np.memmap(path, dtype=np.uint16, mode='r', shape=(rows, width))
idx = np.linspace(0, rows - 1, min(sample, rows)).astype(int)
per_row, seen, repeating = [], np.zeros(65536, dtype=bool), 0
for i in idx:
    r = np.asarray(m[i])
    u, c = np.unique(r, return_counts=True)
    per_row.append(len(u))
    repeating += int(c[c > 1].sum())
    seen[u] = True
print(f"  sampled {len(idx)} rows: distinct 16-bit values per row mean {np.mean(per_row):.1f} ({100 * np.mean(per_row) / width:.2f}% of the row), "
      f"elements sharing a value inside their row {100 * repeating / (len(idx) * width):.2f}%")
print(f"  distinct 16-bit values over all sampled elements: {int(seen.sum())} of 65536 possible")
