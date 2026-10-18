"""How often does the same float value appear inside one real vector? usage: float-repeats.py FILE.f32 [WIDTH]
Reads rows of WIDTH float32 values (default 7168) and counts exact bit-pattern repeats."""
import sys
from array import array
from collections import Counter

path = sys.argv[1]
width = int(sys.argv[2]) if len(sys.argv) > 2 else 7168
raw = array('I')
with open(path, 'rb') as f:
    raw.fromfile(f, __import__('os').path.getsize(path) // 4)
rows = len(raw) // width
distinct, repeated_elems, zeros, top, distinct_abs = [], [], [], [], []
whole = set()
for r in range(rows):
    row = raw[r * width:(r + 1) * width]
    c = Counter(row)
    distinct.append(len(c))
    repeated_elems.append(sum(n for n in c.values() if n > 1))
    zeros.append(c.get(0, 0) + c.get(0x80000000, 0))
    top.append(c.most_common(1)[0][1])
    distinct_abs.append(len(Counter(v & 0x7fffffff for v in row)))
    whole.update(c)
n = float(rows)
print(f"{path}: {rows} vectors of {width} floats")
print(f"  distinct values per vector: mean {sum(distinct)/n:.1f} (min {min(distinct)}, max {max(distinct)})  = {100*sum(distinct)/n/width:.2f}% of the elements")
print(f"  elements whose exact value occurs more than once in their vector: {100*sum(repeated_elems)/n/width:.2f}%")
print(f"  distinct magnitudes (sign ignored) per vector: mean {sum(distinct_abs)/n:.1f}")
print(f"  exact zeros per vector: mean {sum(zeros)/n:.2f}; most repeated value occurs {max(top)} times in one vector")
print(f"  distinct values over all {rows*width} elements: {len(whole)}")
for levels in (16, 53, 256):
    mean_d = sum(distinct) / n
    print(f"  one matrix with {levels} weight values: {width}x{levels} multiplications per output row group -> at most {mean_d*levels:.0f} distinct products per vector "
          f"against {width} x rows of them (rows 3584: {100*mean_d*levels/(width*3584):.3f}% distinct)")
