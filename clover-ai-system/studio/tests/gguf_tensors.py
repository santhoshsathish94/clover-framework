"""List what a GGUF file holds (tensor name groups, architecture keys) without the gguf package.
Usage: python gguf_tensors.py file.gguf [...]
"""
import struct
import sys
from collections import Counter

SIMPLE = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def string(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def value(f, t):
    if t in SIMPLE:
        return read(f, SIMPLE[t])
    if t == 8:
        return string(f)
    if t == 9:
        et, n = read(f, "I"), read(f, "Q")
        vals = [value(f, et) for _ in range(n)]
        return vals if n <= 8 else f"[{n} items]"
    raise ValueError(f"unknown GGUF value type {t}")


def show(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF file"
        ver, n_t, n_kv = read(f, "I"), read(f, "Q"), read(f, "Q")
        kv = {}
        for _ in range(n_kv):
            k = string(f)
            kv[k] = value(f, read(f, "I"))
        groups, names = Counter(), []
        for _ in range(n_t):
            name = string(f)
            nd = read(f, "I")
            dims = [read(f, "Q") for _ in range(nd)]
            read(f, "I"), read(f, "Q")
            names.append((name, dims))
            parts = name.split(".")
            groups[".".join(parts[:2]) if parts[0] in ("blk", "v", "a", "mm") and len(parts) > 2 and not parts[1].isdigit()
                   else (parts[0] + ".N" if len(parts) > 1 and parts[1].isdigit() else parts[0])] += 1
    print(f"== {path}\nGGUF v{ver}: {n_t} tensors, {n_kv} keys")
    for k, v in kv.items():
        if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            continue
        print(f"  {k} = {str(v)[:120]}")
    print("  tensor groups:", dict(sorted(groups.items())))
    for name, dims in names:
        if not any(p.isdigit() and p != "0" for p in name.split(".")):  # one layer of each stack, plus the rest
            print(f"    {name} {dims}")


for p in sys.argv[1:]:
    show(p)
