"""Minimal GGUF header reader: architecture, a few keys, and tensor names (no dependencies)."""
import struct
import sys

T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def read_val(f, t):
    if t == 8:
        return read_str(f)
    if t == 9:
        it, n = struct.unpack("<IQ", f.read(12))
        vals = [read_val(f, it) for _ in range(n)]
        return vals if n <= 8 else f"[{n} items]"
    fmt = T[t]
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
    kv = {}
    for _ in range(nkv):
        k = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = read_val(f, t)
    names = []
    for _ in range(nt):
        name = read_str(f)
        nd = struct.unpack("<I", f.read(4))[0]
        f.read(8 * nd + 4 + 8)
        names.append(name)
arch = kv.get("general.architecture")
print("version", ver, "tensors", nt, "architecture", arch)
for k, v in kv.items():
    if k.startswith(f"{arch}.") or k in ("general.name", "general.basename", "tokenizer.ggml.model"):
        print(" ", k, "=", v)
print("tensor names (sample):", sorted(names)[:4], "...", [n for n in names if "blk.2." in n][:12])
