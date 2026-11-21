#!/usr/bin/env python3
"""Convert an original-format Variant-C export (block_trunklm24.bin) into BTD1 so the strict all-position
Variant-D verifier can check it too. usage: convert_c_to_d.py SRC_DIR DST_DIR EPS"""
import struct, sys
from pathlib import Path
import numpy as np

src, dst, eps = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
b = (src / "block_trunklm24.bin").read_bytes()
L, D, B, K, V, bp = struct.unpack_from("<IIIIII", b, 0)
o = 24
nb = D // B
def take(n, dt):
    global o
    a = np.frombuffer(b, dtype=dt, count=n, offset=o); o += a.nbytes; return a
out = bytearray(b"BTD1") + struct.pack("<IIIIIf", L, D, B, K, V, eps)
out += take(V * D, np.float32).tobytes() + take(D, np.float32).tobytes() + np.zeros(D, np.float32).tobytes()
ident = np.arange(D, dtype=np.uint16)
bfly = np.arange(D).reshape(8, 8, 8).transpose(1, 0, 2).reshape(D).astype(np.uint16)
ones = np.ones(D, np.float32).tobytes()
for _ in range(L):
    for _k in range(4):
        out += take(D, np.float32).tobytes()            # norm1, norm2, diag_scale, tw_s
    out += take(D * K, np.uint8).tobytes()
    for pi in range(3):
        perm = bfly if (pi == 2 and bp) else ident
        out += perm.tobytes() + ones + ones + take(nb * nb, np.float32).tobytes() + take(nb * nb * B, np.uint8).tobytes()
assert o == len(b), (o, len(b))
(dst / "block_trunklm24_d.bin").write_bytes(out)
(dst / "sample_d.bin").write_bytes((src / "sample_all.bin").read_bytes())
print("converted", L, D, B, K, V, "bp", bp, "eps", eps)
