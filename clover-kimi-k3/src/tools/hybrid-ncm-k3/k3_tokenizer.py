"""Build and wrap Kimi K3's exact C tokenizer (k3_tok.h) as a Python module (k3_tokenizer.py) via ctypes.
Guarantees 100% bit-identical token IDs (0..163,839) to Clover Kimi K3's C/CUDA engine (`whole-model` / `whole-model-gpu`).
"""
import ctypes, os, subprocess

SO_PATH = "/home/usergpu/merge_lf/libk3tok.so"
C_SRC = r"""
#define _GNU_SOURCE
#include "k3_tok.h"

static Tok g_tok;
static int g_loaded = 0;

int k3tok_init(const char *dir) {
    if (!g_loaded) {
        k3_tok_load(&g_tok, dir);
        g_loaded = 1;
    }
    return K3_TOK_VOCAB;
}

int k3tok_encode(const char *text, int len, int *out_ids, int cap) {
    return tok_encode_mode(&g_tok, text, len, out_ids, cap, 0);
}

int k3tok_decode(const int *ids, int n, char *out_buf, int cap) {
    return tok_decode(&g_tok, ids, n, out_buf, cap);
}
"""

def ensure_lib():
    if os.path.exists(SO_PATH):
        return SO_PATH
    c_file = "/home/usergpu/merge_lf/k3tok_wrap.c"
    with open(c_file, "w", encoding="utf-8") as f:
        f.write(C_SRC)
    inc1 = "/opt/clover-k3/clover-kimi-k3/src/code/third-party/kimi-k3-in-c/src/tokenizer"
    inc2 = "/opt/clover-k3/clover-kimi-k3/src/code/third-party/kimi-k3-in-c/third_party"
    cmd = ["gcc", "-O3", "-shared", "-fPIC", f"-I{inc1}", f"-I{inc2}", c_file, "-o", SO_PATH]
    subprocess.run(cmd, check=True)
    return SO_PATH


class K3Tokenizer:
    def __init__(self, weights_dir="/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights"):
        so = ensure_lib()
        self.lib = ctypes.CDLL(so)
        self.lib.k3tok_init.argtypes = [ctypes.c_char_p]
        self.lib.k3tok_init.restype = ctypes.c_int
        self.lib.k3tok_encode.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.k3tok_encode.restype = ctypes.c_int
        self.lib.k3tok_decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.k3tok_decode.restype = ctypes.c_int

        self.vocab_size = self.lib.k3tok_init(weights_dir.encode("utf-8"))
        self.eos_token_id = 163586
        self.pad_token_id = 163585

    def encode(self, text: str):
        raw = text.encode("utf-8")
        cap = len(raw) + 16
        buf = (ctypes.c_int * cap)()
        n = self.lib.k3tok_encode(raw, len(raw), buf, cap)
        return list(buf[:n])

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [int(i) for i in ids if 0 <= int(i) < 163584]
        else:
            ids = [int(i) for i in ids if 0 <= int(i) < self.vocab_size]
        if not ids:
            return ""
        n = len(ids)
        arr = (ctypes.c_int * n)(*ids)
        cap = n * 32 + 64
        out = ctypes.create_string_buffer(cap)
        bl = self.lib.k3tok_decode(arr, n, out, cap)
        if bl <= 0:
            return ""
        return out.raw[:bl].decode("utf-8", errors="replace")


if __name__ == "__main__":
    tk = K3Tokenizer()
    for fn in ("p1b.ids", "p2.ids", "p3.ids", "p4.ids"):
        ids = [int(x) for x in open(f"/home/usergpu/k3gpu/results/{fn}").read().replace(",", " ").split() if int(x) < 163584]
        txt = tk.decode(ids)
        re_ids = tk.encode(txt)
        print(f"{fn}: len={len(ids)} re_len={len(re_ids)} 100%_bit_identical={ids == re_ids}")
