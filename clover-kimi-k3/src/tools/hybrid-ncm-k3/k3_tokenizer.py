"""Build and wrap Kimi K3's exact C tokenizer (k3_tok.h) as a Python module (k3_tokenizer.py) via ctypes.
Guarantees 100% bit-identical token IDs (0..163,839) to Clover Kimi K3's C/CUDA engine (`whole-model` / `whole-model-gpu`).
"""
import ctypes, os, subprocess
from clover_paths import TOOLS_DIR, ENGINE_DIR, find_open_weights_dir

SO_PATH = os.environ.get("CLOVER_K3TOK_SO") or os.path.join(TOOLS_DIR, "libk3tok.so")
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
    return tok_encode(&g_tok, text, len, out_ids, cap);
}

int k3tok_encode_plain(const char *text, int len, int *out_ids, int cap) {
    return tok_encode_mode(&g_tok, text, len, out_ids, cap, 0);
}

int k3tok_id_of(const char *name) {
    return tok_id_of(&g_tok, name);
}

int k3tok_decode(const int *ids, int n, char *out_buf, int cap) {
    return tok_decode(&g_tok, ids, n, out_buf, cap);
}
"""

def ensure_lib():
    c_file = os.path.join(os.path.dirname(SO_PATH), "k3tok_wrap.c")
    need_build = not os.path.exists(SO_PATH)
    if not need_build and os.path.exists(c_file):
        with open(c_file, "r", encoding="utf-8") as f:
            if "k3tok_encode_plain" not in f.read():
                need_build = True
    elif not os.path.exists(c_file) and need_build:
        need_build = True
    if not need_build:
        return SO_PATH
    with open(c_file, "w", encoding="utf-8") as f:
        f.write(C_SRC)
    tok_base = os.path.join(ENGINE_DIR, "src/code/third-party/kimi-k3-in-c")
    if not os.path.isdir(tok_base) and os.path.isdir("/opt/clover-k3/clover-kimi-k3/src/code/third-party/kimi-k3-in-c"):
        tok_base = "/opt/clover-k3/clover-kimi-k3/src/code/third-party/kimi-k3-in-c"
    inc1 = os.path.join(tok_base, "src/tokenizer")
    inc2 = os.path.join(tok_base, "third_party")
    cmd = ["gcc", "-O3", "-shared", "-fPIC", f"-I{inc1}", f"-I{inc2}", c_file, "-o", SO_PATH]
    subprocess.run(cmd, check=True)
    return SO_PATH


class K3Tokenizer:
    def __init__(self, weights_dir: str | None = None):
        if weights_dir is None:
            weights_dir = find_open_weights_dir()
        so = ensure_lib()
        self.lib = ctypes.CDLL(so)
        self.lib.k3tok_init.argtypes = [ctypes.c_char_p]
        self.lib.k3tok_init.restype = ctypes.c_int
        self.lib.k3tok_encode.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.k3tok_encode.restype = ctypes.c_int
        self.lib.k3tok_encode_plain.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.k3tok_encode_plain.restype = ctypes.c_int
        self.lib.k3tok_id_of.argtypes = [ctypes.c_char_p]
        self.lib.k3tok_id_of.restype = ctypes.c_int
        self.lib.k3tok_decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.k3tok_decode.restype = ctypes.c_int

        self.vocab_size = self.lib.k3tok_init(weights_dir.encode("utf-8"))
        self.bos_token_id = 163584
        self.eos_token_id = 163585
        self.eom_token_id = 163586
        self.open_token_id = 163587
        self.close_token_id = 163588
        self.sep_token_id = 163589
        self.pad_token_id = 163839

    def encode(self, text: str, allow_special: bool = True):
        raw = text.encode("utf-8")
        cap = len(raw) + 16
        buf = (ctypes.c_int * cap)()
        if allow_special:
            n = self.lib.k3tok_encode(raw, len(raw), buf, cap)
        else:
            n = self.lib.k3tok_encode_plain(raw, len(raw), buf, cap)
        return list(buf[:n])

    def encode_plain(self, text: str):
        return self.encode(text, allow_special=False)

    def _chat_message_ids(self, attrs: str, content: str) -> list[int]:
        ids = [self.open_token_id]
        ids.extend(self.encode_plain("message"))
        ids.extend(self.encode_plain(attrs))
        ids.append(self.sep_token_id)
        if content:
            ids.extend(self.encode_plain(content))
        ids.append(self.close_token_id)
        ids.extend(self.encode_plain("message"))
        ids.append(self.sep_token_id)
        ids.append(self.eom_token_id)
        return ids

    def encode_chat_prompt(self, user_text: str, system_text: str | None = None, history: list[dict] | None = None, think: bool = False) -> list[int]:
        """Encodes a chat turn using Kimi K3's exact XTML chat format (`whole-model.c` `prompt_ids`)."""
        ids: list[int] = []
        if system_text:
            ids.extend(self._chat_message_ids(' role="system"', system_text))
        if history:
            for turn in history:
                u = (turn.get("user") or "").strip()
                a = (turn.get("assistant") or "").strip()
                if u:
                    ids.extend(self._chat_message_ids(' role="user"', u))
                if a:
                    ids.extend([self.open_token_id] + self.encode_plain("message") + self.encode_plain(' role="assistant"') + [self.sep_token_id])
                    ids.extend([self.open_token_id] + self.encode_plain("response") + [self.sep_token_id])
                    ids.extend(self.encode_plain(a))
                    ids.extend([self.close_token_id] + self.encode_plain("response") + [self.sep_token_id])
                    ids.extend([self.close_token_id] + self.encode_plain("message") + [self.sep_token_id, self.eom_token_id])
        ids.extend(self._chat_message_ids(' role="user"', user_text.rstrip(" \t")))
        ids.extend([self.open_token_id] + self.encode_plain("message") + self.encode_plain(' role="assistant"') + [self.sep_token_id])
        ids.extend([self.open_token_id] + self.encode_plain("think" if think else "response") + [self.sep_token_id])
        return ids

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
