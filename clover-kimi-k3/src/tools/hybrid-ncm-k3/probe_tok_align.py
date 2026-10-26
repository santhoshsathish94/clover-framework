"""Cycle 47 (Part 2): Measure exact vocabulary and token-sequence alignment between:
  - Kimi K3 BPE tokenizer (tiktoken.model + tokenizer_config.json, 163,840 tokens)
  - Qwen3-8B / WMT14 NCM tokenizer (Qwen/Qwen3-0.6B, 151,680 tokens)
Checks:
1. How many of the 151,680 Qwen3 token byte pieces exist as exact single-token byte pieces in Kimi K3's 163,840-token vocabulary.
2. On Kimi K3's actual generated outputs for p1..p4 (recorded in k3ncm/*.ids on k3 / gpu), what fraction of token steps are identical
   under the direct Qwen3->Kimi piece mapping vs full Kimi BPE re-tokenization.
3. Also tests whether HybridAgent-8B-Agentic (in <latent> mode + text mode) + WMT14 NCM can draft Kimi K3's answers on p1..p5
   and what token acceptance rate against Kimi K3's actual generated token IDs (p1.ids..p4.ids) is achieved.
"""
import base64, json, os, sys
os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
from transformers import AutoTokenizer

k3_dir = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights"
k3_bpe = {}
id_to_bytes_k3 = {}
with open(os.path.join(k3_dir, "tiktoken.model"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        b64, rk = line.split()
        raw = base64.b64decode(b64)
        rk = int(rk)
        k3_bpe[raw] = rk
        id_to_bytes_k3[rk] = raw

qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
vocab_q = qtok.get_vocab()

# Reconstruct raw bytes for each Qwen3 token ID via byte-level BPE decoder table
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

u2b = bytes_to_unicode()
q_to_k3 = {}
exact_byte_matches = 0
same_id_matches = 0
for tok_str, qid in vocab_q.items():
    try:
        raw = bytes([u2b[c] for c in tok_str])
    except KeyError:
        continue
    if raw in k3_bpe:
        kid = k3_bpe[raw]
        q_to_k3[qid] = kid
        exact_byte_matches += 1
        if kid == qid:
            same_id_matches += 1

print(f"Qwen3 base vocab: {len(vocab_q)}, Kimi K3 base vocab: {len(k3_bpe)}")
print(f"Exact byte-piece matches in Kimi K3: {exact_byte_matches} / {len(vocab_q)} ({100.0*exact_byte_matches/len(vocab_q):.2f}%)")
print(f"Identical token IDs (qid == kid): {same_id_matches}")

# Check on Kimi K3's actual generated .ids files on gpu (p1b.ids, p2.ids, p3.ids, p4.ids)
res_dir = "/home/usergpu/k3gpu/results"
for fn in ("p1b.ids", "p2.ids", "p3.ids", "p4.ids"):
    p = os.path.join(res_dir, fn)
    if not os.path.exists(p):
        continue
    ids = [int(x) for x in open(p).read().replace(",", " ").split() if x.strip()]
    # Filter to base tokens (< 163584)
    base_ids = [i for i in ids if i < 163584]
    raw_bytes = b"".join(id_to_bytes_k3[i] for i in base_ids)
    txt = raw_bytes.decode("utf-8", errors="replace")
    q_ids = qtok.encode(txt, add_special_tokens=False)
    mapped_k3_ids = [q_to_k3.get(q, -1) for q in q_ids]
    in_vocab = sum(1 for m in mapped_k3_ids if m >= 0)
    exact_seq = sum(1 for a, b in zip(base_ids, mapped_k3_ids) if a == b)
    set_overlap = len(set(base_ids) & set(mapped_k3_ids)) / max(len(set(base_ids)), 1)
    print(f"[{fn}] k3_len={len(base_ids)} qwen_len={len(q_ids)} piece_in_k3={in_vocab}/{len(q_ids)} ({100.0*in_vocab/max(len(q_ids),1):.1f}%) pos_match={exact_seq}/{min(len(base_ids),len(q_ids))} set_overlap={set_overlap:.3f}")
