"""QA gate (k3 CPU): can the released NCM tokenizers reconstruct reasoning / question-answer text, before any training on it?
Tokenizers (released, frozen, repo loading path unchanged): wmt14 (Qwen3-0.6B vocabulary, 64 tokens), xsum (GPT-2, 48), tinystories (BPE 8192, 128).
Text sets: GSM8K test questions and worked answers (<<calc>> annotations removed), plus each tokenizer's own domain as calibration:
WMT14 English references, XSum reference summaries, TinyStories validation. Every tokenizer is scored on every set.
Window = first L pre-tokens of a text; a text shorter than L gets EOS then PAD (the repo's own padding is PAD-only; EOS added so a text end is visible).
Scores per (tokenizer, set): token accuracy over text positions (before EOS), share of windows whose text positions are all exact,
recall of numbers (digit strings in the decoded original window found in the decoded reconstruction, as a multiset), and two examples."""
import argparse, collections, json, os, pickle, re, sys, time
ROOT = "/root/ncm-cpu/repo"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--tok", default="wmt14,xsum,tinystories")
ap.add_argument("--out", default="/root/ncm-cpu/gate-qa.json")
a = ap.parse_args()
torch.set_num_threads(16)
D = "/root/ncm-cpu"


def load_tok(task):
    cfg = OmegaConf.load(f"config/eval/{task}.yaml")
    ck = f"{D}/artifacts/{task}"
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    if task == "tinystories":
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.eval()
    return tk


def texts():
    out = {}
    g = [json.loads(l) for l in open(f"{D}/data/gsm8k/test.jsonl", encoding="utf-8")]
    out["gsm8k_question"] = [r["question"].strip() for r in g]
    out["gsm8k_answer"] = [re.sub(r"<<[^>]*>>", "", r["answer"]).strip() for r in g]
    # same answers in one line: newlines and the '#### N' marker are the tokens the first run lost
    out["gsm8k_answer_flat"] = [" ".join(re.sub(r"\n?#### *(.+)$", r"\nThe answer is \1.", s).split()) for s in out["gsm8k_answer"]]
    def first_str(x):
        return x if isinstance(x, str) else first_str(x[0])
    out["wmt14_en_ref"] = [first_str(x).strip() for x in pickle.load(open(f"{D}/data/wmt14/reference_translations.pickle", "rb"))]
    out["xsum_summary"] = [first_str(x).strip() for x in pickle.load(open(f"{D}/data/xsum/reference_summaries.pickle", "rb"))]
    ts = open(f"{D}/data/TinyStories-valid.txt", encoding="utf-8").read().split("<|endoftext|>")
    out["tinystories"] = [s.strip() for s in ts if s.strip()]
    return {k: v[:a.n] for k, v in out.items()}


NUM = re.compile(r"\d+(?:[.,]\d+)*")


def score(tk, items):
    L = tk.vqvae.context_length
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    X, ntext, full_len = [], [], []
    for t in items:
        ids = tk.pre_tokenizer.encode(t)
        full_len.append(len(ids))
        w = ids[:L]
        n = len(w)
        if n < L:
            w = w + [EOS] + [PAD] * (L - n - 1)
        X.append(w); ntext.append(n)
    X = torch.tensor(X)
    with torch.no_grad():
        idx = torch.cat([tk.vqvae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])
        R = torch.cat([tk.vqvae.decode(idx[i:i + 32]).argmax(-1) for i in range(0, len(idx), 32)])
    hit = tot = exact = 0
    nums_orig = nums_kept = 0
    ex = []
    for i, n in enumerate(ntext):
        eq = (R[i, :n] == X[i, :n])
        hit += int(eq.sum()); tot += n; exact += int(bool(eq.all()))
        o = tk.pre_tokenizer.decode(X[i, :n].tolist(), skip_special_tokens=True)
        r = tk.pre_tokenizer.decode(R[i, :n].tolist(), skip_special_tokens=True)
        co, cr = collections.Counter(NUM.findall(o)), collections.Counter(NUM.findall(r))
        nums_orig += sum(co.values()); nums_kept += sum((co & cr).values())
        if len(ex) < 2:
            ex.append({"orig": o, "recon": r})
    return {"windows": len(ntext), "L": L, "mean_pretokens": round(sum(full_len) / len(full_len), 1),
            "share_truncated": round(sum(1 for f in full_len if f > L) / len(full_len), 3),
            "token_acc": round(hit / max(tot, 1), 5), "window_exact": round(exact / len(ntext), 4),
            "numbers": nums_orig, "number_recall": round(nums_kept / max(nums_orig, 1), 4), "examples": ex}


T = texts()
print({k: len(v) for k, v in T.items()}, flush=True)
res = {"n": a.n, "results": {}}
for task in a.tok.split(","):
    t0 = time.perf_counter()
    tk = load_tok(task)
    print(f"== {task}: L={tk.vqvae.context_length} levels={list(tk.vqvae.vq.levels)} EOS={tk.pre_tokenizer.eos_token_id} PAD={tk.pre_tokenizer.pad_token_id}", flush=True)
    res["results"][task] = {}
    for name, items in T.items():
        s = score(tk, items)
        res["results"][task][name] = s
        print(f"{task:12s} {name:15s} acc {s['token_acc']:.4f} exact {s['window_exact']:.3f} numbers {s['numbers']:5d} recall {s['number_recall']:.3f} "
              f"mean_len {s['mean_pretokens']} trunc {s['share_truncated']}", flush=True)
    print(f"   ({time.perf_counter() - t0:.1f} s)", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("wrote", a.out)
