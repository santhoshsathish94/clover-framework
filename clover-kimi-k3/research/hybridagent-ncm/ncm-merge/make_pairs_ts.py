"""Story windows from a TinyStories text file: first 128 BPE tokens of each story that has at least 128 tokens, in file order.
Usage: python make_pairs_ts.py TXT N OUT.jsonl [SKIP_STORIES]   -> lines {"ids": [128 ids]}"""
import json, os, sys
ROOT = os.environ.get("NCM_REPO", "/root/ncm-cpu/repo")
ART = os.environ.get("NCM_ART", "/root/ncm-cpu/artifacts/tinystories")
sys.path.insert(0, ROOT); os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.bpe_tokenizer import BPETokenizer
txt, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
CFG = OmegaConf.load("config/eval/tinystories.yaml")
pt = dict(CFG.tokenizer.pre_tokenizer)
pt["vocab_file_path"] = os.path.join(ART, CFG.checkpoints.bpe_vocab)
pt["merges_file_path"] = os.path.join(ART, CFG.checkpoints.bpe_merges)
bpe = BPETokenizer.from_config(OmegaConf.create(pt))
stories = open(txt, encoding="utf-8").read().split("<|endoftext|>")
n = seen = 0
with open(out, "w", encoding="utf-8") as f:
    for s in stories:
        s = s.strip()
        if not s:
            continue
        seen += 1
        if seen <= skip:
            continue
        ids = bpe.encode(s[:1500])
        if len(ids) < 128:
            continue
        f.write(json.dumps({"ids": ids[:128]}) + "\n")
        n += 1
        if n >= N:
            break
print("windows", n, "from stories seen", seen, "->", out)
