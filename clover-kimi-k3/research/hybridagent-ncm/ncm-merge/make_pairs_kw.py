"""SMOKE pairs (k3): keyword plans extracted from the story itself, to test the stage-2 code path. NOT the plan source of the real run.
Window = first 128 BPE tokens of a story with >= 128 tokens (same windows as the gate). Plan = first N distinct content words of the window in order
(len >= 4, not in the stop list), comma separated."""
import json, os, re, sys
ROOT = "/root/ncm-cpu/repo"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
N = int(sys.argv[1]); NW = int(sys.argv[2]); out = sys.argv[3]
ART = "/root/ncm-cpu/artifacts/tinystories"
CFG = OmegaConf.load("config/eval/tinystories.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ART, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    CFG.tokenizer.pre_tokenizer[key] = os.path.join(ART, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some day once upon time little".split())
txt = open("/root/ncm-cpu/data/TinyStories-valid.txt", encoding="utf-8").read()
stories = [s.strip() for s in txt.split("<|endoftext|>") if s.strip()]
n = 0
with open(out, "w", encoding="utf-8") as f:
    for s in stories:
        ids = tk.pre_tokenizer.encode(s)
        if len(ids) < 128:
            continue
        w = ids[:128]
        text = tk.pre_tokenizer.decode(w)
        words = [x for x in dict.fromkeys(re.findall(r"[a-z]+", text.lower())) if len(x) >= 4 and x not in STOP][:NW]
        f.write(json.dumps({"ids": w, "plan": ", ".join(words)}, ensure_ascii=False) + "\n")
        n += 1
        if n >= N:
            break
print("pairs", n, "->", out)
