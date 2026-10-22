"""Plan writer (text mode): HybridAgent-8B-Agentic's Qwen3 weights, used as a plain chat model WITHOUT the HybridLink adapter (no latent steps).
Reads pairs.jsonl ({"ids": [128 BPE ids]}), decodes each story window with the TinyStories BPE, asks the model for a one-line plan, writes
{"ids", "plan", "story", "raw"} lines. Greedy decoding. Not the real run's data until reviewed.
Usage: python plan_writer.py --pairs-in IN.jsonl --out OUT.jsonl --n 8 --device cpu --max-new 64 [--bs 4]"""
import argparse, json, os, sys, time, re
ROOT = os.environ.get("NCM_REPO", "/root/ncm-cpu/repo")
ART = os.environ.get("NCM_ART", "/root/ncm-cpu/artifacts/tinystories")
sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from tokenization.bpe_tokenizer import BPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--pairs-in", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--model", default="togethercomputer/HybridAgent-8B-Agentic")
a = ap.parse_args()

CFG = OmegaConf.load("config/eval/tinystories.yaml")
pt = dict(CFG.tokenizer.pre_tokenizer)
pt["vocab_file_path"] = os.path.join(ART, CFG.checkpoints.bpe_vocab)
pt["merges_file_path"] = os.path.join(ART, CFG.checkpoints.bpe_merges)
bpe = BPETokenizer.from_config(OmegaConf.create(pt))

PROMPT = ("Read this story opening and write a plan for it in one line: the characters (with names), the setting, and the events in order, "
          "as short phrases separated by semicolons. Use at most 35 words. Use only what the text says: do not add details, and do not guess what happens after the text ends. Output only the plan line.\n\n"
          "Example story: Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house. One day, her mom asked her to wipe the table after lunch. Sue was happy to help.\n"
          "Example plan: thoughtful girl Sue; at home with her mom; Sue loves helping; mom asks her to wipe the table after lunch; Sue is happy to help\n\n"
          "Story:\n{story}\n\nPlan:")
PREFILL = "Plan:"
MAXW = 40

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
rows = [json.loads(l) for l in open(a.pairs_in, encoding="utf-8") if l.strip()][a.skip:a.skip + a.n][a.shard::a.nshards]
out = open(a.out, "w", encoding="utf-8")
t0 = time.perf_counter()
for i in range(0, len(rows), a.bs):
    chunk = rows[i:i + a.bs]
    stories = [bpe.decode(r["ids"]).strip() for r in chunk]
    prompts = []
    for s in stories:
        msgs = [{"role": "user", "content": PROMPT.format(story=s)}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(p + PREFILL)
    if i == 0:
        print("PROMPT TAIL:", repr(prompts[0][-200:]), flush=True)
    enc = tok(prompts, return_tensors="pt", padding=True).to(a.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
    for r, s, o in zip(chunk, stories, g[:, enc["input_ids"].shape[1]:]):
        raw = tok.decode(o, skip_special_tokens=True)
        plan = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip().split("\n")[0].strip()
        plan = re.sub(r"^Plan:\s*", "", plan)
        words = plan.split()
        capped = len(words) > MAXW
        plan = " ".join(words[:MAXW])
        if capped and ";" in plan[:-1]:
            plan = plan[:plan.rindex(";")].strip()
        bad = bool(re.search(r"\b(the user|let me|okay,|now, write|write the story)\b", plan.lower())) or len(plan) < 15 or "(" in plan and ")" not in plan
        out.write(json.dumps({"ids": r["ids"], "plan": plan, "story": s, "raw": raw, "capped": capped, "bad": bad}, ensure_ascii=False) + "\n"); out.flush()
        print(f"[{i}] PLAN: {plan}", flush=True)
    print(f"  {i + len(chunk)}/{len(rows)}  {time.perf_counter() - t0:.0f}s", flush=True)
print("done")
