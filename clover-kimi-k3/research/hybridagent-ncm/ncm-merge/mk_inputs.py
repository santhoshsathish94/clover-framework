"""Build NCM inputs from existing HybridAgent traces (no new HybridAgent run). Variants per trace:
 think_head  visible reasoning (latent tags removed, tool calls and tool results removed), first 512 GPT-2 tokens (what the checkpoint reads as is)
 think_tail  the same text, last 500 GPT-2 tokens (so the final conclusion is inside the prefix)
 answer      the text inside <answer> ... </answer> only
Plus a control: the 'smoke1' news-style facts used earlier on the GPU server."""
import json, re
from transformers import AutoTokenizer

J = "/root/runs/hybrid/"
RUNS = {"q4": "trace-q4-search", "q5": "trace-q5-search", "q1": "trace-q1-search"}
tok = AutoTokenizer.from_pretrained("gpt2")


def clean(o):
    ans = re.search(r"<answer>(.*?)</answer>", o, re.S)
    body = o.split("<answer>")[0]
    for tag in ("python", "result", "search"):
        body = re.sub(rf"<{tag}>.*?</{tag}>", " ", body, flags=re.S)
    body = body.replace("<latent></latent>", " ")
    body = re.sub(r"</?think>", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    return body, (ans.group(1).strip() if ans else "")


items = [{"id": "control", "variant": "news_facts",
          "text": "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a cycle lane along Mill Road. Councillor Priya Raman proposed the plan. Work starts in March and should take 14 weeks. Shop owners on Mill Road, led by Tom Eliot, asked for the vote to be delayed because of parking. The council promised 40 temporary parking spaces on Dock Street."}]
for q, name in RUNS.items():
    o = json.load(open(J + name + ".json"))["output"]
    body, ans = clean(o)
    ids = tok(body)["input_ids"]
    items.append({"id": q, "variant": "think_head", "text": tok.decode(ids[:512])})
    items.append({"id": q, "variant": "think_tail", "text": tok.decode(ids[-500:])})
    items.append({"id": q, "variant": "answer", "text": ans})
    print(q, "cleaned reasoning", len(ids), "gpt2 tokens; answer", len(tok(ans)["input_ids"]), "tokens")
json.dump(items, open("/root/ncm-cpu/inputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", len(items), "inputs")
