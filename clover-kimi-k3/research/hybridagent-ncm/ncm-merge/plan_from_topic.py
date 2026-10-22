"""End-to-end front half: HybridAgent-8B (text mode) turns a short request into a one-line story plan, no story text given.
Writes {"topic", "plan", "raw", "bad"} lines; no "ids" (there is no true story). Usage: python plan_from_topic.py OUT.jsonl [device]"""
import json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
out, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
TOPICS = ["a girl who finds a lost puppy", "a boy who is afraid of the dark", "a robot that wants to learn to dance", "two friends who build a treehouse",
          "a little fish who wants to see the ocean", "a cat that cannot catch a mouse", "a rainy day and a red umbrella", "a bird that forgets how to fly",
          "a boy who plants a magic seed", "a bear who loses his honey", "a girl who bakes a cake for her grandma", "a dragon who is scared of fire",
          "a snowman who wants to visit summer", "a mouse who finds a big cheese", "a boy and his kite on a windy day", "a girl who learns to ride a bike",
          "a sleepy owl who wants to play in the day", "a little star that falls from the sky", "a dog who guards a garden", "a girl who finds a shiny key",
          "a boy who shares his toys at the park", "a turtle who is always late", "a rabbit who wants a carrot from the farm", "a girl who talks to a tree",
          "a boy who gets lost at the market", "a duck who cannot swim", "a clown who loses his nose", "a ladybug on a long trip",
          "a boy who paints the sky", "two ducks and a muddy puddle", "a girl and a talking teddy bear", "a fox who learns to say sorry",
          "a little car that is afraid of the highway", "a girl who wants to touch the moon", "a boy who finds a box in the attic", "a frog and a rainy pond",
          "a sad balloon that floats away", "a girl who loses her mitten in the snow", "a baby elephant who is too small", "a boy who opens a lemonade stand",
          "a horse who wants to race", "a girl who finds a bird with a hurt wing", "a bee who is too busy to rest", "a boy who learns to count",
          "a mouse who wants to go to the city", "a girl who has a secret garden", "a pirate who is afraid of water", "a boy who wants to catch a rainbow"]
PROMPT = ("Plan a short children's story about: {t}. Write the plan in one line: the characters (with names), the setting, and the events in order, "
          "as short phrases separated by semicolons. Use at most 35 words. Output only the plan line.\n\n"
          "Example request: a girl who helps her mom\nExample plan: thoughtful girl Sue; at home with her mom; Sue loves helping; mom asks her to wipe the table after lunch; Sue is happy to help\n\n"
          "Request: {t}\n\nPlan:")
tok = AutoTokenizer.from_pretrained("togethercomputer/HybridAgent-8B-Agentic", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16).to(device).eval()
prompts = []
for t in TOPICS:
    m = [{"role": "user", "content": PROMPT.format(t=t)}]
    try:
        p = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    prompts.append(p + "Plan:")
enc = tok(prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    g = model.generate(**enc, max_new_tokens=64, do_sample=False)
rows = []
for t, o in zip(TOPICS, g[:, enc["input_ids"].shape[1]:]):
    raw = tok.decode(o, skip_special_tokens=True)
    plan = re.sub(r"^Plan:\s*", "", raw.strip().split("\n")[0].strip())
    w = plan.split()
    if len(w) > 40:
        plan = " ".join(w[:40])
        if ";" in plan[:-1]:
            plan = plan[:plan.rindex(";")].strip()
    bad = bool(re.search(r"\b(the user|let me|okay,|write the story)\b|story ends|\.\.\.", plan.lower())) or len(plan) < 15
    rows.append({"topic": t, "plan": plan, "raw": raw, "bad": bad})
    print(t, "->", plan, flush=True)
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "bad", sum(r["bad"] for r in rows))
