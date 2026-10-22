"""Write the items (plan + story only, no system labels) from human/rate.html to ../rater_pack/items.txt for model raters."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
html = open(os.path.join(HERE, "human", "rate.html"), encoding="utf-8").read()
items = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
out = os.path.join(HERE, "..", "rater_pack")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "items.txt"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(f"=== ITEM {it['id']}\nPLAN: {it['plan']}\nSTORY:\n{it['story'].strip()}\n\n")
print(len(items), "items;", sum(len(i["story"].split()) for i in items) // len(items), "words on average")
