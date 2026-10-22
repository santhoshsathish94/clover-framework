"""Build the blind human-rating set: 20 validation plans x {true story, E3 s=0, E3 s=1, E3 s=4}, balanced chunks, plus 4 repeats.
Outputs human/rate.html (give this to the rater) and human/key.json (system of every item; do not open before rating)."""
import json, os, random, html

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "human")
os.makedirs(OUT, exist_ok=True)
d = json.load(open(os.path.join(HERE, "gpu-out", "data37.json"), encoding="utf-8"))
true = json.load(open(os.path.join(HERE, "gpu-out", "true_valid.json"), encoding="utf-8"))
systems = {"true": true, "s0": d["e3_s0"], "s1": d["e3_s1"], "s4": d["e3_s4"]}
rng = random.Random(37)
ok = [i for i in range(len(d["plans"])) if all(len(systems[k][i].split()) >= 40 for k in systems)]
plans = sorted(rng.sample(ok, 20))
rng.shuffle(plans)
items, key = [], {}
n = 0
for c in range(5):
    chunk = []
    for p in plans[c * 4:(c + 1) * 4]:
        for s in systems:
            chunk.append((p, s, False))
    if c >= 1:  # repeats of earlier items, spread over chunks 1..5
        pool = [(p, s) for p in plans[: c * 4] for s in systems]
        for p, s in rng.sample(pool, 1 if c < 5 else 2):
            chunk.append((p, s, True))
    rng.shuffle(chunk)
    for p, s, rep in chunk:
        n += 1
        iid = f"r{n:03d}"
        items.append({"id": iid, "chunk": c + 1, "plan": d["plans"][p], "story": systems[s][p]})
        key[iid] = {"plan_idx": p, "system": s, "repeat": rep}
json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)
payload = json.dumps(items).replace("<", "\\u003c")

page = """<!doctype html><meta charset="utf-8"><title>Story rating</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:24px auto;padding:0 12px;line-height:1.45}
.box{border:1px solid #bbb;border-radius:6px;padding:12px;margin:10px 0;background:#fafafa}.plan{background:#eef3fb}
.story{white-space:pre-wrap;font-size:17px}label{margin-right:14px;cursor:pointer}button{padding:6px 14px;margin-right:8px}
small{color:#555}</style>
<h2>Story rating</h2>
<p><small>Each item is a short children's story written for a story plan. Rate it on two separate 1-5 scales. Rate each item on its own;
do not try to guess where it came from. You can stop at any time (progress is saved in this browser; use Download at the end or whenever you stop).</small></p>
<div class="box"><b>Fluency</b> (is it clean English?): 1 = many garbled or made-up words, doubled words, or repeated phrases; 3 = readable with several glitches;
5 = reads like a clean story, no glitches.<br><b>Plan</b> (does it follow the plan?): 1 = unrelated to the plan; 3 = some plan elements, others missing or wrong order; 5 = covers the plan's events in order.</div>
<div id="app"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const items = JSON.parse(document.getElementById('data').textContent);
const KEY = 'story_rating_v1';
let state = JSON.parse(localStorage.getItem(KEY) || '{"i":0,"r":{}}');
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function esc(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function radios(name, cur){ let h=''; for(let v=1; v<=5; v++) h += '<label><input type="radio" name="'+name+'" value="'+v+'"'+(cur==v?' checked':'')+'> '+v+'</label>'; return h; }
function render(){
  const it = items[state.i], r = state.r[it.id] || {};
  const done = Object.values(state.r).filter(x => x.f && x.p).length;
  document.getElementById('app').innerHTML =
   '<p><b>Item '+(state.i+1)+' of '+items.length+'</b> (chunk '+it.chunk+' of 5; rated '+done+')</p>'+
   '<div class="box plan"><b>Plan:</b> '+esc(it.plan)+'</div>'+
   '<div class="box story">'+esc(it.story)+'</div>'+
   '<div class="box"><b>Fluency</b> '+radios('f', r.f)+'<br><b>Plan</b> '+radios('p', r.p)+'<br>'+
   '<input id="c" placeholder="optional comment" style="width:95%;margin-top:6px" value="'+esc(r.c||'')+'"></div>'+
   '<button onclick="go(-1)">Back</button><button onclick="go(1)">Next</button><button onclick="dl()">Download ratings.csv</button>'+
   '<p><small>Chunks 1-5 each contain every kind of item equally, so stopping after a whole chunk keeps the comparison balanced.</small></p>';
}
function cap(){
  const it = items[state.i], r = state.r[it.id] || {};
  const f = document.querySelector('input[name=f]:checked'), p = document.querySelector('input[name=p]:checked');
  if(f) r.f = +f.value; if(p) r.p = +p.value; r.c = document.getElementById('c').value; state.r[it.id] = r; save();
}
function go(d){ cap(); state.i = Math.min(items.length-1, Math.max(0, state.i+d)); save(); render(); window.scrollTo(0,0); }
function dl(){ cap();
  let rows = ['id,fluency,plan,comment'];
  for(const it of items){ const r = state.r[it.id]; if(r && r.f && r.p) rows.push([it.id, r.f, r.p, '"'+(r.c||'').replace(/"/g,'""')+'"'].join(',')); }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.join('\\n')], {type:'text/csv'})); a.download = 'ratings.csv'; a.click(); }
render();
</script>""".replace("__DATA__", payload)
open(os.path.join(OUT, "rate.html"), "w", encoding="utf-8").write(page)
print(len(items), "items;", len(set(v["plan_idx"] for v in key.values())), "plans;", sum(v["repeat"] for v in key.values()), "repeats")
