"""Experiment 3a: dictionary layer v1 - choosing what a word means in a sentence (an OEWN synset), including meanings
never seen used, by reading their definitions. EXPERIMENTS.md has the pre-registration.
  prep       data from OEWN 2025 usage examples (splits by seed 3101) and SemCor 3.0 (natural text, test only)
  train      bi-encoder (context encoder: target-word vector; gloss encoder: [CLS]; dot product; cross-entropy)
  baselines  random, most-frequent sense, simplified Lesk, the untrained encoder
  report     criteria D1-D5 with paired bootstrap intervals
  (3a2: prep --semcor-train adds SemCor train/dev/test documents; train --train-splits train train_n; evaluate scores
   a saved model; report-usage gives criteria N1-N3)
Usage: python dict_meaning.py prep <data dir> --oewn oewn.xml.gz --semcor semcor.zip [--semcor-train]
       python dict_meaning.py train <data dir> <out json> --seed 1 [--shuffle-glosses] [--train-splits train train_n]
       python dict_meaning.py baselines <data dir> <out json>
       python dict_meaning.py evaluate <data dir> <out json> --model <saved model dir>
       python dict_meaning.py report <data dir> <out json> --runs r1.json ... --control c.json --baselines b.json
       python dict_meaning.py report-usage <data dir> <out json> --runs ... --v1 v1.json --v1-runs ... --baselines b.json"""
import argparse
import collections
import gzip
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
import zipfile

SEED = 3101
POS_GROUP = {"n": "n", "v": "v", "a": "a", "s": "a"}
TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
STOP = set("""a an the of to in on at for with by from as is was were be been are am it its this that these those and
or but not no he she they we you i his her their our your my me him them us who whom which what when where how there
here than then so if into out up down over under about after before while because one some any all each both very
can could will would shall should may might must do does did has have had having being such only also just more most
other another same own s""".split())


# ---------- prep ----------
def load_oewn(path):
    entries, synsets = {}, {}
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as fh:
        for _, el in ET.iterparse(fh):
            if el.tag == "LexicalEntry":
                lem = el.find("Lemma")
                entries[el.get("id")] = {"lemma": lem.get("writtenForm"), "pos": lem.get("partOfSpeech"),
                                         "senses": [(s.get("id"), s.get("synset")) for s in el.findall("Sense")],
                                         "forms": [f.get("writtenForm") for f in el.findall("Form")]}
                el.clear()
            elif el.tag == "Synset":
                d = el.find("Definition")
                synsets[el.get("id")] = {"pos": el.get("partOfSpeech"), "definition": d.text if d is not None else "",
                                         "examples": [e.text or "" for e in el.findall("Example")],
                                         "members": (el.get("members") or "").split()}
                el.clear()
    return entries, synsets


def inflections(lemma, pos):
    l = lemma
    out = {l}
    if pos == "n":
        out |= {l + "s", l + "es"} | ({l[:-1] + "ies"} if l.endswith("y") else set())
    elif pos == "v":
        out |= {l + "s", l + "es", l + "ed", l + "d", l + "ing", l + l[-1] + "ed", l + l[-1] + "ing"}
        out |= ({l[:-1] + "ing"} if l.endswith("e") else set()) | ({l[:-1] + "ied", l[:-1] + "ies"} if l.endswith("y") else set())
    elif pos == "a":
        out |= {l + "er", l + "est", l + "r", l + "st", l + l[-1] + "er", l + l[-1] + "est"}
        out |= {l[:-1] + "ier", l[:-1] + "iest"} if l.endswith("y") else set()
    return out


def prep(a):
    entries, synsets = load_oewn(a.oewn)
    lemma_synsets = collections.defaultdict(list)  # (lemma, pos group) -> synsets in sense order (MFS first)
    forms = collections.defaultdict(set)
    sense_synset = {}
    for eid, e in sorted(entries.items(), key=lambda kv: (kv[1]["pos"] == "s", kv[1]["lemma"] != (kv[1]["lemma"] or "").lower(),
                                                         kv[0])):  # head adjectives, then lower-case lemmas, first (MFS)
        g = POS_GROUP.get(e["pos"])
        for sid, syn in e["senses"]:
            sense_synset[sid.lower()] = syn
        if g is None or not re.fullmatch(r"[A-Za-z]+", e["lemma"] or ""):
            continue
        key = (e["lemma"].lower(), g)
        lemma_synsets[key] += [syn for _, syn in e["senses"] if syn not in lemma_synsets[key]]
        forms[key] |= inflections(key[0], g) | {f.lower() for f in e["forms"]}
    entry_lemma = {eid: e["lemma"].lower() for eid, e in entries.items() if e["lemma"]}
    items = []
    for syn_id, s in sorted(synsets.items()):
        g = POS_GROUP.get(s["pos"])
        if g is None:
            continue
        member_keys = [(entry_lemma[m], g) for m in s["members"] if m in entry_lemma and (entry_lemma[m], g) in forms]
        for k, ex in enumerate(s["examples"]):
            found = None
            for m in TOKEN.finditer(ex):
                w = m.group(0).lower()
                hit = next((key for key in member_keys if w in forms[key]), None)
                if hit:
                    found = (hit, m.start(), m.end())
                    break
            if not found or len(lemma_synsets[found[0]]) < 2:
                continue
            (lemma, pos), st, en = found
            items.append({"id": f"{syn_id}#{k}", "synset": syn_id, "lemma": lemma, "pos": pos, "text": ex,
                          "start": st, "end": en, "candidates": lemma_synsets[(lemma, pos)]})
    rng = random.Random(SEED)
    syns = sorted({it["synset"] for it in items})
    rng.shuffle(syns)
    z_test, z_dev = set(syns[:len(syns) // 10]), set(syns[len(syns) // 10:len(syns) * 15 // 100])
    lemmas = sorted({(it["lemma"], it["pos"]) for it in items if it["synset"] not in z_test | z_dev})
    rng.shuffle(lemmas)
    l_test = set(lemmas[:len(lemmas) // 20])
    for it in items:
        it["split"] = ("test_z" if it["synset"] in z_test else "dev_z" if it["synset"] in z_dev
                       else "test_l" if (it["lemma"], it["pos"]) in l_test else "train")
    rest = [it for it in items if it["split"] == "train"]
    left = collections.Counter(it["synset"] for it in rest)
    pool = [it for it in rest if left[it["synset"]] >= 2]
    rng.shuffle(pool)
    for split, frac in (("test_s", 0.10), ("dev_s", 0.05)):
        want, got = int(frac * len(rest)), 0
        for it in pool:
            if got >= want:
                break
            if it["split"] == "train" and left[it["synset"]] >= 2:
                it["split"] = split
                left[it["synset"]] -= 1
                got += 1
    held = z_test | z_dev
    for it in items:
        if it["split"] == "train":
            it["train_candidates"] = [c for c in it["candidates"] if c not in held]
    items = [it for it in items if it["split"] != "train" or len(it["train_candidates"]) >= 2]
    docs = semcor_docs(a.semcor)
    dev_docs = sorted(random.Random(SEED).sample(docs, 20))  # the 3a SemCor test documents
    if not a.semcor_train:
        items += semcor_items(a.semcor, dev_docs, "test_n", lemma_synsets, sense_synset)
    else:  # Experiment 3a2: the 3a documents become dev-N; 20 fresh test documents; the rest trains
        rest = [d for d in docs if d not in dev_docs]
        test_docs = sorted(random.Random(SEED + 1).sample(rest, 20))
        items += semcor_items(a.semcor, dev_docs, "dev_n", lemma_synsets, sense_synset)
        items += semcor_items(a.semcor, test_docs, "test_n2", lemma_synsets, sense_synset)
        train_n = semcor_items(a.semcor, [d for d in rest if d not in test_docs], "train_n", lemma_synsets, sense_synset)
        for it in train_n:
            it["train_candidates"] = [c for c in it["candidates"] if c not in held]
        items += [it for it in train_n if it["synset"] not in held and (it["lemma"], it["pos"]) not in l_test
                  and len(it["train_candidates"]) >= 2]
    os.makedirs(a.data, exist_ok=True)
    with open(os.path.join(a.data, "items.jsonl"), "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    needed = {c for it in items for c in it["candidates"]}
    json.dump({sid: {"definition": synsets[sid]["definition"], "examples": synsets[sid]["examples"]} for sid in sorted(needed)},
              open(os.path.join(a.data, "synsets.json"), "w"))
    stats = {"items_by_split": dict(collections.Counter(it["split"] for it in items)),
             "held_out_synsets": {"test_z": len(z_test), "dev_z": len(z_dev)}, "held_out_lemmas": len(l_test),
             "mean_candidates_by_split": {s: round(sum(len(it["candidates"]) for it in items if it["split"] == s)
                                                / max(1, sum(it["split"] == s for it in items)), 2)
                                          for s in sorted({it["split"] for it in items})},
             "synsets_with_definition": len(needed)}
    json.dump(stats, open(os.path.join(a.data, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def semcor_docs(path):
    return sorted(n for n in zipfile.ZipFile(path).namelist() if re.search(r"/brown[12]/tagfiles/br-.*\.xml$", n))


def semcor_items(path, chosen, split, lemma_synsets, sense_synset):
    z = zipfile.ZipFile(path)
    out, unmapped = [], collections.Counter()
    for name in chosen:
        root = ET.fromstring(z.read(name))
        for s in root.iter("s"):
            words, pos_list = [], []
            for w in s:
                words.append((w.text or "").replace("_", " "))
                pos_list.append(w)
            text, offs = "", []
            for t in words:
                if text:
                    text += " "
                offs.append((len(text), len(text) + len(t)))
                text += t
            for (st, en), w in zip(offs, pos_list):
                if w.tag != "wf" or w.get("cmd") != "done" or not w.get("lexsn") or not w.get("lemma"):
                    continue
                tag, lemma, lexsn = w.get("pos", ""), w.get("lemma"), w.get("lexsn")
                g = "n" if tag in ("NN", "NNS") else "v" if tag.startswith("VB") else "a" if tag.startswith("JJ") else None
                if g is None or ";" in lexsn or "_" in (w.text or "_") or not re.fullmatch(r"[A-Za-z]+", lemma):
                    continue
                syn = sense_synset.get(f"oewn-{lemma}__{lexsn.replace(':', '.')}".lower())
                cands = lemma_synsets.get((lemma.lower(), g), [])
                if syn is None or syn not in cands:
                    unmapped["no OEWN sense" if syn is None else "sense not among candidates"] += 1
                    continue
                if len(cands) < 2:
                    continue
                out.append({"id": f"{name.split('/')[-1]}:{s.get('snum')}:{st}", "synset": syn, "lemma": lemma.lower(),
                            "pos": g, "text": text, "start": st, "end": en, "candidates": cands, "split": split})
    print("semcor", split, "documents", len(chosen), chosen[:20], "items", len(out), "unmapped", dict(unmapped))
    return out


# ---------- shared ----------
def load_data(d):
    items = [json.loads(l) for l in open(os.path.join(d, "items.jsonl"), encoding="utf-8")]
    return items, json.load(open(os.path.join(d, "synsets.json")))


EVAL_SPLITS = ("dev_s", "dev_z", "dev_n", "test_s", "test_z", "test_l", "test_n", "test_n2")
DEV_SPLITS = ("dev_s", "dev_z", "dev_n")


class Encoders:
    def __init__(self, name, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        saved = os.path.isdir(os.path.join(name, "context"))  # a model saved by train --save
        self.tok = AutoTokenizer.from_pretrained(name)
        self.ctx = AutoModel.from_pretrained(os.path.join(name, "context") if saved else name).to(device)
        self.gls = AutoModel.from_pretrained(os.path.join(name, "gloss") if saved else name).to(device)

    def context_vectors(self, batch):
        enc = self.tok([it["text"] for it in batch], return_offsets_mapping=True, truncation=True, max_length=128,
                       padding=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        h = self.ctx(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state
        mask = self.torch.zeros(h.shape[:2], device=self.device)
        for i, it in enumerate(batch):
            sel = (offs[i, :, 0] < it["end"]) & (offs[i, :, 1] > it["start"]) & (offs[i, :, 1] > offs[i, :, 0])
            mask[i, sel.to(self.device)] = 1.0
            if mask[i].sum() == 0:
                mask[i, 0] = 1.0  # target cut off by truncation: fall back to [CLS] (counted, see stats)
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    def gloss_vectors(self, texts, pool="cls"):
        enc = self.tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt").to(self.device)
        h = self.gls(**enc).last_hidden_state
        if pool == "cls":
            return h[:, 0]
        m = enc["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)


def evaluate(enc, items, synsets, gloss_of, pool="cls", cosine=False, bs=64):
    torch = enc.torch
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        need = sorted({c for it in items if it["split"] in EVAL_SPLITS for c in it["candidates"]})
        gv = {}
        for i in range(0, len(need), 256):
            chunk = need[i:i + 256]
            v = enc.gloss_vectors([gloss_of(c) for c in chunk], pool).float()
            if cosine:
                v = torch.nn.functional.normalize(v, dim=-1)
            gv.update({c: v[k] for k, c in enumerate(chunk)})
        for split in EVAL_SPLITS:
            sel = [it for it in items if it["split"] == split]
            if not sel:
                continue
            correct = []
            for i in range(0, len(sel), bs):
                batch = sel[i:i + bs]
                cv = enc.context_vectors(batch).float()
                if cosine:
                    cv = torch.nn.functional.normalize(cv, dim=-1)
                for it, v in zip(batch, cv):
                    scores = torch.stack([gv[c] for c in it["candidates"]]) @ v
                    correct.append(int(it["candidates"][int(scores.argmax())] == it["synset"]))
            out[split] = {"n": len(sel), "acc": round(sum(correct) / max(1, len(sel)), 4), "correct": correct}
    return out


# ---------- train ----------
def train(a):
    import torch
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    items, synsets = load_data(a.data)
    tr = [it for it in items if it["split"] in a.train_splits]
    gloss = {sid: s["definition"] for sid, s in synsets.items()}
    if a.shuffle_glosses:
        train_syns = sorted({c for it in tr for c in it["train_candidates"]})
        perm = train_syns[:]
        random.Random(a.seed + 1000).shuffle(perm)
        shuffled = dict(zip(train_syns, (gloss[p] for p in perm)))
        train_gloss = lambda c: shuffled[c]  # noqa: E731
    else:
        train_gloss = lambda c: gloss[c]  # noqa: E731
    enc = Encoders(a.model, "cuda")
    params = list(enc.ctx.parameters()) + list(enc.gls.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    best, best_state, log = -1.0, None, []
    rng = random.Random(a.seed)
    for epoch in range(a.epochs):
        enc.ctx.train()
        enc.gls.train()
        rng.shuffle(tr)
        t0, tot = time.time(), 0.0
        for i in range(0, len(tr), a.batch):
            batch = tr[i:i + a.batch]
            cand_lists = []
            for it in batch:
                others = [c for c in it["train_candidates"] if c != it["synset"]]
                cand_lists.append([it["synset"]] + rng.sample(others, min(len(others), a.max_candidates - 1)))
            uniq = sorted({c for cl in cand_lists for c in cl})
            idx = {c: k for k, c in enumerate(uniq)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cv = enc.context_vectors(batch)
                gv = enc.gloss_vectors([train_gloss(c) for c in uniq])
                loss = 0.0
                for k, cl in enumerate(cand_lists):
                    s = gv[[idx[c] for c in cl]].float() @ cv[k].float()
                    loss = loss + torch.nn.functional.cross_entropy(s.unsqueeze(0), torch.zeros(1, dtype=torch.long, device="cuda"))
                loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        enc.ctx.eval()
        enc.gls.eval()
        dev = evaluate(enc, [it for it in items if it["split"] in DEV_SPLITS], synsets, lambda c: gloss[c])
        score = sum(v["acc"] for v in dev.values()) / len(dev)
        log.append({"epoch": epoch + 1, "loss": round(tot / max(1, len(tr) // a.batch), 4),
                    **{s: v["acc"] for s, v in dev.items()}, "seconds": round(time.time() - t0)})
        print(json.dumps(log[-1]), flush=True)
        if score > best:
            best = score
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.ctx.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in enc.gls.state_dict().items()})
    enc.ctx.load_state_dict(best_state[0])
    enc.gls.load_state_dict(best_state[1])
    res = evaluate(enc, items, synsets, lambda c: gloss[c])
    out = {"experiment": "E3a dictionary v1", "learner": "bi-encoder", "model": a.model, "seed": a.seed,
           "train_splits": a.train_splits,
           "shuffle_glosses": a.shuffle_glosses, "n_train": len(tr), "log": log, "best_dev": round(best, 4),
           "results": res}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({s: (v["n"], v["acc"]) for s, v in res.items()}))
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        enc.ctx.half().save_pretrained(os.path.join(a.save, "context"))
        enc.gls.half().save_pretrained(os.path.join(a.save, "gloss"))
        enc.tok.save_pretrained(a.save)


# ---------- baselines ----------
def content(text, drop=()):
    return {w for w in (m.group(0).lower() for m in TOKEN.finditer(text)) if w not in STOP and w not in drop}


def baselines(a):
    items, synsets = load_data(a.data)
    ev = [it for it in items if it["split"] in EVAL_SPLITS]
    res = {"random": {}, "mfs": {}, "lesk": {}}
    for split in EVAL_SPLITS:
        sel = [it for it in ev if it["split"] == split]
        res["random"][split] = {"n": len(sel), "acc": round(sum(1 / len(it["candidates"]) for it in sel) / max(1, len(sel)), 4),
                                "correct": [round(1 / len(it["candidates"]), 4) for it in sel]}
        mfs = [int(it["candidates"][0] == it["synset"]) for it in sel]
        res["mfs"][split] = {"n": len(sel), "acc": round(sum(mfs) / max(1, len(sel)), 4), "correct": mfs}
        lesk = []
        for it in sel:
            target = it["text"][it["start"]:it["end"]].lower()
            ctx = content(it["text"], drop={target, it["lemma"]})
            def overlap(c):
                s = synsets[c]
                sig = content(" ".join([s["definition"]] + [e for e in s["examples"] if e != it["text"]]), drop={it["lemma"]})
                return len(ctx & sig)
            best = max(range(len(it["candidates"])), key=lambda k: (overlap(it["candidates"][k]), -k))
            lesk.append(int(it["candidates"][best] == it["synset"]))
        res["lesk"][split] = {"n": len(sel), "acc": round(sum(lesk) / max(1, len(sel)), 4), "correct": lesk}
    train_n = [it for it in items if it["split"] == "train_n"]
    if train_n:  # Experiment 3a2: the frequency table learned from the SemCor training documents
        freq = collections.defaultdict(collections.Counter)
        for it in train_n:
            freq[(it["lemma"], it["pos"])][it["synset"]] += 1
        res["mfs_train"] = {}
        for split in EVAL_SPLITS:
            sel = [it for it in ev if it["split"] == split]
            if not sel:
                continue
            ok = []
            for it in sel:
                f = freq.get((it["lemma"], it["pos"]), {})
                pred = max(range(len(it["candidates"])), key=lambda k: (f.get(it["candidates"][k], 0), -k))
                ok.append(int(it["candidates"][pred] == it["synset"]))
            res["mfs_train"][split] = {"n": len(sel), "acc": round(sum(ok) / len(sel), 4), "correct": ok}
    enc = Encoders(a.model, "cuda")
    enc.ctx.eval()
    enc.gls.eval()
    res["untrained_encoder"] = evaluate(enc, items, synsets, lambda c: synsets[c]["definition"], pool="mean", cosine=True)
    json.dump({"experiment": "E3a dictionary v1 baselines", "model": a.model, "results": res}, open(a.out, "w"))
    print(json.dumps({b: {s: v["acc"] for s, v in r.items()} for b, r in res.items()}, indent=1))


# ---------- report ----------
def paired_ci(x, y, n_boot=10000, seed=SEED):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return round(float(x.mean() - y.mean()), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def report(a):
    runs = [json.load(open(p)) for p in a.runs]
    control = json.load(open(a.control))["results"]
    base = json.load(open(a.baselines))["results"]
    crit = {"D1": ("test_s", ("mfs", "lesk", "untrained_encoder")), "D2": ("test_z", ("mfs", "lesk", "untrained_encoder")),
            "D3": ("test_l", ("mfs", "lesk", "untrained_encoder")), "D5": ("test_n", ("lesk", "untrained_encoder"))}
    out = {"accuracy": {}, "criteria": {}}
    for split in EVAL_SPLITS:
        if split not in base["mfs"]:
            continue
        out["accuracy"][split] = {**{f"D seed {r['seed']}": r["results"][split]["acc"] for r in runs},
                                  "shuffled-gloss control": control[split]["acc"],
                                  **{b: base[b][split]["acc"] for b in base}, "n": base["mfs"][split]["n"]}
    for name, (split, against) in crit.items():
        rows, ok = {}, True
        for r in runs:
            for b in against:
                diff, lo, hi = paired_ci(r["results"][split]["correct"], base[b][split]["correct"])
                rows[f"seed {r['seed']} - {b}"] = [diff, lo, hi]
                ok &= lo > 0
        out["criteria"][name] = {"split": split, "pass": ok, "differences_with_95ci": rows}
    gap = round(control["test_z"]["acc"] - base["random"]["test_z"]["acc"], 4)
    out["criteria"]["D4"] = {"split": "test_z", "control_minus_random": gap, "pass": abs(gap) <= 0.05}
    out["seed_range"] = {s: round(max(r["results"][s]["acc"] for r in runs) - min(r["results"][s]["acc"] for r in runs), 4)
                         for s in ("test_s", "test_z")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def evaluate_saved(a):
    items, synsets = load_data(a.data)
    enc = Encoders(a.model, "cuda")
    enc.ctx.eval()
    enc.gls.eval()
    res = evaluate(enc, items, synsets, lambda c: synsets[c]["definition"])
    json.dump({"experiment": "E3a2 evaluation of a saved model", "model": a.model, "results": res}, open(a.out, "w"))
    print(json.dumps({s: (v["n"], v["acc"]) for s, v in res.items()}))


def report_usage(a):
    runs = [json.load(open(p)) for p in a.runs]
    v1 = json.load(open(a.v1))["results"]
    v1_z = sum(json.load(open(p))["results"]["test_z"]["acc"] for p in a.v1_runs) / len(a.v1_runs)
    base = json.load(open(a.baselines))["results"]
    out = {"accuracy": {s: {**{f"D2 seed {r['seed']}": r["results"][s]["acc"] for r in runs}, "dictionary v1 (3a seed 1)": v1[s]["acc"],
                            **{b: base[b][s]["acc"] for b in base if s in base[b]}, "n": base["mfs"][s]["n"]}
                        for s in ("dev_n", "test_n2", "test_s", "test_z", "test_l") if s in base["mfs"]},
           "criteria": {}}
    checks = {"N1": ("test_n2", {"mfs_train": base["mfs_train"]["test_n2"]["correct"]}),
              "N2": ("test_n2", {"dictionary v1": v1["test_n2"]["correct"]}),
              "N3": ("test_z", {b: base[b]["test_z"]["correct"] for b in ("mfs", "lesk", "untrained_encoder")})}
    for name, (split, against) in checks.items():
        rows, ok = {}, True
        for r in runs:
            for b, ref in against.items():
                diff, lo, hi = paired_ci(r["results"][split]["correct"], ref)
                rows[f"seed {r['seed']} - {b}"] = [diff, lo, hi]
                ok &= lo > 0
        out["criteria"][name] = {"split": split, "pass": ok, "differences_with_95ci": rows}
    floor = round(v1_z - 0.02, 4)
    kept = all(r["results"]["test_z"]["acc"] >= floor for r in runs)
    out["criteria"]["N3"].update({"test_z_floor": floor, "test_z_kept": kept, "pass": out["criteria"]["N3"]["pass"] and kept})
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("data")
    p.add_argument("--oewn", required=True)
    p.add_argument("--semcor", required=True)
    p.add_argument("--semcor-train", action="store_true", help="Experiment 3a2: SemCor train/dev/test documents")
    for name in ("train", "baselines", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("data")
        p.add_argument("out")
        p.add_argument("--model", default="bert-base-uncased")
        if name == "train":
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--shuffle-glosses", action="store_true")
            p.add_argument("--epochs", type=int, default=8)
            p.add_argument("--batch", type=int, default=16)
            p.add_argument("--max-candidates", type=int, default=16)
            p.add_argument("--lr", type=float, default=2e-5)
            p.add_argument("--save", default=None)
            p.add_argument("--train-splits", nargs="+", default=["train"])
    p = sub.add_parser("report")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--baselines", required=True)
    p = sub.add_parser("report-usage")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--v1", required=True, help="evaluate output of the saved 3a model on the 3a2 data")
    p.add_argument("--v1-runs", nargs="+", required=True, help="the 3a seed results (test-Z floor)")
    p.add_argument("--baselines", required=True)
    a = ap.parse_args()
    {"prep": prep, "train": train, "baselines": baselines, "evaluate": evaluate_saved, "report": report,
     "report-usage": report_usage}[a.cmd](a)


if __name__ == "__main__":
    main()
