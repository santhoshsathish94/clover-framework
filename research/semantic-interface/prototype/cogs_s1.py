"""Experiment 6b S1 (sequence learner): LFM3-Omni-2B S3 + LoRA learns COGS sentence -> logical form.
Train on COGS train, choose the epoch by dev exact match, then score test and gen once (greedy decoding).
Usage:
  LFM3_DIR=<model> python cogs_s1.py train <COGS data dir> <out dir> [--epochs 3] [--limit N] [--dev-limit N]
  LFM3_DIR=<model> python cogs_s1.py eval  <COGS data dir> <out dir> --split gen [--shard 0/3] [--limit N]"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cogs import parse_lf  # noqa: E402

SEP = " =>"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]


def load(d, split, limit=0):
    rows = []
    for line in open(os.path.join(d, f"{split}.tsv"), encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        rows.append((c[0], c[1], c[2] if len(c) > 2 else "in_distribution"))
    return rows[:limit] if limit else rows


def load_model(lora_dir=None, rank=16):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = os.environ["LFM3_DIR"]
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, trust_remote_code=True).cuda()
    ids = {"bos": model.config.bos_token_id, "eos": tok.eos_token_id if tok.eos_token_id is not None else
           model.config.eos_token_id, "pad": tok.pad_token_id if tok.pad_token_id is not None else model.config.pad_token_id}
    assert None not in ids.values(), ids
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                                 target_modules=TARGETS))
    return tok, model, ids


def prompt_ids(tok, ids, sent):
    return [ids["bos"]] + tok.encode(sent + SEP, add_special_tokens=False)


def norm(s):
    return " ".join(s.split())


@torch.no_grad()
def generate(tok, model, ids, sents, max_new, bs=48):
    """Greedy decoding; prompts grouped by exact token length so no padding is ever needed."""
    model.eval()
    groups = collections.defaultdict(list)
    for i, s in enumerate(sents):
        p = prompt_ids(tok, ids, s)
        groups[len(p)].append((i, p))
    out = [None] * len(sents)
    for n, items in sorted(groups.items()):
        for k in range(0, len(items), bs):
            chunk = items[k:k + bs]
            x = torch.tensor([p for _, p in chunk], device="cuda")
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                               do_sample=False, eos_token_id=ids["eos"], pad_token_id=ids["pad"])
            for (i, _), row in zip(chunk, y[:, n:].tolist()):
                if ids["eos"] in row:
                    row = row[:row.index(ids["eos"])]
                out[i] = norm(tok.decode(row, skip_special_tokens=True))
    return out


def role_match(pred_lf, gold_lf):
    """Experiment 6a metric on logical forms: main event verb + agent/theme/recipient heads equal."""
    try:
        ge, gm = parse_lf(gold_lf)
        pe, pm = parse_lf(pred_lf)
    except (ValueError, KeyError):
        return False
    if gm is None or pm is None:
        return False
    g, p = ge[gm], pe[pm]
    return g["verb"] == p["verb"] and g["roles"] == p["roles"]


def score(rows, preds):
    by = collections.defaultdict(lambda: [0, 0, 0])
    for (s, lf, case), p in zip(rows, preds):
        b = by[case]
        b[0] += 1
        b[1] += norm(lf) == p
        b[2] += role_match(p, lf)
    tot = [sum(v[i] for v in by.values()) for i in range(3)]
    return {"n": tot[0], "lf_exact": round(tot[1] / tot[0], 4), "role_exact": round(tot[2] / tot[0], 4),
            "by_case": {k: {"n": v[0], "lf_exact": round(v[1] / v[0], 4), "role_exact": round(v[2] / v[0], 4)}
                        for k, v in sorted(by.items())}}


def max_new_for(tok, rows):
    return int(1.5 * max(len(tok.encode(" " + lf, add_special_tokens=False)) for _, lf, _ in rows)) + 8


def train(a):
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok, model, ids = load_model(rank=a.rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows = load(a.data, "train", a.limit)
    dev = load(a.data, "dev", a.dev_limit)
    ex = []
    for s, lf, _ in rows:
        p = prompt_ids(tok, ids, s)
        t = tok.encode(" " + lf, add_special_tokens=False) + [ids["eos"]]
        ex.append((p + t, [-100] * len(p) + t))
    order = sorted(range(len(ex)), key=lambda i: len(ex[i][0]))
    batches, cur, longest = [], [], 0
    for i in order:
        n = len(ex[i][0])
        if cur and max(longest, n) * (len(cur) + 1) > a.token_budget:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    batches.append(cur)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total, warm = a.epochs * len(batches), 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    max_new_dev = max_new_for(tok, dev)
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(batches)
        run = 0.0
        for b in batches:
            n = max(len(ex[i][0]) for i in b)
            x = torch.full((len(b), n), ids["pad"], dtype=torch.long)
            y = torch.full((len(b), n), -100, dtype=torch.long)
            m = torch.zeros((len(b), n), dtype=torch.long)
            for r, i in enumerate(b):
                xi, yi = ex[i]
                x[r, :len(xi)], y[r, :len(yi)], m[r, :len(xi)] = torch.tensor(xi), torch.tensor(yi), 1
            loss = model(input_ids=x.cuda(), attention_mask=m.cuda(), labels=y.cuda()).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            run += loss.item()
            if step % 25 == 0:
                rec = {"epoch": epoch, "step": step, "of": total, "loss": round(run / 25, 4),
                       "s_per_step": round((time.time() - t0) / step, 3)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run = 0.0
        preds = generate(tok, model, ids, [s for s, _, _ in dev], max_new_dev)
        r = score(dev, preds)
        rec = {"epoch": epoch, "dev": {k: r[k] for k in ("n", "lf_exact", "role_exact")}}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if r["lf_exact"] > best:
            best = r["lf_exact"]
            model.save_pretrained(os.path.join(a.out, "lora-best"))
            json.dump({"epoch": epoch, "dev": rec["dev"], "args": vars(a)}, open(os.path.join(a.out, "best.json"), "w"),
                      indent=1)


def evaluate(a):
    tok, model, ids = load_model(os.path.join(a.out, "lora-best"))
    model = model.merge_and_unload()
    rows = load(a.data, a.split, a.limit)
    i, n = (int(x) for x in a.shard.split("/"))
    rows = rows[i::n]
    t0 = time.time()
    preds = generate(tok, model, ids, [s for s, _, _ in rows], max_new_for(tok, rows))
    r = score(rows, preds)
    r.update({"split": a.split, "shard": a.shard, "seconds": round(time.time() - t0, 1),
              "predictions": [{"s": s, "gold": lf, "case": c, "pred": p} for (s, lf, c), p in zip(rows, preds)]})
    tag = f"{a.split}_shard{i}of{n}"
    json.dump(r, open(os.path.join(a.out, f"eval_{tag}.json"), "w"), indent=1)
    print(tag, json.dumps({k: v for k, v in r.items() if k not in ("predictions", "by_case")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--token-budget", type=int, default=4096, help="padded tokens per batch (fp32 logits ~0.5 MB/token)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)


if __name__ == "__main__":
    main()
