"""Experiment 4e: character-level transformer (from scratch) for verb lemmatisation on the Experiment 4c splits.
split: c1 (fold 0 lemmas held out), c2 (three-letter ending families held out), all (train on every verb; test = the
4c nonce verbs). Usage: python char_seq2seq.py <out json> --unimorph <eng> --split c1|c2|all [--epochs 30]"""
import argparse
import json
import math
import os
import random
import sys
import time
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morph_v2  # noqa: E402

PAD, BOS, EOS = 0, 1, 2


class CharS2S(nn.Module):
    def __init__(self, n_chars, d=256, heads=4, layers=4, max_len=40):
        super().__init__()
        self.emb = nn.Embedding(n_chars, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.Transformer(d, heads, layers, layers, dim_feedforward=4 * d, dropout=0.1, batch_first=True)
        self.out = nn.Linear(d, n_chars)

    def embed(self, x):
        return self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))[None]

    def forward(self, src, tgt_in):
        mask = self.tf.generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)
        h = self.tf(self.embed(src), self.embed(tgt_in), tgt_mask=mask,
                    src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD, memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, max_len=40):
        memory = self.tf.encoder(self.embed(src), src_key_padding_mask=src == PAD)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            mask = self.tf.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            h = self.tf.decoder(self.embed(ys), memory, tgt_mask=mask, memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


def encode(words, chars, max_len=40):
    t = torch.full((len(words), max_len), PAD, dtype=torch.long)
    for i, w in enumerate(words):
        ids = [chars[c] for c in w][:max_len - 1] + [EOS]
        t[i, :len(ids)] = torch.tensor(ids)
    return t


def decode(rows, inv):
    out = []
    for r in rows.tolist():
        s = []
        for c in r:
            if c in (EOS, PAD):
                break
            s.append(inv.get(c, "?"))
        out.append("".join(s))
    return out


def splits(name):
    verbs4 = morph_v2.DATA["verbs4"]
    fold_seed, ending_seed, _ = morph_v2.seeds()
    if name == "c1":
        test = set(morph_v2.folds(verbs4, 5, fold_seed)[0])
        return [l for l in verbs4 if l not in test], morph_v2.verb_entries(sorted(test))
    if name == "c2":
        held = set(morph_v2.folds(sorted({l[-3:] for l in verbs4}), 5, ending_seed)[0])
        return [l for l in verbs4 if l[-3:] not in held], morph_v2.verb_entries([l for l in verbs4 if l[-3:] in held])
    nonce = morph_v2.nonce_verbs()
    test = [(l, f, "verb", t) for cls, lem in nonce.items() for l, fs in lem.items() for t, f in fs.items()]
    return sorted(morph_v2.DATA["by_class"]["verb"]), test, nonce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--unimorph", required=True)
    ap.add_argument("--split", choices=["c1", "c2", "all"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fold-seed", type=int, default=11)
    ap.add_argument("--ending-seed", type=int, default=13)
    ap.add_argument("--nonce-seed", type=int, default=19)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    morph_v2.setup(types.SimpleNamespace(unimorph=a.unimorph, ewt_train=None, fold_seed=a.fold_seed,
                                         ending_seed=a.ending_seed, nonce_seed=a.nonce_seed))
    sp = splits(a.split)
    train_lemmas, test = sp[0], sp[1]
    bc = morph_v2.DATA["by_class"]["verb"]
    rng = random.Random(a.seed)
    lem = train_lemmas[:]
    rng.shuffle(lem)
    dev_l, tr_l = set(lem[:len(lem) // 50]), lem[len(lem) // 50:]
    pairs = lambda ls: [(f, l) for l in ls for f in bc[l].values()]
    train, dev = pairs(tr_l), pairs(sorted(dev_l))
    chars = {c: i + 3 for i, c in enumerate(sorted({c for f, l in train + dev + [(e[1], e[0]) for e in test] for c in f + l}))}
    inv = {i: c for c, i in chars.items()}
    model = CharS2S(len(chars) + 3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    X, Y = encode([f for f, _ in train], chars), encode([l for _, l in train], chars)
    steps = a.epochs * math.ceil(len(train) / 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    best, best_state, t0 = -1, None, time.time()

    def accuracy(items):
        model.eval()
        ok = 0
        for i in range(0, len(items), 2048):
            chunk = items[i:i + 2048]
            pred = decode(model.greedy(encode([f for f, _ in chunk], chars).cuda()), inv)
            ok += sum(p == l for p, (_, l) in zip(pred, chunk))
        model.train()
        return ok / len(items)
    for ep in range(a.epochs):
        perm = torch.randperm(len(train))
        for i in range(0, len(train), 512):
            idx = perm[i:i + 512]
            src, tgt = X[idx].cuda(), Y[idx].cuda()
            tgt_in = torch.cat([torch.full((len(idx), 1), BOS, device="cuda"), tgt[:, :-1]], 1)
            loss = nn.functional.cross_entropy(model(src, tgt_in).reshape(-1, len(chars) + 3), tgt.reshape(-1),
                                               ignore_index=PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        d = accuracy(dev)
        print(json.dumps({"epoch": ep + 1, "loss": round(loss.item(), 4), "dev_acc": round(d, 4),
                          "seconds": round(time.time() - t0)}), flush=True)
        if d > best:
            best, best_state = d, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_pairs = [(f, l) for l, f, _, _ in test]
    res = {"experiment": "E4e char transformer", "split": a.split, "n_train_pairs": len(train), "best_dev": round(best, 4),
           "test_acc": round(accuracy(test_pairs), 4), "n_test": len(test_pairs)}
    by_type = {}
    for t in morph_v2.VERB_TYPES:
        sel = [(f, l) for l, f, _, tt in test if tt == t]
        if sel:
            by_type[t] = round(accuracy(sel), 4)
    res["by_form_type"] = by_type
    if a.split == "all":
        res["nonce_by_class"] = {cls: round(accuracy([(f, l) for l, fs in lem.items() for f in fs.values()]), 4)
                                 for cls, lem in sp[2].items()}
    model.eval()
    sample = test_pairs[:: max(1, len(test_pairs) // 400)]
    preds = decode(model.greedy(encode([f for f, _ in sample], chars).cuda()), inv)
    res["error_sample"] = [f"{f} -> {p} (want {l})" for p, (f, l) in zip(preds, sample) if p != l][:30]
    print(json.dumps({k: v for k, v in res.items() if k != "error_sample"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
