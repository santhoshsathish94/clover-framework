"""CSI dictionary layer, meaning in context (Dictionary v1.1, Experiments 3a/3a2; contract in ARCHITECTURE.md 2b).
For a single-word noun, verb or adjective in a sentence it returns the OEWN meanings its lemma can have, ranked by a
bi-encoder: the context encoder's vector for the target word against the gloss encoder's vector for each definition.
Not a complete dictionary: no adverbs, multiword expressions, words outside OEWN, relations, or calibrated confidence."""
import collections
import gzip
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET

VERSION = "dictionary-v1.1"
UPOS = {"NOUN": "n", "VERB": "v", "ADJ": "a"}
GROUP = {"n": "n", "v": "v", "a": "a", "s": "a"}
MAX_TOKENS, WINDOW_CHARS = 128, 250


def load_inventory(oewn_path):
    """-> ({(lemma, pos): synset ids}, {synset id: definition}), built exactly as in dict_meaning.prep: OEWN sense order
    within an entry; where entries merge, head adjectives before satellites and lower-case lemmas before capitalised."""
    entries, definitions = [], {}
    with (gzip.open if oewn_path.endswith(".gz") else open)(oewn_path, "rb") as fh:
        for _, el in ET.iterparse(fh):
            if el.tag == "LexicalEntry":
                lem = el.find("Lemma")
                entries.append((el.get("id"), lem.get("writtenForm") or "", lem.get("partOfSpeech"),
                                [s.get("synset") for s in el.findall("Sense")]))
                el.clear()
            elif el.tag == "Synset":
                d = el.find("Definition")
                definitions[el.get("id")] = d.text if d is not None else ""
                el.clear()
    inventory = collections.defaultdict(list)
    for _, lemma, pos, syns in sorted(entries, key=lambda e: (e[2] == "s", e[1] != e[1].lower(), e[0])):
        g = GROUP.get(pos)
        if g is None or not re.fullmatch(r"[A-Za-z]+", lemma):
            continue
        key = (lemma.lower(), g)
        inventory[key] += [s for s in syns if s not in inventory[key]]
    return dict(inventory), definitions


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Senses:
    def __init__(self, checkpoint, oewn_path, device="cpu", tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.inventory, self.definitions = load_inventory(oewn_path)
        tok_dir = tokenizer or checkpoint
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        self.ctx = AutoModel.from_pretrained(os.path.join(checkpoint, "context"), torch_dtype=torch.float32).to(device).eval()
        self.gls = AutoModel.from_pretrained(os.path.join(checkpoint, "gloss"), torch_dtype=torch.float32).to(device).eval()
        assert all(p.dtype == torch.float32 for m in (self.ctx, self.gls) for p in m.parameters())
        usage = os.path.join(checkpoint, "usage_lemmas.json")
        self.usage = {tuple(k.split("|")) for k in json.load(open(usage))} if os.path.exists(usage) else None
        self.model_calls = 0
        self._gloss = {}
        files = {f"{p}/{f}": _sha256(os.path.join(checkpoint, p, f)) for p in ("context", "gloss")
                 for f in ("model.safetensors", "config.json")}
        tok_file = os.path.join(tok_dir, "tokenizer.json")
        self.provenance = {"component": "csi.senses", "version": VERSION, "code_sha256": _sha256(__file__),
                           "checkpoint_sha256": files, "weights_stored": "float16", "compute": "float32",
                           "tokenizer": os.path.basename(os.path.normpath(tok_dir)),
                           "tokenizer_sha256": _sha256(tok_file) if os.path.exists(tok_file) else None,
                           "inventory": os.path.basename(oewn_path), "inventory_sha256": _sha256(oewn_path)}

    def meanings(self, lemma, upos):
        p = UPOS.get(upos)
        return list(self.inventory.get(((lemma or "").lower(), p), [])) if p else []

    def _gloss_vectors(self, synsets):
        todo = [s for s in synsets if s not in self._gloss]
        if todo:
            self.model_calls += 1
            enc = self.tok([self.definitions[s] for s in todo], padding=True, truncation=True, max_length=64,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                cls = self.gls(**enc).last_hidden_state[:, 0].clone()  # clone: do not keep the whole output alive
            self._gloss.update(zip(todo, cls))
        return self.torch.stack([self._gloss[s] for s in synsets])

    def _encode(self, text, start, end):
        self.model_calls += 1
        enc = self.tok([text], truncation=True, max_length=MAX_TOKENS, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0]
        pick = (offs[:, 0] < end) & (offs[:, 1] > start) & (offs[:, 1] > offs[:, 0])
        if not pick.any():
            return None
        with self.torch.no_grad():
            hid = self.ctx(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0]
        return hid[pick.to(self.device)].mean(0)

    def choose(self, text, start, end, lemma, upos):
        """Meanings of the word at text[start:end] with this lemma and UD part of speech, best first. status: chosen |
        single_meaning | not_in_dictionary | out_of_scope | invalid_span | target_not_located. Scores are not calibrated."""
        p = UPOS.get(upos)
        out = {"lemma": lemma, "upos": upos, "span": [start, end], "scores_calibrated": False,
               "usage_trained": None if self.usage is None or p is None else ((lemma or "").lower(), p) in self.usage,
               "provenance": self.provenance}
        if p is None or not re.fullmatch(r"[A-Za-z]+", lemma or ""):
            return {**out, "status": "out_of_scope", "senses": []}
        if not (0 <= start < end <= len(text)) or not text[start:end].strip():
            return {**out, "status": "invalid_span", "senses": []}
        cands = self.meanings(lemma, upos)
        if not cands:
            return {**out, "status": "not_in_dictionary", "senses": []}
        if len(cands) == 1:
            return {**out, "status": "single_meaning",
                    "senses": [{"synset": cands[0], "definition": self.definitions[cands[0]], "score": None, "p": 1.0}]}
        vec, windowed = self._encode(text, start, end), False
        if vec is None:  # the word lies beyond the token limit: encode a character window around it
            lo = max(0, start - WINDOW_CHARS)
            vec, windowed = self._encode(text[lo:end + WINDOW_CHARS], start - lo, end - lo), True
        if vec is None:
            return {**out, "status": "target_not_located", "senses": []}
        scores = (self._gloss_vectors(cands) @ vec).tolist()
        top = max(scores)
        z = sum(math.exp(s - top) for s in scores)
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
        return {**out, "status": "chosen", "windowed": windowed,
                "senses": [{"synset": s, "definition": self.definitions[s], "score": round(sc, 4),
                            "p": round(math.exp(sc - top) / z, 4)} for s, sc in ranked]}
