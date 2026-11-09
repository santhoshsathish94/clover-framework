"""Caution signal: a routing reading of one message, used only to add caution.

The reading is act, gather, clarify or verify, taken from the chat model's label probabilities in one forward pass
(GPT-AGI/OpenJev's single-pass method with the fixes found in research/superagent-jev-dory: criteria shown with the
labels, multi-token labels scored whole, the null-state prior removed, then temperature). A confident gather, clarify
or verify reading can make Clover ask before an action runs. An act reading changes nothing: it never grants an
action and never removes a question Clover would ask anyway (human direction, 2026-11-09, review item 1).

Whether a model's reading is good enough to be used at all is decided outside this file, by the pre-registered gate
in research/superagent-jev-dory/05-plain-message-routing (held-out tasks: confident act readings of ambiguous or
unverified requests were the failure seen there)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

LABELS = ("act", "gather", "clarify", "verify")
CAUTION = ("gather", "clarify", "verify")
CRITERIA = {
    "act": "Directly perform the task or answer now because all inputs are present",
    "gather": "Read files, logs, or system state to gather missing evidence first",
    "clarify": "Ask the user a clarifying question because the request is ambiguous or high-risk",
    "verify": "Run a check or compare against raw evidence to verify an unconfirmed claim",
}
INSTRUCTIONS = "Given the state, what should the agent do next?"
NULL_STATE, NULL_INSTRUCTIONS = "[Neutral baseline state with no specific task signal]", "Select a label."
CHAT_SYSTEM = ("You are a calibrated decision engine. Read the state and the allowed label definitions carefully. "
               "Reply with the single best label and nothing else.")
MARKDOWN_HEAD = ("You are a decision engine. Read the state, then answer the question by replying "
                 "with exactly one of the allowed labels and nothing else.\n\n")
ANSWER = "### Answer\n"
WHY = {
    "gather": "it may need information Clover has not read yet, so Clover asks before it runs",
    "clarify": "it may be missing a detail only you can give, so Clover asks before it runs",
    "verify": "it relies on something that has not been checked, so Clover asks before it runs",
}


def question_suffix(instructions=INSTRUCTIONS, labels=LABELS, criteria=CRITERIA):
    options = "\n".join(f"- {lbl}: {criteria[lbl]}" if criteria.get(lbl) else f"- {lbl}" for lbl in labels)
    return f"### Question\n{instructions}\n\n### Allowed labels\n{options}\n\n{ANSWER}"


def softmax(logits, temperature=1.0):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def confidence(probs):
    """1 - normalized entropy: 1.0 when all mass is on one label, 0.0 when uniform."""
    if len(probs) <= 1:
        return 1.0
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - ent / math.log(len(probs))))


def debias(raw, prior, alpha):
    mean = sum(prior) / len(prior)
    return [s - alpha * (p - mean) for s, p in zip(raw, prior)]


@dataclass
class Reading:
    label: str
    confidence: float
    probabilities: dict
    caution: bool
    ms: float
    notes: list = field(default_factory=list)

    def layer(self):
        return {"layer": "caution", "reading": self.label, "confidence": self.confidence, "caution": self.caution,
                "probabilities": self.probabilities, "ms": round(self.ms)}


class LlamaServerScorer:
    """Exact label log-probabilities from a running llama-server. Its /completion reports log-probabilities only for
    the n_probs most likely tokens, and a label often is not among them (after "### Answer" the model would rather
    write "The" or "-"). So each message takes two passes over the same cached prompt: one for the most likely
    tokens, and one with the labels' first tokens biased to the top, whose post-sampling probabilities are the exact
    distribution among those first tokens. One label present in the first pass then fixes the others' absolute
    values. A label longer than one token takes one more pass per further token. request(method, path, body,
    timeout) -> (status, bytes), for example LlamaServer._request, which carries the server's key."""

    def __init__(self, request, template="markdown", n_probs=200, timeout=60):
        if template not in ("markdown", "chat"):
            raise ValueError("template must be markdown or chat")
        self.request, self.template, self.n_probs, self.timeout = request, template, int(n_probs), timeout
        self._bos = self._seqs = None
        self.floors = 0  # label tokens whose log-probability had to be estimated (none of them among n_probs)

    def _post(self, path, body):
        status, data = self.request("POST", path, body, self.timeout)
        if status != 200:
            raise RuntimeError(f"llama-server {path} HTTP {status}: {bytes(data)[:200]!r}")
        return json.loads(data)

    def tokenize(self, text):
        return [int(t) for t in self._post("/tokenize", {"content": text, "add_special": False,
                                                         "parse_special": True})["tokens"]]

    def _setup(self):
        if self._seqs is None:
            status, data = self.request("GET", "/props", None, self.timeout)
            bos_text = (json.loads(data).get("bos_token") or "") if status == 200 else ""
            bos = self.tokenize(bos_text) if bos_text else []
            self._bos = bos[0] if len(bos) == 1 else None  # every prompt starts like the model's own chat template
            self._seqs = [self.tokenize(lbl) for lbl in LABELS]
            firsts = [s[0] for s in self._seqs if s]
            if len(firsts) != len(LABELS) or len(set(firsts)) != len(LABELS):
                raise RuntimeError("the labels' first tokens are not distinct")

    def prompt(self, state, instructions=INSTRUCTIONS):
        body, suffix = f"### State\n{state}\n\n", question_suffix(instructions)
        if self.template == "markdown":
            return MARKDOWN_HEAD + body + suffix
        msgs = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": body + suffix[:-len(ANSWER)]}]
        text = self._post("/apply-template", {"messages": msgs})["prompt"]
        if text.endswith("<think>\n"):  # read the answer position, not the start of a reasoning block
            text = text[:-len("<think>\n")] + "<think></think>\n\n"
        elif text.endswith("<think>"):
            text += "</think>\n\n"
        return text + ANSWER

    def ids(self, state, instructions=INSTRUCTIONS):
        self._setup()
        ids = self.tokenize(self.prompt(state, instructions))
        if self._bos is not None and (not ids or ids[0] != self._bos):
            ids = [self._bos] + ids
        return ids

    def _top(self, ids):
        out = self._post("/completion", {"prompt": ids, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0,
                                         "cache_prompt": True})
        probs = (out.get("completion_probabilities") or [{}])[0]
        return {int(p["id"]): float(p["logprob"]) for p in probs.get("top_logprobs", [])}

    def _among(self, ids, tokens):
        """The exact distribution among `tokens` at the next position (biased to the top, then renormalized)."""
        out = self._post("/completion", {"prompt": ids, "n_predict": 1, "n_probs": len(tokens), "temperature": 1.0,
                                         "post_sampling_probs": True, "samplers": ["top_k", "temperature"],
                                         "top_k": len(tokens), "logit_bias": [[t, 100.0] for t in tokens],
                                         "cache_prompt": True})
        probs = (out.get("completion_probabilities") or [{}])[0]
        got = {int(p["id"]): float(p["prob"]) for p in probs.get("top_probs", [])}
        return [got.get(t, 0.0) for t in tokens]

    def _logprob(self, top, token):
        if token in top:
            return top[token]
        self.floors += 1
        return min(top.values()) - 1.0

    def raw_scores(self, state, instructions=INSTRUCTIONS):
        """Per label: log-probability of its first token, or for a longer label the mean over its tokens."""
        ids = self.ids(state, instructions)
        top = self._top(ids)
        firsts = [s[0] for s in self._seqs]
        q = self._among(ids, firsts)
        known = [i for i, t in enumerate(firsts) if t in top and q[i] > 0]
        if known:
            k = max(known, key=lambda i: q[i])
            log_z = top[firsts[k]] - math.log(q[k])
            first_lp = [math.log(max(qi, 1e-300)) + log_z for qi in q]
        else:
            first_lp = [self._logprob(top, t) for t in firsts]
        scores = []
        for i, seq in enumerate(self._seqs):
            total = first_lp[i]
            for j in range(1, len(seq)):
                total += self._logprob(self._top(ids + seq[:j]), seq[j])
            scores.append(total / len(seq))
        return scores


class CautionEngine:
    """scorer.raw_scores(state, instructions) -> one score per label in LABELS. alpha: how much of the null-state
    prior is removed; threshold: the confidence a gather/clarify/verify reading needs before it adds caution."""

    def __init__(self, scorer, alpha=0.8, temperature=0.7, threshold=0.6):
        self.scorer, self.alpha, self.temperature, self.threshold = scorer, alpha, temperature, threshold
        self._prior = None

    def prior(self):
        if self._prior is None:
            self._prior = self.scorer.raw_scores(NULL_STATE, NULL_INSTRUCTIONS)
        return self._prior

    def reading(self, text):
        t0 = time.perf_counter()
        raw = self.scorer.raw_scores(text)
        logits = debias(raw, self.prior(), self.alpha) if self.alpha > 0 else raw
        probs = softmax(logits, self.temperature)
        best = max(range(len(LABELS)), key=lambda i: probs[i])
        conf = round(confidence(probs), 4)
        label = LABELS[best]
        return Reading(label=label, confidence=conf, probabilities={k: round(p, 4) for k, p in zip(LABELS, probs)},
                       caution=label in CAUTION and conf >= self.threshold, ms=(time.perf_counter() - t0) * 1000)


def apply(frame, reading):
    """Caution only. A confident gather/clarify/verify reading of an action Clover would run without asking makes it
    ask first; any other reading, and any other kind of message, is left exactly as it was. -> True when it changed."""
    if reading is None or not reading.caution or frame.act != "action" or frame.discernment != "proceed":
        return False
    frame.discernment, frame.why = "ask", WHY[reading.label]
    return True
