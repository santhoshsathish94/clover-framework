"""The simulated person's side of an interaction: the messages they write for a scenario, and how they react to what
Clover did - accept, correct (in their own correction style), answer a clarifying question, or ask for a different form.
Their ground truth is used only here and for scoring; Clover sees only the text."""
import random

from . import personas as P
from . import world as W

ORDINALS = ("first", "second", "third", "fourth")


def general_vp(intent, slots, rng, avoid=()):
    phrases = [p for p in W.INTENTS[intent][2] if not any(p.startswith(a + " ") or p == a for a in avoid)]
    return W.fill(rng.choice(phrases or W.INTENTS[intent][2]), slots)


def person_frame(person, part, rng):
    frames = W.TRAIN_FRAMES if part == "train" else W.EVAL_FRAMES
    polite = [f for f in frames if "please" in f.lower() or "pls" in f or "can you" in f.lower() or "could you" in f.lower()
              or "when you get" in f.lower()]
    plain = [f for f in frames if f not in polite]
    pool = polite if (rng.random() < person["politeness"] and polite) else (plain or polite)
    return rng.choice(pool)


def avoid_heads(person, context):
    """General phrasings this person does not use in this area: their habits give those words another meaning."""
    out = set()
    for hid in person["habits"]:
        h = P.HABIT_BY_ID[hid]
        if context in h["contexts"]:
            out |= P.heads(h)
    return out


class Person:
    def __init__(self, spec, seed):
        self.spec = spec
        self.rng = random.Random(seed)
        self.prefs = spec["preferences"]

    def correction(self, scenario):
        """How this person says the answer was wrong, and what the text reveals."""
        style = self.rng.choices(list(self.spec["correction_style"]), weights=list(self.spec["correction_style"].values()))[0]
        if scenario.get("steps"):
            vp = " and then ".join(general_vp(i, s, self.rng) for i, s in scenario["steps"])
        elif scenario["intent"] == "notes.open":
            vp = "tell me what's still open"
        else:
            vp = general_vp(scenario["intent"], scenario["slots"], self.rng, avoid_heads(self.spec, scenario["context"]))
        if style == "explicit":
            return {"type": "correct", "style": style, "text": f"No, I meant: {vp}."}
        if style == "terse":
            return {"type": "correct", "style": style, "text": f"No. {vp[0].upper() + vp[1:]}."}
        if style == "vague":
            return {"type": "correct", "style": style, "text": "That's not what I meant."}
        return {"type": "correct", "style": style, "text": W.frame(self.rng.choice(W.GENERAL_FRAMES), vp)}

    def answer_clarification(self, scenario, options):
        """options: list of candidate intents (or 'routine:<phrase>') Clover offered."""
        target = "routine" if scenario.get("steps") else scenario["intent"]
        for k, o in enumerate(options[:len(ORDINALS)]):
            if o == target or (target == "routine" and str(o).startswith("routine")):
                return {"type": "clarify_answer", "text": f"The {ORDINALS[k]} one.", "choice": k}
        if scenario.get("steps"):
            vp = " and then ".join(general_vp(i, s, self.rng) for i, s in scenario["steps"])
            return {"type": "clarify_answer", "text": f"None of those. I mean: {vp}.", "choice": None}
        vp = general_vp(scenario["intent"], scenario["slots"], self.rng, avoid_heads(self.spec, scenario["context"]))
        return {"type": "clarify_answer", "text": f"None of those. I mean: {vp}.", "choice": None}

    def presentation_feedback(self, dims_wrong):
        dim = dims_wrong[0]
        return {"type": "presentation", "dim": dim, "text": self.rng.choice(W.PRESENTATION_PHRASES[(dim, self.prefs[dim])]).capitalize() + "."}

    def react(self, scenario, response):
        """One reaction to Clover's response (dict with intent / steps / items / presentation / clarify)."""
        if response.get("clarify"):
            return self.answer_clarification(scenario, response["clarify"])
        if scenario.get("steps"):
            right = [i for i, _ in scenario["steps"]] == [i for i, _ in response.get("steps") or []]
        elif scenario["intent"] == "notes.open":
            right = response.get("intent") == "notes.open" and sorted(response.get("items") or []) == sorted(scenario["expected_open"])
        else:
            right = response.get("intent") == scenario["intent"]
        if not right:
            if self.rng.random() < self.spec.get("inattention", 0.0):  # did not notice the wrong answer
                return {"type": "accept", "text": self.rng.choice(("Thanks.", "Great, thanks.", "Perfect.", "Ok.")),
                        "inattentive": True}
            return self.correction(scenario)
        dims = presentation_dims(scenario)
        wrong = [d for d in dims if (response.get("presentation") or {}).get(d, W.DEFAULT_PRESENTATION[d]) != self.prefs[d]]
        if wrong:
            return self.presentation_feedback(wrong)
        return {"type": "accept", "text": self.rng.choice(("Thanks.", "Great, thanks.", "Perfect.", "Ok."))}


def presentation_dims(scenario):
    if scenario.get("steps"):
        return ("plan",)
    if scenario["intent"] in W.INTENTS:
        return W.INTENTS[scenario["intent"]][1]
    return ()
