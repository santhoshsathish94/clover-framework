"""Training sessions and fresh evaluation scenarios for one simulated person.
Training: the person's habits (training phrasings, frames and objects), the same expressions in areas where they mean
what everyone means (controls), ordinary requests, their routine, open items, and occasional explicit statements.
Evaluation: the habits in held-out frames, constructions, objects and (for area-general habits) new areas; controls;
ordinary requests; the routine; questions about open items."""
import math
import random

from . import personas as P
from . import simulator as S
from . import world as W

TASK_INTENTS = [i for i in W.INTENTS if i not in ("notes.later", "notes.done", "notes.open")]


def slot_values(types, part, rng, fixed=None):
    out = dict(fixed or {})
    for t in types:
        if t not in out:
            out[t] = rng.choice(W.objects(t, part))
    return out


def placeholders(template):
    return [p.split("}")[0] for p in template.split("{")[1:]]


def habit_item(person, h, part, rng, template=None):
    pool = (h["train"] if part == "train" else h["eval"] + h["eval_msgs"])
    template = template or rng.choice(pool)
    context = P.template_context(h, template)
    slots = slot_values(placeholders(template), part, rng)
    slots = slot_values(W.INTENTS[h["intent"]][0], part, rng, slots)
    text = W.fill(template, slots)
    if h["vp"] and template not in h["eval_msgs"]:
        text = W.frame(S.person_frame(person, part, rng), text)
    focus = slots.get("code") or rng.choice(W.objects("code", part))
    return {"kind": "habit", "habit": h["id"], "context": context, "focus": focus, "text": text, "intent": h["intent"],
            "slots": slots, "template": template}


def control_item(person, h, control, part, rng):
    context, template, intent = control
    slots = slot_values(placeholders(template), part, rng)
    slots = slot_values(W.INTENTS[intent][0], part, rng, slots)
    text = W.frame(S.person_frame(person, part, rng), W.fill(template, slots))
    return {"kind": "control", "habit": h["id"], "context": context, "focus": rng.choice(W.objects("code", part)),
            "text": text, "intent": intent, "slots": slots, "template": template}


def generic_item(person, part, rng, intent=None):
    if intent is None:
        weights = [3 if W.INTENTS[i][1] else 1 for i in TASK_INTENTS]
        intent = rng.choices(TASK_INTENTS, weights=weights)[0]
    context = W.CONTEXT_OF[intent]
    slots = slot_values(W.INTENTS[intent][0], part, rng)
    vp = S.general_vp(intent, slots, rng, S.avoid_heads(person, context))
    return {"kind": "generic", "habit": None, "context": context, "focus": rng.choice(W.objects("code", part)),
            "text": W.frame(S.person_frame(person, part, rng), vp), "intent": intent, "slots": slots}


def routine_item(person, part, rng):
    phrase = person["routine"]["phrase"]
    forms = (("Do {p}, please.", "{P}, please.", "Time for {p}.") if part == "train"
             else ("Could you do {p}?", "I'd like you to do {p}.", "When you get a chance, do {p}."))
    text = rng.choice(forms).format(p=phrase, P=phrase[0].upper() + phrase[1:])
    return {"kind": "routine", "habit": None, "context": "notes", "focus": None, "text": text, "intent": "routine",
            "slots": {}, "steps": [list(s) for s in person["routine"]["steps"]]}


def oneoff_item(person, oo, part, rng, kind):
    """kind 'oneoff': the ordinary phrasing, meant differently this once (training); 'oneoff_probe': the same phrasing
    later, meant as everyone means it (evaluation)."""
    phrase = W.INTENTS[oo["intent"]][2][oo["phrase"]]
    slots = slot_values(W.INTENTS[oo["intent"]][0], part, rng)
    slots = slot_values(placeholders(phrase), part, rng, slots)
    return {"kind": kind, "habit": None, "context": W.CONTEXT_OF[oo["intent"]], "focus": None,
            "text": W.frame(S.person_frame(person, part, rng), W.fill(phrase, slots)),
            "intent": oo["meant"] if kind == "oneoff" else oo["intent"], "slots": slots, "oneoff": oo}


def open_item(person, kind, topic, part, rng, expected=None):
    intent = {"open_create": "notes.later", "open_resolve": "notes.done", "open_query": "notes.open"}[kind]
    slots = {"topic": topic} if topic else {}
    vp = S.general_vp(intent, slots, rng)
    item = {"kind": kind, "habit": None, "context": "notes", "focus": None,
            "text": W.frame(S.person_frame(person, part, rng), vp), "intent": intent, "slots": slots}
    if kind == "open_query":
        item["expected_open"] = sorted(expected)
    return item


def training_sessions(person, seed, per_habit=7, per_control=2, n_generic=60, sessions=10):
    rng = random.Random(seed)
    items = []
    for hid in person["habits"]:
        h = P.HABIT_BY_ID[hid]
        items += [habit_item(person, h, "train", rng, h["train"][k % len(h["train"])]) for k in range(per_habit)]
        for c in h["controls"]:
            items += [control_item(person, h, c, "train", rng) for _ in range(per_control)]
        if rng.random() < person["states_habit"]:
            items.append({"kind": "statement", "habit": hid, "context": h["contexts"][0], "focus": None,
                          "text": h["statement"], "intent": None, "slots": {}})
    for (d, v), texts in W.PREFERENCE_STATEMENTS.items():
        if person["preferences"][d] == v and v != W.DIMS[d][0] and rng.random() < person["states_habit"]:
            items.append({"kind": "statement", "habit": None, "context": "notes", "focus": None, "text": rng.choice(texts),
                          "intent": None, "slots": {}, "pref": [d, v]})
    items += [generic_item(person, "train", rng) for _ in range(n_generic)]
    items += [routine_item(person, "train", rng) for _ in range(4)]
    rng.shuffle(items)
    topics = person["open_topics"]
    creates = [open_item(person, "open_create", t, "train", rng) for t in topics]
    resolves = [open_item(person, "open_resolve", t, "train", rng) for t in topics[:2]]
    for k, it in enumerate(creates):  # creations in the first half, resolutions in the second
        items.insert(rng.randrange(0, len(items) // 2 + 1), it)
    for it in resolves:
        items.insert(rng.randrange(len(items) // 2 + 1, len(items) + 1), it)
    orng = random.Random(seed + 101)
    for oo in person.get("oneoffs", []):
        items.insert(orng.randrange(0, len(items) + 1), oneoff_item(person, oo, "train", orng, "oneoff"))
    size = math.ceil(len(items) / sessions)
    for k, it in enumerate(items):
        it["session"], it["split"] = k // size, "train"
    return items


def evaluation_items(person, seed, per_habit=6, per_control=3, n_generic=36):
    rng = random.Random(seed)
    items = []
    for hid in person["habits"]:
        h = P.HABIT_BY_ID[hid]
        pool = h["eval"] + h["eval_msgs"]
        items += [habit_item(person, h, "eval", rng, pool[k % len(pool)]) for k in range(per_habit)]
        for c in h["controls"]:
            items += [control_item(person, h, c, "eval", rng) for _ in range(per_control)]
    items += [generic_item(person, "eval", rng) for _ in range(n_generic)]
    items += [routine_item(person, "eval", rng) for _ in range(3)]
    still_open = person["open_topics"][2:]
    items += [open_item(person, "open_query", None, "eval", rng, still_open) for _ in range(2)]
    orng = random.Random(seed + 101)
    items += [oneoff_item(person, oo, "eval", orng, "oneoff_probe") for oo in person.get("oneoffs", []) for _ in range(2)]
    rng.shuffle(items)
    for it in items:
        it["session"], it["split"] = "eval", "eval"
    return items
