"""Experiment 1 (E6+E7) data: generated minimal pairs with a SEEN lexicon (development) and an UNSEEN lexicon (final
run only); Experiment 1b adds HELDOUT2 (nouns and verbs in neither earlier lexicon, 60-pair dative bucket, own seed).
Verb-form tables here belong to the test generator, not to CSI (CSI never reads them).
label 1 = same meaning (grammatical change only), 0 = different meaning."""
import random

# (base, past, past participle, gerund)
TRANSITIVE = {
    "seen": [("open", "opened", "opened", "opening"), ("close", "closed", "closed", "closing"),
             ("paint", "painted", "painted", "painting"), ("clean", "cleaned", "cleaned", "cleaning"),
             ("repair", "repaired", "repaired", "repairing"), ("carry", "carried", "carried", "carrying"),
             ("lift", "lifted", "lifted", "lifting"), ("wash", "washed", "washed", "washing"),
             ("break", "broke", "broken", "breaking"), ("take", "took", "taken", "taking"),
             ("bring", "brought", "brought", "bringing")],
    "unseen": [("lock", "locked", "locked", "locking"), ("fix", "fixed", "fixed", "fixing"),
               ("polish", "polished", "polished", "polishing"), ("load", "loaded", "loaded", "loading"),
               ("inspect", "inspected", "inspected", "inspecting"), ("drag", "dragged", "dragged", "dragging"),
               ("cover", "covered", "covered", "covering"), ("fill", "filled", "filled", "filling"),
               ("hide", "hid", "hidden", "hiding"), ("steal", "stole", "stolen", "stealing"),
               ("throw", "threw", "thrown", "throwing")],
    "heldout2": [("push", "pushed", "pushed", "pushing"), ("pull", "pulled", "pulled", "pulling"),
                 ("fold", "folded", "folded", "folding"), ("wrap", "wrapped", "wrapped", "wrapping"),
                 ("scrub", "scrubbed", "scrubbed", "scrubbing"), ("shake", "shook", "shaken", "shaking"),
                 ("tear", "tore", "torn", "tearing"), ("bend", "bent", "bent", "bending"),
                 ("weigh", "weighed", "weighed", "weighing"), ("buy", "bought", "bought", "buying"),
                 ("drop", "dropped", "dropped", "dropping")],
    "heldout3": [("grab", "grabbed", "grabbed", "grabbing"), ("squeeze", "squeezed", "squeezed", "squeezing"),
                 ("twist", "twisted", "twisted", "twisting"), ("crush", "crushed", "crushed", "crushing"),
                 ("stack", "stacked", "stacked", "stacking"), ("rinse", "rinsed", "rinsed", "rinsing"),
                 ("seal", "sealed", "sealed", "sealing"), ("smash", "smashed", "smashed", "smashing"),
                 ("hold", "held", "held", "holding"), ("lose", "lost", "lost", "losing"),
                 ("find", "found", "found", "finding")],
    # Experiment 4d: invented verbs (seed 20261104; not in UniMorph, OEWN or the 4c/4f/4g nonce sets)
    "nonce1": [("smon", "smonned", "smonned", "smonning"), ("snam", "snammed", "snammed", "snamming"),
               ("thulk", "thulked", "thulked", "thulking"), ("thomp", "thomped", "thomped", "thomping"),
               ("cribe", "cribed", "cribed", "cribing"), ("bove", "boved", "boved", "boving"),
               ("shommy", "shommied", "shommied", "shommying"), ("lubby", "lubbied", "lubbied", "lubbying"),
               ("swesh", "sweshed", "sweshed", "sweshing"), ("sproch", "sproched", "sproched", "sproching"),
               ("vick", "vicked", "vicked", "vicking"), ("shey", "sheyed", "sheyed", "sheying"),
               ("broy", "broyed", "broyed", "broying")],
    # Experiment 4h: seed 20261105, same exclusions plus nonce1 and nonce seed 127 (-ck, vowel+w/y exhausted)
    "nonce2": [("blig", "bligged", "bligged", "bligging"), ("screm", "scremmed", "scremmed", "scremming"),
               ("thont", "thonted", "thonted", "thonting"), ("drord", "drorded", "drorded", "drording"),
               ("vube", "vubed", "vubed", "vubing"), ("scrune", "scruned", "scruned", "scruning"),
               ("drunny", "drunnied", "drunnied", "drunnying"), ("breggy", "breggied", "breggied", "breggying"),
               ("bex", "bexed", "bexed", "bexing"), ("trich", "triched", "triched", "triching")],
}
ANIMATE_VERBS = {
    "seen": [("thank", "thanked", "thanked", "thanking"), ("help", "helped", "helped", "helping"),
             ("call", "called", "called", "calling"), ("follow", "followed", "followed", "following"),
             ("greet", "greeted", "greeted", "greeting"), ("praise", "praised", "praised", "praising"),
             ("meet", "met", "met", "meeting")],
    "unseen": [("admire", "admired", "admired", "admiring"), ("warn", "warned", "warned", "warning"),
               ("invite", "invited", "invited", "inviting"), ("blame", "blamed", "blamed", "blaming"),
               ("rescue", "rescued", "rescued", "rescuing"), ("ignore", "ignored", "ignored", "ignoring"),
               ("catch", "caught", "caught", "catching")],
    "heldout2": [("hug", "hugged", "hugged", "hugging"), ("visit", "visited", "visited", "visiting"),
                 ("trust", "trusted", "trusted", "trusting"), ("chase", "chased", "chased", "chasing"),
                 ("hire", "hired", "hired", "hiring"), ("forgive", "forgave", "forgiven", "forgiving"),
                 ("see", "saw", "seen", "seeing")],
    "heldout3": [("kiss", "kissed", "kissed", "kissing"), ("scold", "scolded", "scolded", "scolding"),
                 ("tease", "teased", "teased", "teasing"), ("obey", "obeyed", "obeyed", "obeying"),
                 ("comfort", "comforted", "comforted", "comforting"), ("phone", "phoned", "phoned", "phoning"),
                 ("fight", "fought", "fought", "fighting")],
    "nonce1": [("shret", "shretted", "shretted", "shretting"), ("frilt", "frilted", "frilted", "frilting"),
               ("chuze", "chuzed", "chuzed", "chuzing"), ("snommy", "snommied", "snommied", "snommying"),
               ("sprash", "sprashed", "sprashed", "sprashing")],
    "nonce2": [("plep", "plepped", "plepped", "plepping"), ("kost", "kosted", "kosted", "kosting"),
               ("thrube", "thrubed", "thrubed", "thrubing"), ("dratty", "drattied", "drattied", "drattying"),
               ("swux", "swuxed", "swuxed", "swuxing")],
}
DITRANSITIVE = {
    "seen": [("give", "gave", "given", "giving"), ("send", "sent", "sent", "sending"),
             ("show", "showed", "shown", "showing"), ("hand", "handed", "handed", "handing")],
    "unseen": [("lend", "lent", "lent", "lending"), ("offer", "offered", "offered", "offering"),
               ("sell", "sold", "sold", "selling"), ("pass", "passed", "passed", "passing")],
    "heldout2": [("toss", "tossed", "tossed", "tossing"), ("mail", "mailed", "mailed", "mailing"),
                 ("loan", "loaned", "loaned", "loaning"), ("rent", "rented", "rented", "renting"),
                 ("ship", "shipped", "shipped", "shipping"), ("kick", "kicked", "kicked", "kicking"),
                 ("post", "posted", "posted", "posting"), ("leave", "left", "left", "leaving"),
                 ("owe", "owed", "owed", "owing"), ("slide", "slid", "slid", "sliding")],
    "heldout3": [("pay", "paid", "paid", "paying"), ("lease", "leased", "leased", "leasing"),
                 ("fling", "flung", "flung", "flinging"), ("pitch", "pitched", "pitched", "pitching"),
                 ("hurl", "hurled", "hurled", "hurling"), ("roll", "rolled", "rolled", "rolling"),
                 ("bounce", "bounced", "bounced", "bouncing"), ("flip", "flipped", "flipped", "flipping"),
                 ("forward", "forwarded", "forwarded", "forwarding"), ("sling", "slung", "slung", "slinging")],
    "nonce1": [],
    "nonce2": [],
}
ANIMATE = {
    "seen": ["child", "teacher", "student", "doctor", "farmer", "girl", "boy", "woman", "nurse", "pilot"],
    "unseen": ["lawyer", "sailor", "painter", "baker", "dancer", "soldier", "driver", "judge", "tourist", "singer"],
    "heldout2": ["miner", "clerk", "chef", "guard", "poet", "banker", "plumber", "actor", "monk", "waiter"],
    "heldout3": ["tailor", "nanny", "butcher", "jockey", "pirate", "senator", "barber", "cashier", "sheriff", "teenager"],
    "nonce1": ["florist", "surgeon", "janitor", "porter", "rancher", "broker", "glazier", "usher", "vicar", "courier"],
}
THINGS = {
    "seen": ["door", "window", "box", "car", "table", "letter", "bottle", "fence", "bag", "chair"],
    "unseen": ["drawer", "gate", "crate", "truck", "desk", "parcel", "jar", "wall", "basket", "bench"],
    "heldout2": ["coin", "rope", "lamp", "shelf", "blanket", "ladder", "kettle", "mirror", "bucket", "carpet"],
    "heldout3": ["ball", "pen", "key", "cup", "plate", "brick", "towel", "hammer", "pillow", "sponge"],
    "nonce1": ["lantern", "saddle", "funnel", "trowel", "barrel", "anchor", "basin", "candle", "satchel", "goblet"],
}
DATIVE_PER_VERB = {"seen": 3, "unseen": 3, "heldout2": 6, "heldout3": 6, "nonce1": 0, "nonce2": 0}
SEED = {"seen": 1234, "unseen": 1234, "heldout2": 20261102, "heldout3": 20261103, "nonce1": 20261104, "nonce2": 20261105}
ANIMATE["nonce2"], THINGS["nonce2"] = ANIMATE["heldout3"], THINGS["heldout3"]  # isolates the verb lemmatiser (4h H3)


def _pairs_for(agent, v, patient, other_v, animate_patient):
    base = f"The {agent} {v[1]} the {patient}."
    out = [
        (base, f"The {patient} was {v[2]} by the {agent}.", 1, "equiv:passive"),
        (f"The {agent} did not {v[0]} the {patient}.", f"The {patient} was not {v[2]} by the {agent}.", 1,
         "equiv:negated_passive"),
        (base, f"The {agent} did not {v[0]} the {patient}.", 0, "dist:negation"),
        (base, f"The {agent} will {v[0]} the {patient}.", 0, "dist:future"),
        (base, f"The {agent} was {v[3]} the {patient}.", 0, "dist:progressive"),
        (base, f"The {agent} might {v[0]} the {patient}.", 0, "dist:modality"),
        (base, f"The {agent} tried to {v[0]} the {patient}.", 0, "dist:attempt"),
        (base, f"The {agent} {other_v[1]} the {patient}.", 0, "dist:predicate"),
        (f"The {patient} was {v[2]} by the {agent}.", f"The {agent} did not {v[0]} the {patient}.", 0,
         "dist:passive_vs_negated_active"),
    ]
    if animate_patient:
        out.append((base, f"The {patient} {v[1]} the {agent}.", 0, "dist:role_reversal"))
    return out


def generate(lexicon, n_per_verb=3, seed=None):
    rng = random.Random(f"{SEED[lexicon] if seed is None else seed}:{lexicon}")
    rows = []
    for verbs, patients, animate_patient in ((TRANSITIVE[lexicon], THINGS[lexicon], False),
                                             (ANIMATE_VERBS[lexicon], ANIMATE[lexicon], True)):
        for v in verbs:
            for _ in range(n_per_verb):
                agent = rng.choice(ANIMATE[lexicon])
                patient = rng.choice([p for p in patients if p != agent])
                other_v = rng.choice([o for o in verbs if o[0] != v[0]])
                rows += _pairs_for(agent, v, patient, other_v, animate_patient)
    for v in DITRANSITIVE[lexicon]:
        for _ in range(DATIVE_PER_VERB[lexicon]):
            agent, recipient = rng.sample(ANIMATE[lexicon], 2)
            thing = rng.choice(THINGS[lexicon])
            rows.append((f"The {agent} {v[1]} the {recipient} the {thing}.",
                         f"The {agent} {v[1]} the {thing} to the {recipient}.", 1, "equiv:dative"))
            rows.append((f"The {agent} {v[1]} the {recipient} the {thing}.",
                         f"The {agent} {v[1]} the {recipient} to the {thing}.", 0, "dist:dative_roles_swapped"))
    return [{"a": a, "b": b, "label": y, "phenomenon": p, "lexicon": lexicon} for a, b, y, p in rows]


if __name__ == "__main__":
    import collections
    for lex in ("seen", "unseen", "heldout2", "heldout3", "nonce1", "nonce2"):
        rows = generate(lex)
        print(lex, len(rows), dict(collections.Counter(r["phenomenon"] for r in rows)))
        print("  e.g.", rows[0]["a"], "|", rows[0]["b"])
