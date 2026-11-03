"""Simulated persons: a pool of personal habits (how a person expresses an intent differently from general usage), and
persons sampled from it with presentation and working-style preferences, a correction style, routines and open items.
The pool is split: development persons use only 'dev' habits, test persons only 'test' habits."""
import random

from . import world as W

# A habit: in `contexts` this person means `intent` by the expression in `train` / `eval` templates.
# vp: templates are verb phrases wrapped by request frames; otherwise whole messages.
# controls: (context, template, intent) - the same expression where this person means what everyone means.
HABITS = [
    # same verb, a personal meaning in one area
    dict(id="tidy_code", pool="dev", contexts=("code",), intent="code.format", vp=True,
         train=("tidy {code}", "tidy up {code}"), eval=("tidy {code} up", "give {code} a tidy"),
         eval_msgs=("{code} needs tidying.", "{code} could use some tidying."),
         controls=(("docs", "tidy up {doc}", "docs.proofread"), ("files", "tidy away {file}", "files.archive")),
         statement="When I say tidy for code, I mean format it."),
    dict(id="check_code", pool="dev", contexts=("code",), intent="code.test", vp=True,
         train=("check {code}", "check on {code}"), eval=("double-check {code}", "give {code} a check"),
         eval_msgs=("{code} needs checking.",),
         controls=(("docs", "check {doc} for typos", "docs.proofread"),),
         statement="When I say check a code file, I mean run its tests."),
    dict(id="send_docs", pool="dev", contexts=("docs",), intent="docs.export", vp=True,
         train=("send {doc}", "send {doc} over"), eval=("send off {doc}", "get {doc} sent"),
         eval_msgs=("{doc} should be sent.",),
         controls=(("email", "send {msg}", "email.send"),),
         statement="If I just say send a document, I mean export it."),
    dict(id="bump_email", pool="dev", contexts=("email",), intent="email.flag", vp=True,
         train=("bump {msg}",), eval=("bump {msg} up", "give {msg} a bump"), eval_msgs=("{msg} needs a bump.",),
         controls=(("calendar", "bump {event} to {time}", "calendar.move"),),
         statement="Bumping an email means flag it for me."),
    dict(id="ship_code", pool="dev", contexts=("code",), intent="code.commit_push", vp=True,
         train=("ship it", "ship the changes"), eval=("ship this", "get the changes shipped"),
         eval_msgs=("The changes can be shipped now.",),
         controls=(("docs", "ship {doc} to {person}", "docs.share"),),
         statement="When I say ship it, I mean commit and push."),
    dict(id="park", pool="dev", contexts=("docs", "code", "email"), train_contexts=("docs", "code"),
         eval_contexts=("email",), intent="notes.todo", vp=True,
         train=("park {doc}", "park {code} for now"), eval=("park {msg}", "park {msg} for later"),
         eval_msgs=("{msg} can be parked.",), controls=(),
         statement="When I say park, I mean put it on my todo list."),
    dict(id="zap", pool="dev", contexts=("files",), intent="files.compress", vp=True,
         train=("zap {file}",), eval=("zap {file} down", "give {file} a zap"), eval_msgs=("{file} needs zapping.",),
         controls=(), statement="Zap a file means compress it."),
    dict(id="mess_code", pool="dev", contexts=("code",), intent="code.format", vp=False,
         train=("{code} is a mess.", "{code} is such a mess."),
         eval=("{code} has become a mess.", "Honestly, {code} is a complete mess."), eval_msgs=(), controls=(),
         statement="When I say mess for a code file, I mean format it."),
    dict(id="see_docs", pool="dev", contexts=("docs",), intent="docs.share", vp=False,
         train=("{person} should see {doc}.",), eval=("{person} needs to see {doc}.", "{person} ought to see {doc}."),
         eval_msgs=(), controls=(), statement="When I say someone should see a document, I mean share it with them."),
    dict(id="clash_cal", pool="dev", contexts=("calendar",), intent="calendar.move", vp=False,
         train=("{event} clashes with {time}.",), eval=("{event} collides with {time}.", "{event} overlaps with {time}."),
         eval_msgs=(), controls=(), statement="When I say a meeting clashes, I mean move it."),
    dict(id="arrow_export", pool="dev", contexts=("docs",), intent="docs.export", vp=False,
         train=("{doc} -> {fmt}",), eval=("{doc} → {fmt}", "{doc} => {fmt} pls"), eval_msgs=(), controls=(),
         statement="An arrow from a document to a format means export it."),
    dict(id="arrow_file", pool="dev", contexts=("files",), intent="files.move", vp=False,
         train=("{file} -> {folder}",), eval=("{file} → {folder}", "{file} => {folder}"), eval_msgs=(), controls=(),
         statement="An arrow from a file to a folder means move it there."),
    # test pool
    dict(id="tidy_files", pool="test", contexts=("files",), intent="files.delete", vp=True,
         train=("tidy up {file}", "tidy {file}"), eval=("tidy {file} up", "give {file} a tidy"),
         eval_msgs=("{file} needs tidying.",),
         controls=(("code", "tidy up {code}", "code.refactor"), ("docs", "tidy up {doc}", "docs.proofread")),
         statement="When I say tidy up a file, I mean delete it."),
    dict(id="clean_code", pool="test", contexts=("code",), intent="code.format", vp=True,
         train=("clean up {code}", "clean {code}"), eval=("clean {code} up", "give {code} a clean"),
         eval_msgs=("{code} needs cleaning up.",),
         controls=(("files", "clean up {file}", "files.delete"), ("docs", "clean up {doc}", "docs.proofread")),
         statement="When I say clean up code, I mean format it."),
    dict(id="look_code", pool="test", contexts=("code",), intent="code.explain", vp=True,
         train=("look at {code}", "have a look at {code}"), eval=("take a look at {code}", "look into {code}"),
         eval_msgs=("{code} needs a look.",),
         controls=(("docs", "have a look at {doc}", "docs.summarize"),),
         statement="When I say look at code, I mean explain it."),
    dict(id="kill_email", pool="test", contexts=("email",), intent="email.archive", vp=True,
         train=("kill {msg}",), eval=("kill off {msg}", "just kill {msg}"), eval_msgs=("{msg} can be killed.",),
         controls=(("calendar", "kill {event}", "calendar.cancel"),),
         statement="Killing an email means archive it."),
    dict(id="polish_docs", pool="test", contexts=("docs",), intent="docs.shorten", vp=True,
         train=("polish {doc}", "polish up {doc}"), eval=("give {doc} a polish", "polish off {doc}"),
         eval_msgs=("{doc} needs polishing.",),
         controls=(("code", "polish {code}", "code.refactor"),),
         statement="Polishing a document means shorten it."),
    dict(id="ship_docs", pool="test", contexts=("docs",), intent="docs.export", vp=True,
         train=("ship {doc}",), eval=("ship off {doc}", "get {doc} shipped"), eval_msgs=("{doc} is ready to ship.",),
         controls=(("code", "ship it to {env}", "code.deploy"),),
         statement="Shipping a document means export it."),
    dict(id="stash", pool="test", contexts=("files", "docs", "code"), train_contexts=("files", "docs"),
         eval_contexts=("code",), intent="files.backup", vp=True,
         train=("stash {file}", "stash {doc} somewhere"), eval=("stash {code}", "stash {code} somewhere safe"),
         eval_msgs=("{code} should be stashed.",), controls=(),
         statement="Stash means back it up."),
    dict(id="ptal", pool="test", contexts=("code",), intent="code.review", vp=False,
         train=("ptal {code}", "{code} ptal"), eval=("ptal on {code} pls", "Can you ptal {code}?"), eval_msgs=(),
         controls=(), statement="ptal means please review it."),
    dict(id="long_docs", pool="test", contexts=("docs",), intent="docs.shorten", vp=False,
         train=("{doc} is way too long.", "{doc} is too long."),
         eval=("{doc} runs far too long.", "{doc} got way too long."), eval_msgs=(), controls=(),
         statement="When I say a document is too long, I mean shorten it."),
    dict(id="red_tests", pool="test", contexts=("code",), intent="code.test", vp=False,
         train=("The tests on {code} are red.", "{code} tests are red again."),
         eval=("{code} is red again.", "Everything in {code} is red."), eval_msgs=(), controls=(),
         statement="Red tests means run the tests again."),
    dict(id="answer_email", pool="test", contexts=("email",), intent="email.reply", vp=False,
         train=("{msg} needs an answer.",), eval=("{msg} still needs a reply.", "{msg} is waiting for an answer."),
         eval_msgs=(), controls=(), statement="When I say an email needs an answer, I mean reply to it."),
    dict(id="arrow_move", pool="test", contexts=("calendar",), intent="calendar.move", vp=False,
         train=("{event} -> {time}",), eval=("{event} → {time}", "{event} => {time}"), eval_msgs=(), controls=(),
         statement="An arrow from a meeting to a time means move it."),
    dict(id="arrow_share", pool="test", contexts=("docs",), intent="docs.share", vp=False,
         train=("{doc} -> {person}",), eval=("{doc} → {person}", "{doc} => {person}"), eval_msgs=(), controls=(),
         statement="An arrow from a document to a person means share it with them."),
    dict(id="fyi_docs", pool="test", contexts=("docs",), intent="docs.share", vp=False,
         train=("fyi {person} on {doc}",), eval=("{doc} fyi {person}", "fyi {person}: {doc}"), eval_msgs=(),
         controls=(), statement="fyi someone on a document means share it with them."),
]
HABIT_BY_ID = {h["id"]: h for h in HABITS}

ROUTINE_PHRASES = ("the usual", "the Friday routine", "the end-of-day stuff", "the weekly thing", "the morning routine",
                   "the wrap-up")
ROUTINE_STEPS = (
    (("docs.export", {"doc": "the weekly update", "fmt": "PDF"}), ("docs.share", {"doc": "the weekly update", "person": "the team"})),
    (("code.commit_push", {}), ("calendar.remind", {"event": "the standup"})),
    (("files.backup", {"file": "data.csv"}), ("files.archive", {"file": "old.log"})),
    (("email.archive", {"msg": "the newsletter"}), ("calendar.find_slot", {"time": "tomorrow morning"})),
)
CORRECTION_STYLES = ("explicit", "terse", "vague", "rephrase")


def heads(habit):
    """The words a verb-phrase habit starts with ('tidy up', 'have a look at'): general phrasings that start the same way
    are not put in this person's mouth in the habit's areas, since there they mean the habit."""
    out = set()
    for t in habit["train"] + habit["eval"]:
        if habit["vp"]:
            words = []
            for w in t.split():
                if w.startswith("{") or w in ("it", "this", "the"):
                    break
                words.append(w)
            if words:
                out.add(" ".join(words))
    return out


def template_context(habit, template):
    for slot, ctx in (("{code}", "code"), ("{doc}", "docs"), ("{msg}", "email"), ("{file}", "files"), ("{event}", "calendar")):
        if slot in template:
            return ctx
    return habit["contexts"][0]


def sample_person(seed, pool, n_habits=8, n_oneoffs=4):
    """A simulated person: habits from one pool, preferences that differ from the general default in about half the
    dimensions, a correction style mix, a routine and open items."""
    rng = random.Random(seed)
    chosen = [h["id"] for h in rng.sample([h for h in HABITS if h["pool"] == pool], n_habits)]
    prefs = {d: (rng.choice(v[1:]) if rng.random() < 0.5 else v[0]) for d, v in W.DIMS.items()}
    weights = [rng.random() for _ in CORRECTION_STYLES]
    total = sum(weights)
    spec = {
        "seed": seed, "pool": pool, "habits": chosen, "preferences": prefs,
        "correction_style": {s: round(w / total, 3) for s, w in zip(CORRECTION_STYLES, weights)},
        "politeness": round(rng.uniform(0.1, 0.7), 2), "states_habit": round(rng.uniform(0.0, 0.35), 2),
        "routine": {"phrase": rng.choice(ROUTINE_PHRASES), "steps": rng.choice(ROUTINE_STEPS)},
        "open_topics": rng.sample(W.objects("topic", "train"), 4),
    }
    # noise (added after development run 1, see EXPERIMENTS.md): inattentive acceptance of a wrong answer, and
    # one-off meanings - an ordinary phrasing that once meant something else and never again
    spec["inattention"] = round(rng.uniform(0.0, 0.08), 3)
    spec["oneoffs"] = sample_oneoffs(spec, rng, n_oneoffs)
    return spec


def sample_oneoffs(spec, rng, n):
    blocked = set()
    for hid in spec["habits"]:
        blocked |= {(c, hd) for c in HABIT_BY_ID[hid]["contexts"] for hd in heads(HABIT_BY_ID[hid])}
    habit_intents = {HABIT_BY_ID[h]["intent"] for h in spec["habits"]}
    out, used = [], set()
    candidates = [i for i in W.INTENTS if i.split(".")[0] not in ("notes",) and i not in habit_intents]
    while len(out) < n:
        i = rng.choice(candidates)
        ctx, slots = W.CONTEXT_OF[i], W.INTENTS[i][0]
        phrases = [k for k, p in enumerate(W.INTENTS[i][2])
                   if not any(p.startswith(hd + " ") for c, hd in blocked if c == ctx)]
        others = [j for j in W.INTENTS if j != i and W.CONTEXT_OF[j] == ctx and W.INTENTS[j][0] == slots]
        if not phrases or not others or i in used:
            continue
        used.add(i)
        out.append({"intent": i, "phrase": rng.choice(phrases), "meant": rng.choice(others)})
    return out
