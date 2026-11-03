"""The controlled world of Personal Process Learning 0.1: application areas, intents with their slots and presentation
dimensions, object lexicons (split into person-training, person-evaluation and general use), the general phrasings
everyone uses, and sentence frames."""

CONTEXTS = ("code", "docs", "email", "calendar", "files", "research", "notes")

# presentation and working-style dimensions; the first value is the general default
DIMS = {"format": ("prose", "bullets", "table"), "length": ("long", "short"), "confirm": ("ask", "just_do"),
        "plan": ("act", "plan_first"), "options": ("one", "several"), "sources": ("without", "with")}
DEFAULT_PRESENTATION = {d: v[0] for d, v in DIMS.items()}

ANSWERING = ("format", "length")
# intent -> (slot types, presentation dimensions that apply, general phrasings)
INTENTS = {
    "code.format": (("code",), (), ("format {code}", "reformat {code}", "prettify {code}", "run the formatter on {code}")),
    "code.test": (("code",), (), ("run the tests for {code}", "test {code}", "run the test suite on {code}")),
    "code.commit_push": ((), ("plan",), ("commit and push the changes", "commit and push my work", "commit the changes and push them")),
    "code.deploy": (("env",), ("plan",), ("deploy to {env}", "ship it to {env}", "ship the build", "release the build to {env}")),
    "code.revert": (("code",), ("confirm",), ("revert {code}", "roll back {code}", "undo the last change to {code}")),
    "code.review": (("code",), ANSWERING, ("review {code}", "look over {code}", "check {code}", "look at {code}")),
    "code.explain": (("code",), ANSWERING, ("explain {code}", "walk me through {code}", "tell me what {code} does")),
    "code.refactor": (("code",), (), ("refactor {code}", "clean up {code}", "tidy up {code}", "restructure {code}", "polish {code}")),
    "docs.summarize": (("doc",), ANSWERING, ("summarize {doc}", "give me the gist of {doc}", "sum up {doc}", "have a look at {doc}")),
    "docs.proofread": (("doc",), (), ("proofread {doc}", "check {doc} for typos", "clean up {doc}", "tidy up {doc}", "polish {doc}")),
    "docs.shorten": (("doc",), (), ("shorten {doc}", "trim {doc}", "cut down {doc}")),
    "docs.expand": (("doc",), (), ("expand {doc}", "flesh out {doc}", "add more detail to {doc}")),
    "docs.translate": (("doc", "lang"), (), ("translate {doc} into {lang}", "put {doc} into {lang}")),
    "docs.export": (("doc", "fmt"), (), ("export {doc} as {fmt}", "convert {doc} to {fmt}", "save {doc} as {fmt}")),
    "docs.share": (("doc", "person"), (), ("share {doc} with {person}", "send {doc} to {person}", "ship {doc} to {person}")),
    "email.reply": (("msg",), (), ("reply to {msg}", "answer {msg}", "write back to {msg}")),
    "email.send": (("msg",), ("confirm",), ("send {msg}", "send off {msg}")),
    "email.archive": (("msg",), (), ("archive {msg}", "file away {msg}")),
    "email.flag": (("msg",), (), ("flag {msg}", "mark {msg} as important", "star {msg}")),
    "email.forward": (("msg", "person"), (), ("forward {msg} to {person}", "pass {msg} on to {person}")),
    "calendar.create": (("event", "time"), (), ("schedule {event} for {time}", "set up {event} at {time}", "book {event} for {time}")),
    "calendar.move": (("event", "time"), (), ("move {event} to {time}", "reschedule {event} to {time}", "bump {event} to {time}", "push {event} to {time}")),
    "calendar.cancel": (("event",), ("confirm",), ("cancel {event}", "call off {event}", "kill {event}")),
    "calendar.find_slot": (("time",), (), ("find me a free slot {time}", "when am I free {time}", "find time {time}")),
    "calendar.remind": (("event",), (), ("remind me about {event}", "set a reminder for {event}")),
    "files.open": (("file",), (), ("open {file}", "pull up {file}")),
    "files.rename": (("file", "name"), (), ("rename {file} to {name}",)),
    "files.move": (("file", "folder"), (), ("move {file} to {folder}", "put {file} in {folder}")),
    "files.archive": (("file",), (), ("archive {file}", "tidy away {file}", "put {file} in the archive")),
    "files.delete": (("file",), ("confirm",), ("delete {file}", "remove {file}", "trash {file}", "bin {file}", "clean up {file}", "nuke {file}")),
    "files.compress": (("file",), (), ("zip {file}", "compress {file}")),
    "files.backup": (("file",), (), ("back up {file}", "make a backup of {file}")),
    "research.search": (("topic",), ANSWERING + ("sources",), ("search for {topic}", "look up {topic}", "find information on {topic}")),
    "research.define": (("term",), ANSWERING, ("define {term}", "tell me what {term} means", "explain the term {term}")),
    "research.compare": (("topic", "topic2"), ANSWERING + ("options",), ("compare {topic} and {topic2}", "weigh {topic} against {topic2}")),
    "notes.save": (("text",), (), ("note down {text}", "save a note that says {text}", "write down {text}")),
    "notes.find": (("topic",), ANSWERING, ("find my notes on {topic}", "look through my notes for {topic}")),
    "notes.todo": (("text",), (), ("add a todo to {text}", "put {text} on my todo list", "add {text} to my list")),
    "notes.later": (("topic",), (), ("remind me later about {topic}", "let's come back to {topic} later",
                                     "keep {topic} on the list for later")),
    "notes.done": (("topic",), (), ("{topic} is done", "we can drop {topic}", "mark {topic} as done")),
    "notes.open": ((), (), ("tell me what's still open", "list what I have left", "show my open items")),
}
CONTEXT_OF = {i: i.split(".")[0] for i in INTENTS}

# how anyone asks for a different form; labels "pres:<dimension>=<value>"
PRESENTATION_PHRASES = {
    ("format", "bullets"): ("bullet points, please", "use bullets", "as a bulleted list"),
    ("format", "table"): ("put it in a table", "as a table, please", "make it a table"),
    ("format", "prose"): ("just write it out in sentences", "no bullets, plain text", "write it as prose"),
    ("length", "short"): ("shorter, please", "make it shorter", "too long, trim it"),
    ("length", "long"): ("more detail, please", "make it longer", "go into more depth"),
    ("confirm", "just_do"): ("just do it, no need to ask", "don't ask, just do it", "no need to check with me"),
    ("confirm", "ask"): ("ask me before you do that", "check with me first", "please confirm first next time"),
    ("plan", "plan_first"): ("show me the plan first", "plan it out before doing it", "give me the plan before you start"),
    ("plan", "act"): ("no plan needed, just do it", "skip the plan", "just go ahead without a plan"),
    ("options", "several"): ("give me a few options", "show me some options", "I'd like a couple of options"),
    ("options", "one"): ("just pick one", "one recommendation is enough", "just give me your pick"),
    ("sources", "with"): ("with sources, please", "add sources", "include links to the sources"),
    ("sources", "without"): ("no need for sources", "skip the sources", "leave out the sources"),
}
# how anyone states a standing preference (general language, so the general interpreter knows these forms)
PREFERENCE_STATEMENTS = {
    ("format", "bullets"): ("I prefer bullet points.", "I like bullet points."),
    ("format", "table"): ("I prefer tables.", "I like things in tables."),
    ("format", "prose"): ("I prefer plain sentences.", "I like answers written as prose."),
    ("length", "short"): ("I like short answers.", "I prefer brief answers."),
    ("length", "long"): ("I like detailed answers.", "I prefer thorough answers."),
    ("confirm", "just_do"): ("You don't need to ask before doing things.", "No need to ask me first."),
    ("confirm", "ask"): ("Always ask before you do anything risky.", "Check with me before acting."),
    ("plan", "plan_first"): ("I like to see a plan before you start.", "Show me a plan first."),
    ("plan", "act"): ("Don't bother with plans, just act.", "I don't need plans."),
    ("options", "several"): ("I like having options to choose from.", "I prefer a few options."),
    ("options", "one"): ("Just give me one recommendation.", "I prefer a single pick."),
    ("sources", "with"): ("I like sources with research answers.", "I prefer answers with sources."),
    ("sources", "without"): ("I don't need sources.", "Sources aren't necessary for me."),
}

# objects by slot type; persons in training see only "train", evaluation uses only "eval", general data uses all
LEXICON = {
    "code": ["utils.py", "parser.rs", "api.ts", "main.go", "models.py", "router.js", "cache.c", "auth.py", "schema.sql",
             "index.html", "server.py", "client.ts", "build.rs", "queue.go", "views.py", "store.js", "lexer.c", "app.py"],
    "doc": ["the Q3 report", "the proposal", "the onboarding guide", "the budget memo", "the release notes", "the grant draft",
            "the meeting minutes", "the style guide", "the press release", "the user manual", "the pitch deck",
            "the research summary", "the travel plan", "the hiring plan", "the policy draft", "the case study",
            "the weekly update", "the roadmap"],
    "msg": ["the email from Raj", "Mia's message", "the invoice email", "the newsletter", "the thread about hiring",
            "the note from Tom", "the vendor email", "Priya's reply", "the HR announcement", "the client email",
            "Leo's question", "the shipping notice", "the support ticket", "Ana's invite", "the bank email",
            "the survey email", "Omar's update", "the security alert"],
    "event": ["the standup", "my 1:1 with Raj", "the design review", "lunch with Mia", "the sprint planning",
              "the dentist appointment", "the team sync", "the client call", "the board meeting", "the demo",
              "the retro", "coffee with Leo", "the budget review", "the interview", "the all-hands", "the workshop",
              "the gym session", "the parent meeting"],
    "file": ["old.log", "photos.zip", "draft_v2.docx", "notes.txt", "backup.tar", "invoice.pdf", "data.csv",
             "slides.pptx", "report_old.pdf", "temp.json", "export.xlsx", "scan01.png", "archive_2023.zip", "debug.log",
             "budget.ods", "readme_old.md", "dump.sql", "video.mp4"],
    "folder": ["the archive folder", "Documents", "the shared drive", "the old folder", "Downloads", "the project folder",
               "the backup drive", "the team folder", "Desktop", "the cold storage folder", "the finance folder",
               "the media folder"],
    "person": ["Raj", "Mia", "Tom", "Priya", "Ana", "Leo", "Sara", "Omar", "Kim", "Ben", "Nina", "Ravi", "Eva", "Jon",
               "Lily", "Sam", "Zoe", "Arjun"],
    "time": ["3pm", "tomorrow morning", "Friday at 10", "next Monday", "4:30", "noon tomorrow", "Thursday afternoon",
             "9am", "next week", "Wednesday at 2", "this evening", "Monday at 11", "tomorrow at 5", "Tuesday morning",
             "6pm", "the 14th", "after lunch", "Saturday"],
    "fmt": ["PDF", "Markdown", "Word", "HTML", "plain text", "LaTeX", "a slide deck", "EPUB", "CSV", "RTF", "OpenDocument",
            "JSON"],
    "lang": ["Spanish", "French", "German", "Hindi", "Japanese", "Portuguese", "Italian", "Dutch", "Korean", "Tamil",
             "Arabic", "Swedish"],
    "topic": ["quantum dots", "vector databases", "solar panel efficiency", "the history of Jaipur", "sourdough baking",
              "electric bikes", "sleep cycles", "rust lifetimes", "heat pumps", "coral reefs", "index funds",
              "noise-cancelling headphones", "remote work", "language learning", "ancient Rome", "home composting",
              "marathon training", "film photography"],
    "topic2": ["graph databases", "wind power", "the history of Delhi", "rye bread", "e-scooters", "naps", "go channels",
               "gas boilers", "kelp forests", "bonds", "earplugs", "office work"],
    "term": ["entropy", "idempotent", "latency", "amortization", "inflation", "recursion", "photosynthesis", "liquidity",
             "osmosis", "polymorphism", "bandwidth", "half-life", "throughput", "allele", "monad", "torque", "yield",
             "jitter"],
    "text": ["call the plumber", "buy milk", "renew my passport", "book the flights", "pay the electricity bill",
             "water the plants", "email the landlord", "fix the bike", "order printer ink", "call mum", "file the taxes",
             "return the library books", "update my CV", "clean the garage", "pick up the parcel", "send the birthday card",
             "back up the laptop", "cancel the old subscription"],
    "env": ["staging", "production"],
    "name": ["final.docx", "summary.txt", "v3.pptx", "clean.csv", "draft.md", "notes_2026.txt"],
}
SPLIT_TYPES = ("code", "doc", "msg", "event", "file", "folder", "person", "time", "topic", "term", "text")


def objects(slot, part):
    """part: 'train' | 'eval' | 'all'. Persons see disjoint halves in training and evaluation."""
    items = LEXICON[slot]
    if slot not in SPLIT_TYPES or part == "all":
        return list(items)
    half = len(items) // 2
    return list(items[:half]) if part == "train" else list(items[half:])


# request frames; general users use all, a person's training uses TRAIN, evaluation uses EVAL
TRAIN_FRAMES = ("{VP}.", "Please {vp}.", "Can you {vp}?", "{VP}, please.", "{VP} pls.")
EVAL_FRAMES = ("Could you {vp}?", "I'd like you to {vp}.", "Go ahead and {vp}.", "When you get a chance, {vp}.",
               "I need you to {vp}.")
GENERAL_FRAMES = TRAIN_FRAMES + EVAL_FRAMES + ("Would you {vp}?", "Kindly {vp}.", "I want you to {vp}.")


def frame(f, vp):
    return f.format(vp=vp, VP=vp[0].upper() + vp[1:])


def fill(template, slots):
    return template.format(**slots)
