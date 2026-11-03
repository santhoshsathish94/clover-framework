# Personal Process Learning: system design (v0.1)

## 1. Layers

```
message + context
   |
   v
CSI (frozen)              general meaning: propositions, roles, mood, polarity, lemmas; shared by everyone
   |
   v
General interpreter       intent from CSI features, learned once from population data; shared, never updated by a person
   |
   v
Personal understanding    per person: interpretation hypotheses, presentation and working-style preferences,
   |                      routines, open items - each with the experiences that support or contradict it
   v
Assistant decision        intent + slots + presentation, or a clarifying question
   |
   v
Outcome                   the person accepts, corrects, rephrases, answers a question, or asks for a different form
   |
   v
Personal process learning experience -> observation -> hypothesis -> test -> validate / retire -> adapt
```

CSI and the general interpreter are read-only for personal learning: nothing a person does changes them. Each person's
knowledge lives in that person's own store, so one person's habits cannot reach another person or general meaning.

## 2. Experience

One experience = one request and everything that followed until it was resolved:
- the message and its context (application area, the object in focus, time);
- CSI's reading and the features derived from it;
- Clover's interpretation and **why** (the source: general interpreter with its confidence, a named hypothesis, a
  routine, a clarifying question), the presentation it chose and why;
- what happened: accepted, corrected (and how: explicit, terse, "not what I meant", rephrased), the answer to a
  clarifying question, requests to change the form (shorter, bullets, ask first, plan first, options, sources);
- the resolved intent and presentation; explicit statements ("When I say X I mean Y");
- the person's retention choice for it: keep, stop learning, deleted.

## 3. Features (what a hypothesis may depend on)

From CSI: the lemmas of the content words (by part of speech; `tidying`, `tidied` and `a tidy` all give `tidy`),
excluding light words (please, can, get, give, ...) and the names of objects, so that habits are not tied to particular
files or people. From the message: arrow or colon patterns, the kinds of objects mentioned (document, person, time,
format, ...). From the context: the application area. Object identities never enter a hypothesis.

The general interpreter also uses CSI's predicate, mood (request or statement) and polarity. Personal hypotheses do not:
a habit learned from requests ("tidy utils.py") must still be recognised in a statement ("main.go needs tidying") or
another construction ("give main.go a tidy"), where the predicate and mood differ but the lemma does not.

## 4. The learning loop

1. **Reconstruct.** Every interpretation is stored with its source and the features it used.
2. **Observe.** When the outcome reveals the intended meaning (acceptance, correction, rephrase, answer to a
   question), an observation records: features, what Clover understood, what the person meant, and why Clover
   understood it that way.
3. **Hypothesise.** When the general interpreter's reading differs from what the person meant (or was below the
   confidence threshold), the observation creates or extends a hypothesis "in contexts matching T, this person means
   I". T starts as everything the message offered (area, object kinds, content lemmas, patterns) and is generalised by
   keeping only what agreeing observations share - but never so far that T would cover an earlier experience in which
   the person meant something else, and never below one word or pattern. A contradiction in an area where the
   hypothesis has no support narrows it back to the areas that supported it.
4. **Test.** Every later outcome that reveals the intended meaning counts for or against every hypothesis whose trigger
   matches, whether or not it was applied. Whether a provisional hypothesis is also applied before validation is set
   on development persons.
5. **Validate or retire.** Validated after enough support with high precision; retired when contradicted. Thresholds are
   set on development persons (EXPERIMENTS.md).
6. **Adapt.** Matching validated hypotheses override the general interpreter, the most specific first (then precision,
   then support); when nothing personal applies and the general interpreter is unsure, Clover asks, offering matching
   personal candidates first; preferences change how the answer is presented; routines expand into their steps.

Preferences (format, length, confirmation before destructive actions, plan first, options, sources) follow the same
loop: observations come from requests to change the form and from accepted forms. Routines ("the usual") are learned
from the answer to a clarifying question and validated by later use. Open items are tracked from "remind me later" /
"let's come back to" until resolved.

Explicit statements are evidence like any other observation, not rules.

## 5. Retention and deletion

Every derived item lists the experiences behind it; derived knowledge is recomputed by replaying the eligible
experiences, so it never outlives its evidence.

| Choice | Effect |
|---|---|
| Keep | the experience stays and informs learning |
| Stop learning | the record stays; it is excluded from learning; derived knowledge is recomputed without it |
| Delete | removed from the store and from its git history (`git filter-repo`, reflog expiry, `gc --prune=now`); derived knowledge recomputed; a tombstone (salted hash, no content) prevents re-learning from a re-imported copy; known copies (clones, backups) are listed for separate deletion |

## 6. Storage

Per person, a git repository: `experiences/<session>/<n>.json`, `retention.json`, `knowledge/` (derived:
hypotheses, preferences, routines, open items), `copies.json`. One commit per change, as in Clover's existing records.

## 7. How it is evaluated

Simulated persons with known habits interact with Clover in training sessions; then fresh evaluation scenarios -
new objects, new sentence frames and constructions, new contexts for general habits, control contexts where a habit
must not apply, ordinary requests, routines and open items - are answered with learning frozen. Compared: no
personalisation; explicit-statement memory (today's "remember"); conversation memory (retrieve similar past messages);
personal process learning; and ablations (without CSI features, without validation, without context scoping).
Separation, CSI integrity and all three retention choices are tested directly. Details and criteria: EXPERIMENTS.md.

## 8. In the runtime (Phase 2; on by default since 2026-11-03, with a switch)

The human's direction (2026-11-03): "keep the learning on by default as I said this is research we can just give option
to disable not the other way around". The store lives on the person's device (`<base>/personal`); the switch is kept in
the store (`.git/ppl-learning-off`), and while it is off nothing is read or recorded but viewing and deleting work.
`CLOVER_PPL=off` turns it off for a process.

`clover-ai-system/mobile/personal.py` (the laptop Studio's staging tree carries an identical copy), reached through
`semantics.Runtime.personal`: with none, every frame is what it was. What PPL-1 tested, and what it became:

| PPL-1 (simulation) | runtime |
|---|---|
| intents from a closed set of 41 | the capabilities a message asks for (`semantics.asked_caps`), i.e. the tools offered |
| general interpreter: logistic regression over CSI features, with a confidence | semantics' rules; no confidence, so a learning opportunity is a revealed reading that differs from the general one (PPL-1 chose C = 0, the same rule) |
| features: CSI lemmas, object kinds from the world's lexicon, area | CSI lemmas when the CSI service is on, else the words as written (PPL-1's surface ablation); object kinds link / path / number; the device as area |
| revealed meaning: acceptance, correction, answer to a clarifying question | thanks or Helpful; a correction that names what was wanted ("No, look it up online", "That's wrong, search the web"); there are no clarifying questions |
| six presentation dimensions, applied by the simulator | the same six, given to the model as a second system message; "ask before acting" can only add a question |
| routines, open items | not integrated |
| store, three retention choices, deletion through history | the same; knowledge is not stored but recomputed; without git-filter-repo the repository is rebuilt from what remains; deleting a conversation deletes its experiences |

Runtime additions that PPL-1 did not test: a retry right after a personal reading counts against it; corrections are
recognised by their opening ("No, ...", "I meant ..."); requests for another form by fixed patterns. The learner and
its registered settings are otherwise PPL-1's, checked identical on shared experiences. Checked without a model only
(`test_personal.py`); real use is what the consented pilot is for.
