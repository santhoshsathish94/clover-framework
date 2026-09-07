# Changelog

All notable changes to this project will be documented in this file.

## v2.4.0 — 2026-09-08

The home page argues once instead of twice. Each stage's real-world section now carries what should
happen, what actually happened, and the evidence for it in one place. `VERSION` is `2.4.0` and the
site's `?v=` cache strings match it.

**Five sections replace ten.** `#context-misuse`, `#accountability`, `#rollout`, `#outcome` and
`#growth` are gone; their arguments and evidence links moved into `#real-world-context`,
`#real-world-direction`, `#real-world-execution`, `#real-world-outcome` and `#real-world-growth`.
The home page is 22 sections, down from 27. `#clover-teaches` is now headed "What needs to change in
the real world?" and closes on a line about who acts when governments do not.

**A new section, `#systems`.** "What systems should adopt?" — every system was built around humans,
and process built to catch human error does not fit an actor that is not human. Accountability rests
on a named human, the environment enforces what an instruction cannot, and AI actions are monitored.

**The mark has two states instead of three.** The autumn palette and the six-step `data-decay` drain
are removed: green while the framework is explained, grey from the real-world sections to the end.
The `is-decayed` state is deleted entirely — after the merge no section in its range could light a
leaf, so its fallen-leaf rules could never match. The grey accessible name moved to `is-real-world`,
where the colour actually changes, so it now describes the whole grey half rather than part of it.

**AGENTS.md states what it does and does not bind.** A new section says the cycle is written for the
agent to understand where the work sits, not as a process the human must adopt — no stage is required
before starting, no vocabulary is corrected, no stage name is returned where an answer was wanted.
What binds the agent is the boundaries. Four of those are strengthened: intellectual property must be
recorded in the artifact rather than only in the conversation, so the creator can verify what was
taken and refuse it; system-level data exposure must be reported, not only credentials the agent
trips over; the Growth record is offered to the human to keep, correct or discard; and Direction may
not be skipped and explained afterwards.

**Spelling normalized.** The home page carried British forms against the American convention used by
every other page. `organisation`, `licence` and `acknowledgement` now match the rest of the site.

## v2.3.0 — 2026-09-06

The home page gained a sourced argument about why responsibility for AI was sidelined, and a
supporting dossier under `docs/ai-responsibility/`. `VERSION` is `2.3.0` and the site's `?v=` cache
strings match it.

**Seven closing sections** run from `#responsibility` to `#clover-teaches`, one per cycle stage plus
a resolution, followed by `#closing` — the mark whole again in new green with the closing line
beneath it. The site names no company, government or person; the documents name every source.

**A new evidence dossier.** `docs/ai-responsibility/` holds the master claim and eight supporting
papers, plus `docs/responsibility-cannot-be-optional.md`. Every paper ends with an evidence ledger
whose last column states what the source does **not** establish. Every source was opened and read
rather than taken from a search summary.

**Claims were tightened against their sources.** Three citations in
`08-incidents-and-adoption-outcomes.md` were corrected: a claim about congressional testimony that
neither cited article contained, an unverifiable detail attributed to a court order, and an
unattributed casualty figure. Two remaining limits are recorded rather than papered over — the Minab
strike establishes no AI involvement, and the enterprise-adoption figure is a single contested study.

**Visual:** accent green is reserved for the opening line of each section; `.evidence-link` puts the
source row below the argument in a quieter register; the decayed mark recovers to `#leaf-revive` at
the end of the page.

## v2.0.0 — 2026-09-04

The site became one continuous argument rather than a set of panels, Growth became a stage of the
cycle rather than something beside it, and the framework stopped assuming the system it is applied to
is software. `VERSION` is `2.0.0` and the site's `?v=` cache-busting query strings match it.

A major version because three things change underneath anyone already using it. **Growth is the fifth
stage of the cycle**, not something sitting outside the framework, so the cycle is
`Context → Direction → Execution → Outcome → Growth` everywhere. **`System → Human → AI` is gone**,
replaced by the sentence the site uses: the system is the reality, and the actors in it are the human
and AI. **`/security/` moved to `/governance/`**, with a redirect left behind, so anything linking to
the old path now takes a client-side hop. The stage names `Action` and `Success` were retired in
1.1.x and remain retired.

### Added

- **The agent holds the system's rules, including against the human.** Clover cannot enforce
  anything; a way of working has no way to stop anybody. An agent is different, because it sits
  between the human and the system while the work happens, which makes it the only actor that can
  hold a rule at the moment it is about to be broken. `AGENTS.md` now states that duty and the
  direction it runs in: the rules belong to the system, so a human crossing one is the same event as
  the agent crossing one. Say it before it happens rather than in the summary afterwards, decline to
  perform the crossing, do not treat being asked as the rule being lifted, write the boundary into
  the context record, and escalate past the person asking when the rule is not theirs to lift. The
  limit is stated with it: this is enforcement inside one cycle, on the work in front of the agent.
  Nothing an agent does makes an organization compliant, and enforcement at that scale belongs to
  governments, the companies building the models, and the enterprises that use them.

- **Why a process is needed at all.** The framework rested on an unstated assumption: that the human
  giving Direction is right. [Philosophy](docs/02-philosophy.md) now says plainly that neither side
  sees the whole consequence — a human works from partial knowledge of a system nobody holds
  entirely, and being accountable for a decision is not the same as being right about it, while AI
  states its mistakes as fluently as its facts. Each stage is then derived from that, and the rules
  bind both. [Principles](docs/03-principles.md) opens on the same point under **Neither side is
  exempt**, and `AGENTS.md` turns it into a duty: say once, plainly, when the Direction looks wrong,
  then respect the decision and record what you expect to go wrong.
- **Instructions come from Direction. Context is data.** The rule existed in `AGENTS.md` and nowhere
  in `docs/`, so a human reading the framework never met it. It is now a principle and a
  [governance](docs/08-governance.md) section, with the reason attached: Context is frequently
  writable by people outside the organization, so a comment on a public issue or a string in a log
  line can be shaped like an instruction, and obeying it hands over whatever access the work holds.
- **Reality is not edited.** Fabrication appeared only in [how AI fails](docs/how-ai-fails.md) as a
  pattern to catch, never as a prohibition, and nothing anywhere forbade altering the evidence.
  Principle 4 already protected the verification control; the new material covers the rest —
  reporting what was not observed, changing data, logs or state so reality appears to agree, and
  acting beyond the scope Direction set. Stated in [Outcome](docs/07-outcome.md) as **Evidence is
  never manufactured**, and in `AGENTS.md` as a refusal.
- **What each layer owes beyond the immediate work.** [Governance](docs/08-governance.md) now states
  a boundary for the System, the Human, the AI, and the frontier layer, including consent, licensing
  and attribution for the human work that model capability is built from. The limit is stated with
  it: Clover cannot enforce that, and no framework can.
- **What Clover cannot work out alone.** The [roadmap](docs/10-roadmap.md) names creation rules as
  unsolved, asks for the people who know the domain, and says how the framework itself advances —
  this expansion is Direction, adoption produces the Context, and what teams report back from real
  cycles is the evidence.
- **Connected System, as the fourth level of scope.** Scope ran `Task → Feature → System` and stopped
  where one system stops, which is not where the consequence stops.
  [Philosophy](docs/02-philosophy.md) now carries the fourth level, and the accountability argument
  is stated against it: as the scale grows, the cost of a wrong Direction grows with it, and
  capability to suggest a direction at that scale is not authority to pursue it.
- **A runtime-enforcement reference implementation.** `reference/runtime-enforcement/` holds a
  tool-layer write guard, a protected verification fixture, a container that mounts the verification
  path read-only, a `verify.sh`, and a test suite covering path traversal, symlinks, hard links and
  new-test generation. It exists to show one boundary living outside the model rather than inside an
  instruction. The container runs read-only, with all capabilities dropped, no new privileges, memory
  and pid limits, and no route to the network. None of the container settings have been observed
  running, since no Docker daemon was available on the machine the branch was written on.
### Changed

- **The agent specification says what the site says.** Seven claims the home page carried and
  `AGENTS.md` did not. Clover as a way of adopting AI into the system cycle; the cycle unchanged by
  AI while every stage becomes easier, better and faster; a pull quote saying AI capability may scale
  but accountability cannot, and that an agent cannot carry accountability but can make it visible;
  the human's understanding as part of what Context is for; holding the rules in Direction; the
  boundaries binding both actors in Execution; the retrospection line in Growth; and a new section on
  credit staying with whoever earned it.
- **The documents say what the home page says.** Nine documents answered *what did AI change* with
  only the loss of accountability, while the home page answers it the other way round first. Each of
  them keeps its accountability paragraph and now carries the positive half before it, so
  accountability reads as the one thing AI cannot do rather than the only thing AI did.
  `System → Human → AI` is gone from the documents in favour of the sentence the site uses — the
  system is the reality, the actors in it are the human and AI — because an arrow chain reads as a
  sequence or a hierarchy. Seventeen occurrences across fourteen files. Five site-only claims are now
  in the documents, and [governance](docs/08-governance.md) names who should enforce at real-world
  scale.
- **Getting started, the quickstart and the brief describe any system.** They assumed a codebase from
  the first line, so a reader arriving from a home page that never mentions software met a different
  framework on the second page. The problem to start with no longer has to be software, setup step
  one is read-only access to whatever the system keeps rather than an MCP server in front of a
  repository, and step three is to start where a mistake is cheap rather than in development. MCP is
  kept and named as the software route. The [orchestration brief](templates/orchestration-brief.md)
  says in its header that nothing in it assumes software.
- **Security is now [Governance](https://cloverframework.com/governance/).** The page was always
  about access, attribution, approvals and delegated execution; security was the narrower name. It
  moved from `/security/` to `/governance/`, with a stub left behind so the old URL still lands, and
  its software assumptions are gone: reading across code, tickets, logs and data became reading
  across every part of the system at once, and *there are secrets in our code* became *there are
  things in there we would rather nobody read*.
- **The evidence page names what it is evidence of.** The case studies are software work, and
  rewording them would have falsified the one thing that page holds, so the evidence is untouched.
  The framing says plainly that everything on the page is software work, that this is where the
  author works, and that no cycle outside software has been run and recorded yet.
- **Five glossary terms stop assuming software**, in both copies. Orchestration environment,
  MCP server, Telemetry, Non-production and Autonomy.
- **Growth is a stage of the cycle, not something outside it.** The framework was four stages with a
  growth clover sitting beside them. It is now
  `Context → Direction → Execution → Outcome → Growth`, and Growth is where what the Outcome taught
  is written back so the next cycle starts from it. The change reaches the mark on every page, the
  five leaves, [the framework](docs/04-framework.md), both glossaries, the four case-study and
  reference diagrams, and the reference implementations, each of which now states its own Growth.
- **The home page is one continuous argument in nineteen sections.** It opened as a set of panels
  and cards. It now runs from what Clover is, through what AI changes and why humans matter, through
  a section for each of the five stages in the author's own words, a worked example, scale, evidence,
  five sections on what each stage means in the real world, and what the real world should enforce.
  One white background throughout, no tints, no dividing rules, and no eyebrows.
- **One clover mark serves the whole page.** The five-leaf mark is pinned beside the text on a
  zero-height sticky rail, turns one full revolution across the length of the page, and lights the
  leaf whose section is being read. Rotation is applied inside the `viewBox` rather than to the
  element, because rotating the element grew its box to the diagonal and pushed the page sideways.
  From the evidence section onward the mark ripens from green to autumn and its leaves repoint from
  the stages to the closing sections, with each `aria-label` taken from the heading it now leads to.
- **The site says what the docs say.** The tagline, the stage names, the accountability claim and
  the definitions were carrying different words in different places. One tagline now identifies
  Clover everywhere, `Success` is gone as a stage name, and the glossary on the site and the glossary
  in `docs/` are kept in step.
- **Section prose, headings and the nav are set at seventeen pixels or above.** Body copy inside a
  section, sub-headings, the nav links and the brand were all below the size the rest of the site
  reads at.
- **The nav opens with Clover, and clicking it on the home page scrolls rather than reloads.**

### Removed

- **`CLOVER-CONTEXT.md` is no longer checked in.** The framework asks for a context file beside the
  work, and this branch keeps one. It is not published. A working context file records where the
  framework is weak, which assumptions have not been tested and where the next attempt is likely to
  fail, which is exactly what makes it useful to whoever is doing the work and exactly what makes it
  the wrong thing to hand a reader. It is in `.gitignore` and stays on the machine doing the work.
  What it teaches reaches the repository through the documents it changes, which is what Growth is
  for.
- **`assets/widening-what-ai-determines.svg`**, which nothing referenced.
- **The model review protocol.** `docs/evaluations/model-review-protocol.md` set out how to publish a
  reproducible model rating. It was procedurally careful — frozen commit, shared prompt, raw response
  kept, mechanical scoring — and it still could not produce a number worth publishing. There was no
  baseline and no blinding, so a model asked to rate a framework it is told is the subject will score
  it high; no repeated sampling, so a single run stood in for a distribution; and the categories and
  their weights were written by the project being scored. A protocol that cannot separate the
  framework's quality from its own framing should not exist rather than be run carefully.
- **The dead `.run` component** in `site/styles.css`. Fourteen rules for a worked-example layout that
  no page has used since the example was rewritten in prose, and the one place on the site still
  carrying `Success` as a stage name.
- **From the home page**: the hero buttons, the How It Works nav link, the Why now section, the
  `System → Human → AI` arrow chain, the green stage labels, the scale worked examples, the five
  enforcement asks, and the closing restatement section. The arrow chain is deliberate — it remains
  in `docs/`, where the reader has the room for it, and stays off the site.

### Fixed

- **The React case study contradicted its own sourcing caveat.** It said the quantitative evidence
  came from a public reproduction rather than the private application, and then gave pod-level object
  counts and a fleet A/B, both of which are production observations. The caveat now separates the two
  kinds of evidence and says which can be reproduced by a reader. The same pass brought the pull
  request up to date: it is two files and +20/−1 rather than one file and +11/−0, because review moved
  the stack-limit toggle into a helper with a `try/finally` and added a regression test; three people
  have since reproduced the result independently, one of them on a production storefront; a React
  collaborator has objected that the regression test asserts an invariant rather than a reclaim, and
  that objection is recorded rather than answered. The A/B evidence now states what it does not
  establish, since setting the stack limit process-wide suppresses capture at every site rather than
  at the one the fix changes.
- **A hard link defeated the write guard.** `resolve()` follows symbolic links, but a hard link is the
  file under a second name and there is nothing to follow, so `src/notatest.js` pointing at
  `tests/example.test.js` was writable. The policy now compares filesystem identity against the
  verification artifacts under the protected root, and a test covers it. Path folding also normalizes
  to NFC before comparing; every protected name is ASCII today, so that guards the list rather than a
  live bypass, and the test says so.
- **`is_verification_path` reported False for anything outside the protected root.** `authorize_write`
  was safe because it checks the root first, but a caller using the helper on its own was told a test
  file outside the root was not a verification path. It now matches on the whole path instead.
- **The container was a boundary in name only.** It ran on a writable root filesystem with every
  capability, no privilege ceiling and no resource limits. It is now `read_only` with a `tmpfs` for
  `/tmp`, `cap_drop: ALL`, `no-new-privileges`, and memory and pid limits, alongside the read-only
  verification mount and the internal network it already had.
- **The reference did not say which of its two layers is the boundary.** The Python policy is a gate
  that only protects against code that calls it; a plain `open()` goes straight past. The mount is
  what refuses the write. The README now says that in the opening paragraph rather than leaving a
  reader to infer it from the entrypoint script.
- **`verify.sh` still said `Success`**, and the site validator no longer checked the `/security/`
  redirect stub after the page moved. Both corrected.
- **Two claims that reached past their evidence.** [How AI fails](docs/how-ai-fails.md) said every
  failure is caught by evidence rather than by better prompting; prompting demonstrably reduces
  several of them, and what it cannot do is tell you whether it worked this time, which is the actual
  argument. [The orchestration environment](docs/orchestration-environment.md) said reading cannot
  corrupt anything, which is true and reads as though read-only were harmless; reading is the whole
  surface for finding credentials and assembling a picture no single person has.
- **A missing failure mode.** [How AI fails](docs/how-ai-fails.md) covered fabrication and unchecked
  success but never named the case where an agent adopts a goal from something it read. It is now the
  ninth mode, caught by Direction, with the point that a redirected agent does damage without needing
  write access.
- **The runtime-enforcement tests could not run.** The suite imported
  `reference.runtime_enforcement`, but the directory is hyphenated and therefore not importable, and
  no `conftest.py`, `__init__.py` or packaging file existed anywhere. The command printed in the
  reference README failed with `ModuleNotFoundError`. A `conftest.py` now puts the package directory
  on the path, and the symlink test skips with a reason on platforms that refuse symlink creation
  rather than failing the run.
- **The Outcome rename reached six files out of forty-five.** `Success` survived as the stage name in
  the site, the quickstart, both glossaries, the video material, examples, templates, both case
  studies and four SVG assets, so the site said `Context → Direction → Action → Success` while the
  README said Outcome. 131 occurrences renamed across 33 text files and 4 SVGs. The changelog keeps
  its history, and the sentences that deliberately contrast the two names are untouched.
- **`docs/07-success.md` renamed to [`docs/07-outcome.md`](docs/07-outcome.md)**, with all nine
  inbound links repointed.
- **Three broken anchors in [field practices](docs/field-practices.md).** Links pointed at principle
  headings that had been renamed. Repo-wide markdown links and anchors now all resolve.
- **The social card said the wrong thing, and drew the wrong clover.** `assets/social-preview.svg`
  described an "AI Orchestration Framework" ending in Success, and its mark was built from four
  leaves at 90 degrees while the framework has five. Both are corrected, and
  `assets/social-preview.png` has been re-rendered from the source at 1280×640.

## v1.1.3 — 2026-08-29
### Added

- **The hero says what kind of thing Clover is.** "A way of working rather than software. There is
  nothing to install, and it runs on top of the systems a team already has." On GitHub, "framework"
  reads as a library, and "AI orchestration" is a category that mostly contains software, so a
  developer could reasonably arrive expecting an SDK. The page already answered that under **What it
  is not**, five screens down, which is after the point the wrong impression forms. The repository
  description on GitHub now carries the same line, since that is the first thing anyone reads.
- **A step on what the written context is actually worth.** The site mentioned context files eight
  times and every one of them framed the benefit as continuity between cycles. The consequence that
  matters more — the next pass does not have to be the same agent, the same tool or the same session,
  because the picture is written down beside the code rather than held in a conversation — appeared
  in four documents and on no page. It closes the argument the page opens with: the context used to
  leave when the human did, and now it stays when the agent does. Getting started says it too.
- **A step on what happens when Success does not hold.** The sharpest claim in the framework — that a
  second attempt on the same information lands in the same place, faster — was in the docs and nowhere
  on the website. The lucky clover now ends on it, and the leaf that lights up while you read it is
  Context, so the loop is shown rather than described.
- **How much the agent gets to decide, on the security page.** A review that settles what an agent can
  read still has to settle how much of the path it works out for itself. The three rules were in
  governance and reachable from the site only through a link: results decide rather than confidence,
  blast radius overrides track record, and it is granted per context rather than globally.

### Fixed

- **The README lost the claim that context outlives the agent.** The rewrite into an index dropped
  "that summary is what lets any agent pick the job up, so no single agent has to hold the work",
  which is one of the few things about the framework that is genuinely not obvious. Restored.

## v1.1.0 — 2026-08-29
### Added

- **Each clover act opens with a heading saying what it is** — three leaves and where most teams are
  today, four leaves and the one to adopt, five leaves and what adoption leads to. The acts were
  separated only by a line of small text before, so a reader could not tell at a glance which clover
  was which. Step headings dropped to `h4` under the act `h3`, giving the page a real outline.
- **The clover story.** The three clover sections and the closing question became one scrolling
  story: a clover pinned beside the steps, growing from three leaves to four to five as the argument
  does, with the leaf for the current stage lit. Each act has its own clover, so a leaf always jumps
  to a step in the act being read. Every leaf is labeled with the stage it stands for, and the label
  lights with the leaf. The four-stage act says only what *changes* when Context arrives rather than
  explaining Direction, Action and Success again. It ends on the unknown clover, where the whole mark
  goes to ink.
- **The common clover now says what is missing from it.** Two of its three stages rest on one human,
  who supplies both the Direction and the context behind it, which leaves very little for AI to act
  on. The lucky clover answers that: Context goes first because it is already there, in the systems
  the organization runs, so the work starts from reality rather than from somebody's memory of it.
- Scrolling drives the highlight and clicking a leaf jumps to where that stage is introduced.
  **Scrolling itself is never taken over.** The highlight is the step nearest the middle of the
  viewport, which reads the same going up as going down. The leaves are SVG anchors, so they are
  keyboard reachable, and without JavaScript every clover shows in full and every step reads as
  prose.
- **The site is six pages instead of one.** Getting started, Security and Evidence each have their
  own page, so a security lead can be sent `/security/` rather than an anchor two thirds of the way
  down. The home page keeps the whole argument — three stages, then four, then five, and the worked
  example — because that arc is the thing that would break if it were chopped up.
- **The setup instructions moved out of the lucky clover and into Getting started.** Read-only MCP
  servers, scoping to the human, starting in development. It was instruction sitting inside a story;
  the lucky clover now makes the argument and hands off.
- **`site/check.ps1`.** Six pages with no build step means the header, nav and footer are copied six
  times and will drift. It resolves every nav link to a site-root path and fails if the pages
  disagree, checks the asset versions match, and follows every relative link and every anchor.
- **Buttons in the hero.** Read the story, Getting started, and a plain link to the evidence. The
  page opened on prose with nothing to act on; there is now somewhere obvious to go from the top.
- **A section between the capabilities and the story, saying why any of this needs to change.** The
  page went straight from what AI can do to a three-leaf clover, with nothing joining them. It now
  says that AI went from finishing the line you were typing to working through a whole task on its
  own, faster than anybody's habits could change; that the habit which stuck is the common clover,
  where the human supplies the context by hand; and that the trade has gone the wrong way, because
  the context is already sitting in the systems the organization runs and AI can read them directly.
  The human moves from being the source of the context to pointing at it. It is the first place on
  the page where the four stages are named in order.
- **`.band--wide`**, which drops the 68ch measure so a section's prose runs the full width of the
  wrap. Used by that section only.
- **Four diagrams on the evidence page, and the five field lessons.** Each case study and each
  reference implementation now carries a diagram, and the lessons from running this on real work sit
  under them: focus beats parallelism, read the system before fixing it, a workaround is not the
  destination, you do not have to hold the context to be accountable for it, and write the context
  down or pay for it again.

### Removed

- **The "Pick a leaf, or keep scrolling" hint** under the first two clovers. The story's opening line
  already says it once, which is enough.
- **The five PNG infographics.** Four were replaced by the SVG diagrams above, and one,
  `ai-orchestration-lifecycle.png`, had not been referenced by anything for some time. Together they
  were 7.1 MB of pictures that disagreed with the documents around them, and leaving them in place
  invited somebody to use one again. `assets/` is now 195 KB in total, most of it the social preview.

### Changed

- **The four infographics were replaced, because they contradicted the documents they illustrated.**
  The two reference-implementation images were built on "Opportunity → Understand → Plan → Execute →
  Proof → Grow", the six stages that were deleted from the framework, and they were the last place
  those stages survived anywhere in the repository. The migration image showed **Cutover** as a
  completed step when the production cutover has not run, and repeated "16× faster" and "0
  vulnerabilities" without the case study's own warning that those are local measurements taken on one
  developer machine. The memory-leak image said the fix was contributed to **Next.js** when it went to
  React, claimed it was **verified in production** when only the mitigation ever ran there, and had
  two labels overlapping each other.
- They are now hand-written SVG in the site's own palette, with no icon art, and each one ends on a
  strip naming what the work does not show. Four files at about 4 KB each replace five PNGs at about
  1.4 MB each, and the same file serves the case study and the website.
- **The two diagrams in the framework doc were redrawn to match.** The capability progression ended on
  a box labeled "Learning", eighteen lines below the section that names the fifth leaf **Growth**, so
  the same document had two words for the same idea. It now ends on Growth, drawn dotted the way the
  growth clover draws it, and the file no longer has "model" in its name.
- The second one was the deleted autonomy ladder — five boxes marching to "Goal-Directed Autonomous
  AI", with a caption underneath insisting it was not a progression to climb. It was the last picture
  of a ladder left in the repository, and it illustrated a claim the surrounding prose does not make.
  It has been replaced by what that section is actually about: what moves is how much of the path AI
  determines, from drafting steps a human approves, through executing an agreed plan, to planning
  within stated constraints. Who owns the objective, the constraints and the outcome does not move,
  and the three rules that decide how far it widens sit underneath.
- **The README is an index now, rather than a second copy of the framework.** It re-explained all
  four stages at length, restated the arc from three leaves to five, and walked through the MCP
  setup, so it competed with the documents it was supposed to point at. It now opens on the clover
  mark, tells the same short story the website tells in three paragraphs, keeps the four-stage table
  and the three steps to start, and puts everything else in linked tables: the framework, putting it
  to work, and the evidence and lessons. A quarter fewer words, and every document in the repository
  is reachable from it — `docs/clover-origin.md` was not linked from anywhere before.
- **AI capabilities is about the range of the work, rather than a scoreboard.** The section listed
  four job functions and claimed AI was past what any individual human can do. It now describes the
  breadth plainly — the same tool reasons like a developer, an operator, a security reviewer, a
  tester, an analyst and a writer — and closes on the point that covering that range would normally
  take several people. It is short on purpose. The argument about what AI is missing belongs to the
  story that follows it.
- **Headings that say what the section is for.** The bridge is "AI evolved. Let's change the way we
  use it." and the story it hands to is "What should we change?". The story used to be called "Three
  stages, then four, then five", which described the picture rather than the question the reader is
  there to answer. Its opening line now says what the clovers are doing: each one is a way of
  working, each leaf is a stage in it, and the first is where most teams are today.

- **Movement between places on the site is eased.** Clicking a nav link used to arrive almost as
  abruptly as no animation at all, because the browser's own smooth scroll runs at a fixed speed no
  matter how far it has to go. Anchor scrolling now eases in and out, and the duration scales with the
  distance, so a jump from the hero to the story takes about a second and a short hop still feels
  immediate. Same-origin navigations cross-fade instead of flashing white.
- Both are off when the reader asks for reduced motion, and both degrade to what the site did before
  in browsers that do not support them.
- **The nav is seven plain links, and the first one is "The Story".** It was a "Framework" dropdown
  holding the three clover sections and the worked example, which meant the first thing in the nav
  could not be clicked — it only opened a panel. It now goes straight to the story, matching the hero
  button. The dropdown's CSS, its JavaScript and its mobile accordion went with it, and the story
  section labels its own leaves, so the deep links it held are still one click away once you are there.
- Fixed a dead `closeAllGroups()` call left behind on the window resize handler, which would have
  thrown a ReferenceError on every resize.
- Security is reachable from the nav, which it was not before the six-page split.
- Every link now points at `santhoshsathish94/clover-framework` after the repository rename.

### Fixed

- **The Growth leaf never appeared to highlight.** The rule existed but filled the active leaf with
  the same color every inactive leaf already had, so nothing visibly happened when it was the step
  being read.
- **"Two of the three stages rest on one person"** and two nearby sentences said *person* where the
  framework says *human*. Direction is held by a human throughout, and the page now says so.
- **The author and glossary pages had no Open Graph tags**, so sharing either one produced a bare
  link with no title, description or image. Both now carry the same card as the other four pages.
- `<main>` takes `tabindex="-1"` on every page, so activating the skip link moves focus rather than
  only scrolling.
- A dead `closeAllGroups()` call on the window resize handler, left behind when the nav dropdown was
  removed, which would have thrown a ReferenceError on every resize.

## v1.0.2 — 2026-08-28

### Changed

- **The site has its own domain, cloverframework.com.** `site/CNAME` carries it, and the canonical
  links, `og:url` and the social card now point there rather than at the GitHub Pages subpath. The
  old address keeps working and redirects.
- **The header reads "Clover Framework"** rather than "Clover four stages".

## v1.0.1 — 2026-08-28

### Fixed

- **The navigation on phones.** Nine links wrapped onto three rows inside a sticky header, so the
  header alone took 183px — nearly a quarter of the screen, on every screen, all the way down the
  page. Below 1000px the links now collapse behind a **Menu** button and the header is 52px. The
  panel closes when a link is followed, when Escape is pressed, and when the window widens past the
  breakpoint. The button is hidden without JavaScript, where the nav stays open as before.
- **1000px is the breakpoint** because that is the narrowest width at which all nine links still sit
  on one row. An earlier 860px left a band of widths where the desktop nav wrapped to two rows.
- **Versioned `styles.css` and `app.js` URLs.** GitHub Pages sends `Cache-Control: max-age=600` on
  every file and they expire independently, so a returning visitor could get new HTML paired with a
  ten-minute-old stylesheet. That happened on the 1.0.0 deploy: the clover marks rendered unsized and
  the header collapsed, while the server was serving entirely correct files.

## v1.0.0 — 2026-08-28

### Renamed and rewritten as Clover

The framework is now **Clover**, an AI Orchestration Framework connecting real-world Context, human
Direction, AI-driven Action, and validated Success into a repeatable cycle. Four stages replace six:
Understand became Context, Opportunity became Direction, Plan and Execute merged into Action, and
Proof became Success. The six principles became five.

Context comes first. The common way of working is Direction → Action → Success, where context is
whatever the human remembers to hand over, so it arrives as a consequence of the direction. Clover
starts with Context because the systems are already running before anyone asks for anything, and
Direction is then given against what is actually there. Iteration feeds Context too: after each
success and each failure, the context files are written before the next attempt.

The names carry the argument, so they are worth stating. **Context** rather than Understand, because
understanding happens inside a head, while context is the material a system has to work from.
**Direction** rather than Control, because control implies people can perfectly steer a system that
keeps getting more capable. **Action** rather than Plan plus Execute, because the plan usually
changes once the work meets reality and splitting the two invites a plan that gets quietly
abandoned. **Success** rather than Proof or Results, because the question is whether the intended
outcome occurred and whether the real environment said so.

The rename ran through the repository as one change: the docs, `AGENTS.md`, the quickstart, the
brief template, the worked example, the issue and pull request templates, the case studies, the
website, and the README. Both case studies are retold through the stages with every fact left
as it was, including the two that are still open — the migration's production cutover has not run,
and the upstream React pull request is CI-green and not merged.

The clover carries the argument. Three leaves is the common clover, how AI is used almost everywhere
today. Four is the lucky clover, where Context becomes the systems an organization already runs and
arrives before the direction, and those four stages are the framework. Five is the growth clover:
Grow became **Growth**, and it sits outside the framework. It is what AI learns out of the four
stages, it belongs to the frontier AI companies and the volume of data everyone's usage generates,
and nobody in an organization operates it. Where repeated Growth ends is followed separately in
`hypothesis/ai-future.md`, kept out of the framework material, and it is a question rather than a
prediction.

This is a breaking change to the framework's vocabulary. Anything written against the six stages
needs remapping.

### Added
- **Rewrote the website's security and governance section around three separate questions**, because the old version mixed them together and was hard to reason about: *what is actually happening* (no new access, no new controls, and one real change — the layer reads across code, tickets, logs and data at once, and that aggregation is the genuinely new thing), *what needs to happen* (five concrete actions), and *what is not solved yet*. Adds the objection most security reviews actually open with — "there are secrets in our code" — and answers it directly: that is a true statement about your systems, not about AI. The credential is already readable by everyone with repository access. Declining AI does not remove the risk, it removes the thing most likely to find it. Rotate, redact, close the path, and the objection is gone while the system is genuinely safer. *The dangerous thing is not an agent reading a secret; it is a secret sitting there unread for three years.*
- **Monitoring is now framed as the primary recommendation rather than an open problem**, on the site and in `docs/08-governance.md`. Record every system an agent touched, compare what it did against what it was asked to do, and alert when it reaches outside the task — a repository unrelated to the ticket, a table it had no reason to query, a write where it was granted read. The honest caveat stays (tooling barely exists), but with the addition that most teams already collect the raw material in access logs and audit trails, so a coarse first version is tractable. It is the only control that operates *while* the agent is working.
- **A walkthrough section on the website, placed after the agent file** — the missing half of the story. Everything else describes the model; this runs one real problem through it end to end: new listings stopped appearing after a data load, the overnight investigation had already failed, and the team that built the synchronization had left. It shows what each piece of access actually answers (the repository, work tracking, logs, the datastore, a non-production environment, the running application), then the six stages with the human/AI split marked at each one, and closes on where the human matters — state the outcome so there is something to diverge from, notice divergence early while it is one wrong assumption rather than a day built on it, hold the approvals, and decide when the evidence is genuinely enough. "You stop doing the retrieval and start doing the direction."
- **An "agent file" section on the website, placed directly after the lifecycle** rather than buried in the closing steps. It explains what `AGENTS.md` is and why it is the intended route in: a person reading the docs takes an afternoon, an agent takes one file, and the person learns the loop by watching it run on their own work. Three tabs — what it makes an agent do, what it stops, and how it teaches you. Added to the primary navigation, and the Start step no longer repeats the explanation.
- **Adoption marked as delivered** in the README status, rather than pending. It is covered by the adoption guide, the orchestration environment doc, and the agent file, which together carry the model into a team without anyone having to read a specification first.
- **"How to carry yourself" in `AGENTS.md` section 1 — humility and courtesy as a working goal.** An agent that is technically right and exhausting to work with has failed at the part that matters. Assume you are the one more likely to be wrong; treat the person's account as outranking your reading of an artifact *about their own world*, since documents go stale and they were there — though not for conclusions you can check yourself; ask rather than accuse; admit error in one sentence without performing contrition; verify what genuinely matters and take the rest in good faith. **Courtesy governs how you say something, never whether you say it** — soften the delivery, never the substance, because an agent that goes along with whatever it is told is worth nothing and the person can tell. Guide rather than concede: if the direction will not work, say so once with the reason and a better option beside it, then respect the decision, note what you expect to go wrong, and get on with the work. The person should finish feeling helped, not inspected — and better off for having disagreed with you.
- **`AGENTS.md` section 9 — teach the person you are working with.** Most people who encounter this framework will encounter it through an agent, not through `docs/`, because asking is faster and clearer than reading a specification. The agent is therefore the most likely teacher whether or not anyone planned it. The section instructs it to name the stage it is in and why, give the reason behind every request for access or approval, offer to set the flow up rather than wait to be asked, explain deliberate-looking hesitation, and match the person's experience level. Bounded by two limits that pull against each other: teaching is not lecturing, and teaching is not criticising — correct the work rather than the person, raise a concern once and let it go, and do not narrate every reservation, since flattery and nagging are two ways of failing the same person.
- `AGENTS.md` — the entire operating model written as **instructions for an AI agent**, in one self-contained file. Covers the six stages, the session-start context read, evidence gathering, a self-check against known failure modes, stating what the evidence actually was, the limits of an agent's authority, teaching the person it works with, and what to write back before finishing. A human needs the full framework; an agent can be pointed at this one file.
- `site/llms.txt` — the site made legible to models: a short description and curated links to the raw markdown, grouped by core, practice, and evidence, telling an agent to read `AGENTS.md` first.
- **"Spend capability where it counts"** in `docs/orchestration-environment.md` — a strong model holds the plan and owns the outcome while cheaper models do scoped work beneath it, supervised rather than trusted. Includes the two cautions: do not under-resource the thinking to save money, and do not confuse fanning out with progress.
- **"Where context lives"** in `docs/05-context-engineering.md` — the mechanism behind reusable context: plain markdown committed beside the code, updated continuously, answering what the goal is, what is established, what remains, and what was learned. Grow made concrete.
- **Field Lesson 4 — you do not have to hold the context to be accountable for it.** Being out of date on an area is an access problem, not a competence problem. The domain was already in the repository and the running application; what changed was that it became reachable.
- **Field Lesson 5 — write the understanding down, or pay for it again.** Two efforts in the same period: one kept persistent context files and converged; the other did not, and degenerated into repeated attempts until the loop was applied deliberately.
- **A three-step Start section** in the README and on the website: copy `AGENTS.md` into your project, give the agent read-only access to the evidence, run one real problem through the loop.
- **"Three things that matter most"** on the website — people are still the point, set it up with your own agent, and spend capability where it counts.
- `docs/orchestration-environment.md` — **the access layer** that makes orchestration possible: what it connects to (repositories, work tracking, pipelines, logs, datastores, non-production environments, the running application), how context forms a chain with a threshold, how it accumulates across passes without becoming thrashing, a practical build sequence ordered by increasing risk, the rules for granting access, and what it deliberately does not require.
- A third reference implementation, **Multi-Repository Defect Remediation** — working a batch of defects across service boundaries rather than one at a time, ending in verified closure for those genuinely fixed and better context for those not.
- A **"Watching the watcher"** section in `docs/orchestration-environment.md` and a matching gap statement in `docs/08-governance.md`: there is currently no monitoring layer that detects an agent doing something unintended inside access it was legitimately granted. Stated as an open problem rather than worked around.
- Guidance in `docs/08-governance.md` on **attributing every action to a person** — running work under the credentials of the accountable human, for traceability and for the restraint that comes with it.
- A concrete adoption sequence in `docs/09-adoption.md`, replacing four abstract bullets: start on one real problem, teach the loop rather than the tool, grant read-only access to real systems first, wire into existing environments and pipelines, let people work conversationally, widen blast radius only as proof accumulates, and capture what each cycle taught.
- A **"Proven in practice"** section in the README, stating the two delivered outcomes up front: the CMS API migration estimated at 8–10 weeks and delivered in about a day, and the production memory leak traced past its workaround and fixed at the root.
- `assets/social-preview.png` — a 1280×640 card used for the site's `og:image` and the repository social preview, with `assets/social-preview.svg` as the design source.

### Changed
- **Governance now states that the framework proposes no new access model.** AI works within the access the accountable person already holds — inherit, do not expand; existing controls apply unchanged; and the organization's own experts make the call. The question is not "is it safe to give AI access?" but "is this person's existing access appropriate, and am I comfortable with work being done through it?"
- **The monitoring gap is now stated as a requirement**, not an observation: `AI governance monitoring is required`. Approval gates stop the largest mistakes and attribution explains them afterwards, but nothing yet detects an agent acting outside intent within access it legitimately holds. Named as the most valuable unsolved problem in the space, including here.
- **Adoption is stated as carrying more weight than anything else.** Access can be arranged and controls written in a week; whether people actually work this way is what decides whether any of it mattered. Governance closes by deferring to it.
- **The website leads with the framework's name** rather than a positioning line, and the primary action is now Start rather than Explore.
- **README navigation now reaches every document.** Six docs — context engineering, proof, governance, adoption, roadmap and the agent-orchestration page — were previously unreachable from the README and discoverable only through internal links. Grouped now into the framework, going deeper, putting it to work, and beyond.
- **Orchestration is now framed as a layer above existing systems**, in the README, the website and the adoption doc — not a platform to migrate onto. Adoption is incremental and reversible; the systems underneath are unchanged, and removing the layer leaves no trace. Promoted from a caveat in "what it is not" to a positioning statement, with the honest qualifier that the layer still needs access, and that access should be scoped deliberately.
- Regraded the reference implementations. Cross-Team Knowledge Access has resolved **real production and support incidents** and reaches **rung 4–5** for those cases; Production Exception Remediation and Multi-Repository Defect Remediation have both run on real work through existing review and deployment approvals. None is an always-on capability or adopted organization-wide, and all still depend on a human providing the map, reviewing the output, and holding the approvals.

### Removed
- **The evidence ladder and the autonomy ladder, everywhere.** Numbered rungs 1–5 and autonomy levels L0–L4 are gone from the docs, `AGENTS.md`, the Quickstart, the brief template, the PR template, the worked example, both glossaries and the website. Grading work on a scale invited the behaviour the framework exists to prevent: "rung 4" is a score, and a score is easier to claim than to justify — it also let a narrative anecdote and a harness with 36 enumerated cases sit at the same number. What replaces it is the same discipline in plain language, **state what you checked, what you observed, and where you stopped**, and for autonomy three rules instead of five levels: results decide rather than confidence, blast radius overrides track record, and it is granted per context rather than globally. The standard did not change; only the pretence that it could be expressed as a number.
- **The website's "Beyond software" section.** Three tabs restating the same six stages in slightly different vocabulary. The claim it existed to make — the loop is not software-specific — is now one line on the lifecycle.
- **Case study 03, "Contextual Reasoning in a Newborn Care Scenario."** It illustrated a model reasoning well about insufficient context, which is a property of the model rather than of orchestration. Nothing in it was set up, governed, or proven — it was good AI use, not orchestration work, and presenting it as evidence for this framework overstated what the framework had shown. The case-studies index and the website now carry outcome case studies and reference implementations only.
- `docs/06-agent-orchestration.md`. It was twenty lines, had no inbound links, and restated Principles 3 and 4 without adding to them. Its one distinctive line — keep execution observable, attributable, and reversible where practical — moved into governance.

### Changed — the website is shorter
- **Deleted the "Run one cycle on something real" section.** Once the agent-file section existed, it was saying the same thing twice — the three start steps now live at the end of that section, where someone has just read what the file does and is ready to use it. The quickstart, brief template and worked example are one line of links rather than three cards.
- **The README lost a third of its length.** Cut "Why this matters" (six bullets that could have described any framework), the duplicate reference-implementations list, and the "Current Status" checklist of green ticks. The thirty-link "Explore the Framework" index became five grouped lines. 239 lines to 189.
- **Merged "Why orchestration" and "The problem" into one section.** Both argued that capability was never the bottleneck; saying it once says it harder.
- **"What it is not" is a two-column list rather than six cards**, with each entry cut to a line.
- **The evidence section is one tab group** — Contentful API migration, React memory leak, reference implementations — instead of stacked cards under two subheadings.
- Net effect: sixteen screens down to twelve, fifteen sections down to eleven, six tab groups down to four.

### Changed — how the work is described
- **The migration's performance figures now say where they were measured.** They come from a committed script running both services on one developer machine — warm-up, then 400 requests at 40 concurrency across four endpoints, plus a 120-request invalid-slug burst. No load test has been run against deployed infrastructure; the multi-regime test that was written was never executed, and the comparison report it was meant to populate was never generated. The numbers are real and reproducible; they are not production evidence, and the case study now says so above the table.
- **Performance is credited to the architecture, not to AI.** One REST fetch instead of 2–3 GraphQL round trips, a slug allow-list rejecting junk before it reaches the CMS, and Brotli on a previously uncompressed payload. A team writing the same design by hand would have measured the same gains. What AI changed was the cost of attempting the rewrite — a claim about effort, not latency. The results table is split accordingly.
- **Stated that the parity suite was a one-time migration gate**, removed from CI once it had served its purpose because it needs both services running side by side. 36/36 is point-in-time, not continuously enforced.
- **"About a day" now says what it measures.** It is the *execution* time for the CMS API migration, run with agents and subagents. Testing and parity validation took roughly another day, and stakeholder agreement longer still; the work was not continuous. Setting one day against an 8–10 week estimate that included analysis, review and coordination overstates the gain, and the case study, README, index, website and profile now all say so. Added as a takeaway in its own right: execution stopped being the expensive part, which moves the bottleneck rather than removing it.
- **`docs/reference-implementations.md` said "Both" while describing three patterns** — left over from when there were two.
- **Case studies are referred to by name, not by number.** "Contentful API migration," not "case study 01." The filenames keep their numeric prefix so the directory stays ordered, but nothing user-facing counts them.
- **Dropped "honestly reported."** Saying so implies other people do not, or that earlier reports here were not. The work either states what it has not proven or it does not; an adverb adds nothing.
- **The React case study now records what the investigation actually cost:** several days of repeated profiling, four candidate fixes implemented and disproved, two separate false conclusions (a single run draining on its own, and a dead local endpoint that manufactured a leak), and the fact that it looked like a Next.js defect until `global.Error` was instrumented to find the pinning object in React.
- **The React case study now separates the two paths to the same pin.** Renders that *fail* retain a reason `Error` via an `AbortSignal` reason, a Flight stream's closed-reason, or a rejected promise — this is the dominant production path, measured at 8,833 aborted-with-reason signals and ~1.96M retained promises on one leaking pod. Renders that *succeed* hit the same pin because React constructs an `Error` itself on the completion path — the defect fixed upstream. Conflating the two cost time during the investigation and had been flattened into a single mechanism in the write-up.
- **Added the fleet A/B that validated the mitigation:** the same container image under sustained load, differing only by `--stack-trace-limit=0` — 113 restarts across 53 pods without it, zero across 36 pods with it over ~6.5 hours. Stated as validation of the blunt mitigation, not of the upstream patch, and noting that the surviving ~1.8 GB is bounded rather than eliminated.

### Moved
- `docs/ai-future-hypothesis.md` → `hypothesis/ai-future.md`, out of `docs/` entirely. It is speculative and explicitly not part of the framework; keeping it beside the doctrine invited confusion.
- Broadened `docs/01-problem.md` so the problem statement is not framed as software-engineering-only. Software engineering remains where the framework was built and where its evidence comes from; the problem it addresses is not confined there.
- Replaced a competing slogan on the website ("An agent can act. Only orchestration learns.") with the canonical **"Agentic workflows repeat. Orchestration learns."**
- Added a framework-versus-runtime clarification to the README and the website: the framework defines the operating model around AI orchestration and does not prescribe the runtime used to execute it. Agent frameworks, workflow engines and tool protocols sit inside it.

## v0.4.0 — 2026-08-24

A review pass over the whole framework, plus the first website. The model itself did not change; what
changed is how much of it is usable, and how honestly it states what it knows.

### Added — making the model usable
- `QUICKSTART.md` — apply the lifecycle to your first task in about 15 minutes.
- `templates/orchestration-brief.md` — a reusable one-page brief covering all six stages, with copy-and-paste prompts and explicit human/AI ownership.
- `examples/production-exception-remediation/` — an end-to-end walkthrough applying the lifecycle (and the orchestration brief) to a recurring production 5xx exception, labeled as an illustrative scenario.
- `site/` — an interactive website deployed to GitHub Pages: the lifecycle explorer (including the Proof → Understand failure edge), the evolution from models to orchestration, the evidence ladder, the AI failure-mode matcher, the autonomy ladder, and a searchable glossary. No build step and no dependencies; readable with JavaScript disabled.

### Added — new instruments
- **Evidence ladder** in the Proof stage — five rungs from "asserted" to "observed in the real environment," with the rule to name the rung you actually reached. Wired into the Quickstart, the brief template, and the worked example.
- **Autonomy ladder** in the AI Orchestration Model — five levels of goal-directed autonomy, each earned through consistently achieved Proof, capped by blast radius and revocable. Applied in `docs/08-governance.md`.
- "When Proof fails" guidance — a failed proof returns to Understand, not to Execute.
- `docs/how-ai-fails.md` — eight AI-specific failure modes and the lifecycle stage that catches each one.
- `docs/glossary.md` — plain-language definitions for every term the framework uses.

### Added — scope and evidence
- Case study 03 — "Contextual Reasoning in a Newborn Care Scenario," illustrating how orchestration can identify a missing signal, support evidence gathering, and keep the consequential decision with the appropriate human expert.
- `case-studies/README.md` — an index that distinguishes outcome case studies from reasoning patterns.
- "What it is not" scope boundaries in the README.
- GitHub issue and pull request templates that follow Opportunity → Change → Proof.

### Changed
- Reduced the principles from eleven to **six — one per lifecycle stage**, folding context, gap-identification, execution ownership, and collective capability into the stage they belong to.
- Collapsed five competing loop formulations onto the single canonical lifecycle across the philosophy, case study 03, and the AI Future hypothesis.
- Expanded the framework philosophy beyond engineering workflows, adding the guiding statement: **AI expands what humans can understand, reduces what humans can miss, and helps prove what humans accomplish.**
- Converted the philosophy's numbered list into prose beliefs so the principles are the only numbered list.
- Reframed the AI Future hypothesis to acknowledge prior art on competitive AI-safety dynamics and to connect back to the framework; demoted it in the README.
- Sharpened the AI Future hypothesis: added the unsolved technical barriers to autonomy (continual learning, long-horizon credit assignment, compounding reliability, physical sample efficiency, open-ended goal evaluation), the asymmetry of restraint as a unilateral cost, the follower's shortcut (compete on leash length rather than model quality), the physical axis as a non-substitutable resource, and a "what would make this wrong" falsifiability section. The argument names no actor by design.
- Rewrote the README navigation to surface the Quickstart, template, and worked example, and corrected the release status.
- Renamed `docs/07-validation.md` to `docs/07-proof.md` to match the stage name.

### Removed
- `docs/AI-FUTURE.md` — an orphaned pointer file describing a superseded version of the hypothesis.
- Eight unused AI Future illustrations, the unused `assets/ai-orchestration-model.png`, and the empty `diagrams/` placeholder.
- Career- and resume-positioning references from the philosophy.

### Fixed
- Redrew the three AI Future illustrations, which still carried headlines from the pre-rewrite "collective intelligence" version of that hypothesis and contradicted the text beside them.

### Known gaps
- `docs/05`, `06`, `08`, `09` and `10` remain short next to `docs/04`; governance and adoption are the ones most worth depth.
- The framework's own evidence base is three case studies from one practitioner. Outside case studies are welcome.

## v0.3.0 — 2026-08-14

### Added
- `docs/reference-implementations.md` — Cross-Team Knowledge Access and Production Exception Remediation, shown as applications of the core lifecycle.
- Diagrams for the lifecycle and both reference implementations.
- "The real bottleneck" opening in `docs/01-problem.md` — turning distributed capability (human and AI) into outcomes.
- `CONTRIBUTING.md` — how to raise an issue or pull request, what we look for, and an open, feedback-driven stance.

### Changed
- Reframed the core model as **Opportunity → Understand → Plan → Execute → Proof → Grow**, and widened the framing from orchestrating AI agents to orchestrating capability (human and AI).
- Rewrote the principles and aligned the philosophy, docs 01–10, field-practices, and case study 02 to the new model.
- Tagline changed to "Transform Opportunity into Outcomes."

## v0.2.0 — 2026-08-14

### Added
- Case study 02 — "Fixing a React Server Components Memory Leak Upstream," a real-world contribution to upstream React, with a visual of the progression: production OOM → workaround → continued investigation → root cause → validated fix → upstream contribution.
- `docs/field-practices.md` — field lessons (focus over parallelism; understand before fixing; a workaround is not the destination).

### Changed
- Reworked the case-study voice across case studies 01 and 02 to emphasize that AI lowers the barrier to meaningful work (removed "built entirely by AI" / single-engineer framing).
- Renamed the case-studies milestone to "Real-World Engineering Case Studies" and refreshed the README status/navigation.

### Fixed
- Resolved duplicate `05` documentation numbering (Practices moved out of the numbered sequence to `docs/field-practices.md`).

## v0.1.0

- Repository created and initial project structure established.
- Initial documentation scaffold and folder structure.
- Added `docs/`, `case-studies/`, `diagrams/`, `templates/`, and `examples/` README files.
