# Website

The public site for Clover:
**https://cloverframework.com/**

> The website explains. The repository documents. The repository is the source of truth — if the two
> disagree, the repository is right and the site is a bug.

## Running it locally

There is no build step and no dependencies. Open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000 --directory site
```

## Structure

```
site/
  index.html        the framework: one cycle, at any scale
  start/            getting started: read-only access, the agent file, a first cycle
  governance/       governance: access, attribution, approvals, delegated execution
  security/         a redirect stub, kept so the old URL still resolves
  evidence/         case studies and reference implementations
  glossary/         searchable terms
  author/           who wrote this, in the first person
  assets/           the peacock feather, and the four evidence diagrams (SVG)
  styles.css
  app.js            tabs, the mobile menu, the scrolling story, the glossary filter
  check.ps1         run before committing: nav consistency, versions, every link
  llms.txt          the repository map for agents
  .nojekyll
```

The page is one continuous argument rather than one section per stage.

1. `#overview` — Hero: the name, the tagline, what Clover is in a sentence, the mark, the stages
2. `#capability` — What is Clover: the cycle as it ran before AI, with the human as the actor
3. `#why-clover` — Why we need Clover: the same model produces different outcomes, and the difference is a structure that can be learned
4. `#ai-changes` — What AI changes: how each stage is done, and which of those changes are improvements
5. `#humans` — Why humans matter: accountability, boundaries, and carrying the consequence
6. `#systems` — What systems should adopt: process built for an actor that is not human
7. `#stage-context` — What is Context?
8. `#stage-direction` — What is Direction?
9. `#stage-execution` — What is Execution?
10. `#stage-outcome` — What is Outcome?
11. `#stage-growth` — What is Growth?
12. `#why-cycle` — Why is it a cycle? Growth becomes the next Context
13. `#try` — A common example: something changed since last time, and each stage explained against it
14. `#scale` — Complexity can grow: the same cycle for individuals, teams, organizations and AI
15. `#evidence-preview` — The cycle exists in every system: what Clover is still working out, and where contribution matters
16. `#real-world-context` — Context was misused: what was taken to train the models, and what it cost
17. `#real-world-direction` — Direction had no accountability: who set the Direction, and why nobody had to answer for it
18. `#real-world-execution` — Execution was not phased: no bounded phases, no external body, released to everyone at once
19. `#real-world-outcome` — Outcomes did not show incapabilities: sold on capability, adopted without knowing the limits
20. `#real-world-growth` — Growth was not fully addressed: why unaddressed mistakes still delay what AI could give
21. `#responsibility` — Great AI capability should mean greater responsibility: why responsibility lost to the pursuit of dominance
22. `#clover-teaches` — Who should take responsibility? The source, and everyone who adopts and uses it
23. `#what-now` — What should we learn from the mistakes? Stop racing, the law compensates rather than prevents, and own the infrastructure the model runs on
24. `#real-growth` — How can we see real growth? Build on what the organization already holds, and the whole mark with the closing line beneath it

The five real-world sections lead into the four closing sections, 21 to 24. They state the author's
conclusions and link to the sourced documents rather than reproducing their detail. No company,
government or person is named anywhere on the site; the documents in `docs/ai-responsibility/` name
every source.

None of the four maps to a stage. `#responsibility` and `#clover-teaches` carry
`.responsibility-band` and are excluded from centre-scroll alignment.

Sections 23 and 24 sit **outside** `[data-story]`, so the sticky mark reserves no space for them.
Section 24 carries its own static mark and the closing line, and its prose stays left-aligned — only
the mark and the verse centre themselves, so the section does not need `.closing`.

**The mark has two states: green, then grey.** Once `#evidence-preview` reaches the middle of the
viewport, `app.js` puts `is-real-world` on `<html>`, CSS drains the gradient to grey, the mark's
accessible name changes to describe the grey, and each leaf repoints from `#stage-*` to its
`#real-world-*` section with an `aria-label` taken from that section's heading. Only the three
gradient stops change: the same five leaves, labels, veins, shape and continuous scroll-driven turn
remain. Scrolling back restores green above the real-world sections.

**The grey does not end the page.** `#what-now` carries its own static mark with all five leaves and
veins intact, filled from a second gradient, `#leaf-revive`, so the drained `--leaf-*` values on
`:root` cannot reach it. When `#what-now` comes within 90% of the viewport `app.js` sets `is-closing`
on `<html>` and the pinned mark fades out; `visibility` is delayed to the end of the fade so its five
leaf links leave the tab order rather than staying focusable while invisible.

Each of the five stages is its own `.band.stage` section carrying `data-leaf`. `app.js` reads those
inside `[data-story]` and inks the matching leaf on the pinned mark as the section reaches the middle
of the viewport. The `stage` class exists for that hook alone and carries no styling.

Each leaf of the mark is an SVG `<a>` to its own stage, so the mark is also the way into them. The
generic in-page click handler scrolls them; nothing leaf-specific is needed. The leaves are links,
not `tabindex` groups, so the keyboard reaches them for free.

There is no author section on this page. The nav links to `author/` instead.

The final Responsibility section links to
`docs/ai-responsibility/why-responsibility-was-sidelined.md`, which holds the detailed argument,
sources, limits and links to the supporting papers.

## How it is built

- **No framework, no build step, no dependencies.** A framework repo whose own site needed a
  toolchain to render five leaves would undercut its own argument.
- **Progressive enhancement.** Every panel is in the DOM and readable with JavaScript disabled;
  `app.js` only reveals and highlights.
- **Inline SVG** for the clover marks — themeable, tiny, and no image assets.
- Deployed by `.github/workflows/pages.yml` on any push to `main` that touches `site/`.

## The clover marks

The marks are the site's identity and they carry the argument, so they have rules of their own.

- One `<path id="clover-leaf">` lives in a hidden sprite `<svg>` at the top of each page. Every mark
  is that single leaf `<use>`d and rotated around the point `(50, 44)` over a short stem. There is no
  second leaf path anywhere — add one and the identity drifts.
- **One mark carries the story**, drawn with five leaves at 72 degrees and labeled Context,
  Direction, Execution, Outcome and Growth. All five are solid: Growth is a stage of the cycle, not a
  possibility hanging off the end of it, and drawing it dotted said the opposite. It is the
  `.clover--hero` mark in the pinned rail and the only labeled mark on the site; the small header
  mark is decorative and stays solid.
- Leaf names are real `<text>` labels, each set on a `<textPath>` along an arc outside its own leaf
  tip, inside a `viewBox` of `-6 -10 112 110` so the labels have room. They are content, not
  decoration — never replace them with a legend.
- Fill state is CSS, not geometry. `.clover__leaf.is-soft` is an established stage,
  `.clover__leaf.is-new` is the stage that just arrived, and `.clover__leaf.is-next` is Growth drawn
  with a dashed stroke and no fill. `.clover__leaf.is-filled` stays for the solid leaves in the
  header and hero.
- Marks that carry meaning get `role="group"` and an `aria-label` naming the leaves. Decorative
  duplicates, such as the one in the header, get `aria-hidden="true"`.
- An arc mark has to hold up between about 200px and 420px; the pinned mark between about 110px and
  400px, and the header mark is 26px. Nothing else is allowed to carry the identity: no brains,
  robots, circuit boards, neural networks, hexagons or glowing AI graphics.

## Editing rules

- **One cycle, shown once.** The page carries a single labelled clover in `div.pinned__mark`, the
  first child of `div.pinned` and a sticky rail that spans every section. It replaced a three-act arc
  that grew the mark from three leaves to four to five. Do not reintroduce separate marks per act.
- **The site does not use the `System → Human → AI` chain.** Say it in words: the system is the
  reality, and the human and AI are **the actors**. The docs still carry the chain; the site does not.
  `Context → Direction → Execution → Outcome → Growth` is **the system cycle**. Say those names when
  referring to either set. Never introduce a competing arrow-chain, never append "→ repeat" to the
  cycle, and never bring back the short form "Where → Know → Do → Validate". Direction is *what*, not
  *where*.
- **Clover is not a new way of working, and the site has to say so.** Every system that worked in the
  past has worked this way: somebody understood the situation, somebody decided what mattered and
  answered for it, the work got done, reality showed what happened, and what it taught carried into
  the next attempt. What AI did was remove accountability out of scope. Execution moved to something
  that cannot be accountable, and a model can perform the work, report that it worked, and hold
  nothing when it did not. Clover establishes accountability back into the system, through the human
  actor who can truly take up the role, and AI takes its place as an actor inside the existing cycle
  rather than as a replacement for it. `#capability` on the home page and `#policy` on the governance
  page both carry this claim, and `llms.txt` carries it for agents. Do not soften it into "AI needs
  oversight".
- **For a task or piece of work, accountability lands on a named human.** Governance is where the
  site shows that the naming is real: whose access was used, who approved, and who answers when it
  goes wrong. This does not transfer model-level accountability away from the organization that
  builds and releases the model. Capability is never authority, and authority is never
  accountability. The home page carries the work-level rule in `#humans`; the named claim
  **Capability may scale. Direction remains human.** now lives on the governance page and in
  `llms.txt`, not on the home page.
- **The system cycle has five stages** — Context, Direction, Execution, Outcome, Growth. All five get
  named wherever the cycle is enumerated: the mark, the five stage sections, the agent file
  description on `start/`, and `llms.txt`. `#stage-growth` and `#why-cycle` are where the home page
  explains the fifth stage and why the five come round again.
- **Growth is whatever the Outcome taught, carried back into Context**, at any size. No repetition and
  no scale is required. One wrong answer, understood and written down, is Growth. It is not reserved
  for frontier AI providers with volumes of interaction data. Writing the context file after each
  success and each failure is Growth, and what that file keeps becomes the next cycle's Context — so a
  walkthrough may end on that write, and must not present it as something outside the cycle.
- **"Leaves" is for the picture.** Say leaves when describing a clover mark. Everywhere else on the
  page, say stages.
- **Never call it a model.** Clover is a way of working with System, Human, and AI to produce
  meaningful outcomes. "Model" collides with "AI model".
- **Say "human"**, never "person" or "the user", for whoever holds Direction. Direction is the human
  controlling what matters, the desired outcome, constraints, boundaries, and what must not happen,
  and approving. Execution is AI determining how the work should happen and executing within those
  boundaries. Never give the human the detailed "how" — it empties Execution.
- **No rungs, levels, scores or grades.** The evidence ladder and the autonomy ladder were deleted
  from the framework. State what was checked, what was observed, and where the work stopped.
- **Never invent** metrics, customers, adoption, or results. The React memory leak is a CI-green pull
  request that is *not merged*, and the Contentful production cutover *has not run* — the site must
  keep saying so.
- **`#capability` is short on purpose.** A lede and one paragraph, in the same voice as the rest of
  the page. It says the cycle ran before AI, with the human as the actor, and stops there. Do not
  turn it into a benchmark table, do not add scores, and do not start arguing about what AI is
  missing — that argument belongs to `#humans` below it, and making it twice weakens both.
- The home page says nothing about production. `#setup` on `start/` is where the environments are
  named: development first, then the other non-production environments. Access is read-only and
  scoped to what the human already holds.
- Growth is the next stage, never a hypothesis, a prediction or "next phase", and never described as
  dangerous. Do not explain the *Black Clover* or devil association anywhere.
- Never assert that any AI provider trains on customer or enterprise work. Keep the accumulation
  argument structural and unattributed.
- One background for the whole site: white. Sections carry no tint and no dividing rule; the
  `.band` padding is what separates them. There is no `.band--soft` and no `.card--flat`; do not
  reintroduce either. The footer keeps its top rule, since nothing else marks where it starts.- **No green labels above headings.** The `.eyebrow` component is gone from every page and its rule
  is out of the stylesheet. A section is introduced by its heading alone.
- **Section prose runs at the hero's scale on every page** — `clamp(1.12rem, 2.2vw, 1.28rem)`, which
  puts a heading at about 1.6 times its text rather than twice. It applies only to prose directly
  inside a section, plus the two unstyled columns in `#scale`; cards, tab panels, `.panel__cols`,
  definition lists and the verse keep their own density. The floor is 1.12rem rather than 1.05rem
  because 1.05rem is 16.8px, below the 17px body size, and would have shrunk narrow screens. Check a
  new selector against `.lede`, which is 1.12rem and will win on specificity if you are careless.
- **Accent is reserved for the opening line of a section.** `main section.band > .wrap > p:first-of-type
  strong` is the accent green. Bold in any later paragraph stays ink, and headings stay ink, so a
  section reads heading, then one coloured claim, then plain emphasis. That selector is deliberately
  specific: the band prose rule above outranks a bare class.
- **`.evidence-link` is the quiet row under an argument** — 0.95rem, muted label, link in `--ink-2`
  with a pale underline, accent only on hover. Six carry an `Evidence:` label; the one pointing at
  `AGENTS.md` does not, because that is the rules file rather than a source.
- The mark turns one full revolution between the top and the bottom of the home page, eased toward
  the scroll position in `app.js` rather than tracking it exactly. `prefers-reduced-motion` stops it.
  The turn is an SVG `rotate(deg 50 44)` on `.clover__spin`, inside the viewBox. Rotating the `<svg>`
  element instead grows its box to the diagonal, which pushed the page sideways at narrow widths.
- On mobile the mark is a sticky white band under the header. Its `top` is deliberately a little less
  than the header height, so the two overlap; at an exact match the page scrolled through the sliver
  between them.
- **Use the width.** Sections must not stack down the left in a 68ch column. `.wrong__flow` and
  `.wrong__cols` run a `.wrong` block across both columns, `.split` puts a heading and lead on the
  left with the detail on the right, `.stack--2` gives a two-column definition list, and
  `.panel__cols` spreads a tab panel across the full measure.
- Tabs: one generic `initTabs` over `[data-tabs]` groups. The markup decides which tab opens — whichever
  carries `aria-selected="true"` and `is-active` — so nothing flashes before `app.js` runs, and
  `app.js` does not override the choice.
- No page links to any home-page anchor. Keep all six navs identical.
- `author/index.html` is the one page written in the first person about a life rather than about the
  framework. It is also the one place *Black Clover* may be named, as a favorite anime. It ends on
  the verse, with no closing section and no site footer. Leave all of that alone.
- **The clover is not used on the author page.** Its mark is a peacock feather,
  `site/assets/peacock-feather.png`, shown twice — beside the name, and above the verse. It is the
  only raster asset the site loads, cropped with the white knocked out to transparency. The clover
  stands for the framework; the feather stands for the author. Do not swap one for the other.
- **Six pages, no build step, so the header, nav and footer are copied into each one.** They will
  drift. Run `pwsh -File site/check.ps1` before committing: it resolves every nav link to a
  site-root path and fails if the six pages disagree, checks the `?v=` versions match, and follows
  every relative link and every anchor.
- **The primary nav is seven plain links, and it does not point at any section of the home page.** It
  opens on **Clover**, which is the way back to the home page from anywhere; on the home page itself
  that link carries `aria-current="page"`. A nav link to the page you are already on eases to the top
  instead of fetching the page again, and clears any fragment from the address bar. `app.js` decides
  that by comparing resolved paths with a trailing `index.html` stripped, since the nav says `./`
  where the address bar says `index.html`. It was
  a "Framework" dropdown holding the three clover sections and the worked example, and later carried a
  first link to `/#story`. Both are gone, because the pinned mark labels its own leaves and links
  each one to its stage section, so the reader can pick one there. Do not reintroduce a dropdown to
  hold anchors that the page already exposes. Nav labels are the only Title Case text on the site;
  everything else, including the headings those links point at, stays sentence case.
- **Anchor scrolling is eased in `app.js`, and the duration scales with the distance.** The browser's
  own `scroll-behavior: smooth` runs at a fixed speed, so a jump from the hero down to the story
  arrived almost as abruptly as no animation at all. The CSS rule stays for the no-JS case, and
  `app.js` sets `scroll-behavior: auto` when it takes over so the two are not fighting over the same
  scroll. Anything measuring scroll positions in a test has to account for the animation.
- **Same-origin navigations cross-fade** through `@view-transition`. Browsers without it navigate the
  way they always did. Both the eased scroll and the cross-fade are inside
  `prefers-reduced-motion: no-preference` and turn into instant jumps when reduce is set.
- **`.btn` is the only button style.** Plain is the default, `.btn--primary` is the filled one, and
  `.btn--plain` is a bare link that keeps the same height. One primary per group.
- **The evidence diagrams are hand-written SVG, and the same four files serve the docs.** They live in
  `assets/` at the repository root and are copied into `site/assets/`; if you change one, copy it
  again or the site and the case study will disagree. They are drawn in the site's palette, they carry
  no icon art, and every one of them ends on a strip naming what the work does **not** show — the
  cutover that has not run, the pull request that is not merged, the pattern that is not always-on.
  Keep that strip. It is the reason the diagrams are allowed on the page at all.
- **The PNG infographics they replaced were wrong** and must not come back. Two of them advertised the
  deleted six stages, one showed the production cutover as a completed step, and one said the memory
  leak fix went to Next.js when it went to React. Any new diagram gets checked against the case study
  text before it ships.
- **`[data-story]` is `div.pinned`**, the wrapper holding every section and the mark itself. It was
  three separate acts, each with its own mark, because a leaf had to link to a step in the act being
  read. With one mark spanning the page that constraint is gone.
- **Each `section.band.stage` carries `data-leaf`.** `app.js` picks, per block, the stage whose
  centre is nearest the middle of the viewport, and highlights that leaf and its label. It also still
  reads `data-caption` and `data-ink`, but nothing on the page carries either. Nearest-to-centre is
  used rather than an IntersectionObserver because it gives the same answer scrolling up as scrolling
  down; an observer fires on entry and leaves the highlight stale on the way back up. Scrolling is
  never taken over.
- **Every leaf carries a `<text class="clover__label story__label">`** naming the stage, and the
  label highlights with its leaf. The labels are content, not decoration — a reader has to be able to
  tell which stage is being described without counting leaves.
- **Leaf links are SVG `<a class="clover__leaf-hit">`**, one around each leaf, its vein and its
  label, carrying an `href` to that stage section, an `aria-label` and a `<title>`, so they are
  clickable and focusable and still jump to the right section without JavaScript.
- **`#stage-context`, `#stage-direction`, `#stage-execution`, `#stage-outcome` and `#stage-growth`
  are the anchors the clover leaves link to.** Nothing in the nav points at them, but keep them —
  `check.ps1` follows every one. Each leaf, its label and its section share a `data-leaf` value, so
  the five values must stay paired.
- American spelling. No employer, product, cluster or infrastructure names.
- **Bump the `?v=` on `styles.css` and `app.js` whenever either changes.** All six pages carry it.
  GitHub Pages sends `Cache-Control: max-age=600` on every file and they expire independently, so
  without it a returning visitor gets new HTML with a ten-minute-old stylesheet and the page renders
  broken. Match it to the version in `VERSION`.
- New terminology goes in [`docs/glossary.md`](../docs/glossary.md) first, then the site. The site
  glossary must define **System**, **actors**, **system cycle**, **Growth** and **accountability** the way
  the docs define them, and its `<div><dt>…</dt><dd>…</dd></div>` rows are what the filter in `app.js`
  reads, so keep that markup.

## Social preview

`assets/social-preview.svg` is the design source; `assets/social-preview.png` is the exported
**1280×640** card used for both the site's `og:image` and the repository social preview
(Settings → General → Social preview, which has no API).

Light background, teal accents, matching the site. Keep important content inside a ~40px margin —
GitHub crops the edges at some sizes.

The mark is five leaves at 72 degrees, the same as everywhere else. It was four at 90 degrees once,
which no amount of correct wording beside it would have fixed.

To re-export after editing the SVG, rasterize with headless Chrome. Set the scale factor explicitly:
without it a display running above 100% produces a scaled, soft card that still reports 1280×640.

```powershell
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' --headless=new --disable-gpu `
  --hide-scrollbars --force-device-scale-factor=1 --window-size=1280,640 `
  --screenshot=assets\social-preview.png file:///$PWD/assets/social-preview.svg
```

The `og:image` tags on every page point at the PNG and pick up a new file with no markup change.
