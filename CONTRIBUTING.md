# Contributing

This is an open, feedback-driven project. It grows through real engineering experience — yours
included. Corrections, questions, ideas, and case studies are all welcome.

## The short version

1. **Start with the problem, not the solution.** Open an issue describing the direction: what you
   observed, why it matters, and the outcome you're after.
2. **Discuss before large changes.** For anything beyond a small fix, let's align in the issue first
   so the work stays focused.
3. **Raise a focused pull request.** Fork, branch, make one coherent change, and open a PR that
   explains the direction, the change, and the evidence behind it.

## How to raise a pull request

1. Fork the repository and create a branch (for example, `docs/clarify-the-outcome-stage`).
2. Make a small, focused change. Match the existing structure and writing style.
3. In the PR description, briefly cover:
   - **Direction** — the problem or gap this addresses, and the outcome you're after.
   - **Execution** — what you changed, and why this approach rather than another.
   - **Outcome** — the evidence. State what you checked, what you observed, and where you stopped.
     For a docs change, "I checked every link and heading anchor" is real evidence.
   - **Growth** — what the change taught that should outlive it, and where that is written down. For
     a docs change, that might be a wording rule the next contributor will run into too.
4. Open the PR against `main`. Expect a friendly review and a little back-and-forth.

## What we're likely to accept

- Clarity and correctness improvements to the docs and the framework.
- Field lessons and practices drawn from real experience.
- Reference implementations that apply the stages to a common problem.
- Case studies with honest outcomes, including the ones where the outcome was never reached.
- Anything that keeps the framework **simple, universal, practical, and explicit about human
  Direction and accountability**.

## What we'll usually push back on

- Added stages, jargon, or complexity the reader doesn't need. Clover is deliberately five stages:
  **Context → Direction → Execution → Outcome → Growth.**
- Rungs, levels, scores, or grades for evidence or delegated execution. They were removed on purpose,
  because a number is easier to claim than to justify. Say what you checked, what you observed, and
  where you stopped.
- Outcome or production claims without evidence.
- Overselling AI. AI is a capability within the system, not the hero.
- Treating model capability as authority. **Capability may scale. Direction remains human.**
- Treating competitive pressure as a reason to transfer Direction to AI. **Direction is not delegated
  by competition.**
- Making it about a single person or tool rather than the shared idea.

## Writing style

Use the [repository map](docs/repository-organization.md) to choose the document's
home and role. Keep documents in their purpose-based folders, without forwarding copies. Keep historical
measurements distinct from current guidance and label proposals as proposals.

Before a documentation or site PR, run these from the repository root (Python 3.12
is used in CI):

```sh
python -m unittest discover -s scripts/tests
python scripts/check-links.py
python scripts/check-site.py
```

These checks do not rerun model benchmarks or establish external claims. State any
additional checks the change requires and any evidence that remains unavailable.

Three rules: **simple, direct, human.**

**Simple.** Short words. One idea per sentence. If a sentence needs a comma to hold itself together,
make it two sentences.

**Direct.** Say the thing, then explain it. No build-up. Name who does what — "the team decides" is
better than "a decision is made".

**Human.** Write like you are explaining it to a colleague at their desk. Contractions are fine.
Leave out the clever closing line.

Start with the problem, then what you found, then the principle. Say *we* and *teams*, not *you*. If
a reader thinks *"this is simple, we already have this problem, I can apply this,"* it is working.

Signs it is not working:

- A sentence shaped like "not X, it is Y".
- A paragraph that ends on a quotable phrase.
- Any sentence you would call elegant.
- Three items in a list because three sounds good.

## Feedback and discussion

Not sure it's worth a PR? Open an issue anyway. Questions, disagreement, and lessons from your own
work are genuinely useful — that feedback is how the framework improves.
