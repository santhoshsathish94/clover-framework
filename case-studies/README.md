# Case Studies

> **Context → Direction → Execution → Outcome → Growth**

Real work, including the parts that are still unproven.

Both are delivered outcomes rather than illustrations: a migration whose implementation took about a
day against an 8–10 week estimate, and a production out-of-memory failure traced to its root cause in
a shared framework, with a focused fix contributed upstream. Neither is finished. The migration's
production cutover has not run, and the upstream fix has not been merged. Both say so.

The two span different domains on purpose. The stages are the same in each, and only the
vocabulary around them changes.

| Case study | Domain | What it shows |
|---|---|---|
| [Contentful API migration](01-contentful-migration.md) | Software engineering | A large GraphQL → REST migration whose implementation took about a day against an 8–10 week estimate, with the public contract preserved and parity checked endpoint by endpoint against the current API. *(Execution time only — validation and sign-off took longer. Production cutover not yet run.)* |
| [A React Server Components memory leak](02-react-rsc-memory-leak.md) | Open-source engineering | Production OOM → workaround → root cause → validated fix → upstream contribution. The workaround was not the destination. *(Pull request CI-green, not merged.)* |

## A note on evidence

The three [reference implementations](../docs/guides/reference-implementations.md) are described separately.
Cross-team knowledge access has resolved real production and support incidents, and the two
remediation patterns have run on real work through existing review and deployment approvals. None is
always-on or adopted organization-wide.

Each case study says what was checked, what was observed, and where the work stopped. See
[Outcome](../docs/framework/07-outcome.md#how-strong-is-your-evidence) for how evidence is described here.

Each one then ends at Growth, in a section named *Growth — what went back into Context*, which
records what the cycle taught. That section keeps within the evidence above it and adds no result of
its own.

## Contributing a case study

Honest case studies are welcome, including the ones where the outcome was never reached. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
