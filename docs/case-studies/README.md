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

The three [reference implementations](../guides/reference-implementations.md) are described separately:
cross-team knowledge access, production exception remediation, and multi-repository defect remediation.
Knowledge access has answered real production and support incidents, including a data-synchronization
failure that had survived an overnight investigation after the owning team had gone. The remediation
patterns have run on real work through existing review and deployment approvals. None is an always-on
capability or adopted organization-wide, and each still depends on a human providing the map and
holding the approvals.

Each case study says what was checked, what was observed, and where the work stopped. See
[Outcome](../framework/07-outcome.md#how-strong-is-your-evidence) for how evidence is described here.

Each one then ends at Growth, in a section named *Growth — what went back into Context*, which
records what the cycle taught. That section keeps within the evidence above it and adds no result of
its own.

## Five lessons from running this on real work

These came out of the work above, including the parts that went badly (see
[Practices and field lessons](../guides/field-practices.md) and
[How AI fails](../guides/how-ai-fails.md) for the full guidance):

1. **Focus beats parallelism.** Agents made it easy to run three initiatives at once, and context and
   human direction spread too thin across all of them. Dropping the parallelism and working one
   stream at a time is what finally resolved the hardest issue at its root. Human attention is the
   scarce resource, rather than AI throughput.
2. **Read the system before fixing it.** AI jumped straight to attempted fixes and repeatedly failed,
   because each attempt was a guess. Gathering what the system was really doing end to end, and
   adding logging where the picture had holes, found the root cause far faster than the fix-first
   thrashing that came before it. Thrashing is the signal to stop fixing and go back for the
   information.
3. **A workaround is not the destination.** Disabling stack-trace capture stopped the production
   memory growth, and that was the right immediate mitigation. Continuing past it found the retention
   mechanism underneath and produced an upstream fix instead of a workaround carried forever. Ask
   what a workaround is hiding or disabling, rather than only whether the incident stopped.
4. **You do not have to hold the context to be accountable for it.** A defect arrived in a flow the
   responsible human had never worked on. The orchestration layer read across the repositories,
   reproduced the behavior in the running application, and showed where it happened. The knowledge
   was in the system the whole time. Treat "I am not current on this area" as a question about
   access. Accountability stayed with the named human throughout.
5. **Write the context down, or pay for it again.** Two investigations, the same human, the same
   tools, the same period. The one that kept findings written down beside the work converged. The one
   where nothing persisted between passes degenerated into repeated attempts at a fix. A file
   maintained during the work is memory; a write-up produced afterwards is a report. That write is
   Growth, the fifth stage of the system cycle, and what it keeps becomes the next cycle's Context.

## Contributing a case study

Honest case studies are welcome, including the ones where the outcome was never reached. See
[CONTRIBUTING.md](../../CONTRIBUTING.md).
