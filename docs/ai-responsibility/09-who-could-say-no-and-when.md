# Who Could Say No, and When

Status: evidence checked 9 September 2026.

## The question

Consent is usually argued as a question of principle: should permission have been obtained before
human work was used to train a general-purpose model?

This paper asks a narrower question that can be answered from documents and dates:

**Who was able to refuse, by what mechanism, and at what point in time?**

The interesting finding is not that protection was absent. Protection existed, arrived early, and
worked. It simply did not arrive for everyone at the same time.

---

## The short answer

**Observed.** By August 2021, one provider's terms already prohibited scraping its systems, cloning
its model weights, building a competing model from its outputs, and bringing a class action against
it. The first documented mechanism allowing a website to signal that its content should not be used
to train that provider's models appears roughly two years later, in August 2023.

**Conclusion.** The party that had something to protect wrote its own protection, made it a condition
of access, and had it operating before any dispute arose. The parties whose work made the model
possible were offered a comparable mechanism afterwards, and only for future collection.

---

## What the model was built from, and when

OpenAI published
[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) on **28 May 2020**,
describing GPT-3, an autoregressive language model with 175 billion parameters. The abstract notes
"methodological issues related to training on large web corpora."

[Data, Consent and Compensation](03-data-consent-and-compensation.md) sets out the corpus in detail:
filtered Common Crawl covering 41 monthly shards from 2016-2019, WebText2, two internet book corpora
and English Wikipedia. The authors thanked "the millions of people who created content that was used
in the training of the model."

**Observed.** The corpus predates the model, and the people in it were not parties to anything. They
had no account, no contract, and no notice. There was no relationship through which permission could
have been asked or refused.

---

## What the provider protected, and when

The OpenAI API Terms of Use marked **"Updated August 6, 2021"**
([archived copy](https://web.archive.org/web/20211230012443/https://openai.com/api/policies/terms/))
contain four restrictions relevant here.

Section 3(b)(vii) — a developer may not "use the APIs to develop competing products or services."

Section 3(d) — "You will not use the APIs to discover any underlying components of our models,
algorithms, and systems, such as **exfiltrating the weights of our models by cloning via logits**."

Section 3(e) — "You may not use **web scraping, web harvesting, or web data extraction methods** to
extract data from the APIs, the Content, or OpenAI's or its affiliates' software, models or systems."

Section 12(l) — "any Disputes between them must be brought against each other on an individual basis
only… neither party can bring a Dispute as a plaintiff or class member in a class action,
consolidated action, or representative action."

**Observed.** Fifteen months after GPT-3 was published, the technique of extracting a model by
querying it was already prohibited by name, scraping the provider was prohibited in terms, and the
collective legal remedy was contractually removed.

The equivalent clauses remain in force. The current
[Terms of Use](https://openai.com/policies/terms-of-use/), effective 1 January 2026, still prohibit
users from "automatically or programmatically extract data or Output" and from "use Output to develop
models that compete with OpenAI," and still state that "class arbitrations, class actions, and
representative actions are prohibited."

Anthropic's [Commercial Terms](https://www.anthropic.com/legal/commercial-terms), effective 17 June
2025, carry the same pairing in a single document. Section B: "Anthropic may not train models on
Customer Content from Services." Section D.4: a customer may not "access the Services to build a
competing product or service, including to train competing AI models."

**Observed.** Both directions of the training question are settled in these contracts. Both are
settled between the provider and its customer. Neither party to that agreement is a person whose
work is inside the base model.

---

## What a website could do, and when

OpenAI's [crawler documentation](https://developers.openai.com/api/docs/bots) describes GPTBot:
"GPTBot is used to crawl content that may be used in training our generative AI foundation models.
Disallowing GPTBot indicates a site's content should not be used in training generative AI
foundation models."

The Internet Archive holds 1,738 captures of the GPTBot documentation page. **The earliest is dated
7 August 2023.**

**Observed.** The first evidenced date on which a website could signal "not for training" to this
provider is approximately two years after the same provider's terms prohibited the reverse.

**What this does not establish.** The archive date is the earliest capture, not a confirmed launch
date, and the August 2021 terms may not be the first version containing those clauses. Both
uncertainties run in the same direction: the clauses could be older, and GPTBot could be days
earlier. Neither closes a two-year gap.

---

## An opt-out existed, for a different purpose

A crawler opt-out did exist before 2023, and the honest version of this argument has to account for
it.

[Common Crawl](https://commoncrawl.org/ccbot) publishes the exact instruction to exclude its crawler:

```
User-agent: CCBot
Disallow: /
```

Common Crawl also now runs an
[opt-out registry](https://commoncrawl.org/blog/common-crawl-foundation-opt-out-registry).

Its stated purpose is not commercial model training. Common Crawl describes itself as "a non-profit
foundation founded with the goal of democratizing access to web information," maintaining an open
repository "universally accessible and analyzable by anyone," so that "organizations, academia, and
non-profits can work together to address complex challenges."

**Conclusion.** A site owner who permitted that crawler was consenting to inclusion in an open
research archive. Permission given for that purpose was later relied upon for a different one. The
opt-out matching the actual use did not exist until 2023, and no opt-out removes material already
collected.

The accurate claim is therefore not that refusal was impossible. It is that **the refusal on offer
was for something else.**

---

## Two instruments, two kinds of protection

The asymmetry is not which rules existed. It is which instrument each side could use.

| | The provider | The people in the corpus |
|---|---|---|
| Instrument | Contract, written by itself | Copyright and related rights |
| When it acts | Before access is granted | After the use has occurred |
| What it costs to invoke | Nothing; acceptance is a condition of entry | Litigation, per claimant |
| Who decides the terms | The party protected by them | A court, years later |
| Collective action | Removed by the class waiver | Needed most, available least |

**Conclusion.** Contract is preventive and self-authored. Copyright is remedial and must be enforced
by the person harmed, one work and one jurisdiction at a time. Handing one side a preventive
instrument and the other a remedial one produces exactly the outcome observed, without anyone having
to break a law.

The absent third instrument is procedural. [The Missing Trial Gate](02-the-missing-trial-gate.md)
sets out what a pre-market approval path would have required. No such gate existed, so there was no
moment at which anyone could withhold permission before release rather than object after it.

---

## The consent objection answers itself

The standard defence of training without permission is that the scale made it impossible: there were
millions of rightsholders, and no practical way to reach them.

**Observed.** Obtaining agreement from millions of people, individually, before granting access, at
negligible cost, is precisely what the terms of service in this paper do. It is the mechanism by
which every clause quoted above became binding.

**Conclusion.** The capability to obtain consent at scale was not missing. It was built, it works,
and it was pointed at protecting the service rather than at the people whose work the service was
built from. Impracticality does not explain the difference; sequence does. Consent machinery is
built when there is a relationship, and a relationship was created only once there was a product to
sell.

---

## What the law was asked, and what it answered

Deferral is also a decision.

[Data, Consent and Compensation](03-data-consent-and-compensation.md) records that there is no
general statutory royalty for model training in the United States, and that the
[US Copyright Office](https://www.copyright.gov/ai/) recommended allowing voluntary markets to
develop first.

The same paper checks what those voluntary markets produced. Licensing agreements between large
organisations exist and prove that permission can be negotiated. The checked announcements "do not
disclose financial terms or promise payment to each journalist, photographer, commenter or other
underlying contributor."

**Conclusion.** The institution with authority over the question was asked, and chose to wait for a
market. The market that followed transacts between organisations. It does not reach the individual
whose work is inside the corpus.

---

## What an organisation can do without waiting

Nothing above obliges anyone to repeat it. The conditions that made the original taking possible —
no relationship, no notice, no mechanism to refuse — do not apply to an organisation building for
its own use.

**Observed.** Provenance is now a documented property of a model rather than a matter of trust.
[Article 53(1)(d)](https://artificialintelligenceact.eu/article/53/) requires providers of
general-purpose AI models to "draw up and make publicly available a sufficiently detailed summary
about the content used for training of the general-purpose AI model, according to a template provided
by the AI Office." Article 53(1)(c) requires a policy to comply with Union copyright law, including
identifying and complying with a reservation of rights expressed under Article 4(3) of Directive (EU)
2019/790.

**Observed.** Article 53(2) exempts models released under a free and open-source licence from points
(a) and (b) — the two documentation duties — only. The copyright policy and the training-content
summary survive open-sourcing. No part of the exemption applies to models with systemic risk. These
obligations entered into force on 2 August 2025.

**Conclusion.** An organisation choosing a base model can ask what it was trained on and expect an
answer, or notice that no answer exists. That turns a claim of care into a document.

**Observed.** An organisation's own systems already hold what a model for its own work needs: its
code, incident history, tickets, runbooks, correspondence and operational records. That material is
unambiguously its own, and no other party holds it.

**Conclusion.** The corpus the frontier had to take is not the corpus an organisation needs. A model
that performs one organisation's work does not have to match a general-purpose model on general
tasks. It has to be useful on the work in front of it.

A model built this way inherits whatever its base carries. That cannot be undone by the layer added
on top, and it should not be described as though it could. The honest claim is not "we took nothing."
It is **we added nothing taken, and we chose our base knowing what it was.**

**Conclusion.** Where the boundary is the point, running the model on infrastructure the organisation
controls is stronger than a term promising the same thing. A clause is an instruction and can be
revised on notice; an environment that the data never leaves does not depend on anyone's continued
intention. [Why Model Rules Are Not Enforcement](05-why-model-rules-are-not-enforcement.md) makes the
same argument about safeguards inside a model. It applies equally to a data boundary written into a
contract.

---

## What this paper does not establish

- It does not show that training on public web content is unlawful. That question is contextual and
  unresolved; see [Data, Consent and Compensation](03-data-consent-and-compensation.md).
- It does not show that any provider broke a contract, a licence or a law.
- It examines one provider's terms in detail. Whether every provider adopted equivalent clauses on
  equivalent dates has not been checked.
- It does not establish that robots.txt has legal force anywhere. It is a convention, and honouring
  it is voluntary.
- It says nothing about whether any particular model was distilled from any other. No such
  allegation is relied on here, and none is needed.
- It does not establish that a model trained on one organisation's own records performs well enough
  for that organisation's work. That depends on the task and can only be answered by trying it.
- It does not compare the cost of self-hosted inference against paid API access. That comparison
  decides whether the approach is practical and has not been made here.

---

## What would make this right

**Proposals.** These are Clover's positions, not findings.

1. **Permission before collection, not refusal after.** An opt-out that arrives after training has
   occurred protects nothing already taken. Where a corpus is assembled for commercial training, the
   default should be that inclusion requires permission.
2. **Symmetry of instrument.** A provider that protects its model by contract before access should
   not rely, for its own inputs, on a remedial regime the supplier must fund a lawsuit to invoke.
3. **Do not remove the only affordable remedy.** A class waiver in a consumer agreement removes the
   sole realistic route for claims too small to litigate individually. Where harm is small per person
   and large in aggregate, collective remedy is the remedy.
4. **Purpose-bound consent.** Permission given to an open research archive is not permission for
   commercial model training. Downstream reuse for a materially different purpose needs its own
   basis.
5. **Publish the dates.** Providers should state when each protective clause entered their terms and
   when each opt-out mechanism became available. The sequence is a fact about the record and should
   not require an archive to reconstruct.
6. **Select a base model on its disclosed provenance.** An organisation building on someone else's
   weights should read the training-content summary before choosing, keep it on file, and treat its
   absence as information. Provenance is now disclosable; choosing not to look is a decision.
7. **Enforce the data boundary in the environment, not only in the contract.** Where it matters that
   material never reaches a third party, the control should be the infrastructure rather than a term
   that can be revised on notice.

---

## Evidence ledger

| Claim | Evidence | What it does not prove |
|---|---|---|
| GPT-3 was published 28 May 2020 with 175 billion parameters, trained on large web corpora | [GPT-3 paper](https://arxiv.org/abs/2005.14165) | The corpus composition, which is cited in [paper 03](03-data-consent-and-compensation.md), not re-derived here |
| By 6 August 2021 the API terms barred competing products, weight extraction "by cloning via logits", and scraping OpenAI's systems | [OpenAI API Terms, updated 6 Aug 2021](https://web.archive.org/web/20211230012443/https://openai.com/api/policies/terms/), §§3(b)(vii), 3(d), 3(e) | That this was the first version to contain them, or that other providers matched the date |
| The same terms removed class, consolidated and representative actions | ibid., §12(l) | That the waiver has been tested or enforced against a claimant |
| The equivalent prohibitions remain current | [OpenAI Terms of Use, eff. 1 Jan 2026](https://openai.com/policies/terms-of-use/) | That the clauses are identical in scope to the 2021 versions |
| One provider's contract bars itself from training on customer content and bars the customer from training a competing model | [Anthropic Commercial Terms, eff. 17 June 2025](https://www.anthropic.com/legal/commercial-terms), §B and §D.4 | That either clause has been audited or litigated |
| Disallowing GPTBot signals that content should not be used for training | [OpenAI crawler documentation](https://developers.openai.com/api/docs/bots) | That the signal is honoured, or that it has legal effect |
| The GPTBot documentation page is first archived 7 August 2023 | [Internet Archive captures](https://web.archive.org/web/*/https://platform.openai.com/docs/gptbot) | The launch date; an archive capture is a lower bound |
| A general crawler opt-out predated it, for an open research archive | [Common Crawl CCBot](https://commoncrawl.org/ccbot) | The date CCBot exclusion became available, which was not checked |
| Common Crawl's stated purpose is open research access, not commercial training | ibid. | That its data was not lawfully available for other uses |
| There is no general statutory royalty for training in the US, and the Copyright Office recommended voluntary markets first | [US Copyright Office](https://www.copyright.gov/ai/), as cited in [paper 03](03-data-consent-and-compensation.md) | The outcome of pending litigation, or that voluntary markets cannot mature |
| Providers of general-purpose AI models must publish a sufficiently detailed summary of training content, and must hold a copyright-compliance policy honouring reserved rights | [EU AI Act, Article 53(1)(c) and 53(1)(d)](https://artificialintelligenceact.eu/article/53/), in force 2 August 2025 | That published summaries are complete, accurate or enforced |
| Open-source release exempts a provider from the two documentation duties only; the copyright policy and training-content summary remain, and no exemption applies to systemic-risk models | [EU AI Act, Article 53(2)](https://artificialintelligenceact.eu/article/53/) | Anything outside the Union, or how "sufficiently detailed" will be interpreted |

---

**The principle.** Protection that arrives before the harm, written by the party it protects, is a
different thing from protection that must be bought in a courtroom afterwards. Both are law. Only one
of them was available to the people who made the model possible, and it arrived second.

**They took because they could. No one else has to, because the work an organisation needs is
already its own.**
