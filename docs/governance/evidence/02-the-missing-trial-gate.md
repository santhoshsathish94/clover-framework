# The Missing Trial Gate

Status: evidence checked 6 September 2026.

## The question

Why could a general-purpose AI system move from a laboratory to the whole world before its effects
were understood, when medicines normally move through bounded trials before reaching a market?

## The short answer

**Observed:** Medicine has a legal gate. General-purpose AI did not have an equivalent gate when
ChatGPT launched. The developer decided when the public experiment began, and public use became part
of how the product was tested and improved.

**Conclusion:** The difference is not that medicine is cautious and technology is careless by
nature. Medicine acquired an external authority, evidence stages and the power to stop after people
died under the older approach. General-purpose software inherited a release-and-update model instead.
Digital distribution then allowed that model to reach a population faster than law could respond.

**Proposal:** Do not copy clinical trials mechanically. Borrow their governing idea: greater possible
harm and greater reach require stronger evidence, independent review and a bounded step before wider
exposure.

---

## What a medicine gate actually does

The [FDA drug-development process](https://www.fda.gov/patients/learn-about-drug-and-device-approvals/drug-development-process)
has five stages: discovery, preclinical research, clinical research, FDA review and post-market
monitoring.

Before a US clinical trial starts, a sponsor must submit an Investigational New Drug application.
The FDA can place the investigation on hold if participants face unreasonable risk or the application
lacks enough information. Clinical work then normally moves from a small Phase 1 study focused on
safety and dosage, through Phase 2 efficacy and side effects, to larger Phase 3 studies. Before a
drug can be marketed, the developer submits the evidence and the FDA decides whether to approve it.
The [FDA's own description](https://www.fda.gov/patients/drug-development-process/step-3-clinical-research)
lists 20-100 participants in Phase 1, up to several hundred in Phase 2 and 300-3,000 in Phase 3. The
[marketing review](https://www.fda.gov/patients/drug-development-process/step-4-fda-drug-review)
includes inspection for fabrication, manipulation or withholding of data.

The system did not arise from foresight alone. In 1937, an untested liquid formulation called Elixir
Sulfanilamide killed more than 100 people in 15 states. The law at the time did not require a new drug
to be safety-tested. The FDA records that the disaster hastened the 1938 statute that created a new
system of drug control. Its
[historical account](https://www.fda.gov/about-fda/histories-product-regulation/sulfanilamide-disaster)
quotes the manufacturer saying there was no responsibility on the company's part because it had not
foreseen the result.

The hard lesson is not that medicine always governed well. It is that its binding gate was built
after preventable deaths demonstrated what voluntary caution could not do.

---

## What happened with general-purpose AI

OpenAI released ChatGPT on 30 November 2022 as a free
["research preview"](https://openai.com/index/chatgpt/). The launch post named important known
limitations: plausible but false answers, sensitivity to prompt wording, guessing rather than
clarifying, harmful responses, and false negatives and positives in its external content filter. It
also said broad use would provide feedback on real-world harms and teach OpenAI how to improve future
systems.

Nothing in that description made participation a controlled trial. Access was public. Reuters
reported that, according to a UBS study using Similarweb data, ChatGPT had reached an estimated
[100 million monthly active users by January 2023](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/).

The same act was therefore three things at once:

- a product launch;
- an experiment from which the provider learned; and
- exposure of people and institutions that were not research subjects under a controlled protocol.

Calling it a research preview described the developer's intent. It did not create an external gate,
a bounded population, independent approval or a duty to compensate anyone harmed.

---

## The EU now supplies part of the missing structure

The [EU AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689)
is important precisely because it shows that staged, supervised AI deployment is possible.

For defined high-risk systems it requires risk management, data governance, documentation, logs,
human oversight, testing before market entry and a conformity assessment. It also provides
post-market monitoring, incident reporting, regulator access and powers to withdraw a system.
Real-world testing outside a sandbox requires a plan, regulatory approval, bounded duration,
oversight, reversibility and, with stated exceptions, informed consent. Member states must provide
regulatory sandboxes for limited, supervised testing before market entry.

But it is not a clinical-trial regime for every general-purpose model:

- most AI systems are not classified as high-risk;
- most stand-alone high-risk uses follow provider internal control rather than mandatory independent
  third-party assessment;
- general-purpose models have a separate set of transparency, copyright and systemic-risk duties;
- a sandbox is available, not a mandatory passage for every model;
- most of these obligations arrived after mass public adoption.

As of this evidence date, the European Commission says general-purpose-model obligations began in
August 2025, transparency duties in August 2026, and the remaining high-risk-system duties are being
phased in later. The law is a material change in authority. It does not retroactively make the first
world-scale deployments controlled trials.

---

## Why the gate was missing

No source can prove a single intention shared by an industry. The record supports five structural
reasons.

### 1. The product inherited software's legal path

A medicine is in a category that requires external permission before marketing. General-purpose AI
was offered as software and a service. The main US cross-sector risk framework, the
[NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), says explicitly that it is
voluntary. Sector laws can still apply, and the FTC and other agencies retain powers, but there was
no general FDA-like decision that every general-purpose model had to obtain before public release.

### 2. Capability and distribution scaled together

The GPT-3 paper documented smooth performance gains from scaling model size and data. A digital
service could then be made available globally without manufacturing a physical unit for every user.
The first public deployment reached a population comparable to a large country within two months.
There was no natural pause between proving capability and distributing it.

### 3. The developer also controlled the experiment

OpenAI described ChatGPT as iterative deployment: feedback from the public would reveal unknown
risks and improve later systems. That can produce valuable evidence. It also means the party seeking
the benefit of release defined the test, held the data and decided whether the evidence justified
continuing.

### 4. The warnings were mostly non-binding

The Asilomar Principles, OECD AI Principles, UNESCO Recommendation and NIST AI RMF each named parts
of responsible development. None was a universal market-access decision. The 2023 request for a
pause likewise had signatures but no authority.

### 5. The harms do not look like one clinical endpoint

A medicine trial can be designed around a dose, an indicated use, a population and specified health
outcomes. A general-purpose model can be used for writing, hiring, education, fraud, art, software,
warfare or private conversation. Its harms may fall on people who never used it. That makes testing
harder. It does not make testing optional; it means the test must follow capability, use and reach
rather than one universal endpoint.

---

## A trial gate suitable for AI

This is Clover's proposal, not a description of existing law.

### Stage 1: bounded development

Document the intended capability, data provenance, people who could be affected, prohibited uses,
known limitations and the conditions that would stop development. Keep model access inside the team
until those questions have evidence.

### Stage 2: independent adversarial evaluation

Give qualified outside evaluators sufficient access to test foreseeable misuse, privacy leakage,
deception, security, discrimination and attempts to remove safeguards. Publish methods, material
results and unresolved risks, not only a safety score selected by the provider.

### Stage 3: limited real-world trial

Use a bounded population, duration and geography. Tell participants that the system is being tested,
what data is collected and how it will be used. Provide a route to withdraw, report harm and seek
redress. Do not use people who cannot meaningfully refuse as the first trial population.

### Stage 4: evidence-based expansion

Expand only when the observed outcome supports it. Define reach thresholds in advance. A material
increase in capability, autonomy, tool access or user population should require a new decision, not
be treated as a routine update.

### Stage 5: post-release accountability

Keep incident reporting, independent access, rollback, withdrawal and compensation mechanisms for
the life of the system. For open-weight release, assess the model after adversarial modification,
because the [UK AI Security Institute](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems)
notes that released weights can be modified without oversight and cannot be rolled back.

The gate should become more demanding as capability, reach and irreversibility rise. A small bounded
assistant and a model released as modifiable weights to the world are not the same decision.

---

## Evidence ledger

| Claim | Evidence | What it does not prove |
|---|---|---|
| Drugs move through preclinical work, phased human trials, review and post-market monitoring | [FDA process](https://www.fda.gov/patients/learn-about-drug-and-device-approvals/drug-development-process) | That every drug follows identical timing, or that AI and medicine have identical risks |
| FDA can hold a trial and approve or refuse marketing | [FDA clinical research](https://www.fda.gov/patients/drug-development-process/step-3-clinical-research); [FDA review](https://www.fda.gov/patients/drug-development-process/step-4-fda-drug-review) | That the FDA model can be copied unchanged for AI |
| The modern safety gate followed a lethal failure under weaker law | [FDA history of Elixir Sulfanilamide](https://www.fda.gov/about-fda/histories-product-regulation/sulfanilamide-disaster) | That catastrophe is necessary before regulation |
| ChatGPT was a public research preview with known limitations | [OpenAI launch post](https://openai.com/index/chatgpt/) | The full internal pre-release test programme |
| ChatGPT reached about 100 million monthly users in two months | [Reuters, citing UBS/Similarweb](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/) | An audited count from OpenAI |
| NIST's main cross-sector framework was voluntary | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | That no binding sector law applied to any AI use |
| The EU now requires controlled real-world testing and pre-market duties for defined high-risk uses | [EU AI Act, Articles 9, 43, 57, 60 and 61](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689) | A mandatory clinical-style trial for every general-purpose model |
| Open-weight release is difficult to reverse | [UK AISI](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems) | That closed access eliminates misuse |
