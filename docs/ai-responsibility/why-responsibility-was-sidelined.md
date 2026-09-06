# Why Responsibility for AI Was Sidelined

Status: evidence checked 6 September 2026.

This is the master document for the claim made in the closing section of the Clover website.

**The website makes this argument without naming anyone. This document names the specifics.** The
public page stays neutral on purpose: the failure is systemic rather than the conduct of one
organisation, and naming actors there would turn a shared problem into an argument about one of them.
Here the record is set out in full, with every source named, so the claim can be checked and
challenged rather than simply believed.

## The claim

**Responsibility for AI was known, flagged and then sidelined whenever it conflicted with
capability, commercial expansion and the pursuit of dominance.**

That is Clover's conclusion from the record below. It is not a claim that every organisation acted
from one motive, that responsibility work was insincere, or that any single decision had one proven
cause.

## How each case is read

Clover describes one cycle: **Context → Direction → Execution → Outcome → Growth.**

- **Context** is the relevant evidence about the real system, established before acting.
- **Direction** is the human-owned outcome, its boundaries, and who answers for it.
- **Execution** is the work done inside those boundaries.
- **Outcome** is what the real system shows, favourable or not.
- **Growth** is what the Outcome taught, carried into the next cycle so it is not repeated.

A skipped stage does not disappear. Its cost lands on somebody who did not choose it. Each case below
is named for the stage that was skipped, and each is walked through the whole cycle.

## What was already known before any of it reached the world

- The 2017 [Asilomar AI Principles](https://futureoflife.org/open-letter/ai-principles/), signed by
  1,797 AI and robotics researchers, named safety, responsibility, human control, shared benefit and
  the need to avoid cutting corners in a race.
- The [OECD AI Principles](https://oecd.ai/en/ai-principles), adopted by member governments in 2019,
  included accountability, transparency, robustness and human-centred values.
- UNESCO's 193 member states adopted the
  [Recommendation on the Ethics of Artificial Intelligence](https://unesdoc.unesco.org/ark:/48223/pf0000381137)
  in 2021, setting out responsibility, audit, risk assessment and human oversight.
- Emily Bender, Timnit Gebru, Angelina McMillan-Major and Margaret Mitchell's peer-reviewed 2021
  paper, [*On the Dangers of Stochastic Parrots*](https://doi.org/10.1145/3442188.3445922), warned
  about scale, environmental and financial cost, undocumented training data, and the need to weigh
  affected people before development.

These do not prove that every later consequence was predicted. They establish that the questions were
already on the table before these systems reached the public.

The people raising them did not hold release authority.
[MIT Technology Review reported](https://www.technologyreview.com/2020/12/04/1013294/google-ai-ethics-research-paper-forced-out-timnit-gebru/)
that Google said the *Stochastic Parrots* draft did not meet its publication bar, while Timnit Gebru
said the company forced her out and cut off her access before a negotiated departure; the sequence
remains disputed, and more than 1,400 Google staff signed a protest letter. In February 2021 Google
fired Margaret Mitchell, the founder and other co-head of the same Ethical AI team, saying an
investigation found she had moved files outside the company in breach of its code of conduct. The
[BBC's contemporaneous account](https://www.bbc.com/news/technology-56135817) also records that she
had been locked out of Google's systems for five weeks, had criticised Gebru's departure, and that
both researchers had raised concerns about censorship.

That record does not prove competitive dominance was the sole motive for either decision. It does
establish that both leaders of that ethics team lost their roles around disputes over this work,
while product and release authority stayed elsewhere.

The full chronology and its limits are in
[It Was Not Unforeseen](01-it-was-not-unforeseen.md).

## Case 1 — Context taken instead of established: training on the world's work

This is the first breach, and the one everything else rests on.

**Context.** Building a general-purpose model out of the world's material required knowing what that
material was, who made it, and what authority permitted its use. That work was not done to the depth
the use demanded. Public disclosure settled at categories rather than sources: Meta's
[Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B) says only "publicly available
online data"; Google's [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card) describes
web documents, code and mathematics; OpenAI's
[GPT-4 technical report](https://arxiv.org/html/2303.08774v6) explicitly withheld dataset
construction, citing competition and safety.
[LAION-5B](https://arxiv.org/abs/2210.08402), a public image-text dataset of 5.85 billion pairs, was
assembled by crawling the web. An ordinary person cannot ask whether their work is inside.

The assumption that reachable means permitted had already been rejected elsewhere. Canada's privacy
commissioners [found](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/)
that Clearview AI's scraped facial-image database amounted to illegal mass surveillance, and the
[Dutch data-protection authority](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition)
fined the same company for unlawful collection. In both cases the images were already public. The US
Copyright Office's 2025 [report on generative-AI training](https://www.copyright.gov/ai/) concludes
that several stages of training implicate rights holders' exclusive rights, and that fair use turns
on source, purpose and output. In *Bartz v. Anthropic*, as
[summarised by the Authors Guild](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/),
lawfully acquired books and books taken from pirate libraries were treated differently.

**Direction.** The people whose work formed the corpus were not asked and had no way to refuse.
Direction over that material was exercised by the parties able to reach, copy and hold it at scale.
**Clover's conclusion:** permission was treated as unnecessary because the material was reachable,
and because the reach itself was concentrated in very few hands.

**Execution.** Training proceeded at world scale while permission, provenance and payment remained
unresolved. Nothing in the execution required them to be settled first, so they were not.

**Outcome.** The value settled where the data was held. The capability built from that material is
owned and monetised by the parties who assembled it. Compensation exists, but as islands rather than
a rule: Shutterstock reports paying a
[contributor fund](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year)
and ongoing royalties, and OpenAI has licensed content from publishers including
[Axel Springer](https://openai.com/index/axel-springer-partnership/) and
[News Corp](https://openai.com/index/news-corp-and-openai-sign-landmark-multi-year-global-partnership/).
Those are private contracts, not a general right to attribution, refusal or payment for everyone
whose work contributed.

**Growth.** The lesson has not been made binding. The European Commission's
[public training-content summary template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models)
is an explicit minimum baseline rather than a work-level account, and OpenAI's
[data-use policy](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/)
still treats consumer and business content differently by default. The next model can be built the
same way.

**This case does not establish** that every training use was unlawful, that categorical disclosure is
itself illegal, or the outcome of any pending litigation. The detail is in
[Data, Consent and Compensation](03-data-consent-and-compensation.md).

## Case 2 — Direction set by reach: releasing to everyone at once

**Context.** Known limitations were documented in OpenAI's
[ChatGPT launch post](https://openai.com/index/chatgpt/). The gap was not ignorance of the flaws; it
was the absence of any external body holding evidence about them.

**Direction.** How many people would be exposed, and how quickly, was decided by the party doing the
releasing. No independent authority had to agree that the reach was proportionate to what was known.

**Execution.** The free public "research preview" reached an estimated
[100 million monthly users](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/)
within two months, as reported by Reuters citing UBS and Similarweb. Nine days after launch its own
chief executive [wrote publicly](https://x.com/sama/status/1601731295792414720) that the system was
"incredibly limited, but good enough at some things to create a misleading impression of greatness",
that "it's a mistake to be relying on it for anything important right now", and that it was "a preview
of progress". The warning did not slow the reach, because nothing in the release required it to.

A prototype demonstrates that something can work. A product has been through the phases that make it
usable by people who did not build it. Where that distinction was not held, the demonstration itself
became the claim. Google's "Hands-on with Gemini" video of December 2023 appeared to show the model
responding to speech and live video; the company
[confirmed](https://techcrunch.com/2023/12/07/googles-best-gemini-demo-was-faked/) that it had
"prompted Gemini using still image frames from the footage, and prompting via text", after Bloomberg
reported the discrepancy, and a Google DeepMind research vice-president said the video "illustrates
what the multimodal user experiences built with Gemini could look like. We made it to inspire
developers." What was shown was aspirational. What it created was demand.

Compare a field that keeps the gate outside the developer: under the US FDA, medicines move through
preclinical work, permission before human trials, bounded phases, and
[review before marketing](https://www.fda.gov/patients/drug-development-process/step-4-fda-drug-review).
That structure exists because an earlier failure under weaker law
[killed more than a hundred people](https://www.fda.gov/about-fda/histories-product-regulation/sulfanilamide-disaster).

**Outcome.** The public became the test population, and supplied the evidence the developer used.
Calling a worldwide release a preview did not make it a controlled trial.

**Growth.** Binding duties arrived later and partially. The US
[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) says
explicitly that it is voluntary; the
[EU AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689) now imposes
real pre-market duties, but for defined high-risk uses rather than as a universal gate for
general-purpose models. Once weights are released openly, the UK AI Security Institute
[notes](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems)
that the decision is difficult to reverse.

**This case does not establish** that AI and medicine are the same risk class, that no internal
testing took place, or that any provider released a prototype *in order to* create hype. What it
establishes is that the prototype reached the world at product scale, and that the gap between the
demonstration and the product was left for the public to discover. The detail is in
[The Missing Trial Gate](02-the-missing-trial-gate.md).

## Case 3 — Execution inside boundaries nobody can hold

**Context.** Providers know their safeguards are imperfect; their own reports record refusals being
bypassed and classifiers making mistakes.

**Direction.** Prohibited-use rules are published, and models are trained to refuse. The intent is
stated clearly enough.

**Execution.** A rule inside a model is guidance, not a boundary. It cannot verify consent it has
never seen, cannot bind the human who reroutes around it, and cannot survive being modified after an
open-weight release. Licence terms commonly place responsibility for the output on the downstream
user.

**Outcome.** When the boundary fails, the consequence lands outside the system that failed — on the
person depicted, quoted, imitated or harmed.

And it did not stay with the providers. The same treatment of reachable work as available work
reappeared wherever the tools went. NewsGuard's AI Tracking Center has identified
[3,749 AI content-farm news and information sites](https://www.newsguardtech.com/special-reports/ai-tracking-center/)
across sixteen languages, typically publishing dozens of articles a day with little or no human
oversight and funded by programmatic advertising that pays regardless of provenance. Detection does
not close the gap: NewsGuard found that leading chatbots
[failed to identify AI-generated video](https://www.newsguardtech.com/special-reports/top-ai-chatbots-dont-recognize-ai-generated-videos)
in 78-95% of prompts, including the assistant made by the company that built the video tool. The law
protecting a person's likeness remains partial in the meantime — the US
[TAKE IT DOWN Act](https://www.govinfo.gov/app/details/PLAW-119publ12) covers intimate visual
depictions, and ordinary impersonation largely falls outside it, as
[Likeness, Privacy and the Moral Gap](04-likeness-privacy-and-the-moral-gap.md) sets out.

One release shows the gap between a stated rule and an enforceable boundary. Sora 2 launched on 30
September 2025 **already requiring opt-in for the use of an individual's voice and likeness**. Within
weeks, unauthorised clips using Bryan Cranston's voice and likeness were circulating on the app; the
company had blocked depictions of Martin Luther King Jr. at his estate's request after users generated
"disrespectful depictions"; and Robin Williams's daughter had asked the public to stop sending her
AI-generated videos of her father. On 20 October OpenAI issued a joint statement with Cranston,
SAG-AFTRA, United Talent Agency, the Association of Talent Agents and Creative Artists Agency
committing to stronger guardrails and to responding expeditiously to complaints, as
[CNBC reported](https://www.cnbc.com/2025/10/20/open-ai-sora-bryan-cranston-sag-aftra.html). The rule
had been in place the entire time. What did not exist was anything that could refuse the generation
before the depicted person found out about it.

The same shape appears where the stakes are highest. Anthropic holds a defence contract worth up to
$200 million and, [as NBC News reported](https://www.nbcnews.com/tech/security/anthropic-ai-defense-war-venezuela-maduro-rcna259603)
on 20 February 2026, was the first AI company allowed to offer services on classified networks,
through a 2024 partnership with Palantir. Palantir's
[own announcement](https://investors.palantir.com/news-details/2024/Anthropic-and-Palantir-Partner-to-Bring-Claude-AI-Models-to-AWS-for-U.S.-Government-Intelligence-and-Defense-Operations/)
said the models would support government operations by "helping U.S. officials to make more informed
decisions in time-sensitive situations". Anthropic has maintained that it will not allow its systems to
be used in lethal autonomous weapons or for domestic surveillance. After the Wall Street Journal and
Axios reported that its products featured in the operation to capture Venezuela's president, NBC
recorded two things that belong together: the Pentagon said it was reviewing its relationship with the
company, and **"it is unclear how Anthropic's Claude was used."**

**Clover's conclusion:** the boundary in that arrangement is a supplier's published policy. It is not
enforced by the environment the model runs in; the provider cannot always establish afterwards whether
it was crossed; and the counterparty with the power to withdraw the contract is the one pressing
against it. A red line that only one party can see, and that the other party can price, is not a
boundary.

**Clover's conclusion:** the breach at the source became the default behaviour downstream. Nobody
using these tools was told that being able to reach something is not permission to take it, because
the systems they were using had been built on exactly that assumption.

**Growth.** Enforcement that would actually hold has to exist in the product, the environment, the
organisation and the law, not only in the model. That has not been made general.

**This case does not establish** that safeguards are worthless, that most people using these tools
intended to take anyone's work, or that every content farm was built with a general-purpose
assistant. The detail is in
[Why Model Rules Are Not Enforcement](05-why-model-rules-are-not-enforcement.md).

## Case 4 — Outcome carried by people who never participated

**Context.** Two groups were affected without ever being users: people whose likeness, voice or
identity can now be reproduced, and people whose occupations were exposed to automation.

**Direction.** Neither group was represented where the decisions were made. Their exposure was a
by-product of someone else's outcome.

**Execution.** Capability was shipped ahead of the rights and protections that would have made it
answerable. Legal protection for a person's likeness remains fragmented, so a real harm can fall
between copyright, privacy and publicity law.

**Outcome.** The measured labour evidence is uneven, not apocalyptic: reduced work and earnings in
some exposed freelance occupations, sharp demand declines in selected writing and translation tasks,
a widening relative employment gap for younger workers in exposed occupations, alongside growing
demand for complementary work and no large aggregate earnings effect in some national records.
Aggregate stability and a destroyed livelihood are not contradictory. Both are true, and only one of
them is felt by a person.

**Growth.** No general duty exists to measure displacement, protect entry paths, or give an affected
person a route to explanation, removal, appeal or remedy.

**This case does not establish** economy-wide replacement, or that every harmful use is illegal. The
detail is in [Likeness, Privacy and the Moral Gap](04-likeness-privacy-and-the-moral-gap.md) and
[Work, Livelihoods and Accountability](06-work-livelihoods-and-accountability.md).

## Case 5 — Growth recorded but never made binding

**Context.** The [2026 Stanford AI Index](https://hai.stanford.edu/ai-index/2026-ai-index-report/responsible-ai)
records 362 AI incidents in 2025, up from 233 in 2024, while responsible-AI benchmark reporting
remains sparse beside capability reporting, and average Foundation Model Transparency Index scores
rose from 37 in 2023 to 58 in 2024 before falling to 40 in 2025.

**Direction.** Frontier-risk frameworks are published and revised, so the intent to learn is real.
But decisive authority usually stays inside the organisation that benefits from the release: under
OpenAI's [Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/), an
internal Safety Advisory Group recommends whether safeguards are sufficient while leadership makes
the final deployment decision, and requirements may be adjusted if another frontier developer
releases a high-risk system without comparable safeguards.

**Execution.** An incident may change one filter, one policy or one product surface. It rarely
changes the structure that produced it, and almost never changes it for anyone else.

**Outcome.** The same categories of failure recur across providers and jurisdictions.

**Growth.** Real Growth would need an owner, an investigation, an established cause, a correction,
evidence that the correction works, and a rule that binds the next release. No mechanism currently
makes all six happen. Learning that cannot bind the next cycle is not Growth; it is a record of
Outcome.

**This case does not establish** that no organisation learns anything. The detail is in
[Why Incidents Still Do Not Become Accountability](07-why-incidents-do-not-become-accountability.md).

## Why no one takes responsibility

Four structural reasons, none of which requires anyone to have acted in bad faith.

**Responsibility is divided until nobody holds the whole Outcome.** It is split across whoever builds
the model, whoever deploys it, whoever distributes it, whoever uses it, and several regulators. Each
can hold one piece honestly while the whole remains unowned.

**The interested party assesses itself.** Where safety review is internal, the organisation that
gains from releasing is also the one judging whether releasing is acceptable.

**Restraint reads as losing.** When a competitor ships without comparable protection, holding a
boundary looks like surrendering position rather than avoiding a risk. A floor that any one
participant can lower is not a floor.

**Those who could enforce it are also competing.** Governments are at once regulator, investor,
purchaser, promoter and geopolitical rival. The 2025 US
[AI Action Plan](https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/)
is titled *Winning the AI Race*; the White House says it will "cement U.S. dominance in artificial
intelligence", and quotes AI and Crypto Czar David Sacks saying the United States must win the AI
race to remain the leading economic and military power. The same reflex appears in other strategic
industries: a 2024
[White House trade fact sheet](https://bidenwhitehouse.archives.gov/briefing-room/statements-releases/2024/05/14/fact-sheet-president-biden-takes-action-to-protect-american-workers-and-businesses-from-chinas-unfair-trade-practices/)
reported that Chinese EV exports grew 70% from 2022 to 2023, that China controlled over 80% of
certain segments of the EV-battery supply chain and 80-90% of certain parts of the global solar
supply chain, and answered with tariffs and domestic investment. Those documents do not say that
China's clean-technology position caused the AI plan; reading them as one continuous pursuit of
technological dominance is Clover's interpretation.

**Clover's conclusion:** no actor is required to hold the entire cycle, so the stage that gets skipped
is reliably the one with no owner — and the cost is reliably carried by whoever was not in the room.

## Clover's conclusion

The record does not support saying that nobody knew. It does not support saying that every safety
commitment was false. It does not establish one shared intention behind every organisation.

It supports something narrower and harder to dismiss:

> Responsibility was sidelined because capability, commercial direction and competitive dominance
> held decision power, while responsibility remained advice. The risks were known and flagged, but no
> common, binding authority could make those warnings control what was built or released next.

That is what "sidelined" means here. Responsibility existed — in research, in principles, in teams, in
published frameworks. It simply never held equal authority over what was pursued.

And it is not finished. Every case above describes a cycle still running today.

## What this claim does not establish

- It does not establish that any particular organisation or country acted in bad faith.
- It does not establish that competitive dominance was the sole motive for Timnit Gebru's departure
  or Margaret Mitchell's firing.
- It does not establish that China's position in EVs, batteries or solar caused the US AI Action Plan.
- It does not establish that all responsibility work was insincere or ineffective.
- It does not establish that every training use was unlawful, or predict any pending legal outcome.
- It does not establish that every later harm was foreseeable or preventable.

Those limits do not weaken the pattern. They keep the conclusion at the scale the evidence can carry.

## Further documents

- [AI Responsibility: Evidence and the Ask](README.md) indexes the complete dossier and its evidence
  standard.
- [It Was Not Unforeseen](01-it-was-not-unforeseen.md) documents the warning chronology and the
  disputed employment record.
- [The Missing Trial Gate](02-the-missing-trial-gate.md) compares world-scale iterative release with
  staged external approval.
- [Data, Consent and Compensation](03-data-consent-and-compensation.md) examines training sources,
  permission, attribution and payment.
- [Likeness, Privacy and the Moral Gap](04-likeness-privacy-and-the-moral-gap.md) covers harms that
  fall between existing legal rights.
- [Why Model Rules Are Not Enforcement](05-why-model-rules-are-not-enforcement.md) separates model,
  product, deployment, human and government control.
- [Work, Livelihoods and Accountability](06-work-livelihoods-and-accountability.md) records the
  measured labour effects and their limits.
- [Why Incidents Still Do Not Become Accountability](07-why-incidents-do-not-become-accountability.md)
  explains why learning often fails to bind the next release.
- [Incidents and Adoption Outcomes](08-incidents-and-adoption-outcomes.md) records what adoption
  produced in five documented cases, and what each source does not establish.
- [Responsibility Cannot Be Optional](../responsibility-cannot-be-optional.md) contains the broader
  request to model providers, governments, enterprises and users.

## Evidence ledger

| Claim used in this document | Evidence | What it does not establish |
|---|---|---|
| Safety, responsibility, human control and race avoidance were named before mass adoption | [Asilomar AI Principles, 2017](https://futureoflife.org/open-letter/ai-principles/) | That every later consequence was predicted |
| Governments adopted AI accountability and oversight principles before these systems reached the public | [OECD AI Principles, 2019](https://oecd.ai/en/ai-principles); [UNESCO Recommendation, 2021](https://unesdoc.unesco.org/ark:/48223/pf0000381137) | That those principles created a release gate |
| Researchers warned about model scale, training data and affected people beforehand | [Bender, Gebru, McMillan-Major and Mitchell, 2021](https://doi.org/10.1145/3442188.3445922) | That every recommendation was rejected |
| Both co-leads of Google's Ethical AI team lost their roles around disputes over that work | [MIT Technology Review, 4 December 2020](https://www.technologyreview.com/2020/12/04/1013294/google-ai-ethics-research-paper-forced-out-timnit-gebru/); [BBC, 20 February 2021](https://www.bbc.com/news/technology-56135817) | That dominance was the sole motive for either employment decision |
| Training sources were disclosed as categories, and GPT-4 withheld dataset detail | [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B); [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card); [GPT-4 technical report](https://arxiv.org/html/2303.08774v6) | That categorical disclosure is itself unlawful |
| Public availability does not by itself make collection lawful | [Canadian privacy commissioners](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/); [Dutch DPA](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition) | A universal rule for all non-biometric training data |
| Training implicates exclusive rights, and acquisition route matters | [US Copyright Office, 2025](https://www.copyright.gov/ai/); [*Bartz v. Anthropic* summary](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/) | The outcome of any pending litigation |
| Contributor and publisher compensation is possible but contractual, not general | [Shutterstock contributor fund](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year); [Axel Springer](https://openai.com/index/axel-springer-partnership/); [News Corp](https://openai.com/index/news-corp-and-openai-sign-landmark-multi-year-global-partnership/) | Fair compensation across the wider training corpus |
| The same take-what-is-reachable pattern reappeared downstream at scale, with 3,749 AI content-farm news sites identified across 16 languages | [NewsGuard AI Tracking Center](https://www.newsguardtech.com/special-reports/ai-tracking-center/) | That any specific provider's tool produced them, or that most users intended to deceive |
| Leading chatbots could not identify AI-generated video in 78-95% of prompts, including one made by the video tool's own developer | [NewsGuard, chatbots and AI-generated video](https://www.newsguardtech.com/special-reports/top-ai-chatbots-dont-recognize-ai-generated-videos) | That no detection method works, or that detection would be sufficient protection |
| A video tool required opt-in for a person's voice and likeness at launch, yet unauthorised generations of named individuals appeared within weeks, and the developer then agreed stronger guardrails with performer unions and agencies | [CNBC, 20 October 2025](https://www.cnbc.com/2025/10/20/open-ai-sora-bryan-cranston-sag-aftra.html) | That the developer intended those generations, or that the strengthened guardrails have prevented recurrence since |
| A frontier provider's models reached classified defence and intelligence networks under a contract worth up to $200 million, its partner said they would help officials "make more informed decisions in time-sensitive situations", and after reported use in a capture operation it remained unclear how the model had been used | [NBC News, 20 February 2026](https://www.nbcnews.com/tech/security/anthropic-ai-defense-war-venezuela-maduro-rcna259603); [Palantir announcement, 2024](https://investors.palantir.com/news-details/2024/Anthropic-and-Palantir-Partner-to-Bring-Claude-AI-Models-to-AWS-for-U.S.-Government-Intelligence-and-Defense-Operations/) | **That the model selected a target, made any decision, or that any policy was breached.** None of that is established |
| A worldwide research preview reached roughly 100 million monthly users in two months | [OpenAI launch post](https://openai.com/index/chatgpt/); [Reuters, citing UBS/Similarweb](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/) | An audited count, or the full internal test programme |
| The developer's own chief executive said publicly that the system was "incredibly limited", created "a misleading impression of greatness", and should not be relied on for anything important | [Sam Altman, 11 December 2022](https://x.com/sama/status/1601731295792414720) | That the release was made in order to create hype; the statement is a warning, not an admission of intent |
| A flagship capability demonstration was assembled from still frames and text prompts rather than the live interaction it appeared to show, as the provider confirmed | [TechCrunch, 7 December 2023, quoting Google and Bloomberg's first report](https://techcrunch.com/2023/12/07/googles-best-gemini-demo-was-faked/) | That the model lacked the underlying capability, or that every provider demonstration was staged |
| Medicine keeps the approval gate outside the developer | [FDA marketing review](https://www.fda.gov/patients/drug-development-process/step-4-fda-drug-review); [FDA history of the failure that created it](https://www.fda.gov/about-fda/histories-product-regulation/sulfanilamide-disaster) | That the FDA model transfers unchanged to AI |
| Binding duties arrived later and partially, and open release is hard to reverse | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework); [EU AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689); [UK AISI](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems) | A universal pre-market gate for general-purpose models |
| EU training-content summaries are a minimum baseline, and provider data-use defaults still vary | [European Commission template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models); [OpenAI data-use policy](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/) | That the duties have no enforcement value |
| Incidents rose while transparency fell and responsible-AI reporting lagged capability | [Stanford AI Index 2026, Responsible AI](https://hai.stanford.edu/ai-index/2026-ai-index-report/responsible-ai) | A complete census of incidents, their causes or preventability |
| An internal group advises while leadership decides, and requirements may shift after a competitor's release | [OpenAI Preparedness Framework update](https://openai.com/index/updating-our-preparedness-framework/) | That any specific safeguard was lowered |
| US AI policy is framed around winning a race and cementing dominance | [AI Action Plan announcement, 23 July 2025](https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/); [White House trade fact sheet, 14 May 2024](https://bidenwhitehouse.archives.gov/briefing-room/statements-releases/2024/05/14/fact-sheet-president-biden-takes-action-to-protect-american-workers-and-businesses-from-chinas-unfair-trade-practices/) | That China's clean-technology position caused the AI plan, or that dominance was the only objective |
| Responsibility was sidelined because capability and dominance held decision power | Clover's synthesis of the records above and the [structural evidence](07-why-incidents-do-not-become-accountability.md) | A single motive for every actor or decision |