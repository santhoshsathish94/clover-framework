# Likeness, Privacy and the Moral Gap

Status: evidence checked 6 September 2026. This is a policy argument, not legal advice. Laws differ
by jurisdiction and facts.

## The ordinary example

A person gives an AI tool a photograph of a friend and asks for a convincing video in which the
friend makes a false claim.

The intuitive question is simple: **did the friend agree?**

The legal questions are not simple. Who took the photograph? Where is each person? Is the use
commercial? Is the video intimate, defamatory, satirical, political, journalistic or private? Was it
shared? Could a reasonable viewer believe it? Different answers activate different bodies of law.

That fragmentation creates a moral gap. A use can be harmful before a court has a clean statute to
apply, and a person can lack a practical remedy even when a claim theoretically exists.

---

## Copyright protects the work, not automatically the person depicted

The US Copyright Office says that, as a general rule, the author and initial owner of a photograph is
[the person who takes it](https://www.copyright.gov/engage/photographers/), subject to exceptions such
as work made for hire and later transfers.

That means the friend in the photograph is not necessarily its copyright owner. Copyright may give
the photographer a claim over copying or adapting the image while giving the depicted person no
copyright claim at all. Their interests instead may fall under privacy, publicity, defamation,
consumer protection, biometric-data or digital-replica law.

The distinction matters for an AI agent. Asking "is this file copyrighted?" does not answer "did this
person consent to having their identity used this way?"

---

## Existing US protection is incomplete by design and geography

The US Copyright Office studied the issue in
[Copyright and Artificial Intelligence, Part 1: Digital Replicas](https://www.copyright.gov/ai/).
Its conclusion was not that existing law was enough. It recommended prompt federal action because
existing remedies were insufficient and inconsistent.

The report's recommendations are especially relevant to the friend's-photo case:

- protection should cover **all people**, not only celebrities or people with commercial value;
- liability should not be limited to commercial uses, because much of the harm is personal;
- a federal right should create a national floor while allowing stronger state protection;
- both injunctions and monetary remedies are needed so protection is not available only to people
  who can fund litigation;
- free-speech interests require balancing, not an indiscriminate ban.

That is evidence from the agency responsible for US copyright policy that copyright, publicity,
privacy, consumer and communications laws left a real gap. It is not itself a law.

### TAKE IT DOWN covers one severe category

The TAKE IT DOWN Act became
[Public Law 119-12 on 19 May 2025](https://www.govinfo.gov/app/details/PLAW-119publ12).
It criminalises specified publication of non-consensual intimate imagery, including digital
forgeries, and creates platform removal duties.

It is an important protection. It does not cover every non-intimate fake, every false endorsement or
every reputational harm. A fake ordinary video of a friend making a non-sexual statement sits outside
its defining category.

### NO FAKES would cover more, but is not law

The revised NO FAKES Act would create a federal right over highly realistic digital replicas of a
person's voice or visual likeness. It covers living and deceased people, requires written and
specific licences, creates liability for unauthorised public distribution, and preserves exclusions
for news, documentary, history, commentary, criticism, scholarship, satire and parody.

As of 6 September 2026, the official
[GovInfo record](https://www.govinfo.gov/app/details/BILLS-119s4591rs) shows S. 4591 as "Reported in
Senate" on 24 June 2026. The bill had not passed both chambers or been signed into law. Its safeguards
are therefore proposed rights, not current federal remedies.

The reported bill also shows why responsibility cannot be reduced to "make the model refuse." It
creates liability for a product primarily designed, substantially used or marketed to make
unauthorised replicas of specifically identified people. A general-purpose tool is not automatically
liable merely because someone misuses it. Platform safe harbours depend substantially on notice and
response, and the bill says there is no general duty to monitor.

---

## Europe shows both stronger rules and their limit

The EU AI Act prohibits untargeted scraping of internet or CCTV facial images to create or expand a
facial-recognition database. It also requires providers of content-generating systems to make
synthetic outputs machine-detectable where technically feasible, and requires deployers to disclose
deepfakes clearly. Those Article 50 transparency duties began applying on 2 August 2026.

A label answers "was this generated?" It does not answer "did the person consent?" A perfectly
labelled fake can still humiliate, defame or exploit someone.

The Act also preserves exceptions and other law. That is necessary. A documentary reconstruction,
political satire, news report, fraud and private joke are not the same act merely because each uses a
likeness. A responsible boundary needs context and review, not only face detection.

---

## Public is not permission

Clearview AI is a useful demonstration because the company made the argument directly: consent was
unnecessary because the images it scraped were publicly available.

Canadian privacy commissioners
[rejected that argument](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/).
They found that collecting billions of images into a biometric identification service without
knowledge or consent was illegal mass surveillance. The Dutch Data Protection Authority later found
Clearview's database unlawful and insufficiently transparent, imposing a
[EUR 30.5 million fine](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition).

Those are decisions under Canadian and European privacy law about biometric processing. They do not
create a worldwide legal rule for every use of public information. They establish something narrower
and important: changing the purpose and power of public data can change the rights at stake.

A photograph shared so friends can see it is not necessarily offered as raw material for identity
search, synthetic video, advertising or false speech.

---

## The moral rule

Law asks whether a recognised claim, jurisdiction and remedy apply. Responsibility asks one question
earlier: **what are we doing to this person, and did they agree?**

Clover's proposed rule is:

> Do not use an identifiable person's face, voice or likeness to make them appear to say or do
> something they did not say or do without their informed permission, unless a legitimate public
> interest such as reporting, criticism, documentary evidence or satire clearly justifies the use.
> Make the synthetic nature visible and do not present it as authentic.

This is deliberately not "never depict a person." An absolute ban would suppress journalism,
history, accessibility, criticism and art. It is also not "anything legal is acceptable." Laws are
uneven, slow and bounded by jurisdiction. The rule protects dignity in the space where law is silent.

---

## What each actor should do

### The model provider

- Detect when a request seeks a realistic replica of an identifiable person.
- Require the user to state the authority or consent for that use, while acknowledging that an
  attestation is not proof.
- Refuse clear deception, harassment, sexual exploitation and false endorsement.
- Preserve provenance and make generated media detectable.
- Give the depicted person a usable report, removal and appeal path.
- Publish how often these controls fail, not only that they exist.

### The deploying company

- Establish consent and purpose outside the model.
- Restrict who can provide identity material and what tools may act on it.
- Keep records of source, permission, generation, approval and distribution.
- Review the actual output and intended audience before publication.
- Stop and notify affected people when the boundary fails.

### The human using the tool

- Do not treat possession of a photograph as permission from the person in it.
- Do not override a refusal or move to another tool to obtain the same result.
- Do not claim the model's output is authentic.
- Check what was produced and where it will travel.
- Remain answerable for the act even when the model performed the transformation.

### Government

- Create a consistent right that protects private individuals as well as public figures.
- Preserve legitimate speech with reviewable exceptions rather than leaving everyone unprotected.
- Put duties on model providers, deployers and distribution platforms instead of assigning the whole
  burden to victims after publication.
- Provide fast removal, evidence preservation, appeal and affordable redress.

---

## Evidence ledger

| Claim | Evidence | What it does not prove |
|---|---|---|
| A photograph is generally owned initially by its photographer | [US Copyright Office](https://www.copyright.gov/engage/photographers/) | That the subject has no privacy, publicity, contract or other right |
| Existing US law leaves digital-replica gaps, including for private people and non-commercial harm | [US Copyright Office, Part 1](https://www.copyright.gov/ai/) | That every replica should be unlawful |
| TAKE IT DOWN is enacted and limited to intimate visual depictions | [Public Law 119-12](https://www.govinfo.gov/app/details/PLAW-119publ12) | Coverage for ordinary non-intimate impersonation |
| NO FAKES had been reported by the Senate committee but was not law on the evidence date | [GovInfo, S. 4591 reported version](https://www.govinfo.gov/app/details/BILLS-119s4591rs) | Its eventual fate or how courts would interpret it |
| EU law requires machine-readable marking and deepfake disclosure | [EU AI Act, Article 50](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689#art_50) | Consent or legality of the depicted use |
| Regulators rejected "publicly available means no consent needed" for Clearview biometrics | [Canada](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/); [Netherlands](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition) | A universal rule covering every kind of public data or jurisdiction |
