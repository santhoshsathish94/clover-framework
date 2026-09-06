# Why Model Rules Are Not Enforcement

Status: evidence checked 6 September 2026.

## The question

If an AI company knows that impersonation, unlicensed copying, deception and other harms are wrong,
why not put a rule in the model and make the model enforce it?

## The short answer

**Observed:** Providers already put rules and refusal behaviour into many models. Those measures
reduce harm, but providers' own evaluations and independent research show that they do not create a
hard boundary. Open-weight safeguards can be modified after release. Closed services can control
access more strongly, but still face jailbreaks, false positives and false negatives.

**Conclusion:** A rule in a model is one control in a system. It is not the system's enforcement.
Responsibility must be held at every layer that can still change the outcome.

---

## Rules already exist

Meta's [Llama 3 Acceptable Use Policy](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
prohibits infringement, non-consensual impersonation, fraud, disinformation, defamation and several
other harms. OpenAI's GPT-4 report describes reinforcement learning, rule-based reward models,
external experts and deployment monitoring intended to make the model refuse disallowed content.
Google's Gemma model card describes content filters, red teaming, memorisation evaluation and a
prohibited-use policy.

It would therefore be false to say that model companies put no rules in their systems. The harder
question is what those rules can actually hold.

---

## A refusal is probabilistic

OpenAI's [GPT-4 technical report](https://arxiv.org/html/2303.08774v6) says its model-level
interventions increase the difficulty of eliciting bad behaviour, but that jailbreaks still exist.
The original ChatGPT launch post likewise expected false negatives and false positives from its
external content filter.

This is a classification problem as well as a policy problem. The system must infer what the user is
asking, whether an identity is real, whether consent exists, which jurisdiction applies, and whether
the use is fraud, abuse, news, criticism, parody, research or something else. Much of that Context is
not present in the prompt or media.

A model cannot inspect a private consent agreement it was never given. It cannot know whether the
person uploading a photograph took it, licensed it, stole it, or received permission from the person
shown. It can ask, warn and refuse obvious cases. Treating its inference as proof would block lawful
work and still allow harmful work phrased differently.

---

## Open weights remove the provider's control point

The UK AI Security Institute explains the difference in
[Managing risks from increasingly capable open-weight AI systems](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems).
A closed-weight deployer controls access points and can add filters, monitor use and enforce an
acceptable-use policy. Open weights can be shared and modified without that oversight.

AISI reports that current safety fine-tuning can be undone with dozens of examples in minutes, that
external safeguards can be trivially disabled, that provenance measures can be circumvented, and
that an open-weight release cannot be rolled back. It recommends full-access audits, staged release,
monitoring, training-data curation and, for high-risk systems, not releasing the weights.

A separate [2026 investigation](https://www.irishtimes.com/business/2026/05/25/ai-guardrails-stripped-from-meta-and-google-models-in-minutes/)
reported that a free tool called Heretic removed safeguards from Meta and Google open-weight models
in minutes on consumer hardware. The result demonstrates that safety alignment in those tested models
was a removable feature, not a structural property. This is evidence about those models, not proof
that every safeguard in every future architecture will be removable.

---

## Closed access gives control, not certainty

A provider running a model on its own servers can do things an open-weight provider no longer can:

- authenticate and rate-limit users;
- run input and output classifiers;
- retain abuse signals;
- suspend accounts;
- update the model or policy centrally;
- block a tool or capability;
- investigate incidents;
- withdraw a version.

Those are real enforcement mechanisms. They are stronger than an instruction embedded in a prompt.
They still do not establish that the user has consent, that an output will remain private, that a
classifier understands the law, or that nobody will reproduce the capability elsewhere.

The point is not "companies cannot enforce anything." It is: **companies can enforce only at the
control points they retain, and they are responsible for using those control points.**

---

## Responsibility is layered

### Model layer

The model can carry rules, name uncertainty, ask for missing authority, refuse its part in a clear
violation and explain the boundary to the human. That is the role `AGENTS.md` can play in one Clover
cycle.

It cannot bind whoever edits the instruction, fine-tunes the weights, changes the application or
moves to another model.

### Product and infrastructure layer

The provider can create the controls listed above. For identity-sensitive generation it can also
require consent attestations, preserve provenance, limit bulk generation, create victim reporting and
removal routes, and keep evidence when abuse is alleged.

The provider cannot honestly describe those measures as a guarantee. It can disclose their tested
coverage and observed failures.

### Deployment layer

The enterprise or organisation decides what data the model can reach, what action it can take, who
reviews its work, what evidence is required and whether a refusal can be bypassed. It can make an
agent's policy binding through permissions, isolation, approval gates and audit logs.

This is why Clover says the environment should enforce what matters. A model refusing to write a
file is weaker than an environment in which it has no permission to write the file.

### Human layer

The human controls the route around the system. If an agent refuses an unlicensed or deceptive use,
the human can rewrite the request, remove the instruction, use another model or do the act manually.
No model can prevent all four.

The human's responsibility is therefore not passive supervision. It is not to override the boundary,
not to obtain the same prohibited outcome elsewhere, and to inspect what the agent actually did.

### Legal and market layer

Governments can define duties that neither provider nor deployer may waive, require independent
access, assign liability and provide redress. Markets and procurement can make compliance a condition
of purchase. These are the layers that can turn a voluntary promise into a cost for breaking it.

---

## Why companies do not simply block every disputed use

There are legitimate conflicts, not only technical weakness.

A digital likeness may be fraud, or it may be documentary reconstruction, accessibility, criticism,
parody or art. Copyright may require permission, or an exception such as fair use may apply. Privacy,
publicity and speech rules differ by jurisdiction. A global model cannot resolve every contested
legal question from text and pixels alone.

Overbroad refusal causes harm too: it can suppress lawful speech, prevent a person from working with
their own material, and give a private platform control over criticism. The NO FAKES debate includes
exactly this concern, which is why the proposed bill contains public-interest and speech exceptions
and a counter-notice process.

That complexity explains why no model classifier can be the final judge. It does not justify doing
nothing. It supports escalation to a human with the Context and authority to decide, plus an external
review route for the person affected.

---

## What responsibility requires from each actor

### AI companies

1. Put the strongest feasible protections into training, model behaviour and product access.
2. Test the safeguards under adversarial modification, not only as shipped.
3. Match release form to risk. Do not publish high-risk weights merely because an API would be less
   open.
4. Publish failure rates, bypasses and incidents with enough detail for independent scrutiny.
5. Preserve provenance and make generated media detectable.
6. Provide a fast path for affected people to report, remove, appeal and preserve evidence.
7. Keep responsibility for provider-controlled choices instead of assigning all misuse to the user.

### Enterprises and deployers

1. Put permissions, isolation, approvals and logging around the model.
2. Know which model and release form are in use, including unsanctioned local models.
3. Give one named human the authority and competence to stop an action.
4. Monitor actual actions and outcomes, not prompts and policy documents alone.
5. Suspend use and report when the system crosses a boundary.

### Humans using the tools

1. Give honest Context and Direction.
2. Do not remove or route around a safeguard to get a prohibited result.
3. Verify rights and consent outside the model.
4. Review the outcome in the real system.
5. Remain accountable for choosing to act on it.

### Governments

1. Define minimum controls proportionate to capability, reach and irreversibility.
2. Require independent evaluation where self-assessment is not enough.
3. Assign duties across the value chain so each actor owns what it can control.
4. Require incident reporting, regulator access, withdrawal powers and meaningful remedies.
5. Protect lawful expression through review and appeal, not by leaving harmful use ungoverned.

---

## Evidence ledger

| Claim | Evidence | What it does not prove |
|---|---|---|
| Major model releases carry safety rules and evaluations | [Llama 3 model card and AUP](https://huggingface.co/meta-llama/Meta-Llama-3-8B); [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card); [GPT-4 report](https://arxiv.org/html/2303.08774v6) | Consistent compliance in deployment |
| GPT-4's mitigations reduced unsafe behaviour but jailbreaks remained | [GPT-4 report, section 6](https://arxiv.org/html/2303.08774v6) | Current failure rates of later models |
| Closed deployments retain access-control tools that open weights do not | [UK AISI](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems) | That closed systems are safe or misuse-free |
| Open-weight safeguards can be removed and releases cannot be recalled | [UK AISI](https://www.aisi.gov.uk/blog/managing-risks-from-increasingly-capable-open-weight-ai-systems) | That openness has no research, competition or transparency benefit |
| Safeguards were removed from tested models in minutes | [Financial Times report republished by The Irish Times](https://www.irishtimes.com/business/2026/05/25/ai-guardrails-stripped-from-meta-and-google-models-in-minutes/) | Independent reproduction across all model families |
| EU law assigns duties to both providers and deployers | [EU AI Act, Articles 16, 26, 53 and 55](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689) | A complete global allocation of responsibility |
| The proposed US likeness law preserves speech exceptions and notice/counter-notice | [NO FAKES reported bill](https://www.govinfo.gov/app/details/BILLS-119s4591rs) | That the bill will become law or operate as intended |
