# Data, Consent and Compensation

Status: evidence checked 6 September 2026.

## The questions

What human work was used to build general-purpose AI? Did the people who made it consent? Can they
find out whether their work was used? Who was credited or paid? What happens to the new material a
person gives an AI service while using it?

There is no one answer across providers, models, countries or kinds of data. That inconsistency is
itself part of the problem.

---

## The scale is real; "the entire world's data" is not a literal fact

Large models were built from human-created material at a scale that makes individual provenance hard
to see.

OpenAI's 2020 GPT-3 paper,
[Language Models are Few-Shot Learners](https://arxiv.org/html/2005.14165), describes a training mix
of filtered Common Crawl, WebText2, two internet book corpora and English Wikipedia. It says the
Common Crawl source covered 41 monthly shards from 2016-2019: 45 terabytes of compressed text before
filtering and 570 gigabytes after filtering. GPT-3 trained on 300 billion tokens. The authors closed
by thanking "the millions of people who created content that was used in the training of the model."

Meta's [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B) says only that its
more than 15 trillion pretraining tokens came from "a new mix of publicly available online data."
Google's [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card) describes six trillion
tokens from broad categories: web documents, code and mathematics. It describes filtering, but not a
work-by-work source list.

For images, LAION-5B contains
[5.85 billion image-text pairs](https://arxiv.org/abs/2210.08402) extracted through Common Crawl.
LAION's own release page says the metadata is CC BY 4.0 but **the linked images retain their own
copyright**, recommends the dataset for research, and says it does not recommend using the uncurated
dataset to create ready-to-go industrial products while basic safety research remains in progress.

These records support "internet-scale human data." They do not support the literal statement that
any model contains every person's data or the entire internet.

---

## Category disclosure does not answer the individual question

The question a creator needs answered is concrete: **Was my book, photograph, code, voice or post
used? Under what authority?**

Public documentation commonly answers at a different level. "Publicly available online data,"
"web documents" and "licensed third-party data" describe categories, not the works inside them.
OpenAI's [GPT-4 technical report](https://arxiv.org/html/2303.08774v6) says the model used public and
licensed data, then states that, because of the competitive landscape and safety implications, it
provides no further detail about dataset construction, architecture, hardware, compute or training
method.

The EU AI Act now requires providers of general-purpose models to publish a sufficiently detailed
summary of training content and maintain a copyright-compliance policy. But the Act itself says the
summary should be comprehensive rather than technically detailed, naming main collections and
providing a narrative of other sources. It also says the AI Office does not perform a work-by-work
copyright assessment. The Commission calls its template a
["common minimal baseline"](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models).

This is a material improvement in public visibility. It still does not necessarily let one person
look up one work.

---

## Public access is not the same thing as permission

There are three separate questions:

1. Was the material technically accessible?
2. Was its use legally permitted?
3. Was its use acceptable to the person who created it or appears in it?

A yes to the first does not answer the other two.

The clearest regulatory example is not a generative model but facial recognition. Clearview AI argued
that consent was unnecessary because the images it scraped were publicly available. Canadian privacy
commissioners rejected that argument, finding that the collection of billions of images without
knowledge or consent was
[illegal mass surveillance](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/).
The Dutch Data Protection Authority later fined Clearview EUR 30.5 million for a database of more
than 30 billion scraped photographs and found both unlawful collection and inadequate transparency.
The [Dutch decision summary](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition)
says plainly that an internet photograph can place someone in the database without their knowledge
or consent.

Those findings apply privacy law in particular jurisdictions to biometric processing. They are not a
court holding that every public web page requires consent before any model can learn from it. They do
establish the narrower principle: public accessibility does not erase privacy rights or decide the
lawfulness of a new purpose.

---

## Copyright does not produce one answer either

The US Copyright Office's May 2025
[report on generative AI training](https://www.copyright.gov/ai/) concludes that several stages of
model development use copyrighted works in ways that implicate owners' exclusive rights. Whether
those acts are excused as fair use depends on the works, source, purpose and output controls.

Its conclusion draws a line rather than declaring all training legal or illegal:

- analytical or research uses whose outputs do not substitute for the source are more likely to be
  transformative;
- commercial use of vast collections to make expressive content that competes in existing markets,
  especially through illegal access, can go beyond established fair-use boundaries;
- licensing markets are emerging but inconsistent;
- government intervention in those markets was premature at the time of the report, though targeted
  intervention should be considered where market failure persists.

A US district court's June 2025 ruling in *Bartz v. Anthropic* illustrates the split. According to the
[Authors Guild's case summary](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/),
the judge treated training on legally acquired books as fair use for the named plaintiffs, but did
not treat building a permanent library from pirate sites as fair use. The certified class and the
USD 1.5 billion settlement process concern piracy, not a general royalty for training. The official
[settlement site](https://www.anthropiccopyrightsettlement.com/) describes the settlement as
preliminarily approved; this dossier does not claim a later final status that it could not confirm
from the court record.

One case in one court does not settle the national or international question. It does show why
"trained without consent" and "infringed copyright" cannot be used as interchangeable claims.

---

## Compensation exists where bargaining power exists

It is false to say nobody has been paid.

- Shutterstock gave OpenAI a six-year licence to image, video and music data. Shutterstock says its
  [Contributor Fund](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year)
  has compensated hundreds of thousands of artists for training use and pays ongoing royalties tied
  to licensing newly generated assets.
- OpenAI's [Axel Springer partnership](https://openai.com/index/axel-springer-partnership/) permits
  use of publisher content for model training and display with attribution and links.
- Its [News Corp partnership](https://openai.com/index/news-corp-and-openai-sign-landmark-multi-year-global-partnership/)
  grants access to current and archived publications for display and product enhancement.
- Reddit's [Google partnership](https://redditinc.com/news/reddit-and-google-expand-partnership)
  provides structured access to public posts and comments for product improvement and model
  training.

The first example says contributors participate in compensation. The checked announcements for the
publisher and Reddit deals do not disclose financial terms or promise payment to each journalist,
photographer, commenter or other underlying contributor. A licence between organisations proves
that permission can be negotiated. It does not prove that value reaches every person whose work is
inside the licensed corpus.

There is no general statutory royalty for model training in the United States. The Copyright Office
recommended allowing voluntary markets to develop first. That is the current policy position, not a
finding that voluntary markets will reach everyone or distribute value fairly.

---

## Data supplied while using the product is a separate issue

Pretraining data and user conversations are different collections and should not be blurred together.
Provider practices also differ.

As checked on 6 September 2026, OpenAI's
[data-use policy](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/)
says content from individual services such as ChatGPT may be used to train models unless the person
opts out. Business products and API traffic are excluded from training by default unless the
organisation opts in.

Anthropic's current
[consumer policy](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training)
says consumer chats are used when a person chooses model improvement, explicitly opts in, or when a
conversation is flagged for safety review. Its
[commercial policy](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)
says inputs and outputs are not used for training by default, except where feedback or permission is
given.

These are current published policies, not proof of what every person understood on first use or what
older versions said. They do show why a responsible account must name the product, account type,
setting and date instead of saying simply "AI companies train on your chats."

---

## What responsibility would require

This is Clover's proposal.

1. **A searchable source record.** A creator should be able to ask whether a work or domain was used,
   not only read a category such as "web data." Where a public work-level list would itself expose
   private or dangerous information, an independent authority should be able to answer the query.
2. **A lawful basis before ingestion.** Record whether each source is owned, licensed, public domain,
   covered by a statutory exception, or unresolved. Access is not a category of permission.
3. **Consent for identity.** A person's face, voice and biometric identity should not be treated as
   ordinary content merely because somebody else owns the file.
4. **Compensation that reaches contributors.** Catalogue licences should disclose how value reaches
   the authors, artists and users who supplied the material. Collective licensing may be needed where
   individual negotiation cannot scale.
5. **A first-use data statement.** Before a person enters content, say what is retained, whether a
   human may review it, whether it trains models, the default, and how to refuse or delete it. Do not
   bury the decision in a general privacy policy.
6. **No retroactive permission.** A later licence can govern future use and settle past claims. It
   should not rewrite the record to suggest that permission existed when data was acquired.

---

## Evidence ledger

| Claim | Evidence | What it does not prove |
|---|---|---|
| GPT-3 used hundreds of billions of tokens from web and book corpora | [GPT-3 paper, section 2.2](https://arxiv.org/html/2005.14165) | Which individual works were present, or whether every use required permission |
| Llama 3 and Gemma disclose broad source categories | [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B); [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card) | That the categories are unlawful, or a work-level inventory |
| LAION-5B contains 5.85 billion web-derived image-text pairs and does not own image copyright | [LAION paper](https://arxiv.org/abs/2210.08402); [LAION release](https://laion.ai/blog/laion-5b/) | That every model using LAION reproduced every image |
| GPT-4 withheld dataset-construction detail | [GPT-4 report, section 2](https://arxiv.org/html/2303.08774v6) | That OpenAI had no internal provenance records |
| EU summaries are a minimum baseline, not work-by-work review | [Commission template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models); [AI Act recitals 107-108](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689) | That the summary duty has no enforcement value |
| Publicly available biometric data can still be unlawfully collected | [Canadian commissioners](https://www.priv.gc.ca/en/opc-news/news-and-announcements/2021/nr-c_210203/); [Dutch DPA](https://autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition) | A universal rule for all non-biometric training data |
| US training fair use is contextual, not categorical | [US Copyright Office, Part 3](https://www.copyright.gov/ai/) | The outcome of any pending lawsuit |
| Lawfully acquired training and pirate-library acquisition were treated differently in *Bartz* | [Authors Guild case summary](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/) | Binding precedent outside that case or a royalty ruling |
| Licensed training and contributor compensation are possible | [Shutterstock](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year) | Fair compensation across the wider training corpus |
| Current consumer and business data defaults differ | [OpenAI](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/); [Anthropic consumer](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training); [Anthropic commercial](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training) | What users saw or understood at an earlier launch date |
