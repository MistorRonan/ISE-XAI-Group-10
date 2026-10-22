# Slide Plan — 2026-10-06

**Title:** Using Feature Attribution to Audit Social Bias in LLM Pretraining Data Filters

**Research question:** Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?

**Description:** Data filters silently decide whose language LLMs learn from. SHAP and LIME can show which words drive a removal, and our experiments show they reveal real problems with our chosen filters. However, despite LIME and SHAP being useful for finding suspicious behaviour, they shouldn't be used as a verdict due to their instability and position as being descriptive, rather than normative (they describe what the model does but can't say what it should do).

> **Note:** If you are using a reference in the PowerPoint, make a write-up of it using the template in daily notes and push it to Git.

---

## Slide 1 — Title

- Stick our title and problem description on this.

---

## Slide 2 — The Data Filters

The automated tools that decide which text from the internet gets into an LLM's training data.

- **Pipeline diagram:** flow diagram showing the process of how data gets filtered, from Common Crawl to testing.
- **A concrete case where each filter has caused harm/faulted** (reference these):
  - Dodge 2021 (C4 blocklist removes LGBTQ+ and AAE text)
  - Gururangan 2022 (quality filter favours richer, urban schools)
  - Sap 2019 (AAE gets flagged as toxic)
- Check if these papers are already in our Git research, and if not, add them.

---

## Slide 3 — Filter types and their common gap

- **Quality filters:** Gururangan (linear LR)
- **Toxicity:** Mendu HarmFormer (black-box Longformer)
- **Bias:** Udagawa regard RoBERTa
- All three audit only in aggregate, and none explains an individual removal.

---

## Slide 4 — Why we need post-hoc explanation

- **Interpretability spectrum diagram:** Linear → GAM → Decision Tree → Random Forest → Gradient Boosting → Deep Neural Network. Put the filters we are discussing on the diagram.
- **Rudin (2019):** post-hoc explanations of black-box filters can mislead, which is what we are testing in our experiments.

---

## Slide 5 — SHAP and LIME for text

Contrasting example of how both work.

**SHAP:** *"The club condemned a post calling fans disgusting idiots"* (or any other example sentence of your choosing)
- Splits the score fairly between words
- Contributions add up to the final score
- Same answer every run

**LIME:** *"The club condemned a post calling fans disgusting idiots"* (or any other example sentence of your choosing)
- Deletes random words and fits a simple model
- Only trustworthy near this sentence
- Can change every run

- Maybe try to make a cool visual of how this works for both, and then how the end results will differ.
- Have a quick sentence at the bottom of the slide tying this back into our experiment slides, which are next.

---

## Slide 6 — Exp 1: What drives a removal?

- Visual representation of the results of the experiment.
- Explanation of what it is and why we did it.
- Relevance to XAI and our presentation.

---

## Slide 7 — Exp 2: Does the identity word carry the decision?

- Visual representation of the results of the experiment.
- Explanation of what it is and why we did it.
- Relevance to XAI and our presentation.

---

## Slide 8 — Overall result and can we trust the explainers?

- **Chart on one side:** LIME runs 10 times on the same sentence. One bar per word, with error bars showing how much it jumped around. Differentiation in results.
- **One big number:** SHAP and LIME agreed on the top word in 35/40 sentences (Exp 2; add Exp 1's result when it's in).
- Conclusion that they're useful for spotting problems, not proof.

---

## Slide 9 — Critical Analysis

- **Explanations disagree and are unstable:** Krishna et al. 2022 and experiment findings
- **Can be fooled:** Slack et al. 2020
- **Can say which words, not whether the definition is legitimate**
- **Word removal is not neutral:** link back to lecture content
- **Not causal:** textbook page 47 and page 53
- **Accuracy isn't equal to explainability:** Ngueajio
- **Scope honesty**

---

## Slide 10 — Verdict and Next Steps

- Link it all back to our title and description, our research and the lecture content.
- Give some next steps, and how we'll use this in our research paper.

---

## Slide 11 — References
