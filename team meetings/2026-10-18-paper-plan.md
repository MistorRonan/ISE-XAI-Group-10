# Paper Plan — 2026-10-18

**Title:** Do Attributions Tell the Truth About Data Filters? Measuring the Faithfulness of SHAP and LIME in Identity-Term Bias Audits

**Research question:** Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?

**Thesis:** SHAP and LIME give faithful explanations of a toxicity filter when the identity-term signal is strong. Their attributions match the measured effect of swapping the identity word, and they recover exact Shapley values on a transparent filter. However, faithfulness and stability drop when the signal is weak, and they shift with the choice of removal baseline. So attribution can triage a filter for bias, but it cannot certify the filter is fair.

**In one line:** feature attribution is faithful enough to flag identity-term bias in a pretraining filter, but not reliable enough to certify the filter is fair.

**Description:** The presentation showed that SHAP and LIME point at real problems in Detoxify, and that they agree with each other most of the time. But two explainers agreeing doesn't prove either one is right. The paper goes a step further: we test the explainers against **references whose answers we already know**, and we measure *where* they stop being trustworthy. The verdict stays the same as the presentation ("enough to find the problem, not enough to deliver the verdict"), but this time it's backed by measurement.

---

| Exp | Name | Status | Spec |
|---|---|---|---|
| 1 | Report vs promote | Done (presentation) | [EXP1](../experiments/report%20vs%20promote/EXP1_report_vs_promote.md) |
| 2 | Identity-term swap | Done (presentation) | [EXP2](../experiments/identity%20term%20swap/EXP2_identity_swap.md) |
| 3 | Counterfactual swap + baseline check | New, core | [EXP3](../experiments/counterfactual%20swap/EXP3_counterfactual_swap.md) |
| 4 | Glass-box filter | New, core | [EXP4](../experiments/glass%20box%20filter/EXP4_glass_box.md) |
| 5 | Deletion test | New, core | [EXP5](../experiments/deletion%20test/EXP5_deletion.md) |
| 6 | LIME stability vs signal strength | New, core | [EXP6](../experiments/lime%20stability/EXP6_lime_stability.md) |
| 7 | Second filter family | Optional

---


---

### Abstract (about 150 words)

- One sentence each, in this order:
  - **Problem:** pretraining filters silently remove text, and they've been shown to fail specific groups.
  - **Gap:** these filters are audited only in aggregate, and nobody checks whether the explanations an auditor would use are faithful.
  - **What we did:** tested SHAP and LIME on Detoxify against two references (counterfactual identity swaps and exact Shapley values on a transparent filter), plus a deletion test and a stability test.
  - **Main numbers:** 2–3 headline results (e.g. correlation with the counterfactual effect; instability on weak signals; baseline sensitivity).
  - **Verdict:** the one-line thesis.

---

### I. Introduction (about 0.4 page)

- **Opening hook:** data filters decide whose language LLMs learn from. One concrete harm with a citation, e.g. C4's blocklist removing LGBTQ+ and AAE text (Dodge 2021).
- **Why explanations are needed:** the filters that matter most are deep nets (Detoxify, HarmFormer, regard classifiers), so they can't be read directly. An auditor (W1-L01 p5: "can I verify this system is non-discriminatory?") would reach for post-hoc tools like SHAP and LIME.
- **The problem with that:** post-hoc explanations are a simpler model of the model, so they can be wrong (Rudin 2019). Two explainers agreeing doesn't make them right (Krishna 2022). So before an auditor relies on them, we need to check whether they're **faithful**.
- **Research question** 
- **Contributions:** a short numbered list of 3–4 items, one per new experiment:
  1. A counterfactual reference for attribution, built from minimal-pair identity swaps (Exp 3).
  2. An exact-Shapley reference from a transparent filter, which also tests the accuracy–interpretability trade-off (Exp 4).
  3. A deletion-based faithfulness comparison of SHAP vs LIME vs random (Exp 5).
  4. A systematic link between signal strength and explainer stability, testing whether LIME's R² or SHAP–LIME disagreement can warn an auditor (Exp 6).
- **Last sentence:** a one-line preview of the answer (the thesis).
- **Draws on:** slides 1–4; Dodge, Sap, Gururangan, Rudin, Krishna; W1-L01.

---

### II. Background and Related Work (about 0.5 page)

- **A. Bias in pretraining filters** (1 paragraph)
  - Blocklist: Dodge 2021. Quality filter: Gururangan 2022. Toxicity: Sap 2019 (AAE flagged), Mendu 2025 (HarmFormer flags *reporting* pages). Bias filter: Udagawa 2025.
  - Identity-term templates as a bias test: Dixon 2018. The Jigsaw unintended-bias data that Detoxify `unbiased` was built for: Borkan 2019.
  - **Gap:** all of these audit in aggregate (rates, prevalence); none explains an individual removal. (Slide 3's point.)
- **B. Feature attribution for text** (1 paragraph)
  - SHAP: Shapley values from game theory (Lundberg & Lee 2017). The four axioms, especially efficiency and null player (textbook p48). Note that SHAP's text explainer uses a *partition* approximation, not exact Shapley.
  - LIME: a local surrogate fitted to random perturbations (Ribeiro 2016; W4-L02). Local fidelity is measured by R².
  - Both are post-hoc, local and model-agnostic (W1-L01 p8 taxonomy).
- **C. Evaluating explanations: faithfulness vs agreement** (1 paragraph)
  - Faithfulness vs plausibility (Jacovi & Goldberg 2020). Deletion-based metrics, comprehensiveness and sufficiency (DeYoung 2020, ERASER).
  - Explainers disagree (Krishna 2022). Explainers can be fooled (Slack 2020). Deleted-word inputs are out-of-distribution (Hase 2021). "Accuracy ≠ explainability" (Ngueajio 2025).
  - **Gap:** nobody has checked attribution faithfulness *for filter auditing* specifically, against references with known answers.
- Rudin's argument can go here or in the Discussion, but not both.
- **Draws on:** slides 2, 3, 5, 9; all source reviews in `journal/daily notes/`.

---

### III. Method (about 0.6 page)

Short, precise sub-sections. Someone should be able to rerun everything from this section plus the repo.

- **A. Filters**
  - Detoxify `original` (BERT, uncased) and `unbiased` (RoBERTa, cased).
  - The transparent filter from Exp 4: binary unigram+bigram counts + logistic regression, trained on Civil Comments. Give its test AUC next to Detoxify's.
  - Say up front that the two Detoxify models differ in architecture as well as debiasing.
- **B. Sentence sets**
  - Exp 1: 48 sentences, 16 topics × promote/report/counter.
  - Exp 2: 190 sentences, 10 templates × (15 identity + 4 control terms).
  - Exp 4 only: 200 real comments sampled from Civil Comments.
  - One example of each in the text, or a tiny table.
- **C. Explainers**
  - SHAP: Partition explainer, word-level tokens, explaining **log-odds**, `max_evals` = 500. The efficiency check passed on every run (W4-L01 p3–4).
  - LIME: `LimeTextExplainer`, `labels=(1,)` (sign trap, friday-01), 5000 samples unless stated, fixed `random_state`.
  - Removal baseline: what a "removed" word is replaced with. Default `"..."`; Exp 3 also tests deletion and the model's mask token.
- **D. References** (the core of the method; explain these carefully)
  - **Counterfactual effect:** logit(identity sentence) − mean logit(the 4 control sentences in the same template). Explain *why* this is a fair yardstick: only one word differs, so the whole score change is caused by that word.
  - **Exact Shapley values:** computed by brute-force enumeration of every word subset on the transparent filter (feasible because sentences are ≤ 16 words).
- **E. Metrics**
  - Faithfulness to the references: Spearman ρ, sign agreement, MAE (log-odds).
  - Deletion test: comprehensiveness, sufficiency, AOPC at k = 1–3.
  - Stability: top-word flip rate and mean pairwise Spearman across 10 seeds.
  - Signal strength: |logit − SHAP base value|.

---

### IV. Results (about 1 page, the biggest section)

**Figures and tables we have room for:** 2 figures + 1 table. Don't add more.
- **Fig. 1 (Exp 3):** scatter of SHAP's estimate vs the counterfactual effect, one dot per identity sentence, two panels (original / unbiased), y = x line.
- **Fig. 2 (Exp 6):** LIME instability vs signal strength, one dot per sentence, 500 vs 5000 samples.
- **Table I:** faithfulness summary. Rows = SHAP / LIME / random. Columns = ρ with the counterfactual effect (Exp 3), MAE vs exact Shapley (Exp 4), comprehensiveness / sufficiency (Exp 5), top-word flip rate (Exp 6).

Sub-sections:
- **A. What the explainers say** (short; 3–4 sentences recapping the presentation)
  - Exp 1: reports still flagged 7/16; SHAP puts about 87% of the toxic push on the insult words.
  - Exp 2: harmless identity sentences score far above control sentences; SHAP puts the decision on the identity word.
  - Then: "But are these explanations faithful?" This sets up B–E.
- **B. Faithfulness to the references** (Exp 3 + Exp 4; Fig. 1 + Table I)
  - How closely SHAP and LIME match the counterfactual effect.
  - How closely they match exact Shapley on the transparent filter, especially where bigrams create interactions.
  - **The "unbiased model leans on the group word more (46% → 60%)" result from the presentation:** say plainly whether it survives once we look at absolute SHAP values and the counterfactual effect, not just the share. If it doesn't survive, say so. That's critical analysis, not a failure.
- **C. Deletion test** (Exp 5; Table I)
  - Do SHAP's top words cause a bigger score drop than LIME's and than random words?
- **D. Where the explainers break** (Exp 6; Fig. 2)
  - Stability vs signal strength.
  - Whether low R² or SHAP–LIME disagreement predicts high error. This tests the presentation's own "next steps" recommendations.
- **E. Sensitivity to the removal baseline** (Exp 3's baseline check; 2–3 sentences, numbers in text)
  - How much the identity-term ranking moves when "removed" means deleted vs `"..."` vs `[MASK]`.
- **Rules:** every number must come from a results CSV in the repo. Give n for every statistic. Present results first; save interpretation for Section V.
- **Draws on:** slides 6–8 and 13; Exp 1–6 results.

---

### V. Discussion and Critical Analysis (about 0.4 page)

- **Triage vs certify:** what our results mean for an auditor in practice. An explainer that's reliable on strong signals and shaky on weak ones can flag bias, but "no bias found" can't be trusted. Use the smoke-alarm vs fire-certificate idea, in academic wording.
- **Agreement ≠ faithfulness:** compare the presentation's 78/88 agreement figure with what the references show (Krishna 2022).
- **The removal baseline isn't neutral:** deleted or masked sentences are unnatural inputs (Hase 2021; textbook p48 "choosing a good baseline can be difficult"). Our baseline-check numbers make this concrete.
- **Descriptive, not normative:** SHAP shows *which words* drive the score, not whether the filter's idea of "toxic" is right.
- **Not causal:** attribution describes the model, not harm in the world (textbook p53).
- **Can be fooled:** Slack 2020. Our references only work because we built them ourselves. A real auditor facing a black box doesn't have them.
- **Rudin's debate (W1-L01 p10), answered with Exp 4:** how big is the accuracy gap between the transparent filter and Detoxify? If the gap is small, Rudin's "use interpretable models instead" applies to filters. If it's big, post-hoc explanation is necessary but needs the checks we propose.
- **Draws on:** slide 9; Krishna, Slack, Hase, Rudin, Ngueajio; textbook pp. 48 and 53.

---

### VI. Conclusion and Limitations (about 0.25 page)

The rubric rewards honesty about scope (10%), so be explicit.

- **Conclusion:** restate the thesis in one or two sentences, plus 2–3 practical recommendations for auditors:
  - report LIME's R² and run it several times;
  - use SHAP/LIME disagreement as a flag for human review, but only if Exp 6 supports this;
  - check a counterfactual swap before trusting any attribution.
- **Limitations** (list them; it's fine for this to be a dense paragraph):
  - Template sentences, English only, single-word identity terms.
  - One filter family (Detoxify), unless Exp 7 is done.
  - `original` vs `unbiased` also differ in architecture (BERT vs RoBERTa).
  - Exp 4's filter is ours, not a real pipeline filter.
  - Only lexical bias: dialect bias like AAE (Sap 2019) isn't a single word, so word attribution can't see it.
  - Our references only exist because we controlled the inputs.
  - We didn't test quality filters or blocklists.
- **Future work:** real web text (C4), the quality filter, human rationales (HateXplain).

---

### References Identified so far (page 4)

- IEEE numbered style, generated by BibTeX in Overleaf.:
  - **Course:** Rudin 2019; Doshi-Velez & Kim 2017; Munn & Pitman 2022 (the textbook); Molnar *Interpretable ML*; Ribeiro 2016; Lundberg & Lee 2017.
  - **Ours (already reviewed):** Dodge 2021, Gururangan 2022, Sap 2019, Mendu 2025, Udagawa 2025, Dixon 2018, Krishna 2022, Slack 2020, Ngueajio 2025.
  - **New (need source reviews):** DeYoung 2020, Jacovi & Goldberg 2020, Garg 2019, Hase 2021, Borkan 2019.

---
