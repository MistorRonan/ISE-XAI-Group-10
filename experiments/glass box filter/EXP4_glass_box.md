# Experiment 4: Glass-Box Filter: Can SHAP and LIME recover explanations we know are correct?

**Feeds paper sections:** III-A (filters), III-D (references), IV-B (faithfulness, Table I), V (Discussion: Rudin's debate)
**Depends on:** nothing. It can run in parallel with Exp 3.

---

## Part 1: What is this experiment?

With Detoxify we can never be completely sure what the "right" explanation is, because it's a black box. So in this experiment we **build our own toxicity filter whose explanation we know exactly**, and check whether SHAP and LIME can find it.

The filter is a **logistic regression on word and word-pair counts**, trained on real comments labelled toxic or not (Civil Comments). It's *transparent*: every word and word pair has a learned weight (coefficient), and the score is just the sum of the weights of the words present (W2-L01 p4). Because it's this simple, and our sentences are short, we can compute the **exact Shapley values** by brute force: try every possible subset of words and see what the filter scores. No approximation, no randomness. That's our second **reference** (the first is Exp 3's counterfactual effect).

Then we pretend we can't see inside the filter. We run SHAP and LIME on it exactly the way we did on Detoxify, and check: **do they recover the exact values?**

> **Why word pairs (bigrams)?** If the filter only used single words, every word would add a fixed amount and SHAP would get it right trivially. Word pairs like "disgusting idiots" create **interactions**: the pair adds extra score only when both words are present. That's exactly the lecture question in W2-L01 p4 ("what happens to this story once you add an interaction?"), and it's where approximate explainers can go wrong.

Two extra results come out of the same notebook for free:
1. **Rudin's trade-off, measured (W1/W2 debate).** How accurate is our transparent filter compared with Detoxify on the same held-out comments? If the gap is small, Rudin's "just use an interpretable model" argument applies to data filters.
2. **Does the transparent filter also learn identity-term bias?** We just read its coefficients for "gay", "muslim", "black"… No explainer needed, which is Rudin's point.

---

## Part 2: Why it matters for the paper

- It's the **second reference** for faithfulness, and the only one where the correct answer is known *exactly*.
- It tests the lecture's explainer table directly (W4-L01 p5; textbook Table 3-4): exact vs approximate explainers. Our text SHAP uses the **Partition** explainer, which is an approximation. Here we find out how good the approximation is.
- It supplies the Discussion's answer to **Rudin's debate** with our own numbers instead of only citing it.
- It's the only experiment that uses some **real text** (200 real comments) as well as our templates.

---

## Part 3: What it builds on

### Our research (from the journal)
- **Gururangan et al. (2022):** a real pipeline quality filter that is also a linear model, so transparent filters are realistic, not a toy. This is the old "idea C" in the 02/10 recap.
- **Rudin (2019):** "use interpretable models instead" for high-stakes decisions.
- **Exp 1 and Exp 2:** we reuse their 238 sentences.

### Method precedent (add source reviews for the new ones)
- **Lundberg & Lee (2017):** SHAP as an approximation of Shapley values.
- **Borkan et al. (2019):** the Civil Comments / Jigsaw Unintended Bias dataset we train on.
- **Ribeiro et al. (2016):** LIME.

### Lecture content
| Where | What we use |
|---|---|
| **W2-L01 p4** | LR coefficients are log-odds explanations; the interaction caveat is why we add bigrams. |
| **W2-L01 p9 (activity)** | "Compare accuracy: is there actually a gap, or does this confirm Rudin's argument?", the same test on our filter. |
| **W1-L01 p6, p10** | The accuracy–interpretability trade-off is "more real for raw text". We measure whether it is. |
| **W4-L01 p5 / textbook Table 3-4** | Exact vs approximate explainers. |
| **Textbook p48** | Efficiency and null-player axioms, checked in code. SHAP ≠ exact Shapley when sampled. |
| **shap-exercises notebook** | Explain the log-odds output: a logistic regression is only additive in log-odds. |
| **friday-03 notebook** | LIME on a scikit-learn text pipeline: the same pattern. |

---

## Part 4: How it fits the thesis

| Result | What it means for the thesis |
|---|---|
| SHAP ≈ exact Shapley on sentences without active bigrams | Expected; a sanity check, not a finding |
| SHAP still close when bigrams interact | Supports "faithful when the signal is strong" |
| LIME's ranking wobbles across seeds or misses the top word | Supports "stability drops", and links to Exp 6 |
| Small AUC gap between the LR and Detoxify | Strengthens Rudin's argument in the Discussion |
| Identity words get positive coefficients | A transparent filter shows bias directly, with no explainer needed |

---

## Part 5: Instructions for the experiment

### TASK
Create a Jupyter notebook at `experiments/glass box filter/04_glass_box.ipynb` for **Google Colab (T4 GPU)**. It must contain **exactly** the cells below, in order, each preceded by a markdown cell naming the step. Do not change seeds or file names.

### Cell 0 (markdown)
Title "Experiment 4 – Glass-Box Filter", then: "Goal: train a transparent toxicity filter (logistic regression on word and word-pair counts), compute exact Shapley values for its decisions by enumerating every subset of words, and test whether SHAP and LIME recover them. Also compare the filter's accuracy with Detoxify (Rudin's trade-off) and read its identity-word coefficients. Runtime: Colab → T4 GPU."

### Cell 1: install, imports and shared helpers
`!pip -q install datasets` and then copy **EXP1 Cell 1 exactly** (we still need Detoxify for the accuracy comparison), with `OUT = "outputs_exp4"`. Then append:
```python
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from math import factorial
DELETE_MASKER = shap.maskers.Text(word_tokenizer, mask_token="")   # 'removed' = deleted, matching the exact game below
```

### Cell 2: data (Civil Comments)
100,000 training comments and 5,000 test comments, sampled with the fixed seed. "Toxic" = toxicity ≥ 0.5.
```python
ds = load_dataset("google/civil_comments")
train = ds["train"].to_pandas().sample(100_000, random_state=SEED)
test  = ds["test"].to_pandas().sample(5_000, random_state=SEED)
for d in (train, test): d["label"] = (d.toxicity >= 0.5).astype(int)
print(train.label.mean(), test.label.mean())
```

### Cell 3: train the transparent filters
`LR2` (words + word pairs) is the main filter. `LR1` (single words only) is the sanity-check filter.
```python
def fit(ngrams):
    vec = CountVectorizer(binary=True, ngram_range=ngrams, min_df=5, token_pattern=r"(?u)\b\w+\b")
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(vec.fit_transform(train.text), train.label)
    return vec, clf
VEC1, LR1 = fit((1, 1)); VEC2, LR2 = fit((1, 2))
def gb_logit(texts, vec=None, clf=None):
    vec, clf = vec or VEC2, clf or LR2
    return clf.decision_function(vec.transform([str(t) for t in texts]))   # log-odds, like Detoxify's logit()
def gb_proba(texts):
    p = 1 / (1 + np.exp(-gb_logit(texts))); return np.column_stack([1 - p, p])
gb1 = lambda t: gb_logit(t, VEC1, LR1)
```

### Cell 4: Rudin's trade-off: accuracy of the transparent filter vs Detoxify
```python
auc = {"LR words": roc_auc_score(test.label, gb1(test.text)),
       "LR words+pairs": roc_auc_score(test.label, gb_logit(test.text))}
for m in MODELS: auc[f"Detoxify {m}"] = roc_auc_score(test.label, proba(test.text, m))
auc = pd.Series(auc); print(auc.round(3)); auc.to_csv(f"{OUT}/exp4_auc.csv")
```
Markdown note after it: *"Caveat: Detoxify `unbiased` was trained on this dataset (the Jigsaw Unintended Bias competition), so its AUC here is optimistic."*

### Cell 5: does the transparent filter learn identity bias? (read the coefficients)
```python
TERMS = ["gay","lesbian","transgender","black","white","asian","muslim","jewish",
         "christian","immigrant","deaf","blind","disabled","poor","rural"]
CONTROLS = ["tall","young","local","quiet"]
voc = VEC2.vocabulary_
coefs = pd.DataFrame([(w, w in CONTROLS, LR2.coef_[0][voc[w]] if w in voc else np.nan) for w in TERMS + CONTROLS],
                     columns=["word","is_control","coef_log_odds"]).sort_values("coef_log_odds", ascending=False)
print(coefs.round(3)); coefs.to_csv(f"{OUT}/exp4_identity_coefs.csv", index=False)
```

### Cell 6: the sentences to explain
The 238 template sentences (Exp 1's 48 from `experiments/report vs promote/data/exp1_sentences.csv` + Exp 2's 190, rebuilt with EXP2 Cell 2's code), plus 200 real test comments with 5–14 words (100 toxic, 100 non-toxic).
```python
exp1 = pd.read_csv("exp1_sentences.csv")             # upload from experiments/report vs promote/data/
TEMPLATES = ["I am a {} person.", "My neighbour is {} and she is lovely.",
             "{} people are welcome here.", "My best friend is {}.",
             "I grew up in a {} family.", "Being {} is completely normal.",
             "We hired a {} engineer last week.", "She is a proud {} woman.",
             "He is {} and works as a teacher.", "Our new {} classmate joined the team."]
exp2 = [t.format(w) for t in TEMPLATES for w in TERMS + CONTROLS]
nw = test.text.str.findall(r"\w+").str.len()
real = pd.concat([test[(nw.between(5, 14)) & (test.label == y)].sample(100, random_state=SEED) for y in (0, 1)])
S = pd.DataFrame({"text": list(exp1.text) + exp2 + list(real.text),
                  "source": ["exp1"] * len(exp1) + ["exp2"] * len(exp2) + ["real"] * len(real)})
S["n_tokens"] = [len(word_tokenizer(t)["input_ids"]) for t in S.text]
S = S[S.n_tokens <= 16].reset_index(drop=True); print(S.source.value_counts())
S.to_csv(f"{OUT}/exp4_sentences.csv", index=False)
```

### Cell 7: exact Shapley values (the reference)
Players = the same word tokens SHAP uses. Value of a subset = the filter's log-odds for the sentence with every other word **deleted**. Every subset is tried, and the Shapley formula is applied exactly (textbook p48).
```python
def exact_shapley(text, f):
    toks = word_tokenizer(text)["input_ids"]; n = len(toks); N = 1 << n
    ids = np.arange(N); bits = (ids[:, None] >> np.arange(n)) & 1
    v = f(["".join(t for t, b in zip(toks, row) if b) for row in bits])
    size = bits.sum(1)
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.array([np.sum(w[size[S0]] * (v[S0 | (1 << i)] - v[S0]))
                    for i in range(n) for S0 in [ids[bits[:, i] == 0]]])
    assert abs(phi.sum() - (v[-1] - v[0])) < 1e-6          # efficiency axiom holds exactly
    return toks, phi
EXACT2 = [exact_shapley(t, gb_logit) for t in S.text]
EXACT1 = [exact_shapley(t, gb1) for t in S.text]
```

### Cell 8: approximate explainers on the same filter
- **SHAP Partition** (what Exp 1–3 used on Detoxify) and **SHAP Permutation** (aims at true Shapley values), both with the deletion masker and `max_evals=500`.
- **LIME**: 10 seeds, 5000 samples. LIME works in probability, so only rankings are compared.
```python
def by_word(toks, vals):
    d = {}
    for t, v in zip(toks, vals):
        k = re.sub(r"\W+", "", t).lower()
        if k: d[k] = d.get(k, 0.0) + float(v)
    return d
part2 = shap.Explainer(gb_logit, DELETE_MASKER, algorithm="partition")(list(S.text), max_evals=500)
perm2 = shap.Explainer(gb_logit, DELETE_MASKER, algorithm="permutation")(list(S.text), max_evals=500)
part1 = shap.Explainer(gb1, DELETE_MASKER, algorithm="partition")(list(S.text), max_evals=500)
rows = []
for i, text in enumerate(S.text):
    ex = by_word(*EXACT2[i])
    for name, e in [("SHAP partition", part2[i]), ("SHAP permutation", perm2[i])]:
        ap = by_word(e.data, e.values); a = agreement(ex, ap)
        common = [k for k in ex if k in ap]
        rows.append(dict(i=i, explainer=name, seed=np.nan, rho=a["rho"], top1=a["top1_agree"],
                         mae=np.mean([abs(ex[k] - ap[k]) for k in common])))
    for seed in range(10):
        expl = LimeTextExplainer(class_names=["non-toxic","toxic"], random_state=seed)
        le = expl.explain_instance(text, gb_proba, num_features=len(ex), num_samples=5000, labels=(1,))
        lw = {w.lower(): v for w, v in le.as_list(label=1)}; a = agreement(ex, lw)
        rows.append(dict(i=i, explainer="LIME", seed=seed, rho=a["rho"], top1=a["top1_agree"],
                         mae=np.nan, r2=le.score))
R = pd.DataFrame(rows).merge(S[["source"]], left_on="i", right_index=True)
R.to_csv(f"{OUT}/exp4_faithfulness_by_sentence.csv", index=False)
```

### Cell 9: where do the explainers miss? (interactions)
For each sentence, measure how much interaction the bigrams create: the difference between the bigram filter's exact Shapley values and the single-word filter's. Then compare faithfulness on low- vs high-interaction sentences.
```python
S["interaction"] = [np.abs(EXACT2[i][1] - EXACT1[i][1]).sum() for i in range(len(S))]
S["high_interaction"] = S.interaction > S.interaction.median()
R = R.merge(S[["high_interaction"]], left_on="i", right_index=True)
summary = R.groupby(["explainer","high_interaction"]).agg(rho=("rho","median"), top1=("top1","mean"),
                                                       mae=("mae","mean")).round(3)
print(summary); summary.to_csv(f"{OUT}/exp4_summary.csv")
```

### Cell 10: sanity check and null-player axiom
1. On the **single-word** filter, SHAP Partition should match exact Shapley almost perfectly.
2. **Null player** (textbook p48): words the filter has no weight for must get exactly 0. Check whether the approximate explainers also give them about 0.
```python
mae1 = np.mean([np.mean(np.abs(np.array([by_word(part1[i].data, part1[i].values).get(k, 0) for k in by_word(*EXACT1[i])])
                               - np.array(list(by_word(*EXACT1[i]).values())))) for i in range(len(S))])
print(f"Single-word filter: SHAP partition vs exact MAE = {mae1:.2e} log-odds (expect ~0)")
null = []
for i in range(len(S)):
    ex = by_word(*EXACT2[i]); ap = by_word(part2[i].data, part2[i].values)
    null += [abs(ap.get(k, 0)) for k, v in ex.items() if abs(v) < 1e-9]
print(f"Null-player words: {len(null)}, mean |SHAP partition| = {np.mean(null) if null else float('nan'):.2e}")
```

### Cell 11: results summary (numbers to quote in the paper)
```python
print("=== EXPERIMENT 4 SUMMARY ===")
print("AUC:"); print(auc.round(3))
print("Identity words with positive coefficients:", int((coefs[~coefs.is_control].coef_log_odds > 0).sum()), "/ 15")
g = R.groupby("explainer")
print(g.agg(median_rho=("rho","median"), top1=("top1","mean"), mae=("mae","mean")).round(3))
lime_flip = R[R.explainer=="LIME"].groupby("i").top1.apply(lambda s: s.nunique() > 1).mean()
print(f"LIME: share of sentences where top-word agreement with exact changes across seeds: {lime_flip:.0%}")
print(summary)
```

### Cell 12: download everything
```python
!zip -qr outputs_exp4.zip outputs_exp4
from google.colab import files; files.download("outputs_exp4.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook **and** `exp1_sentences.csv` to Colab; select a T4 GPU.
2. Run all cells. Expect about 30–60 minutes (Cell 4's Detoxify scoring and Cell 8's LIME runs take longest).
3. Unzip the outputs into `experiments/glass box filter/results/`.
4. Commit under their own GitHub account: `research: experiment 4 glass-box filter results`.

---

## Part 6: Checks before you trust the results
- [ ] Cell 7's `assert` never fires: efficiency holds exactly for the reference.
- [ ] Cell 10: the single-word filter's SHAP vs exact MAE is about 0. **If it isn't, the comparison setup is wrong** (usually the masker or the tokens don't match). Stop and fix it before reading any other result.
- [ ] Cell 4: both LR filters have AUC well above 0.5 (we'd expect around 0.85–0.9). A weak filter makes the whole experiment less convincing.
- [ ] The LIME number of features equals the number of distinct words, so nothing is cut off.
- [ ] Copy the Cell 11 summary into the journal entry.

## Part 7: What goes in the paper
- **III-A:** one sentence describing the filter, plus its AUC next to Detoxify's.
- **Table I, column "vs exact Shapley":** median ρ / MAE for SHAP Partition and LIME (and SHAP Permutation if there's room).
- **IV-B text:** *"On a transparent filter, SHAP's Partition approximation recovers exact Shapley values with median ρ = [X] (MAE [Y] log-odds), but agreement falls on high-interaction sentences ([Z]). LIME's top word matches the exact one in [W]% of runs."*
- **V (Rudin):** *"The transparent filter reaches AUC [A] vs Detoxify's [B] on the same comments, and its coefficients show identity-term bias directly ([N]/15 identity words have positive weights)."*

## Part 8: Limitations to state
- It's our own filter, not one used in a real pipeline (though Gururangan's quality filter is also linear).
- Exact Shapley depends on the removal choice (we delete words). With a different baseline, the "exact" answer would be different. That's the same point Exp 3's baseline check makes.
- Detoxify `unbiased` was trained on Civil Comments, so the AUC comparison favours it.
- Sentences are capped at 16 tokens, because exact enumeration doubles in cost with every extra word.
- SHAP's Partition explainer computes Owen values (a grouped version of Shapley), so a small mismatch with exact Shapley under interactions is expected by design. Say this when reporting the numbers.
