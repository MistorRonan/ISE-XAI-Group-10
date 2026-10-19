# Experiment 5: Deletion Test: If we remove the words the explainer blames, does the score actually drop?

---

## Part 1: What is this experiment?

A simple common-sense test of whether an explanation is faithful. **If the explainer says "these words caused the toxic score", then deleting those words should make the score drop a lot.** And keeping *only* those words should keep most of the score.

For every sentence, we take the top 1, 2 and 3 words that SHAP blames, the top words that LIME blames, and the same number of **random** words, and we measure two things:

| Measure | What we do | What a faithful explainer gets |
|---|---|---|
| **Comprehensiveness** | Delete the top-k words; measure how much the toxic score drops | **High** (the important words really were important) |
| **Sufficiency** | Keep *only* the top-k words; measure how much the score drops | **Low** (those words alone are enough to keep the score) |

> **Example:** "Those fans are disgusting idiots and should be banned." SHAP blames *idiots* and *disgusting*. Delete them → "Those fans are and should be banned." If the score falls from very toxic to almost clean, comprehensiveness is high and SHAP was right. If random words give the same drop, the explanation isn't telling us anything.

The **random** words are the baseline. An explainer is only useful if it beats random.

This is the standard faithfulness test from the ERASER benchmark (DeYoung et al. 2020). It's also the same "perturb it and measure the change" idea as permutation importance in the textbook (pp. 41–42), applied to words.

---

## Part 2: Why it matters for the paper

- It's a **third, independent faithfulness check** that works on any sentence. Exp 3's reference needs minimal pairs, and Exp 4's needs a transparent filter. This one works on the Exp 1 report/counter sentences too, which Exp 3 can't cover.
- It lets us **compare SHAP vs LIME vs random** on one fair scale (Table I).
- It feeds the Discussion's point that **deleting words creates unnatural sentences** (Hase 2021). The test itself has that problem, and we say so honestly.

---

## Part 3: What it builds on

### Our research (from the journal)
- **Ngueajio et al. (2025):** the survey that pointed us to ERASER's faithfulness metrics (comprehensiveness and sufficiency).
- **Exp 1 and Exp 2:** we reuse their 238 sentences.

### Method precedent (add source reviews for the new ones)
- **DeYoung et al. (2020), *ERASER: A Benchmark to Evaluate Rationalized NLP Models*:** defines comprehensiveness, sufficiency and AOPC.
- **Jacovi & Goldberg (2020), *Towards Faithfully Interpretable NLP Systems*:** faithfulness vs plausibility.
- **Hase et al. (2021):** deleted-word inputs are out-of-distribution.

### Lecture content
| Where | What we use |
|---|---|
| **Textbook pp. 41–42** | Perturbation-based attribution: perturb features, measure the change in prediction. That's what this test does. |
| **Textbook p48** | "Choosing a good baseline can be difficult": deletion is one baseline choice. |
| **W4-L01 p2** | SHAP should "catch a model leaning on the wrong signal". This checks it really does. |
| **shap-exercises notebook** | Measure score changes in log-odds. |

---

## Part 4: How it fits the thesis

| Result | What it means for the thesis |
|---|---|
| SHAP and LIME clearly beat random on comprehensiveness | Supports "faithful": the blamed words really drive the score |
| The gap over random shrinks for weak-signal sentences (control words, low-scoring sentences) | Supports "faithfulness drops when the signal is weak" |
| SHAP beats LIME, or the reverse | Tells the auditor which tool to prefer |
| Neither beats random | Major finding against the explainers; tell the group |

---

## Part 5: Instructions for the experiment

### TASK
Create a Jupyter notebook at `experiments/deletion test/05_deletion.ipynb` for **Google Colab (T4 GPU)**, with **exactly** the cells below, each preceded by a markdown cell naming the step. Do not change seeds or file names.

### Cell 0 (markdown)
Title "Experiment 5 – Deletion Test", then: "Goal: test whether the words SHAP and LIME blame for a toxic score actually drive it, by deleting them (comprehensiveness) or keeping only them (sufficiency), compared with random words. Runtime: Colab → T4 GPU."

### Cell 1: install, imports and shared helpers
Copy **EXP1 Cell 1 exactly**, with `OUT = "outputs_exp5"`. Then append:
```python
KS = [1, 2, 3]
N_RANDOM = 10
def toks_of(text): return word_tokenizer(text)["input_ids"]
def key(tok): return re.sub(r"\W+", "", tok).lower()
def ranking(scores):
    """Token positions ordered from most to least 'toward toxic'."""
    return list(np.argsort(-np.asarray(scores, float), kind="stable"))
def drop(toks, keep_idx):
    return "".join(t for j, t in enumerate(toks) if j in keep_idx).strip()
```

### Colour scheme (append to the end of Cell 1)
All figures use the **same pink-and-black theme as the Exp 1 and Exp 2 figures**. Paste the colour-scheme block from [EXP3 Part 5](../counterfactual%20swap/EXP3_counterfactual_swap.md) ("Colour scheme") at the end of Cell 1. It overrides the old blue/orange colours in EXP1's Cell 1. SHAP = `COLORS["SHAP"]` (dark pink), LIME = `COLORS["LIME"]` (light pink), random = `COLORS["random"]` (grey).

### Cell 2: the sentences
The 48 Exp 1 sentences (upload `experiments/report vs promote/data/exp1_sentences.csv`) and the 190 Exp 2 sentences (rebuild with EXP2 Cell 2's code).
```python
exp1 = pd.read_csv("exp1_sentences.csv")
TEMPLATES = ["I am a {} person.", "My neighbour is {} and she is lovely.",
             "{} people are welcome here.", "My best friend is {}.",
             "I grew up in a {} family.", "Being {} is completely normal.",
             "We hired a {} engineer last week.", "She is a proud {} woman.",
             "He is {} and works as a teacher.", "Our new {} classmate joined the team."]
TERMS = ["gay","lesbian","transgender","black","white","asian","muslim","jewish",
         "christian","immigrant","deaf","blind","disabled","poor","rural"]
CONTROLS = ["tall","young","local","quiet"]
exp2 = pd.DataFrame([(t.format(w), "control" if w in CONTROLS else "identity") for t in TEMPLATES for w in TERMS + CONTROLS],
                    columns=["text", "group"])
S = pd.concat([pd.DataFrame({"text": exp1.text, "group": exp1.condition, "source": "exp1"}),
               exp2.assign(source="exp2")], ignore_index=True)
print(S.groupby(["source","group"]).size())          # expect 16/16/16 and 150/40
```

### Cell 3: attributions per token (SHAP and LIME, both models)
SHAP gives a value per token directly. LIME gives a value per word, which we map back onto the tokens.
```python
ATTR = {}
for m in MODELS:
    sv = shap_explainer(m)(list(S.text), max_evals=500, batch_size=32)
    check_efficiency(sv, list(S.text), m)
    ATTR[(m, "SHAP")] = [list(sv[i].values) for i in range(len(S))]
    lime_scores = []
    for text in S.text:
        lw, _ = lime_word_weights(text, m, seed=SEED, num_samples=5000)
        lime_scores.append([lw.get(key(t), 0.0) for t in toks_of(text)])
    ATTR[(m, "LIME")] = lime_scores
    # sanity: SHAP's tokens are the same as word_tokenizer's
    assert all(len(ATTR[(m, "SHAP")][i]) == len(toks_of(S.text[i])) for i in range(len(S)))
```

### Cell 4: comprehensiveness and sufficiency
Every perturbed sentence is collected first and then scored in one batch per model (much faster).
```python
rng = np.random.RandomState(SEED)
rows = []
for m in MODELS:
    base = logit(S.text, m)
    batch, meta = [], []
    for i, text in enumerate(S.text):
        toks = toks_of(text); n = len(toks)
        for k in KS:
            if k >= n: continue
            sets = {"SHAP": ranking(ATTR[(m, "SHAP")][i])[:k], "LIME": ranking(ATTR[(m, "LIME")][i])[:k]}
            for r in range(N_RANDOM): sets[f"random{r}"] = list(rng.choice(n, k, replace=False))
            for name, top in sets.items():
                top = set(top)
                batch += [drop(toks, set(range(n)) - top), drop(toks, top)]
                meta.append((i, k, name))
    scores = logit(batch, m).reshape(-1, 2)          # column 0 = top-k removed, column 1 = only top-k kept
    for (i, k, name), (removed, kept) in zip(meta, scores):
        rows.append(dict(model=m, i=i, k=k, explainer="random" if name.startswith("random") else name,
                         comp=base[i] - removed, suff=base[i] - kept))
D = pd.DataFrame(rows).groupby(["model","i","k","explainer"], as_index=False)[["comp","suff"]].mean()  # averages the 10 random draws
D = D.merge(S, left_on="i", right_index=True)
D.to_csv(f"{OUT}/exp5_deletion_by_sentence.csv", index=False)
```

### Cell 5: AOPC summary (Table I numbers)
AOPC = the average over k = 1, 2, 3. Higher comprehensiveness is better; lower sufficiency is better.
```python
aopc = D.groupby(["model","explainer"])[["comp","suff"]].mean().round(3)
print(aopc); aopc.to_csv(f"{OUT}/exp5_aopc.csv")
by_group = D.groupby(["model","source","group","explainer"])[["comp","suff"]].mean().round(3)
print(by_group); by_group.to_csv(f"{OUT}/exp5_aopc_by_group.csv")
```

### Cell 6: does SHAP / LIME beat random in each sentence?
```python
piv = D.pivot_table(index=["model","i","k"], columns="explainer", values="comp").reset_index()
for m in MODELS:
    p = piv[piv.model == m]
    print(f"[{m}] SHAP beats random on comprehensiveness in {(p.SHAP > p.random).mean():.0%} of cases; "
          f"LIME in {(p.LIME > p.random).mean():.0%}; SHAP beats LIME in {(p.SHAP > p.LIME).mean():.0%}")
```

### Cell 7: backup figure (not planned for the paper; for the journal and questions)
Comprehensiveness by k for SHAP, LIME and random (original model), one line each.
```python
fig, ax = plt.subplots(figsize=(6, 4))
for name, g in D[D.model == "original"].groupby("explainer"):
    s = g.groupby("k").comp.mean(); ax.plot(s.index, s.values, marker="o", label=name, color=COLORS[name])
ax.set_xlabel("k (words removed)"); ax.set_ylabel("Comprehensiveness (log-odds drop)")
ax.set_xticks(KS); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp5_comprehensiveness.png", dpi=300)
```

### Cell 8: results summary
```python
print("=== EXPERIMENT 5 SUMMARY ===")
print(aopc)
print("Biggest SHAP-over-random gap by group (original):")
g = by_group.xs("original", level="model")["comp"].unstack("explainer")
print((g["SHAP"] - g["random"]).sort_values(ascending=False).round(3))
```

### Cell 9: download everything
```python
!zip -qr outputs_exp5.zip outputs_exp5
from google.colab import files; files.download("outputs_exp5.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook **and** `exp1_sentences.csv` to Colab; select a T4 GPU.
2. Run all cells. Expect about 30–45 minutes (mostly Cell 3's LIME runs).
3. Unzip the outputs into `experiments/deletion test/results/`.
4. Commit under their own GitHub account: `research: experiment 5 deletion test results`.

---

## Part 6: Checks before you trust the results
- [ ] Every `check_efficiency` printed **"efficiency check passed"**.
- [ ] The Cell 3 assert passed (SHAP's tokens line up with ours).
- [ ] Print 3 perturbed sentences from Cell 4 and read them. Do the deletions look right?
- [ ] Random comprehensiveness should be small and similar across groups. If random alone causes big drops, the deletion itself is moving the score (the out-of-distribution problem). Note it.
- [ ] Copy the Cell 8 summary into the journal entry.

## Part 7: What goes in the paper
- **Table I, columns "Comprehensiveness ↑" and "Sufficiency ↓":** AOPC for SHAP / LIME / random (original model; unbiased in text).
- **IV-C text:** *"Deleting SHAP's top-3 words lowers the toxic log-odds by [X] on average, vs [Y] for LIME and [Z] for random words. The gap over random is largest for [group] and smallest for [weak group]."*
- **V (Discussion):** one sentence saying that deletion creates unnatural sentences, so this test has the same baseline problem as Exp 3's baseline check.

## Part 8: Limitations to state
- Deleted-word sentences are unnatural, so part of the score drop may come from the sentence being broken rather than from the word's meaning (Hase 2021).
- We use k = 1–3 words, not ERASER's percentage bins, because our sentences are short.
- One LIME run per sentence (seed 42). Exp 6 shows how much LIME varies between runs.
- SHAP's attributions were computed with the `"..."` baseline, but the test deletes words. Write this mismatch down, and check whether the ranking changes much using Exp 3's baseline results.
