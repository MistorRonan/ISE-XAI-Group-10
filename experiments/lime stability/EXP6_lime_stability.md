# Experiment 6: LIME Stability vs Signal Strength: Are the explainers least reliable exactly where an auditor needs them?

**Feeds paper sections:** III-E (metrics), IV-D (where the explainers break, Fig. 2, Table I), V (Discussion: triage vs certify), VI (recommendations)
**Depends on:** **Exp 3**. Its `exp3_results.csv` is needed for Cell 6. Everything else can run before Exp 3 is finished.

---

## Part 1: What is this experiment?

LIME works by randomly deleting words thousands of times and fitting a simple model to the results. Because it's random, **running LIME twice on the same sentence can give different answers** (W5-L02 p2). The presentation showed this on 7 hand-picked "hard cases": the top word changed in 4/7 at 500 samples, and LIME's own fit score (R²) fell to 0.14. But 7 hand-picked sentences are an anecdote, not a finding.

This experiment does it properly on **all 238 sentences**: LIME is run 10 times per sentence (seeds 0–9) at two sample sizes (500 and 5000), and we measure how much the answer moves. Then we ask:

1. **Is LIME less stable when the signal is weak?** Signal strength = how far the sentence's score is from the "everything removed" score (|logit − SHAP base value|). Strong signal = one word clearly drives the score, e.g. "Being gay is completely normal". Weak signal = nothing stands out, e.g. "We hired a young engineer last week".
2. **Can an auditor tell when not to trust the explanation?** The presentation recommended *"report LIME's R²"* and *"use SHAP/LIME disagreement as a flag for human review."* We test those recommendations: **do low R² or SHAP–LIME disagreement actually predict that the explanation is wrong** (a large error against Exp 3's counterfactual effect)?

> **Analogy:** asking a witness the same question 10 times. If they give the same answer every time, you can trust it more. If the answer changes, especially when the event was hard to see, you shouldn't rely on them. We want to know whether there's a warning sign *before* you ask 10 times.

---

## Part 2: Why it matters for the paper

- It's the main evidence for **"stability drops when the signal is weak"** in the thesis.
- It turns slide 8's 7 hand-picked cases into a **systematic relationship** across every sentence. The rubric rewards depth, and this is the difference between an anecdote and a result.
- It **tests our own recommendations from the presentation**. If R² doesn't predict errors, we have to drop that recommendation from the paper's conclusion, and that's a good critical-analysis point either way.
- It supports **"triage, not certify"**. If the explainers are least reliable on weak signals, then "the explainer found nothing" can't be read as "there is nothing".

---

## Part 3: What it builds on

### Our research (from the journal)
- **Exp 2's LIME stability test (slide 8, slide 13 backup chart):** the same idea, 5–7 sentences. Now scaled up.
- **Krishna et al. (2022):** explainers disagree; the rank-correlation and top-k agreement measures come from here.
- **Exp 3:** supplies the counterfactual error used to test whether R² and disagreement are useful warnings.

### Method precedent
- **Ribeiro et al. (2016):** LIME and its local fidelity score.
- **Krishna et al. (2022):** agreement metrics.

### Lecture content
| Where | What we use |
|---|---|
| **W5-L02 p2** | Without `random_state`, each `explain_instance` call draws new samples (5000 by default) and fits a slightly different surrogate. This experiment measures how different. |
| **W4-L02 p3** | Local fidelity: LIME's R² says how well the surrogate fits *near this sentence*. We test whether it's a usable warning. |
| **Textbook pp. 45–46** | Permutation importance: "different shuffles may produce different results… multiple runs are necessary", with n_repeats 5–30. The same reasoning applied to LIME's random sampling. |
| **friday-01 notebook** | `labels=(1,)` so weights refer to the toxic class. |

---

## Part 4: How it fits the thesis

| Result | What it means for the thesis |
|---|---|
| Instability rises as signal strength falls | Supports "stability drops when the signal is weak" |
| 5000 samples is much more stable than 500 | Practical recommendation: always use ≥ 5000 (LIME's default) |
| Low R² / SHAP–LIME disagreement predicts large Exp 3 error | The presentation's recommendations hold. Keep them in the Conclusion |
| They don't predict error | Drop or soften those recommendations, and say so. An auditor has no built-in warning sign, which strengthens "cannot certify" |

---

## Part 5: Instructions for the experiment

### TASK
Create a Jupyter notebook at `experiments/lime stability/06_lime_stability.ipynb` for **Google Colab (T4 GPU)**, with **exactly** the cells below, each preceded by a markdown cell naming the step. Do not change seeds or file names. **This is the longest-running notebook, so it saves progress as it goes and can be resumed if Colab disconnects.**

### Cell 0 (markdown)
Title "Experiment 6 – LIME Stability vs Signal Strength", then: "Goal: measure how much LIME's explanation changes between runs (10 seeds × 500 and 5000 samples) on all 238 Exp 1 + Exp 2 sentences, relate it to signal strength, and test whether LIME's R² or SHAP–LIME disagreement predicts when an explanation is wrong (using Exp 3's counterfactual error). Runtime: Colab → T4 GPU. Expect about 2–3 hours. Progress is saved after every sentence."

### Cell 1: install, imports and shared helpers
Copy **EXP1 Cell 1 exactly**, with `OUT = "outputs_exp6"`. Then append:
```python
from itertools import combinations
from sklearn.metrics import roc_auc_score
SEEDS = list(range(10)); SAMPLE_SIZES = [500, 5000]
CKPT = f"{OUT}/exp6_lime_runs.csv"     # one row per (sentence, n, seed); reused if the notebook is restarted
```

### Colour scheme (append to the end of Cell 1)
All figures use the **same pink-and-black theme as the Exp 1 and Exp 2 figures**. Paste the colour-scheme block from [EXP3 Part 5](../counterfactual%20swap/EXP3_counterfactual_swap.md) ("Colour scheme") at the end of Cell 1. It overrides the old blue/orange colours in EXP1's Cell 1. 500 samples = `COLORS["n500"]` (dark pink) and 5000 samples = `COLORS["n5000"]` (light pink), the same as the slide 8 and slide 13 charts.

### Cell 2: the sentences
Same as EXP5 Cell 2: the 48 Exp 1 sentences (upload `exp1_sentences.csv`) + the 190 Exp 2 sentences, with `source` and `group` columns. Keep the Exp 2 `term` column so Cell 6 can join with Exp 3.
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
exp2 = pd.DataFrame([(t.format(w), w, "control" if w in CONTROLS else "identity") for t in TEMPLATES for w in TERMS + CONTROLS],
                    columns=["text","term","group"])
S = pd.concat([pd.DataFrame({"text": exp1.text, "term": np.nan, "group": exp1.condition, "source": "exp1"}),
               exp2.assign(source="exp2")], ignore_index=True)
print(len(S))     # expect 238
```

### Cell 3: SHAP once per sentence (signal strength + the comparison for agreement)
```python
SV = shap_explainer("original")(list(S.text), max_evals=500, batch_size=32)
check_efficiency(SV, list(S.text), "original")
S["logit"] = logit(S.text, "original")
S["base"] = [float(SV[i].base_values) for i in range(len(S))]
S["signal"] = (S.logit - S.base).abs()                       # how far the score is from 'everything removed'
SHAP_W = [shap_word_weights(SV[i]) for i in range(len(S))]
S["shap_top"] = [max(w, key=w.get) for w in SHAP_W]
```

### Cell 4: LIME, 10 seeds × 2 sample sizes, with checkpointing
```python
done = set()
if os.path.exists(CKPT):
    old = pd.read_csv(CKPT); done = set(zip(old.i, old.n, old.seed))
for i, text in enumerate(S.text):
    new = []
    for n in SAMPLE_SIZES:
        for seed in SEEDS:
            if (i, n, seed) in done: continue
            lw, exp = lime_word_weights(text, "original", seed=seed, num_samples=n)
            new.append(dict(i=i, n=n, seed=seed, top=max(lw, key=lw.get), r2=exp.score,
                            weights=repr(lw)))
    if new:
        pd.DataFrame(new).to_csv(CKPT, mode="a", header=not os.path.exists(CKPT), index=False)
    if i % 20 == 0: print(f"{i}/{len(S)} sentences done")
runs = pd.read_csv(CKPT); runs["weights"] = runs.weights.apply(eval)
print(len(runs), "LIME runs (expect", len(S) * 20, ")")
```

### Cell 5: stability per sentence
- `flip_rate`: share of runs whose top word differs from the most common top word.
- `rank_stability`: mean Spearman between every pair of the 10 runs.
- `r2`: mean LIME fit.
- `agree`: share of runs whose top word matches SHAP's top word.
```python
def pairwise_rho(ws):
    vals = []
    for a, b in combinations(ws, 2):
        common = sorted(set(a) & set(b))
        if len(common) > 2: vals.append(spearmanr([a[k] for k in common], [b[k] for k in common]).correlation)
    return np.nanmean(vals) if vals else np.nan
rows = []
for (i, n), g in runs.groupby(["i","n"]):
    mode = g.top.mode()[0]
    rows.append(dict(i=i, n=n, flip_rate=float((g.top != mode).mean()), distinct_top=g.top.nunique(),
                     rank_stability=pairwise_rho(list(g.weights)), r2=g.r2.mean(),
                     agree=float((g.top == S.shap_top[i]).mean())))
ST = pd.DataFrame(rows).merge(S, left_on="i", right_index=True)
ST.to_csv(f"{OUT}/exp6_stability_by_sentence.csv", index=False)
for n in SAMPLE_SIZES:
    d = ST[ST.n == n]
    print(f"n={n}: top word changed in {(d.distinct_top > 1).mean():.0%} of sentences; "
          f"rho(signal, rank_stability) = {spearmanr(d.signal, d.rank_stability, nan_policy='omit').correlation:.2f}; "
          f"rho(signal, r2) = {spearmanr(d.signal, d.r2).correlation:.2f}")
```

### Cell 6: do R² and disagreement warn us when the explanation is wrong? (uses Exp 3)
Upload `exp3_results.csv` from `experiments/counterfactual swap/results/`. For the 150 identity sentences, the **error** = |counterfactual effect − SHAP estimate| (from Exp 3). We test whether sentences with low R² or SHAP–LIME disagreement are the ones with big errors.
```python
e3 = pd.read_csv("exp3_results.csv")[["text","cf_original","est_original"]]
e3["error"] = (e3.cf_original - e3.est_original).abs()
W = ST[(ST.n == 5000) & (ST.group == "identity")].merge(e3, on="text")
big = W.error > W.error.quantile(0.75)                     # 'explanation is badly wrong' = top quarter of errors
warn = pd.DataFrame({
    "rho with error": {"low R2 (1-R2)": spearmanr(1 - W.r2, W.error).correlation,
                       "SHAP-LIME disagreement": spearmanr(1 - W.agree, W.error).correlation,
                       "weak signal (-signal)": spearmanr(-W.signal, W.error).correlation},
    "AUC for flagging the worst 25%": {"low R2 (1-R2)": roc_auc_score(big, 1 - W.r2),
                                       "SHAP-LIME disagreement": roc_auc_score(big, 1 - W.agree),
                                       "weak signal (-signal)": roc_auc_score(big, -W.signal)}})
print(warn.round(3)); warn.to_csv(f"{OUT}/exp6_warning_signs.csv")
```
Markdown note after it: *"AUC 0.5 = no better than guessing; above about 0.7 = a useful warning sign. If neither R² nor disagreement gets there, the presentation's recommendation needs softening."*

### Cell 7: Figure 2 (paper Fig. 2)
x = signal strength, y = instability (1 − rank stability), one dot per sentence, two colours (500 and 5000 samples), log scale on x if the values spread over orders of magnitude. Size it for one IEEE column.
```python
fig, ax = plt.subplots(figsize=(3.5, 1.6))   # one IEEE column, short (2026-10-22)
for n, c in [(500, COLORS["n500"]), (5000, COLORS["n5000"])]:
    d = ST[ST.n == n]
    ax.scatter(d.signal, 1 - d.rank_stability, s=4, alpha=0.6, color=c, linewidths=0, label=f"{n} samples")
ax.set_xscale("symlog"); ax.set_xlabel("Signal strength |logit − base| (log-odds)", fontsize=6)
ax.set_ylabel("LIME instability\n(1 − ρ across seeds)", fontsize=6)
ax.legend(frameon=False, fontsize=5.5, markerscale=1.5)
ax.tick_params(labelsize=5.5, width=0.5, length=2)
for s in ax.spines.values(): s.set_linewidth(0.5)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_exp6_instability_vs_signal.pdf"); fig.savefig(f"{OUT}/fig_exp6_instability_vs_signal.png", dpi=300)
```

### Cell 8: results summary
```python
print("=== EXPERIMENT 6 SUMMARY ===")
summ = ST.groupby(["n","source","group"]).agg(flip_rate=("flip_rate","mean"), changed=("distinct_top", lambda s: (s > 1).mean()),
                                             rank_stability=("rank_stability","mean"), r2=("r2","mean"),
                                             agree=("agree","mean")).round(3)
print(summ); summ.to_csv(f"{OUT}/exp6_summary.csv")
print(warn.round(3))
```

### Cell 9: download everything
```python
!zip -qr outputs_exp6.zip outputs_exp6
from google.colab import files; files.download("outputs_exp6.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook, `exp1_sentences.csv` and (once Exp 3 is done) `exp3_results.csv` to Colab; select a T4 GPU.
2. Run all cells. Expect about 2–3 hours, almost all in Cell 4. If Colab disconnects, re-run from the top: Cell 4 skips finished runs. **Download `exp6_lime_runs.csv` now and then**, because Colab deletes files when the session ends.
3. Unzip the outputs into `experiments/lime stability/results/`.
4. Commit under their own GitHub account: `research: experiment 6 LIME stability results`.

---

## Part 6: Checks before you trust the results
- [ ] Cell 4 printed 238 × 20 = 4,760 runs.
- [ ] `check_efficiency` passed in Cell 3.
- [ ] The obvious case from slide 8 ("Being gay is completely normal") should have `distinct_top = 1` at both sample sizes. If it doesn't, something has changed since the presentation. Tell the group.
- [ ] Cell 6 merged 150 rows (all identity sentences). Fewer means the sentence text doesn't match between Exp 3 and here.
- [ ] Copy the Cell 8 summary into the journal entry.

## Part 7: What goes in the paper
- **Fig. 2:** `fig_exp6_instability_vs_signal.pdf`.
- **Table I, column "Top word changes across seeds":** the share of sentences with `distinct_top > 1` at n = 5000 (and n = 500 in the text).
- **IV-D text:** *"Across 238 sentences, LIME's top word changed between seeds in [X]% of sentences at 500 samples and [Y]% at 5000. Instability rises as signal strength falls (ρ = [Z]). LIME's R² flags the worst explanations with AUC [A], and SHAP–LIME disagreement with AUC [B]."*
- **VI (Conclusion):** keep, soften or drop the "report R²" and "use disagreement as a flag" recommendations, depending on Cell 6.

## Part 8: Limitations to state
- Only LIME is re-run with different seeds. SHAP's Partition explainer is deterministic at a fixed `max_evals` (checked in Exp 2), so it has no run-to-run variance to measure.
- Signal strength is measured with SHAP's base value, which depends on the removal baseline (see Exp 3's baseline check).
- The warning-sign test uses only the 150 identity sentences, because only they have a counterfactual reference.
- Original model only, to keep runtime manageable.
