# Experiment 3: Counterfactual Swap: Does SHAP's explanation match what the identity word actually does?

**Feeds paper sections:** III-D (references), IV-B (faithfulness, Fig. 1, Table I), IV-E (baseline sensitivity), V (Discussion)
**Depends on:** nothing. **Exp 6 uses this experiment's results**, so run this first.

---

## Part 1: What is this experiment?

Exp 2 showed that SHAP blames the identity word. For example, in "My best friend is **gay**", SHAP says *gay* pushes the score toward toxic. But SHAP's number is an *estimate* of what the word does, and the presentation never checked whether that estimate is right.

Because our Exp 2 sentences come in minimal pairs, we can check it. "My best friend is **gay**" and "My best friend is **tall**" differ by exactly one word. So we can just **run the filter on both and measure the difference**. That measured difference is the **counterfactual effect** of the word: what the word really does to the score. It's our **reference**, an answer we know independently of SHAP.

Then we check: **does SHAP's estimate for the word match the counterfactual effect?**

> **Example:** the filter gives the "gay" sentence a log-odds of −1.0 and the four control sentences an average of −6.0. The counterfactual effect is **+5.0**. If SHAP gives *gay* +4.8 more than it gives the control words, SHAP is faithful here. If it gives +2.0, the other 3.0 has been spread onto other words, and SHAP is understating the identity word.

**Why the two numbers can differ at all.** SHAP's efficiency axiom guarantees that all the word contributions add up to the full score. So the *total* change between the two sentences is always explained. The question is whether SHAP puts that change **on the identity word** or **spills it onto other words** ("best", "friend"…). That spill-over is exactly what we measure.

We also do two extra checks in the same notebook:
1. **Re-test the presentation's "the unbiased model leans on the group word more (46% → 60%)" claim.** That number was a *share* of SHAP, which can rise just because the other words' SHAP values shrink. We check it against absolute SHAP values and the counterfactual effect.
2. **Baseline check.** To work out a word's contribution, SHAP has to "remove" it, and there's more than one way to do that: delete it, replace it with `...`, or replace it with the model's `[MASK]` token. We run all three and see how much the answer changes.

---

## Part 2: Why it matters for the paper

**Research question:** *Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?*

- This is the main evidence for **"faithful when the signal is strong"** in our thesis. Without it, "reliably" rests only on SHAP and LIME agreeing, which proves nothing (Krishna 2022).
- It's the first of our two **references** (the other is exact Shapley values, Exp 4).
- The 46% → 60% re-test is **critical analysis of our own earlier result**. Whichever way it comes out, it goes in the paper and the journal.
- The baseline check backs up **"they shift with the choice of removal baseline"** in the thesis. It turns slide 9's "removing a word isn't neutral" from a claim into a number.

---

## Part 3: What it builds on

### Our research (from the journal)
- **Exp 2 (identity-term swap):** same sentence grid, same models. This experiment re-runs its SHAP with the tokenizer fix and adds the reference.
- **Dixon et al. (2018):** the identity-term template method.
- **Krishna et al. (2022):** agreement between explainers isn't the same as being right. That's the reason we need a reference.

### Method precedent (add source reviews for the new ones)
- **Garg et al. (2019), *Counterfactual Fairness in Text Classification*:** swapping identity terms and measuring the score change ("counterfactual token fairness"). This is our counterfactual effect.
- **Hase et al. (2021), *The Out-of-Distribution Problem in Explainability*:** removed-word inputs are unnatural, which is why the baseline matters.
- **Lundberg & Lee (2017):** SHAP and its efficiency property.

### Lecture content
| Where | What we use |
|---|---|
| **W4-L01 p3–4** | Additivity / efficiency: contributions sum to the score. Checked in code, and it's why "spill-over" is the right way to describe a mismatch. |
| **W4-L01 p10** | Mean \|SHAP\| hides direction, so we use signed values. |
| **Textbook p48** | The four axioms, and *"choosing a good baseline can be difficult"*, which is the baseline check. |
| **Textbook p57** | Normalised Shapley values may not add up cleanly. That's the reason to distrust the share metric behind 46% → 60%. |
| **shap-exercises notebook** | Explain log-odds, not probability. |
| **friday-01 notebook** | LIME label sign trap, fixed with `labels=(1,)`. |

---

## Part 4: How it fits the thesis

| Result | What it means for the thesis |
|---|---|
| SHAP estimate ≈ counterfactual effect for strong identity terms (gay, transgender, lesbian…) | Supports "faithful when the signal is strong" |
| Bigger mismatch / sign errors for weak terms (asian, christian, immigrant…) | Supports "faithfulness drops when the signal is weak" |
| Term ranking changes between maskers | Supports "shifts with the choice of removal baseline" |
| SHAP doesn't match even on strong terms | **Thesis flips** to "unreliable even for triage". Tell the group straight away. |

---

## Part 5: Instructions for the experiment

### TASK
Create a Jupyter notebook at `experiments/counterfactual swap/03_counterfactual_swap.ipynb` for **Google Colab (T4 GPU)**. It must contain **exactly** the cells below, in order. Each code cell is preceded by a markdown cell naming the step. Do not change seeds or file names. These outputs go directly into the paper.

### Cell 0 (markdown)
Title "Experiment 3 – Counterfactual Swap", then: "Goal: test whether SHAP's attribution to an identity word matches the word's counterfactual effect (the measured change in logit when the identity word is swapped for a neutral control word), re-test the Exp 2 'debiasing' share result, and measure how much SHAP changes with the removal baseline. Runtime: Colab → Runtime → Change runtime type → T4 GPU."

### Cell 1: install, imports and shared helpers
Copy **EXP1 Cell 1 exactly** (it has the fixed `word_tokenizer` masker), but set `OUT = "outputs_exp3"`. Then append:
```python
from scipy.stats import pearsonr
print("default mask token:", repr(MASKER.mask_token))   # we expect '...'; write down what it prints

def make_masker(mask_token):
    """Same word-level tokens as MASKER, different replacement for a 'removed' word."""
    return shap.maskers.Text(word_tokenizer, mask_token=mask_token)

def shap_explainer(model="original", masker=None):
    return shap.Explainer(lambda t: logit(t, model), masker or MASKER)

def faith(ref, est):
    """Faithfulness of an estimate to a reference: correlation, sign agreement, mean absolute error."""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    return dict(pearson=pearsonr(ref, est)[0], spearman=spearmanr(ref, est).correlation,
                sign_agree=float(np.mean(np.sign(ref) == np.sign(est))),
                mae=float(np.mean(np.abs(ref - est))))
```

### Colour scheme (append to the end of Cell 1)
All figures use the **same pink-and-black theme as the Exp 1 and Exp 2 figures**: white background, black text and axes, pink bars and points, grey for controls and baselines. The EXP1 spec's Cell 1 still lists the old blue/orange colours, so **paste this block after it so the new colours override them**. The values are copied from `02_identity_swap.ipynb`.
```python
BG, FG = "#FFFFFF", "#000000"                     # pink theme (same as Exp 1 and 2): white background, black text, pink marks
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
                     "xtick.color": FG, "ytick.color": FG, "legend.labelcolor": FG})
COLORS = {"original": "#D6006F", "unbiased": "#FF8FC4", "control": "#9A8590",
          "n500": "#A3134F", "n5000": "#FF7AB8", "err": "#333333",
          "SHAP": "#D6006F", "LIME": "#FF8FC4", "random": "#9A8590",
          "SHAP partition": "#D6006F", "SHAP permutation": "#7A0A3E"}
PINK_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("pink_black", ["#FFF0F6", "#FF4FA3", "#7A0A3E"])
```
(`import matplotlib` first if Cell 1 doesn't already import it.) Never use matplotlib's default colours: every `plot`, `bar` and `scatter` call passes `color=COLORS[...]`, and heatmaps use `PINK_CMAP`.

### Cell 2: the Exp 2 grid (do not edit)
Copy **EXP2 Cell 2 exactly** (same `TEMPLATES`, `TERMS`, `CONTROLS`), but save to `f"{OUT}/exp3_sentences.csv"`. Expect 190 sentences.

### Cell 3: scores
```python
for m in MODELS:
    grid[f"p_{m}"] = proba(grid.text, m)
    grid[f"logit_{m}"] = logit(grid.text, m)
```

### Cell 4: SHAP with the fixed tokenizer (default masker)
```python
SV = {}
for m in MODELS:
    SV[m] = shap_explainer(m)(list(grid.text), max_evals=500, batch_size=32)
    check_efficiency(SV[m], list(grid.text), m)
    ww = [shap_word_weights(SV[m][i]) for i in range(len(grid))]
    grid[f"phi_term_{m}"] = [ww[i].get(grid.term[i], np.nan) for i in range(len(grid))]
    grid[f"phi_share_{m}"] = [abs(grid[f"phi_term_{m}"][i]) / max(np.abs(SV[m][i].values).sum(), 1e-9)
                              for i in range(len(grid))]
    grid[f"base_{m}"] = [float(SV[m][i].base_values) for i in range(len(grid))]
```

### Cell 5: counterfactual effect vs SHAP estimate (the headline measure)
For each identity sentence:
- `cf` = its logit − the mean logit of the 4 control sentences in the same template (**the reference**).
- `est` = its identity word's SHAP value − the mean control-word SHAP value in the same template (**SHAP's estimate**).
- `spill` = `cf − est` (how much of the real effect SHAP put on other words).
```python
for m in MODELS:
    c = grid[grid.is_control].groupby("template")
    grid[f"cf_{m}"]    = grid[f"logit_{m}"]    - grid.template.map(c[f"logit_{m}"].mean())
    grid[f"est_{m}"]   = grid[f"phi_term_{m}"] - grid.template.map(c[f"phi_term_{m}"].mean())
    grid[f"spill_{m}"] = grid[f"cf_{m}"] - grid[f"est_{m}"]
ids = grid[~grid.is_control].copy()

rows = []
for m in MODELS:
    r = faith(ids[f"cf_{m}"], ids[f"est_{m}"]); r.update(model=m, level="sentence (n=150)"); rows.append(r)
    t = ids.groupby("term")[[f"cf_{m}", f"est_{m}"]].mean()
    r = faith(t[f"cf_{m}"], t[f"est_{m}"]); r.update(model=m, level="term (n=15)"); rows.append(r)
faith_shap = pd.DataFrame(rows); print(faith_shap.round(3))
faith_shap.to_csv(f"{OUT}/exp3_faithfulness_shap.csv", index=False)
```

### Cell 6: the same check for LIME
LIME weights are in probability units, so compare them with the counterfactual effect **in probability** and use rank correlation only. One LIME run per sentence: seed 42, 5000 samples (LIME's default, W5-L02 p2).
```python
for m in MODELS:
    lw = [lime_word_weights(t, m, seed=SEED, num_samples=5000)[0] for t in grid.text]
    grid[f"lime_term_{m}"] = [lw[i].get(grid.term[i], np.nan) for i in range(len(grid))]
    c = grid[grid.is_control].groupby("template")
    grid[f"cf_p_{m}"]     = grid[f"p_{m}"]         - grid.template.map(c[f"p_{m}"].mean())
    grid[f"est_lime_{m}"] = grid[f"lime_term_{m}"] - grid.template.map(c[f"lime_term_{m}"].mean())
ids = grid[~grid.is_control].copy()
faith_lime = pd.DataFrame([dict(model=m, **faith(ids[f"cf_p_{m}"], ids[f"est_lime_{m}"])) for m in MODELS])
print(faith_lime[["model","spearman","sign_agree"]].round(3))
faith_lime.to_csv(f"{OUT}/exp3_faithfulness_lime.csv", index=False)
grid.to_csv(f"{OUT}/exp3_results.csv", index=False)      # Exp 6 needs this file
```

### Cell 7: Figure 1 (paper Fig. 1)
Two panels side by side (original, unbiased). x = counterfactual effect, y = SHAP estimate, one dot per identity sentence, dashed y = x line, a short label at each term's mean point. Size it for one IEEE column.
```python
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharex=True, sharey=True)   # one IEEE column (2026-10-22)
for ax, m in zip(axes, MODELS):
    ax.scatter(ids[f"cf_{m}"], ids[f"est_{m}"], s=4, alpha=0.6, color=COLORS[m], linewidths=0)
    lim = [min(ids[f"cf_{m}"].min(), ids[f"est_{m}"].min()) - 0.5, max(ids[f"cf_{m}"].max(), ids[f"est_{m}"].max()) + 0.5]
    ax.plot(lim, lim, "k--", lw=0.6)
    for term, g in ids.groupby("term"):
        ax.annotate(term, (g[f"cf_{m}"].mean(), g[f"est_{m}"].mean()), fontsize=4.5)
    r = faith(ids[f"cf_{m}"], ids[f"est_{m}"])
    ax.set_title(f"Detoxify {m} (ρ = {r['spearman']:.2f})", fontsize=7)
    ax.set_xlabel("Counterfactual effect (log-odds)", fontsize=6)
    ax.tick_params(labelsize=5.5, width=0.5, length=2)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("SHAP estimate (log-odds)", fontsize=6)
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(f"{OUT}/fig_exp3_cf_vs_shap.pdf"); fig.savefig(f"{OUT}/fig_exp3_cf_vs_shap.png", dpi=300)
```

### Cell 8: re-test the "46% → 60%" claim
Per model: the mean share (the presentation's measure) next to the mean **absolute** SHAP value of the identity word, the mean SHAP estimate and the mean counterfactual effect.
```python
rt = pd.DataFrame({m: {"share_of_|SHAP| (presentation measure)": ids[f"phi_share_{m}"].mean(),
                       "mean phi_term (log-odds)": ids[f"phi_term_{m}"].mean(),
                       "mean SHAP estimate (log-odds)": ids[f"est_{m}"].mean(),
                       "mean counterfactual effect (log-odds)": ids[f"cf_{m}"].mean()} for m in MODELS})
print(rt.round(3)); rt.to_csv(f"{OUT}/exp3_debias_retest.csv")
per_term = ids.groupby("term")[[f"{k}_{m}" for m in MODELS for k in ("phi_share","phi_term","cf")]].mean()
per_term.round(3).to_csv(f"{OUT}/exp3_debias_retest_per_term.csv")
```
Markdown cell after it: *"If the share went up but the absolute SHAP value and the counterfactual effect went down, the presentation's '46% → 60%' was a side-effect of the share measure, not a sign that the unbiased model leans on the word more. Record which it is in the journal."*

### Cell 9: baseline check (how much does the removal baseline matter?)
Three ways of "removing" a word: `"..."` (the default, already done in Cell 4), deletion, and each model's own mask token (`[MASK]` for original, `<mask>` for unbiased).

Two shap quirks (tested on shap 0.52):
- `mask_token=""` crashes the text masker, so deletion uses a single space `" "`.
- Mask tokens get a trailing space (`"[MASK] "`) so they don't glue onto the next word.
```python
BASELINES = {m: {"dots": "...", "delete": " ", "model_mask": MODELS[m].tokenizer.mask_token + " "} for m in MODELS}
print(BASELINES)
bl_rows, term_rank = [], {}
for m in MODELS:
    for name, tok in BASELINES[m].items():
        if name == "dots":
            phi = grid[f"phi_term_{m}"]
        else:
            sv = shap_explainer(m, make_masker(tok))(list(grid.text), max_evals=500, batch_size=32)
            check_efficiency(sv, list(grid.text), m)
            ww = [shap_word_weights(sv[i]) for i in range(len(grid))]
            phi = pd.Series([ww[i].get(grid.term[i], np.nan) for i in range(len(grid))], index=grid.index)
        est = phi - grid.template.map(phi[grid.is_control].groupby(grid.template).mean())
        r = faith(grid.loc[~grid.is_control, f"cf_{m}"], est[~grid.is_control])
        r.update(model=m, baseline=name, control_mean_abs_phi=float(phi[grid.is_control].abs().mean()))
        bl_rows.append(r)
        term_rank[(m, name)] = phi[~grid.is_control].groupby(grid.term).mean()
bl = pd.DataFrame(bl_rows); print(bl.round(3)); bl.to_csv(f"{OUT}/exp3_baselines.csv", index=False)
for m in MODELS:   # do the three baselines rank the 15 identity terms the same way?
    for a, b in [("dots","delete"), ("dots","model_mask"), ("delete","model_mask")]:
        print(m, a, "vs", b, "term-ranking rho =", round(spearmanr(term_rank[(m,a)], term_rank[(m,b)]).correlation, 3))
```

### Cell 10: results summary (numbers to quote in the paper)
```python
print("=== EXPERIMENT 3 SUMMARY ===")
for m in MODELS:
    s = faith(ids[f"cf_{m}"], ids[f"est_{m}"])
    print(f"[{m}] SHAP vs counterfactual effect: Spearman {s['spearman']:.2f}, Pearson {s['pearson']:.2f}, "
          f"sign agreement {s['sign_agree']:.0%}, MAE {s['mae']:.2f} log-odds (n=150)")
    print(f"[{m}] mean spill-over onto other words: {ids[f'spill_{m}'].mean():.2f} log-odds")
    print(f"[{m}] LIME vs counterfactual effect (prob.): Spearman "
          f"{faith(ids[f'cf_p_{m}'], ids[f'est_lime_{m}'])['spearman']:.2f}")
print("Weakest-faithfulness terms (largest mean |spill|, original):")
print(ids.groupby("term")["spill_original"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).head(5).round(2))
print(bl[["model","baseline","spearman","mae","control_mean_abs_phi"]].round(3))
```

### Cell 11: download everything
```python
!zip -qr outputs_exp3.zip outputs_exp3
from google.colab import files; files.download("outputs_exp3.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook to Colab and select a T4 GPU.
2. Run all cells. Expect about 45–60 minutes, mostly Cells 6 and 9.
3. Unzip the outputs into `experiments/counterfactual swap/results/`.
4. Commit under their own GitHub account with the message `research: experiment 3 counterfactual swap results`.

---

## Part 6: Checks before you trust the results
- [ ] Cell 1 printed the default mask token. Write it in the journal: it's the baseline Exp 1 and Exp 2 actually used.
- [ ] Every `check_efficiency` call printed **"efficiency check passed"**. If not, raise `max_evals` to 1000.
- [ ] Control words have SHAP values near 0 under the default masker. The tokenizer fix should make this *better* than Exp 2. If not, tell the group.
- [ ] The P(toxic) scores match Exp 2's to about 3 decimal places. They should, because the sentences and models are the same.
- [ ] Cell 9: look at one masked sentence per baseline (e.g. `make_masker("")(np.array([True]*3+[False]*3), grid.text[0])`) and check the masking does what you think.
- [ ] Copy the Cell 10 summary into the journal entry.

## Part 7: What goes in the paper
- **Fig. 1:** `fig_exp3_cf_vs_shap.pdf`.
- **Table I, column "ρ with counterfactual effect":** the sentence-level Spearman for SHAP and LIME (original model; put unbiased in the text or a second column).
- **IV-B text:** *"SHAP's estimate of the identity word's effect correlates with its counterfactual effect at ρ = [X] (n = 150), with sign agreement [Y]%. Faithfulness is lowest for [weak terms], where on average [Z] log-odds of the effect is assigned to other words."*
- **IV-B text, 46 → 60:** one or two sentences saying whether the claim survives, using `exp3_debias_retest.csv`.
- **IV-E text:** *"Changing the removal baseline from '...' to deletion / [MASK] changes the identity-term ranking (ρ = [X]) and the agreement with the counterfactual effect (ρ = [Y] vs [Z])."*

## Part 8: Limitations to state
- The counterfactual effect measures the word's *total* effect, including how it interacts with the rest of the sentence. A Shapley value splits interactions between words, so a perfect match isn't expected. A small mismatch isn't automatically SHAP's fault. Say this in Method.
- The reference depends on our choice of the 4 control words. A different "neutral" set would give a slightly different reference.
- Templates only, English only, single-word terms (same as Exp 2).
- The original and unbiased models differ in architecture (BERT vs RoBERTa), not just in debiasing.
- This reference only exists because we controlled the input. A real auditor looking at web text wouldn't have it.
