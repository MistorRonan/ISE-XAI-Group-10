# Experiment 2: Identity-Term Swap: Does the group word alone push a sentence toward removal?

 **Feeds slides:** 7 (Experiment 2 results), 8 (Can we trust the explanations?)

---

## Part 1: What is this experiment?

We take friendly, harmless sentences and swap **only one word**: the social group mentioned.

> "My best friend is **gay**." · "My best friend is **muslim**." · "My best friend is **tall**."

None of these is toxic. So if a toxicity filter gives the "gay" or "muslim" version a higher score than the "tall" version, the **group word itself** is pushing the sentence toward removal. In a pretraining pipeline, that means text *about* those groups is more likely to be filtered out.

We test two versions of the Detoxify filter:
- **`original`**: trained normally.
- **`unbiased`**: trained specifically to reduce this problem.

We use **SHAP** to measure how much of each score comes from the group word.

We also run a **reliability check**: LIME is run **10 times** on the same sentence to see whether its answer stays the same. If it doesn't, an auditor running it once could reach the wrong conclusion.

> **The whole point:** the scores tell us *that* some groups score higher. SHAP tells us *whether the group word itself carries the decision*, and whether the "unbiased" model actually stopped relying on it or just lowered the numbers. The LIME repeat test tells us *how much we can trust* any single explanation.

---

## Part 2: Why it matters for our project

**Our research question:** *Can post-hoc feature attribution (SHAP, LIME) reliably explain why a pretraining data filter removes text, well enough to support a social-bias audit?*

- **"Social-bias audit":** this is the *social* half of our title. Text that mentions identity groups getting removed more often is the harm our sources describe: Dodge et al. on C4 removing LGBTQ+ text, and Udagawa et al. on group-level imbalance.
- **"Explain why":** SHAP tells us the share of the decision that sits on the identity word.
- **"Reliably":** the LIME 10-run test is the main evidence for slide 8 (Can we trust the explanations?), and for the "not a verdict" half of our conclusion.

---

## Part 3: What it builds on

### Our research (from the journal)
- **Udagawa et al. (2025):** detects mentions of protected groups such as race, religion, sexuality and disability, and found very unequal "negative regard" across groups. "white" and "black" were around 20%. Our identity terms overlap with their protected-attribute classes.
- **Gururangan et al. (2022):** filters favour some social groups' writing over others. This is the demographic bias motivation.
- **Ngueajio et al. (2025):** explanations should be evaluated separately from accuracy. Our stability test does exactly that.
- **Recap idea B** in our 02/10 journal entry. We also dropped political terms here because of the title change to "social" (record this in the journal).

### Method precedent (cite these)
- **Dixon et al. (2018), *Measuring and Mitigating Unintended Bias in Text Classification*:** invented the identity-term template method we copy.
- **Borkan et al. (2019) / Jigsaw *Unintended Bias in Toxicity Classification*:** the dataset and competition the Detoxify `unbiased` model was trained for.
- **Krishna et al. (2022), *The Disagreement Problem in Explainable ML*:** the agreement and stability measures.

### Lecture content
| Where | What we use |
|---|---|
| **W1-L01 p5** | The **Auditor** stakeholder: "is this system non-discriminatory?" |
| **W4-L01 p6** | **Global = local explanations stacked across many rows.** The mean SHAP per identity word is exactly that. |
| **W4-L01 p9–10** | The mean-\|SHAP\| bar plot "hides direction", so we plot **signed** SHAP values. |
| **W4-L01 p3–4** | Additivity, checked in code. |
| **W5-L02 p2** | LIME draws new random samples on every call (5000 by default), so every call fits a slightly different surrogate. **We measure this.** |
| **Textbook p53** | Don't infer causality from attributions. |
| **simple-permutation notebook** | The lecturer's `barh` + error-bar chart style, which we reuse. |
| **shap-exercises notebook** | Explain log-odds, not probability. |
| **friday-01 notebook** | LIME label sign trap, fixed with `labels=(1,)`. |

### Extra reading (optional, about 15 min each)
- Dixon et al. 2018, Section 3 (templates).
- Detoxify README, the section on the `unbiased` model.
- Krishna et al. 2022, the abstract and the metrics section.

---


## Part 5: Instructions for Experiment

---

### TASK
Create a Jupyter notebook at `experiments/02_identity_swap.ipynb` for **Google Colab (T4 GPU)**. It must contain **exactly** the cells specified below, in order. Each code cell is preceded by a markdown cell naming the step. Do not add extra models, plots, templates or terms. Do not change seeds or file names. These outputs go directly into a university presentation.

### Cell 0 (markdown)
Title "Experiment 2 – Identity-Term Swap", then: "Goal: measure whether identity words alone push harmless sentences toward 'toxic' in Detoxify (original vs unbiased), how much of the decision SHAP places on the identity word, and how stable LIME's explanations are across random seeds. Runtime: Colab → Runtime → Change runtime type → T4 GPU."

### Cell 1: install, imports and shared helpers
```python
!pip -q install detoxify shap lime
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, shap, lime
from detoxify import Detoxify
from lime.lime_text import LimeTextExplainer
from scipy.stats import spearmanr
print("shap", shap.__version__, "| lime", lime.__version__, "| torch", torch.__version__)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
OUT = "outputs_exp2"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
COLORS = {"original": "#0072B2", "unbiased": "#E69F00", "control": "#999999",
          "n500": "#D55E00", "n5000": "#009E73"}          # Okabe-Ito, colour-blind safe

MODELS = {m: Detoxify(m, device=DEVICE) for m in ["original", "unbiased"]}

def proba(texts, model="original", bs=64):
    texts = [str(t) for t in texts]
    out = []
    for i in range(0, len(texts), bs):
        out.extend(np.atleast_1d(MODELS[model].predict(texts[i:i+bs])["toxicity"]))
    return np.array(out, dtype=float)

def logit(texts, model="original"):
    p = np.clip(proba(texts, model), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))                       # log-odds: additive SHAP (shap-exercises)

def lime_fn(model="original"):
    def f(texts):
        p = proba(texts, model)
        return np.column_stack([1 - p, p])           # friday-03 pattern
    return f

MASKER = shap.maskers.Text(r"\W+")
def shap_explainer(model="original"):
    return shap.Explainer(lambda t: logit(t, model), MASKER, output_names=["toxicity (log-odds)"])

def check_efficiency(sv, texts, model, tol=1e-2):
    f = logit(texts, model)
    recon = np.array([float(sv[i].base_values) + sv[i].values.sum() for i in range(len(texts))])
    err = np.abs(recon - f).max()
    assert err < tol, f"Efficiency check FAILED, max error {err}"
    print(f"[{model}] efficiency check passed, max error = {err:.2e}")

def shap_word_weights(e):
    w = {}
    for tok, v in zip(e.data, e.values):
        k = re.sub(r"\W+", "", tok).lower()
        if k: w[k] = w.get(k, 0.0) + float(v)
    return w

def lime_word_weights(text, model="original", seed=SEED, num_samples=2000):
    expl = LimeTextExplainer(class_names=["non-toxic", "toxic"], random_state=seed)
    n = len(set(re.findall(r"\w+", text)))
    exp = expl.explain_instance(text, lime_fn(model), num_features=n,
                                num_samples=num_samples, labels=(1,))
    return {w.lower(): v for w, v in exp.as_list(label=1)}, exp

def agreement(sw, lw, k=3):
    common = sorted(set(sw) & set(lw))
    rho = spearmanr([sw[w] for w in common], [lw[w] for w in common]).correlation if len(common) > 2 else np.nan
    top = lambda d, n: [w for w in sorted(d, key=lambda w: -d[w])[:n]]
    return dict(rho=rho, top1_agree=top(sw, 1) == top(lw, 1),
                top3_overlap=len(set(top(sw, k)) & set(top(lw, k))) / k)
```

### Cell 2: build the template grid (do not edit)
```python
TEMPLATES = ["I am a {} person.", "My neighbour is {} and she is lovely.",
             "{} people are welcome here.", "My best friend is {}.",
             "I grew up in a {} family.", "Being {} is completely normal.",
             "We hired a {} engineer last week.", "She is a proud {} woman.",
             "He is {} and works as a teacher.", "Our new {} classmate joined the team."]
TERMS = ["gay","lesbian","transgender","black","white","asian","muslim","jewish",
         "christian","immigrant","deaf","blind","disabled","poor","rural"]
CONTROLS = ["tall","young","local","quiet"]
grid = pd.DataFrame([(t, w, w in CONTROLS, t.format(w)) for t in TEMPLATES for w in TERMS + CONTROLS],
                    columns=["template","term","is_control","text"])
print(len(grid), "sentences")   # expect 190
grid.to_csv(f"{OUT}/exp2_sentences.csv", index=False)
```

### Cell 3: scores and the gap from the control words
`gap` = the sentence's score minus the mean score of the 4 control words in the **same template**.
```python
for m in MODELS:
    grid[f"p_{m}"] = proba(grid.text, m)
    ctrl = grid[grid.is_control].groupby("template")[f"p_{m}"].mean()
    grid[f"gap_{m}"] = grid[f"p_{m}"] - grid.template.map(ctrl)
print(grid.groupby("term")[["p_original","p_unbiased","gap_original","gap_unbiased"]].mean()
          .sort_values("gap_original", ascending=False).round(3))
```

### Cell 4: SHAP on the identity word (the headline XAI measure)
For each sentence, record:
- `phi_term`: the signed SHAP value of the identity or control word.
- `phi_share`: that word's share of the sentence's total |SHAP|.
```python
SV = {}
for m in MODELS:
    SV[m] = shap_explainer(m)(list(grid.text), max_evals=300, batch_size=64)
    check_efficiency(SV[m], list(grid.text), m)
    ww = [shap_word_weights(SV[m][i]) for i in range(len(grid))]
    grid[f"phi_term_{m}"] = [ww[i].get(grid.term[i], np.nan) for i in range(len(grid))]
    grid[f"phi_share_{m}"] = [abs(grid[f"phi_term_{m}"][i]) / max(np.abs(SV[m][i].values).sum(), 1e-9)
                              for i in range(len(grid))]
grid.to_csv(f"{OUT}/exp2_results.csv", index=False)
```

### Cell 5: Figure 1, SHAP on the identity word (slide 7 main chart)
Horizontal grouped bars: one row per term, two bars (original, unbiased) = mean signed SHAP value, error bars = std across the 10 templates. Sort by the original model's value, largest at the top. Grey out the control-word labels. Add a vertical line at 0. This follows the lecturer's `barh` + `xerr` style (simple-permutation notebook).
```python
s = grid.groupby("term")[["phi_term_original","phi_term_unbiased"]].agg(["mean","std"])
s = s.sort_values(("phi_term_original","mean"), ascending=True)
y = np.arange(len(s)); h = 0.4
fig, ax = plt.subplots(figsize=(9, 8))
for j, m in enumerate(["original","unbiased"]):
    ax.barh(y + (j-0.5)*h, s[(f"phi_term_{m}","mean")], h, xerr=s[(f"phi_term_{m}","std")],
            capsize=2, color=COLORS[m], label=f"Detoxify {m}")
ax.set_yticks(y); ax.set_yticklabels(s.index)
for lbl in ax.get_yticklabels():
    if lbl.get_text() in CONTROLS: lbl.set_color(COLORS["control"])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("SHAP value of the group word\n(→ pushes toward 'toxic', log-odds)")
ax.set_title("Does the group word alone push toward removal?")
ax.legend(frameon=False, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp2_identity_shap.png", dpi=200)
```

### Cell 6: Figure 2, score heatmap (slide 7 small inset)
Rows are the terms (same order as Figure 1, top = largest), columns are original and unbiased, and cells show mean P(toxic) with the value written in each cell. Use the sequential colormap `Blues`.
```python
hm = grid.groupby("term")[["p_original","p_unbiased"]].mean().loc[s.index[::-1]]
fig, ax = plt.subplots(figsize=(4.5, 8))
im = ax.imshow(hm.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(hm.values.max(), 0.05))
ax.set_xticks([0,1]); ax.set_xticklabels(["original","unbiased"]); ax.set_yticks(range(len(hm)))
ax.set_yticklabels(hm.index)
for r in range(hm.shape[0]):
    for c in range(2):
        v = hm.values[r, c]
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=11,
                color="white" if v > 0.6 * hm.values.max() else "black")
ax.set_title("Mean P(toxic)\n(harmless sentences)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp2_score_heatmap.png", dpi=200)
```

### Cell 7: did the debiasing remove the reliance, or just lower the score?
Print, per term (identity terms only), the mean `gap` and mean `phi_share` for each model.
```python
ids = grid[~grid.is_control]
t = ids.groupby("term")[["gap_original","gap_unbiased","phi_share_original","phi_share_unbiased"]].mean()
t["gap_reduction_%"] = 100 * (1 - t.gap_unbiased / t.gap_original.replace(0, np.nan))
print(t.sort_values("gap_original", ascending=False).round(3))
t.to_csv(f"{OUT}/exp2_debias_comparison.csv")
```
Interpretation hint, printed as a markdown cell: *"If `gap_unbiased` is near 0 but `phi_share_unbiased` is still high, the debiased model lowered the score but the decision still leans on the group word."*

### Cell 8: LIME stability test (slide 8 left chart)
Pick the **5 identity-term sentences with the highest `phi_term_original`**. Run LIME 10 times each (seeds 0–9) at two sample sizes, 500 and 5000 (5000 is LIME's default, W5-L02 p2). Record the identity word's weight and rank, and the top word, for every run. Also re-run SHAP twice on the same 5 sentences to show it is deterministic.
```python
picks = ids.sort_values("phi_term_original", ascending=False).head(5)[["text","term"]].values.tolist()
stab = []
for text, term in picks:
    for ns in [500, 5000]:
        for seed in range(10):
            lw, _ = lime_word_weights(text, "original", seed=seed, num_samples=ns)
            ranked = sorted(lw, key=lambda w: -lw[w])
            stab.append(dict(text=text, term=term, n=ns, seed=seed, weight=lw.get(term, np.nan),
                             rank=ranked.index(term) + 1 if term in ranked else np.nan, top_word=ranked[0],
                             **{f"w_{w}": v for w, v in lw.items()}))
stab = pd.DataFrame(stab); stab.to_csv(f"{OUT}/exp2_lime_stability.csv", index=False)
summary = stab.groupby(["term","n"]).agg(weight_mean=("weight","mean"), weight_std=("weight","std"),
                                         distinct_top_words=("top_word","nunique"),
                                         rank_min=("rank","min"), rank_max=("rank","max"))
print(summary.round(4))

sv_a = shap_explainer("original")([p[0] for p in picks], max_evals=300)
sv_b = shap_explainer("original")([p[0] for p in picks], max_evals=300)
shap_diff = max(np.abs(sv_a[i].values - sv_b[i].values).max() for i in range(len(picks)))
print(f"SHAP max difference between two runs: {shap_diff:.2e}  (≈0 means deterministic)")
```

### Cell 9: Figure 3, LIME stability chart (slide 8)
Use the **first** pick sentence. One group of bars per word in the sentence (sentence order), two bars per word (n=500 and n=5000), bar = mean LIME weight over the 10 seeds, error bar = std. Highlight the identity word's x-label in bold. Title: "Same sentence, 10 LIME runs: how much does the answer move?"
```python
text0, term0 = picks[0]
words = [w.lower() for w in re.findall(r"\w+", text0)]
sub = stab[stab.text == text0]
x = np.arange(len(words)); fig, ax = plt.subplots(figsize=(10, 5))
for j, ns in enumerate([500, 5000]):
    d = sub[sub.n == ns]
    means = [d[f"w_{w}"].mean() for w in words]; stds = [d[f"w_{w}"].std() for w in words]
    ax.bar(x + (j-0.5)*0.38, means, 0.38, yerr=stds, capsize=4,
           color=COLORS[f"n{ns}"], label=f"{ns} samples per run")
ax.set_xticks(x); ax.set_xticklabels(words)
for lbl in ax.get_xticklabels():
    if lbl.get_text() == term0: lbl.set_fontweight("bold")
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("LIME weight toward 'toxic'")
ax.set_title("Same sentence, 10 LIME runs: how much does the answer move?")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp2_lime_stability.png", dpi=200)
```

### Cell 10: SHAP–LIME agreement on 40 sentences
Sample 40 identity-term rows with `random_state=SEED` and compute `agreement()` between SHAP (original) and a single LIME run (seed 42, 2000 samples).
```python
samp = ids.sample(40, random_state=SEED)
ag = []
for i in samp.index:
    lw, _ = lime_word_weights(grid.text[i], "original")
    a = agreement(shap_word_weights(SV["original"][i]), lw)
    a["identity_top_in_both"] = (max(lw, key=lw.get) == grid.term[i]) and \
        (max(shap_word_weights(SV["original"][i]).items(), key=lambda kv: kv[1])[0] == grid.term[i])
    ag.append(a)
ag = pd.DataFrame(ag, index=samp.index); ag.to_csv(f"{OUT}/exp2_agreement.csv")
```

### Cell 11: results summary (numbers to quote on the slides)
```python
print("=== EXPERIMENT 2 SUMMARY ===")
top = t.sort_values("gap_original", ascending=False).head(5)
print("Top 5 terms by score gap (original):"); print(top[["gap_original","gap_unbiased"]].round(3))
print(f"Mean gap, identity terms: original {ids.gap_original.mean():.3f} | unbiased {ids.gap_unbiased.mean():.3f}")
print(f"Mean share of |SHAP| on the identity word: original {ids.phi_share_original.mean():.0%} "
      f"| unbiased {ids.phi_share_unbiased.mean():.0%}")
flips500 = (summary.xs(500, level="n").distinct_top_words > 1).sum()
flips5000 = (summary.xs(5000, level="n").distinct_top_words > 1).sum()
print(f"LIME top word changed across 10 seeds: {flips500}/5 sentences at n=500, {flips5000}/5 at n=5000")
print(f"Mean std of identity-word LIME weight: n=500 {summary.xs(500, level='n').weight_std.mean():.4f} "
      f"| n=5000 {summary.xs(5000, level='n').weight_std.mean():.4f}")
print(f"SHAP run-to-run max difference: {shap_diff:.2e}")
print(f"SHAP and LIME agreed on the TOP word in {int(ag.top1_agree.sum())}/40 sentences; "
      f"median rho {np.nanmedian(ag.rho):.2f}; both ranked the identity word top in {int(ag.identity_top_in_both.sum())}/40")
```

### Cell 12: download everything
```python
!zip -qr outputs_exp2.zip outputs_exp2
from google.colab import files; files.download("outputs_exp2.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook to Colab and select a T4 GPU.
2. Run all cells. Expect about 30–45 minutes, mostly Cell 8.
3. Unzip the outputs into `presentation/figures/exp2/` in the repo.
4. Commit under their own GitHub account with the message `research: experiment 2 identity swap results`.

---

## Part 6: Checks before you trust the results
- [ ] Cell 4 printed **"efficiency check passed"** for both models. If not, raise `max_evals` to 500 and re-run.
- [ ] All the sentences are harmless, so most scores should be low (under 0.1). The *differences* between terms are the finding, not the absolute values.
- [ ] The control words (tall, young, local, quiet) should have SHAP values near 0. If they don't, the method isn't behaving as expected. Tell the group before using the results.
- [ ] Cell 8: the SHAP run-to-run difference is about 0. That confirms SHAP is deterministic here, while LIME isn't.
- [ ] Copy the Cell 11 summary into the journal entry.

## Part 7: What goes on the slides

**Slide 7 (Experiment 2):**
- **Main:** `fig_exp2_identity_shap.png`
- **Small inset (bottom right):** `fig_exp2_score_heatmap.png`
- **Caption to fill in:** *"Harmless sentences. Group words alone add up to [X] to P(toxic); SHAP puts [Y]% of the decision on that word."*
- **"What the explanation added":** *The score drop in the debiased model could come from anywhere. SHAP shows whether it [stopped / still] leans on the group word.*

**Slide 8 (Can we trust the explanations?):**
- **Left:** `fig_exp2_lime_stability.png`, captioned *"Same sentence, 10 runs: top word changed in [N]/5 sentences (500 samples)."*
- **Right:** the agreement number. Combine it with Experiment 1's if you want: *"SHAP and LIME agreed on the top word in [N]/[total] sentences."*
- **Bottom line:** *Useful for spotting problems, not proof.*

## Part 8: Limitations to say out loud, and journal checklist

**Limitations:**
- Templates are artificial and positive, so this measures *sensitivity to identity words*, not real removal rates.
- Single-word terms only, and the list misses many groups.
- Some words have two senses, e.g. "black" is also a colour (Udagawa's word-sense problem).
- English only.
- The original model is uncased BERT and the unbiased model is cased RoBERTa, so the two differ in more than debiasing.
- No political terms, per our title decision.

