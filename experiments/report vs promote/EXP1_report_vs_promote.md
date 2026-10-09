# Experiment 1: "Report vs Promote": What makes a filter remove a sentence?

 **Feeds slides:** 5 (SHAP vs LIME example), 6 (Experiment 1 results), 8 (Can we trust the explanations?)

---

## Part 1: What is this experiment? 

A toxicity filter is one of the "gatekeepers" that decides which web text gets into an LLM's training data. Research shows these filters often remove pages that **report** or **argue against** abuse, not just pages that are abusive.

We write sets of three sentences that use **the same insult** but with **different intent**:

| Type | Example |
|---|---|
| **Promote** (actually abusive) | Those fans are disgusting idiots and should be banned. |
| **Report** (news about abuse) | The club condemned a post calling its fans disgusting idiots who should be banned. |
| **Counter** (arguing against abuse) | Calling the fans disgusting idiots is unfair and wrong. |

We run a real toxicity filter (**Detoxify**) on all of them. Then we use **SHAP and LIME** to see **which words** made it score each one as toxic.

> **The whole point:** the toxicity score alone only tells us *that* the news report got flagged. SHAP and LIME tell us *why*: is the filter reacting to the insult words, or does it understand the sentence is a report? We also check whether SHAP and LIME **agree** on the answer.

---

## Part 2: Why it matters for our project

**Our research question:** *Can post-hoc feature attribution (SHAP, LIME) reliably explain why a pretraining data filter removes text, well enough to support a social-bias audit?*

The experiment answers both halves of that question:
1. **"Can it explain why?"** If SHAP/LIME show the quoted insult driving the score even in the report and counter sentences, the explanation has revealed the *mechanism* behind a known filter failure. That's something aggregate statistics can't show.
2. **"Reliably?"** We measure how often SHAP and LIME pick the same words. That number goes on slide 8, which asks whether the explanations can be trusted.

The talk's throughline is that explanations are **an aid to auditing, not a verdict**. This experiment shows the "aid" part: the explanation points us at the problem. The agreement number starts the "not a verdict" part.

---

## Part 3: What it builds on

### Our research (from the journal)
- **Mendu et al. (2025), "Towards Safer Pretraining":** most false positives of their toxicity filters (HarmFormer, TTP) were **reporting pages**, i.e. pages that describe hate without endorsing it. They found this by manual inspection, with no explanation of *which words* caused it. **We add the word-level explanation.**
- **Udagawa et al. (2025):** their own bias filter downsampled a **news report of a racist attack**. This is the same "discussing vs doing harm" problem in a different kind of filter.
- **Ngueajio et al. (2025) survey:** "accuracy ≠ explainability". A filter can score well and still be reacting to the wrong signal.
- **Recap idea A** in our 02/10 journal entry.

### Lecture content
| Where | What we use |
|---|---|
| **W1-L01 p5** | The **Auditor** stakeholder: "can I verify this system is… non-discriminatory?" We are the auditor. |
| **W1-L01 p8** | Taxonomy: SHAP and LIME are **post-hoc, local, model-agnostic**. Say this on slide 5. |
| **W4-L01 p2** | SHAP helps "catch a model leaning on the wrong signal", which is exactly what we're testing. |
| **W4-L01 p3–4** | Words are players and the score is the payout; **additivity**: base value + contributions = prediction. We check this in code. |
| **W4-L02 p2–4** | LIME recipe (perturb → weight → fit a simple surrogate) and **local fidelity**. We report LIME's fidelity score. |
| **W5-L02 p2** | LIME is random every run, so we fix `random_state`. |
| **Textbook p48 / p53** | Efficiency axiom; **don't read attributions as causal**. |
| **friday-03 notebook** | The LIME-on-text pattern: a function that turns strings into probabilities. |
| **friday-01 notebook** | LIME explains class index 1 by default, which is a sign trap. We pass `labels=(1,)` explicitly. |
| **shap-exercises notebook** | Explain **log-odds, not probability**, so contributions add up properly. |

### Readings
- Detoxify README: github.com/unitaryai/detoxify. Explains the `original` (BERT) and `unbiased` (RoBERTa) models.
- Mathew et al. (2019), *Thou shalt not hate: A study of counter speech*. Use it to cite "counter-speech" as a concept.
- Krishna et al. (2022), *The Disagreement Problem in Explainable ML*. Source for comparing explainers via top-k and rank agreement.

---

## Part 4: How to tie it back when presenting

**Say (slide 6, about 50 seconds):**

---

## Part 5: Instructions Experiment

---

### TASK
Create a Jupyter notebook at `experiments/01_report_vs_promote.ipynb` for **Google Colab (T4 GPU)**. It must contain **exactly** the cells specified below, in order. Each cell starts with a markdown heading cell naming the step. Do not add extra models, plots or sentences. Do not change seeds, the sentence list or file names. These outputs go directly into a university presentation.

After creating the notebook, also create `experiments/data/exp1_sentences.csv` from the sentence table in Cell 2. Columns: `pair_id,topic,condition,text,target_words`.

### Cell 0 (markdown)
Title "Experiment 1 – Report vs Promote", then one paragraph: "Goal: explain *why* a toxicity filter (Detoxify) flags sentences that report or argue against abuse, using SHAP and LIME, and measure whether the two explainers agree. Runtime: Colab → Runtime → Change runtime type → T4 GPU."

### Cell 1: install, imports and shared helpers
```python
!pip -q install detoxify shap lime
import importlib.metadata, os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, shap, lime, sklearn
from matplotlib import colors as mcolors
from detoxify import Detoxify
from lime.lime_text import LimeTextExplainer
from scipy.stats import spearmanr
print("shap", shap.__version__, "| lime", importlib.metadata.version("lime"), "| torch", torch.__version__)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
OUT = "outputs_exp1"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
COLORS = {"original": "#0072B2", "unbiased": "#E69F00"}   # Okabe-Ito, colour-blind safe

MODELS = {m: Detoxify(m, device=DEVICE) for m in ["original", "unbiased"]}

def proba(texts, model="original", bs=64):
    """P(toxic) for a list of strings, batched so LIME's thousands of samples don't run out of memory."""
    texts = [str(t) for t in texts]
    out = []
    for i in range(0, len(texts), bs):
        out.extend(np.atleast_1d(MODELS[model].predict(texts[i:i+bs])["toxicity"]))
    return np.array(out, dtype=float)

def logit(texts, model="original"):
    """Explain log-odds, not probability, so SHAP contributions are additive (shap-exercises notebook)."""
    p = np.clip(proba(texts, model), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def lime_fn(model="original"):
    """friday-03 pattern: list of strings -> (n, 2) array of [P(non-toxic), P(toxic)]."""
    def f(texts):
        p = proba(texts, model)
        return np.column_stack([1 - p, p])
    return f

def word_tokenizer(s, return_offsets_mapping=True):
    """Word-level tokens that keep their trailing punctuation. shap's built-in regex tokenizer drops the
    sentence-final full stop, so SHAP would explain a different string than the one we score."""
    spans = [m.span() for m in re.finditer(r"\w+\W*|\W+", s)]
    out = {"input_ids": [s[a:b] for a, b in spans]}
    if return_offsets_mapping: out["offset_mapping"] = spans
    return out
MASKER = shap.maskers.Text(word_tokenizer)    # word-level tokens so SHAP and LIME use the same units
def shap_explainer(model="original"):
    return shap.Explainer(lambda t: logit(t, model), MASKER)

def check_efficiency(sv, texts, model, tol=1e-2):
    """Efficiency/additivity axiom (W4-L01 p4, textbook p48): base value + sum of SHAP values = model output."""
    f = logit(texts, model)
    recon = np.array([float(sv[i].base_values) + sv[i].values.sum() for i in range(len(texts))])
    err = np.abs(recon - f).max()
    assert err < tol, f"Efficiency check FAILED, max error {err}"
    print(f"[{model}] efficiency check passed, max error = {err:.2e}")

def shap_word_weights(e):
    """Collapse SHAP tokens into lower-case words; sum repeated words."""
    w = {}
    for tok, v in zip(e.data, e.values):
        k = re.sub(r"\W+", "", tok).lower()
        if k: w[k] = w.get(k, 0.0) + float(v)
    return w

def lime_word_weights(text, model="original", seed=SEED, num_samples=2000):
    """LIME on one sentence. labels=(1,) so weights refer to the TOXIC class (friday-01 sign trap)."""
    expl = LimeTextExplainer(class_names=["non-toxic", "toxic"], random_state=seed)
    n = len(set(re.findall(r"\w+", text)))
    exp = expl.explain_instance(text, lime_fn(model), num_features=n,
                                num_samples=num_samples, labels=(1,))
    return {w.lower(): v for w, v in exp.as_list(label=1)}, exp

def agreement(sw, lw, k=3):
    """Krishna et al. (2022)-style agreement: rank correlation, top-1 match, top-k overlap.
    Ranks are used because SHAP is in log-odds and LIME is in probability, so raw sizes are not comparable."""
    common = sorted(set(sw) & set(lw))
    rho = spearmanr([sw[w] for w in common], [lw[w] for w in common]).correlation if len(common) > 2 else np.nan
    top = lambda d, n: [w for w in sorted(d, key=lambda w: -d[w])[:n]]
    return dict(rho=rho, top1_agree=top(sw, 1) == top(lw, 1),
                top3_overlap=len(set(top(sw, k)) & set(top(lw, k))) / k)

def highlight_png(pairs, title, path):
    """Draw a sentence with each word coloured by its attribution (red = toward toxic, blue = away).
    Colours are scaled per figure, so compare colour patterns between SHAP and LIME, not exact shades."""
    vmax = max([abs(v) for _, v in pairs] + [1e-9])
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax); cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(14, 1.4)); ax.axis("off")
    fig.canvas.draw(); r = fig.canvas.get_renderer(); x = 0.005
    for w, v in pairs:
        t = ax.text(x, 0.5, w, fontsize=20, va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.25", fc=cmap(norm(v)), ec="none"))
        x = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted()).x1 + 0.01
    ax.set_title(title, loc="left", fontsize=16)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
```

### Cell 2: the sentences (do not edit)
Build `df` from this exact table: 16 topics × 3 conditions = 48 rows. `target_words` are the insult words the filter could key on.

```python
ROWS = [
 (1,"football fans","promote","Those fans are disgusting idiots and should be banned.","disgusting idiots"),
 (1,"football fans","report","The club condemned a post calling its fans disgusting idiots who should be banned.","disgusting idiots"),
 (1,"football fans","counter","Calling the fans disgusting idiots is unfair and wrong.","disgusting idiots"),
 (2,"immigrants","promote","Immigrants are worthless and we hate them here.","worthless hate"),
 (2,"immigrants","report","Police are investigating leaflets claiming immigrants are worthless.","worthless"),
 (2,"immigrants","counter","Nobody is worthless, and saying immigrants are is hateful.","worthless hateful"),
 (3,"politicians","promote","Politicians are lying scumbags who should all be thrown out.","lying scumbags"),
 (3,"politicians","report","A survey found many voters describe politicians as lying scumbags.","lying scumbags"),
 (3,"politicians","counter","It is lazy and unfair to call all politicians lying scumbags.","lying scumbags"),
 (4,"vegans","promote","Vegans are annoying freaks and nobody likes them.","annoying freaks"),
 (4,"vegans","report","The article quoted a comment calling vegans annoying freaks.","annoying freaks"),
 (4,"vegans","counter","Vegans are not annoying freaks, they just eat differently.","annoying freaks"),
 (5,"teenagers","promote","Teenagers are useless brats who ruin everything.","useless brats"),
 (5,"teenagers","report","The headteacher criticised an advert that called teenagers useless brats.","useless brats"),
 (5,"teenagers","counter","Teenagers are not useless brats and deserve respect.","useless brats"),
 (6,"neighbours","promote","My neighbours are filthy pigs and I hate them.","filthy pigs hate"),
 (6,"neighbours","report","The court heard the defendant called his neighbours filthy pigs.","filthy pigs"),
 (6,"neighbours","counter","It was cruel of him to call his neighbours filthy pigs.","filthy pigs"),
 (7,"refugees","promote","Refugees are parasites draining our country.","parasites"),
 (7,"refugees","report","The report documents politicians describing refugees as parasites.","parasites"),
 (7,"refugees","counter","Refugees are not parasites, and that language dehumanises people.","parasites"),
 (8,"women","promote","Women are too stupid to drive.","stupid"),
 (8,"women","report","Researchers studied online comments claiming women are too stupid to drive.","stupid"),
 (8,"women","counter","The idea that women are too stupid to drive is simply false.","stupid"),
 (9,"poor people","promote","Poor people are lazy losers who deserve nothing.","lazy losers"),
 (9,"poor people","report","The charity condemned a tweet calling poor people lazy losers.","lazy losers"),
 (9,"poor people","counter","Poor people are not lazy losers, and that stereotype hurts.","lazy losers"),
 (10,"referee","promote","That referee is a pathetic moron.","pathetic moron"),
 (10,"referee","report","Fans were fined for chanting that the referee is a pathetic moron.","pathetic moron"),
 (10,"referee","counter","Shouting that the referee is a pathetic moron is unacceptable.","pathetic moron"),
 (11,"gamers","promote","Gamers are smelly idiots.","smelly idiots"),
 (11,"gamers","report","A columnist was criticised for writing that gamers are smelly idiots.","smelly idiots"),
 (11,"gamers","counter","Gamers are not smelly idiots, that is a lazy insult.","smelly idiots"),
 (12,"tourists","promote","Tourists are disgusting and should stay away.","disgusting"),
 (12,"tourists","report","Locals put up signs saying tourists are disgusting, the paper reported.","disgusting"),
 (12,"tourists","counter","It is wrong to say tourists are disgusting.","disgusting"),
 (13,"cyclists","promote","Cyclists are arrogant idiots and should be banned from roads.","arrogant idiots"),
 (13,"cyclists","report","The council received complaints calling cyclists arrogant idiots.","arrogant idiots"),
 (13,"cyclists","counter","Cyclists are not arrogant idiots, most ride safely.","arrogant idiots"),
 (14,"students","promote","Students are lazy scroungers.","lazy scroungers"),
 (14,"students","report","A headline describing students as lazy scroungers sparked complaints.","lazy scroungers"),
 (14,"students","counter","Students are not lazy scroungers, many work two jobs.","lazy scroungers"),
 (15,"rural people","promote","Rural people are dumb hicks.","dumb hicks"),
 (15,"rural people","report","The study found online posts calling rural people dumb hicks.","dumb hicks"),
 (15,"rural people","counter","Rural people are not dumb hicks, that is a nasty stereotype.","dumb hicks"),
 (16,"older people","promote","Old people are useless and a burden.","useless burden"),
 (16,"older people","report","The charity reported that older people are often told they are useless and a burden.","useless burden"),
 (16,"older people","counter","Old people are not useless or a burden.","useless burden"),
]
df = pd.DataFrame(ROWS, columns=["pair_id","topic","condition","text","target_words"])
df["n_words"] = df.text.str.findall(r"\w+").str.len()
print(df.groupby("condition").n_words.mean())   # length check: report sentences are longer (a stated limitation)
df.to_csv(f"{OUT}/exp1_sentences.csv", index=False)
```

### Cell 3: score every sentence with both filters
```python
for m in MODELS:
    df[f"p_{m}"] = proba(df.text, m)
print(df.groupby("condition")[["p_original","p_unbiased"]].agg(["mean","std"]).round(3))
```

### Cell 4: SHAP for every sentence, both models, with the efficiency check
```python
SV = {}
for m in MODELS:
    SV[m] = shap_explainer(m)(list(df.text), max_evals=500, batch_size=32)
    check_efficiency(SV[m], list(df.text), m)
print("SHAP base value (score of a fully masked sentence, log-odds):",
      {m: round(float(SV[m][0].base_values), 3) for m in MODELS})
```

### Cell 5: target-word share (the key measure)
The share of the positive (toward-toxic) SHAP attribution that lands on the insult words.
```python
def target_share(e, targets):
    w = shap_word_weights(e); t = set(targets.lower().split())
    pos = {k: v for k, v in w.items() if v > 0}
    return sum(v for k, v in pos.items() if k in t) / max(sum(pos.values()), 1e-9)

for m in MODELS:
    df[f"share_{m}"] = [target_share(SV[m][i], df.target_words[i]) for i in range(len(df))]
print(df.groupby("condition")[["share_original","share_unbiased"]].mean().round(3))
```

### Cell 6: LIME for every sentence (original model) and SHAP–LIME agreement
```python
rows, LIME_W = [], []
for i, text in enumerate(df.text):
    lw, exp = lime_word_weights(text, "original")
    LIME_W.append(lw)
    a = agreement(shap_word_weights(SV["original"][i]), lw)
    a["lime_fidelity"] = exp.score          # local fidelity R² of LIME's surrogate (W4-L02 p3)
    rows.append(a)
df = pd.concat([df, pd.DataFrame(rows)], axis=1)
print(df.groupby("condition")[["rho","top1_agree","top3_overlap","lime_fidelity"]].mean().round(3))
df.to_csv(f"{OUT}/exp1_results.csv", index=False)
```

### Cell 7: Figure 1, the scores chart (slide 6)
Grouped bar chart: x = condition in the order promote, report, counter; one bar per model; error bars = std.
```python
order = ["promote","report","counter"]
g = df.groupby("condition")[["p_original","p_unbiased"]]
mean, std = g.mean().loc[order], g.std().loc[order]
x = np.arange(len(order)); fig, ax = plt.subplots(figsize=(8,5))
for j, m in enumerate(["original","unbiased"]):
    ax.bar(x + (j-0.5)*0.38, mean[f"p_{m}"], 0.38, yerr=std[f"p_{m}"], capsize=4,
           color=COLORS[m], label=f"Detoxify {m}")
ax.set_xticks(x); ax.set_xticklabels(["Promotes abuse","Reports abuse","Argues against abuse"])
ax.set_ylabel("P(toxic)"); ax.set_ylim(0, 1); ax.legend(frameon=False)
ax.set_title("Same insult, different intent: still flagged?")
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
fig.savefig(f"{OUT}/fig_exp1_scores.png", dpi=200)
```

### Cell 8: Figure 2, target-word share (slide 6)
A strip plot (one dot per sentence, with jitter) plus a horizontal line at the mean for each condition. Original model only.
```python
fig, ax = plt.subplots(figsize=(8,5)); rng = np.random.RandomState(SEED)
for j, c in enumerate(order):
    vals = df.loc[df.condition == c, "share_original"]
    ax.scatter(j + rng.uniform(-0.12, 0.12, len(vals)), vals, color=COLORS["original"], alpha=0.7)
    ax.hlines(vals.mean(), j-0.25, j+0.25, color="black", lw=2.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["Promotes","Reports","Argues against"])
ax.set_ylabel("Share of 'toxic' push on insult words"); ax.set_ylim(0, 1.05)
ax.set_title("SHAP: how much of the decision is the insult itself?")
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
fig.savefig(f"{OUT}/fig_exp1_target_share.png", dpi=200)
```

### Cell 9: Figures 3 and 4, SHAP vs LIME highlights for the hero sentence (slides 5 and 6)
The hero sentence is **pair 1, the report version**. Draw it once with SHAP colours and once with LIME colours using `highlight_png`. Also save shap's own HTML version for backup.
```python
i = df.index[(df.pair_id == 1) & (df.condition == "report")][0]
text = df.text[i]
e = SV["original"][i]
shap_pairs = [(tok.strip(), float(v)) for tok, v in zip(e.data, e.values) if tok.strip()]
lime_pairs = [(w, LIME_W[i].get(w.lower(), 0.0)) for w in re.findall(r"\w+", text)]
highlight_png(shap_pairs, f"SHAP  (P(toxic) = {df.p_original[i]:.2f})", f"{OUT}/fig_exp1_hero_shap.png")
highlight_png(lime_pairs, "LIME", f"{OUT}/fig_exp1_hero_lime.png")
with open(f"{OUT}/hero_shap.html", "w") as fh:
    fh.write(shap.plots.text(e, display=False))
for c in order:   # also save SHAP highlights for all three versions of pair 1 (slide 6 option)
    k = df.index[(df.pair_id == 1) & (df.condition == c)][0]; ek = SV["original"][k]
    highlight_png([(t.strip(), float(v)) for t, v in zip(ek.data, ek.values) if t.strip()],
                  f"{c}  (P(toxic) = {df.p_original[k]:.2f})", f"{OUT}/fig_exp1_pair1_{c}.png")
```

### Cell 10: results summary (numbers to quote on the slides)
Print a clear block like this:
```python
s = df.groupby("condition").agg(p_orig=("p_original","mean"), p_unb=("p_unbiased","mean"),
                                share=("share_original","mean")).loc[order].round(2)
print("=== EXPERIMENT 1 SUMMARY ===")
print(s)
print(f"Report sentences scored > 0.5 toxic (original): {(df[df.condition=='report'].p_original>0.5).sum()}/16")
print(f"Counter sentences scored > 0.5 toxic (original): {(df[df.condition=='counter'].p_original>0.5).sum()}/16")
print(f"SHAP and LIME agreed on the TOP word in {int(df.top1_agree.sum())}/{len(df)} sentences")
print(f"Median SHAP-LIME rank correlation (Spearman rho): {np.nanmedian(df.rho):.2f}")
print(f"Mean top-3 overlap: {df.top3_overlap.mean():.2f}")
print(f"Median LIME local fidelity (R²): {df.lime_fidelity.median():.2f}")
s.to_csv(f"{OUT}/exp1_summary.csv")
```

### Cell 11: download everything
```python
!zip -qr outputs_exp1.zip outputs_exp1
from google.colab import files; files.download("outputs_exp1.zip")
```

### Final step for Claude Code
Print a short checklist reminding the user to:
1. Upload the notebook to Colab (File → Upload notebook) and select a T4 GPU.
2. Run all cells.
3. Unzip the outputs into `presentation/figures/exp1/` in the repo.
4. Commit the notebook and figures under their own GitHub account with the message `research: experiment 1 report vs promote results`.

---

## Part 6: Checks before you trust the results
- [ ] Cell 4 printed **"efficiency check passed"** for both models. If not, don't use the SHAP numbers. Raise `max_evals` to 1000 and re-run.
- [ ] Promote sentences score high (mostly > 0.5). If they don't, something is wrong with the model loading.
- [ ] Cell 2's length check: report sentences are longer. Mention this as a limitation, because longer sentences tend to get lower scores.
- [ ] Look at the hero PNGs. Are the words readable at slide size?
- [ ] Copy the Cell 10 summary into the journal entry.

## Part 7: What goes on the slides

**Slide 5 (SHAP vs LIME on text):** `fig_exp1_hero_shap.png` stacked above `fig_exp1_hero_lime.png`, with the two short comparison columns from the plan underneath.

**Slide 6 (Experiment 1):**
- **Left:** `fig_exp1_scores.png`
- **Right:** `fig_exp1_pair1_promote/report/counter.png` stacked, or `fig_exp1_target_share.png` if the pair images are too busy
- **Caption to fill in:** *"Reports still flagged: [X]/16. SHAP puts [Y]% of the 'toxic' push on the insult words, whatever the intent."*
- **"What the explanation added":** *The score shows **that** reports get flagged; SHAP shows **why**: the words, not the intent.*

**Slide 8 (Can we trust the explanations?):** the right-hand number: *"SHAP and LIME agreed on the top word in [N]/48 sentences"*. Add Experiment 2's agreement count if you want one combined number.

## Part 8: Limitations

**Limitations:**
- 48 hand-written sentences: illustrative, not representative of web pages
- Report sentences are longer than promote ones
- Quotation marks are invisible to the word tokenizer
- Detoxify is a stand-in, not the filter actually used on C4 or FineWeb
- SHAP and LIME "remove" words differently: SHAP masks a word and LIME deletes it, so they use different baselines. That's one reason they can disagree.


