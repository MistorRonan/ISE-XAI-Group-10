# ISE XAI Group 10

## Using Feature Attribution to Audit Social Bias in LLM Pretraining Data Filters

**Research question:** Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?

**Topic:** Data filters silently decide whose language LLMs learn from. Pipelines built on Common Crawl remove "toxic" or "low-quality" text using blocklists, quality classifiers and toxicity classifiers. Each of these has been shown to fail specific groups: C4's blocklist removes LGBTQ+ and African American English text, quality filters favour text from wealthier, urban sources, and hate-speech detectors flag African American English as toxic. These filters are audited only in aggregate, and none explains an individual removal.

We test whether SHAP and LIME can fill that gap. They can show which words drive a removal, and our experiments show they reveal real problems in the filters we tested. But they are useful for *finding* suspicious behaviour, not for reaching a *verdict*: they are least reliable on subtle cases, and they are descriptive rather than normative (they describe what the model does, but can't say what it should do).

**Scope:** one filter family (Detoxify, `original` and `unbiased` models), plus a transparent glass-box filter we built for Exp 4; short English template sentences, one word changed at a time. Results show what the filter reacts to, not real-world removal rates.

**Team:** Julia Hooper · Bláithín Kavanagh · Rory Linnane · April Gilhool

---

## How the journal is organised

Every entry is a dated Markdown file named `YYYY-MM-DD-….md`. Entries are grouped by type:

| Folder | Entry type | What's in each entry |
|---|---|---|
| [journal/weekly/](journal/weekly/) | **Weekly recap** | What we decided that week and why, the sources we looked at with a verdict for each, what's still open, and who did what |
| [journal/meetings/](journal/meetings/) | **Meeting minutes** | Agenda, discussion, decisions and action items |
| [journal/sources/](journal/sources/) | **Source review**, one per paper | Full reference, TL;DR, key findings, why it matters to us, concerns, and the report sections it feeds |
| [journal/notes/](journal/notes/) | **Ad-hoc note** | Early ideas that came before a weekly entry |
| [journal/templates/](journal/templates/) | Templates | [Source review](journal/templates/source-review.md) |
| [plans/](plans/) | **Plans agreed at meetings** | The [presentation plan](plans/2026-10-06-presentation-plan.md) (06/10) and the [paper plan](plans/2026-10-18-paper-plan.md) (18/10), updated as decisions change |

---

## Paper

**Do Attributions Tell the Truth About Data Filters? Measuring the Faithfulness of SHAP and LIME in Identity-Term Bias Audits**

**Thesis:** feature attribution is faithful enough to flag identity-term bias in a pretraining filter, but not reliable enough to certify the filter is fair.

**Findings:**
- Both explainers rank identity-word effects almost exactly as counterfactual swaps do (Spearman ρ ≥ 0.92 for SHAP, 0.97 for LIME) and recover exact Shapley values on the glass-box filter (ρ ≥ 0.97).
- The words they blame matter: deleting them drops the score far more than deleting random words (comprehensiveness 3.09 vs 0.35).
- But SHAP overstates the size of the effect, and its magnitudes shift with the removal baseline.
- On weak signals LIME's surrogate fit falls (R² 0.77 → 0.41), and SHAP–LIME disagreement is no better than chance at flagging wrong explanations (AUC 0.46).

**Recommendations for auditors:** use attribution to triage a filter, not to certify it as fair. (1) Treat weak signal strength or low LIME R² as a warning that an explanation may be wrong. (2) Don't treat SHAP–LIME agreement or seed stability as validation. (3) Confirm any attribution finding with a counterfactual swap.

- **Final paper (PDF):** [paper/drafts/draft-2026-10-23.pdf](paper/drafts/draft-2026-10-23.pdf). Earlier snapshots are listed in [paper/drafts/](paper/drafts/).
- **LaTeX source:** [paper/main.tex](paper/main.tex), with each section in [paper/sections/](paper/sections/) and figures in [paper/figures/](paper/figures/).
- **Live build:** every push to `paper/` builds the PDF. Go to the Actions tab → *Build paper* → latest run → *Artifacts* → *paper-pdf*.
- **How to contribute:** read [paper/README.md](paper/README.md) first.


---

## Experiments

All notebooks run on **Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU). Each folder has the spec, the notebook and the saved results. Exp 1–2 began as presentation experiments; their sentences are reused across Exp 3–6, and all six feed the paper.

**Running a notebook:**
- **Exp 4 and 5:** upload `exp1_sentences.csv` from [experiments/report vs promote/data/](experiments/report%20vs%20promote/data/) to the Colab session before running. Both rebuild the Exp 2 sentences with EXP2 Cell 2's code.
- **Saving results:** the last cell of each notebook downloads its outputs as `outputs_expN.zip`. Unzip it into that experiment's `results/` folder and commit it. Exp 4 and 5 keep the unzipped `outputs_exp4/` and `outputs_exp5/` folders inside `results/`.

| Experiment | Question | Spec | Notebook | Results |
|---|---|---|---|---|
| **1. Report vs promote** | Does the filter react to the insult words or to the intent? Same insult in sentences that promote, report, or argue against abuse. | [EXP1 spec](experiments/report%20vs%20promote/EXP1_report_vs_promote.md) | [01_report_vs_promote.ipynb](experiments/report%20vs%20promote/01_report_vs_promote.ipynb) | [results/](experiments/report%20vs%20promote/results/) |
| **2. Identity-term swap** | Does the group word alone push a harmless sentence toward removal? E.g. "My best friend is gay" vs "…is tall". | [EXP2 spec](experiments/identity%20term%20swap/EXP2_identity_swap.md) | [02_identity_swap.ipynb](experiments/identity%20term%20swap/02_identity_swap.ipynb) | [results/](experiments/identity%20term%20swap/results/) |
| **3. Counterfactual swap** | Does SHAP's number for the identity word match what swapping that word actually does to the score? Plus: does the choice of removal baseline matter? | [EXP3 spec](experiments/counterfactual%20swap/EXP3_counterfactual_swap.md) | [03_counterfactual_swap.ipynb](experiments/counterfactual%20swap/03_counterfactual_swap.ipynb) | [results/](experiments/counterfactual%20swap/results/) |
| **4. Glass-box filter** | On a transparent filter whose exact Shapley values we can compute, do SHAP and LIME recover them? | [EXP4 spec](experiments/glass%20box%20filter/EXP4_glass_box.md) | [04_glass_box.ipynb](experiments/glass%20box%20filter/04_glass_box.ipynb) | [results/](experiments/glass%20box%20filter/results/) |
| **5. Deletion test** | If we delete the words SHAP/LIME blame, does the score drop more than when deleting random words? | [EXP5 spec](experiments/deletion%20test/EXP5_deletion.md) | [05_deletion.ipynb](experiments/deletion%20test/05_deletion.ipynb) | [results/](experiments/deletion%20test/results/) |
| **6. LIME stability** | Is LIME least stable on weak signals, and can its R² or SHAP–LIME disagreement warn an auditor? | [EXP6 spec](experiments/lime%20stability/EXP6_lime_stability.md) | [06_lime_stability.ipynb](experiments/lime%20stability/06_lime_stability.ipynb) | [results/](experiments/lime%20stability/results/) |

---

## Repo map

| Folder | What's in it |
|---|---|
| [journal/](journal/) | The research journal (see [How the journal is organised](#how-the-journal-is-organised)) |
| [plans/](plans/) | Presentation and paper plans agreed at meetings |
| [experiments/](experiments/) | Experiment specs, notebooks and results |
| [paper/](paper/) | LaTeX source of the paper (one file per section), references, [figures](paper/figures/) (copied from the experiment results) and dated draft PDFs |
| [.github/workflows/](.github/workflows/) | The GitHub Actions workflow that builds the paper PDF on every push to `paper/` |

---

## Note on commits

Commit messages follow the format `type: short summary`:

| Prefix | Used for | Example |
|---|---|---|
| `journal:` | a new or edited journal entry | `journal: week 2 entry, drop LIME paper and keep SHAP` |
| `report:` | changes to the report draft | `report: draft background section from week 2 sources` |
| `research:` | notes on sources or data outside the weekly entry | `research: add summary of counterfactual explanations paper` |
| `docs:` | README, template and repo housekeeping | `docs: add journal template` |

