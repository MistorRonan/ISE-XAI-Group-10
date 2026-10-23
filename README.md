# ISE XAI Group 10

## Using Feature Attribution to Audit Social Bias in LLM Pretraining Data Filters

**Research question:** Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?

**Topic:** Data filters silently decide whose language LLMs learn from. Pipelines built on Common Crawl remove "toxic" or "low-quality" text using blocklists, quality classifiers and toxicity classifiers. Each of these has been shown to fail specific groups: C4's blocklist removes LGBTQ+ and African American English text, quality filters favour text from wealthier, urban sources, and hate-speech detectors flag African American English as toxic. These filters are audited only in aggregate, and none explains an individual removal.

We test whether SHAP and LIME can fill that gap. They can show which words drive a removal, and our experiments show they reveal real problems in the filters we tested. But they are useful for *finding* suspicious behaviour, not for reaching a *verdict*: they are least reliable on subtle cases, and they are descriptive rather than normative (they describe what the model does, but can't say what it should do).

**Scope:** one filter family (Detoxify, `original` and `unbiased` models), short English template sentences, one word changed at a time. Results show what the filter reacts to, not real-world removal rates.

**Team:** Julia Hooper · Blaithin Kavanagh · Rory Linnane · April Gilhool

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

- **Latest draft:** [paper/main.tex](paper/main.tex), with each section in [paper/sections/](paper/sections/). Dated PDF snapshots go in [paper/drafts/](paper/drafts/).
- **Live build:** every push to `paper/` builds the PDF. Go to the Actions tab → *Build paper* → latest run → *paper-pdf*.
- **LaTeX source and how to contribute:** [paper/](paper/). Read [paper/README.md](paper/README.md) first.


---

## Experiments

All notebooks run on **Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU). Each folder has the spec, the notebook and the saved results. Exp 1 and 2 were for the presentation; Exp 3–6 are for the paper.

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
| [paper/](paper/) | LaTeX source of the paper (one file per section), references, figures and dated draft PDFs |

---

## Note on commits

Commit messages follow the format `type: short summary`:

| Prefix | Used for | Example |
|---|---|---|
| `journal:` | a new or edited journal entry | `journal: week 2 entry, drop LIME paper and keep SHAP` |
| `report:` | changes to the report draft | `report: draft background section from week 2 sources` |
| `research:` | notes on sources or data outside the weekly entry | `research: add summary of counterfactual explanations paper` |
| `docs:` | README, template and repo housekeeping | `docs: add journal template` |

