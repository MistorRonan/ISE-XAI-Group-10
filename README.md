# ISE XAI Group 10

## Using Feature Attribution to Audit Social Bias in LLM Pretraining Data Filters

**Research question:** Can post-hoc feature attribution (SHAP, LIME) explain why a pretraining data filter removes text, reliably enough to support a social-bias audit?

**Topic:** Data filters silently decide whose language LLMs learn from. Pipelines built on Common Crawl remove "toxic" or "low-quality" text using blocklists, quality classifiers and toxicity classifiers. Each of these has been shown to fail specific groups: C4's blocklist removes LGBTQ+ and African American English text, quality filters favour text from wealthier, urban sources, and hate-speech detectors flag African American English as toxic. These filters are audited only in aggregate, and none explains an individual removal.

We test whether SHAP and LIME can fill that gap. They can show which words drive a removal, and our experiments show they reveal real problems in the filters we tested. But they are useful for *finding* suspicious behaviour, not for reaching a *verdict*: they are least reliable on subtle cases, and they are descriptive rather than normative (they describe what the model does, but can't say what it should do).

**Scope:** one filter family (Detoxify, `original` and `unbiased` models), short English template sentences, one word changed at a time. Results show what the filter reacts to, not real-world removal rates.

**Team:** Julia Hooper · Blaithin Kavanagh · Rory Linnane · April Gilhool



---

## Experiments

Both notebooks run on **Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU). Each folder has the spec, the notebook and the saved results.

| Experiment | Question | Spec | Notebook | Results |
|---|---|---|---|---|
| **1. Report vs promote** | Does the filter react to the insult words or to the intent? Same insult in sentences that promote, report, or argue against abuse. | [EXP1 spec](experiments/report%20vs%20promote/EXP1_report_vs_promote.md) | [01_report_vs_promote.ipynb](experiments/report%20vs%20promote/01_report_vs_promote.ipynb) | [results/](experiments/report%20vs%20promote/results/) |
| **2. Identity-term swap** | Does the group word alone push a harmless sentence toward removal? E.g. "My best friend is gay" vs "…is tall". | [EXP2 spec](experiments/identity%20term%20swap/EXP2_identity_swap.md) | [02_identity_swap.ipynb](experiments/identity%20term%20swap/02_identity_swap.ipynb) | [results/](experiments/identity%20term%20swap/results/) |


---

## Repo map

| Folder | What's in it |
|---|---|
| [journal/daily notes/](journal/daily%20notes/) | Source reviews (one per paper) and daily notes |
| [journal/weekly recaps/](journal/weekly%20recaps/) | Weekly summaries of decisions and sources |
| [team meetings/](team%20meetings/) | Meeting minutes and the [presentation slide plan](team%20meetings/2026-10-06-slide-plan.md) |
| [experiments/](experiments/) | Experiment specs, notebooks and results (see above) |
| [presentation/](presentation/) | Figures used in the slides |

### Templates

- **Source review:** [journal/daily notes/research_paper_template.md](journal/daily%20notes/research_paper_template.md). Use this for every paper we cite in the presentation or report.

---

## Note on commits

Commit messages follow the format `type: short summary`:

| Prefix | Used for | Example |
|---|---|---|
| `journal:` | a new or edited journal entry | `journal: week 2 entry, drop LIME paper and keep SHAP` |
| `report:` | changes to the report draft | `report: draft background section from week 2 sources` |
| `research:` | notes on sources or data outside the weekly entry | `research: add summary of counterfactual explanations paper` |
| `docs:` | README, template and repo housekeeping | `docs: add journal template` |
