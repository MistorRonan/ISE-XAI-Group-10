# Paper: how we write it

**Title:** Do Attributions Tell the Truth About Data Filters? Measuring the Faithfulness of SHAP and LIME in Identity-Term Bias Audits

**What goes in each section:** [plans/2026-10-18-paper-plan.md](../plans/2026-10-18-paper-plan.md). The same bullet points are also at the top of each section file as comments.

---

## Where things are

| File | What it is |
|---|---|
| `main.tex` | The IEEE two-column template. Title, authors, and it pulls in the section files. You shouldn't need to edit it. |
| `sections/*.tex` | **One file per section.** This is where the writing goes. |
| `refs.bib` | All references. Add yours here (BibTeX format). |
| `figures/` | Figures used in the paper, copied from `experiments/*/results/`. |
| `drafts/` | Dated PDF snapshots of the paper (see below). |

---

## How to write your section

1. Open your section file in `sections/` (in VS Code, or directly on github.com with the pencil icon).
2. Read the comments at the top of the file: they say what has to go in it.
3. Replace the pink `\todo{...}` line with your text.
4. **Commit it under your own GitHub account.** The journal is marked partly on who contributed what, and the commit history is how the lecturer checks that. Don't send your text to someone else to commit for you.
   - Commit message format: `report: draft method section A–C` or `report: add Exp 3 results to section IV-B`.
5. Push. GitHub builds the PDF automatically (see below). Check it built and looks right.

**Only edit your own section file** unless you've agreed otherwise. That way two people never edit the same lines and nothing gets overwritten.

---

## Getting the PDF

Every push that changes anything in `paper/` builds the PDF on GitHub. **Nobody needs LaTeX installed.**

1. Go to the repo on GitHub → **Actions** tab → **Build paper**.
2. Click the latest run (green tick = built fine, red cross = LaTeX error).
3. Scroll to **Artifacts** → download **paper-pdf** (it's a zip with `main.pdf` inside).

**If the build fails (red cross):** click the run → "Compile LaTeX" step → scroll to the first line starting with `!`. That's the error and the line number. The usual causes are:
- an unescaped `%`, `&`, `_` or `#` (write `\%`, `\&`, `\_`, `\#`);
- a `\cite{key}` whose key isn't in `refs.bib`;
- a missing `}`.

---

## Live preview in Overleaf (optional)

If you want to see the PDF update as you type:
1. Download the `paper/` folder as a zip (or zip it yourself).
2. On Overleaf: **New Project → Upload Project** → choose the zip.
3. Write there with live preview.
4. **When you're done, copy your section's text back into `sections/<your section>.tex` in the repo and commit it yourself.** The GitHub version is the official one. Anything that only exists in Overleaf doesn't count towards the journal.

(Overleaf's automatic GitHub sync is a paid feature, which is why we copy by hand.)

---

## Citing

- Cite with `\cite{key}`, e.g. `\cite{rudin2019stop}`. The keys are in `refs.bib`.
- **Adding a new source:**
  1. Add its BibTeX entry to `refs.bib`. Google Scholar → "Cite" → BibTeX gives you one; check it against the paper.
  2. Write a source review in `journal/sources/` using the [template](../journal/templates/source-review.md).
- IEEE numbered style ([1], [2], …) is generated automatically, in order of first citation. Don't type numbers by hand.
- `main.tex` currently has `\nocite{*}` so the reference list builds before anything is cited. **Delete it before submitting**, because the rubric checks that every listed reference is actually cited.

---

## Figures

1. Copy the figure PDF from `experiments/<exp>/results/` into `paper/figures/`.
2. Use the same file name.
3. The figure environments in `sections/results.tex` already point at the planned names: `fig_exp3_cf_vs_shap.pdf` and `fig_exp6_instability_vs_signal.pdf`. They show a placeholder box until the file exists.

---

## Draft snapshots (for the journal)

At each milestone (first full draft, after feedback, final), save the built PDF into `drafts/` as `draft-YYYY-MM-DD.pdf`, add a line to `drafts/README.md`, update the **Latest draft** link in the main README, and commit with `report: paper draft YYYY-MM-DD`. This is what lets the lecturer trace how the paper developed.

---

## Before submitting

- [ ] No pink `[TODO]` text left in the PDF.
- [ ] `\nocite{*}` removed from `main.tex`; every reference in the list is cited in the text.
- [ ] Within the page limit: 3 pages of body + references on their own page (confirm the limit with the lecturer).
- [ ] Every entry in `refs.bib` checked against the real paper (search for `VERIFY`).
- [ ] Every entry in `refs.bib` in the journal section of the repo.
- [ ] Every number in the paper traces to a results CSV in `experiments/*/results/`.
- [ ] Final PDF saved in `drafts/` and linked from the main README.
