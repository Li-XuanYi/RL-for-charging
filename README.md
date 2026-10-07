# MBA-RL Paper — IEEE T-ASE Format

LaTeX source for *Multi-Balancing-Agent Reinforcement Learning for Charging Control of Lithium-ion Batteries*.

## Structure

```text
.
|-- figures/          # Figures referenced by the manuscript
|-- sections/         # Section-level LaTeX sources
|-- IEEEtran.cls      # IEEE journal document class
|-- IEEEtran.bst      # IEEE BibTeX bibliography style
|-- main.tex          # Manuscript entry point
|-- main-authors.tex  # Same manuscript with author information enabled
|-- references.bib    # Bibliography database
`-- README.md
```

The manuscript uses the IEEE Transactions on Automation Science and Engineering (T-ASE) regular-paper layout: IEEEtran, 10-point type, US Letter paper, and two columns. Figure and table captions retain IEEEtran styling; subfigures use `subfig` with `caption=false`. References use the IEEEtran BibTeX style.

The first-page order is Abstract, Note to Practitioners, and Index Terms. The existing abstract has 135 words, within T-ASE's 200-word maximum for regular papers. The added Note to Practitioners has 173 words, within the required 100–300 words, and explains implementation requirements, the simulation-based validation, and deployment limitations.

`main.tex` produces the anonymous review version: author names, affiliations, email addresses, and ORCID links are omitted, and PDF author metadata is empty. The original author block remains in the source. `main-authors.tex` produces the version with that author block restored. Submit the compiled anonymous PDF for review, rather than the source containing author details.

Both versions render revision text in black. Set `\TASEshowrevisionstrue` in `main.tex` to display earlier revisions in blue again. Hyperlinks remain active without visible borders or link coloring.

## Compile

Run from the repository root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For the version with author information, run the same four commands with `main-authors.tex` and `bibtex main-authors` instead. Alternatively, use `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` or the corresponding command for `main-authors.tex`.

Both validated builds are 9 pages. Citations and cross-references resolve, and neither build has overfull boxes. The compiled manuscript PDFs and auxiliary files are intentionally excluded from version control. PDF files under figures/ are source figures required for compilation, not compiled manuscript outputs.

## Formatting Sources

Requirements checked on October 7, 2026:

- [T-ASE Information for Authors](https://www.ieee-ras.org/publications/t-ase/information-for-authors/)
- [IEEE RAS Rules for Double-Anonymous Review](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/)

AI tools assisted with this LaTeX format conversion and drafting the Note to Practitioners from the existing manuscript.
