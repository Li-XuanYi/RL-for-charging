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

The first-page order is Abstract, Note to Practitioners, and Index Terms. The existing abstract has 135 words, within T-ASE's 200-word maximum for regular papers. The current Note to Practitioners has 179 words, within the required 100–300 words, and explains practical deployment conditions and future hardware and larger-pack validation.

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

Both validated builds are 10 pages after the TASE citation update on October 8, 2026. All 38 cited references and cross-references resolve, and neither build has overfull boxes. The compiled manuscript PDFs and auxiliary files are intentionally excluded from version control. PDF files under figures/ are source figures required for compilation, not compiled manuscript outputs.

## TASE Citation Update

Four TTE papers were replaced with four related TASE papers. Two additional TASE papers on calendar/cycling aging coupling and SOC/SOE co-estimation were added to the general background in the Introduction. The six new records include verified DOI, authors, formal publication year, volume, and pages. See [TASE citation audit](TASE_citation_audit.md) for the replacement mapping, citation contexts, and verification scope, and [verified metadata](TASE_verified_metadata.json) for the publication records retrieved from Crossref.

## Formatting Sources

Requirements checked on October 7, 2026:

- [T-ASE Information for Authors](https://www.ieee-ras.org/publications/t-ase/information-for-authors/)
- [IEEE RAS Rules for Double-Anonymous Review](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/)

AI tools assisted with this LaTeX format conversion and drafting the Note to Practitioners from the existing manuscript.
