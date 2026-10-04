# MBA-RL Paper

LaTeX source for *Multi-Balancing-Agent Reinforcement Learning for Charging Control of Lithium-ion Batteries*.

## Structure

```text
.
|-- figures/          # Figures referenced by the manuscript
|-- sections/         # Section-level LaTeX sources
|-- IEEEtran.cls      # IEEE journal document class
|-- main.tex          # Manuscript entry point
|-- references.bib    # Bibliography database
`-- README.md
```

The review version uses black for the original text, blue for earlier revisions, and red for the October 4, 2026 revision.

## Compile

Run from the repository root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The validated review build is 9 pages. Generated PDF and auxiliary files are intentionally excluded from version control.
