# Paper maintenance

The `paper` branch contains the maintained manuscript and its supporting assets.

## Main files

- LaTeX source: `Multi_Balancing_Agent_Reinforcement_Learning_for_Charging_Control_of_Lithium_ion_Batteries/MARL charging balance.tex`
- Bibliography: `Multi_Balancing_Agent_Reinforcement_Learning_for_Charging_Control_of_Lithium_ion_Batteries/conf.bib`
- Chinese revision report: `MBA-RL论文修改说明与中英对照_20261003.docx`

The manuscript currently uses black for original text, blue for earlier revisions, and red for the latest revision. Remove the color macros only when preparing the final all-black submission version.

## Compile

Run the following commands from the manuscript directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error "MARL charging balance.tex"
bibtex "MARL charging balance"
pdflatex -interaction=nonstopmode -halt-on-error "MARL charging balance.tex"
pdflatex -interaction=nonstopmode -halt-on-error "MARL charging balance.tex"
```

The validated October 3, 2026 review build is 10 pages. Compiled PDFs are not tracked; generate them locally from the maintained source.

## Figures and reports

- `tools/build_manuscript_figures.py` regenerates the added comparison and ablation figures.
- `evidence/figures/` stores review images and provenance metadata for those figures.
- `tools/build_change_report.py` regenerates the Chinese revision report.

Keep generated LaTeX auxiliary files (`.aux`, `.bbl`, `.blg`, `.log`, `.fls`, and `.fdb_latexmk`) out of commits.
