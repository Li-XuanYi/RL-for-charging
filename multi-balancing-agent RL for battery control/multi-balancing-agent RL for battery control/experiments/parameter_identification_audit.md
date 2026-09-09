# Parameter-identification provenance audit

The original three fitted parameter dictionaries cannot be recovered from the
available repository snapshot.

Evidence:

- `data_true/parameter identification.py` prints `global_best_position` but
  does not set a random seed or save the optimum, objective history, or run
  metadata.
- The script fits only one selected voltage trace in its checked-in call, even
  though three voltage/temperature trace pairs are present.
- `data_true/main.py` contains one manually embedded six-parameter candidate
  labelled in the plot as Battery 3. It does not contain Battery 1 and Battery 2
  dictionaries or a PSO run identifier.
- The embedded Battery 3 negative active-material fraction is 0.560, whereas
  the PSO source bounds encode 0.60--0.65. Its positive particle radius is also
  3.98 micrometres, below the encoded 4--5 micrometre range. Therefore the
  hard-coded candidate cannot safely be presented as an output of the archived
  PSO program without author evidence.
- The archived PSO objective uses voltage only. Temperature traces exist, but
  the objective does not use them, so it does not establish cell-specific
  thermal-parameter identification.
- The archived script uses the old PyBaMM keys `Positive/Negative electrode
  diffusivity`. PyBaMM 26.8 uses `Positive/Negative particle diffusivity`;
  leaving the old names unchanged emits a warning and can leave the intended
  override inactive. The new loader explicitly migrates these two keys and
  records that migration in each run.

`identify_cell_parameters.py` provides a seeded, saving, voltage-only PSO
re-identification path. It deliberately requires explicit acknowledgement of
the inconsistent legacy bounds and writes to `cell_parameters.reidentified.json`.
That output is a new analysis, not recovery of the original parameters, and
does not automatically replace the confirmatory manifest's author-approved
`cell_parameters.identified.json` gate.
