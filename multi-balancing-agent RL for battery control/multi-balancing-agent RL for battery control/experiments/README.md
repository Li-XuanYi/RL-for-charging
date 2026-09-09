# Supplementary experiment runner

This directory implements the first executable layer of the revision plan. It
does not contain fabricated results. A run is reportable only after its
`summary.json`, `evaluation_trace.json`, training history, software versions,
and exact command have been archived.

## Implemented

- Paper-aligned QMIX configuration: 1200 episodes and target updates every 50
  training episodes.
- Matched VDN mixer using the same recurrent agents, observations, action grid,
  reward, safety mask, and stopping rule.
- A centralized recurrent DQL baseline that observes the nine-dimensional pack
  state and selects the Cartesian product of the three per-cell current actions
  under the same action masks, reward, decision interval, and stopping rule.
- Proportional-feedback and per-cell CC-CV rule baselines using the same
  independently controlled current-source interface.
- Reward-component ablations and temperature-threshold sensitivity arguments.
- Seeded, randomized run schedule and bootstrap confidence-interval aggregation.

## Still blocking

- The controller environment does not contain three archived PSO-identified
  parameter dictionaries; all three cells otherwise inherit Chen2020. Populate
  `cell_parameters.identified.json` from the template before changing any
  confirmatory row to `ready`.
  See `parameter_identification_audit.md`. A seeded re-identification utility is
  provided, but its output is explicitly labelled as a new voltage-only fit and
  requires author approval of the inconsistent source-coded bounds.
- The larger-`n` experiment is not runnable because only three identified cell
  models are currently defined. A scientifically defensible virtual-cell
  generation rule must be approved before extending the environment.
- The original environment has no archived package versions. Old figures cannot
  be called reproduced until the provenance runs identify a matching setting.

## Usage

Use Python 3.10--3.12. The current system-default Python 3.14 environment does
not contain PyTorch or PyBaMM.

1. Generate the frozen schedule:

   `python experiments/build_manifest.py`

2. Execute only rows whose `implementation_status` is `ready`, following
   `run_order`.

   At the current checkpoint, only legacy provenance rows are ready. They are
   diagnostic runs and must not be used for controller-performance claims.
   The generator also writes these rows, in randomized order, to
   `commands_ready.ps1`; inspect that script before launching it.

   For logged sequential execution, first inspect the next command with
   `python experiments/run_manifest.py`, then add `--execute`. The default is
   one run; use `--run-id` or `--max-runs` to expand the scope deliberately.

3. Aggregate completed summaries:

   `python experiments/analyze_results.py --results-root results`

The independent replicate is the training seed for learned methods. For
deterministic baselines, repeated seeds are explicitly treated as paired
environment blocks, not as independent training replicates.
