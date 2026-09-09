# Revision Tracking Record

## Paper Information

| Field | Value |
|---|---|
| Paper title | Multi-Balancing-Agent Reinforcement Learning for Charging Control of Lithium-ion Batteries |
| Revision round | 1 (continued text-level revision) |
| Date | 2026-08-29 |
| Previous decision | Major Revision |
| Manuscript source | `MARL charging balance.tex` |

## Revision Tracking Table

| # | Issue | Severity | Resolution summary | Location | Status | Remaining requirement |
|---|---|---|---|---|---|---|
| M1/DA1 | Discharge-elimination claim contradicted by code | Critical | Removed the claim; documented the -2.0 to 7.0 A grid and negative-current fallback | Abstract; Introduction; Action Space; Performance Evaluation; Conclusion | RESOLVED | Rerun only if the authors want to restore a non-negative-action claim |
| M2 | Released reward implementation is incomplete/non-runnable | Critical | Added pure reward/terminal functions, unit tests, PyBaMM step return, and complete pack reward | Code; supplementary protocol; response letter | IMPLEMENTED_NOT_VERIFIED | Execute clean-environment reproduction and replace figures if required |
| M3 | Episode count, target-update period, initialization, and implementation differ | Major | Defaults now match 1200/50; periodic target update, SOC-consistent confirmatory initialization, and legacy provenance schedule implemented | Configuration; environment; policy; experiment manifest | IMPLEMENTED_PROVENANCE_PENDING | Run paper-aligned and legacy diagnostics; replace figures if provenance fails |
| M4 | SPM/SPMe and thermal-model mismatch | Major | Changed residual SPM references to SPMe; retained lumped thermal ODE | Sections II and III; parameter identification | RESOLVED | Verify equations against the exact PyBaMM version |
| M5 | Chen2020 metadata/model-origin mismatch and missing cell-specific parameter archive | Major | Clarified template/fitting distinction; added a validated loader with PyBaMM key migration, provenance audit, template, and explicitly new voltage-only re-identification utility | Parameter identification; code; supplement; `conf.bib` | TEXT_RESOLVED_CODE_BLOCKING | Recover/approve three parameter dictionaries or narrow the identified-aging claim |
| M6 | No statistical repeats | Major | Frozen five training seeds, blocked run order, trace schema, and bootstrap CI aggregation | Supplementary protocol; experiment manifest; analysis script | EXPERIMENT_DESIGNED | Execute all independent runs and insert results |
| D1 | Absolute MARL novelty claim | Major | Replaced with a bounded comparison to cited swapping/V2G studies | Introduction | RESOLVED | A systematic novelty search would be needed for an absolute claim |
| D2/D3/DA3/DA4 | Missing matched baselines and hardware/algorithm confound | Major | Added matched VDN, centralized recurrent DQL, proportional, and per-cell CC–CV implementations under the same current interface | Code; supplement; response letter | IMPLEMENTED_EXECUTION_BLOCKING | Execute all methods after cell-parameter provenance is resolved |
| D4/D5/DA5 | Inconsistent stopping rules; sustained balance is design-dependent | Major | Both time-to-balance and time-to-reference-SOC are computed from every trace | Code; supplementary protocol | METRICS_IMPLEMENTED_RUNS_PENDING | Execute and report both outcomes for all methods |
| P1/DA2 | Hardware-validation overclaim | Critical | Explicitly separated NEWARE parameter-identification tests from PyBaMM policy simulation | Abstract; Results; Limitations; Conclusion | RESOLVED | Hardware-in-the-loop work remains future validation |
| P2/DA6 | Scalability not demonstrated | Major | Restricted claims to three-cell cases; documented why copying identical cells is not a valid larger-n experiment | Abstract; Limitations; supplementary protocol | DESIGN_PAUSED_AUTHOR_DECISION | Approve additional-cell identification or a bounded virtual-cell distribution |
| P3 | Per-cell source feasibility | Major | Added isolated-converter cost and deployment boundary | Limitations | RESOLVED IN TEXT | Hardware design/efficiency study remains future work |
| P4 | $T_{max}=309$ K lacks engineering justification | Major | Frozen local sensitivity at 306/309/312 K and retained requirement for an engineering source | Supplementary protocol; manifest; response letter | EXPERIMENT_DESIGNED | Execute sensitivity and add battery-specific rationale |
| P5/DA7 | No quantitative energy efficiency | Minor | Removed energy-efficiency claims | Abstract; Introduction; Results; Conclusion | CLAIM WITHDRAWN | Report $\eta$ and Ah-throughput before restoring the claim |
| FMT1 | IEEE caption-package warning | Editorial | Replaced `caption`/`subcaption` with IEEE-compatible `subfig` setup | Preamble and all composite figures | RESOLVED | Confirmed by compilation |
| CIT1 | Duplicate bibliography record | Editorial | Reused citation key `10` and removed duplicate Ghaeminezhad entry | Results; `conf.bib` | RESOLVED | None |
| CIT2 | Inaccurate/incomplete metadata | Editorial | Corrected verified metadata and added selected DOIs | `conf.bib` | PARTLY RESOLVED | Complete DOI audit for all remaining records |

## Round Summary

| Metric | Count |
|---|---:|
| Resolved or resolved in text | 8 |
| Implemented but awaiting validated execution | 2 |
| Experiment designed or partly implemented | 4 |
| Deliberate limitations / claims withdrawn | 2 |
| Partial citation audit | 1 |
| New references added | 0 |
| Duplicate references removed | 1 |

## Resubmission Gate

The manuscript is not ready for resubmission. M2 and M3 have implementation-level fixes that pass compilation, unit tests, and short end-to-end smoke tests in a locked local environment, but still require formal provenance closure. M5 now blocks confirmatory execution because the three cell-specific PSO parameter dictionaries are not archived in the controller environment. M6, D2/D3, D4, and P4 have frozen designs and implementations but no formal numerical results. P2 requires an author decision on virtual-cell generation or continued restriction to three cells. P5 remains closed by claim withdrawal unless a quantitative efficiency claim is restored.

## New Experiment Artifacts

- supplement/supplementary_experiment_protocol.md: frozen design, estimands, replicate definition, and submission gates.
- supplement/supplementary_experiments.tex: manuscript-ready section skeleton with explicit result placeholders.
- experiments/build_manifest.py: seeded, randomized schedule covering confirmatory, ablation, threshold, and provenance runs.
- experiments/analyze_results.py: independent-run aggregation with mean, SD, and bootstrap 95% CI.
- tests/test_reward.py: unit tests for the manuscript reward and joint terminal equations.
