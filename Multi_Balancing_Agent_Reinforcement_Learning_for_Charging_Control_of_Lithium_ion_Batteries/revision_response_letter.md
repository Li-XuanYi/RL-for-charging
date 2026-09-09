# Response to Reviewers — Point-by-Point Revision Letter

**Manuscript:** Multi-Balancing-Agent Reinforcement Learning for Charging Control of Lithium-ion Batteries  
**Authors:** Hejiakang Cao, Bing-Chuan Wang, Biao Luo, Zhongmei Li  
**Decision:** Major Revision  
**Response Date:** 2026-08-29 (working draft; not submission-ready)

---

## Overview

We thank the editor and reviewers for their thorough and constructive evaluation. This document is a working revision record. Text-level overclaims and method-description inconsistencies have been corrected. The first implementation pass now includes the complete reward, paper-aligned target-update logic, a matched VDN mixer, a centralized matched DQL controller, two rule-based baselines, and a frozen multi-seed experiment schedule. Compilation, unit tests, and short end-to-end smoke tests pass in a locked local PyTorch/PyBaMM environment; formal multi-seed and provenance runs have not yet been completed, and the larger-n environment remains incomplete. The response must not be submitted until every promised experiment has either been completed and reported or explicitly withdrawn with an editor-approved limitation statement.

**Summary of verified manuscript changes:**

1. Removed the claim that MBA-RL eliminates discharge actions. The manuscript now reports the released action grid and state-dependent mask, including the negative-current fallback.
2. Separated the per-cell current-source architecture from the QMIX/MARL contribution; no topology or efficiency benefit is attributed to the learning algorithm alone.
3. Distinguished experimental parameter-identification data from simulation-only policy evaluation and stated that the learned controller was not executed on the NEWARE tester.
4. Aligned the model description (SPM→SPMe, distributed thermal→lumped thermal), corrected the target equation terminal mask, and clarified the joint terminal condition.
5. Added a Limitations and Engineering Scope subsection covering n=3, single-run results, unmatched baselines, stopping-rule differences, hardware cost, and lack of hardware validation.
6. Replaced the IEEE-incompatible caption setup, consolidated a duplicate bibliography entry, and corrected verified metadata for Chen2020, PyBaMM, and selected IEEE references.

**Current blockers:**

| Blocker | Status | Required closure |
|---|---|---|
| M2 public reward implementation | IMPLEMENTED; RERUN BLOCKING | Validate the clean environment and reproduce or replace every reported trajectory |
| M3 hyperparameters/target updates | IMPLEMENTED; PROVENANCE BLOCKING | Execute paper-aligned and legacy provenance runs; synchronize Table II with archived results |
| M5 identified parameter sets | CODE-PROVENANCE BLOCKING | Recover and archive three cell-specific parameter dictionaries or narrow the manuscript claim |
| M6 statistical repeats | SCHEDULE FROZEN; RUNS PENDING | Complete five independent training seeds and report uncertainty intervals |
| D2/D3 matched baselines | IMPLEMENTED; EXECUTION BLOCKING | VDN, centralized DQL, and rule baselines pass implementation-level tests; formal runs await cell parameters |
| D4 stopping rules | METRICS IMPLEMENTED; RUNS PENDING | Report time-to-balance and time-to-reference-SOC for every method |
| M8/P2/P4/P5 | MIXED | M8/P4 are scheduled; P2 awaits a defensible virtual-cell rule; P5 remains claim-withdrawn |

---

## Reviewer 1 — Methodology

### M1 [CRITICAL] — "Eliminates discharging actions" claim contradicted by code — RESOLVED IN TEXT

**Reviewer's concern:** The code `actions = np.arange(-2, 7.5, 0.5)` includes negative currents; `SPM.py` explicitly allows discharge when SOC>0.95, contradicting the paper's core claim.

**Response:** We agree. We removed every claim that discharge actions are eliminated. The revised manuscript now states that the released implementation uses a discrete grid from -2.0 to 7.0 A. Its action mask admits non-negative commands under normal operating states but retains negative commands near the upper-SOC or safety limits. The contribution is therefore limited to coordinated current control without a separate equalization network under the assumed per-cell current-source architecture. No energy-loss advantage is claimed without measured energy-throughput data.

---

### M2 [CRITICAL] — Reward function implementation incomplete — IMPLEMENTED; VALIDATION BLOCKING

**Reviewer's concern:** `SPM.step()` returns None; `multi_step()` only computes $r_{bal}$, missing $r_{time}$ and $r_{safety}$.

**Response:** We acknowledge the gap between the manuscript and the released code. A pure, unit-testable implementation of the reported reward and terminal equations has now been added, and the PyBaMM environment returns observations and evaluates the complete pack reward. Unit and short end-to-end smoke tests pass, but this item remains blocking until the formal provenance runs are completed and all reported trajectories are reproduced or replaced. Before resubmission we will:

1. Completed in code: include $r_{time} = -0.75$, $r_{bal}$, $r_{volt}$, and $r_{temp}$ as described in Eqs. 9–13.
2. Pending execution: re-verify that the published results are reproducible with the complete reward implementation.
3. Add a supplementary note documenting the normalization constants used in `normalize_outputs`.

---

### M3 [MAJOR] — Hyperparameter inconsistencies — IMPLEMENTED; PROVENANCE BLOCKING

**Reviewer's concern:** Paper Table II says $N_{episode}=1200$, code says 800; $N_t=50$ in paper vs 200 in code; target network hard-copied every step.

**Response:** The code defaults now match Table II (1200 episodes and target updates every 50 training episodes). A seeded provenance schedule also retains 800/200 and 800/every-step diagnostic runs. It preserves the legacy mixed voltage/SOC initialization only for source investigation, while confirmatory runs use SOC-consistent PyBaMM initialization. We will reconcile the reported figures as follows:

1. Pending execution: compare the paper-aligned and legacy configurations and identify whether any setting reproduces the published figures.
2. Completed in code: implement periodic target updates every $N_t$ training episodes instead of per-step hard copy.
3. Add missing hyperparameters to Table II: $\gamma=0.99$, buffer size, gradient clip norm, DRQN hidden dimension (128), QMIX hidden dimension (256), optimizer (RMSprop).

---

### M4 [MAJOR] — Model description vs. implementation (SPM vs. SPMe, thermal)

**Reviewer's concern:** Paper describes SPM but code uses SPMe; thermal model is PDE but code uses lumped.

**Response:** ✅ **Addressed in revision.**  
1. §II.A: Changed all references from "SPM" to "SPMe" (Single Particle Model with electrolyte) and updated the description to reflect electrolyte dynamics.
2. §II.A Thermal Model: Replaced the distributed PDE formulation with the lumped-parameter ODE that matches the `"thermal": "lumped"` implementation, including convective heat transfer term $hA_s(T_{amb}-T)$.

---

### M5 [MAJOR] — Battery parameter set mismatch (Samsung 40T vs. Chen2020)

**Reviewer's concern:** Samsung 40T (NCA) but code defaults to Chen2020 (LG M50 NMC).

**Response:** **Addressed in text; code provenance remains blocking.**  
Section IV.A explains that Chen2020 is the structural template and that PSO is intended to fit cell-specific parameters. However, the controller environment currently instantiates all three cells from Chen2020, and no three machine-readable PSO optimum dictionaries were found. The archived PSO objective uses voltage only, does not save its unseeded optimum, and its encoded bounds conflict with the single Battery~3 candidate hard-coded in a plotting script. Moreover, the archived diffusivity keys were renamed in current PyBaMM; the new loader explicitly migrates those keys and records the migration. A validated loader, template, provenance audit, and deterministic voltage-only re-identification utility have been added. The latter is explicitly a new analysis, not recovery of the original optima. Before confirmatory runs, we will either recover and archive the author-approved Battery1--Battery3 dictionaries, including a file hash, or approve and report a new identification protocol. If neither is possible, we will relabel the evaluation as a shared-parameter Chen2020 simulation and withdraw the claim that controller robustness was tested using three aging-specific identified models.

---

### M6 [MAJOR] — Lack of statistical repeats

**Reviewer's concern:** RL results from single run; no variance/error bars; no convergence curves.

**Response:** We agree this is a significant gap.

1. The five training seeds (7, 17, 29, 43, and 71), SOC-set blocks, and randomized run order have been frozen.
2. We will report mean, standard deviation, and bootstrap 95% confidence intervals for both completion-time outcomes and safety metrics.
3. We will include reward convergence curves in a supplementary figure.

---

### M7 [MINOR] — Undocumented normalization constants

**Response:** The normalization equations and constants have been drafted in the supplementary methods section and will be included with the completed experimental results.

### M8 [MINOR] — No ablation/sensitivity analysis

**Response:** The first frozen robustness stage contains reward-component ablations and $T_{max}\in\{306,309,312\}$ K. Broader $\beta$ and decision-cycle sensitivity will be added only after the confirmatory runs pass the reproducibility gate.

---

## Reviewer 2 — Domain

### D1 [MAJOR] — Overclaim "MARL has not yet been explored" — RESOLVED IN TEXT

**Response:** ✅ **Addressed in revision.**  
The absolute novelty claim has been removed. The revised introduction instead states what the cited MARL studies address (battery swapping/charging-system operation and V2G scheduling) and explains why they motivate, but do not establish, cell-level coordinated charging control.

---

### D2 [MAJOR] — Insufficient baselines

**Response:** A parameter-free VDN mixer, a centralized recurrent DQL controller, and proportional-feedback/per-cell CC–CV baselines have been implemented under the same per-cell current interface. DQL observes the nine-dimensional joint state and selects all three cell-current commands from the Cartesian joint-action set; it therefore removes the previous hardware-interface confound while retaining a single-agent value-learning baseline. Unit and short end-to-end tests pass. Their commands are frozen in the supplementary manifest, but formal comparison remains blocked by the missing cell-specific parameter archive.

---

### D3 [MAJOR] — Unfair comparison (hardware vs. algorithm confound) — PARTLY RESOLVED

**Response:** The manuscript now explicitly attributes topology removal to the per-cell current-source architecture and coordination to MBA-RL. It also labels the present comparison as confounded and withdraws the quantitative energy-efficiency claim. The experimental component remains unresolved: DQL and the rule-based baseline must still be run under the same per-cell current interface.

---

### D4 [MAJOR] — Inconsistent termination conditions

**Response:** Both outcomes are now computed from every archived evaluation trace: (a) first attainment of $\sigma\leq\beta$ and (b) first attainment of $SOC_i\geq SOC_{ref}$ for all cells. The joint terminal condition is unchanged. Numerical closure awaits execution of all matched methods.

---

### D5 [MAJOR] — "Prevents re-emerging imbalance" is a stopping-rule artifact

**Response:** ✅ **Addressed in revision.**  
We have reworded this claim to acknowledge that sustained balance is achieved by the design choice of continuing to charge under the balancing constraint until $SOC_{ref}$, rather than being an intrinsic algorithmic property. The revised text in §IV.B now states: "MBA-RL integrates balancing and charging into a unified process that continues under the balancing constraint until the reference SOC is reached, thereby maintaining the balanced state at termination."

---

### D6 [MINOR] — Missing key references

**Response:** ⚡ We will add recent (2024–2025) references on MARL-based charging/equalization and safety-constrained RL charging.

### D7 [MINOR] — Reference [22] description mismatch

**Response:** ✅ **Addressed in revision.**  
Corrected the in-text description to match the actual paper title: "proposed an MARL mechanism for intelligent vehicle-to-grid (V2G) integration in future transportation systems."

---

## Reviewer 3 — Perspective

### P1 [CRITICAL] — "Real battery validation" overclaim

**Response:** ✅ **Addressed in revision.**  
All instances of "validated MBA-RL on real battery data/equipment" have been corrected to "validated MBA-RL through simulation using experimentally identified battery models" or "validated MBA-RL using battery models whose parameters are identified from real battery cells with different cycle lifetimes." Changes made in: Contribution #3, §IV opening paragraph, and Conclusion.

⚡ **Stronger fix:** We plan to deploy the trained MBA-RL policy on the NEWARE CT-4008 for hardware-in-the-loop validation and will include those results if time permits.

---

### P2 [MAJOR] — Scalability not verified (n=3 only)

**Response:** The larger-n design is paused rather than implemented by duplicating identical cells. The current environment has only three identified cell models. Before running n=5, we will either identify additional cells or define and justify a bounded virtual-cell parameter distribution. Until then, all claims remain restricted to the simulated three-cell cases.

### P3 [MAJOR] — Hardware feasibility of per-cell current sources — RESOLVED IN TEXT

**Response:** ✅ Added a Limitations and Engineering Scope subsection explaining that independently controllable channels require isolated converters or equivalent power electronics and that cost and complexity may limit direct scaling. The manuscript no longer claims demonstrated large-pack applicability.

### P4 [MAJOR] — Safety constraints overly conservative

**Response:** A local sensitivity schedule at 306, 309, and 312 K has been frozen. The 309 K value will also be supported by a battery-specific engineering source before submission; sensitivity alone does not establish physical appropriateness.

### P5 [MINOR] — No quantitative energy efficiency — CLAIM WITHDRAWN; MEASUREMENT PENDING

**Response:** All claims of demonstrated energy-efficiency improvement have been removed. A quantitative comparison still requires energy efficiency $\eta$ and Ah-throughput for every method under matched hardware assumptions.

---

## Devil's Advocate

### DA1 [CRITICAL] — "Eliminates discharging" falsified by code

**Response:** Same as M1. The discharge-elimination claim has been removed. The paper now reports the negative-current fallback in the released implementation and makes no energy-loss claim.

### DA2 [CRITICAL] — "Real battery validation" is simulation

**Response:** Same as P1. All overclaims corrected; validation accurately described as simulation with experimentally identified parameters.

### DA3 [MAJOR] — No comparison with simple rule-based baseline

**Response:** Same as D2/D3. Simple proportional feedback baseline will be added.

### DA4 [MAJOR] — Hardware/architecture confound

**Response:** Same as D3. Architecture and algorithm contributions are now separated in the manuscript; matched-architecture experiments remain pending.

### DA5 [MAJOR] — "Prevents re-emerging imbalance" = stopping rule

**Response:** Same as D5. Claim reworded as a design choice benefit, not intrinsic advantage.

### DA6 [MAJOR] — Overgeneralization to "industrial applications" / "large-scale"

**Response:** ✅ **Addressed in revision.**  
Conclusion revised to: (1) acknowledge that validation is simulation-based, (2) state that scalability to large-scale systems requires further investigation, and (3) add future work on hardware-in-the-loop experiments.

### DA7 [MINOR] — No quantified engineering value

**Response:** Same as P5. The unsupported efficiency claim has been withdrawn; matched energy-throughput measurements remain pending.

---

## Checklist of Items Requiring New Experiments (⚡)

| Item | Description | Estimated Effort |
|------|-------------|-----------------|
| M2 | Share complete reward code, verify reproducibility | 1 day |
| M3 | Reconcile hyperparameters, re-run with corrected target update | 2–3 days GPU |
| C1 (M6) | 5-seed statistical repeats with error bars | 5 days GPU |
| C2 (D4) | Unified termination condition comparison | 2 days GPU |
| C3 (D2/DA3) | Add VDN + proportional feedback baselines | 3–5 days GPU |
| C4 (D3/DA4) | Run DQL/rule under same hardware setting | 2 days GPU |
| C5 (M8/P4) | Sensitivity analysis on β, T_max, decision cycle | 3 days GPU |
| D1 (P2) | n=5 scalability experiment | 2–3 days GPU |
| D2 (P5) | Energy efficiency quantification | 1 day (post-processing) |
| P1 strong | Hardware-in-the-loop validation on NEWARE | 1–2 weeks lab |

If the authors wish to restore a strict "non-negative actions" contribution, the action space must first be restricted, all affected experiments rerun, and the new action traces reported. That optional change is not assumed in the current manuscript.

**Total estimated effort:** ~3–4 weeks (GPU experiments) + optional 1–2 weeks (hardware-in-the-loop)

---

*End of Response Letter*
