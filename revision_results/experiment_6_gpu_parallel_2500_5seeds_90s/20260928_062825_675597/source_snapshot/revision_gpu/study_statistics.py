"""Paired seed-level comparisons for E3-E6, conditional on fixed test cases.

Case trajectories are not independent training replicates. A deterministic rule
is evaluated once, matched to each learned seed when necessary, and never
counted as additional independent training seeds. Comparisons use the prespecified
reference within each panel/action-domain/cell-count/test-family group.

Two-sided sign-flip tests assume sign exchangeability of independent seed-level
paired effects under the null. Five nonzero seed effects permit a minimum exact
two-sided p of 2/32 = .0625. This intentionally does not manufacture significance
by treating cells or scenarios as independent learned policies. Bootstrap
intervals describe seed variability conditional on the fixed scenario set.

The module does not launch experiments and makes no automatic superiority claim.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import dump, write_csv


_GROUP = ("study_panel", "action_kind", "n_agents", "test_family")
_METRICS = (
    ("safe_success_difference", "all_paired_cases", "higher", "primary"),
    ("capped_charge_time_difference_s", "all_paired_cases", "lower", "primary"),
    ("joint_success_charge_time_difference_s", "joint_success_only", "lower", "secondary"),
    ("joint_success_imbalance_integral_difference_soc_s", "joint_success_only", "lower", "secondary"),
    ("joint_success_terminal_input_difference_Wh", "joint_success_only", "neither_efficiency_nor_loss", "secondary"),
)


def _group(row):
    return (str(row.get("study_panel", "main")), str(row["action_kind"]),
            int(row["n_agents"]), str(row.get("test_family", "nominal")))


def _finite(value, name):
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Paired comparison requires finite {name}")
    return float(value)


def _success(row):
    value = row["task_success"]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("task_success must be a Boolean, not a string or missing value")
    return bool(value)


def _rng(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.default_rng((int(seed) + int.from_bytes(digest[:8], "little")) % 2**64)


def _sign_flip(values, rng, monte_carlo):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 2:
        return None, "insufficient_independent_seeds", 0
    observed = abs(float(values.mean()))
    tolerance = 1e-12 * max(1.0, observed)
    greater = 0
    if count <= 20:
        draws = 2**count
        shifts = np.arange(count, dtype=np.uint32)[None, :]
        for start in range(0, draws, 4096):
            numbers = np.arange(start, min(start + 4096, draws), dtype=np.uint32)[:, None]
            signs = 2.0 * ((numbers >> shifts) & 1).astype(np.float64) - 1.0
            effects = (signs @ values) / count
            greater += int(np.count_nonzero(np.abs(effects) >= observed - tolerance))
        return greater / draws, "exact_two_sided_seed_sign_flip", draws
    for start in range(0, monte_carlo, 4096):
        size = min(4096, monte_carlo - start)
        signs = 2.0 * rng.integers(0, 2, (size, count)) - 1.0
        effects = (signs @ values) / count
        greater += int(np.count_nonzero(np.abs(effects) >= observed - tolerance))
    return (greater + 1) / (monte_carlo + 1), "monte_carlo_two_sided_seed_sign_flip_plus_one", monte_carlo


def _bootstrap(values, rng, replicates):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None, None
    sampled_means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 4096):
        size = min(4096, replicates - start)
        sampled_means[start:start + size] = rng.choice(values, (size, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(low), float(high)


def _holm(rows):
    # Undefined endpoints retain a planned place in the family as p=1, avoiding
    # a smaller correction merely because a difficult endpoint had no successes.
    for position, row in enumerate(rows):
        row["holm_family_size"] = len(rows)
        row["holm_adjusted_p"] = None
    order = sorted(range(len(rows)), key=lambda i: (1.0 if rows[i]["p_value"] is None else rows[i]["p_value"], i))
    running = 0.0
    for rank, index in enumerate(order):
        p_value = rows[index]["p_value"]
        running = max(running, min(1.0, (len(rows) - rank) * (1.0 if p_value is None else p_value)))
        if p_value is not None:
            rows[index]["holm_adjusted_p"] = running


def _case_difference(reference, candidate, group, ref_id, candidate_id, paired_seed, cap):
    ref_success, candidate_success = _success(reference), _success(candidate)
    ref_time = _finite(reference["charging_time_s"], "reference charge time") if ref_success else cap
    candidate_time = _finite(candidate["charging_time_s"], "candidate charge time") if candidate_success else cap
    if not (0 <= ref_time <= cap and 0 <= candidate_time <= cap):
        raise ValueError("Successful charge time lies outside the predeclared physical horizon")
    joint = ref_success and candidate_success
    result = {**dict(zip(_GROUP, group)), "reference_method": ref_id,
              "comparison_method": candidate_id, "paired_seed": paired_seed,
              "reference_seed": int(reference["seed"]), "comparison_seed": int(candidate["seed"]),
              "case_id": str(reference["case_id"]), "reference_success": ref_success,
              "comparison_success": candidate_success, "joint_success": joint,
              "deterministic_rule_reused_for_pairing": reference["seed"] == -1 or candidate["seed"] == -1,
              "safe_success_difference": int(candidate_success) - int(ref_success),
              "capped_charge_time_difference_s": candidate_time - ref_time,
              "joint_success_charge_time_difference_s": candidate_time - ref_time if joint else None,
              "joint_success_imbalance_integral_difference_soc_s": None,
              "joint_success_terminal_input_difference_Wh": None,
              "joint_success_terminal_soc_max_absolute_difference": None}
    if joint:
        result["joint_success_imbalance_integral_difference_soc_s"] = (
            _finite(candidate["soc_std_integral_soc_s"], "candidate SOC integral")
            - _finite(reference["soc_std_integral_soc_s"], "reference SOC integral")
        )
        result["joint_success_terminal_input_difference_Wh"] = (
            _finite(candidate["sum_cell_terminal_input_Wh"], "candidate terminal input Wh")
            - _finite(reference["sum_cell_terminal_input_Wh"], "reference terminal input Wh")
        )
        if "final_charge_soc_vector" in reference and "final_charge_soc_vector" in candidate:
            ref_soc = np.asarray(reference["final_charge_soc_vector"], dtype=float)
            candidate_soc = np.asarray(candidate["final_charge_soc_vector"], dtype=float)
            if ref_soc.shape != (group[2],) or candidate_soc.shape != ref_soc.shape or not np.isfinite(ref_soc).all() or not np.isfinite(candidate_soc).all():
                raise ValueError("Malformed terminal SOC vectors in paired comparison")
            result["joint_success_terminal_soc_max_absolute_difference"] = float(np.max(np.abs(candidate_soc - ref_soc)))
    return result


def compare(records, out, cfg, references=None):
    """Write paired case/seed effects and adjusted inference, then return a report.

    references maps action kind to a canonical method_id. Defaults are discrete
    qmix and continuous mappo_continuous; absent references are reported and
    skipped, never replaced by an outcome-selected comparator. In each contrast,
    missing seeds/cases invalidate CI/p-value inference but remain documented.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    references = {"discrete": "qmix", "continuous": "mappo_continuous"} if references is None else dict(references)
    records = list(records)
    bootstrap_seed = int(cfg.get("bootstrap_seed", 390903))
    replicates = int(cfg.get("bootstrap_replicates", 2000))
    sign_flip_replicates = int(cfg.get("sign_flip_replicates", 100000))
    if replicates < 2 or sign_flip_replicates < 1:
        raise ValueError("Invalid statistical resampling counts")
    cap = _finite(cfg["horizon_s"], "horizon_s") + _finite(cfg["hold_s"], "hold_s")
    if cap <= 0:
        raise ValueError("Failure-capped charge time must be positive")
    groups = defaultdict(lambda: defaultdict(dict))
    for row in records:
        group, method = _group(row), str(row["method_id"])
        key = (int(row["seed"]), str(row["case_id"]))
        if key in groups[group][method]:
            raise ValueError(f"Duplicate case record would double-count evidence: {group}, {method}, {key}")
        groups[group][method][key] = row
    pairs, seed_rows, summaries, unmatched, skipped = [], [], [], [], []
    for group, methods in sorted(groups.items()):
        ref_id = references.get(group[1])
        if not ref_id or ref_id not in methods:
            skipped.append({**dict(zip(_GROUP, group)), "reason": "prespecified_reference_absent", "reference_method": ref_id})
            continue
        reference = methods[ref_id]
        ref_seeds = {key[0] for key in reference}
        if -1 in ref_seeds and len(ref_seeds) != 1:
            raise ValueError("A method cannot mix deterministic and learned seed identifiers")
        ref_rule = ref_seeds == {-1}
        family_rows = []
        for candidate_id, candidate in sorted(methods.items()):
            if candidate_id == ref_id:
                continue
            candidate_seeds = {key[0] for key in candidate}
            if -1 in candidate_seeds and len(candidate_seeds) != 1:
                raise ValueError("A method cannot mix deterministic and learned seed identifiers")
            candidate_rule = candidate_seeds == {-1}
            if ref_rule and candidate_rule:
                paired_seeds = {-1}
            elif ref_rule:
                paired_seeds = candidate_seeds
            elif candidate_rule:
                paired_seeds = ref_seeds
            else:
                paired_seeds = ref_seeds | candidate_seeds
            contrast_pairs = []
            complete = True
            case_sets = []
            for seed in sorted(paired_seeds):
                ref_seed = -1 if ref_rule else seed
                candidate_seed = -1 if candidate_rule else seed
                ref_cases = {case for run_seed, case in reference if run_seed == ref_seed}
                candidate_cases = {case for run_seed, case in candidate if run_seed == candidate_seed}
                shared = ref_cases & candidate_cases
                if ref_cases != candidate_cases or not shared:
                    complete = False
                    for case in sorted(ref_cases ^ candidate_cases):
                        unmatched.append({**dict(zip(_GROUP, group)), "reference_method": ref_id,
                                          "comparison_method": candidate_id, "paired_seed": seed, "case_id": case,
                                          "missing_method": ref_id if case not in ref_cases else candidate_id})
                case_sets.append(shared)
                for case in sorted(shared):
                    contrast_pairs.append(_case_difference(reference[(ref_seed, case)], candidate[(candidate_seed, case)],
                                                           group, ref_id, candidate_id, seed, cap))
            if case_sets and any(cases != case_sets[0] for cases in case_sets[1:]):
                complete = False
            pairs.extend(contrast_pairs)
            contrast_seed_rows = []
            for seed in sorted(paired_seeds):
                matched = [row for row in contrast_pairs if row["paired_seed"] == seed]
                if not matched:
                    continue
                seed_row = {**dict(zip(_GROUP, group)), "reference_method": ref_id,
                            "comparison_method": candidate_id, "paired_seed": seed,
                            "matched_cases": len(matched), "joint_success_cases": sum(row["joint_success"] for row in matched),
                            "complete_case_and_seed_pairing": complete,
                            "is_independent_training_replicate": seed != -1}
                for metric, _, _, _ in _METRICS:
                    values = [row[metric] for row in matched if row[metric] is not None]
                    seed_row[metric] = float(np.mean(values)) if values else None
                    seed_row[metric + "_case_count"] = len(values)
                contrast_seed_rows.append(seed_row)
            seed_rows.extend(contrast_seed_rows)
            for metric, conditioning, direction, status in _METRICS:
                available = [row for row in contrast_seed_rows if row[metric] is not None]
                values = [row[metric] for row in available]
                stochastic = [row[metric] for row in available if row["paired_seed"] != -1]
                infer = complete and len(stochastic) >= 2
                ci = _bootstrap(stochastic, _rng(bootstrap_seed, group, ref_id, candidate_id, metric, "bootstrap"), replicates) if infer else (None, None)
                p_value, test, permutations = _sign_flip(
                    stochastic, _rng(bootstrap_seed, group, ref_id, candidate_id, metric, "sign_flip"), sign_flip_replicates
                ) if infer else (None, "inference_unavailable_missing_pairs_or_insufficient_seeds", 0)
                nonzero = sum(abs(value) > 1e-12 for value in stochastic)
                summary = {**dict(zip(_GROUP, group)), "reference_method": ref_id, "comparison_method": candidate_id,
                           "metric": metric, "endpoint_status": status, "difference_direction": "comparison_minus_reference",
                           "preferred_direction": direction, "conditioning": conditioning,
                           "mean_seed_difference": float(np.mean(values)) if values else None,
                           "seed_bootstrap_ci_low": ci[0], "seed_bootstrap_ci_high": ci[1],
                           "independent_paired_training_seeds": len(stochastic),
                           "nonzero_seed_effects": nonzero,
                           "minimum_two_sided_exact_p_if_no_ties": min(1.0, 2.0 / (2**nonzero)) if nonzero <= 1023 else 0.0,
                           "matched_case_pairs": len(contrast_pairs),
                           "contributing_case_pairs": sum(row[metric + "_case_count"] for row in available),
                           "complete_case_and_seed_pairing": complete, "p_value": p_value,
                           "test": test, "sign_assignments_or_mc_draws": permutations,
                           "automatic_superiority_claim": False,
                           "note": "Seed bootstrap conditional on fixed cases. Joint-success endpoints are selected subsets; input Wh is not loss or efficiency."}
                if conditioning == "joint_success_only" and len(available) < len(contrast_seed_rows):
                    summary["note"] += " Seeds without joint successes have undefined conditional effects and are reported as missing."
                family_rows.append(summary)
        _holm(family_rows)
        summaries.extend(family_rows)
    write_csv(out / "paired_case_differences.csv", pairs)
    write_csv(out / "paired_seed_differences.csv", seed_rows)
    write_csv(out / "paired_comparison_summary.csv", summaries)
    write_csv(out / "unmatched_pair_records.csv", unmatched)
    report = {"references": references, "failure_capped_charge_time_s": cap,
              "paired_case_rows": len(pairs), "paired_seed_rows": len(seed_rows),
              "summaries": summaries, "skipped_groups": skipped, "unmatched_records": unmatched,
              "bootstrap_replicates": replicates, "bootstrap_seed": bootstrap_seed,
              "holm_family": "All planned candidate contrasts and all five endpoints within each study_panel/action_kind/n_agents/test_family; undefined p values retain a place as p=1.",
              "independence_unit": "Independent learned training seed; deterministic rules are reused only for paired subtraction, never as new replicates.",
              "small_sample_warning": "Five nonzero paired seeds have minimum exact two-sided p=.0625 before Holm correction. Low power is not evidence of equivalence.",
              "conditional_endpoint_warning": "Charging time, imbalance integral, and terminal input Wh use only cases where both policies succeeded; successful subsets can differ across seeds.",
              "scope": "Conditional on the finite fixed simulation scenarios, with sign-exchangeability required for sign-flip inference. Separate study families are not a global multiplicity correction.",
              "energy_warning": "Input Wh cannot establish loss, efficiency, or lifetime improvement. Inspect terminal SOC differences.",
              "automatic_superiority_claim": False}
    dump(out / "paired_comparison_report.json", report)
    return report
