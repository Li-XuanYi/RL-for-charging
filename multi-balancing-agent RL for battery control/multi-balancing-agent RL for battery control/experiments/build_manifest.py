"""Generate a seeded, blocked run order for the supplementary experiments."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


TRAINING_SEEDS = (7, 17, 29, 43, 71)
SOC_SETS = ("A", "B")
SCHEDULE_SEED = 20260829


def learned_command(row: dict) -> str:
    result_label = row["variant"] if row["variant"] != "full" else row["algorithm"]
    output = (
        f"results/{row['phase']}/{result_label}/"
        f"set_{row['soc_set'].lower()}_seed_{row['seed']}"
    )
    parts = [
        r".\.venv\Scripts\python.exe main.py",
        f"--algorithm {row['algorithm']}",
        f"--soc-set {row['soc_set']}",
        f"--seed {row['seed']}",
        f"--episodes {row['episodes']}",
        f"--target-update {row['target_update']}",
        f"--variant {row['variant']}",
        f"--initialization-mode {row['initialization_mode']}",
        f"--beta {row['beta']}",
        f"--max-temperature {row['max_temperature_k']}",
        f"--sample-time {row['sample_time_s']}",
        f"--time-penalty {row['time_penalty']}",
        f"--balance-penalty-scale {row['balance_scale']}",
        f"--voltage-penalty-scale {row['voltage_scale']}",
        f"--temperature-penalty-scale {row['temperature_scale']}",
        f'--output-dir "{output}"',
    ]
    if row["cell_parameter_file"]:
        parts.append(f'--cell-parameters "{row["cell_parameter_file"]}"')
    return " ".join(parts)


def baseline_command(row: dict) -> str:
    result_label = row["variant"] if row["variant"] != "full" else row["algorithm"]
    output = (
        f"results/{row['phase']}/{result_label}/"
        f"set_{row['soc_set'].lower()}_seed_{row['seed']}"
    )
    command = (
        r".\.venv\Scripts\python.exe experiments/run_rule_baseline.py "
        f"--algorithm {row['algorithm']} --soc-set {row['soc_set']} "
        f"--seed {row['seed']} --output-dir \"{output}\""
    )
    if row["cell_parameter_file"]:
        command += f' --cell-parameters "{row["cell_parameter_file"]}"'
    return command


def base_row(phase: str, algorithm: str, soc_set: str, seed: int) -> dict:
    return {
        "phase": phase,
        "algorithm": algorithm,
        "variant": "full",
        "soc_set": soc_set,
        "seed": seed,
        "episodes": 1200,
        "target_update": 50,
        "beta": 0.02,
        "max_temperature_k": 309.0,
        "sample_time_s": 90,
        "time_penalty": -0.75,
        "balance_scale": -50.0,
        "voltage_scale": -20.0,
        "temperature_scale": -2.0,
        "n_cells": 3,
        "initialization_mode": "soc_consistent",
        "cell_parameter_file": "experiments/cell_parameters.identified.json",
        "review_items": "",
        "implementation_status": "pending_identified_parameter_sets",
        "required_for_resubmission": "yes",
        "replicate_unit": "",
    }


def build_rows() -> list[dict]:
    rows = []
    for soc_set in SOC_SETS:
        for seed in TRAINING_SEEDS:
            for algorithm in ("qmix", "vdn", "proportional", "per_cell_cccv"):
                row = base_row("confirmatory", algorithm, soc_set, seed)
                row["review_items"] = "M6;D2;D3;D4"
                row["replicate_unit"] = (
                    "training_seed"
                    if algorithm in {"qmix", "vdn"}
                    else "environment_block"
                )
                rows.append(row)

            dql = base_row("confirmatory", "dql", soc_set, seed)
            dql["review_items"] = "M6;D2;D3;D4"
            dql["replicate_unit"] = "training_seed"
            rows.append(dql)

            for algorithm, overrides in (
                ("qmix_no_time", {"time_penalty": 0.0}),
                ("qmix_no_balance", {"balance_scale": 0.0}),
                (
                    "qmix_no_safety",
                    {"voltage_scale": 0.0, "temperature_scale": 0.0},
                ),
            ):
                row = base_row("ablation", "qmix", soc_set, seed)
                row.update(overrides)
                row["variant"] = algorithm
                row["review_items"] = "M8"
                row["replicate_unit"] = "training_seed"
                rows.append(row)

            for threshold in (306.0, 312.0):
                row = base_row("temperature_sensitivity", "qmix", soc_set, seed)
                row["max_temperature_k"] = threshold
                row["variant"] = f"tmax_{int(threshold)}"
                row["review_items"] = "P4"
                row["replicate_unit"] = "training_seed"
                rows.append(row)

            for episodes, target_update, label in (
                (800, 200, "legacy_declared"),
                (800, 1, "legacy_effective_every_step"),
            ):
                row = base_row("provenance", "qmix", soc_set, seed)
                row["episodes"] = episodes
                row["target_update"] = target_update
                row["initialization_mode"] = "legacy_mixed"
                row["variant"] = label
                row["review_items"] = "M2;M3"
                row["replicate_unit"] = "training_seed"
                row["cell_parameter_file"] = ""
                row["implementation_status"] = "ready"
                rows.append(row)

    rng = random.Random(SCHEDULE_SEED)
    rng.shuffle(rows)
    for order, row in enumerate(rows, start=1):
        row["run_order"] = order
        row["run_id"] = f"SUP-{order:03d}"
        if row["implementation_status"] != "ready":
            row["command"] = ""
        elif row["algorithm"] in {"proportional", "per_cell_cccv"}:
            row["command"] = baseline_command(row)
        else:
            row["command"] = learned_command(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("experiment_manifest.csv"),
    )
    parser.add_argument(
        "--commands-output",
        type=Path,
        default=Path(__file__).with_name("commands_ready.ps1"),
        help="PowerShell script containing only currently unblocked runs.",
    )
    args = parser.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ready_commands = [row["command"] for row in rows if row["command"]]
    args.commands_output.parent.mkdir(parents=True, exist_ok=True)
    with args.commands_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("$ErrorActionPreference = 'Stop'\n")
        for command in ready_commands:
            handle.write(f"{command}\n")
    print(f"Wrote {len(rows)} scheduled runs to {args.output}")
    print(
        f"Wrote {len(ready_commands)} unblocked commands to "
        f"{args.commands_output}"
    )


if __name__ == "__main__":
    main()
