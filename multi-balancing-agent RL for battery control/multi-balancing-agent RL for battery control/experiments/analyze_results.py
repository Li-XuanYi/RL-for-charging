"""Aggregate independent runs without treating time steps as replicates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "time_to_balance_s",
    "time_to_reference_soc_s",
    "terminal_soc_std",
    "max_voltage_v",
    "max_temperature_k",
    "charge_energy_throughput_wh",
    "discharge_energy_throughput_wh",
)


def percentile(sorted_values, probability):
    if not sorted_values:
        return None
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(values, seed=20260829, draws=10_000):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [rng.choice(values) for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def load_summaries(results_root: Path):
    for path in results_root.rglob("summary.json"):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["_path"] = str(path)
        row["soc_set"] = row.get(
            "soc_set",
            "A" if row.get("initial_socs") == [0.1, 0.2, 0.3] else "B",
        )
        yield row


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("algorithm"), row.get("soc_set"), row.get("variant", "full"))
        groups[key].append(row)

    output = []
    for (algorithm, soc_set, variant), group in sorted(groups.items()):
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            if not values:
                continue
            ci_low, ci_high = bootstrap_mean_ci(values)
            output.append(
                {
                    "algorithm": algorithm,
                    "soc_set": soc_set,
                    "variant": variant,
                    "metric": metric,
                    "n_independent_runs": len(values),
                    "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else None,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("aggregated_metrics.csv"),
    )
    args = parser.parse_args()
    aggregated = aggregate(list(load_summaries(args.results_root)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "algorithm",
            "soc_set",
            "variant",
            "metric",
            "n_independent_runs",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)
    print(f"Wrote {len(aggregated)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
