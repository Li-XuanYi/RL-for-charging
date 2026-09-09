"""Deterministic PSO re-identification from the three archived voltage traces.

This utility creates a *new* parameter-identification result. It cannot recover
the unseeded, unarchived optima used for an earlier manuscript version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pybamm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_FILES = (
    ("Battery 1", "data_1C_begin"),
    ("Battery 2", "data_1C_middle"),
    ("Battery 3", "data_1C_age"),
)
PARAMETER_NAMES = (
    "Positive particle diffusivity [m2.s-1]",
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle radius [m]",
    "Negative particle radius [m]",
    "Negative electrode active material volume fraction",
    "Positive electrode active material volume fraction",
)
# These reproduce the encoded bounds in data_true/parameter identification.py.
# The fifth coordinate is scaled by 0.1, hence [6, 6.5] -> [0.60, 0.65].
LOWER = np.array([5.0, 4.0, 4.0, 4.0, 6.0, 5.0])
UPPER = np.array([6.0, 5.0, 5.0, 5.0, 6.5, 5.8])
INITIAL = np.array([5.77, 4.30, 4.06, 4.16, 5.58, 5.17])


def decoded_parameters(vector: np.ndarray) -> dict[str, float]:
    values = (
        vector[0] * 1e-15,
        vector[1] * 1e-14,
        vector[2] * 1e-6,
        vector[3] * 1e-6,
        vector[4] * 0.1,
        vector[5] * 0.1,
    )
    return dict(zip(PARAMETER_NAMES, (float(value) for value in values)))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate_voltage(vector: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
    parameters = pybamm.ParameterValues("Chen2020")
    parameters.update(decoded_parameters(vector))
    parameters["Current function [A]"] = 5.0
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3),
    )
    solution = simulation.solve(t_eval=times)
    return (
        np.asarray(solution["Time [s]"].entries, dtype=float),
        np.asarray(solution["Voltage [V]"].entries, dtype=float),
    )


def voltage_rmse(vector: np.ndarray, times: np.ndarray, measured: np.ndarray) -> float:
    try:
        simulated_time, simulated_voltage = simulate_voltage(vector, times)
    except Exception:
        return 1e6
    if simulated_time[-1] + 1e-6 < times[-1]:
        missing_fraction = (times[-1] - simulated_time[-1]) / times[-1]
        return 1e3 + float(missing_fraction)
    predicted = np.interp(times, simulated_time, simulated_voltage)
    return float(np.sqrt(np.mean((measured - predicted) ** 2)))


def fit_trace(
    times: np.ndarray,
    measured: np.ndarray,
    *,
    particles: int,
    iterations: int,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, float]:
    center = np.clip(INITIAL, LOWER, UPPER)
    positions = np.clip(
        center + 0.05 * rng.standard_normal((particles, len(center))),
        LOWER,
        UPPER,
    )
    velocity = rng.uniform(-0.03, 0.03, positions.shape)
    personal_best = positions.copy()
    personal_scores = np.array(
        [voltage_rmse(value, times, measured) for value in positions]
    )
    best_index = int(np.argmin(personal_scores))
    global_best = personal_best[best_index].copy()
    global_score = float(personal_scores[best_index])

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        inertia = 0.9 + progress * (0.4 - 0.9)
        cognitive = 2.5 + progress * (0.5 - 2.5)
        social = 0.5 + progress * (2.5 - 0.5)
        for particle in range(particles):
            r1 = rng.random(len(center))
            r2 = rng.random(len(center))
            velocity[particle] = (
                inertia * velocity[particle]
                + cognitive * r1 * (personal_best[particle] - positions[particle])
                + social * r2 * (global_best - positions[particle])
            )
            positions[particle] = np.clip(
                positions[particle] + velocity[particle], LOWER, UPPER
            )
            score = voltage_rmse(positions[particle], times, measured)
            if score < personal_scores[particle]:
                personal_best[particle] = positions[particle].copy()
                personal_scores[particle] = score
            if score < global_score:
                global_best = positions[particle].copy()
                global_score = float(score)
        print(
            json.dumps(
                {
                    "cell": label,
                    "iteration": iteration + 1,
                    "iterations": iterations,
                    "voltage_rmse_v": global_score,
                }
            ),
            flush=True,
        )
    return global_best, global_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data_true" / "data_image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("cell_parameters.reidentified.json"),
    )
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Use every kth time point; values above 1 are exploratory only.",
    )
    parser.add_argument(
        "--acknowledge-legacy-bounds",
        action="store_true",
        help="Required because the archived bounds conflict with one hard-coded candidate.",
    )
    args = parser.parse_args()
    if not args.acknowledge_legacy_bounds:
        raise RuntimeError(
            "Re-identification is paused: inspect and approve the source-coded "
            "bounds, then pass --acknowledge-legacy-bounds."
        )
    if args.particles < 2 or args.iterations < 1 or args.downsample < 1:
        raise ValueError("particles>=2, iterations>=1, and downsample>=1 are required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    rng = np.random.default_rng(args.seed)
    cells = []
    for label, filename in TRACE_FILES:
        path = args.data_root / filename
        measured_all = np.loadtxt(path, dtype=float)
        # Exclude samples below the model's 2.5 V termination threshold.
        indices = np.flatnonzero(measured_all >= 2.5)[:: args.downsample]
        if len(indices) < 2:
            raise ValueError(f"Insufficient usable samples in {path}")
        times = indices.astype(float)
        measured = measured_all[indices]
        best, score = fit_trace(
            times,
            measured,
            particles=args.particles,
            iterations=args.iterations,
            rng=rng,
            label=label,
        )
        cells.append(
            {
                "label": label,
                "parameters": decoded_parameters(best),
                "fit": {
                    "objective": "voltage_rmse",
                    "voltage_rmse_v": score,
                    "trace": str(path.resolve()),
                    "trace_sha256": file_sha256(path),
                    "usable_samples": int(len(indices)),
                    "downsample": args.downsample,
                },
            }
        )

    result = {
        "description": (
            "New deterministic voltage-only PSO re-identification; this is not "
            "a recovery of the unarchived original optima."
        ),
        "status": "new_reidentification_not_original_recovery",
        "template": "Chen2020",
        "current_a": 5.0,
        "seed": args.seed,
        "particles": args.particles,
        "iterations": args.iterations,
        "encoded_bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(PARAMETER_NAMES, LOWER, UPPER)
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote new re-identification result to {args.output}")


if __name__ == "__main__":
    main()
