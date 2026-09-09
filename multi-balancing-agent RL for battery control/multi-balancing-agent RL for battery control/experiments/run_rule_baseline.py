"""Matched per-cell current-source baselines for reviewer-requested comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from main import (
    SOC_SETS,
    build_environment,
    load_cell_parameters,
    set_global_seed,
    summarize_trace,
)


def nearest_available_action(env, agent_id: int, desired_current: float) -> int:
    available = env.get_avail_agent_actions(agent_id).astype(bool)
    indices = np.nonzero(available)[0]
    if indices.size == 0:
        raise RuntimeError(f"No available action for agent {agent_id}")
    distances = np.abs(env.action_space[indices] - desired_current)
    return int(indices[np.argmin(distances)])


def proportional_currents(env, gain: float = 20.0) -> list[float]:
    cells = (env.spm1, env.spm2, env.spm3)
    return [
        float(np.clip(gain * (env.soc_ref - cell.soc), 0.0, 7.0))
        for cell in cells
    ]


def per_cell_cccv_currents(
    env, cc_current: float = 4.0, cv_voltage: float = 4.0, cv_gain: float = 20.0
) -> list[float]:
    cells = (env.spm1, env.spm2, env.spm3)
    currents = []
    for cell in cells:
        if cell.voltage < cv_voltage:
            current = cc_current
        else:
            current = np.clip(cv_gain * (env.max_voltage - cell.voltage), 0.0, cc_current)
        currents.append(float(current))
    return currents


def run_baseline(
    algorithm: str,
    soc_set: str,
    seed: int,
    output_dir: Path,
    cell_parameters: Path | None = None,
) -> dict:
    conf = Config(seed=seed)
    if cell_parameters is not None:
        (
            conf.cell_parameter_overrides,
            conf.cell_parameter_source,
        ) = load_cell_parameters(cell_parameters)
    initial_socs = SOC_SETS[soc_set]
    set_global_seed(10_000 + seed)
    env = build_environment(conf, initial_socs)
    env.reset(initial_socs=initial_socs)

    trace = []
    for step in range(1, env.episode_limit + 1):
        if algorithm == "proportional":
            desired = proportional_currents(env)
        elif algorithm == "per_cell_cccv":
            desired = per_cell_cccv_currents(env)
        else:
            raise ValueError(f"Unknown baseline: {algorithm}")
        actions = [
            nearest_available_action(env, agent_id, current)
            for agent_id, current in enumerate(desired)
        ]
        _, terminated = env.multi_step(actions, conf.beta)
        transition = dict(env.last_transition)
        transition["step"] = step
        transition["time_s"] = step * conf.sample_time
        trace.append(transition)
        if terminated:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
    summary = summarize_trace(trace, conf)
    summary.update(
        {
            "algorithm": algorithm,
            "variant": "full",
            "seed": seed,
            "soc_set": soc_set,
            "initial_socs": list(initial_socs),
            "controller_note": (
                "Deterministic controller evaluated under the same seeded "
                "environment reset; seed is an environment block, not a training replicate."
            ),
            "cell_parameter_source": conf.cell_parameter_source,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", choices=("proportional", "per_cell_cccv"), required=True
    )
    parser.add_argument("--soc-set", choices=tuple(SOC_SETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-parameters", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_baseline(
        args.algorithm,
        args.soc_set,
        args.seed,
        args.output_dir,
        args.cell_parameters,
    )
    print(json.dumps(result, indent=2))
