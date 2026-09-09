"""Reproducible training entry point for the supplementary experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
from pathlib import Path
import shlex
import sys

import numpy as np
import pybamm
import torch

from agent import Agents
from config import Config
from SPM import MultiSPM
from utils import ReplayBuffer, RolloutWorker


SOC_SETS = {
    "A": (0.10, 0.20, 0.30),
    "B": (0.30, 0.50, 0.70),
}
LEGACY_PARAMETER_KEYS = {
    "Positive electrode diffusivity [m2.s-1]": (
        "Positive particle diffusivity [m2.s-1]"
    ),
    "Negative electrode diffusivity [m2.s-1]": (
        "Negative particle diffusivity [m2.s-1]"
    ),
}


def normalize_parameter_keys(parameters: dict) -> tuple[dict, dict]:
    """Map archived PyBaMM names to current names without silent conflicts."""
    normalized = dict(parameters)
    migrations = {}
    for legacy, current in LEGACY_PARAMETER_KEYS.items():
        if legacy not in normalized:
            continue
        if current in normalized:
            raise ValueError(
                f"Parameter mapping contains both legacy key {legacy!r} "
                f"and current key {current!r}"
            )
        normalized[current] = normalized.pop(legacy)
        migrations[legacy] = current
    return normalized, migrations


def load_cell_parameters(path: Path | str):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    cells = data.get("cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise ValueError("cell-parameter file must define exactly three cells")
    overrides = []
    all_migrations = {}
    for index, cell in enumerate(cells, start=1):
        parameters = cell.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(
                f"cell {index} must contain a non-empty parameters mapping"
            )
        parameters, migrations = normalize_parameter_keys(parameters)
        if "Current function [A]" in parameters:
            raise ValueError(
                "Current function [A] is controlled by the environment and "
                "must not appear in cell-parameter overrides"
            )
        overrides.append(parameters)
        if migrations:
            all_migrations[f"cell_{index}"] = migrations
    source = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "description": data.get("description", ""),
        "parameter_key_migrations": all_migrations,
    }
    return tuple(overrides), source


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_environment(conf: Config, initial_socs) -> MultiSPM:
    num_agents = 3
    obs_shape = 3
    state_shape = num_agents * obs_shape
    actions = np.arange(-2.0, 7.5, 0.5)
    episode_limit = conf.episode_limit_config
    return MultiSPM(
        num_agents,
        state_shape,
        obs_shape,
        len(actions),
        episode_limit,
        actions,
        beta=conf.beta,
        soc_ref=conf.soc_ref,
        time_penalty=conf.time_penalty,
        balance_penalty_scale=conf.balance_penalty_scale,
        voltage_penalty_scale=conf.voltage_penalty_scale,
        temperature_penalty_scale=conf.temperature_penalty_scale,
        max_voltage=conf.max_voltage,
        max_temperature=conf.max_temperature,
        sample_time=conf.sample_time,
        initial_socs=initial_socs,
        initialization_mode=conf.initialization_mode,
        cell_parameter_overrides=conf.cell_parameter_overrides,
    )


def summarize_trace(trace, conf: Config) -> dict:
    balance_times = [
        row["time_s"]
        for row in trace
        if row["reward_components"]["soc_std"] <= conf.beta
    ]
    reference_times = [
        row["time_s"]
        for row in trace
        if all(soc >= conf.soc_ref for soc in row["socs"])
    ]
    charge_energy_wh = 0.0
    discharge_energy_wh = 0.0
    for row in trace:
        for current, voltage in zip(row["actions_a"], row["voltages_v"]):
            energy = abs(current) * voltage * conf.sample_time / 3600.0
            if current >= 0:
                charge_energy_wh += energy
            else:
                discharge_energy_wh += energy

    return {
        "time_to_balance_s": balance_times[0] if balance_times else None,
        "time_to_reference_soc_s": reference_times[0] if reference_times else None,
        "joint_terminal_reached": bool(trace and trace[-1]["terminated"]),
        "terminal_soc_std": (
            trace[-1]["reward_components"]["soc_std"] if trace else None
        ),
        "max_voltage_v": max(
            (max(row["voltages_v"]) for row in trace), default=None
        ),
        "max_temperature_k": max(
            (max(row["temperatures_k"]) for row in trace), default=None
        ),
        "voltage_violation_steps": sum(
            max(row["voltages_v"]) >= conf.max_voltage for row in trace
        ),
        "temperature_violation_steps": sum(
            max(row["temperatures_k"]) >= conf.max_temperature for row in trace
        ),
        "charge_energy_throughput_wh": charge_energy_wh,
        "discharge_energy_throughput_wh": discharge_energy_wh,
    }


def train(
    conf: Config,
    initial_socs,
    output_dir: Path,
    variant: str = "full",
    progress_interval: int = 25,
) -> dict:
    set_global_seed(conf.seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "algorithm": conf.algorithm,
        "variant": variant,
        "seed": conf.seed,
        "initial_socs": list(initial_socs),
        "episodes": conf.n_epochs,
        "target_update_interval": conf.update_target_params,
        "episode_limit": conf.episode_limit_config,
        "initialization_mode": conf.initialization_mode,
        "cell_parameter_source": conf.cell_parameter_source,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pybamm_version": pybamm.__version__,
        "device": str(conf.device),
    }
    run_record_path = output_dir / "run_record.json"
    with run_record_path.open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)

    env = build_environment(conf, initial_socs)
    conf.set_env_info(env.get_env_info())
    agents = Agents(conf)
    rollout_worker = RolloutWorker(env, agents, conf)
    replay = ReplayBuffer(conf)

    rewards = []
    losses = []
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["episode", "reward", "loss", "epsilon"]
        )
        writer.writeheader()
        for epoch in range(conf.n_epochs):
            fraction = (epoch + 1) / conf.n_epochs
            epsilon = conf.start_epsilon + fraction * (
                conf.end_epsilon - conf.start_epsilon
            )
            episode, episode_reward, *_ = rollout_worker.generate_episode(epsilon)
            replay.store_episode(episode)
            loss = None
            if replay.current_size >= 5:
                mini_batch = replay.sample(
                    min(replay.current_size, conf.batch_size)
                )
                loss = agents.train(mini_batch, training_episode=epoch + 1)
            rewards.append(float(episode_reward))
            losses.append(loss)
            writer.writerow(
                {
                    "episode": epoch + 1,
                    "reward": float(episode_reward),
                    "loss": loss,
                    "epsilon": epsilon,
                }
            )
            handle.flush()
            if progress_interval > 0 and (
                (epoch + 1) % progress_interval == 0
                or epoch + 1 == conf.n_epochs
            ):
                agents.policy.save_model(output_dir / "checkpoint.latest.pt")
                print(
                    json.dumps(
                        {
                            "progress_episode": epoch + 1,
                            "episodes": conf.n_epochs,
                            "latest_reward": float(episode_reward),
                            "latest_loss": loss,
                        }
                    ),
                    flush=True,
                )

    agents.policy.save_model(output_dir / "model.pt")

    # Use a fixed, independent evaluation reset for paired method comparisons.
    set_global_seed(10_000 + conf.seed)
    _, evaluation_reward, *_ = rollout_worker.generate_episode(
        epsilon=0.0, initial_socs=initial_socs
    )
    trace = rollout_worker.last_trace
    with (output_dir / "evaluation_trace.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(trace, handle, indent=2)

    summary = summarize_trace(trace, conf)
    summary.update(
        {
            "algorithm": conf.mixer,
            "variant": variant,
            "seed": conf.seed,
            "initial_socs": list(initial_socs),
            "episodes": conf.n_epochs,
            "target_update_interval": conf.update_target_params,
            "episode_limit": conf.episode_limit,
            "initialization_mode": conf.initialization_mode,
            "cell_parameter_source": conf.cell_parameter_source,
            "evaluation_reward": float(evaluation_reward),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pybamm_version": pybamm.__version__,
            "device": str(conf.device),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    run_record["status"] = "complete"
    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_record["summary_path"] = str((output_dir / "summary.json").resolve())
    with run_record_path.open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", choices=("qmix", "vdn", "dql"), default="qmix"
    )
    parser.add_argument("--soc-set", choices=tuple(SOC_SETS), default="B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--target-update", type=int, default=50)
    parser.add_argument("--episode-limit", type=int, default=50)
    parser.add_argument(
        "--initialization-mode",
        choices=("soc_consistent", "legacy_mixed"),
        default="soc_consistent",
    )
    parser.add_argument("--cell-parameters", type=Path)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max-temperature", type=float, default=309.0)
    parser.add_argument("--sample-time", type=int, default=90)
    parser.add_argument("--time-penalty", type=float, default=-0.75)
    parser.add_argument("--balance-penalty-scale", type=float, default=-50.0)
    parser.add_argument("--voltage-penalty-scale", type=float, default=-20.0)
    parser.add_argument("--temperature-penalty-scale", type=float, default=-2.0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conf = Config(
        seed=args.seed,
        n_epochs=args.episodes,
        target_update_interval=args.target_update,
        mixer=args.algorithm,
        episode_limit=args.episode_limit,
        initialization_mode=args.initialization_mode,
    )
    conf.beta = args.beta
    conf.max_temperature = args.max_temperature
    conf.sample_time = args.sample_time
    conf.time_penalty = args.time_penalty
    conf.balance_penalty_scale = args.balance_penalty_scale
    conf.voltage_penalty_scale = args.voltage_penalty_scale
    conf.temperature_penalty_scale = args.temperature_penalty_scale
    if args.cell_parameters is not None:
        (
            conf.cell_parameter_overrides,
            conf.cell_parameter_source,
        ) = load_cell_parameters(args.cell_parameters)
    output_dir = args.output_dir or Path("results") / args.algorithm / (
        f"set_{args.soc_set.lower()}_seed_{args.seed}"
    )
    summary = train(
        conf,
        SOC_SETS[args.soc_set],
        output_dir,
        variant=args.variant,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
