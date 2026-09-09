"""Pure reward and stopping-rule functions for reproducible experiments.

This module deliberately has no PyBaMM or PyTorch dependency, so the equations
reported in the manuscript can be unit-tested independently of the simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def population_soc_std(socs: Sequence[float]) -> float:
    if not socs:
        raise ValueError("socs must contain at least one cell")
    mean_soc = sum(socs) / len(socs)
    return math.sqrt(sum((soc - mean_soc) ** 2 for soc in socs) / len(socs))


def balancing_reward(
    socs: Sequence[float], beta: float, scale: float = -50.0
) -> tuple[float, float]:
    sigma = population_soc_std(socs)
    reward = scale * (sigma - beta) if sigma >= beta else 0.0
    return float(reward), float(sigma)


def safety_reward(
    voltages: Sequence[float],
    temperatures: Sequence[float],
    max_voltage: float = 4.2,
    max_temperature: float = 309.0,
    voltage_scale: float = -20.0,
    temperature_scale: float = -2.0,
) -> tuple[float, float, float]:
    if len(voltages) != len(temperatures):
        raise ValueError("voltages and temperatures must have equal length")
    voltage_term = sum(
        voltage_scale * (voltage - max_voltage)
        for voltage in voltages
        if voltage >= max_voltage
    )
    temperature_term = sum(
        temperature_scale * (temperature - max_temperature)
        for temperature in temperatures
        if temperature >= max_temperature
    )
    return (
        float(voltage_term + temperature_term),
        float(voltage_term),
        float(temperature_term),
    )


def terminal_status(
    socs: Sequence[float], beta: float = 0.02, soc_ref: float = 0.90
) -> tuple[bool, float]:
    sigma = population_soc_std(socs)
    done = sigma <= beta and all(soc >= soc_ref for soc in socs)
    return bool(done), float(sigma)


def pack_reward(
    socs: Sequence[float],
    voltages: Sequence[float],
    temperatures: Sequence[float],
    *,
    beta: float = 0.02,
    time_penalty: float = -0.75,
    balance_scale: float = -50.0,
    max_voltage: float = 4.2,
    max_temperature: float = 309.0,
    voltage_scale: float = -20.0,
    temperature_scale: float = -2.0,
) -> tuple[float, dict[str, float]]:
    balance_term, sigma = balancing_reward(socs, beta, balance_scale)
    safety_term, voltage_term, temperature_term = safety_reward(
        voltages,
        temperatures,
        max_voltage,
        max_temperature,
        voltage_scale,
        temperature_scale,
    )
    total = time_penalty + balance_term + safety_term
    components = {
        "time": float(time_penalty),
        "balance": balance_term,
        "safety": safety_term,
        "voltage": voltage_term,
        "temperature": temperature_term,
        "soc_std": sigma,
    }
    return float(total), components
