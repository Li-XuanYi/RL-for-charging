"""Experiment 2: separate SPMe plant and nominal finite-horizon safety screen.

The filter receives only measured SOC, terminal voltage and cell temperature.
It does not receive a plant object, plant parameters, internal concentration
profiles or future plant trajectories.  Its own nominal model is advanced with
the executed currents.  A screened interval followed by one zero-current
interval is a finite-horizon test, NOT an invariant-set safety guarantee.

No experiment is launched by importing this module.  This revision was authored
and statically reviewed; the requested experiments are to be run by the user.
"""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

import numpy as np
import pybamm

from model import MESH, parameter_values, values


ACTIONS = np.arange(0.0, 7.500001, 0.5)
DT = 90.0
TARGET = 0.9
BETA = 0.02
VMAX = 4.2
TMAX = 309.0
NUMERICAL_VMAX = 4.5
NUMERICAL_TMAX = 330.0
_V = "Voltage [V]"
_T = "X-averaged cell temperature [K]"
_C = "R-averaged negative particle concentration [mol.m-3]"
_TIME_TOL = 1e-5


class _NumericalFailure(RuntimeError):
    """A failed trajectory, distinct from a source-code/configuration error."""


def _validated_currents(currents, n):
    a = np.asarray(currents, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"Expected {n} finite charging currents")
    if np.any(a < -1e-10) or np.any(a > ACTIONS[-1] + 1e-10):
        raise ValueError("Charging currents must be in [0, 7.5] A")
    return np.clip(a, 0.0, ACTIONS[-1])


class _Cell:
    def __init__(self, vector, soc, ambient_K, options):
        p = parameter_values(vector, temperature=ambient_K, ambient=ambient_K)
        p["Upper voltage cut-off [V]"] = VMAX
        p["Current function [A]"] = "[input]"
        # Determine BOTH electrodes from the same SOC convention.  Numerical
        # overvoltage stopping must not silently change the definition of SOC.
        p.set_initial_stoichiometries(0.0)
        self.c0 = float(p["Initial concentration in negative electrode [mol.m-3]"])
        p.set_initial_stoichiometries(1.0)
        self.c1 = float(p["Initial concentration in negative electrode [mol.m-3]"])
        p.set_initial_stoichiometries(float(soc))
        if abs(self.c1 - self.c0) < 1e-12:
            raise ValueError("SOC reference concentrations are indistinguishable")
        p["Upper voltage cut-off [V]"] = NUMERICAL_VMAX
        model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
        model.events.append(pybamm.Event(
            "Diagnostic numerical temperature ceiling 330 K",
            NUMERICAL_TMAX - model.variables[_T],
        ))
        if options["physical_interlock"]:
            # Ideal continuously monitored plant protection. It terminates a
            # FAILED episode; it is not a proof that a discrete policy is safe.
            # The nominal screen keeps these events disabled so a rejected
            # candidate can still report its predicted constraint excess.
            model.events.append(pybamm.Event(
                "Protective voltage ceiling 4.2 V", VMAX - model.variables[_V],
            ))
            model.events.append(pybamm.Event(
                "Protective temperature ceiling 309 K", TMAX - model.variables[_T],
            ))
            soc_expression = (pybamm.x_average(model.variables[_C]) - self.c0) / (self.c1 - self.c0)
            model.events.append(pybamm.Event(
                "Protective SOC upper bound", float(options["physical_soc_max"]) - soc_expression,
            ))
        self.sample_dt_s = float(options["sample_dt_s"])
        self.sim = pybamm.Simulation(
            model, parameter_values=p,
            solver=pybamm.CasadiSolver(
                mode="safe", rtol=float(options["rtol"]),
                atol=float(options["atol"]),
            ),
            var_pts=dict(options["var_pts"]),
        )
        # The microscopic zero-current solve only establishes a consistent DAE
        # state.  Its clock offset is removed from every exported trajectory.
        self.initial = self.sim.solve(
            np.array([0.0, 1e-6]),
            inputs={"Current function [A]": 0.0},
        ).last_state
        self.functions = {}
        self.mapped = {}
        for key in (_V, _T, _C):
            variable = self.initial[key]
            # Compatible fallback for other supported PyBaMM builds; the pinned
            # runtime exposes the same compiled expressions used by variables.
            if hasattr(variable, "base_variables_casadi"):
                self.functions[key] = variable.base_variables_casadi[0]
        self.reset()

    def read(self, sol, key):
        if key not in self.functions:
            data = values(sol, key)
        else:
            count = len(sol.t)
            cache_key = (key, count)
            if cache_key not in self.mapped:
                self.mapped[cache_key] = self.functions[key].map(count)
            current = float(np.asarray(
                sol.all_inputs[0]["Current function [A]"]
            ).reshape(-1)[0])
            data = self.mapped[cache_key](
                np.asarray(sol.t).reshape(1, -1), sol.y,
                np.full((1, count), current),
            )
            data = np.asarray(data).reshape(-1, count).mean(axis=0)
        result = np.asarray(data, dtype=float).reshape(-1)
        if len(result) != len(sol.t) or not np.isfinite(result).all():
            raise _NumericalFailure(f"Non-finite or malformed model output: {key}")
        return result

    def reset(self):
        self.sol = self.initial
        self.current_A = 0.0
        self.update()

    def update(self):
        self.voltage = float(self.read(self.sol, _V)[-1])
        self.temperature = float(self.read(self.sol, _T)[-1])
        self.soc = float((self.read(self.sol, _C)[-1] - self.c0)
                         / (self.c1 - self.c0))

    def propose(self, current, duration, old_solution=None):
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError("A solver interval must have a positive finite duration")
        count = max(2, int(math.ceil(duration / self.sample_dt_s)) + 1)
        # Include the consistent post-switch algebraic voltage, not only the
        # pre-switch observation and the endpoint ninety seconds later.
        return self.sim.solver.step(
            self.sol if old_solution is None else old_solution,
            self.sim.built_model, float(duration),
            t_eval=np.linspace(0.0, float(duration), count), save=False,
            inputs={"Current function [A]": -float(current)},
        )

    def trajectory(self, sol, origin=None):
        if origin is None:
            origin = float(self.sol.t[-1])
        local_t = np.asarray(sol.t, dtype=float) - float(origin)
        if len(local_t) == 0 or not np.isfinite(local_t).all():
            raise _NumericalFailure("Missing or non-finite solver time")
        # BaseSolver shifts the start by one floating-point ULP to distinguish
        # adjacent steps.  Export that post-switch sample at exactly local t=0.
        if abs(local_t[0]) < _TIME_TOL:
            local_t[0] = 0.0
        if np.any(np.diff(local_t) < 0.0):
            raise _NumericalFailure("Solver time is not monotone")
        return {
            "time_s": local_t,
            "voltage_V": self.read(sol, _V),
            "temperature_K": self.read(sol, _T),
            "soc": (self.read(sol, _C) - self.c0) / (self.c1 - self.c0),
        }

    def commit(self, sol, current, duration, pack_time):
        before = dict(soc=self.soc, voltage_V=self.voltage,
                      temperature_K=self.temperature)
        if duration <= 0.0:
            return {
                "requested_current_A": float(current), "applied_current_A": 0.0,
                "interval_duration_s": 0.0,
                "time_s": [float(pack_time)],
                "voltage_V": [self.voltage], "temperature_K": [self.temperature],
                "soc": [self.soc], "mean_voltage_V": self.voltage,
                "max_voltage_V": self.voltage,
                "max_temperature_K": self.temperature,
                "max_soc": self.soc, "pre_switch_voltage_V": self.voltage,
                "post_switch_voltage_V": self.voltage,
                "pre_switch_state": before,
            }
        dense = self.trajectory(sol)
        if abs(dense["time_s"][-1] - duration) > _TIME_TOL:
            raise _NumericalFailure("Cell endpoint does not equal committed interval")
        voltage, temperature = dense["voltage_V"], dense["temperature_K"]
        # The duplicate time at the step boundary deliberately retains both
        # sides of the current switch.  A zero-width jump adds no energy/time.
        result = {
            "requested_current_A": float(current),
            "applied_current_A": float(current),
            "interval_duration_s": float(duration),
            "time_s": np.r_[pack_time, pack_time + dense["time_s"]].tolist(),
            "voltage_V": np.r_[self.voltage, voltage].tolist(),
            "temperature_K": np.r_[self.temperature, temperature].tolist(),
            "soc": np.r_[self.soc, dense["soc"]].tolist(),
            "mean_voltage_V": float(np.trapz(voltage, dense["time_s"]) / duration),
            "max_voltage_V": float(max(self.voltage, voltage.max())),
            "max_temperature_K": float(max(self.temperature, temperature.max())),
            "max_soc": float(max(self.soc, dense["soc"].max())),
            "pre_switch_voltage_V": float(self.voltage),
            "post_switch_voltage_V": float(voltage[0]),
            "pre_switch_state": before,
        }
        self.sol = sol.last_state
        self.current_A = float(current)
        self.update()
        return result


class Pack:
    """Independent cell plants sharing a synchronized experiment clock.

    Currents are nonnegative per-cell charging currents; the original sources
    contain no validated electrical balancer topology or converter loss model.
    A numerical failure is absorbing. Reaching the SOC goal is reported but
    does not prohibit a caller from observing a later holding interval.
    """

    def __init__(self, vectors, initial_soc, ambient_K=298.15,
                 decision_s=DT, model_options=None):
        self.vectors = np.asarray(vectors, dtype=float).copy()
        self.initial_soc = np.asarray(initial_soc, dtype=float).copy()
        if (self.vectors.ndim != 2 or self.vectors.shape[1] != 8
                or self.initial_soc.shape != (len(self.vectors),)
                or not np.isfinite(self.vectors).all()
                or not np.isfinite(self.initial_soc).all()
                or np.any(self.initial_soc <= 0.0)
                or np.any(self.initial_soc >= 1.0)):
            raise ValueError("Expected N eight-parameter vectors and N SOCs in (0,1)")
        if not np.isfinite(ambient_K) or not 250.0 < ambient_K < 330.0:
            raise ValueError("Ambient temperature must be in (250,330) K")
        if decision_s <= 0 or not np.isfinite(decision_s):
            raise ValueError("decision_s must be positive and finite")
        self.n = len(self.initial_soc)
        if self.n < 1:
            raise ValueError("At least one cell is required")
        self.ambient_K = float(ambient_K)
        self.decision_s = float(decision_s)
        options = dict(
            rtol=1e-7, atol=1e-9, var_pts=dict(MESH), sample_dt_s=1.0,
            numerical_failure_penalty=-2000.0, physical_interlock=False,
            physical_soc_max=0.95,
        )
        provided = dict(model_options or {})
        if "failure_penalty" in provided:
            provided["numerical_failure_penalty"] = provided.pop("failure_penalty")
        unknown = set(provided).difference(options)
        if unknown:
            raise ValueError(f"Unknown model_options: {sorted(unknown)}")
        options.update(provided)
        if not 0.0 < float(options["sample_dt_s"]) <= 1.0:
            raise ValueError("Constraint audit sampling must be in (0,1] seconds")
        if min(float(options["rtol"]), float(options["atol"])) <= 0:
            raise ValueError("Solver tolerances must be positive")
        if float(options["numerical_failure_penalty"]) > 0:
            raise ValueError("A numerical failure penalty cannot be positive")
        if not isinstance(options["physical_interlock"], (bool, np.bool_)):
            raise ValueError("physical_interlock must be boolean")
        if not TARGET <= float(options["physical_soc_max"]) < 1.0:
            raise ValueError("physical_soc_max must be in [0.9,1)")
        self.model_options = options
        self.failure_penalty = float(options["numerical_failure_penalty"])
        self.cells = [
            _Cell(v, s, self.ambient_K, options)
            for v, s in zip(self.vectors, self.initial_soc)
        ]
        self.reset()

    def reset(self):
        for cell in self.cells:
            cell.reset()
        self.time_s = 0.0
        self.time = 0.0  # Convenient compatibility alias, updated with time_s.
        self.failed = False
        self.protective_stop = False
        self.failure_reason = ""
        return self.obs()

    def physical(self):
        return np.array([[c.soc, c.voltage, c.temperature] for c in self.cells])

    def obs(self):
        return ((self.physical() - [0.5, 3.5, 308.0])
                / [0.5, 1.0, 11.0]).astype(np.float32)

    def mask(self):
        mask = np.ones((self.n, len(ACTIONS)), dtype=bool)
        for i, cell in enumerate(self.cells):
            if cell.soc >= 0.95:
                mask[i, 1:] = False
            elif cell.voltage >= VMAX or cell.temperature >= TMAX:
                mask[i, ACTIONS > 1.0] = False
        return mask

    def _proposals(self, currents, requested_duration):
        trials = [cell.propose(a, requested_duration)
                  for cell, a in zip(self.cells, currents)]
        durations = np.array([
            float(sol.t[-1] - cell.sol.t[-1])
            for cell, sol in zip(self.cells, trials)
        ])
        if (not np.isfinite(durations).all()
                or np.any(durations < -_TIME_TOL)
                or np.any(durations > requested_duration + _TIME_TOL)):
            raise _NumericalFailure("Invalid solver interval duration")
        actual = max(0.0, float(durations.min()))
        reasons = [str(sol.termination) for sol in trials
                   if str(sol.termination) != "final time"]
        failed = bool(actual < requested_duration - _TIME_TOL or reasons)
        if failed and actual > _TIME_TOL:
            # Resolve cells that initially reached a later time from their OLD
            # states to the earliest event. No future state is extrapolated back.
            for i, (cell, a) in enumerate(zip(self.cells, currents)):
                if durations[i] > actual + _TIME_TOL:
                    trials[i] = cell.propose(a, actual)
        if actual <= _TIME_TOL:
            actual = 0.0
            trials = [c.sol for c in self.cells]
        else:
            for cell, trial in zip(self.cells, trials):
                if abs(float(trial.t[-1] - cell.sol.t[-1]) - actual) > _TIME_TOL:
                    raise _NumericalFailure("Cells could not synchronize at the first event")
                dense = cell.trajectory(trial)
                if abs(dense["time_s"][-1] - actual) > _TIME_TOL:
                    raise _NumericalFailure("Trajectory duration differs from solver duration")
        reason = "; ".join(reasons) if reasons else (
            "Model terminated before the requested endpoint" if failed else ""
        )
        return trials, actual, failed, reason

    def step(self, currents, duration_s=None):
        currents = _validated_currents(currents, self.n)
        requested = self.decision_s if duration_s is None else float(duration_s)
        if requested <= 0.0 or not np.isfinite(requested):
            raise ValueError("duration_s must be positive and finite")
        if self.failed:
            raise RuntimeError("The failed pack is absorbing; reset before stepping again")
        start = self.time_s
        try:
            trials, duration, failure, reason = self._proposals(currents, requested)
        except (pybamm.SolverError, _NumericalFailure) as error:
            # No synchronized valid interval is available. Keep the last valid
            # states, explicitly fail at the last valid time, and never pretend
            # the requested ninety seconds were simulated.
            trials = [c.sol for c in self.cells]
            duration, failure, reason = 0.0, True, str(error)[:1000]
        stats = [c.commit(s, a, duration, start)
                 for c, s, a in zip(self.cells, trials, currents)]
        if duration > 0.0:
            # A cell stopped by an event keeps its native truncated grid; the
            # other cells were re-solved to that exact time and may have a
            # different grid. Export their union so pack SOC statistics compare
            # simultaneous samples. Every original peak is retained, and all
            # interpolation stays inside each cell's actually solved interval.
            common_time = np.unique(np.concatenate([
                np.asarray(stat["time_s"][1:], dtype=float) for stat in stats
            ]))
            for stat in stats:
                native_time = np.asarray(stat["time_s"][1:], dtype=float)
                if (common_time[0] < native_time[0] - _TIME_TOL
                        or common_time[-1] > native_time[-1] + _TIME_TOL):
                    raise _NumericalFailure("Shared audit grid exceeds a solved interval")
                for key in ("soc", "voltage_V", "temperature_K"):
                    pre_switch = float(stat[key][0])
                    post_switch = np.asarray(stat[key][1:], dtype=float)
                    stat[key] = np.r_[pre_switch, np.interp(
                        common_time, native_time, post_switch
                    )].tolist()
                stat["time_s"] = np.r_[start, common_time].tolist()
        self.time_s += duration
        self.time = self.time_s
        self.failed = bool(failure)
        self.failure_reason = reason
        self.protective_stop = bool(failure and "Protective" in reason)
        states = self.physical()
        terms = dict(time=0.0, balance=0.0, voltage=0.0, temperature=0.0,
                     numerical_failure=self.failure_penalty if failure and not self.protective_stop else 0.0,
                     protective_stop=self.failure_penalty if self.protective_stop else 0.0)
        if duration > 0.0:
            count = max(2, int(math.ceil(duration / self.model_options["sample_dt_s"])) + 1)
            grid = np.linspace(start, self.time_s, count)
            # np.interp selects the post-switch value at a duplicate boundary.
            # A jump has zero duration; both values remain in the audit output.
            soc = np.array([np.interp(grid, s["time_s"], s["soc"]) for s in stats])
            voltage = np.array([np.interp(grid, s["time_s"], s["voltage_V"]) for s in stats])
            temperature = np.array([np.interp(grid, s["time_s"], s["temperature_K"]) for s in stats])
            # Fixed 90-second normalization preserves reward units when E2
            # compares different decision periods (e.g. 45 s versus 90 s).
            terms["time"] = -0.75 * duration / DT
            terms["balance"] = float(-50.0 * np.trapz(
                np.maximum(soc.std(axis=0) - BETA, 0.0), grid) / DT)
            terms["voltage"] = float(-20.0 * np.trapz(
                np.maximum(voltage - VMAX, 0.0).sum(axis=0), grid) / DT)
            terms["temperature"] = float(-2.0 * np.trapz(
                np.maximum(temperature - TMAX, 0.0).sum(axis=0), grid) / DT)
        goal = bool(states[:, 0].min() >= TARGET and states[:, 0].std() <= BETA)
        return {
            "reward": float(sum(terms.values())), "reward_terms": terms,
            "goal": goal, "failure": bool(failure), "failure_reason": reason,
            "protective_stop": self.protective_stop,
            "numerical_failure": bool(failure and not self.protective_stop),
            "post_switch_peak_unknown": bool(failure and duration == 0.0),
            "terminal": bool(goal or failure), "time_s": float(self.time_s),
            "interval_duration_s": float(duration),
            "requested_duration_s": requested, "states": states,
            "stats": stats,
        }


class SafetyController:
    """Nominal prediction with observed discrepancy and finite backup screening.

    For every cell, candidates descend from the requested action toward zero.
    A candidate must satisfy tightened V/T/SOC limits throughout its sampled
    next interval AND a following zero-current interval. Conservative additive
    biases use positive observed-minus-nominal V/T and absolute SOC discrepancy.
    These are heuristic uncertainty margins, not proven worst-case error bounds.
    """

    def __init__(self, nominal_vectors, initial_soc, ambient_K=298.15,
                 decision_s=DT, voltage_margin_V=0.03,
                 temperature_margin_K=1.0, soc_max=0.95,
                 model_options=None):
        self.nominal_vectors = np.asarray(nominal_vectors, dtype=float).copy()
        self.initial_soc = np.asarray(initial_soc, dtype=float).copy()
        self.ambient_K = float(ambient_K)
        self.decision_s = float(decision_s)
        self.voltage_margin_V = float(voltage_margin_V)
        self.temperature_margin_K = float(temperature_margin_K)
        self.soc_max = float(soc_max)
        if not 0 <= self.voltage_margin_V < 0.5:
            raise ValueError("voltage_margin_V must be in [0,0.5)")
        if not 0 <= self.temperature_margin_K < 20.0:
            raise ValueError("temperature_margin_K must be in [0,20)")
        if not TARGET <= self.soc_max < 1.0:
            raise ValueError("soc_max must be in [0.9,1)")
        self.model_options = dict(model_options or {})
        if self.model_options.get("physical_interlock", False):
            raise ValueError("The nominal predictor must not use plant physical-interlock events")
        self.nominal = Pack(
            self.nominal_vectors, self.initial_soc, self.ambient_K,
            self.decision_s, self.model_options,
        )
        self.observer_failed = False
        self.observer_failure_reason = ""
        self.last_details = None

    def reset(self, initial_soc=None, ambient_K=None):
        desired = self.initial_soc if initial_soc is None else np.asarray(initial_soc, dtype=float)
        ambient = self.ambient_K if ambient_K is None else float(ambient_K)
        if (desired.shape != self.initial_soc.shape
                or not np.array_equal(desired, self.initial_soc)
                or ambient != self.ambient_K):
            self.initial_soc = desired.copy()
            self.ambient_K = ambient
            self.nominal = Pack(
                self.nominal_vectors, self.initial_soc, self.ambient_K,
                self.decision_s, self.model_options,
            )
        else:
            self.nominal.reset()
        self.observer_failed = False
        self.observer_failure_reason = ""
        self.last_details = None

    def filter(self, requested, observation):
        start_wall = time.perf_counter()
        requested = _validated_currents(requested, self.nominal.n)
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.nominal.n, 3) or not np.isfinite(observation).all():
            raise ValueError("Safety observation must be a finite N x 3 SOC/V/T array")
        nominal_state = self.nominal.physical()
        bias_soc = np.abs(observation[:, 0] - nominal_state[:, 0])
        bias_v = np.maximum(observation[:, 1] - nominal_state[:, 1], 0.0)
        bias_t = np.maximum(observation[:, 2] - nominal_state[:, 2], 0.0)
        applied = np.zeros_like(requested)
        details = {
            "requested_current_A": requested.tolist(),
            "applied_current_A": applied.tolist(),
            "intervention": False, "infeasible": False, "reason": "",
            "voltage_margin_V": self.voltage_margin_V,
            "temperature_margin_K": self.temperature_margin_K,
            "soc_max": self.soc_max,
            "voltage_guard_V": VMAX - self.voltage_margin_V,
            "temperature_guard_K": TMAX - self.temperature_margin_K,
            "nominal_state": nominal_state.tolist(),
            "observation": observation.tolist(),
            "positive_voltage_bias_V": bias_v.tolist(),
            "positive_temperature_bias_K": bias_t.tolist(),
            "absolute_soc_bias": bias_soc.tolist(),
            "cells": [], "candidate_solve_count": 0,
            "screen_horizon_s": self.decision_s,
            "zero_current_backup_horizon_s": self.decision_s,
            "safety_guarantee": "finite-horizon sampled screen only",
        }
        if self.observer_failed:
            details["infeasible"] = True
            details["reason"] = "Nominal observer unavailable: " + self.observer_failure_reason
        else:
            for i, (cell, command) in enumerate(zip(self.nominal.cells, requested)):
                cell_details = {"cell_index": i, "candidate_count": 0,
                                "feasible": False, "rejections": []}
                candidates = ACTIONS[ACTIONS <= command + 1e-10][::-1]
                for candidate in candidates:
                    cell_details["candidate_count"] += 1
                    try:
                        details["candidate_solve_count"] += 1
                        trial = cell.propose(float(candidate), self.decision_s)
                        first = cell.trajectory(trial)
                        if (str(trial.termination) != "final time"
                                or first["time_s"][-1] < self.decision_s - _TIME_TOL):
                            raise _NumericalFailure("Candidate reached a numerical stop")
                        first_peaks = {
                            "voltage_V": float(first["voltage_V"].max()),
                            "temperature_K": float(first["temperature_K"].max()),
                            "soc": float(first["soc"].max()),
                        }
                        first_adjusted = {
                            "voltage_V": first_peaks["voltage_V"] + float(bias_v[i]),
                            "temperature_K": first_peaks["temperature_K"] + float(bias_t[i]),
                            "soc": first_peaks["soc"] + float(bias_soc[i]),
                        }
                        early_limits = {
                            "voltage_V": VMAX - self.voltage_margin_V,
                            "temperature_K": TMAX - self.temperature_margin_K,
                            "soc": self.soc_max,
                        }
                        early_exceeded = [key for key in early_limits
                                          if first_adjusted[key] > early_limits[key]]
                        if early_exceeded:
                            # Already infeasible in the requested interval: a
                            # second solve cannot make this candidate feasible.
                            cell_details["rejections"].append({
                                "current_A": float(candidate),
                                "nominal_next_peaks": first_peaks,
                                "nominal_backup_peaks": None,
                                "bias_adjusted_peaks": first_adjusted,
                                "reason": "next_interval:" + ",".join(early_exceeded),
                            })
                            continue
                        details["candidate_solve_count"] += 1
                        backup = cell.propose(0.0, self.decision_s,
                                              old_solution=trial.last_state)
                        second = cell.trajectory(backup, origin=float(trial.t[-1]))
                        if (str(backup.termination) != "final time"
                                or second["time_s"][-1] < self.decision_s - _TIME_TOL):
                            raise _NumericalFailure("Zero-current backup reached a numerical stop")
                        backup_peaks = {
                            "voltage_V": float(second["voltage_V"].max()),
                            "temperature_K": float(second["temperature_K"].max()),
                            "soc": float(second["soc"].max()),
                        }
                        peaks = {key: max(first_peaks[key], backup_peaks[key])
                                 for key in first_peaks}
                        conservative = {
                            "voltage_V": peaks["voltage_V"] + float(bias_v[i]),
                            "temperature_K": peaks["temperature_K"] + float(bias_t[i]),
                            "soc": peaks["soc"] + float(bias_soc[i]),
                        }
                        exceeded = []
                        if conservative["voltage_V"] > VMAX - self.voltage_margin_V:
                            exceeded.append("voltage")
                        if conservative["temperature_K"] > TMAX - self.temperature_margin_K:
                            exceeded.append("temperature")
                        if conservative["soc"] > self.soc_max:
                            exceeded.append("soc")
                        record = {
                            "current_A": float(candidate), "nominal_next_peaks": first_peaks,
                            "nominal_backup_peaks": backup_peaks,
                            "bias_adjusted_peaks": conservative,
                        }
                        if exceeded:
                            record["reason"] = ",".join(exceeded)
                            cell_details["rejections"].append(record)
                            continue
                        applied[i] = float(candidate)
                        cell_details.update(record)
                        cell_details["feasible"] = True
                        break
                    except (pybamm.SolverError, _NumericalFailure) as error:
                        cell_details["rejections"].append({
                            "current_A": float(candidate),
                            "reason": "nominal_prediction_failure: " + str(error)[:500],
                        })
                if not cell_details["feasible"]:
                    details["infeasible"] = True
                details["cells"].append(cell_details)
        if details["infeasible"]:
            # This is a diagnostic shutdown request, NOT a certified safe zero
            # action. The caller must fail/terminate without stepping the plant.
            applied[:] = 0.0
            if not details["reason"]:
                details["reason"] = "No screened feasible current for one or more cells; zero is not certified safe"
        elif np.any(np.abs(applied - requested) > 1e-10):
            details["reason"] = "Reduced requested current by nominal V/T/SOC and backup screening"
        else:
            details["reason"] = "Requested currents passed nominal screening"
        details["applied_current_A"] = applied.tolist()
        details["intervention"] = bool(np.any(np.abs(applied - requested) > 1e-10))
        details["filter_wall_time_s"] = float(time.perf_counter() - start_wall)
        self.last_details = details
        return applied, details

    def advance(self, applied, duration_s=None):
        """Advance only this controller's nominal state after plant execution.

        The caller must pass the actually executed duration, including partial
        numerical intervals. An observer failure is returned and must be logged.
        """
        currents = _validated_currents(applied, self.nominal.n)
        duration = self.decision_s if duration_s is None else float(duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("Observer duration must be nonnegative and finite")
        if self.observer_failed:
            return {"failure": True, "failure_reason": self.observer_failure_reason,
                    "interval_duration_s": 0.0, "time_s": self.nominal.time_s}
        if duration == 0.0:
            return {"failure": False, "failure_reason": "", "interval_duration_s": 0.0,
                    "time_s": self.nominal.time_s}
        result = self.nominal.step(currents, duration_s=duration)
        if result["failure"]:
            self.observer_failed = True
            self.observer_failure_reason = result["failure_reason"]
        return {
            "failure": bool(result["failure"]),
            "failure_reason": result["failure_reason"],
            "interval_duration_s": result["interval_duration_s"],
            "time_s": result["time_s"], "states": result["states"],
        }
