"""Shared electrochemical model; units and source assumptions are explicit.

The user runs this module through a launcher. It has not been executed during
code delivery. Source parameters are preserved separately in original_parameters.json.
"""
import os
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import json
from pathlib import Path

import numpy as np
import pybamm

BASE = json.loads((Path(__file__).parent / "original_parameters.json").read_text(encoding="utf-8"))
KEYS = ["Positive electrode diffusivity [m2.s-1]", "Negative electrode diffusivity [m2.s-1]",
        "Positive particle radius [m]", "Negative particle radius [m]",
        "Negative electrode active material volume fraction", "Positive electrode active material volume fraction"]
SCALES = np.array([1e-15, 1e-14, 1e-6, 1e-6, .1, .1], dtype=float)
HEAT_KEYS = [name + " specific heat capacity [J.kg-1.K-1]" for name in
             ["Negative current collector", "Positive current collector", "Negative electrode",
              "Positive electrode", "Separator"]]
INITIAL = np.r_[np.array([BASE[key] for key in KEYS]) / SCALES, 25., 1.]
# Radii remain fixed. These bounds contain the original active-material fractions.
# This is an explicitly declared reconstruction prior, not a measured confidence interval.
BOUNDS = np.array([[5., 6.], [4., 5.], [3.8, 5.], [3.8, 5.],
                   [4.5, 7.5], [4., 6.6], [5., 50.], [.4, 2.5]])
FREE = np.array([0, 1, 4, 5, 6, 7], dtype=int)
MESH = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 10, "r_p": 10}
REFINED_MESH = {"x_n": 60, "x_s": 60, "x_p": 60, "r_n": 20, "r_p": 20}


def parameter_values(vector=None, temperature=298.15, ambient=298.15):
    p = pybamm.ParameterValues("Chen2020")
    p.update(BASE)
    p.update({"Ambient temperature [K]": float(ambient),
              "Initial temperature [K]": float(temperature), "Current function [A]": 0.})
    if vector is not None:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (8,) or not np.isfinite(vector).all():
            raise ValueError("A finite eight-component parameter vector is required")
        p.update(dict(zip(KEYS, vector[:6] * SCALES)))
        p["Total heat transfer coefficient [W.m-2.K-1]"] = float(vector[6])
        for key in HEAT_KEYS:
            p[key] = BASE[key] * float(vector[7])
    return p


def values(sol, key):
    """Spatial average of PyBaMM's evaluated variable, keeping its time axis."""
    return np.asarray(sol[key].entries).reshape(-1, len(sol.t)).mean(axis=0)


class FitSimulator:
    """Repeated constant-current discharge with fixed geometry and explicit inputs.

    Positive current denotes discharge in PyBaMM. Initial electrode concentrations
    follow the supplied source/Chen2020 defaults, without an initial_soc=1 reset.
    """
    def __init__(self, family="SPMe", mesh=None, rtol=1e-6, atol=1e-8):
        if family not in ("SPM", "SPMe"):
            raise ValueError("family must be SPM or SPMe")
        p = parameter_values()
        dynamic = [KEYS[i] for i in (0, 1, 4, 5)] + HEAT_KEYS + [
            "Total heat transfer coefficient [W.m-2.K-1]", "Current function [A]",
            "Initial temperature [K]", "Ambient temperature [K]"]
        for key in dynamic:
            p[key] = "[input]"
        model = getattr(pybamm.lithium_ion, family)({"thermal": "lumped"})
        self.sim = pybamm.Simulation(model, parameter_values=p,
            solver=pybamm.CasadiSolver(mode="safe", rtol=rtol, atol=atol), var_pts=mesh or MESH)
        self.sim.build()

    def solve(self, vector, current, duration, temperature=298.15, ambient=298.15, points=301):
        vector = np.asarray(vector, dtype=float)
        if not np.allclose(vector[2:4], INITIAL[2:4], atol=1e-12, rtol=0):
            raise ValueError("Particle radii are fixed to source values in the cached mesh")
        if float(duration) <= 0 or int(points) < 2:
            raise ValueError("duration must be positive and points at least two")
        inputs = {KEYS[i]: float(vector[i] * SCALES[i]) for i in (0, 1, 4, 5)}
        inputs.update({"Current function [A]": float(current), "Initial temperature [K]": float(temperature),
                       "Ambient temperature [K]": float(ambient),
                       "Total heat transfer coefficient [W.m-2.K-1]": float(vector[6])})
        inputs.update({key: BASE[key] * float(vector[7]) for key in HEAT_KEYS})
        sol = self.sim.solve(np.linspace(0., float(duration), int(points)), inputs=inputs)
        times = np.asarray(sol.t, dtype=float)
        voltage = values(sol, "Voltage [V]")
        temp = values(sol, "X-averaged cell temperature [K]") - 273.15
        if not (np.isfinite(times).all() and np.isfinite(voltage).all() and np.isfinite(temp).all()):
            raise ValueError("Nonfinite solver output")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Solver output times are not strictly increasing")
        return times, voltage, temp
