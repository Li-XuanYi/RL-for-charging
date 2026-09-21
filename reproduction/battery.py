"""Reconstructed SPMe environment. This is a simulation, not a hardware controller."""
import os
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import numpy as np
import pybamm

DT, BETA, TARGET, VMAX, TMAX = 90.0, 0.02, 0.9, 4.2, 309.0
ACTIONS = np.arange(0.0, 7.5001, 0.5)
KEYS = ["Positive electrode diffusivity [m2.s-1]",
        "Negative electrode diffusivity [m2.s-1]",
        "Positive particle radius [m]", "Negative particle radius [m]",
        "Negative electrode active material volume fraction",
        "Positive electrode active material volume fraction"]
SCALES = np.array([1e-15, 1e-14, 1e-6, 1e-6, .1, .1])
# First six search bounds are from the supplied PSO script. The last two are
# explicitly reconstructed thermal fit parameters, absent from that script.
BOUNDS = np.array([[5, 6], [4, 5], [4, 5], [4, 5], [6, 6.5], [5, 5.8], [5, 50], [.5, 5]])
INITIAL = np.array([5.77, 4.30, 4.06, 4.16, 6.1, 5.17, 25., 2.])


def parameters(vector, initial_temperature=298.15):
    p = pybamm.ParameterValues("Chen2020")
    if vector is not None:
        p.update(dict(zip(KEYS, np.asarray(vector[:6]) * SCALES)))
        p["Total heat transfer coefficient [W.m-2.K-1]"] = float(vector[6])
        for name in ("Negative current collector", "Positive current collector",
                     "Negative electrode", "Positive electrode", "Separator"):
            key = name + " specific heat capacity [J.kg-1.K-1]"
            p[key] = p[key] * float(vector[7])
    p.update({"Ambient temperature [K]": 298.15,
              "Initial temperature [K]": float(initial_temperature),
              "Upper voltage cut-off [V]": VMAX,
              "Lower voltage cut-off [V]": 2.5})
    return p


def series(sol, name):
    data = np.asarray(sol[name].entries)
    return data.reshape(-1, len(sol.t)).mean(axis=0)


def simulation(p):
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
    # An event protects each simulated interval; a rejected action is retried
    # from the same preceding state. This extra filter is NOT in the paper.
    model.events.append(pybamm.Event("Reproduction temperature limit",
                                    TMAX - model.variables["X-averaged cell temperature [K]"]))
    return pybamm.Simulation(model, parameter_values=p,
        solver=pybamm.CasadiSolver(mode="safe", rtol=1e-5, atol=1e-7),
        var_pts={"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10})


def discharge(vector, duration, current, temperature, points=100):
    p = parameters(vector, temperature)
    p["Current function [A]"] = float(current)
    # No 309 K charging event during fitting of measured discharge curves.
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
    sim = pybamm.Simulation(model, parameter_values=p,
        solver=pybamm.CasadiSolver(mode="safe", rtol=1e-5, atol=1e-7),
        var_pts={"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10})
    sol = sim.solve(np.linspace(0, duration, points), initial_soc=1.0)
    return sol.t, series(sol, "Voltage [V]"), series(sol, "X-averaged cell temperature [K]") - 273.15


class Cell:
    def __init__(self, vector, soc):
        p = parameters(vector)
        p.set_initial_stoichiometries(0.0)
        self.c0 = float(p["Initial concentration in negative electrode [mol.m-3]"])
        p.set_initial_stoichiometries(1.0)
        self.c1 = float(p["Initial concentration in negative electrode [mol.m-3]"])
        p.set_initial_stoichiometries(float(soc))
        p["Current function [A]"] = "[input]"
        self.sim = simulation(p)
        self.sol = self.sim.solve([0, 1e-6], inputs={"Current function [A]": 0.0}).last_state
        self.update()
        self.soc = float(soc)

    def update(self):
        self.voltage = float(series(self.sol, "Voltage [V]")[-1])
        self.temperature = float(series(self.sol, "X-averaged cell temperature [K]")[-1])
        c = float(series(self.sol, "R-averaged negative particle concentration [mol.m-3]")[-1])
        self.soc = (c - self.c0) / (self.c1 - self.c0)

    def step(self, requested):
        old = self.sol
        # Reduce magnitude, never reverse the sign. MBA-RL supplies >= 0 only.
        candidates = np.arange(abs(requested), -.01, -.5) * np.sign(requested)
        if not len(candidates) or abs(candidates[-1]) > 1e-8:
            candidates = np.append(candidates, 0.)
        last_reason = ""
        for action in candidates:
            try:
                sol = self.sim.solver.step(old, self.sim.built_model, DT,
                    npts=10, inputs={"Current function [A]": -float(action)}, save=False)
                volt = series(sol, "Voltage [V]")
                temp = series(sol, "X-averaged cell temperature [K]")
                elapsed = sol.t[-1] - old.t[-1]
                if elapsed < DT - 1e-3 or np.max(volt) > VMAX + 1e-5 or np.max(temp) > TMAX + 1e-5:
                    last_reason = str(sol.termination)
                    continue
                self.sol = sol.last_state
                self.update()
                return float(action), float(np.mean(volt)), float(np.max(volt)), float(np.max(temp))
            except pybamm.SolverError as exc:
                last_reason = str(exc)
        raise RuntimeError("No feasible simulated current, including zero: " + last_reason)


class Pack:
    def __init__(self, vectors, initial):
        self.cells = [Cell(v, s) for v, s in zip(vectors, initial)]
        self.time = 0.

    def physical(self):
        return np.array([[c.soc, c.voltage, c.temperature] for c in self.cells])

    def obs(self):
        x = self.physical()
        return ((x - np.array([.5, 3.5, 308.])) / np.array([.5, 1., 11.])).astype(np.float32)

    def mask(self):
        result = np.ones((3, len(ACTIONS)), dtype=bool)
        for i, cell in enumerate(self.cells):
            if cell.soc >= .95:
                result[i, 1:] = False
            elif cell.voltage >= VMAX - .01 or cell.temperature >= TMAX - .3:
                result[i, ACTIONS > .5] = False
        return result

    def step(self, currents):
        stats = [cell.step(float(a)) for cell, a in zip(self.cells, currents)]
        self.time += DT
        x = self.physical()
        std = float(x[:, 0].std())
        reward = -.75 - 50 * max(std - BETA, 0)
        reward -= 20 * np.maximum(x[:, 1] - VMAX, 0).sum()
        reward -= 2 * np.maximum(x[:, 2] - TMAX, 0).sum()
        done = bool(std <= BETA and np.all(x[:, 0] >= TARGET))
        return float(reward), done, np.asarray(stats)
