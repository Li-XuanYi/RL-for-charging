"""Experiment 1: source audit, seeded identification, temporal holdout, precision.

No data are fabricated. Heldout suffix prediction is explicitly NOT independent
experimental validation. The current assumption is declared before optimization.
"""
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybamm

from common import dump, write_csv, sha256
from model import (FitSimulator, INITIAL, BOUNDS, FREE, KEYS, SCALES, MESH,
                   REFINED_MESH, BASE)

CELL_NAMES = ("begin", "middle", "age")
FAILURE_SCORE = 1e12
PARAMETER_NAMES = KEYS + ["Total heat transfer coefficient [W.m-2.K-1]", "Heat capacity multiplier"]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _dump(path, data):
    dump(path, _plain(data))


def _csv(path, rows):
    write_csv(path, [_plain(row) for row in rows])


def _project_data(root):
    candidates = list(root.glob("multi-balancing-agent*/multi-balancing-agent*/data_true/data_image"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one supplied data_image directory; found {len(candidates)}")
    return candidates[0]


def _audit(root, out, cfg):
    source = _project_data(root)
    metadata_path = root / "data_metadata.json"
    configured_metadata = cfg.get("metadata_file", cfg.get("metadata_path"))
    if configured_metadata:
        metadata_path = Path(configured_metadata)
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("data_metadata.json must contain a JSON object")
    records, datasets, problems = [], [], []
    for number, name in enumerate(CELL_NAMES, 1):
        detail = dict(metadata)
        detail.update(metadata.get("cells", {}).get(name, {}))
        # A template uses null for unconfirmed facts. Null is not confirmation,
        # but can use the same explicit simulation assumptions as an absent file.
        def known(key, fallback):
            value = detail.get(key)
            return fallback if value is None or value == "" else value

        dt = float(known("time_step_s", 1.))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("time_step_s must be finite and positive")
        v_unit = known("voltage_unit", "V")
        t_unit = known("temperature_unit", "degC")
        if v_unit not in ("V", "mV") or t_unit not in ("degC", "C", "K"):
            raise ValueError("Supported metadata units: V/mV and degC/C/K")
        files = [source / ("data_1C_" + name), source / ("temp_data 1C " + name)]
        v, temp = [np.loadtxt(file, dtype=float) for file in files]
        if v.ndim != 1 or temp.ndim != 1 or len(v) != len(temp) or len(v) < 30:
            raise ValueError(f"{name}: expected paired one-dimensional arrays of equal length >=30")
        if not np.isfinite(v).all() or not np.isfinite(temp).all():
            raise ValueError(f"{name}: raw measurements contain nonfinite values; no automatic imputation permitted")
        if v_unit == "mV":
            v = v / 1000.
        if t_unit == "K":
            temp = temp - 273.15
        if not ((v > 0).all() and (v < 6).all() and (temp > -80).all() and (temp < 150).all()):
            raise ValueError(f"{name}: voltage/temperature fail broad unit plausibility bounds; confirm metadata")
        ambient = float(known("ambient_temperature_K", cfg.get("ambient_temperature_K", 298.15)))
        if not np.isfinite(ambient) or not 200 < ambient < 400:
            raise ValueError("ambient_temperature_K must be a plausible finite Kelvin temperature")
        times = np.arange(len(v), dtype=float) * dt
        required = ["time_step_s", "voltage_unit", "temperature_unit", "current_A", "ambient_temperature_K",
                    "cell_identity", "capacity_Ah", "measurement_source", "initialization_protocol"]
        missing = [key for key in required if key not in detail or detail[key] in (None, "")]
        confirmed_current = detail.get("current_A")
        if confirmed_current is not None and (not np.isfinite(float(confirmed_current)) or float(confirmed_current) <= 0):
            raise ValueError("Discharge metadata current_A must be positive")
        if confirmed_current is not None and not np.isclose(float(confirmed_current), float(cfg["current_for_control"]), rtol=0., atol=1e-12):
            problems.append(f"Cell {name}: supplied metadata current {confirmed_current} A differs from configured "
                            f"control calibration current {cfg['current_for_control']} A")
        record = {"cell": number, "name": name, "samples": len(v), "duration_s": times[-1],
            "voltage_file": str(files[0]), "temperature_file": str(files[1]),
            "voltage_sha256": sha256(files[0]), "temperature_sha256": sha256(files[1]),
            "sampling_interval_s": dt, "voltage_input_unit": v_unit, "temperature_input_unit": t_unit,
            "voltage_range_V": [v.min(), v.max()], "temperature_range_C": [temp.min(), temp.max()],
            "ambient_temperature_K": ambient, "metadata_missing_fields": missing,
            "metadata_status": "PROVIDED_NOT_INDEPENDENTLY_VERIFIED" if not missing else "UNCONFIRMED",
            "metadata_values": detail,
            "time_axis": "synthetic equally spaced indices; timestamps are not present in source arrays",
            "independent_cycles_available": False}
        records.append(record)
        datasets.append({"cell": number, "name": name, "time": times, "voltage": v, "temperature": temp,
                         "initial_temperature_K": temp[0] + 273.15, "ambient_temperature_K": ambient})
        _csv(out / f"raw_cell_{number}.csv", [{"time_s": t, "voltage_V": vv, "temperature_C": tt}
             for t, vv, tt in zip(times, v, temp)])
    audit = {"source_directory": str(source), "metadata_file": str(metadata_path),
        "metadata_file_sha256": sha256(metadata_path) if metadata_path.is_file() else None,
        "records": records, "conflicts": problems, "array_integrity_passed": True,
        "metadata_fields_complete": all(not r["metadata_missing_fields"] for r in records),
        "metadata_independently_verified": False,
        "current_note": "Source script uses 5 A; manuscript states 4 Ah at 1 C. A smaller fitted error cannot identify the true experimental current.",
        "model_note": "Supplied simulator is SPMe with lumped thermal dynamics, using modified Chen2020. This is not a measured Samsung40T parameter validation.",
        "soc_note": "Identification preserves source/Chen2020 initial electrode concentrations; subsequent control uses stoichiometric SOC initialization, not voltage-as-SOC.",
        "temperature_note": "First measured temperature initializes the model. Ambient is metadata or explicit 298.15 K assumption; sensor location/uncertainty are unavailable."}
    _dump(out / "data_audit.json", audit)
    if problems:
        raise ValueError("Metadata conflict must be resolved before calibration: " + "; ".join(problems))
    return datasets, audit


def _predict(sim, vector, current, data, indices=None, dense=True):
    times = data["time"] if indices is None else data["time"][indices]
    st, sv, tt = sim.solve(vector, current, times[-1], data["initial_temperature_K"],
                           data["ambient_temperature_K"], points=len(times) if dense else min(401, len(times)))
    # np.interp would silently extend the final event value. Mask both endpoints
    # first, and never assign a prediction beyond the physical solver interval.
    covered = (times >= st[0] - 1e-8) & (times <= st[-1] + 1e-8)
    pv, pt = np.full(len(times), np.nan), np.full(len(times), np.nan)
    pv[covered] = np.interp(times[covered], st, sv)
    pt[covered] = np.interp(times[covered], st, tt)
    return {"time": times, "voltage": pv, "temperature": pt, "covered": covered,
            "model_end_s": float(st[-1]), "solver_points": len(st), "solver_failure": None}


def _safe_final_prediction(sim, vector, current, data, n_train):
    """Retain a failed suffix or comparison as a failed result, not missing work."""
    try:
        return _predict(sim, vector, current, data)
    except (pybamm.SolverError, ValueError, RuntimeError) as exc:
        n = len(data["time"])
        result = {"time": data["time"], "voltage": np.full(n, np.nan), "temperature": np.full(n, np.nan),
                  "covered": np.zeros(n, dtype=bool), "model_end_s": 0., "solver_points": 0,
                  "solver_failure": f"{type(exc).__name__}: {exc}"}
        if n_train < n:
            try:
                prefix = _predict(sim, vector, current, data, slice(0, n_train))
                for key in ("voltage", "temperature", "covered"):
                    result[key][:n_train] = prefix[key]
                result["model_end_s"] = prefix["model_end_s"]
                result["solver_points"] = prefix["solver_points"]
            except (pybamm.SolverError, ValueError, RuntimeError) as prefix_exc:
                result["solver_failure"] += f"; prefix retry failed: {type(prefix_exc).__name__}: {prefix_exc}"
        return result


def _residual(actual, predicted, mask):
    valid = mask & np.isfinite(predicted)
    error = predicted[valid] - actual[valid]
    if not len(error):
        return {"n": 0, "rmse": None, "mae": None, "bias": None, "max_abs": None}
    return {"n": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "max_abs": float(np.max(np.abs(error)))}


def _metrics(data, prediction, mask=None):
    mask = np.ones(len(data["time"]), dtype=bool) if mask is None else mask
    n = int(mask.sum())
    covered = prediction["covered"] & mask
    return {"requested_samples": n, "covered_samples": int(covered.sum()),
            "coverage_fraction": float(covered.sum() / n) if n else None,
            "full_coverage": bool(covered.sum() == n), "model_end_s": prediction["model_end_s"],
            "solver_failure": prediction.get("solver_failure"),
            "voltage_V": _residual(data["voltage"], prediction["voltage"], mask),
            "temperature_C": _residual(data["temperature"], prediction["temperature"], mask),
            "metric_scope": "covered samples only; uncovered samples are missing, never extrapolated"}


def _fit(sim, data, current, n_train, seed, cfg, target):
    target.mkdir(parents=True, exist_ok=True)
    low, high = BOUNDS[FREE].T
    rng = np.random.default_rng(seed)
    particles, iterations = int(cfg["particles"]), int(cfg["iterations"])
    positions = rng.uniform(low, high, (particles, len(FREE)))
    positions[0] = INITIAL[FREE]
    velocity = np.zeros_like(positions)
    personal, personal_score = positions.copy(), np.full(particles, FAILURE_SCORE)
    global_best, global_score = positions[0].copy(), FAILURE_SCORE
    history, failure_examples = [], []
    failed, evaluated = 0, 0
    train_slice = slice(0, n_train)
    actual_v, actual_t = data["voltage"][train_slice], data["temperature"][train_slice]
    loss_v, loss_t = float(cfg.get("loss_voltage_scale_V", .03)), float(cfg.get("loss_temperature_scale_C", .3))
    if not np.isfinite([loss_v, loss_t]).all() or loss_v <= 0 or loss_t <= 0:
        raise ValueError("Loss scales must be finite and positive")

    def score(z):
        nonlocal failed, evaluated
        evaluated += 1
        vector = INITIAL.copy()
        vector[FREE] = z
        try:
            pred = _predict(sim, vector, current, data, train_slice, dense=False)
            good = pred["covered"]
            if int(good.sum()) < 2:
                raise ValueError("Fewer than two covered fit samples")
            rv = np.sqrt(np.mean((pred["voltage"][good] - actual_v[good]) ** 2))
            rt = np.sqrt(np.mean((pred["temperature"][good] - actual_t[good]) ** 2))
            missing = 1. - good.mean()
            # Fixed coverage penalty prevents an optimizer from dropping the bad
            # end of a discharge curve. It is declared, not a measurement loss.
            result = (rv / loss_v) ** 2 + (rt / loss_t) ** 2 + 200. * missing + 1000. * missing ** 2
            if not np.isfinite(result):
                raise ValueError("Nonfinite objective")
            return float(result)
        except (pybamm.SolverError, ValueError, RuntimeError) as exc:
            failed += 1
            if len(failure_examples) < 20:
                failure_examples.append({"evaluation": evaluated, "type": type(exc).__name__, "reason": str(exc), "vector": vector.tolist()})
            return FAILURE_SCORE

    for iteration in range(iterations + 1):
        for particle in range(particles):
            cost = score(positions[particle])
            if cost < personal_score[particle]:
                personal_score[particle], personal[particle] = cost, positions[particle].copy()
            if cost < global_score:
                global_score, global_best = cost, positions[particle].copy()
        history.append({"iteration": iteration, "best_objective": global_score, "solver_failures": failed, "evaluations": evaluated})
        _dump(target / "progress.json", {"iteration": iteration, "iterations": iterations, "objective": global_score,
            "failed_candidates": failed, "evaluations": evaluated, "best_free_parameters": global_best})
        if iteration % 10 == 0 or iteration == iterations:
            print(f"E1 {data['name']} {current:g}A n={n_train} seed={seed}: PSO {iteration}/{iterations}, loss={global_score:.6g}", flush=True)
        if iteration == iterations:
            break
        inertia = .9 - .5 * iteration / max(1, iterations)
        velocity = (inertia * velocity + 1.5 * rng.random(positions.shape) * (personal - positions)
                    + 1.5 * rng.random(positions.shape) * (global_best - positions))
        velocity = np.clip(velocity, -.2 * (high - low), .2 * (high - low))
        positions = np.clip(positions + velocity, low, high)
    vector = INITIAL.copy()
    vector[FREE] = global_best
    result = {"seed": seed, "vector": vector.tolist(), "objective": global_score,
        "training_samples": n_train, "training_end_s": float(data["time"][n_train - 1]),
        "solver_failures": failed, "evaluations": evaluated, "failure_examples": failure_examples,
        "success": bool(global_score < FAILURE_SCORE), "selection_uses_suffix": False}
    _csv(target / "pso_history.csv", history)
    _dump(target / "fit.json", result)
    return result


def _segments(data, prediction, fraction):
    n = len(data["time"])
    split = int(n * fraction)
    fractions = np.arange(n) / max(n - 1, 1)
    masks = {"full": np.ones(n, dtype=bool), "early_0_20pct": fractions <= .2,
             "middle_20_80pct": (fractions > .2) & (fractions < .8), "late_80_100pct": fractions >= .8,
             "prefix_fit": np.arange(n) < split, "suffix_forecast": np.arange(n) >= split}
    return {name: _metrics(data, prediction, mask) for name, mask in masks.items()}


def _prediction_csv(path, data, pred, split, label):
    _csv(path, [{"time_s": float(t), "measured_voltage_V": float(v), "predicted_voltage_V": float(pv),
        "measured_temperature_C": float(temp), "predicted_temperature_C": float(pt),
        "covered": bool(cov), "split": "prefix" if i < split else "suffix", "prediction_kind": label}
        for i, (t, v, pv, temp, pt, cov) in enumerate(zip(data["time"], data["voltage"], pred["voltage"],
                                                       data["temperature"], pred["temperature"], pred["covered"]))])


def _precision(vector, current, data, out, cfg):
    reference = _safe_final_prediction(FitSimulator(), vector, current, data, len(data["time"]))
    variants = [("strict_tolerance", {"rtol": 1e-8, "atol": 1e-10}),
                ("refined_mesh_and_strict_tolerance", {"mesh": REFINED_MESH, "rtol": 1e-8, "atol": 1e-10})]
    thresholds = {"max_voltage_difference_V": .002, "max_temperature_difference_C": .05,
                  "max_end_time_difference_s": max(1., float(data["time"][1] - data["time"][0])),
                  "min_common_coverage_fraction": .95, **cfg.get("precision_thresholds", {})}
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in thresholds.values()):
        raise ValueError("Precision thresholds must be finite and positive")
    if thresholds["min_common_coverage_fraction"] > 1:
        raise ValueError("Minimum common coverage cannot exceed one")
    records = []
    for name, settings in variants:
        try:
            pred = _safe_final_prediction(FitSimulator(**settings), vector, current, data, len(data["time"]))
        except (pybamm.SolverError, ValueError, RuntimeError) as exc:
            records.append({"variant": name, "passed": False, "failure": f"{type(exc).__name__}: {exc}"})
            continue
        common = reference["covered"] & pred["covered"]
        if not common.any():
            records.append({"variant": name, "passed": False, "failure": "No common valid output times",
                            "reference_failure": reference.get("solver_failure"), "variant_failure": pred.get("solver_failure")})
            continue
        record = {"variant": name, "max_voltage_difference_V": float(np.max(np.abs(reference["voltage"][common] - pred["voltage"][common]))),
            "max_temperature_difference_C": float(np.max(np.abs(reference["temperature"][common] - pred["temperature"][common]))),
            "end_time_difference_s": abs(reference["model_end_s"] - pred["model_end_s"]),
            "common_coverage_fraction": float(common.mean()), "reference_coverage_fraction": float(reference["covered"].mean()),
            "variant_coverage_fraction": float(pred["covered"].mean())}
        record["passed"] = bool(record["max_voltage_difference_V"] <= thresholds["max_voltage_difference_V"]
            and record["max_temperature_difference_C"] <= thresholds["max_temperature_difference_C"]
            and record["end_time_difference_s"] <= thresholds["max_end_time_difference_s"]
            and record["common_coverage_fraction"] >= thresholds["min_common_coverage_fraction"])
        records.append(record)
        _prediction_csv(out / f"{name}.csv", data, pred, len(data["time"]), "numerical_precision_check")
    result = {"thresholds_predeclared": thresholds, "variants": records,
              "passed": all(x["passed"] for x in records),
              "interpretation": "Numerical convergence on the available discharge trajectory only; this does not establish physical accuracy or charging-domain validity."}
    _dump(out / "precision.json", result)
    return result


def _plot(path, data, full, prefix, spm, split, current):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    t = data["time"]
    for ax, residual_ax, key, measured_key, unit in [(axes[0, 0], axes[1, 0], "voltage", "voltage", "V"),
                                                   (axes[0, 1], axes[1, 1], "temperature", "temperature", "deg C")]:
        actual = data[measured_key]
        ax.plot(t, actual, "k", label="Measured", lw=1.2)
        for prediction, label, style in [(full, "SPMe full-curve fit (in-sample)", "-"),
                                       (prefix, "SPMe prefix-fit forecast", "--"),
                                       (spm, "SPM with SPMe-fit parameters", ":")]:
            ax.plot(t, prediction[key], style, label=label, lw=1.)
            residual_ax.plot(t, prediction[key] - actual, style, label=label, lw=1.)
        for target in (ax, residual_ax):
            target.axvline(t[split], color="gray", ls="--", lw=.8)
            target.axvspan(t[split], t[-1], color="gray", alpha=.08)
            target.grid(alpha=.2)
            target.set_xlabel("Time / s")
        ax.set_ylabel(f"{key.capitalize()} / {unit}")
        residual_ax.set_ylabel(f"Prediction - measurement / {unit}")
        residual_ax.axhline(0, color="k", lw=.6)
        ax.legend(fontsize=8)
    fig.suptitle(f"Cell {data['cell']} | current assumption {current:g} A | shaded suffix not used in prefix fit")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(root: Path, out: Path, cfg: dict):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"seeds": [7, 17, 27, 37, 47], "particles": 20, "iterations": 50,
           "train_fraction": .8, "currents": [4., 5.], "current_for_control": 5., **cfg}
    if (not isinstance(cfg["seeds"], list) or not cfg["seeds"]
            or any(type(seed) is not int or seed < 0 for seed in cfg["seeds"])
            or len(set(cfg["seeds"])) != len(cfg["seeds"])):
        raise ValueError("Provide distinct nonnegative integer optimizer seeds")
    if (type(cfg["particles"]) is not int or type(cfg["iterations"]) is not int
            or cfg["particles"] < 2 or cfg["iterations"] < 1
            or not .5 <= float(cfg["train_fraction"]) < .95):
        raise ValueError("Invalid PSO budget or chronological split")
    cfg["train_fraction"] = float(cfg["train_fraction"])
    if not isinstance(cfg["currents"], list) or not cfg["currents"] or any(isinstance(x, bool) for x in cfg["currents"]):
        raise ValueError("currents must be a nonempty list of numeric discharge currents")
    currents = [float(x) for x in cfg["currents"]]
    if len(set(currents)) != len(currents) or any(not math.isfinite(x) or x <= 0 for x in currents):
        raise ValueError("currents must be distinct finite positive discharge currents")
    if len({f"{x:g}" for x in currents}) != len(currents):
        raise ValueError("Current hypotheses are too close to produce unique current_A output directories")
    cfg["current_for_control"] = float(cfg["current_for_control"])
    if not math.isfinite(cfg["current_for_control"]) or sum(np.isclose(cfg["current_for_control"], x, rtol=0., atol=1e-12) for x in currents) != 1:
        raise ValueError("Predeclared current_for_control must match exactly one configured current")
    for key, default in (("loss_voltage_scale_V", .03), ("loss_temperature_scale_C", .3)):
        cfg[key] = float(cfg.get(key, default))
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for group, allowed in [
        ("precision_thresholds", {"max_voltage_difference_V", "max_temperature_difference_C", "max_end_time_difference_s", "min_common_coverage_fraction"}),
        ("model_screening_thresholds", {"minimum_coverage_fraction", "maximum_voltage_RMSE_V", "maximum_temperature_RMSE_C"})]:
        options = cfg.get(group, {})
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError(f"Unknown or invalid {group} keys")
        options = {key: float(value) for key, value in options.items()}
        if any(not math.isfinite(value) or value <= 0 for value in options.values()):
            raise ValueError(f"{group} values must be finite and positive")
        if any("coverage" in key and value > 1 for key, value in options.items()):
            raise ValueError(f"{group} coverage fractions cannot exceed one")
        cfg[group] = options
    protocol = {"config": cfg, "model_family": "SPMe", "parameter_source": "modified Chen2020 plus supplied hardcoded dictionary",
        "fixed_particle_radii_m": (INITIAL[2:4] * SCALES[2:4]).tolist(), "free_parameter_indices": FREE.tolist(),
        "parameter_names": PARAMETER_NAMES, "scaled_parameter_bounds": BOUNDS.tolist(),
        "objective": f"(voltage_RMSE/{float(cfg.get('loss_voltage_scale_V', .03))} V)^2 + (temperature_RMSE/{float(cfg.get('loss_temperature_scale_C', .3))} degC)^2 + 200*m + 1000*m^2; m is missing sample fraction",
        "candidate_solve_grid": "up to 401 equally spaced output points; final metrics use every measured timestamp",
        "selection": "Each current/cell/fit-prefix is independent. Select smallest training objective among optimizer seeds; never select by suffix error or current hypothesis.",
        "validation": "Chronological suffix is a forecast within the same discharge, not an independent cycle or cell validation. Original hardcoded priors may themselves have been derived from these same curves; prefix withholding does not erase that historical information.",
        "spm_comparison": "Model-form sensitivity using refitted SPMe parameters unchanged in SPM; not an independently optimized SPM baseline.",
        "initialization": "Source/Chen2020 initial electrode concentrations; first measured temperature; no initial_soc override",
        "solver": {"mode": "safe", "rtol": 1e-6, "atol": 1e-8, "mesh": MESH},
        "strict_solver": {"rtol": 1e-8, "atol": 1e-10, "refined_mesh": REFINED_MESH},
        "precision_thresholds_predeclared": {"max_voltage_difference_V": .002,
            "max_temperature_difference_C": .05, "max_end_time_difference_s": "max(1 s, one sampling interval)",
            "min_common_coverage_fraction": .95, **cfg.get("precision_thresholds", {})},
        "model_screening_thresholds_predeclared": {"minimum_coverage_fraction": .99,
            "maximum_voltage_RMSE_V": .05, "maximum_temperature_RMSE_C": 1., **cfg.get("model_screening_thresholds", {})},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pybamm": pybamm.__version__},
        "uncertainty": "Optimizer-seed spread quantifies optimization instability only. It is not a confidence interval on physical parameters or repeated measurements."}
    _dump(out / "protocol.json", protocol)
    datasets, audit = _audit(root, out, cfg)
    from fit_recovery import prepare as prepare_recovery, read_fit as read_recovered_fit
    recovery = prepare_recovery(root, out, cfg, audit)
    if recovery:
        protocol['optimizer_reuse'] = recovery
        protocol['selection'] += ' Recovery preserves the original coarse-mesh objectives and selected seeds; predictions are recalculated, not reoptimized.'
        _dump(out / 'protocol.json', protocol)
    sim = FitSimulator()
    spm_sim = FitSimulator(family="SPM")
    all_metrics, all_seed_rows, variability, selected = [], [], [], []
    comparison_rows = []
    for current in currents:
        current_dir = out / f"current_{current:g}A"
        for data in datasets:
            cell_dir = current_dir / f"cell_{data['cell']}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            n, split = len(data["time"]), int(len(data["time"]) * float(cfg["train_fraction"]))
            chosen, predictions = {}, {}
            for kind, n_train in [("full_fit", n), ("prefix_fit", split)]:
                fits = [read_recovered_fit(recovery,current,data['cell'],kind,int(seed),n_train,cell_dir / kind / f"seed_{seed}")
                        if recovery else _fit(sim, data, current, n_train, int(seed), cfg, cell_dir / kind / f"seed_{seed}")
                        for seed in cfg["seeds"]]
                successful = [fit for fit in fits if fit["success"]]
                if not successful:
                    raise RuntimeError(f"Every {kind} optimizer failed for cell {data['cell']} at {current} A; no handoff produced")
                best = min(successful, key=lambda x: x["objective"])
                chosen[kind] = best
                predictions[kind] = _safe_final_prediction(sim, best["vector"], current, data, n_train)
                for fit in fits:
                    row = {"current_A_assumption": current, "cell": data["cell"], "fit_kind": kind,
                           "seed": fit["seed"], "objective": fit["objective"], "success": fit["success"],
                           "selected": fit is best, "solver_failures": fit["solver_failures"]}
                    row['optimizer_reused'] = bool(recovery)
                    row['objective_mesh'] = recovery['original_optimization_mesh'] if recovery else MESH
                    row.update({f"parameter_{i}": x for i, x in enumerate(fit["vector"])})
                    if fit["success"]:
                        seed_pred = _safe_final_prediction(sim, fit["vector"], current, data, n_train)
                        mm = _segments(data, seed_pred, cfg["train_fraction"])
                        for segment in ("full", "prefix_fit", "suffix_forecast"):
                            row[f"{segment}_V_RMSE"] = mm[segment]["voltage_V"]["rmse"]
                            row[f"{segment}_T_RMSE"] = mm[segment]["temperature_C"]["rmse"]
                            row[f"{segment}_coverage"] = mm[segment]["coverage_fraction"]
                        row["prediction_solver_failure"] = seed_pred.get("solver_failure")
                    all_seed_rows.append(row)
                matrix = np.array([fit["vector"] for fit in successful])
                for i in range(8):
                    width = BOUNDS[i, 1] - BOUNDS[i, 0]
                    scale = float(SCALES[i]) if i < 6 else 1.
                    variability.append({"current_A_assumption": current, "cell": data["cell"], "fit_kind": kind,
                        "parameter_index": i, "parameter_name": PARAMETER_NAMES[i], "fixed": i not in FREE,
                        "optimizer_runs": len(matrix), "scaled_mean": float(matrix[:, i].mean()),
                        "scaled_std": float(matrix[:, i].std(ddof=1)) if len(matrix) > 1 else None,
                        "scaled_min": float(matrix[:, i].min()), "scaled_max": float(matrix[:, i].max()),
                        "physical_mean": float(matrix[:, i].mean() * scale),
                        "physical_std": float(matrix[:, i].std(ddof=1) * scale) if len(matrix) > 1 else None,
                        "fraction_at_bound": float(np.mean((matrix[:, i] - BOUNDS[i, 0] < .01 * width) |
                                                              (BOUNDS[i, 1] - matrix[:, i] < .01 * width))) if i in FREE else 0.,
                        "interpretation": "optimizer variability, not physical parameter uncertainty"})
                _prediction_csv(cell_dir / f"{kind}_selected.csv", data, predictions[kind], split, kind)
                _dump(cell_dir / f"{kind}_selected.json", {"selection": best, "metrics": _segments(data, predictions[kind], cfg["train_fraction"])})
            spm = _safe_final_prediction(spm_sim, chosen["full_fit"]["vector"], current, data, n)
            _prediction_csv(cell_dir / "spm_form_sensitivity.csv", data, spm, split, "SPM_with_SPMe_parameters")
            record = {"current_A_assumption": current, "cell": data["cell"],
                "full_fit": _segments(data, predictions["full_fit"], cfg["train_fraction"]),
                "prefix_fit_forecast": _segments(data, predictions["prefix_fit"], cfg["train_fraction"]),
                "spm_form_sensitivity": _segments(data, spm, cfg["train_fraction"])}
            for kind, field in [("SPMe_full_fit", "full_fit"), ("SPMe_prefix_forecast", "prefix_fit_forecast"),
                                ("SPM_same_parameters", "spm_form_sensitivity")]:
                for segment, metric in record[field].items():
                    comparison_rows.append({"current_A_assumption": current, "cell": data["cell"], "model_evaluation": kind,
                        "segment": segment, "V_RMSE": metric["voltage_V"]["rmse"], "V_bias": metric["voltage_V"]["bias"],
                        "T_RMSE": metric["temperature_C"]["rmse"], "T_bias": metric["temperature_C"]["bias"],
                        "coverage_fraction": metric["coverage_fraction"], "full_coverage": metric["full_coverage"]})
            if np.isclose(current, float(cfg["current_for_control"]), rtol=0., atol=1e-12):
                precision = _precision(chosen["full_fit"]["vector"], current, data, cell_dir, cfg)
                record["numerical_precision"] = precision
                selected.append({"cell": data["cell"], "vector": chosen["full_fit"]["vector"],
                                 "seed": chosen["full_fit"]["seed"], "precision": precision,
                                 "fit_metrics": record["full_fit"]["full"]})
            all_metrics.append(record)
            _dump(cell_dir / "metrics.json", record)
            _plot(cell_dir / "fit_and_residuals.png", data, predictions["full_fit"], predictions["prefix_fit"], spm, split, current)
            _dump(out / "metrics.json", all_metrics)
            _csv(out / "optimizer_runs.csv", all_seed_rows)
            _csv(out / "parameter_variability.csv", variability)
            _csv(out / "model_and_current_comparison.csv", comparison_rows)
    selected.sort(key=lambda item: item["cell"])
    precision_passed = len(selected) == 3 and all(item["precision"]["passed"] for item in selected)
    all_full_coverage = all(item["fit_metrics"]["full_coverage"] for item in selected)
    screening = {"minimum_coverage_fraction": .99, "maximum_voltage_RMSE_V": .05,
                 "maximum_temperature_RMSE_C": 1., **cfg.get("model_screening_thresholds", {})}
    if (not 0 < float(screening["minimum_coverage_fraction"]) <= 1
            or float(screening["maximum_voltage_RMSE_V"]) <= 0
            or float(screening["maximum_temperature_RMSE_C"]) <= 0
            or not all(np.isfinite(float(value)) for value in screening.values())):
        raise ValueError("Invalid engineering model-screening thresholds")
    model_screen_passed = len(selected) == 3 and all(
        item["fit_metrics"]["coverage_fraction"] >= screening["minimum_coverage_fraction"]
        and item["fit_metrics"]["voltage_V"]["rmse"] is not None
        and item["fit_metrics"]["voltage_V"]["rmse"] <= screening["maximum_voltage_RMSE_V"]
        and item["fit_metrics"]["temperature_C"]["rmse"] is not None
        and item["fit_metrics"]["temperature_C"]["rmse"] <= screening["maximum_temperature_RMSE_C"]
        for item in selected)
    quality = {"computation_completed": True, "numerical_precision_passed": precision_passed,
        "model_screening_thresholds_predeclared": screening, "model_screening_passed": model_screen_passed,
        "model_screening_interpretation": "Engineering gate for exploratory numerical control experiments; these are not paper accuracy targets or independent physical validation.",
        "all_control_fits_cover_entire_curve": all_full_coverage,
        "metadata_fields_complete": audit["metadata_fields_complete"],
        "metadata_independently_verified": False,
        "independent_experimental_validation": False, "charging_domain_validated": False,
        "paper_temperature_RMSE_reference_C": [.068, .091, .093],
        "paper_temperature_RMSE_thresholds_met_in_sample": all(item["fit_metrics"]["temperature_C"]["rmse"] is not None
            and item["fit_metrics"]["temperature_C"]["rmse"] <= [.068, .091, .093][item["cell"] - 1]
            and item["fit_metrics"]["full_coverage"] for item in selected),
        "warning": "Passing numerical checks does not imply physical validation. Temperature accuracy, incomplete metadata and lack of independent charge/discharge cycles remain explicit limitations."}
    summary = {"experiment": 1, "quality": quality, "selected_control_current_A_assumption": cfg["current_for_control"],
               "selected_cells": selected, "number_of_optimizer_runs": len(all_seed_rows), "outputs": str(out)}
    summary['new_optimizer_runs'] = 0 if recovery else len(all_seed_rows)
    summary['optimizer_reuse'] = recovery
    _dump(out / "summary.json", summary)
    lines = ["# Experiment 1 result", "", "This report is generated from the user's run. No success is assumed in advance.", "",
        f"Numerical precision gate: {precision_passed}", f"All selected full fits cover every sample: {all_full_coverage}",
        f"Model engineering screen: {model_screen_passed}",
        f"Metadata fields supplied: {audit['metadata_fields_complete']}", "",
        "Current is an assumption declared before fitting; compare 4 A and 5 A without identifying the true current from residuals.",
        "The suffix forecast holds out time points from the same curve. It does not replace an independent experimental cycle.",
        "The SPM comparison holds SPMe-fit parameters fixed; it is model-form sensitivity, not best possible SPM performance.",
        "Parameter standard deviations are optimizer-seed variability, not statistical confidence intervals.",
        "Missing predictions after physical cutoff are blank in CSV and excluded from error metrics; always read coverage alongside RMSE.",
        "Missing ambient, sensor and capacity metadata are not invented. Discharge-only fitting does not validate charging-domain safety.", "",
        "See model_and_current_comparison.csv, optimizer_runs.csv, parameter_variability.csv, data_audit.json and per-cell figures.",
        'Optimizer provenance: ' + ('Saved coarse-mesh parameters and original seed selection reused; all new-mesh predictions and gates recomputed. No fine-mesh refitting was performed.' if recovery else 'New optimizer runs on the declared base mesh.')]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not precision_passed or not model_screen_passed:
        raise RuntimeError("E1 completed fitting but numerical precision or model screening failed; results retained, no experiment-2 handoff created. Inspect precision.json and summary.json; do not relax thresholds after observing results merely to force a pass.")
    assumptions = ["Control calibration current was specified before fitting; physical measurement current is unconfirmed unless metadata supplied.",
        "SPMe/lumped-thermal model and modified Chen2020 source parameters are reconstruction assumptions.",
        "Initial discharge concentrations follow original source defaults; control SOC uses consistent electrode stoichiometries.",
        "No independent experimental cycles, charging measurements, sensor uncertainty or hardware validation are supplied.",
        "Heat-capacity multiplier and heat-transfer coefficient may compensate for model mismatch; fitted parameters are not uniquely identified."]
    handoff = {"schema_version": 1, "model_family": "SPMe", "current_A_assumption": float(cfg["current_for_control"]),
        "parameter_vectors": [item["vector"] for item in selected], "source": str(out), "assumptions": assumptions,
        "readiness": {"status": "READY_FOR_EXPLORATORY_SIMULATION", "numerical_precision_passed": precision_passed,
                      "model_screening_passed": model_screen_passed, "physical_validation_complete": False},
        "quality": quality, "numerical_settings": protocol["solver"], "initialization": protocol["initialization"],
        "ambient_temperature_K": [data["ambient_temperature_K"] for data in datasets],
        "source_hashes": [{"cell": item["cell"], "voltage_sha256": item["voltage_sha256"], "temperature_sha256": item["temperature_sha256"]} for item in audit["records"]],
        "artifact_hashes": {name: sha256(out / name) for name in ("protocol.json", "data_audit.json", "summary.json", "metrics.json")},
        "code_hashes": {path.name: sha256(path) for path in (Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("original_parameters.json"))}}
    _dump(out / "handoff.json", handoff)
    return _plain(summary)
