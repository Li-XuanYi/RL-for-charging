"""Auditable partial reproduction of the supplied battery charging research."""
import os
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pybamm
import torch
from battery import ACTIONS, BETA, BOUNDS, DT, INITIAL, Pack, discharge
from learner import Learner

SETS = {"A": [.1, .2, .3], "B": [.3, .5, .7]}
MODES = {"smoke": (2, 1, 2, 4, 10), "quick": (6, 5, 30, 50, 60),
         "full": (30, 100, 1200, 50, 60)}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file
    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()


def raw_data(root, out):
    data_dir = next(root.glob("multi-balancing-agent*/multi-balancing-agent*/data_true/data_image"), None)
    if data_dir is None:
        raise FileNotFoundError("Original data_true/data_image directory not found under " + str(root))
    data, provenance = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, age in enumerate(("begin", "middle", "age")):
        vp, tp = data_dir / ("data_1C_" + age), data_dir / ("temp_data 1C " + age)
        v, temp = np.loadtxt(vp), np.loadtxt(tp)
        if v.ndim != 1 or temp.ndim != 1 or len(v) != len(temp) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(temp)):
            raise ValueError("Invalid measured voltage/temperature pair: " + age)
        t = np.arange(len(v), dtype=float)  # Source scripts assume 1 second sampling.
        data.append((t, v, temp))
        for p in (vp, tp):
            provenance.append({"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "samples": len(v)})
        write_csv(out / "raw" / f"battery_{i+1}.csv", [dict(time_s=float(a), voltage_V=float(b), temperature_C=float(c)) for a,b,c in zip(t,v,temp)])
        axes[0].plot(t, v, label=f"Battery {i+1}")
        axes[1].plot(t, temp, label=f"Battery {i+1}")
    for ax, ylabel in zip(axes, ("Voltage / V", "Temperature / deg C")):
        ax.set(xlabel="Time / s", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("Measured discharge curves from supplied files (Fig. 4 reconstruction)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "measured_discharge.png", dpi=180)
    plt.close(fig)
    write_json(out / "data_provenance.json", provenance)
    return data


def fit_battery(data, index, args, out, particles, iterations):
    t, v, temp = data
    rng = np.random.default_rng(args.seed + index)
    lower, upper = BOUNDS.T
    position = rng.uniform(lower, upper, (particles, len(lower)))
    position[0] = INITIAL
    velocity = np.zeros_like(position)
    best_position = position.copy()
    best_score = np.full(particles, np.inf)
    global_position, global_score = INITIAL.copy(), np.inf
    history, errors = [], []
    def score(vector):
        try:
            ts, vs, temps = discharge(vector, t[-1], args.ident_current, temp[0]+273.15)
            # Compare on the full measured time grid. Early termination is
            # penalized, never rewarded by silently shortening the measurement.
            ve, te = np.interp(t, ts, vs), np.interp(t, ts, temps)
            return float(np.sqrt(np.mean((ve-v)**2))/.05 + np.sqrt(np.mean((te-temp)**2))
                         + 20 * max(0., 1-ts[-1]/t[-1]))
        except (pybamm.SolverError, ValueError) as exc:
            errors.append(str(exc)[:300])
            return 1e6
    for iteration in range(iterations+1):
        for k in range(particles):
            value = score(position[k])
            if value < best_score[k]:
                best_score[k], best_position[k] = value, position[k].copy()
            if value < global_score:
                global_score, global_position = value, position[k].copy()
        history.append({"iteration": iteration, "normalized_objective": float(global_score)})
        print(f"Fit battery {index+1}: {iteration}/{iterations}, objective={global_score:.5f}", flush=True)
        write_json(out / "fit" / f"battery_{index+1}_best.json", {"vector": global_position.tolist(), "objective": float(global_score), "iteration": iteration})
        w = .9 - .5 * iteration/max(iterations, 1)
        velocity = w*velocity + 1.5*rng.random(position.shape)*(best_position-position) + 1.5*rng.random(position.shape)*(global_position-position)
        position = np.clip(position+velocity, lower, upper)
    if global_score >= 1e6:
        raise RuntimeError("All parameter-fitting trials failed: " + " | ".join(errors[-3:]))
    ts, vs, temps = discharge(global_position, t[-1], args.ident_current, temp[0]+273.15, points=len(t))
    ve, te = np.interp(t, ts, vs), np.interp(t, ts, temps)
    covered = t <= ts[-1] + 1e-6
    metrics = {"battery": index+1, "voltage_RMSE_V": float(np.sqrt(np.mean((ve-v)**2))),
        "temperature_RMSE_C": float(np.sqrt(np.mean((te-temp)**2))),
        "model_coverage_fraction": float(covered.mean()), "identification_current_A": args.ident_current,
        "evaluation": "in_sample_fit_not_independent_validation"}
    write_csv(out / "fit" / f"battery_{index+1}_fit.csv", [dict(time_s=float(a), measured_V=float(b), fitted_V=float(c), measured_C=float(d), fitted_C=float(e), model_covers_time=bool(f)) for a,b,c,d,e,f in zip(t,v,ve,temp,te,covered)])
    write_csv(out / "fit" / f"battery_{index+1}_pso_history.csv", history)
    write_json(out / "fit" / f"battery_{index+1}_solver_rejections.json", errors)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, measured, fitted, label in zip(axes, (v,temp), (vs,temps), ("Voltage / V", "Temperature / deg C")):
        ax.plot(t, measured, label="Measured", lw=1.2)
        ax.plot(ts, fitted, label="Refitted SPMe", lw=1.2)
        ax.set(xlabel="Time / s", ylabel=label)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(f"Battery {index+1}: reconstructed in-sample fit; current={args.ident_current:g} A")
    fig.tight_layout()
    fig.savefig(out / "figures" / f"battery_{index+1}_fit.png", dpi=180)
    plt.close(fig)
    return global_position.tolist(), metrics


def rollout(vectors, initial, learner, epsilon, limit):
    pack = Pack(vectors, initial)
    learner.reset()
    episode = {"obs": [pack.obs()], "masks": [pack.mask()], "actions": [], "rewards": [], "done": []}
    rows = state_rows(pack, np.zeros(3), np.zeros((3,4)), np.zeros(3))
    for step in range(limit):
        choice = learner.choose(episode["obs"][-1], episode["masks"][-1], epsilon)
        currents = ACTIONS[choice]
        reward, done, stats = pack.step(currents)
        episode["actions"].append(choice)
        episode["rewards"].append(reward)
        episode["done"].append(done)  # time-limit truncation still bootstraps
        episode["obs"].append(pack.obs())
        episode["masks"].append(pack.mask())
        rows.extend(state_rows(pack, currents, stats, np.full(3, reward)))
        if done:
            break
    return episode, rows


def state_rows(pack, requested, stats, rewards):
    std = float(pack.physical()[:,0].std())
    return [dict(time_s=pack.time, cell=i+1, soc=c.soc, voltage_V=c.voltage,
        temperature_K=c.temperature, requested_current_A=float(requested[i]), applied_current_A=float(stats[i,0]),
        interval_mean_voltage_V=float(stats[i,1]) if pack.time else c.voltage,
        interval_max_voltage_V=float(stats[i,2]) if pack.time else c.voltage,
        interval_max_temperature_K=float(stats[i,3]) if pack.time else c.temperature,
        soc_std=std, team_reward=float(rewards[i])) for i,c in enumerate(pack.cells)]


def reconstructed_cc(vectors, initial, limit, seed):
    # Preserves the supplied CC-CV equalization.py idea: 4 A CC, then paired
    # charge redistribution without external charging. It is NOT a validated
    # constant-voltage circuit model. All cells now advance the SAME 90 s.
    pack, balancing = Pack(vectors, initial), False
    rng = np.random.default_rng(seed)
    rows = state_rows(pack, np.zeros(3), np.zeros((3,4)), np.zeros(3))
    for step in range(limit):
        if max(c.voltage for c in pack.cells) >= 4.0:
            balancing = True
        currents = np.full(3, 4.)
        if balancing:
            currents[:] = 0.
            for a,b in ((0,1),(1,2)):
                delta = pack.cells[a].soc - pack.cells[b].soc
                amount = 3.8 if abs(delta)>=.1 else 2.8 if abs(delta)>=.075 else 1.8 if abs(delta)>=.05 else 1.
                amount += rng.uniform(-.5,.5)
                currents[a] -= np.sign(delta)*amount
                currents[b] += np.sign(delta)*amount
            # Quantize to the reconstructed environment's 0.5 A control grid.
            currents = np.round(currents*2)/2
        reward, _, stats = pack.step(currents)
        rows.extend(state_rows(pack, currents, stats, np.full(3,reward)))
        if pack.physical()[:,0].std() <= BETA:
            break
    return rows


def metrics(rows, method, scenario, seed):
    a = np.array([[r["time_s"], r["soc"], r["soc_std"], r["applied_current_A"], r["interval_mean_voltage_V"], r["requested_current_A"], r["interval_max_voltage_V"], r["interval_max_temperature_K"]] for r in rows]).reshape(-1,3,8)
    balanced = a[:,0,2] <= BETA
    reached = balanced & (a[:,:,1] >= .9).all(axis=1)
    b = int(np.flatnonzero(balanced)[0]) if balanced.any() else None
    charge = a[1:,:,3] * DT/3600
    return {"method": method, "scenario": scenario, "seed": seed,
        "first_balance_time_s": float(a[b,0,0]) if b is not None else None,
        "balance_maintained_after_first": bool(balanced[b:].all()) if b is not None else False,
        "balanced_target90_time_s": float(a[np.flatnonzero(reached)[0],0,0]) if reached.any() else None,
        "target_reached": bool(reached.any()), "final_time_s": float(a[-1,0,0]),
        "final_soc_1": float(a[-1,0,1]), "final_soc_2": float(a[-1,1,1]), "final_soc_3": float(a[-1,2,1]),
        "final_soc_std": float(a[-1,0,2]), "positive_charge_Ah": float(np.maximum(charge,0).sum()),
        "discharged_Ah": float(-np.minimum(charge,0).sum()),
        "terminal_input_energy_Wh": float((np.maximum(charge,0)*a[1:,:,4]).sum()),
        "max_voltage_V": float(a[:,:,6].max()), "max_temperature_K": float(a[:,:,7].max()),
        "safety_filter_interventions": int((abs(a[1:,:,3]-a[1:,:,5])>1e-8).sum()),
        "full_charging_comparison_valid": False}


def plots(rows, path, title):
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for i in (1,2,3):
        r = [x for x in rows if x["cell"]==i]
        t = [x["time_s"] for x in r]
        for ax,key,label in zip(axes.flat,("soc","applied_current_A","voltage_V","temperature_K"),("SOC","Current / A","Voltage / V","Temperature / K")):
            if key == "applied_current_A":
                # Each endpoint row stores the current used on the preceding
                # 90-second interval; draw its actual zero-order hold.
                ax.step(t,[x[key] for x in r],where="pre",label=f"Cell {i}")
            else:
                ax.plot(t,[x[key] for x in r],label=f"Cell {i}")
            ax.set(xlabel="Time / s",ylabel=label)
            ax.grid(alpha=.2)
    axes[0,0].axhline(.9,color="gray",ls="--")
    axes[1,0].axhline(4.2,color="gray",ls="--")
    axes[1,1].axhline(309,color="gray",ls="--")
    axes[0,0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=MODES, default="quick")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--ident-current",type=float,default=4.,help="4 A from paper's 4 Ah/1C; use 5 to match supplied fitting script")
    parser.add_argument("--episodes",type=int)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--fit-from",type=Path,help="Reuse a previous fitted_parameters.json")
    args = parser.parse_args()
    if args.ident_current<=0 or (args.episodes is not None and args.episodes<1):
        parser.error("current and episodes must be positive")
    root = args.root.resolve()
    out = root / "reproduction_results" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.mode)
    for d in ("raw","fit","training","checkpoints","trajectories","figures"):
        (out/d).mkdir(parents=True,exist_ok=True)
    log = (out/"run.log").open("w",encoding="utf-8")
    sys.stdout,sys.stderr = Tee(sys.stdout,log),Tee(sys.stderr,log)
    print("OUTPUT:",out,flush=True)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    np.random.seed(args.seed)
    random.seed(args.seed)
    pybamm.set_logging_level("ERROR")
    particles,iterations,episodes,train_limit,eval_limit = MODES[args.mode]
    episodes = args.episodes or episodes
    started = time.time()
    config = {**vars(args),"root":str(root),"fit_from":str(args.fit_from) if args.fit_from else None,
        "particles":particles,"pso_iterations":iterations,"episodes_per_scenario":episodes,
        "train_step_limit":train_limit,"evaluation_step_limit":eval_limit,"decision_seconds":DT,
        "current_grid_A":ACTIONS.tolist(),"target_network_update_episodes":50,
        "epsilon":"0.5*(1-(episode+1)/episodes), from supplied main.py",
        "runtime_python":sys.version,"dependencies":{p:importlib.metadata.version(p) for p in ("pybamm","torch","numpy","scipy","matplotlib","casadi")},
        "scope":"partial_numerical_reconstruction_not_exact_paper_reproduction",
        "missing":["DQL implementation and checkpoints", "complete original per-cell fitted parameters", "hardware online-control interface", "raw charging-control trajectories"],
        "assumptions":["1 second sampling inferred from original scripts", "SPMe+Chen2020 follows code; paper says SPM and Samsung 40T", "eight-parameter joint in-sample refit is reconstructed", "extra simulation safety filter changes the paper controller", "CC baseline has no physical converter or true CV model", "terminal energy is not topology efficiency or battery degradation"]}
    write_json(out/"config.json",config)
    write_json(out/"code_provenance.json",[{"path":str(p.relative_to(Path(__file__).parent)),
        "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in Path(__file__).parent.rglob("*.py")])
    write_json(out/"status.json",{"status":"running","scope":config["scope"]})
    (root/"reproduction_results"/"LATEST.txt").write_text(str(out),encoding="utf-8")
    summary = []
    try:
        data = raw_data(root,out)
        if args.fit_from:
            fitted = json.loads(args.fit_from.read_text(encoding="utf-8"))
            vectors = fitted["vectors"]
            if np.asarray(vectors).shape != (3,8) or not np.isfinite(vectors).all():
                raise ValueError("Expected three finite eight-parameter vectors")
            if fitted["identification_current_A"] != args.ident_current:
                raise ValueError("--ident-current does not match the reused parameter fit")
        else:
            vectors, fit_metrics = [], []
            for i,d in enumerate(data):
                vector,m = fit_battery(d,i,args,out,particles,iterations)
                vectors.append(vector)
                fit_metrics.append(m)
                write_csv(out/"fit"/"metrics.csv",fit_metrics)
            fitted = {"vectors":vectors,"identification_current_A":args.ident_current,"source":"reconstructed_PSO_fit","in_sample":True}
        write_json(out/"fitted_parameters.json",fitted)
        for name,initial in SETS.items():
            print(f"Scenario {name}: initial SOC {initial}; {episodes} training episodes",flush=True)
            learner = Learner(args.seed,args.device)
            replay = deque(maxlen=800)
            history = []
            for epoch in range(episodes):
                epsilon = .5*(1-(epoch+1)/episodes)
                ep,rows = rollout(vectors,initial,learner,epsilon,train_limit)
                replay.append(ep)
                batch_indices = learner.rng.choice(len(replay),size=min(len(replay),128),replace=False)
                loss = learner.train([replay[int(i)] for i in batch_indices]) if len(replay)>=2 else None
                if (epoch+1)%50==0:
                    learner.sync()
                history.append({"episode":epoch+1,"reward":float(sum(ep["rewards"])),"loss":loss,
                    "epsilon":epsilon,"steps":len(ep["actions"]),"target_reached":bool(ep["done"][-1])})
                write_csv(out/"training"/f"set_{name}.csv",history)
                if (epoch+1)%25==0 or epoch+1==episodes:
                    learner.save(out/"checkpoints"/f"set_{name}.pt",{"episode":epoch+1,"scenario":name,"config":config})
                print(f"  {name} episode {epoch+1}/{episodes}: reward={sum(ep['rewards']):.3f}, loss={loss}, target={ep['done'][-1]}",flush=True)
            # Evaluate the saved checkpoint, independently from rollout memory.
            learner.load(out/"checkpoints"/f"set_{name}.pt")
            _,rows = rollout(vectors,initial,learner,0.,eval_limit)
            write_csv(out/"trajectories"/f"MBA_RL_set_{name}.csv",rows)
            summary.append(metrics(rows,"MBA_RL_reconstructed",name,args.seed))
            plots(rows,out/"figures"/f"MBA_RL_set_{name}.png",f"Reconstructed MBA-RL | Set {name} | {episodes} episodes")
            rows = reconstructed_cc(vectors,initial,eval_limit,args.seed)
            write_csv(out/"trajectories"/f"CC_baseline_set_{name}.csv",rows)
            summary.append(metrics(rows,"CC_then_balancing_reconstructed",name,args.seed))
            plots(rows,out/"figures"/f"CC_baseline_set_{name}.png",f"Reconstructed CC-then-balancing heuristic | Set {name}")
            write_csv(out/"summary.csv",summary)
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot([r["episode"] for r in history],[r["reward"] for r in history])
            ax.set(xlabel="Episode",ylabel="Team reward",title=f"Set {name} training ({args.mode})")
            fig.tight_layout()
            fig.savefig(out/"figures"/f"training_set_{name}.png",dpi=180)
            plt.close(fig)
        report = {"status":"completed", "scope":config["scope"], "mode":args.mode,
            "elapsed_seconds":time.time()-started,"all_MBA_RL_targets_reached":all(r["target_reached"] for r in summary if r["method"]=="MBA_RL_reconstructed"),
            "DQL":"not_run_missing_source", "hardware":"not_run", "results":summary}
        write_json(out/"status.json",report)
        print("Completed PARTIAL NUMERICAL RECONSTRUCTION. See summary.csv and status.json.")
        print("Short-run completion does not establish convergence or reproduce the paper's numeric claims.")
        print("OUTPUT:",out)
    except BaseException as exc:
        write_json(out/"status.json",{"status":"interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
            "error":str(exc),"elapsed_seconds":time.time()-started,"completed_metrics":summary})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
