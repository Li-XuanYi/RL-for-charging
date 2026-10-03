"""Create compact manuscript figures from the archived experiment records."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MANUSCRIPT_OUT = (ROOT /
                  "Multi_Balancing_Agent_Reinforcement_Learning_for_Charging_Control_of_Lithium_ion_Batteries" /
                  "image")
EVIDENCE_OUT = ROOT / "evidence" / "figures"
TRAJECTORIES = (WORKSPACE / "RL-for-charging-text-revision-20260930" /
                "revision_results" / "experiment_3_gpu_parallel_2500_5seeds_90s" /
                "20260922_185259_914157")
ABLATIONS = (WORKSPACE / "RL-for-charging-selected" / "revision_results" /
             "experiment_4_gpu_parallel_2500_5seeds_90s" /
             "20260924_175214_871584")
CASE = "E3_test_000"
SEEDS = [7, 17, 27, 37, 47]
INPUTS = {}


def record(path):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def configure():
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "stix",
    })
    return font_path


def load_trace(method, seed):
    base = TRAJECTORIES / method / f"seed_{seed}" / "test" / CASE
    frame = pd.read_csv(record(base / "trajectory.csv.gz"))
    metrics = json.loads(record(base / "metrics.json").read_text(encoding="utf-8-sig"))
    charge = frame.loc[frame.phase.eq("charge")]
    parts = [charge.loc[charge.cell.eq(cell)] for cell in [1, 2, 3]]
    time = parts[0].time_s.to_numpy()
    soc = np.column_stack([part.soc.to_numpy() for part in parts])
    assert all(np.array_equal(time, part.time_s.to_numpy()) for part in parts[1:])
    std = soc.std(axis=1, ddof=0)
    assert np.isclose(np.trapezoid(std, time), metrics["soc_std_integral_soc_s"],
                      rtol=1e-9, atol=1e-8)
    assert time[-1] == metrics["charging_time_s"] and metrics["task_success"]
    return time, soc.mean(axis=1), std


def save(fig, stem):
    fig.savefig(MANUSCRIPT_OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(EVIDENCE_OUT / f"{stem}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def comparative_charging_discrete():
    """Compare MBA-RL with discrete-action learning baselines.

    The action authority and test case are held fixed for all curves.  The
    continuous-action references remain in the aggregate table rather than
    being mixed into this direct trajectory comparison.
    """
    specs = [
        ("qmix", 7, "MBA-RL", "#0000FF", "-", "o"),
        ("iql_shared", 7, "Shared IQL", "#E18A00", "-.", "D"),
        ("dqn", 7, "Centralized DQN", "#FF0000", "--", "s"),
    ]
    axis_font = {"fontfamily": "Arial"}
    legend_font = font_manager.FontProperties(family="Arial", size=6.4)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.08))
    fig.subplots_adjust(left=.14, right=.985, bottom=.215, top=.78, wspace=.44)
    handles = []
    for method, seed, label, color, linestyle, marker in specs:
        time, mean, std = load_trace(method, seed)
        mark_indices = np.arange(0, len(time), 420)
        for ax, values in zip(axes, (mean, std)):
            line, = ax.plot(time, values, color=color, linestyle=linestyle,
                            linewidth=1.35, marker=marker,
                            markevery=mark_indices, markersize=3.3,
                            markerfacecolor="none", markeredgewidth=.9,
                            label=label)
            ax.plot(time[-1], values[-1], marker=marker, color=color,
                    markersize=3.5, markeredgewidth=.9, linestyle="None")
        handles.append(line)
    fig.legend(handles, [s[2] for s in specs], loc="upper center",
               bbox_to_anchor=(.52, .985), ncol=3, frameon=False,
               columnspacing=.72, handlelength=1.8, handletextpad=.3,
               labelspacing=.2, prop=legend_font)
    for col, ax in enumerate(axes):
        ax.set_xlim(0, 3000)
        ax.set_xticks([0, 1500, 3000])
        ax.set_xlabel("Time/s", labelpad=2, fontsize=7.2,
                      fontweight="bold", **axis_font)
        ax.text(.04, .94, f"({chr(97 + col)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, fontweight="bold",
                **axis_font)
        if col == 0:
            ax.set_ylim(.40, 1.0)
            ax.set_yticks([.4, .7, 1.0])
            ax.set_ylabel("Mean SOC", labelpad=2, fontsize=7.2,
                          fontweight="bold", **axis_font)
        else:
            ax.set_ylim(0, .12)
            ax.set_yticks([0, .06, .12])
            ax.set_ylabel("SOC deviation", labelpad=2, fontsize=7.2,
                          fontweight="bold", **axis_font)
            ax.axhline(.02, color="#666666", linewidth=.9,
                       linestyle=(0, (5, 3)), zorder=0)
            ax.text(2940, .022, "0.02", ha="right", va="bottom",
                    color="#555555", fontsize=6.1, fontweight="bold",
                    **axis_font)
        ax.tick_params(axis="both", labelsize=6.5, width=.9, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")
        for spine in ax.spines.values():
            spine.set_linewidth(.9)
    save(fig, "comparative_charging_discrete")


def ablation():
    frame = pd.read_csv(record(ABLATIONS / "episode_metrics.csv"))
    frame = frame.loc[frame.test_family.eq("random_nominal")].copy()
    if frame.task_success.dtype != bool:
        frame["task_success"] = frame.task_success.astype(str).str.lower().eq("true")
    order = [
        ("qmix_reference", "MBA-RL"),
        ("qmix_no_mixer_shared", "No mixer"),
        ("qmix_unshared", "No sharing"),
        ("qmix_no_balance_reward", "No balance reward"),
    ]
    grouped = frame.groupby(["method_id", "seed"]).agg(
        cases=("case_id", "size"),
        success_count=("task_success", "sum"),
        mean_imbalance=("soc_std_integral_soc_s", "mean"),
    ).reset_index()
    grouped["completion_percent"] = grouped.success_count / grouped.cases * 100
    fig = plt.figure(figsize=(3.45, 2.18))
    ax = fig.add_axes([.36, .20, .59, .68])
    for row, (method, label) in enumerate(order):
        part = grouped.loc[grouped.method_id.eq(method)].set_index("seed").loc[SEEDS]
        ax.axhline(row, color="#D9D9D9", linewidth=.55, zorder=0)
        mean = part["mean_imbalance"].mean()
        completed = int(part["success_count"].sum())
        cases = int(part["cases"].sum())
        color = "#D62728" if method in {"qmix_unshared", "qmix_no_balance_reward"} else "#0000FF"
        ax.scatter(part["mean_imbalance"], np.full(len(part), row), s=18,
                   marker="o", facecolors="white", edgecolors="#666666",
                   linewidths=.65, zorder=3)
        ax.scatter(mean, row, s=38, marker="D", facecolors=color,
                   edgecolors="black", linewidths=.7, zorder=4)
        ax.text(mean + 14, row, f"{mean:.0f}", va="center", ha="left",
                fontsize=6.8)
        ax.text(1065, row, f"{completed}/{cases}", va="center", ha="right",
                fontsize=6.8)
    reference = grouped.loc[grouped.method_id.eq("qmix_reference"),
                            "mean_imbalance"].mean()
    ax.axvline(reference, color="#777777", linewidth=.8,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(1065, -.58, "Completed", va="bottom", ha="right", fontsize=6.8)
    ax.set_ylim(len(order) - .55, -.45)
    ax.set_yticks(range(len(order)))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_yticklabels([label for _, label in order], fontsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.set_xlim(120, 1080)
    ax.set_xticks([150, 400, 650, 900])
    ax.set_xlabel("Accumulated SOC imbalance (fraction s)", labelpad=3)
    save(fig, "component_ablation_compact")


if __name__ == "__main__":
    MANUSCRIPT_OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.mkdir(parents=True, exist_ok=True)
    configure()
    comparative_charging_discrete()
    ablation()
    (EVIDENCE_OUT / "figure_provenance_compact.json").write_text(
        json.dumps({"inputs": INPUTS, "scope": "Compact manuscript figures"}, indent=2),
        encoding="utf-8")
    print(f"Created manuscript PDFs in {MANUSCRIPT_OUT}")
