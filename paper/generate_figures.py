from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.axis("off")

    boxes = [
        (0.03, 0.55, "Q1 static calibration", "Exact enumeration\nRB granularity + QoS closure"),
        (0.36, 0.55, "Q2 dynamic baseline", "Finite-horizon MPC\nqueues + channel evolution"),
        (0.69, 0.55, "Q3 coupled control", "Hierarchical PPO\nslice budget + power control"),
    ]
    colors = ["#dbeafe", "#dcfce7", "#fee2e2"]
    edge = ["#2563eb", "#16a34a", "#dc2626"]

    for idx, (x, y, title, body) in enumerate(boxes):
        rect = FancyBboxPatch(
            (x, y),
            0.27,
            0.28,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=edge[idx],
            facecolor=colors[idx],
        )
        ax.add_patch(rect)
        ax.text(x + 0.135, y + 0.20, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 0.135, y + 0.09, body, ha="center", va="center", fontsize=10)

    for x0, x1 in [(0.30, 0.36), (0.63, 0.69)]:
        ax.add_patch(FancyArrowPatch((x0, 0.69), (x1, 0.69), arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#374151"))

    ax.text(0.5, 0.33, "Unified objective: long-term average service utility under heterogeneous slice SLA", ha="center", fontsize=11)
    ax.text(0.5, 0.19, "Two time scales: 100 ms slice/power decisions and 1 ms task service execution", ha="center", fontsize=11)
    save(fig, "framework.png")


def draw_q1_static_optima() -> None:
    allocations = {
        "(30,10,10)": [30, 10, 10],
        "(20,20,10)": [20, 20, 10],
        "(20,10,20)": [20, 10, 20],
    }
    colors = ["#2563eb", "#f97316", "#16a34a"]
    labels = ["URLLC", "eMBB", "mMTC"]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    left = [0] * len(allocations)
    y = list(range(len(allocations)))
    for idx, label in enumerate(labels):
        vals = [v[idx] for v in allocations.values()]
        ax.barh(y, vals, left=left, color=colors[idx], label=label)
        for j, val in enumerate(vals):
            ax.text(left[j] + val / 2, j, str(val), ha="center", va="center", color="white", fontsize=10, weight="bold")
        left = [left[j] + vals[j] for j in range(len(vals))]
    ax.set_yticks(y, list(allocations.keys()))
    ax.set_xlim(0, 50)
    ax.set_xlabel("Allocated RBs")
    ax.set_title("Static single-BS optimal allocations, U = 0.980771")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)
    save(fig, "q1_static_optima.png")


def draw_q2_tradeoff() -> None:
    depths = [1, 2, 3]
    objectives = [load_json(f"outputs/q2_mpc/lookahead{h}.json")["objective"] for h in depths]
    times = []
    for h in depths:
        time_path = ROOT / "outputs" / "q2_mpc" / f"lookahead{h}_time.txt"
        raw = time_path.read_bytes()
        text = raw.decode("utf-16", errors="ignore") if raw.startswith(b"\xff\xfe") else raw.decode("utf-8", errors="ignore")
        seconds_line = [line for line in text.splitlines() if line.strip().startswith("TotalSeconds")]
        if seconds_line:
            times.append(float(seconds_line[-1].split(":", 1)[1].strip()))
        else:
            vals = [float(tok) for tok in text.replace("=", " ").replace(",", " ").split() if tok.replace(".", "", 1).isdigit()]
            times.append(vals[-1])

    fig, ax1 = plt.subplots(figsize=(7.4, 4.0))
    ax1.plot(depths, objectives, marker="o", linewidth=2.2, color="#2563eb", label="Objective")
    ax1.set_xlabel("Lookahead depth")
    ax1.set_ylabel("Objective", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_xticks(depths)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(depths, times, width=0.34, alpha=0.35, color="#f97316", label="Runtime")
    ax2.set_ylabel("Runtime (s)", color="#c2410c")
    ax2.tick_params(axis="y", labelcolor="#c2410c")
    ax1.set_title("Q2 lookahead tradeoff")
    save(fig, "q2_lookahead_tradeoff.png")


def draw_q2_actions() -> None:
    data = load_json("outputs/q2_mpc/lookahead2.json")
    steps = [a["step"] for a in data["actions"]]
    vals = {"URLLC": [], "eMBB": [], "mMTC": []}
    for a in data["actions"]:
        vals["URLLC"].append(a["action"][0])
        vals["eMBB"].append(a["action"][1])
        vals["mMTC"].append(a["action"][2])

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bottom = [0] * len(steps)
    colors = {"URLLC": "#2563eb", "eMBB": "#f97316", "mMTC": "#16a34a"}
    for label, series in vals.items():
        ax.bar(steps, series, bottom=bottom, label=label, color=colors[label])
        bottom = [bottom[i] + series[i] for i in range(len(series))]
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Allocated RBs")
    ax.set_ylim(0, 50)
    ax.set_xticks(steps)
    ax.set_title("Q2 selected RB budget sequence, lookahead = 2")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "q2_decision_sequence.png")


def draw_q3_summary() -> None:
    baseline = load_json("outputs/q3_rl/short_run_metrics.json")["best_eval_objective"]
    paths = [
        ("seed 7", "outputs/q3_sb3/q3_combined_eval_fresh_10k.json"),
        ("seed 17", "outputs/q3_sb3/q3_combined_eval_seed17_10k.json"),
        ("seed 27", "outputs/q3_sb3/q3_combined_eval_seed27_10k.json"),
    ]
    labels = ["baseline"] + [p[0] for p in paths]
    objectives = [baseline] + [load_json(p[1])["evaluation"]["summary"]["objective"] for p in paths]
    seed17 = load_json("outputs/q3_sb3/q3_combined_eval_seed17_10k.json")["evaluation"]["summary"]
    ratios = {
        k: seed17["completed"][k] / seed17["total_arrivals"][k]
        for k in ["u", "e", "m"]
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = ["#94a3b8", "#60a5fa", "#2563eb", "#1d4ed8"]
    ax1.bar(labels, objectives, color=colors)
    ax1.set_ylim(0.43, 0.51)
    ax1.set_ylabel("Objective")
    ax1.set_title("Q3 multi-seed objective")
    ax1.grid(axis="y", alpha=0.25)
    for i, v in enumerate(objectives):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(["URLLC", "eMBB", "mMTC"], [ratios["u"], ratios["e"], ratios["m"]], color=["#2563eb", "#f97316", "#16a34a"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Completion ratio")
    ax2.set_title("Best run completion ratio")
    ax2.grid(axis="y", alpha=0.25)
    for i, key in enumerate(["u", "e", "m"]):
        ax2.text(i, ratios[key] + 0.03, f"{ratios[key]:.2%}", ha="center", fontsize=9)
    save(fig, "q3_multiseed_summary.png")


def copy_training_curve() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "docs" / "assets" / "q3_slice_seed17_convergence.png",
        FIG_DIR / "q3_slice_seed17_convergence.png",
    )


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_framework()
    draw_q1_static_optima()
    draw_q2_tradeoff()
    draw_q2_actions()
    draw_q3_summary()
    copy_training_curve()
