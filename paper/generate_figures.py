from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Publication-quality colour palette (colourblind-friendly, muted academic)
# Based on Paul Tol's "muted" scheme with a few custom additions.
# ---------------------------------------------------------------------------
C_BLUE   = "#4477AA"
C_CYAN   = "#66CCEE"
C_GREEN  = "#228833"
C_YELLOW = "#DDCC77"
C_RED    = "#CC6677"
C_PURPLE = "#AA3377"
C_GREY   = "#BBBBBB"
C_DARK   = "#332288"

SLICE_COLORS = {"URLLC": C_BLUE, "eMBB": C_RED, "mMTC": C_GREEN}  # triad, not RGB
SEED_COLORS   = [C_GREY, C_CYAN, C_BLUE, C_DARK]
SENS_COLORS   = [C_BLUE, C_RED, C_GREEN, C_PURPLE]  # one per panel
FRAMEWORK_COLORS = ["#E8F0FA", "#E5F4E6", "#FDEAEA"]  # light tints of B/G/R
FRAMEWORK_EDGE  = [C_BLUE, C_GREEN, C_RED]


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Framework overview diagram
# ---------------------------------------------------------------------------
def draw_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.axis("off")

    boxes = [
        (0.03, 0.55, "Q1  static calibration", "Budget enumeration\nRB granularity + QoS closure"),
        (0.36, 0.55, "Q2  dynamic baseline", "Finite-horizon MPC\nqueues + channel evolution"),
        (0.69, 0.55, "Q3  coupled control", "Hierarchical PPO\nslice budget + power control"),
    ]

    for idx, (x, y, title, body) in enumerate(boxes):
        rect = FancyBboxPatch(
            (x, y), 0.27, 0.28,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=FRAMEWORK_EDGE[idx],
            facecolor=FRAMEWORK_COLORS[idx],
        )
        ax.add_patch(rect)
        ax.text(x + 0.135, y + 0.20, title, ha="center", va="center",
                fontsize=12, weight="bold", color="#222222")
        ax.text(x + 0.135, y + 0.09, body, ha="center", va="center",
                fontsize=10, color="#444444")

    for x0, x1 in [(0.30, 0.36), (0.63, 0.69)]:
        ax.add_patch(FancyArrowPatch(
            (x0, 0.69), (x1, 0.69), arrowstyle="-|>",
            mutation_scale=16, linewidth=1.5, color="#555555",
        ))

    ax.text(0.5, 0.33, "Unified objective: long-term weighted average service utility\nunder heterogeneous slice SLA constraints",
            ha="center", fontsize=11, color="#333333")
    ax.text(0.5, 0.19, "Two time scales: 100 ms slice / power decisions  +  1 ms task service execution",
            ha="center", fontsize=11, color="#333333")
    save(fig, "framework.png")


# ---------------------------------------------------------------------------
# Q1 — static optima bar chart
# ---------------------------------------------------------------------------
def draw_q1_static_optima() -> None:
    allocations = {
        "(30,10,10)": [30, 10, 10],
        "(20,20,10)": [20, 20, 10],
        "(20,10,20)": [20, 10, 20],
    }
    labels = ["URLLC", "eMBB", "mMTC"]
    colors = [SLICE_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    left = [0] * len(allocations)
    y = list(range(len(allocations)))
    for idx, label in enumerate(labels):
        vals = [v[idx] for v in allocations.values()]
        ax.barh(y, vals, left=left, color=colors[idx], label=label, alpha=0.88)
        for j, val in enumerate(vals):
            ax.text(left[j] + val / 2, j, str(val),
                    ha="center", va="center", color="white", fontsize=10, weight="bold")
        left = [left[j] + vals[j] for j in range(len(vals))]
    ax.set_yticks(y, list(allocations.keys()))
    ax.set_xlim(0, 50)
    ax.set_xlabel("Allocated RBs")
    ax.set_title("Static single-BS optimal allocations  (U = 0.9808)")
    ax.grid(axis="x", alpha=0.2, color="#CCCCCC")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)
    save(fig, "q1_static_optima.png")


# ---------------------------------------------------------------------------
# Q2 — lookahead tradeoff (dual-axis: line + bar)
# ---------------------------------------------------------------------------
def draw_q2_tradeoff() -> None:
    depths = [1, 2, 3]
    objectives = [load_json(f"outputs/q2_mpc/lookahead{h}.json")["objective"] for h in depths]
    times = []
    for h in depths:
        time_path = ROOT / "outputs" / "q2_mpc" / f"lookahead{h}_time.txt"
        raw = time_path.read_bytes()
        text = (raw.decode("utf-16", errors="ignore") if raw.startswith(b"\xff\xfe")
                else raw.decode("utf-8", errors="ignore"))
        seconds_line = [ln for ln in text.splitlines() if ln.strip().startswith("TotalSeconds")]
        if seconds_line:
            times.append(float(seconds_line[-1].split(":", 1)[1].strip()))
        else:
            vals = [float(tok) for tok in text.replace("=", " ").replace(",", " ").split()
                    if tok.replace(".", "", 1).isdigit()]
            times.append(vals[-1])

    fig, ax1 = plt.subplots(figsize=(7.4, 4.0))
    ax1.plot(depths, objectives, marker="o", linewidth=2.2, color=C_BLUE, label="Objective")
    ax1.set_xlabel("Lookahead depth")
    ax1.set_ylabel("Objective", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_xticks(depths)
    ax1.grid(alpha=0.2, color="#CCCCCC")

    ax2 = ax1.twinx()
    ax2.bar(depths, times, width=0.34, alpha=0.30, color=C_RED, label="Runtime")
    ax2.set_ylabel("Runtime (s)", color=C_RED)
    ax2.tick_params(axis="y", labelcolor=C_RED)
    ax1.set_title("Q2  lookahead tradeoff: objective vs. runtime")
    save(fig, "q2_lookahead_tradeoff.png")


# ---------------------------------------------------------------------------
# Q2 — decision sequence stacked bar
# ---------------------------------------------------------------------------
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
    for label, series in vals.items():
        ax.bar(steps, series, bottom=bottom, label=label, color=SLICE_COLORS[label], alpha=0.88)
        bottom = [bottom[i] + series[i] for i in range(len(series))]
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Allocated RBs")
    ax.set_ylim(0, 50)
    ax.set_xticks(steps)
    ax.set_title("Q2  selected RB budget sequence  (lookahead = 2)")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.2, color="#CCCCCC")
    save(fig, "q2_decision_sequence.png")


# ---------------------------------------------------------------------------
# Q3 — multi-seed summary (bar + completion ratio)
# ---------------------------------------------------------------------------
def draw_q3_summary() -> None:
    baseline = load_json("outputs/q3_rl/short_run_metrics.json")["best_eval_objective"]
    paths = [
        ("seed 7",  "outputs/q3_sb3/q3_combined_eval_fresh_10k.json"),
        ("seed 17", "outputs/q3_sb3/q3_combined_eval_seed17_10k.json"),
        ("seed 27", "outputs/q3_sb3/q3_combined_eval_seed27_10k.json"),
    ]
    labels = ["baseline"] + [p[0] for p in paths]
    objectives = [baseline] + [
        load_json(p[1])["evaluation"]["summary"]["objective"] for p in paths
    ]
    seed17 = load_json("outputs/q3_sb3/q3_combined_eval_seed17_10k.json")["evaluation"]["summary"]
    ratios = {k: seed17["completed"][k] / seed17["total_arrivals"][k] for k in ["u", "e", "m"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    bar_colors = [C_GREY] + [C_CYAN, C_BLUE, C_DARK]
    ax1.bar(labels, objectives, color=bar_colors)
    ax1.set_ylim(0.43, 0.51)
    ax1.set_ylabel("Objective")
    ax1.set_title("Q3  multi-seed objective comparison")
    ax1.grid(axis="y", alpha=0.2, color="#CCCCCC")
    for i, v in enumerate(objectives):
        ax1.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    comp_colors = [SLICE_COLORS[k] for k in ["u", "e", "m"]]
    ax2.bar(["URLLC", "eMBB", "mMTC"], [ratios["u"], ratios["e"], ratios["m"]],
            color=comp_colors, alpha=0.88)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Completion ratio")
    ax2.set_title("Best model (seed 17)  completion ratio")
    ax2.grid(axis="y", alpha=0.2, color="#CCCCCC")
    for i, key in enumerate(["u", "e", "m"]):
        ax2.text(i, ratios[key] + 0.03, f"{ratios[key]:.1%}", ha="center", fontsize=9)
    save(fig, "q3_multiseed_summary.png")


# ---------------------------------------------------------------------------
# Sensitivity — 2×2 panel
# ---------------------------------------------------------------------------
def draw_sensitivity_summary() -> None:
    q2_alpha_paths = [
        ("0.90", "sensitivity/q2_alpha/q2_alpha_0.90.json"),
        ("0.93", "sensitivity/q2_alpha/q2_alpha_0.93.json"),
        ("0.95", "sensitivity/q2_alpha/q2_alpha_0.95.json"),
        ("0.97", "sensitivity/q2_alpha/q2_alpha_0.97.json"),
    ]
    q2_beta_paths = [
        ("(4,2,1)", "sensitivity/q2_beta/q2_beta_421.json"),
        ("(5,3,1)", "sensitivity/q2_beta/q2_beta_531.json"),
        ("(6,4,2)", "sensitivity/q2_beta/q2_beta_642.json"),
    ]
    q3_alpha_paths = [
        ("0.90", "sensitivity/q3_alpha/q3_eval_alpha_0.90.json"),
        ("0.93", "sensitivity/q3_alpha/q3_eval_alpha_0.93.json"),
        ("0.95", "sensitivity/q3_alpha/q3_eval_alpha_0.95.json"),
        ("0.97", "sensitivity/q3_alpha/q3_eval_alpha_0.97.json"),
    ]
    q3_beta_paths = [
        ("(4,2,1)", "sensitivity/q3_beta/q3_eval_b421.json"),
        ("(5,3,1)", "sensitivity/q3_beta/q3_eval_b531.json"),
        ("(6,4,2)", "sensitivity/q3_beta/q3_eval_b642.json"),
    ]

    def q2_obj(paths):
        return [load_json(p)["objective"] for _, p in paths]

    def q3_obj(paths):
        return [load_json(p)["evaluation"]["summary"]["objective"] for _, p in paths]

    panels = [
        ("Q2  α_u  perturbation",  q2_alpha_paths, q2_obj(q2_alpha_paths)),
        ("Q2  β  perturbation",    q2_beta_paths,  q2_obj(q2_beta_paths)),
        ("Q3  α_u  perturbation",  q3_alpha_paths, q3_obj(q3_alpha_paths)),
        ("Q3  β  perturbation",    q3_beta_paths,  q3_obj(q3_beta_paths)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    for ax, (title, paths, values), color in zip(axes.ravel(), panels, SENS_COLORS):
        labels = [label for label, _ in paths]
        ax.plot(labels, values, marker="o", linewidth=2.0, color=color)
        ax.set_title(title)
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.2, color="#CCCCCC")
        ax.margins(x=0.08)
        ymin = min(values) - 0.015
        ymax = max(values) + 0.015
        ax.set_ylim(ymin, ymax)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.004, f"{value:.4f}", ha="center", fontsize=9)

    fig.tight_layout()
    save(fig, "sensitivity_objectives.png")


# ---------------------------------------------------------------------------
# Copy pre-existing convergence curve
# ---------------------------------------------------------------------------
def copy_training_curve() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "docs" / "assets" / "q3_slice_seed17_convergence.png",
        FIG_DIR / "q3_slice_seed17_convergence.png",
    )


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
    })
    draw_framework()
    draw_q1_static_optima()
    draw_q2_tradeoff()
    draw_q2_actions()
    draw_q3_summary()
    draw_sensitivity_summary()
    copy_training_curve()
