from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path("D:/456")
PROJECT = ROOT / "project"
RESULTS = ROOT / "results"
TABLES = PROJECT / "tables"
FIGURES = PROJECT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

METHOD_NAMES = {
    "clip_zero_shot": "CLIP Zero-Shot",
    "prompt_ensemble": "Prompt Ensemble",
    "source_prototype": "Source Prototype",
    "source_anchored_text": "Text + Source",
    "no_source_anchor": "Text + Target",
    "satpa_no_uncertainty": "SA-TPA w/o uncertainty",
    "t3a": "T3A (protocol-compatible)",
    "tip_adapter_source": "Tip-Adapter source cache",
    "satpa": "SA-TPA",
}
MAIN_METHODS = [
    "clip_zero_shot", "prompt_ensemble", "source_prototype",
    "t3a", "tip_adapter_source", "satpa",
]
COLORS = {
    "clip_zero_shot": "#9CA3AF",
    "prompt_ensemble": "#60A5FA",
    "source_prototype": "#F59E0B",
    "t3a": "#8B5CF6",
    "tip_adapter_source": "#EC4899",
    "satpa": "#0F766E",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_table(frame: pd.DataFrame, stem: str, index: bool = False):
    frame.to_csv(TABLES / f"{stem}.csv", index=index, encoding="utf-8-sig")
    latex = frame.to_latex(index=index, float_format=lambda value: f"{value:.3f}", escape=True)
    (TABLES / f"{stem}.tex").write_text(latex, encoding="utf-8")


def main_table(benchmark: str, tasks: list[str]) -> pd.DataFrame:
    if benchmark == "office31":
        rows = pd.concat([
            read_csv(RESULTS / "office31" / "development_v2.csv"),
            read_csv(RESULTS / "office31" / "additional_baselines_v1.csv"),
        ], ignore_index=True)
    else:
        rows = pd.concat([
            read_csv(RESULTS / "officehome" / "confirmatory_v1.csv"),
            read_csv(RESULTS / "officehome" / "additional_baselines_v1.csv"),
        ], ignore_index=True)
    rows = rows[rows.method.isin(MAIN_METHODS)]
    if rows.duplicated(["task", "method"]).any():
        duplicates = rows[rows.duplicated(["task", "method"], keep=False)][["task", "method"]]
        raise ValueError(f"Duplicate main results: {duplicates.to_dict('records')}")
    pivot = rows.pivot(index="method", columns="task", values="accuracy").reindex(MAIN_METHODS)
    pivot = pivot.reindex(columns=tasks)
    pivot["Mean"] = pivot.mean(axis=1)
    pivot.index = [METHOD_NAMES[item] for item in pivot.index]
    pivot.index.name = "Method"
    return pivot.reset_index()


def build_backbone_table() -> pd.DataFrame:
    b16 = read_csv(RESULTS / "vitb16" / "vitb16_summary.csv")
    b32_rows = []
    for benchmark, path in [
        ("office31", RESULTS / "office31" / "development_v2.csv"),
        ("officehome", RESULTS / "officehome" / "confirmatory_v1.csv"),
    ]:
        data = read_csv(path)
        for method in ["clip_zero_shot", "prompt_ensemble", "no_source_anchor", "satpa"]:
            b32_rows.append({
                "benchmark": benchmark,
                "method": method,
                "backbone": "ViT-B/32",
                "accuracy": data.loc[data.method == method, "accuracy"].mean(),
            })
    b16 = b16.assign(backbone="ViT-B/16")[["benchmark", "method", "backbone", "accuracy"]]
    data = pd.concat([pd.DataFrame(b32_rows), b16], ignore_index=True)
    data["Method"] = data.method.map(METHOD_NAMES)
    pivot = data.pivot(index=["benchmark", "Method"], columns="backbone", values="accuracy").reset_index()
    pivot["B16 - B32"] = pivot["ViT-B/16"] - pivot["ViT-B/32"]
    return pivot.rename(columns={"benchmark": "Benchmark"})


def build_ablation_table() -> pd.DataFrame:
    office31 = read_csv(RESULTS / "ablations" / "ablations_v1.csv")
    officehome = read_csv(RESULTS / "officehome" / "confirmatory_v1.csv")
    methods = ["prompt_ensemble", "source_anchored_text", "no_source_anchor", "satpa_no_uncertainty", "satpa"]
    rows = []
    for method in methods:
        rows.append({
            "Method": METHOD_NAMES[method],
            "Text": "Yes",
            "Source anchor": "Yes" if method in {"source_anchored_text", "satpa_no_uncertainty", "satpa"} else "No",
            "Target prototype": "Yes" if method in {"no_source_anchor", "satpa_no_uncertainty", "satpa"} else "No",
            "Uncertainty": "Yes" if method in {"no_source_anchor", "satpa"} else "No",
            "Office-31": office31.loc[office31.method == method, "accuracy"].mean(),
            "Office-Home": officehome.loc[officehome.method == method, "accuracy"].mean(),
        })
    return pd.DataFrame(rows)


def build_sensitivity_table() -> pd.DataFrame:
    data = read_csv(RESULTS / "sensitivity" / "sensitivity_v1.csv")
    records = []
    prefixes = {
        "alpha_source_": "alpha_source",
        "alpha_target_": "alpha_target",
        "threshold_": "confidence_threshold",
        "top_k_": "top_k",
        "prior_": "class_prior_strength",
    }
    for tag, group in data.groupby("run_tag"):
        for prefix, parameter in prefixes.items():
            if tag.startswith(prefix):
                value = float(tag[len(prefix):].replace("p", "."))
                records.append({
                    "Parameter": parameter,
                    "Value": value,
                    "Mean accuracy": group.accuracy.mean(),
                    "Mean ECE": group.ece.mean(),
                    "Worst-task accuracy": group.accuracy.min(),
                })
                break
    return pd.DataFrame(records).sort_values(["Parameter", "Value"]).reset_index(drop=True)


def build_efficiency_table() -> pd.DataFrame:
    frames = []
    for benchmark in ["office31", "officehome"]:
        frame = read_csv(RESULTS / "robustness" / f"{benchmark}_efficiency.csv")
        frame.insert(0, "Benchmark", benchmark)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["Method"] = data.method.map(METHOD_NAMES)
    return data[[
        "Benchmark", "benchmark_task", "Method", "median_adaptation_ms", "p95_adaptation_ms",
        "peak_python_numpy_memory_mb", "backpropagation", "trainable_parameters",
    ]].rename(columns={
        "benchmark_task": "Representative task",
        "median_adaptation_ms": "Median latency (ms)",
        "p95_adaptation_ms": "P95 latency (ms)",
        "peak_python_numpy_memory_mb": "Peak host allocation (MB)",
        "backpropagation": "Backpropagation",
        "trainable_parameters": "Trainable parameters",
    })


def build_design_validation() -> pd.DataFrame:
    original = read_csv(RESULTS / "office31" / "development_v2.csv")
    dev_tasks = ["A2W", "W2A", "D2W"]
    original_dev = original[(original.method == "satpa") & original.task.isin(dev_tasks)].accuracy.mean()
    adaptive = read_csv(RESULTS / "adaptive" / "p0_development.csv")
    adaptive_means = adaptive[adaptive.method == "adaptive_satpa"].groupby("reliability_tau").accuracy.mean()
    best_tau = adaptive_means.idxmax()
    best_adaptive = adaptive_means.max()

    agreement_parts = [
        read_csv(RESULTS / "adaptive" / "p0_revision.csv").query("alpha_target == 0.1"),
        read_csv(RESULTS / "adaptive" / "p0_heldout_w2d.csv"),
        read_csv(RESULTS / "adaptive" / "p0_remaining_office31.csv"),
    ]
    agreement_o31 = pd.concat(agreement_parts, ignore_index=True).accuracy.mean()
    original_o31 = original[original.method == "satpa"].accuracy.mean()
    agreement_oh = read_csv(RESULTS / "adaptive" / "officehome_locked_extension.csv").accuracy.mean()
    original_oh = read_csv(RESULTS / "officehome" / "confirmatory_v1.csv").query("method == 'satpa'").accuracy.mean()

    iteration = read_csv(RESULTS / "adaptive" / "p1_iteration_a2w.csv").sort_values("target_update_steps")
    iteration_delta = iteration.accuracy.iloc[-1] - iteration.accuracy.iloc[0]
    spt = read_csv(RESULTS / "spt_sa" / "development_epsilon.csv")
    spt_best = spt.groupby("ot_epsilon").accuracy.mean().max()
    return pd.DataFrame([
        {"Extension": f"Class-adaptive fusion (best tau={best_tau:g})", "Scope": "Office-31, 3 development tasks", "Reference": original_dev, "Candidate": best_adaptive, "Delta": best_adaptive-original_dev, "Decision": "Reject"},
        {"Extension": "Agreement filtering", "Scope": "Office-31, 6 tasks", "Reference": original_o31, "Candidate": agreement_o31, "Delta": agreement_o31-original_o31, "Decision": "Not promoted"},
        {"Extension": "Agreement filtering", "Scope": "Office-Home, 12 tasks", "Reference": original_oh, "Candidate": agreement_oh, "Delta": agreement_oh-original_oh, "Decision": "Reject cross-dataset"},
        {"Extension": "Iterative update (3 vs 1 rounds)", "Scope": "Office-31 A2W", "Reference": iteration.accuracy.iloc[0], "Candidate": iteration.accuracy.iloc[-1], "Delta": iteration_delta, "Decision": "Reject"},
        {"Extension": "SPT-SA optimal transport", "Scope": "Office-31, 3 development tasks", "Reference": original_dev, "Candidate": spt_best, "Delta": spt_best-original_dev, "Decision": "Reject at gate"},
    ])


def build_baseline_setting_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["CLIP Zero-Shot", "No", "No", "No", "No", "Frozen inference", "Main table", "Single prompt"],
        ["Prompt Ensemble", "No", "No", "No", "No", "Frozen inference", "Main table", "Four fixed prompts"],
        ["Source Prototype", "Yes", "Yes", "No", "No", "Source-available UDA", "Main table", "Closed-form visual prototypes"],
        ["T3A", "No", "No", "No", "No", "Protocol-compatible batch adaptation", "Main table", "Project implementation; not official reproduction"],
        ["Tip-Adapter source cache", "Yes", "Yes", "No", "No", "Protocol-compatible source cache", "Main table", "Project implementation; not official reproduction"],
        ["SA-TPA", "Yes", "Yes", "No", "No", "Source-available UDA", "Main table", "Fixed tri-prototype fusion"],
        ["DPA", "No", "No", "No", "Yes", "Source-free adaptation", "Setting comparison", "Official code lacks Office configs and reports best target-test epoch"],
        ["DPE", "No", "No", "No", "Yes", "Online TTA", "Setting comparison", "Optimizes prototype residuals; different protocol"],
    ], columns=["Method", "Source data", "Source labels", "Target labels for adaptation", "Gradient update", "Setting", "Placement", "Qualification"])


def save_figure(fig, stem: str):
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_means(office31: pd.DataFrame, officehome: pd.DataFrame):
    means = []
    for benchmark, table in [("Office-31", office31), ("Office-Home", officehome)]:
        for method in MAIN_METHODS:
            name = METHOD_NAMES[method]
            means.append((benchmark, method, float(table.loc[table.Method == name, "Mean"].iloc[0])))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(2)
    width = 0.12
    for index, method in enumerate(MAIN_METHODS):
        values = [value for benchmark, current, value in means if current == method]
        ax.bar(x + (index - 2.5) * width, values, width, label=METHOD_NAMES[method], color=COLORS[method])
    ax.set_xticks(x, ["Office-31", "Office-Home"])
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_ylim(55, 88)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Mean accuracy under unified Office protocols")
    fig.tight_layout()
    save_figure(fig, "main_results_mean_accuracy")


def plot_task_gains():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, benchmark, path in [
        (axes[0], "Office-31", RESULTS / "office31" / "development_v2.csv"),
        (axes[1], "Office-Home", RESULTS / "officehome" / "confirmatory_v1.csv"),
    ]:
        data = read_csv(path)
        pivot = data[data.method.isin(["prompt_ensemble", "satpa"])].pivot(index="task", columns="method", values="accuracy")
        delta = (pivot.satpa - pivot.prompt_ensemble).sort_index()
        colors = ["#0F766E" if value >= 0 else "#DC2626" for value in delta]
        ax.bar(delta.index, delta.values, color=colors)
        ax.axhline(0, color="#374151", linewidth=0.8)
        ax.set_ylabel("Accuracy gain (pp)")
        ax.set_title(f"{benchmark}: SA-TPA minus Prompt Ensemble")
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "task_level_gains_over_prompt")


def plot_sensitivity(sensitivity: pd.DataFrame):
    parameters = ["alpha_source", "alpha_target", "confidence_threshold", "top_k", "class_prior_strength"]
    defaults = {"alpha_source": 0.1, "alpha_target": 0.025, "confidence_threshold": 0.7, "top_k": 1.0, "class_prior_strength": 0.1}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for ax, parameter in zip(axes.flat, parameters):
        group = sensitivity[sensitivity.Parameter == parameter]
        ax.plot(group.Value, group["Mean accuracy"], marker="o", color="#0F766E", label="Mean")
        ax.plot(group.Value, group["Worst-task accuracy"], marker="s", color="#F59E0B", label="Worst task")
        ax.axvline(defaults[parameter], color="#6B7280", linestyle="--", linewidth=1)
        ax.set_title(parameter.replace("_", " "))
        ax.set_xlabel("Value")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower right", bbox_to_anchor=(0.94, 0.08))
    save_figure(fig, "parameter_sensitivity")


def plot_efficiency(efficiency: pd.DataFrame, office31: pd.DataFrame, officehome: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for ax, benchmark, table in [(axes[0], "office31", office31), (axes[1], "officehome", officehome)]:
        subset = efficiency[efficiency.Benchmark == benchmark]
        benchmark_rows = pd.concat([
            read_csv(RESULTS / benchmark / ("development_v2.csv" if benchmark == "office31" else "confirmatory_v1.csv")),
            read_csv(RESULTS / benchmark / "additional_baselines_v1.csv"),
        ], ignore_index=True)
        for _, row in subset.iterrows():
            internal_method = next(key for key, value in METHOD_NAMES.items() if value == row.Method)
            accuracy = float(benchmark_rows.loc[benchmark_rows.method == internal_method, "accuracy"].mean())
            ax.scatter(row["Median latency (ms)"], accuracy, s=65, color="#0F766E" if row.Method == "SA-TPA" else "#60A5FA")
            label = row.Method.replace(" (protocol-compatible)", "")
            offset = (4, 4)
            if benchmark == "officehome" and row.Method == "SA-TPA":
                offset = (4, 8)
            elif benchmark == "officehome" and row.Method == "Text + Target":
                offset = (-62, -13)
            ax.annotate(label, (row["Median latency (ms)"], accuracy), xytext=offset, textcoords="offset points", fontsize=7)
        ax.set_xlabel("Median adaptation latency (ms)")
        ax.set_ylabel("Mean accuracy (%)")
        ax.set_title("Office-31" if benchmark == "office31" else "Office-Home")
        ax.grid(alpha=0.2)
    save_figure(fig, "accuracy_latency_tradeoff")


def plot_method_overview():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    boxes = [
        (0.3, 1.25, 1.8, 0.8, "Frozen CLIP\nfeatures", "#DBEAFE"),
        (2.7, 2.25, 2.0, 0.7, "Text prototypes\n4 fixed prompts", "#E0E7FF"),
        (2.7, 1.25, 2.0, 0.7, "Source prototypes\nlabeled source", "#FEF3C7"),
        (2.7, 0.25, 2.0, 0.7, "Target prototypes\nunlabeled target", "#D1FAE5"),
        (5.5, 1.05, 2.2, 1.1, "Fixed fusion\n0.875 / 0.100 / 0.025", "#CCFBF1"),
        (8.5, 1.25, 2.0, 0.8, "Cosine classifier\nno backpropagation", "#DCFCE7"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#334155", linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + width/2, y + height/2, label, ha="center", va="center", fontsize=10)
    arrows = [((2.1, 1.65), (2.7, 2.6)), ((2.1, 1.65), (2.7, 1.6)), ((2.1, 1.65), (2.7, 0.6)), ((4.7, 2.6), (5.5, 1.75)), ((4.7, 1.6), (5.5, 1.6)), ((4.7, 0.6), (5.5, 1.35)), ((7.7, 1.6), (8.5, 1.65))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#475569", "linewidth": 1.2})
    ax.set_title("SA-TPA: frozen, source-anchored tri-prototype adaptation", fontsize=13, pad=8)
    save_figure(fig, "satpa_method_overview")


def main():
    office31 = main_table("office31", ["A2W", "D2W", "W2A", "A2D", "D2A", "W2D"])
    officehome = main_table("officehome", ["A2C", "A2P", "A2R", "C2A", "C2P", "C2R", "P2A", "P2C", "P2R", "R2A", "R2C", "R2P"])
    backbone = build_backbone_table()
    ablation = build_ablation_table()
    sensitivity = build_sensitivity_table()
    efficiency = build_efficiency_table()
    design = build_design_validation()
    settings = build_baseline_setting_table()
    bootstrap = read_csv(RESULTS / "robustness" / "clustered_bootstrap" / "clustered_bootstrap_summary.csv")

    outputs = {
        "office31_main": office31,
        "officehome_main": officehome,
        "backbone_comparison": backbone,
        "ablation_summary": ablation,
        "sensitivity_summary": sensitivity,
        "efficiency_summary": efficiency,
        "clustered_bootstrap": bootstrap,
        "design_validation": design,
        "baseline_setting_comparison": settings,
    }
    for stem, frame in outputs.items():
        write_table(frame, stem)

    plot_main_means(office31, officehome)
    plot_task_gains()
    plot_sensitivity(sensitivity)
    plot_efficiency(efficiency, office31, officehome)
    plot_method_overview()

    summary = {
        "tables": {name: str(TABLES / f"{name}.csv") for name in outputs},
        "figures": [path.name for path in sorted(FIGURES.glob("*.png"))],
        "headline": {
            "office31_satpa": float(office31.loc[office31.Method == "SA-TPA", "Mean"].iloc[0]),
            "office31_prompt": float(office31.loc[office31.Method == "Prompt Ensemble", "Mean"].iloc[0]),
            "officehome_satpa": float(officehome.loc[officehome.Method == "SA-TPA", "Mean"].iloc[0]),
            "officehome_prompt": float(officehome.loc[officehome.Method == "Prompt Ensemble", "Mean"].iloc[0]),
        },
    }
    (TABLES / "final_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    workbook_data = {
        name: {
            "columns": list(frame.columns),
            "records": frame.replace({np.nan: None}).to_dict(orient="records"),
        }
        for name, frame in outputs.items()
    }
    (TABLES / "final_workbook_data.json").write_text(json.dumps(workbook_data, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
