"""Predeclared T003 Pareto diagnostic; no per-image configuration selection."""

import argparse
import json
from pathlib import Path
from statistics import mean

from .safety_sweep import SETTINGS
from .suite import FAMILIES


UTILITY_CONDITIONS = ("left_right", "quadrants", "smooth_gradient")


def diagnostic_points(rows):
    index = {(r["seed"], r["family"], r["condition"], r["model"]): r for r in rows}
    global_utility = [r for r in rows if r["model"] == "global" and r["family"] == "midtone" and r["condition"] in UTILITY_CONDITIONS]
    def improvement(name):
        return mean(r["mse"] - index[(r["seed"], r["family"], r["condition"], name)]["mse"] for r in global_utility)
    baseline_improvement = improvement("a0_s0")
    baseline_worst = max(r["identity_drift_mse"] for r in rows if r["model"] == "a0_s0" and r["condition"] == "clean")
    points = []
    for name, a, s in SETTINGS:
        clean = [r for r in rows if r["model"] == name and r["condition"] == "clean"]
        family_means = {family: mean(r["identity_drift_mse"] for r in clean if r["family"] == family) for family in FAMILIES}
        worst = max(r["identity_drift_mse"] for r in clean)
        gain = improvement(name)
        point = {"model": name, "lambda_a": a, "lambda_s": s,
                 "clean_mean_drift": mean(r["identity_drift_mse"] for r in clean),
                 "clean_worst_drift": worst, "clean_family_means": family_means,
                 "clean_worst_family_mean": max(family_means.values()),
                 "drift_reduction_factor": baseline_worst / worst,
                 "mean_heterogeneous_mse_improvement": gain,
                 "utility_retention": gain / baseline_improvement}
        point["meets_both_thresholds"] = worst <= baseline_worst / 5 and gain / baseline_improvement >= .7
        points.append(point)
    return {"baseline_worst_clean_drift": baseline_worst, "safety_max_drift": baseline_worst / 5,
            "baseline_heterogeneous_improvement": baseline_improvement, "minimum_utility_retention": .7,
            "qualifying_settings": [p["model"] for p in points if p["meets_both_thresholds"]], "settings": points}


def write_summary(output: Path):
    rows = json.loads((output / "metrics.json").read_text())
    summary = diagnostic_points(rows)
    summary.update({"rows": len(rows), "all_finite": all(r["all_finite"] for r in rows),
                    "utility_formula": "mean(MSE_global-MSE_setting) / mean(MSE_global-MSE_unregularized_spatial4), nine fixed midtone low-frequency inputs",
                    "safety_formula": "mean and maximum over nine clean inputs (three families x three seeds); worst family mean also reported"})
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# T003 complete fixed Pareto diagnostic", "", f"{len(rows)} rows; all finite: {summary['all_finite']}.",
             "", "One setting must reduce worst clean drift by at least 5x AND retain at least 70% of baseline improvement. These are diagnostic thresholds, not a per-image selection policy.",
             "", f"Baseline worst drift: {summary['baseline_worst_clean_drift']:.9g}; safety ceiling: {summary['safety_max_drift']:.9g}; baseline mean MSE improvement: {summary['baseline_heterogeneous_improvement']:.9g}.",
             "", "| Setting (anchor, TV) | Mean clean drift | Worst clean drift | Worst family mean | Drift reduction | Utility retained | Meets both |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for p in summary["settings"]:
        lines.append(f"| ({p['lambda_a']:g}, {p['lambda_s']:g}) | {p['clean_mean_drift']:.7f} | {p['clean_worst_drift']:.7f} | {p['clean_worst_family_mean']:.7f} | {p['drift_reduction_factor']:.3f}x | {100*p['utility_retention']:.3f}% | {p['meets_both_thresholds']} |")
    lines.extend(["", "Qualifying fixed settings: " + (", ".join(summary["qualifying_settings"]) or "NONE"),
                  "", "Safety uses the worst individual input among the nine clean cases; the alternative worst family-mean is displayed without changing the decision rule. Utility is a ratio of mean improvements, not a mean of per-image ratios. Negative utility is retained, not clipped.",
                  "", "All input-level errors, region errors, clipping and grid diagnostics remain in metrics.csv/metrics.json. All prior/anchor/TV/total and gradient trajectories are in trajectories/. Seeds vary synthetic noise only. Passing does not establish natural-image identity safety; failing rules out only this finite diagnostic weight sweep and optimization budget."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("qualifying_settings:", summary["qualifying_settings"], "rows:", len(rows), "all_finite:", summary["all_finite"])
    return summary


def plot_pareto(output: Path):
    # Plotting is local postprocessing; it never feeds back to the sweep.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((output / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1, .85]}, layout="constrained")
    colors = ["black"] + ["#2166ac"] * 4 + ["#d95f02"] * 3 + ["#1b7837"] * 3
    for axis, key, title in zip(axes[:2], ("clean_mean_drift", "clean_worst_drift"), ("Mean clean-input drift", "Worst clean-input drift")):
        for i, (p, color) in enumerate(zip(summary["settings"], colors)):
            axis.scatter(p[key], 100*p["utility_retention"], c=color, s=45, marker="*" if p["meets_both_thresholds"] else "o", zorder=3)
            axis.annotate(str(i), xy=(p[key], 100*p["utility_retention"]), xytext=(.98, .95-i*.078), textcoords="axes fraction", color=color,
                          ha="right", fontsize=9, arrowprops={"arrowstyle": "-", "lw": .5, "color": color, "alpha": .6})
        axis.set_xscale("log")
        axis.axhline(70, color="#555555", ls="--", lw=1, label="70% retention")
        axis.axhline(0, color="#bbbbbb", lw=.8)
        if key == "clean_worst_drift":
            axis.axvline(summary["safety_max_drift"], color="#555555", ls=":", lw=1, label="5x drift reduction")
            axis.legend(fontsize=8, loc="lower left")
        axis.set_title(title)
        axis.set_xlabel("MSE to unchanged input (log scale; lower is better)", fontsize=9)
        axis.set_ylabel("Heterogeneous utility retained (%)")
        axis.grid(alpha=.18)
    axes[2].axis("off")
    axes[2].text(0, .98, "Fixed weight settings\nID: (anchor, TV)", va="top", fontsize=12, weight="bold")
    for i, (p, color) in enumerate(zip(summary["settings"], colors)):
        axes[2].text(0, .85-i*.061, f"{i}: ({p['lambda_a']:g}, {p['lambda_s']:g})", color=color, fontsize=11)
    axes[2].text(0, .1, "Decision: " + (", ".join(summary["qualifying_settings"]) if summary["qualifying_settings"] else "no setting meets both"), fontsize=10, wrap=True)
    axes[2].text(0, .01, "All 3 seeds / 3 clean families.\nUtility: midtone LR, quadrants, gradient.\nEvery setting retained; no per-image selection.", fontsize=9, va="bottom")
    fig.suptitle("T003: fixed identity-anchor / TV sweep", fontsize=15)
    fig.savefig(output / "pareto.png", dpi=180)
    fig.savefig(output / "pareto.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    write_summary(args.path)
    if args.plot:
        plot_pareto(args.path)
