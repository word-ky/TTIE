"""Aggregate the complete T002 matrix without selecting configurations."""

import argparse
import json
from pathlib import Path
from statistics import mean

from .suite import CONDITIONS, FAMILIES, VARIANTS


def summarize(path: Path):
    rows = json.loads((path / "metrics.json").read_text())
    grouped = {(family, condition, name): [r for r in rows if (r["family"], r["condition"], r["model"]) == (family, condition, name)]
               for family in FAMILIES for condition in CONDITIONS for name, _, _ in VARIANTS}
    aggregate = []
    for (family, condition, name), values in grouped.items():
        row = {"family": family, "condition": condition, "model": name, "n": len(values)}
        for key in values[0]:
            if key in ("seed", "family", "condition", "model", "all_finite"):
                continue
            measured = [r[key] for r in values if r[key] is not None]
            row[key] = mean(measured) if measured else None
        aggregate.append(row)
    index = {(r["seed"], r["family"], r["condition"], r["model"]): r for r in rows}
    uniform_differences, spatial1_differences = [], []
    for key, row in index.items():
        if key[-1] == "global":
            uniform_differences.append(abs(row["mse"] - index[(*key[:-1], "uniform96")]["mse"]))
            spatial1_differences.append(abs(row["mse"] - index[(*key[:-1], "spatial1")]["mse"]))
    summary = {"rows": len(rows), "all_finite": all(r["all_finite"] for r in rows),
               "max_uniform_vs_global_mse_abs_difference": max(uniform_differences),
               "max_spatial1_vs_global_mse_abs_difference": max(spatial1_differences),
               "max_uniform_field_variance": max(v for r in rows if r["model"] == "uniform96" for k, v in r.items() if k.startswith("field_variance_")),
               "loss_increase_rows": sum(r["loss_after"] > r["loss_before"] + 1e-8 for r in rows),
               "seed_means": aggregate}
    (path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lookup = {(r["family"], r["condition"], r["model"]): r for r in aggregate}
    lines = ["# T002 complete matrix summary", "", f"{len(rows)} rows; all finite: {summary['all_finite']}. Means across all declared seeds. No per-case selection.",
             "", f"Max |uniform96-global| MSE: {max(uniform_differences):.9g}; max |spatial1-global| MSE: {max(spatial1_differences):.9g}.",
             "", "Uniform96 still renders a six-dimensional global function; extra symmetric latent scalars do not add spatial or nonlinear expressivity."]
    names = [name for name, _, _ in VARIANTS]
    for family in FAMILIES:
        lines.extend(["", "## " + family + " — evaluation MSE", "", "| Condition | " + " | ".join(names) + " |", "| --- | " + " | ".join(["---:"] * len(names)) + " |"])
        for condition in CONDITIONS:
            lines.append("| " + condition + " | " + " | ".join(f"{lookup[(family, condition, name)]['mse']:.7f}" for name in names) + " |")
    lines.extend(["", "## Spatial4 compared with uniform96", "", "| Family | Condition | MSE reduction % | Input clipping % | Spatial4 output clipping % |", "| --- | --- | ---: | ---: | ---: |"])
    for family in FAMILIES:
        for condition in CONDITIONS:
            a, b = lookup[(family, condition, "uniform96")], lookup[(family, condition, "spatial4")]
            improvement = 100 * (1 - b["mse"] / a["mse"])
            lines.append(f"| {family} | {condition} | {improvement:.3f} | {100*b['input_clipping_fraction']:.3f} | {100*b['output_clipping_fraction']:.3f} |")
    lines.extend(["", "Clean-condition MSE is identity drift. Identity PSNR for zero MSE is infinite (null in JSON, blank in CSV). Non-clean input_output_drift_mse is the magnitude of editing, not reference recovery error.",
                  "", "High-frequency stripe results also reflect 8x8 patch-loss averaging/aliasing, content priors, operator flexibility and clipping; they do not isolate representation resolution alone."])
    (path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "seed_means"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    summarize(parser.parse_args().path)
