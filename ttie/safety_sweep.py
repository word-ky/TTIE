"""T003 fixed regularization sweep, reusing unchanged T002 input generation."""

import argparse
import json
import platform
from pathlib import Path

import torch

from .adapt import adapt, local_statistics_loss
from .isp import PARAMETER_NAMES
from .regularization import normalized_correction
from .suite import FAMILIES, CONDITIONS, clean_family, degraded_image, evaluate_case, save_panel, save_rows


WEIGHTS = ((0., 0.), (.01, 0.), (.1, 0.), (1., 0.), (10., 0.),
           (0., .01), (0., .1), (0., 1.), (.1, .1), (1., .1), (1., 1.))
SETTINGS = tuple((f"a{a:g}_s{s:g}", a, s) for a, s in WEIGHTS)
VARIANTS = (("identity", None, 1), ("global", "global", 1)) + tuple((name, "spatial", 4) for name, _, _ in SETTINGS)
VISUAL_COLUMNS = ("clean", "identity", "global", "a0_s0", "a0.1_s0.1", "a1_s0.1", "a10_s0")
VISUAL_CASES = (("dark_structures", "clean"), ("high_key", "clean"), ("midtone", "left_right"))


def adapt_settings(image, *, steps: int = 200, lr: float = .03):
    """Only current image enters all 12 fresh episodes; no evaluation metadata."""
    results = {"global": adapt(image, local_statistics_loss, steps=steps, lr=lr)}
    for name, a, s in SETTINGS:
        results[name] = adapt(image, local_statistics_loss, mode="spatial", grid_size=(4, 4),
                              steps=steps, lr=lr, lambda_a=a, lambda_s=s)
    return results


def evaluate_safety(clean, degraded, results, *, family, condition, seed):
    rows, pack = evaluate_case(clean, degraded, results, family=family, condition=condition, seed=seed, variants=VARIANTS)
    diagnostics = {}
    for row in rows:
        name = row["model"]
        result = results.get(name)
        if result is None:
            components = {"prior": [row["loss_before"]], "anchor": [0.], "smooth": [0.]}
            diag = {"loss_trajectory": [row["loss_before"]], "component_trajectories": components,
                    "gradient_norms": [], "steps": 0, "lr": 0., "lambda_a": 0., "lambda_s": 0.,
                    "parameter_count": 0, "all_finite": row["all_finite"]}
        else:
            # Every episode retains all loss components and gradient norms, compactly.
            diag = {key: result.diagnostics[key] for key in ("loss_trajectory", "component_trajectories", "gradient_norms", "steps", "lr", "lambda_a", "lambda_s", "parameter_count", "all_finite")}
            components = diag["component_trajectories"]
        diagnostics[name] = diag
        row.update({"lambda_a": diag["lambda_a"], "lambda_s": diag["lambda_s"],
                    "gradient_norm_last_update": diag["gradient_norms"][-1] if diag["gradient_norms"] else 0.,
                    "gradient_norm_max": max(diag["gradient_norms"], default=0.),
                    "correction_grid_tv": components["smooth"][-1]})
        for term, trajectory in components.items():
            row[term + "_before"] = trajectory[0]
            row[term + "_after"] = trajectory[-1]
        normalized = normalized_correction(pack[name + "_grid"])
        variance = normalized.double().var(dim=(-2, -1), correction=0).flatten().tolist()
        row.update({"normalized_grid_variance_" + key: value for key, value in zip(PARAMETER_NAMES, variance)})
    return rows, pack, diagnostics


def safety_case(*, family, condition, seed, device="cpu", steps=200, lr=.03):
    clean = clean_family(family, seed)
    degraded = degraded_image(clean, condition)
    results = adapt_settings(degraded.to(device), steps=steps, lr=lr)
    return evaluate_safety(clean, degraded, results, family=family, condition=condition, seed=seed)


def run_sweep(output: Path, *, device="cpu", seeds=(7, 11, 23), steps=200, lr=.03):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    config = {"seeds": list(seeds), "steps": steps, "lr": lr, "device": device, "settings": SETTINGS,
              "families": FAMILIES, "conditions": CONDITIONS, "visual_columns": VISUAL_COLUMNS,
              "visual_cases_seed7": VISUAL_CASES, "normalization": "EV/2; log2(gamma,WB-R,WB-G,WB-B,contrast)",
              "anchor": "mean(c^2) on coarse physical normalized grid",
              "smooth": "mean(abs(dx c))+mean(abs(dy c)); absent axes contribute zero",
              "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    all_rows, panels = [], {}
    for seed in seeds:
        for family in FAMILIES:
            for condition in CONDITIONS:
                rows, pack, diagnostics = safety_case(family=family, condition=condition, seed=seed, device=device, steps=steps, lr=lr)
                all_rows.extend(rows)
                save_rows(all_rows, output)
                case_name = f"seed{seed}_{family}_{condition}"
                (output / "trajectories" / (case_name + ".json")).write_text(json.dumps(diagnostics, separators=(",", ":"), allow_nan=False), encoding="utf-8")
                if seed == 7 and (family, condition) in VISUAL_CASES:
                    torch.save(pack, output / (case_name + ".pt"))
                    panels[(family, condition)] = pack
                print(f"finished {case_name} rows={len(all_rows)}", flush=True)
    if panels:
        save_panel([(family + "/" + condition, panels[(family, condition)]) for family, condition in VISUAL_CASES], output / "content_utility_panel.png",
                   columns=VISUAL_COLUMNS, title="T003: fixed representatives; RGB [0,1]; lower row EV (blue -2, white 0, red +2)")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("research_log/artifacts/T003_cpu"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 23))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.03)
    args = parser.parse_args()
    run_sweep(args.output, device=args.device, seeds=tuple(args.seeds), steps=args.steps, lr=args.lr)
