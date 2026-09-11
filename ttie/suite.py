"""T002 fixed synthetic suite: adaptation first, clean-reference evaluation after."""

import argparse
import csv
import json
import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

from .adapt import adapt, local_statistics_loss
from .demo import make_toy, recovery_metrics
from .isp import ISP, PARAMETER_NAMES


FAMILIES = ("midtone", "dark_structures", "high_key")
CONDITIONS = ("clean", "homogeneous_dark", "homogeneous_bright", "left_right",
              "quadrants", "smooth_gradient", "stripes_4", "stripes_12")
VARIANTS = (("identity", None, 1), ("global", "global", 1),
            ("uniform96", "uniform_control", 4), ("spatial1", "spatial", 1),
            ("spatial2", "spatial", 2), ("spatial4", "spatial", 4),
            ("spatial8", "spatial", 8))


def clean_family(family: str, seed: int) -> Tensor:
    clean, _ = make_toy(seed)
    if family == "dark_structures":
        y, x = torch.meshgrid(torch.arange(64), torch.arange(96), indexing="ij")
        intrinsic_dark = ((x > 10) & (x < 40) & (y > 12) & (y < 53)) | ((x - 72)**2 + (y - 32)**2 < 14**2)
        clean = clean * torch.where(intrinsic_dark, 0.35, 1.0)[None, None]
    elif family == "high_key":
        clean = 0.75 + 0.5 * (clean - 0.5)
    return clean


def gain_map(condition: str) -> Tensor:
    y, x = torch.meshgrid(torch.arange(64), torch.arange(96), indexing="ij")
    if condition == "clean":
        gain = torch.ones(64, 96)
    elif condition == "homogeneous_dark":
        gain = torch.full((64, 96), 0.45)
    elif condition == "homogeneous_bright":
        gain = torch.full((64, 96), 1.55)
    elif condition == "left_right":
        gain = torch.where(x < 48, 0.45, 1.55)
    elif condition == "quadrants":
        gain = torch.where(((x >= 48).int() + (y >= 32).int()) % 2 == 0, 0.45, 1.55)
    elif condition == "smooth_gradient":
        gain = 0.45 + 1.1 * x.float() / 95
    else:
        band = 12 if condition == "stripes_4" else 4
        gain = torch.where((x // band) % 2 == 0, 0.45, 1.55)
    return gain[None, None]


def degraded_image(clean: Tensor, condition: str) -> Tensor:
    return (clean * gain_map(condition)).clamp(0, 1)


def adapt_variants(degraded: Tensor, *, steps: int, lr: float) -> dict:
    """No clean image, family, masks, condition or gain enters adaptation."""
    return {name: adapt(degraded, local_statistics_loss, mode=mode, grid_size=(grid, grid), steps=steps, lr=lr)
            for name, mode, grid in VARIANTS if mode is not None}


def clipping_fraction(image: Tensor) -> float:
    return ((image <= 0) | (image >= 1)).float().mean().item()


def evaluate_case(clean: Tensor, degraded: Tensor, adapted: dict, *, family: str,
                  condition: str, seed: int) -> tuple[list[dict], dict]:
    """Evaluate completed adaptation; this function cannot update any parameters."""
    clean, degraded = clean.cpu(), degraded.cpu()
    rows = []
    pack = {"clean": clean, "identity": degraded}
    initial_loss = local_statistics_loss(degraded).item()
    for name, mode, _ in VARIANTS:
        result = adapted.get(name)
        output = degraded if result is None else result.image.cpu()
        field = ISP().parameter_field(clean.shape[-2:]).detach() if result is None else result.parameter_field.cpu()
        grid = ISP().physical_grid().detach() if result is None else result.parameter_grid.cpu()
        metrics = recovery_metrics(output, clean)
        if metrics["mse"] == 0:
            metrics["psnr_db"] = None  # mathematically +infinity, valid portable JSON
        squared = (output - clean).square()
        row = {"seed": seed, "family": family, "condition": condition, "model": name,
               "loss_before": initial_loss if result is None else result.diagnostics["loss_trajectory"][0],
               "loss_after": initial_loss if result is None else result.diagnostics["loss_trajectory"][-1],
               **metrics,
               "top_left_mse": squared[..., :32, :48].mean().item(),
               "top_right_mse": squared[..., :32, 48:].mean().item(),
               "bottom_left_mse": squared[..., 32:, :48].mean().item(),
               "bottom_right_mse": squared[..., 32:, 48:].mean().item(),
               "input_output_drift_mse": (output - degraded).square().mean().item(),
               "identity_drift_mse": (output - degraded).square().mean().item() if condition == "clean" else None,
               "parameter_count": 0 if result is None else result.diagnostics["parameter_count"],
               "input_clipping_fraction": clipping_fraction(degraded),
               "output_clipping_fraction": clipping_fraction(output),
               "all_finite": bool(torch.isfinite(output).all()) if result is None else result.diagnostics["all_finite"]}
        variances = field.double().var(dim=(-2, -1), correction=0).flatten().tolist()
        row.update({"field_variance_" + key: value for key, value in zip(PARAMETER_NAMES, variances)})
        rows.append(row)
        pack[name] = output
        pack[name + "_ev"] = field[:, :1]
        pack[name + "_grid"] = grid
    return rows, pack


def case_results(*, family: str, condition: str, seed: int, device: str,
                 steps: int, lr: float) -> tuple[list[dict], dict, dict]:
    clean = clean_family(family, seed)
    degraded = degraded_image(clean, condition)
    adapted = adapt_variants(degraded.to(device), steps=steps, lr=lr)
    rows, pack = evaluate_case(clean, degraded, adapted, family=family, condition=condition, seed=seed)
    return rows, pack, {name: result.diagnostics for name, result in adapted.items()}


def save_rows(rows: list[dict], output: Path):
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(rows, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def save_panel(packs: list[tuple[str, dict]], destination: Path):
    """Corrected images and exposure fields use fixed, shared display scales."""
    columns = ("clean", "identity", "global", "uniform96", "spatial4", "spatial8")
    cell_w, cell_h, label_w = 192, 128, 185
    row_h, title_h = 276, 64
    canvas = Image.new("RGB", (label_w + cell_w * len(columns), title_h + row_h * len(packs)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((8, 8), "T002: output RGB [0,1]; lower row = exposure EV (blue -2, white 0, red +2)", fill="black", font=font)
    for index, name in enumerate(columns):
        draw.text((label_w + index * cell_w + 6, 38), name, fill="black", font=font)
    for row_index, (label, pack) in enumerate(packs):
        top = title_h + row_index * row_h
        draw.multiline_text((8, top + 8), label.replace("/", "\n"), fill="black", font=font)
        draw.text((8, top + cell_h + 8), "EV field", fill="black", font=font)
        for index, name in enumerate(columns):
            rgb = (pack[name][0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
            tile = Image.fromarray(rgb).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            canvas.paste(tile, (label_w + index * cell_w, top))
            if name == "clean":
                continue
            ev = pack[name + "_ev"][0, 0] / 2
            colors = torch.stack((1 - (-ev).clamp_min(0), 1 - ev.abs(), 1 - ev.clamp_min(0)), dim=-1)
            tile = Image.fromarray((255 * colors.clamp(0, 1)).round().byte().numpy()).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            canvas.paste(tile, (label_w + index * cell_w, top + cell_h + 8))
    canvas.save(destination)


def run_suite(output: Path, *, device: str = "cpu", seeds: tuple[int, ...] = (7, 11, 23),
              steps: int = 200, lr: float = 0.03):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    config = {"seeds": list(seeds), "steps": steps, "lr": lr, "device": device,
              "families": FAMILIES, "conditions": CONDITIONS, "variants": VARIANTS,
              "loss": "unchanged T001 mean((avg_pool2d(output,8)-0.5)^2)",
              "python": platform.python_version(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    all_rows, degradation_packs, content_packs = [], [], []
    for seed in seeds:
        for family in FAMILIES:
            for condition in CONDITIONS:
                rows, pack, diagnostics = case_results(family=family, condition=condition, seed=seed,
                                                        device=device, steps=steps, lr=lr)
                all_rows.extend(rows)
                # Material case results persist as work proceeds, no model selection.
                save_rows(all_rows, output)
                if seed == 7 and (family == "midtone" or condition == "clean"):
                    label = family + "/" + condition
                    torch.save(pack, output / (family + "_" + condition + ".pt"))
                    (output / (family + "_" + condition + "_diagnostics.json")).write_text(json.dumps(diagnostics, separators=(",", ":"), allow_nan=False), encoding="utf-8")
                    if family == "midtone":
                        degradation_packs.append((label, pack))
                    if condition == "clean":
                        content_packs.append((label, pack))
                print(f"finished seed={seed} family={family} condition={condition} rows={len(all_rows)}", flush=True)
    if degradation_packs:
        save_panel(degradation_packs, output / "degradations.png")
    if content_packs:
        save_panel(content_packs, output / "content_drift.png")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("research_log/artifacts/T002_cpu"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 23))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    run_suite(args.output, device=args.device, seeds=tuple(args.seeds), steps=args.steps, lr=args.lr)
