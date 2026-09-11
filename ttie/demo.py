"""Deterministic controlled toy; clean reference enters evaluation only."""

import argparse
import json
import math
import platform
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch import Tensor

from .adapt import adapt, local_statistics_loss
from .isp import PARAMETER_NAMES


def make_toy(seed: int = 7) -> tuple[Tensor, Tensor]:
    """Return clean and degraded RGB, generated on CPU independently of runtime."""
    generator = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(64), torch.arange(96), indexing="ij")
    base = 0.5 + 0.1 * torch.sin(2 * math.pi * x / 8) + 0.05 * torch.cos(2 * math.pi * y / 8)
    channels = [base + 0.02 * torch.sin(2 * math.pi * (x + y) / 8 + phase)
                for phase in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
    clean = (torch.stack(channels)[None] + 0.005 * torch.randn(1, 3, 64, 96, generator=generator)).clamp(0, 1)
    gain = torch.where(x < 48, 0.45, 1.55)[None, None]
    degraded = (clean * gain).clamp(0, 1)
    return clean, degraded


def recovery_metrics(output: Tensor, clean: Tensor) -> dict:
    """Offline evaluation only; PSNR data_range=1 over all RGB pixels."""
    squared = (output - clean).square()
    mse = squared.mean().item()
    return {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse > 0 else float("inf"),
            "left_mse": squared[..., :48].mean().item(),
            "right_mse": squared[..., 48:].mean().item()}


def run_demo(*, seed: int = 7, device: str = "cpu", grid_size: tuple[int, int] = (4, 4),
             steps: int = 200, lr: float = 0.03) -> tuple[dict, dict]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    clean, degraded_cpu = make_toy(seed)
    degraded = degraded_cpu.to(device)
    # Adaptation receives ONLY degraded image and the fixed, target-free callable.
    adapted = {mode: adapt(degraded, local_statistics_loss, mode=mode,
                           grid_size=grid_size, steps=steps, lr=lr)
               for mode in ("global", "spatial")}
    # Both adaptation episodes finish before the first recovery evaluation.
    images = {"clean": clean, "identity": degraded_cpu,
              **{mode: result.image.cpu() for mode, result in adapted.items()}}
    initial_loss = local_statistics_loss(degraded).item()
    cases = {"identity": {**recovery_metrics(images["identity"], clean),
                          "loss_before": initial_loss, "loss_after": initial_loss,
                          "all_finite": bool(torch.isfinite(degraded).all())}}
    tensors = dict(images)
    for mode, result in adapted.items():
        cases[mode] = {**recovery_metrics(images[mode], clean),
                       "loss_before": result.diagnostics["loss_trajectory"][0],
                       "loss_after": result.diagnostics["loss_trajectory"][-1],
                       "physical_grid": result.parameter_grid.cpu().tolist(),
                       **result.diagnostics}
        tensors[mode + "_raw"] = result.raw_parameters.cpu()
        tensors[mode + "_field"] = result.parameter_field.cpu()
    report = {"config": {"seed": seed, "steps": steps, "lr": lr, "spatial_grid": list(grid_size),
                          "loss": "mean((avg_pool2d(rgb,8)-0.5)^2)",
                          "degradation": "left x0.45, right x1.55, clipped [0,1]",
                          "parameter_names": PARAMETER_NAMES},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "device": device, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None},
              "clipped_input_fraction": (degraded_cpu >= 1).float().mean().item(),
              "cases": cases,
              "spatial_mse_reduction_vs_global_pct": 100 * (1 - cases["spatial"]["mse"] / cases["global"]["mse"]),
              "spatial_better_than_global": cases["spatial"]["mse"] < cases["global"]["mse"],
              "limitations": "One synthetic midtone/color-balanced image and a matching generic statistical prior; no semantics or texture preservation, no real-data/task claim. Operator decomposition is non-identifiable. Coarse interpolation blurs the exposure step; clipping loses information."}
    return report, tensors


def save_comparison(tensors: dict, destination: Path):
    canvas = Image.new("RGB", (384, 84), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(("clean", "identity", "global", "spatial")):
        array = (tensors[name][0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
        canvas.paste(Image.fromarray(array), (96 * index, 20))
        draw.text((96 * index + 3, 3), name, fill="black")
    canvas.resize((1152, 252), Image.Resampling.NEAREST).save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--grid", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--output", type=Path, default=Path("research_log/artifacts/T001_cpu"))
    args = parser.parse_args()
    report, tensors = run_demo(seed=args.seed, device=args.device, grid_size=tuple(args.grid), steps=args.steps, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(tensors, args.output / "tensors.pt")
    save_comparison(tensors, args.output / "comparison.png")
    for name, case in report["cases"].items():
        print(f"{name}: loss={case['loss_before']:.8f}->{case['loss_after']:.8f}, "
              f"MSE={case['mse']:.8f}, PSNR={case['psnr_db']:.4f} dB, finite={case['all_finite']}")
    print(f"Spatial MSE reduction vs global: {report['spatial_mse_reduction_vs_global_pct']:.2f}%")


if __name__ == "__main__":
    main()
