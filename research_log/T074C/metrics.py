"""T074-C phase-3 metrics behind the reference gate: RGB PSNR / RGB-SSIM with the frozen T071-A code.

* Refuses to run unless ``reference_gate.validate_receipt`` accepts the gate receipt (before any GT access).
* GT goes through the identical staging code used for the lows (``stage_target.decode_low`` + ``to_unit``,
  file SHA pinned): np.load -> cv2.resize(960x512, INTER_LINEAR) -> [2,1,0] -> float32 / 255. Its raw
  bytes must match the reference-opaque manifest first.
* Metric = ``research_log.T071A.core.metrics`` (T072-L / T071-B provenance, commit 579c3691, SHA256 pinned
  in ``research_log/T073A/metric_plan.json``), imported unchanged from byte-exact copies in ``frozen_metric/``.
  It takes HWC RGB float64 arrays on a [0, 1] scale (data range 1): output float32 -> float64, clipped to
  [0, 1]; reference float32 -> float64.
* Statistics: image-level point estimates; paired deltas; strict win fraction; cluster bootstrap
  (PCG64 seed 20260922, integers(0, G, (10000, G), int64), image-weighted ratio, 95% linear percentile).
"""

import argparse
import gzip
import importlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_gate as gate_mod  # noqa: E402
from reference_gate import require, sha256_bytes, sha256_file, load_json, utc  # noqa: E402

FROZEN_METRIC_DIR = HERE / "frozen_metric"
FROZEN_METRIC_SHA256 = {  # research_log/T073A/metric_plan.json "metric_source" (commit 579c3691)
    "research_log/T071A/core.py": "45d6b92dee5fb3163da1ea1e9d331a2be36f8991a142bf221167c6d426871858",
    "ttie/ssim_transfer.py": "01d227a8b4caaf8f5705d9c18a5ef685367b6ccf80588d212be92830a04f8556",
    "ttie/__init__.py": "4de7703189366591a75bc1f59f8d0a02da9cf1ba06999293141560c0949512c6",
}
SEED = 20260922
RESAMPLES = 10000
QUANTILES = (0.025, 0.975)
LOW_POWER_MAX_CLUSTERS = 10  # protocol_draft.md section 5: G = 6 / 10 flagged unreliable
METRICS = ("psnr", "rgb_ssim")
HEADLINE = "ours_ttt"
ADAPTATION_PAIRS = (("promptir_dctta", "promptir"), ("ours_ttt", "ours_step0"))
TUNED = gate_mod.TUNED_ROW
TUNED_NOTE = ("knobs selected on these same test images' GT (user decision T074-A); "
              "point estimates and CIs are optimistic, not held-out")


# ---------------------------------------------------------------- frozen code


def import_frozen_metric():
    for rel, digest in FROZEN_METRIC_SHA256.items():
        require(sha256_file(FROZEN_METRIC_DIR / rel) == digest, f"vendored frozen metric file changed: {rel}")
    if str(FROZEN_METRIC_DIR) not in sys.path:
        sys.path.append(str(FROZEN_METRIC_DIR))  # appended: an already importable identical ttie wins
    core = importlib.import_module("research_log.T071A.core")
    ssim = importlib.import_module("ttie.ssim_transfer")
    require(sha256_file(core.__file__) == FROZEN_METRIC_SHA256["research_log/T071A/core.py"], "imported T071A core differs")
    require(sha256_file(ssim.__file__) == FROZEN_METRIC_SHA256["ttie/ssim_transfer.py"], "imported ssim_transfer differs")
    require(core.rgb_ssim is ssim.rgb_ssim, "T071A core bound to another rgb_ssim")
    return core


def import_stage_target(path, expected_sha256):
    path = Path(path)
    require(sha256_file(path) == expected_sha256, f"stage_target.py SHA mismatch at {path}")
    spec = importlib.util.spec_from_file_location("t074b_stage_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoGuard:
    """Phase-3 replacement for the staging DecodeGuard; GT decode is allowed once the gate receipt is valid."""

    def check(self, path):
        return None


# ---------------------------------------------------------------- context


class Context:
    """Gate-validated target context. Constructing it is the only way to reach GT decoding."""

    def __init__(self, target, low_receipt, opaque_manifest, gate_receipt, rows_dir=None, stage_target=None):
        self.target = target
        self.rows_dir = Path(rows_dir) if rows_dir else HERE / target
        self.receipt, self.receipt_sha256 = gate_mod.validate_receipt(
            gate_receipt, target, self.rows_dir, low_receipt, opaque_manifest)
        self.spec = gate_mod.target_spec(target)
        self.low, self.low_sha256, self.opaque, self.opaque_sha256 = gate_mod.check_low_and_opaque(
            self.spec, low_receipt, opaque_manifest)
        self.files = self.low["files"]
        self.names = [f["name"] for f in self.files]
        self.clusters = [f["cluster"] for f in self.files]
        self.stage = import_stage_target(stage_target or HERE.parent / "T074B" / "stage_target.py",
                                         self.spec["stage_script_sha256"])
        self.core = import_frozen_metric()
        self.dataset_root = Path(self.opaque["rule"]["root"])
        self.geometry = self.opaque["rule"]["geometry"]
        self.reads = []

    def reference(self, index):
        """Canonical GT (H, W, 3) float32 RGB in [0, 1] via the official loader path used for the lows."""
        item, pair = self.files[index], self.opaque["pairs"][index]
        require(pair["name"] == item["name"], "pair order mismatch")
        path = self.dataset_root / pair["gt_relpath"]
        opened = utc()
        digest, size = self.stage.opaque_sha256(path)
        require(digest == pair["gt_raw_sha256"] and size == pair["gt_bytes"], f"GT raw bytes changed: {pair['gt_relpath']}")
        rgb, facts = self.stage.decode_low(path, self.geometry, _NoGuard())
        require(facts["source_dtype"] == item["source_dtype"] == "uint8", f"GT/low source dtype not uint8: {facts}")
        require(rgb.dtype == np.uint8 and rgb.shape == (item["height"], item["width"], 3),
                f"low/GT geometry mismatch at {item['name']}: {rgb.shape}")
        self.reads.append({"name": item["name"], "gt_relpath": pair["gt_relpath"], "gt_raw_sha256": digest,
                           "opened_utc": opened, "source_shape": facts["source_shape"]})
        return self.stage.to_unit(rgb)


def load_output(path, row, item):
    """Frozen output tensor as (H, W, 3) float32, after file and tensor hash checks."""
    import torch

    raw = Path(path).read_bytes()
    require(sha256_bytes(raw) == row["output_file_sha256"], f"output file SHA mismatch: {path}")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        tensor = torch.load(stream, map_location="cpu", weights_only=True)
    require(list(tensor.shape) == [1, 3, item["height"], item["width"]] and tensor.dtype == torch.float32,
            f"output geometry/dtype mismatch: {path}")
    require(sha256_bytes(tensor.contiguous().numpy().tobytes()) == row["output_tensor_sha256"], f"output tensor SHA mismatch: {path}")
    require(bool(torch.isfinite(tensor).all()), f"nonfinite output: {path}")
    return tensor[0].permute(1, 2, 0).contiguous().numpy()


def score(output_hwc, reference_hwc, core):
    """Clip the output to [0, 1] in float64, then call the frozen T071-A metrics unchanged.

    Inputs are made C-contiguous HWC first (the T071-B convention: np.load'ed npy / PIL arrays). The frozen
    PSNR's np.mean summation order follows memory layout, so a strided view can move PSNR by one ulp.
    """
    x = np.ascontiguousarray(output_hwc, dtype=np.float64)
    y = np.ascontiguousarray(reference_hwc, dtype=np.float64)
    require(x.shape == y.shape and x.ndim == 3 and x.shape[2] == 3, f"output/GT shape mismatch {x.shape} vs {y.shape}")
    require(bool(np.isfinite(x).all() and np.isfinite(y).all()), "nonfinite metric input")
    below, above = float((x < 0).mean()), float((x > 1).mean())
    values = core.metrics(np.clip(x, 0.0, 1.0), y)
    require(all(np.isfinite(values[k]) for k in METRICS), f"nonfinite metric {values}")
    return {"psnr": values["psnr"], "rgb_ssim": values["rgb_ssim"], "clipped_below": below, "clipped_above": above}


# ---------------------------------------------------------------- statistics


def bootstrap_indices(groups):
    rng = np.random.Generator(np.random.PCG64(SEED))
    return rng.integers(0, groups, size=(RESAMPLES, groups), dtype=np.int64)


def cluster_layout(names, clusters):
    """Clusters in sorted cluster-id order (protocol_draft.md section 5); position of each image."""
    order = sorted(set(clusters))
    position = np.asarray([order.index(c) for c in clusters], dtype=np.int64)
    sizes = np.bincount(position, minlength=len(order)).astype(np.int64)
    if (sizes == 1).all():
        require(order == list(clusters), "singleton clusters must sort like the canonical images (exact T073-A reduction)")
    return order, position, sizes


def cluster_bootstrap(deltas, position, sizes, indices):
    """Image-weighted ratio statistic sum_b S_c / sum_b n_c; 95% linear percentile CI."""
    d = np.asarray(deltas, dtype=np.float64)
    sums = np.bincount(position, weights=d, minlength=len(sizes))
    stats = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    low, high = np.quantile(stats, QUANTILES, method="linear").tolist()
    return {"ci95": [low, high]}, stats


def paired(a, b, position, sizes, indices):
    out = {}
    for metric in METRICS:
        d = np.asarray([x[metric] - y[metric] for x, y in zip(a, b)], dtype=np.float64)
        boot, _ = cluster_bootstrap(d, position, sizes, indices)
        out[metric] = {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                       "win_fraction": float((d > 0).sum() / d.size), "wins": int((d > 0).sum()),
                       "ties": int((d == 0).sum()), "n": int(d.size), **boot}
    return out


def row_summary(table):
    ps = np.asarray([r["psnr"] for r in table], dtype=np.float64)
    ss = np.asarray([r["rgb_ssim"] for r in table], dtype=np.float64)
    return {"images": int(ps.size), "mean_psnr": float(ps.mean()), "median_psnr": float(np.median(ps)),
            "mean_rgb_ssim": float(ss.mean()), "median_rgb_ssim": float(np.median(ss))}


def comparisons(row_ids):
    pairs = {}
    for other in row_ids:
        if HEADLINE in row_ids and other not in (HEADLINE, TUNED):
            pairs.setdefault((HEADLINE, other), []).append("headline")
        if TUNED in row_ids and other != TUNED:
            pairs.setdefault((TUNED, other), []).append("tuned_headline")
    for a, b in ADAPTATION_PAIRS:
        if a in row_ids and b in row_ids:
            pairs.setdefault((a, b), []).append("adaptation_gain")
    return pairs


# ---------------------------------------------------------------- run


def row_sources(ctx, tuned_dir=None):
    """{row_id: (manifest, manifest_sha256, output_root, extra)} for every gated row (+ tuned row)."""
    sources = {}
    for row_id in ctx.receipt["required_rows"]:
        entry = ctx.receipt["rows"][row_id]
        manifest, digest = load_json(ctx.rows_dir / row_id / "output_manifest.json")
        require(digest == entry["output_manifest_sha256"], f"{row_id}: manifest differs from gate receipt")
        extra = {}
        if row_id == "ours_ttt":
            freeze, _ = load_json(ctx.rows_dir / row_id / "freeze_receipt.json")
            extra["no_active_abstentions"] = freeze["ttt_abstain_no_active_gate_count"]
        sources[row_id] = (manifest, digest, Path(entry["remote_output_root"]), extra)
    if tuned_dir is not None:
        manifest, digest = load_json(Path(tuned_dir) / "output_manifest.json")
        require(manifest["method_id"] == TUNED and manifest["gate_receipt_sha256"] == ctx.receipt_sha256,
                "tuned row not produced under this gate receipt")
        require(manifest["count"] == len(manifest["rows"]) == len(ctx.files)
                and manifest["low_receipt_sha256"] == ctx.low_sha256, "tuned row count/low binding mismatch")
        for item, row in zip(ctx.files, manifest["rows"]):
            require(row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"], "tuned row/low mismatch")
        sources[TUNED] = (manifest, digest, Path(tuned_dir), {"no_active_abstentions": manifest["no_active_abstentions"],
                                                              "tuned_on_test_gt": True, "note": TUNED_NOTE})
    return sources


def evaluate(ctx, tuned_dir=None):
    sources = row_sources(ctx, tuned_dir)
    row_ids = list(sources)
    tables = {r: [] for r in row_ids}
    first_access = utc()
    for index, item in enumerate(ctx.files):
        reference = ctx.reference(index)
        for row_id, (manifest, _, root, _) in sources.items():
            row = manifest["rows"][index]
            output = load_output(root / Path(item["name"]).stem / "output.pt.gz", row, item)
            tables[row_id].append({"name": item["name"], "cluster": item["cluster"], **score(output, reference, ctx.core)})
    order, position, sizes = cluster_layout(ctx.names, ctx.clusters)
    indices = bootstrap_indices(len(order))
    result = {
        "task": "T074-C-phase3-metrics",
        "target": ctx.target,
        "gate_receipt_sha256": ctx.receipt_sha256,
        "low_receipt_sha256": ctx.low_sha256,
        "reference_opaque_manifest_sha256": ctx.opaque_sha256,
        "code_sha256": {"metrics.py": sha256_file(__file__), "reference_gate.py": sha256_file(gate_mod.__file__),
                        **{f"frozen_metric/{k}": v for k, v in FROZEN_METRIC_SHA256.items()},
                        "stage_target.py": ctx.spec["stage_script_sha256"]},
        "environment": _environment(),
        "metric": {"implementation": "research_log.T071A.core.metrics (full-RGB PSNR, ttie.ssim_transfer.rgb_ssim)",
                   "ssim": ctx.core.METRIC, "output_conversion": "float32 -> float64, clip [0, 1]",
                   "reference_conversion": "official SNR loader: np.load -> cv2.resize(960x512) -> [2,1,0] -> float32/255 -> float64"},
        "first_reference_access_utc": first_access,
        "reference_reads": len(ctx.reads),
        "n": len(ctx.files),
        "clusters": {"G": len(order), "order": order, "sizes": sizes.tolist(),
                     "low_power": len(order) <= LOW_POWER_MAX_CLUSTERS},
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
                      "indices": f"integers(0,{len(order)},size=({RESAMPLES},{len(order)}),dtype=int64); C order",
                      "indices_sha256": sha256_bytes(indices.tobytes()), "statistic": "sum S_c / sum n_c (image-weighted)",
                      "interval": "95% percentile", "quantiles": list(QUANTILES), "quantile_method": "linear"},
        "rows": {r: {"output_manifest_sha256": sources[r][1], **sources[r][3], **row_summary(tables[r])} for r in row_ids},
        "comparisons": [{"a": a, "b": b, "direction": f"{a} - {b}", "families": fam,
                         **({"tuned_on_test_gt": True, "note": TUNED_NOTE} if TUNED in (a, b) else {}),
                         **paired(tables[a], tables[b], position, sizes, indices)}
                        for (a, b), fam in comparisons(row_ids).items()],
        "per_image": tables,
    }
    return result


def _environment():
    import cv2
    import scipy
    import torch

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "cv2": cv2.__version__, "torch": torch.__version__}


def markdown(result):
    c = result["clusters"]
    lines = [f"# T074-C metrics: {result['target']} (N={result['n']}, G={c['G']}"
             + (", low-power: CIs unreliable" if c["low_power"] else "") + ")", "",
             f"Gate receipt `{result['gate_receipt_sha256']}`. Outputs clipped to [0,1]; frozen T071-A metrics.", "",
             "| Row | N | Mean PSNR | Median PSNR | Mean RGB-SSIM | No-active abstentions |", "|---|---|---|---|---|---|"]
    for r, s in result["rows"].items():
        lines.append(f"| {r}{' (tuned on test GT)' if s.get('tuned_on_test_gt') else ''} | {s['images']} | {s['mean_psnr']:.4f} | {s['median_psnr']:.4f} | {s['mean_rgb_ssim']:.4f} | "
                     f"{s.get('no_active_abstentions', 'N/A')} |")
    lines += ["", "| Comparison | Metric | Mean Δ | Median Δ | Win fraction | 95% cluster CI | Families |",
              "|---|---|---|---|---|---|---|"]
    for comp in result["comparisons"]:
        for m in METRICS:
            v = comp[m]
            lines.append(f"| {comp['direction']} | {m} | {v['mean_delta']:+.4f} | {v['median_delta']:+.4f} | "
                         f"{v['win_fraction']:.3f} ({v['wins']}/{v['n']}) | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                         f"{', '.join(comp['families'])} |")
    if TUNED in result["rows"]:
        lines += ["", f"Note: `{TUNED}`: {TUNED_NOTE}."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path)
    p.add_argument("--stage-target", type=Path)
    p.add_argument("--tuned-row", type=Path, help="phase-4 ours_ttt_target_tuned row directory")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ctx = Context(a.target, a.low_receipt, a.opaque_manifest, a.gate_receipt, a.rows_dir, a.stage_target)
    a.out.mkdir(parents=True, exist_ok=False)
    result = evaluate(ctx, a.tuned_row)
    for name, payload in (("reference_reads.json", json.dumps(ctx.reads, indent=2)),
                          ("metrics_result.json", json.dumps(result, indent=2, allow_nan=False)),
                          ("metrics_table.md", markdown(result))):
        with open(a.out / name, "x") as stream:
            stream.write(payload)
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
