"""Per-target Ours "ceiling" row ``ours_v2_ceiling`` (tuned on the target's test GT; not held-out).

User decision research_log/T075A/ceiling_decision.md. Pinned T074-C code is imported unchanged.

    python ceiling.py tuning  <ours_tuning.py arguments>     (run from the frozen Ours root, frozen env, under flock)
        LSRW/SMID: registers the 18-row v2 gate spec (v2_gate_metrics.register) so the pinned harness accepts the
        v2 gate receipt, injects --rows-dir research_log/T075B/<target>, then calls ours_tuning.main() unchanged
        (round 1 / --round2 / --materialize; default-reproduction check at every start).
    python ceiling.py kappa   --target T --low-receipt L --opaque-manifest O --gate-receipt R [--rows-dir D]
                              --tuned-row <ours_ttt_target_tuned dir> --out <ceiling dir> [--workers 4]
        D (v2 denoiser, identical to denoise_d.apply_d except kappa) on every tuned output for kappa in KAPPAS,
        frozen T071-A metric vs GT (metrics.Context), every kappa logged; selection = max mean PSNR, ties within
        0.01 dB by mean RGB-SSIM, then smaller |kappa - 6|. Then writes the row (T074-B schema), an independent
        re-execution check (verification.json) and freeze_receipt.json, all with tuned_on_test_gt = true.
    python ceiling.py metrics --target T ... --gate-receipt R --tuned-row DIR --ceiling DIR [--dev-rows-dir D] --out DIR
        All gated rows (+ SDSD dev v2 rows) + the tuned row + ours_v2_ceiling; paired comparisons ceiling vs every row.
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T074C = Path(os.environ.get("T074C_CODE", HERE.parents[1] / "T074C"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(T074C))
import reference_gate as rg  # noqa: E402
import metrics as M  # noqa: E402
import targets_t075 as t075  # noqa: E402
import v2_gate_metrics as gm  # noqa: E402
import denoise_d as dd  # noqa: E402
import verify_v2_rows as vv  # noqa: E402

ROW = "ours_v2_ceiling"
KAPPAS = (4.0, 5.0, 6.0, 7.0, 8.0)
KAPPA_DEFAULT = 6.0
TIE_DB = 0.01
NOTE = "ceiling reference: Ours knobs and kappa selected on this target's test GT (not held-out)"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def d_numpy(x, kappa):
    """x: HWC float64 already clipped to [0,1]. Same steps as denoise_d.apply_d with KAPPA replaced by kappa."""
    import cv2

    cv2.setNumThreads(1)
    sigma = dd.immerkaer_sigma(x)
    h = float(kappa) * 255.0 * sigma
    bgr = np.ascontiguousarray(np.round(x[..., ::-1] * 255.0).astype(np.uint8))
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, dd.TEMPLATE_WINDOW, dd.SEARCH_WINDOW)
    chw = np.ascontiguousarray(den[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / np.float32(255)
    return np.ascontiguousarray(chw), sigma, h


def to_hwc_clipped(tensor):
    return np.clip(tensor[0].permute(1, 2, 0).numpy().astype(np.float64), 0.0, 1.0)


def load_checked(path, row):
    import torch

    raw = Path(path).read_bytes()
    need(M.sha256_bytes(raw) == row["output_file_sha256"], f"tuned output file SHA mismatch: {path}")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        tensor = torch.load(stream, map_location="cpu", weights_only=True)
    need(tensor.dtype == torch.float32 and M.sha256_bytes(tensor.contiguous().numpy().tobytes()) == row["output_tensor_sha256"],
         f"tuned output tensor SHA mismatch: {path}")
    return tensor


# ---------------------------------------------------------------- context


def context(a):
    development = a.target == "SDSD_indoor"
    if development:
        rows_dir = a.rows_dir or T074C / "SDSD_indoor"
    else:
        gm.register(a.target)
        rows_dir = a.rows_dir or t075.default_rows_dir(a.target)
    return M.Context(a.target, a.low_receipt, a.opaque_manifest, a.gate_receipt, rows_dir, a.stage_target), development


def tuned_row(ctx, tuned_dir):
    """The pinned harness' tuned row, bound to this gate receipt (same checks as metrics.row_sources)."""
    manifest, digest = M.load_json(Path(tuned_dir) / "output_manifest.json")
    need(manifest["method_id"] == M.TUNED and manifest["gate_receipt_sha256"] == ctx.receipt_sha256,
         "tuned row not produced under this gate receipt")
    need(manifest["count"] == len(manifest["rows"]) == len(ctx.files) and manifest["low_receipt_sha256"] == ctx.low_sha256,
         "tuned row count/low binding")
    for item, row in zip(ctx.files, manifest["rows"]):
        need(row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"], "tuned row/low mismatch")
    return manifest, digest


# ---------------------------------------------------------------- kappa selection + row


def _score_task(task):
    index, x, ref, kappas = task
    core = M.import_frozen_metric()
    out = {"index": index, "kappa": {}}
    for kappa in kappas:
        chw, sigma, h = d_numpy(x, kappa)
        m = M.score(chw.transpose(1, 2, 0), ref, core)
        out["kappa"][str(kappa)] = {"psnr": m["psnr"], "rgb_ssim": m["rgb_ssim"], "h": h}
        out["sigma_hat"] = sigma
    return out


def select_kappa(summary):
    best = max(s["mean_psnr"] for s in summary.values())
    window = [k for k, s in summary.items() if s["mean_psnr"] >= best - TIE_DB]
    chosen = min(window, key=lambda k: (-summary[k]["mean_rgb_ssim"], abs(float(k) - KAPPA_DEFAULT), float(k)))
    return chosen, window


def _row_task(task):
    import torch

    torch.set_num_threads(1)
    index, x, kappa, item, source_row, out_dir = task
    started = time.perf_counter()
    chw, sigma, h = d_numpy(x, kappa)
    tensor = torch.from_numpy(chw)[None]
    d = Path(out_dir) / Path(item["name"]).stem
    d.mkdir(parents=True)
    dd.save_tensor(d / "output.pt.gz", tensor)
    row = {"method": "Ours-v2 ceiling (tuned knobs + D, kappa selected on test GT)", "low_name": item["name"],
           "low_sha256": item["sha256"], "source_output_tensor_sha256": source_row["output_tensor_sha256"],
           "source_output_file_sha256": source_row["output_file_sha256"], "sigma_hat": sigma, "h": h, "kappa": kappa,
           "shape": [1, 3, item["height"], item["width"]], "dtype": "torch.float32",
           "output_tensor_sha256": dd.tensor_sha256(tensor), "output_file_sha256": dd.sha256_file(d / "output.pt.gz"),
           "whole_run_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": 0}
    (d / "decision.json").write_text(json.dumps(row, indent=2))
    return row


def _verify_task(task):
    """Independent re-execution: slicing Immerkaer (vv), recorded h, OpenCV NLM, bitwise compare; hashes; quantisation."""
    import cv2
    import torch

    cv2.setNumThreads(1)
    index, x, row, out_dir, kappa = task
    d = Path(out_dir) / Path(row["low_name"]).stem
    need(json.loads((d / "decision.json").read_bytes()) == row, f"decision.json differs {row['low_name']}")
    raw = (d / "output.pt.gz").read_bytes()
    need(M.sha256_bytes(raw) == row["output_file_sha256"], f"file SHA {row['low_name']}")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        tensor = torch.load(stream, map_location="cpu", weights_only=True)
    array = tensor.numpy()
    need(M.sha256_bytes(tensor.contiguous().numpy().tobytes()) == row["output_tensor_sha256"], f"tensor SHA {row['low_name']}")
    need(np.array_equal(np.rint(array * 255).astype(np.float32) / np.float32(255), array), "not 8-bit quantised")
    sigma = vv.sigma_by_slicing(x)
    need(abs(sigma - row["sigma_hat"]) <= 1e-9 * max(sigma, 1e-12), f"sigma_hat {row['low_name']}")
    need(row["h"] == kappa * 255.0 * row["sigma_hat"] and row["kappa"] == kappa, f"h rule {row['low_name']}")
    u8 = np.ascontiguousarray(np.round(x * 255).astype(np.uint8)[:, :, ::-1])
    den = cv2.fastNlMeansDenoisingColored(u8, None, row["h"], row["h"], 7, 21)[:, :, ::-1]
    need(np.array_equal(den.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255), array), f"re-execution {row['low_name']}")
    return len(raw)


def kappa_cmd(argv):
    p = argparse.ArgumentParser(description="kappa selection + ours_v2_ceiling row")
    common_args(p)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    need(a.workers <= 6, "at most 6 workers")
    need(not a.out.exists(), f"{a.out} exists")
    ctx, development = context(a)
    manifest, tuned_sha = tuned_row(ctx, a.tuned_row)
    xs = []
    for item, row in zip(ctx.files, manifest["rows"]):
        xs.append(to_hwc_clipped(load_checked(a.tuned_row / Path(item["name"]).stem / "output.pt.gz", row)))
    refs = [ctx.reference(i) for i in range(len(ctx.files))]
    a.out.mkdir(parents=True)
    pool = get_context("spawn").Pool(a.workers)
    try:
        per_image = pool.map(_score_task, [(i, xs[i], refs[i], KAPPAS) for i in range(len(xs))], chunksize=1)
        summary = {}
        for kappa in KAPPAS:
            k = str(kappa)
            ps = np.array([r["kappa"][k]["psnr"] for r in per_image])
            ss = np.array([r["kappa"][k]["rgb_ssim"] for r in per_image])
            summary[k] = {"mean_psnr": float(ps.mean()), "median_psnr": float(np.median(ps)), "mean_rgb_ssim": float(ss.mean())}
        chosen, window = select_kappa(summary)
        selection = {
            "task": "T075-A-ceiling-kappa", "target": a.target, "tuned_on_test_gt": True, "note": NOTE,
            "gate_receipt_sha256": ctx.receipt_sha256, "tuned_row_manifest_sha256": tuned_sha,
            "tuned_knobs": manifest["knobs"], "tuned_setting_id": manifest["setting_id"],
            "tuning_log_sha256": manifest["selection"]["tuning_log_sha256"],
            "kappas": list(KAPPAS), "rule": "max mean PSNR; within 0.01 dB: max mean RGB-SSIM, then min |kappa-6|",
            "summary": summary, "tie_window": window, "selected_kappa": float(chosen),
            "per_image": [{"name": item["name"], "cluster": item["cluster"], **r} for item, r in zip(ctx.files, per_image)],
            "ceiling.py_sha256": M.sha256_file(__file__), "denoise_d.py_sha256": M.sha256_file(dd.__file__),
            "metrics.py_sha256": M.sha256_file(M.__file__)}
        with open(a.out / "kappa_selection.json", "x") as stream:
            json.dump(selection, stream, indent=2, allow_nan=False)
        with open(a.out / "reference_reads.json", "x") as stream:
            json.dump(ctx.reads, stream, indent=2)
        kappa = float(chosen)
        row_dir = a.out / ROW
        partial = Path(str(row_dir) + ".partial")
        rows = pool.map(_row_task, [(i, xs[i], kappa, item, manifest["rows"][i], str(partial))
                                    for i, item in enumerate(ctx.files)], chunksize=1)
        row_manifest = {
            "method": "Ours-v2 ceiling (tuned knobs + D, kappa selected on test GT)", "method_id": ROW, "count": len(rows),
            "low_receipt_sha256": ctx.low_sha256, "tuned_on_test_gt": True, "note": NOTE,
            "gate_receipt_sha256": ctx.receipt_sha256, "source_row": M.TUNED, "source_output_manifest_sha256": tuned_sha,
            "tuned_knobs": manifest["knobs"], "tuned_setting_id": manifest["setting_id"],
            "kappa_selection_sha256": M.sha256_file(a.out / "kappa_selection.json"), "selected_kappa": kappa,
            "d_spec": dict(dd.SPEC, kappa=kappa), "producer": "research_log/T075A/v2/ceiling.py",
            "producer_sha256": M.sha256_file(__file__), "reference_reads": 0, "metrics": 0,
            "reference_note": "outputs are D(tuned outputs) at the selected kappa; GT was read only by the selection step",
            "rows": rows}
        with open(partial / "output_manifest.json", "x") as stream:
            json.dump(row_manifest, stream, indent=2)
        os.replace(partial, row_dir)
        sizes = pool.map(_verify_task, [(i, xs[i], row, str(row_dir), kappa) for i, row in enumerate(rows)], chunksize=1)
    finally:
        pool.close()
    manifest_sha = M.sha256_file(row_dir / "output_manifest.json")
    verification = {"classification": "T075A_CEILING_OUTPUTS_VERIFIED", "method_id": ROW, "count": len(rows),
                    "output_manifest_sha256": manifest_sha, "low_receipt_sha256": ctx.low_sha256,
                    "recomputed_images": len(rows), "compressed_output_bytes": int(sum(sizes)),
                    "verifier": "ceiling.py _verify_task (slicing Immerkaer from verify_v2_rows.py, bitwise NLM re-execution)",
                    "reference_reads": 0, "metrics": 0}
    with open(a.out / "verification.json", "x") as stream:
        json.dump(verification, stream, indent=2)
    secs = [r["whole_run_seconds"] for r in rows]
    receipt = {"task": "T075-A-ceiling", "method_id": ROW, "target": ctx.spec["display_name"], "classification": "FROZEN_OUTPUTS",
               "tuned_on_test_gt": True, "note": NOTE, "output_count": len(rows),
               "decision": "research_log/T075A/ceiling_decision.md",
               "canonical_low_receipt_sha256": ctx.low_sha256, "gate_receipt_sha256": ctx.receipt_sha256,
               "source_row": M.TUNED, "source_output_manifest_sha256": tuned_sha, "tuned_knobs": manifest["knobs"],
               "selected_kappa": kappa, "kappa_summary": summary,
               "kappa_selection_sha256": M.sha256_file(a.out / "kappa_selection.json"),
               "runner": "research_log/T075A/v2/ceiling.py", "runner_sha256": M.sha256_file(__file__),
               "output_manifest_sha256": manifest_sha, "independent_verification_sha256": M.sha256_file(a.out / "verification.json"),
               "verification_classification": verification["classification"],
               "per_image_seconds_median": float(np.median(secs)), "per_image_seconds_sum": float(sum(secs)),
               "remote_output_root": str(row_dir), "preregistration_changed": False}
    (a.out / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"selected_kappa": kappa, "summary": summary, "tie_window": window, "manifest": manifest_sha,
                      "freeze": M.sha256_file(a.out / "freeze_receipt.json")}, indent=2), flush=True)


# ---------------------------------------------------------------- metrics


def metrics_cmd(argv):
    p = argparse.ArgumentParser(description="ceiling metrics: ceiling row vs every other row")
    common_args(p)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--ceiling", type=Path, required=True, help="ceiling dir (freeze_receipt.json, ours_v2_ceiling/)")
    p.add_argument("--dev-rows-dir", type=Path, help="SDSD_indoor: v2 development rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--reuse-scores", type=Path,
                   help="metrics_v2_result.json of this gate: per-image scores of the 18 gated rows are taken from it "
                        "(bound by gate receipt + per-row manifest SHA); only the tuned and ceiling rows are scored now")
    a = p.parse_args(argv)
    ctx, development = context(a)
    sources = M.row_sources(ctx, a.tuned_row)
    entries = dict(ctx.receipt["rows"])
    extra_dirs = {}
    if development:
        need(a.dev_rows_dir is not None, "SDSD_indoor needs --dev-rows-dir")
        for row in gm.EXTRA_ROWS:
            entry, _, manifest = rg.check_row(row, a.dev_rows_dir, ctx.spec, ctx.files, ctx.low_sha256, verify_outputs=True)
            entries[row] = entry
            extra_dirs[row] = a.dev_rows_dir / row
            sources[row] = (manifest, entry["output_manifest_sha256"], Path(entry["remote_output_root"]), {"development_after_gt": True})
    gm.check_derivations(ctx.rows_dir, entries, extra_dirs)
    freeze, _ = M.load_json(a.ceiling / "freeze_receipt.json")
    cman, csha = M.load_json(a.ceiling / ROW / "output_manifest.json")
    need(freeze["output_manifest_sha256"] == csha and freeze["gate_receipt_sha256"] == ctx.receipt_sha256
         and freeze["source_output_manifest_sha256"] == sources[M.TUNED][1] == cman["source_output_manifest_sha256"],
         "ceiling row not bound to this gate / tuned row")
    need(M.sha256_file(a.ceiling / "verification.json") == freeze["independent_verification_sha256"], "ceiling verification changed")
    sources[ROW] = (cman, csha, a.ceiling / ROW, {"tuned_on_test_gt": True, "selected_kappa": freeze["selected_kappa"], "note": NOTE})
    a.out.mkdir(parents=True, exist_ok=False)
    reused = None
    if a.reuse_scores is not None:
        prior, reused = M.load_json(a.reuse_scores)
        need(not development and prior["gate_receipt_sha256"] == ctx.receipt_sha256 and prior["n"] == len(ctx.files),
             "reused scores not produced under this gate")
        gated = [r for r in sources if r not in (M.TUNED, ROW)]
        need(set(prior["rows"]) == set(gated), "reused scores cover another row set")
        for r in gated:
            need(prior["rows"][r]["output_manifest_sha256"] == sources[r][1], f"reused scores bound to another {r} manifest")
            need([x["name"] for x in prior["per_image"][r]] == ctx.names, f"reused scores image order {r}")
        tables = gm.score_all(ctx, {r: sources[r] for r in (M.TUNED, ROW)}, a.workers)
        tables.update({r: prior["per_image"][r] for r in gated})
        tables = {r: tables[r] for r in sources}
    else:
        tables = gm.score_all(ctx, sources, a.workers)
    order, position, sizes = M.cluster_layout(ctx.names, ctx.clusters)
    indices = M.bootstrap_indices(len(order))
    comps = [{"a": ROW, "b": b, "direction": f"{ROW} - {b}", **M.paired(tables[ROW], tables[b], position, sizes, indices)}
             for b in sources if b != ROW]
    result = {"task": "T075-A-ceiling-metrics", "target": a.target, "tuned_on_test_gt_rows": [M.TUNED, ROW], "note": NOTE,
              "development_after_gt": development, "gate_receipt_sha256": ctx.receipt_sha256,
              "reused_scores_sha256": reused,
              "code_sha256": {"ceiling.py": M.sha256_file(__file__), "metrics.py": M.sha256_file(M.__file__),
                              "v2_gate_metrics.py": M.sha256_file(gm.__file__)},
              "n": len(ctx.files), "clusters": {"G": len(order), "low_power": len(order) <= M.LOW_POWER_MAX_CLUSTERS},
              "bootstrap": {"seed": M.SEED, "resamples": M.RESAMPLES, "indices_sha256": M.sha256_bytes(indices.tobytes())},
              "rows": {r: {"output_manifest_sha256": sources[r][1], **sources[r][3], **M.row_summary(tables[r])} for r in sources},
              "comparisons": comps, "per_image": tables}
    lines = [f"# Ours ceiling metrics: {a.target} (N={result['n']}, G={len(order)}) — {NOTE}", "",
             "| Row | Mean PSNR | Median PSNR | Mean RGB-SSIM |", "|---|---|---|---|"]
    lines += [f"| {r}{' (tuned on test GT)' if r in (M.TUNED, ROW) else ''} | {s['mean_psnr']:.4f} | {s['median_psnr']:.4f} | "
              f"{s['mean_rgb_ssim']:.4f} |" for r, s in result["rows"].items()]
    lines += ["", "| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |", "|---|---|---|---|---|"]
    for c in comps:
        for m in M.METRICS:
            v = c[m]
            lines.append(f"| {c['direction']} | {m} | {v['mean_delta']:+.4f} | {v['win_fraction']:.3f} | "
                         f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    for name, payload in (("reference_reads.json", json.dumps(ctx.reads, indent=2)),
                          ("ceiling_metrics_result.json", json.dumps(result, indent=2, allow_nan=False)),
                          ("ceiling_metrics_table.md", "\n".join(lines) + "\n")):
        with open(a.out / name, "x") as stream:
            stream.write(payload)
    print("\n".join(lines), flush=True)
    return result


# ---------------------------------------------------------------- tuning wrapper


def tuning_cmd(argv):
    """Pinned ours_tuning.main() against a v2-gated target (run from the frozen Ours root)."""
    need("--target" in argv, "--target required")
    target = argv[argv.index("--target") + 1]
    if target in t075.T075_TARGETS:
        gm.register(target)
        if "--rows-dir" not in argv:
            argv = list(argv) + ["--rows-dir", str(t075.default_rows_dir(target))]
    import ours_tuning

    saved = sys.argv
    sys.argv = ["ours_tuning.py"] + list(argv)
    try:
        return ours_tuning.main()
    finally:
        sys.argv = saved


def common_args(p):
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path)
    p.add_argument("--stage-target", type=Path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    need(__debug__, "run without -O")
    need(argv and argv[0] in ("tuning", "kappa", "metrics"), "usage: ceiling.py {tuning|kappa|metrics} ...")
    return {"tuning": tuning_cmd, "kappa": kappa_cmd, "metrics": metrics_cmd}[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
