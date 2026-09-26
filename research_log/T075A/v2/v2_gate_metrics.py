"""Reference gate + metrics extension for the T075-A v2 rows (pinned T074-C code imported unchanged).

Rows added to the 8 preregistered rows (research_log/T075A/v2_freeze.md):
  ours_ttt_sdsd_knobs, ours_v2, ours_v2_sdsd_knobs, <row>_plus_D for the 7 other preregistered rows.

    python v2_gate_metrics.py gate    --target LSRW|SMID --low-receipt L --opaque-manifest O [--rows-dir D] [--out R]
    python v2_gate_metrics.py metrics --target LSRW|SMID --low-receipt L --opaque-manifest O --gate-receipt R
                                      [--rows-dir D] --out DIR
    python v2_gate_metrics.py metrics --target SDSD_indoor ... --gate-receipt <committed SDSD receipt>
                                      --dev-rows-dir research_log/T075A/sdsd_v2 --out DIR   (development only)

Held-out targets (LSRW, SMID): the gate registers the target with required_rows = 8 preregistered + 10 v2 rows, so
the unchanged ``reference_gate.evaluate`` re-verifies every v2 row (freeze -> manifest -> verification, local and
remote, every output file re-hashed) before GT can be decoded; it also checks here that every derived row is bound
to the exact frozen manifest of its source row. Use this gate (not the 8-row ``targets_t075.py gate``) for
LSRW/SMID, otherwise the v2 rows are not provably frozen before GT.
SDSD-indoor: its gate is already open (8 rows), so v2 rows there are development rows: they are checked with the
same ``check_row`` + derivation bindings at metric time and flagged ``development_after_gt: true``.

Comparisons (paired, cluster bootstrap exactly as metrics.py): each v2 row vs each of the 6 baselines, vs each
baseline_plus_D, v2 - ours_ttt; ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs; ours_ttt_sdsd_knobs - ours_ttt.
"""

import argparse
import json
import os
import sys
from pathlib import Path

T074C = Path(os.environ.get("T074C_CODE", Path(__file__).resolve().parents[2] / "T074C"))
sys.path.insert(0, str(T074C))
import reference_gate as rg  # noqa: E402
import metrics as M  # noqa: E402
import targets_t075 as t075  # noqa: E402

BASELINES = ["retinexformer", "snr_aware", "promptir", "promptir_dctta", "mr_illuminate", "quadprior"]
PREREG = list(t075.REQUIRED_ROWS)
V2_ROWS = ["ours_v2", "ours_v2_sdsd_knobs"]
EXTRA_ROWS = ["ours_ttt_sdsd_knobs"] + V2_ROWS + [f"{r}_plus_D" for r in PREREG if r != "ours_ttt"]
SOURCES = {"ours_v2": "ours_ttt", "ours_v2_sdsd_knobs": "ours_ttt_sdsd_knobs", "ours_ttt_sdsd_knobs": "ours_ttt",
           **{f"{r}_plus_D": r for r in PREREG if r != "ours_ttt"}}
KNOB_OVERRIDES = {"q_joint": 0.35266535990213066, "exposure_target": 0.7}
DEV_NOTE = "SDSD-indoor: v2 designed on this target's GT (development); not held-out"


def register(target, base_spec=None):
    """Register ``target`` with the 8 preregistered rows + the v2 rows (fails if registered differently)."""
    if base_spec is None:
        existing = rg.TARGETS.get(target)
        if existing is not None and existing["required_rows"] == PREREG + EXTRA_ROWS:
            return existing
        rg.require(target in t075.T075_TARGETS, f"unknown held-out target {target!r}")
    base = base_spec or t075.T075_TARGETS[target]
    rg.require(list(base["required_rows"]) == PREREG, "base spec must carry exactly the 8 preregistered rows")
    spec = dict(base, required_rows=PREREG + EXTRA_ROWS)
    existing = rg.TARGETS.setdefault(target, spec)
    rg.require(existing == spec, f"target {target} already registered with a different spec")
    return spec


def check_derivations(rows_dir, entries, extra_dirs=None):
    """Every v2 row's freeze and manifest bind the exact frozen manifest of its source row."""
    for row, source in SOURCES.items():
        directory = Path((extra_dirs or {}).get(row, Path(rows_dir) / row))
        freeze, _ = rg.load_json(directory / "freeze_receipt.json")
        manifest, manifest_sha = rg.load_json(directory / "output_manifest.json")
        rg.require(manifest_sha == entries[row]["output_manifest_sha256"], f"{row}: manifest differs from its checked entry")
        src_sha = entries[source]["output_manifest_sha256"]
        rg.require(freeze["source_row"] == source and freeze["source_output_manifest_sha256"] == src_sha,
                   f"{row}: freeze not bound to frozen {source}")
        if row == "ours_ttt_sdsd_knobs":
            rg.require(manifest["frozen_ours_ttt_manifest_sha256"] == src_sha, f"{row}: manifest not bound to ours_ttt")
            rg.require(all(manifest["knobs"][k] == v for k, v in KNOB_OVERRIDES.items()), f"{row}: knobs changed")
        else:
            rg.require(manifest["source_row"] == source and manifest["source_output_manifest_sha256"] == src_sha,
                       f"{row}: manifest not bound to frozen {source}")
            rg.require(manifest["d_spec"]["kappa"] == 6.0, f"{row}: kappa changed")


def gate(argv):
    p = argparse.ArgumentParser(description="T074-C reference gate with the v2 rows bound")
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    rows_dir = a.rows_dir or t075.default_rows_dir(a.target)
    out = a.out or rows_dir / "reference_gate_receipt.json"
    rg.require(not out.exists(), f"{out} exists; the gate receipt is immutable")
    register(a.target)
    fields = rg.evaluate(a.target, rows_dir, a.low_receipt, a.opaque_manifest)
    check_derivations(rows_dir, fields["rows"])
    fields["target_registry"] = {"module": "research_log/T074C/targets_t075.py", "sha256": rg.sha256_file(t075.__file__),
                                 "v2_module": "research_log/T075A/v2/v2_gate_metrics.py", "v2_sha256": rg.sha256_file(__file__)}
    receipt = rg.write_receipt(fields, out)
    print(json.dumps({"receipt": str(out), "sha256": rg.sha256_file(out), "rows": receipt["required_rows"]}, indent=2))
    return receipt


def comparisons(rows):
    pairs = []
    for v in V2_ROWS:
        pairs += [(v, b, "v2_vs_baseline") for b in BASELINES]
        pairs += [(v, f"{b}_plus_D", "v2_vs_baseline_plus_D") for b in BASELINES]
        pairs.append((v, "ours_ttt", "v2_gain_over_ours_ttt"))
    pairs += [("ours_v2_sdsd_knobs", "ours_ttt_sdsd_knobs", "d_gain_sdsd_knobs"),
              ("ours_ttt_sdsd_knobs", "ours_ttt", "knob_gain")]
    return [c for c in pairs if c[0] in rows and c[1] in rows]


def metrics(argv):
    p = argparse.ArgumentParser(description="metrics for the 8 preregistered rows + the v2 rows")
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path)
    p.add_argument("--stage-target", type=Path)
    p.add_argument("--dev-rows-dir", type=Path, help="SDSD_indoor only: directory holding the v2 development rows")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    development = a.target == "SDSD_indoor"
    if development:
        rg.require(a.dev_rows_dir is not None, "SDSD_indoor needs --dev-rows-dir")
        rows_dir = a.rows_dir or T074C / "SDSD_indoor"
    else:
        rg.require(a.dev_rows_dir is None, "--dev-rows-dir is only for SDSD_indoor")
        register(a.target)
        rows_dir = a.rows_dir or t075.default_rows_dir(a.target)
    ctx = M.Context(a.target, a.low_receipt, a.opaque_manifest, a.gate_receipt, rows_dir, a.stage_target)
    sources = M.row_sources(ctx)
    entries = dict(ctx.receipt["rows"])
    extra_dirs = {}
    if development:
        spec = ctx.spec
        for row in EXTRA_ROWS:
            entry, _, manifest = rg.check_row(row, a.dev_rows_dir, spec, ctx.files, ctx.low_sha256, verify_outputs=True)
            entries[row] = entry
            extra_dirs[row] = a.dev_rows_dir / row
            sources[row] = (manifest, entry["output_manifest_sha256"], Path(entry["remote_output_root"]),
                            {"development_after_gt": True})
    check_derivations(rows_dir, entries, extra_dirs)
    rg.require(set(sources) == set(PREREG + EXTRA_ROWS), f"row set {sorted(sources)}")
    a.out.mkdir(parents=True, exist_ok=False)
    tables = {r: [] for r in sources}
    for index, item in enumerate(ctx.files):
        reference = ctx.reference(index)
        for row_id, (manifest, _, root, _) in sources.items():
            output = M.load_output(root / Path(item["name"]).stem / "output.pt.gz", manifest["rows"][index], item)
            tables[row_id].append({"name": item["name"], "cluster": item["cluster"], **M.score(output, reference, ctx.core)})
    order, position, sizes = M.cluster_layout(ctx.names, ctx.clusters)
    indices = M.bootstrap_indices(len(order))
    result = {
        "task": "T075-A-v2-metrics", "target": a.target, "development_after_gt": development,
        **({"note": DEV_NOTE} if development else {}),
        "gate_receipt_sha256": ctx.receipt_sha256, "low_receipt_sha256": ctx.low_sha256,
        "reference_opaque_manifest_sha256": ctx.opaque_sha256,
        "code_sha256": {"v2_gate_metrics.py": rg.sha256_file(__file__), "metrics.py": rg.sha256_file(M.__file__),
                        "reference_gate.py": rg.sha256_file(rg.__file__), "targets_t075.py": rg.sha256_file(t075.__file__)},
        "metric": "research_log.T071A.core.metrics via metrics.score (frozen T071-A, clip [0,1])",
        "reference_reads": len(ctx.reads), "n": len(ctx.files),
        "clusters": {"G": len(order), "order": order, "sizes": sizes.tolist(), "low_power": len(order) <= M.LOW_POWER_MAX_CLUSTERS},
        "bootstrap": {"seed": M.SEED, "resamples": M.RESAMPLES, "indices_sha256": M.sha256_bytes(indices.tobytes())},
        "rows": {r: {"output_manifest_sha256": sources[r][1], **sources[r][3], **M.row_summary(tables[r])} for r in sources},
        "comparisons": [{"a": x, "b": y, "direction": f"{x} - {y}", "family": fam,
                         **M.paired(tables[x], tables[y], position, sizes, indices)} for x, y, fam in comparisons(sources)],
        "per_image": tables,
    }
    lines = [f"# T075-A v2 metrics: {a.target} (N={result['n']}, G={len(order)})"
             + (" — DEVELOPMENT (v2 designed on this GT)" if development else ""), "",
             "| Row | Mean PSNR | Mean RGB-SSIM |", "|---|---|---|"]
    lines += [f"| {r} | {s['mean_psnr']:.4f} | {s['mean_rgb_ssim']:.4f} |" for r, s in result["rows"].items()]
    lines += ["", "| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |", "|---|---|---|---|---|"]
    for c in result["comparisons"]:
        for m in M.METRICS:
            v = c[m]
            lines.append(f"| {c['direction']} | {m} | {v['mean_delta']:+.4f} | {v['win_fraction']:.3f} | "
                         f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    for name, payload in (("reference_reads.json", json.dumps(ctx.reads, indent=2)),
                          ("metrics_v2_result.json", json.dumps(result, indent=2, allow_nan=False)),
                          ("metrics_v2_table.md", "\n".join(lines) + "\n")):
        with open(a.out / name, "x") as stream:
            stream.write(payload)
    print("\n".join(lines), flush=True)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rg.require(argv and argv[0] in ("gate", "metrics"), "usage: v2_gate_metrics.py {gate|metrics} ...")
    return gate(argv[1:]) if argv[0] == "gate" else metrics(argv[1:])


if __name__ == "__main__":
    main()
